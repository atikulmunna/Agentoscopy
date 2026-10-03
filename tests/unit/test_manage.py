"""Garbage collection, and resuming and cancelling runs from the CLI."""

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
import yaml
from conftest import BASE_TASK

from agentoscopy.cli.main import main
from agentoscopy.housekeeping import ARTIFACTS_DELETED, collect_garbage, run_output_dirs
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import pin_tasks, plan_trials
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import RunSettings, Store
from agentoscopy.testing.fake_sandbox import FakeBackend

LATER = datetime.now(UTC) + timedelta(days=31)  # every run finished today is "old" by then
CONFIG = AgentConfig(
    name="fixer",
    adapter="python",
    entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
    params={"script": [{"write": "app.py", "content": "fixed"}]},
)


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def project(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = FakeBackend(grader=passes_when_fixed)
    for module in ("agentoscopy.cli.run", "agentoscopy.cli.manage"):
        monkeypatch.setattr(f"{module}.DockerBackend", lambda: backend)
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    write_task({**BASE_TASK, "id": "task-a"})
    store.save_task_version(load_task(tasks_dir, "task-a"), "fake-image", [], [])
    yield store, home, tasks_dir, backend
    store.close()


def new_run(store, tasks_dir, trials=2):
    pinned = pin_tasks(store, tasks_dir, ["task-a"])
    settings = RunSettings(
        suite_id=None, suite_version=None, trials_per_task=trials, budget_usd=None, seed=2,
        concurrency=2, harness_version="test",
    )  # fmt: skip
    return store.create_run(CONFIG, settings, plan_trials(pinned, trials, seed=2))


def cli(home, tasks_dir, *args):
    return main([*args, "--home", str(home), "--tasks-dir", str(tasks_dir)])


def finished_run_with_outputs(store, home, tasks_dir):
    run_id = new_run(store, tasks_dir)
    for name in ("trajectories", "artifacts"):
        (home / name / run_id / "trial").mkdir(parents=True)
        (home / name / run_id / "trial" / "attempt-1.jsonl").write_text("x" * 100)
    store.set_run_status(run_id, "completed", finished=True)
    return run_id


# Garbage collection -----------------------------------------------------------------------


def gc(store, home, backend, **options):
    return asyncio.run(collect_garbage(store, backend, home, timedelta(days=30), **options))


def test_old_run_outputs_are_deleted_and_the_run_is_flagged(project):
    store, home, tasks_dir, backend = project
    old = finished_run_with_outputs(store, home, tasks_dir)
    running = new_run(store, tasks_dir)

    report = gc(store, home, backend, now=LATER)

    assert report.cleared_runs == [old] and report.freed_bytes == 200
    assert not (home / "trajectories" / old).exists() and not (home / "artifacts" / old).exists()
    assert ARTIFACTS_DELETED in store.get_run(old)["flags"]
    assert store.get_run(running)["flags"] == []
    assert gc(store, home, backend, now=LATER).cleared_runs == []  # done once


def test_recent_runs_and_runs_awaiting_review_are_kept(project):
    store, home, tasks_dir, backend = project
    recent = finished_run_with_outputs(store, home, tasks_dir)
    reviewed = finished_run_with_outputs(store, home, tasks_dir)
    store.queue_review(store.trial_rows(reviewed)[0]["trial_id"], 0, "", "manual")

    assert gc(store, home, backend).cleared_runs == []
    report = gc(store, home, backend, now=LATER)

    assert reviewed not in report.cleared_runs and recent in report.cleared_runs
    assert report.kept_runs == [(reviewed, "it has trials waiting for review")]
    assert (home / "trajectories" / reviewed).exists()


def test_a_dry_run_deletes_nothing(project):
    store, home, tasks_dir, backend = project
    old = finished_run_with_outputs(store, home, tasks_dir)

    report = gc(store, home, backend, now=LATER, dry_run=True)

    assert report.cleared_runs == [old]
    assert (home / "trajectories" / old).exists()
    assert store.get_run(old)["flags"] == []


def test_a_malformed_run_id_never_leads_outside_its_directory(tmp_path):
    (tmp_path / "trajectories").mkdir()
    (tmp_path / "victim").mkdir()

    assert run_output_dirs(tmp_path, "../victim") == []
    assert run_output_dirs(tmp_path, "..") == []


def test_only_old_sandboxes_of_dead_trials_are_removed(project):
    store, home, tasks_dir, backend = project
    run_id = new_run(store, tasks_dir, trials=3)
    live = store.claim_next(run_id, "me:1:0").trial
    dead = store.claim_next(run_id, "dead:1:0", now=time.time() - 3600, lease_s=1).trial
    now = datetime.now(UTC)

    async def sandboxes():
        made = {}
        for name, trial_id in [
            ("live", live.trial_id),
            ("dead", dead.trial_id),
            ("young", "validate-x"),
        ]:
            made[name] = await backend.create("image", None, trial_id)
        made["live"].created = made["dead"].created = now - timedelta(hours=1)
        return made

    made = asyncio.run(sandboxes())
    report = gc(store, home, backend, now=now)

    assert [record.trial_id for record in report.removed_sandboxes] == [dead.trial_id]
    assert made["dead"].destroyed
    assert not made["live"].destroyed and not made["young"].destroyed


def test_gc_from_the_cli(project, capsys):
    store, home, tasks_dir, _ = project
    finished_run_with_outputs(store, home, tasks_dir)

    assert cli(home, tasks_dir, "gc", "--older-than", "1s", "--dry-run") == 0
    assert "would remove the trajectories and artifacts of 0 runs" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli(home, tasks_dir, "gc", "--older-than", "a while")


# Resume and cancel -------------------------------------------------------------------------


def test_a_crashed_run_is_resumed_from_the_cli(project, capsys):
    store, home, tasks_dir, _ = project
    run_id = new_run(store, tasks_dir, trials=3)
    store.mark_running(run_id)
    # Its process claimed a trial and died long ago.
    store.claim_next(run_id, "dead:1:0", now=time.time() - 3600, lease_s=1)

    assert cli(home, tasks_dir, "run", "--resume", run_id) == 0

    assert f"run {run_id}: resuming at 0 of 3 trials" in capsys.readouterr().out
    assert store.run_status(run_id) == "completed"
    assert [row["outcome"] for row in store.trial_rows(run_id)] == ["pass"] * 3


@pytest.mark.parametrize(
    ("setup", "message"), [("finish", "RUN_FINISHED"), ("change", "TASK_CHANGED")]
)
def test_runs_that_cannot_be_resumed(project, write_task, capsys, setup, message):
    store, home, tasks_dir, _ = project
    run_id = new_run(store, tasks_dir)
    if setup == "finish":
        store.set_run_status(run_id, "completed", finished=True)
    else:
        write_task({**BASE_TASK, "id": "task-a", "instructions": "Something else."})

    assert cli(home, tasks_dir, "run", "--resume", run_id) == 2
    assert message in capsys.readouterr().err


def test_resume_needs_a_known_run_and_a_new_run_needs_an_agent(project, capsys):
    store, home, tasks_dir, _ = project

    assert cli(home, tasks_dir, "run", "--resume", "no-such-run") == 2
    assert cli(home, tasks_dir, "run", "--task", "task-a") == 2
    err = capsys.readouterr().err
    assert "NOT_FOUND" in err and "--agent is required" in err


def test_cancel_from_the_cli(project, capsys):
    store, home, tasks_dir, _ = project
    pending = new_run(store, tasks_dir)
    done = new_run(store, tasks_dir)
    store.set_run_status(done, "completed", finished=True)

    assert cli(home, tasks_dir, "cancel", pending) == 0
    assert cli(home, tasks_dir, "cancel", done) == 2
    assert cli(home, tasks_dir, "cancel", "no-such-run") == 2

    out, err = capsys.readouterr()
    assert f"run {pending}: cancelled" in out
    assert "already finished (completed)" in err and "no run no-such-run" in err
    assert store.run_status(pending) == "cancelled"


def test_a_run_cancelled_from_elsewhere_stops_and_exits_with_130(project, capsys):
    store, home, tasks_dir, _ = project
    sleeper = home.parent / "sleeper.yaml"
    sleeper.write_text(
        yaml.safe_dump({**CONFIG.model_dump(), "params": {"script": [{"sleep": 30}]}})
    )

    def cancel_once_trials_run():  # another process, e.g. `agentoscopy cancel` or the API
        other = Store(home / "agentoscopy.db")
        try:
            while (
                not other.list_runs() or other.count_in_flight(other.list_runs()[0]["run_id"]) < 2
            ):
                time.sleep(0.05)
            other.request_cancel(other.list_runs()[0]["run_id"])
        finally:
            other.close()

    canceller = threading.Thread(target=cancel_once_trials_run)
    canceller.start()
    started = time.monotonic()
    code = cli(
        home,
        tasks_dir,
        "run",
        "--task",
        "task-a",
        "--agent",
        str(sleeper),
        "--trials",
        "3",
        "--concurrency",
        "2",
    )
    canceller.join()

    assert code == 130
    assert time.monotonic() - started < 15  # the sleeping agents were stopped, not waited out
    (run,) = store.list_runs()
    assert run["status"] == "cancelled"
    assert {row["outcome"] for row in store.trial_rows(run["run_id"])} == {"cancelled"}
