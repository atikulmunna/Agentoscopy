"""Adaptive top-up (FR-RUN-10): extra trials only where results were mixed."""

from pathlib import Path

import pytest
import yaml
from conftest import BASE_TASK

from agentoscopy.api.common import ApiError
from agentoscopy.api.runs import parse_request
from agentoscopy.cli.main import main
from agentoscopy.gateway.providers import Providers
from agentoscopy.gateway.session import Usage
from agentoscopy.launch import Directories, RunRequest, create_run
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import PinnedTask, top_up_trials
from agentoscopy.spec import load_task
from agentoscopy.storage.store import TOPPED_UP, Store
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.worker.trial import AttemptResult

TASKS = ["task-a", "task-b"]


class Alternating:
    """Grades two trials pass, then two fail, and so on, so round-robin trials of two tasks
    come out mixed for both."""

    def __init__(self):
        self.graded = 0

    def __call__(self, cmd, files):
        if cmd != "run-tests":
            return ExecResult(0, b"", b"")  # the cache cleanup before grading
        self.graded += 1
        return ExecResult(0 if (self.graded - 1) // 2 % 2 == 0 else 1, b"", b"")


@pytest.fixture
def project(tmp_path, write_task, tasks_dir, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    for task_id in TASKS:
        write_task({**BASE_TASK, "id": task_id})
        store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        yaml.safe_dump(
            {
                "name": "fixer",
                "adapter": "python",
                "entrypoint": "agentoscopy.testing.scripted_agent:ScriptedAgent",
                "params": {"script": [{"model_calls": 1}, {"write": "app.py", "content": "x"}]},
            }
        )
    )
    yield store, home, tasks_dir, agent, monkeypatch
    store.close()


def run(project, grader, *extra):
    store, home, tasks_dir, agent, monkeypatch = project
    monkeypatch.setattr("agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=grader))
    args = ["run", *[a for t in TASKS for a in ("--task", t)], "--agent", str(agent)]
    dirs = ["--home", str(home), "--tasks-dir", str(tasks_dir), "--concurrency", "1"]
    return main([*args, "--trials", "2", "--seed", "1", *dirs, *extra])


def counts(store, run_id):
    rows = store.trial_rows(run_id)
    return {task: sum(row["task_id"] == task for row in rows) for task in TASKS}


def test_mixed_tasks_get_the_extra_trials(project):
    store = project[0]

    assert run(project, Alternating(), "--top-up", "2") == 0

    run_row = store.list_runs()[0]
    assert run_row["status"] == "completed" and TOPPED_UP in run_row["flags"]
    assert counts(store, run_row["run_id"]) == {"task-a": 4, "task-b": 4}
    assert {row["state"] for row in store.trial_rows(run_row["run_id"])} == {"COMPLETED"}
    assert run_row["top_up"] == 2


def test_consistent_tasks_get_none(project):
    store = project[0]

    assert run(project, lambda cmd, files: ExecResult(0, b"", b""), "--top-up", "2") == 0

    run_row = store.list_runs()[0]
    assert counts(store, run_row["run_id"]) == {"task-a": 2, "task-b": 2}
    assert TOPPED_UP in run_row["flags"]  # considered, and nothing was mixed


def test_without_top_up_nothing_is_added(project):
    store = project[0]

    assert run(project, Alternating()) == 0

    run_row = store.list_runs()[0]
    assert counts(store, run_row["run_id"]) == {"task-a": 2, "task-b": 2}
    assert TOPPED_UP not in run_row["flags"]


def test_a_run_that_hit_its_budget_is_not_topped_up(project):
    store, _, _, agent, _ = project
    # One model call spends a little, after which no trial's full cap fits the budget.
    assert run(project, Alternating(), "--top-up", "2", "--budget-usd", "1.0") == 0

    run_row = store.list_runs()[0]
    assert "BUDGET_TRUNCATED" in run_row["flags"] and TOPPED_UP not in run_row["flags"]


def test_a_resumed_run_still_gets_its_top_up(project):
    store, home, tasks_dir, agent, monkeypatch = project
    request = RunRequest(agent=agent, tasks=tuple(TASKS), trials=2, seed=1, top_up=1)
    prepared = create_run(
        store, Directories(tasks_dir, tasks_dir, home), request, providers=Providers()
    )
    store.mark_running(prepared.run_id)
    # Its planned trials finished (round robin: each task passed once and failed once), then
    # the process died before the top-up.
    for outcome in ("pass", "pass", "fail", "fail"):
        trial = store.claim_next(prepared.run_id, "dead:1:0").trial
        store.finish_attempt(trial, "dead:1:0", finished(outcome))
    monkeypatch.setattr(
        "agentoscopy.cli.run.DockerBackend", lambda: FakeBackend(grader=Alternating())
    )

    resumed = main(
        ["run", "--resume", prepared.run_id, "--home", str(home), "--tasks-dir", str(tasks_dir)]
    )

    assert resumed == 0
    assert counts(store, prepared.run_id) == {"task-a": 3, "task-b": 3}
    assert TOPPED_UP in store.get_run(prepared.run_id)["flags"]


def finished(outcome):
    return AttemptResult(
        outcome=outcome, termination="agent_done", score=0.0, grades=[], error_code=None,
        error=None, cleanup_errors=[], usage=Usage(), duration_s=1.0,
        trajectory_path=Path("t.jsonl"), final_state_path=None,
    )  # fmt: skip


def test_top_up_trials_continue_each_tasks_numbering(make_task):
    tasks = [make_task(id=task_id) for task_id in ("mixed", "steady")]
    pinned = [PinnedTask(task, 1, None, frozenset()) for task in tasks]
    rows = [
        {"task_id": "mixed", "trial_index": 0, "dispatch_order": 0, "outcome": "pass"},
        {"task_id": "steady", "trial_index": 0, "dispatch_order": 1, "outcome": "pass"},
        {"task_id": "mixed", "trial_index": 1, "dispatch_order": 2, "outcome": "fail"},
        {"task_id": "steady", "trial_index": 1, "dispatch_order": 3, "outcome": "infra_error"},
    ]

    extra = top_up_trials(rows, pinned, 2)

    assert [(t.task_id, t.trial_index, t.dispatch_order) for t in extra] == [
        ("mixed", 2, 4),
        ("mixed", 3, 5),
    ]
    assert top_up_trials(rows, pinned, 0) == []


@pytest.mark.parametrize(
    ("value", "ok"), [(0, True), (20, True), (21, False), (-1, False), ("2", False)]
)
def test_the_api_bounds_top_up(tmp_path, value, ok):
    (tmp_path / "fixer.yaml").write_text("name: fixer\n")
    body = {"agent": "fixer", "tasks": ["task-a"], "top_up": value}

    if ok:
        assert parse_request(body, tmp_path).top_up == value
    else:
        with pytest.raises(ApiError):
            parse_request(body, tmp_path)
