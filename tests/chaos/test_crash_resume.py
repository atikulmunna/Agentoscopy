"""Crash injection (NFR-MNT-02, acceptance criterion 4): kill the orchestrator process while
trials are in flight, resume the run, and compare it with a run that was never interrupted."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import BASE_TASK, running_gateway

from agentoscopy.launch import Directories, resume_run
from agentoscopy.scheduler.plan import pin_tasks, plan_trials
from agentoscopy.scheduler.runner import RunExecutor
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import RunSettings, Store
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.worker.trial import recorded_spend

HERE = Path(__file__).resolve().parent
TASK_IDS = ["task-a", "task-b"]
TRIALS = 4
LEASE_S = 1.0
CONFIG = AgentConfig(
    name="slow", adapter="python", entrypoint="slow_agent:SlowAgent", params={"pause_s": 0.6}
)


@pytest.fixture
def world(tmp_path, write_task, tasks_dir):
    home = tmp_path / "home"
    store = Store(home / "agentoscopy.db")
    for task_id in TASK_IDS:
        write_task({**BASE_TASK, "id": task_id})
        store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
    yield store, home, tasks_dir
    store.close()


def new_run(store, tasks_dir):
    pinned = pin_tasks(store, tasks_dir, TASK_IDS)
    settings = RunSettings(
        suite_id=None, suite_version=None, trials_per_task=TRIALS, budget_usd=None, seed=5,
        concurrency=2, harness_version="test",
    )  # fmt: skip
    return store.create_run(CONFIG, settings, plan_trials(pinned, TRIALS, seed=5))


def execute_here(store, home, tasks_dir, run_id):
    """Run (or resume) the run in this process, as `agentoscopy run --resume` would."""
    prepared = resume_run(
        store, Directories(tasks_dir, tasks_dir, home), run_id, has_credentials=False
    )
    from orchestrator import passes_when_fixed

    async def scenario():
        async with running_gateway() as gateway:
            await RunExecutor(
                store, run_id, prepared.pinned, prepared.config,
                FakeBackend(grader=passes_when_fixed), gateway, home, concurrency=2, seed=5,
                lease_s=LEASE_S, heartbeat_s=LEASE_S / 4,
            ).execute()  # fmt: skip

    asyncio.run(scenario())


def kill_mid_run(store, home, tasks_dir, run_id):
    """Start the orchestrator in its own process and kill it while a trial is in flight with
    a model call paid for. Returns how many trials were in flight when it died.

    The first trials start together and finish together, so the kill waits for a trial that
    was claimed after one finished: its pause has only just begun."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "orchestrator.py"),
            str(home),
            str(tasks_dir),
            run_id,
            str(LEASE_S),
        ],
        env=env,
    )
    try:
        deadline, first_wave = time.monotonic() + 60, None
        while True:
            assert process.poll() is None, "the orchestrator exited before it could be killed"
            assert time.monotonic() < deadline, "the run never got going"
            spending = spending_in_flight(store, home, run_id)
            if store.progress(run_id)[0] >= 1:
                if first_wave is None:
                    first_wave = spending
                elif spending - first_wave:
                    break
            time.sleep(0.02)
    finally:
        process.kill()  # SIGKILL or TerminateProcess: no cleanup runs, like a crash
        process.wait()
    return store.count_in_flight(run_id)


def spending_in_flight(store, home, run_id):
    """Trials in flight whose current attempt has paid for a model call."""
    return {
        r["trial_id"]
        for r in store.trial_rows(run_id)
        if r["state"] in ("PROVISIONING", "RUNNING", "GRADING")
        and recorded_spend(home, run_id, r["trial_id"], r["attempt"])[0] > 0
    }


def test_a_killed_run_resumes_without_losing_or_repeating_trials(world):
    store, home, tasks_dir = world
    crashed = new_run(store, tasks_dir)

    in_flight_at_crash = kill_mid_run(store, home, tasks_dir, crashed)
    assert in_flight_at_crash >= 1
    assert store.run_status(crashed) == "running"  # nothing marked it finished
    execute_here(store, home, tasks_dir, crashed)

    uninterrupted = new_run(store, tasks_dir)
    execute_here(store, home, tasks_dir, uninterrupted)

    crashed_rows, clean_rows = store.trial_rows(crashed), store.trial_rows(uninterrupted)
    assert store.run_status(crashed) == "completed"
    assert len(crashed_rows) == len(clean_rows) == len(TASK_IDS) * TRIALS
    assert [r["outcome"] for r in crashed_rows] == [r["outcome"] for r in clean_rows]
    assert {r["outcome"] for r in crashed_rows} == {"pass"}

    lost = [store.get_trial(r["trial_id"]) for r in crashed_rows if r["attempt"] > 1]
    assert len(lost) == in_flight_at_crash
    for trial in lost:
        first, second = trial["attempts"]  # one abandoned attempt, then one that finished
        assert (first["outcome"], first["error_code"]) == ("infra_error", "WORKER_LOST")
        # What the dead attempt spent, as its trajectory recorded it, still counts.
        logged, _ = recorded_spend(home, crashed, trial["trial_id"], 1)
        assert first["cost_usd"] == pytest.approx(logged)
        assert second["outcome"] == "pass"
        assert [g["grader_name"] for g in trial["grades"]] == ["tests"]  # graded exactly once
    abandoned = sum(t["attempts"][0]["cost_usd"] for t in lost)
    assert abandoned > 0
    assert store.run_spent(crashed) == pytest.approx(store.run_spent(uninterrupted) + abandoned)
