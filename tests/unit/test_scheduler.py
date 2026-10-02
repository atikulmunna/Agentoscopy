import asyncio

import pytest
from conftest import BASE_TASK, running_gateway

from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import PlanError, check_budget, pin_tasks, plan_trials
from agentoscopy.scheduler.runner import RunExecutor
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import RunSettings, Store
from agentoscopy.testing.fake_sandbox import FakeBackend, exit_with
from agentoscopy.testing.scripted_agent import ScriptedAgent

FIX = {"write": "app.py", "content": "fixed"}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "agentoscopy.db")
    yield store
    store.close()


@pytest.fixture
def validated(store, write_task, tasks_dir):
    """Write and record as validated the given task ids; returns their pinned versions."""

    def _validated(*task_ids, **overrides):
        for task_id in task_ids:
            write_task({**BASE_TASK, "id": task_id, **overrides})
            store.save_task_version(load_task(tasks_dir, task_id), "fake-image", [], [])
        return pin_tasks(store, tasks_dir, list(task_ids))

    return _validated


def config(script):
    return AgentConfig(
        name="scripted",
        adapter="python",
        entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
        params={"script": script},
    )


def execute(store, tmp_path, pinned, backend, script, trials=2, budget=None, concurrency=2):
    settings = RunSettings(
        suite_id=None,
        suite_version=None,
        trials_per_task=trials,
        budget_usd=budget,
        seed=3,
        concurrency=concurrency,
        harness_version="test",
    )
    agent_config = config(script)
    run_id = store.create_run(agent_config, settings, plan_trials(pinned, trials, seed=3))
    progress = []

    async def scenario():
        async with running_gateway() as gateway:
            executor = RunExecutor(
                store,
                run_id,
                pinned,
                agent_config,
                backend,
                gateway,
                tmp_path / "home",
                concurrency=concurrency,
                seed=3,
                adapter_factory=lambda _: ScriptedAgent(),
                on_progress=progress.append,
                retry_backoff_s=0.01,
            )
            await executor.execute()

    asyncio.run(scenario())
    return run_id, progress


def test_plan_is_round_robin_by_trial_index(validated):
    pinned = validated("task-a", "task-b", "task-c")

    trials = sorted(plan_trials(pinned, trials_per_task=2, seed=1), key=lambda t: t.dispatch_order)

    assert [trial.trial_index for trial in trials] == [0, 0, 0, 1, 1, 1]
    first_round = [trial.task_id for trial in trials[:3]]
    assert sorted(first_round) == ["task-a", "task-b", "task-c"]
    assert [trial.task_id for trial in trials[3:]] == first_round  # same order every round


def test_unvalidated_or_changed_tasks_cannot_run(store, validated, write_task, tasks_dir):
    validated("task-a")
    write_task({**BASE_TASK, "id": "task-a", "instructions": "Changed after validation."})

    with pytest.raises(PlanError, match="UNVALIDATED_TASKS: task-a"):
        pin_tasks(store, tasks_dir, ["task-a"])


def test_budget_below_a_trial_cap_is_rejected(validated):
    pinned = validated("task-a", budget={"max_cost_usd": 2.0})

    with pytest.raises(PlanError, match="BUDGET_BELOW_TRIAL_CAP"):
        check_budget(pinned, 1.0)


def test_run_completes_every_trial_within_the_concurrency_limit(store, tmp_path, validated):
    pinned = validated("task-a", "task-b", "task-c")
    backend = FakeBackend(grader=passes_when_fixed)

    run_id, progress = execute(store, tmp_path, pinned, backend, [{"model_calls": 1}, FIX])

    rows = store.trial_rows(run_id)
    assert [row["outcome"] for row in rows] == ["pass"] * 6
    assert all(row["steps"] == 1 and row["cost_usd"] > 0 for row in rows)
    assert backend.max_live_agents <= 2
    assert store.get_run(run_id)["status"] == "completed"
    assert progress[-1].finished == progress[-1].total == 6


def test_retryable_infra_errors_are_retried(store, tmp_path, validated):
    pinned = validated("task-a")
    backend = FakeBackend(grader=passes_when_fixed, create_failures=1)

    run_id, progress = execute(store, tmp_path, pinned, backend, [FIX], trials=1, concurrency=1)

    (row,) = store.trial_rows(run_id)
    assert (row["outcome"], row["attempt"]) == ("pass", 2)
    assert [item.retrying for item in progress] == [True, False]


def test_non_retryable_infra_errors_flag_the_task(store, tmp_path, validated):
    pinned = validated("task-a", environment={"base_image": "img@sha256:abc", "setup": ["prepare"]})
    backend = FakeBackend(agent_handler=exit_with(1))

    run_id, _ = execute(store, tmp_path, pinned, backend, [FIX], trials=1, concurrency=1)

    (row,) = store.trial_rows(run_id)
    assert (row["state"], row["error_code"], row["attempt"]) == ("INFRA_ERROR", "SETUP_BROKEN", 1)
    version = store.task_version("task-a", pinned[0].task.content_hash)
    assert version.flags == ["SETUP_BROKEN"]


def test_run_budget_is_never_exceeded_and_the_rest_is_skipped(store, tmp_path, validated):
    pinned = validated("task-a", "task-b", "task-c", budget={"max_cost_usd": 0.01})
    spend_everything = [{"model_calls": 1000, "max_tokens": 4000}]

    run_id, _ = execute(store, tmp_path, pinned, FakeBackend(), spend_everything, budget=0.025)

    outcomes = [row["outcome"] for row in store.trial_rows(run_id)]
    assert store.run_spent(run_id) <= 0.025
    assert outcomes.count("skipped") >= 3
    assert "skipped" in outcomes and "pass" in outcomes
    assert store.get_run(run_id)["flags"] == ["BUDGET_TRUNCATED"]
    assert all(
        row["termination"] == "budget_cost"
        for row in store.trial_rows(run_id)
        if row["outcome"] == "pass"
    )


def test_critical_tasks_get_enough_trials_to_reach_significance(validated):
    pinned = validated("normal-task") + validated("key-task", critical=True)

    trials = plan_trials(pinned, trials_per_task=2, seed=1)

    per_task = {
        task_id: sum(t.task_id == task_id for t in trials)
        for task_id in ("normal-task", "key-task")
    }
    assert per_task == {"normal-task": 2, "key-task": 6}
    assert len({trial.dispatch_order for trial in trials}) == len(trials)
