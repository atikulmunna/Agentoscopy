import asyncio
import time

import pytest
from conftest import BASE_TASK, running_gateway

from agentoscopy.adapters.base import AgentResult
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.plan import (
    PlanError,
    check_budget,
    check_judge,
    pin_tasks,
    plan_trials,
)
from agentoscopy.scheduler.runner import LOW_CONFIDENCE, RunExecutor
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


def execute(
    store, tmp_path, pinned, backend, script, trials=2, budget=None, concurrency=2, **options
):
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
                **options,
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


JUDGE = {"name": "quality", "type": "llm_judge", "rubric": "Pass if minimal.", "required": False}
TAMPER = {"name": "no_tamper", "type": "tamper_check", "protected_paths": ["/workspace/tests/**"]}


def test_review_rate_samples_finished_trials_as_audits(store, tmp_path, validated):
    pinned = validated("task-a")
    backend = FakeBackend(grader=passes_when_fixed)

    execute(store, tmp_path, pinned, backend, [FIX], trials=3, review_rate=0.0)
    assert store.pending_reviews() == 0

    execute(store, tmp_path, pinned, backend, [FIX], trials=3, review_rate=1.0)
    queue = store.review_queue(limit=10)
    assert [(item["grader_name"], item["sample_source"]) for item in queue] == [("", "random")] * 3


def test_judged_trials_are_sampled_per_judge_and_unsure_verdicts_are_queued(
    store, tmp_path, validated
):
    pinned = validated("task-a", graders=[*BASE_TASK["graders"], JUDGE])
    backend = FakeBackend(grader=passes_when_fixed)

    run_id, _ = execute(
        store, tmp_path, pinned, backend, [FIX], trials=8, judge_model="mock", review_rate=1.0
    )

    queue = store.review_queue(limit=100)
    random_items = [item for item in queue if item["sample_source"] == "random"]
    unsure = [item for item in queue if item["sample_source"] == "low_confidence"]
    assert len(random_items) == 8 and {item["grader_name"] for item in random_items} == {"quality"}
    expected = sum(
        store.grade_for(row["trial_id"], 1, "quality")["metadata"]["confidence"] < LOW_CONFIDENCE
        for row in store.trial_rows(run_id)
    )
    assert len(unsure) == expected > 0
    assert store.run_judge_spent(run_id) > 0


def test_vetoed_trials_are_tagged_as_reward_hacking(store, tmp_path, validated):
    pinned = validated("task-a", graders=[*BASE_TASK["graders"], TAMPER])
    script = [FIX, {"write": "tests/test_app.py", "content": "assert True"}]

    run_id, _ = execute(store, tmp_path, pinned, FakeBackend(grader=passes_when_fixed), script)

    for row in store.trial_rows(run_id):
        assert row["outcome"] == "fail"
        tags = store.get_trial(row["trial_id"])["failure_tags"]
        assert tags == [
            {"tag": "reward_hacking", "source": "auto", "note": "tamper_check vetoed the trial"}
        ]


def test_judge_model_is_only_pinned_for_tasks_with_judges(validated):
    plain = [item.task for item in validated("task-a")]
    judged = [item.task for item in validated("task-b", graders=[*BASE_TASK["graders"], JUDGE])]

    assert check_judge(plain, "claude-opus-5-5", has_credentials=False) is None
    assert check_judge(judged, "mock", has_credentials=False) == "mock"
    assert check_judge(judged, "claude-opus-5-5", has_credentials=True) == "claude-opus-5-5"


@pytest.mark.parametrize(
    ("model", "credentials", "code"),
    [("claude-opus-5-5", False, "JUDGE_NEEDS_CREDENTIALS"), ("gpt-x", True, "UNKNOWN_JUDGE_MODEL")],
)
def test_unusable_judge_models_are_rejected(validated, model, credentials, code):
    judged = [item.task for item in validated("task-b", graders=[*BASE_TASK["graders"], JUDGE])]

    with pytest.raises(PlanError, match=code):
        check_judge(judged, model, has_credentials=credentials)


class PausingAgent:
    """Waits a while, then applies the fix, so its trials stay in flight for a bit."""

    name = "pausing"

    def __init__(self, pause_s):
        self.pause_s = pause_s

    async def setup(self, config):
        pass

    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        await asyncio.sleep(self.pause_s)
        await sandbox.write_file("app.py", b"fixed")
        return AgentResult(final_message="done")

    async def teardown(self):
        pass


def run_paused(store, tmp_path, pinned, pause_s, trials, during=None, before=None, **options):
    """Execute a run of PausingAgent trials; `during(run_id)` runs alongside the executor and
    `before(run_id)` runs first."""
    agent_config = config([])
    settings = RunSettings(
        suite_id=None, suite_version=None, trials_per_task=trials, budget_usd=None, seed=3,
        concurrency=2, harness_version="test",
    )  # fmt: skip
    run_id = store.create_run(agent_config, settings, plan_trials(pinned, trials, seed=3))
    if before:
        before(run_id)

    async def scenario():
        async with running_gateway() as gateway:
            executor = RunExecutor(
                store, run_id, pinned, agent_config, FakeBackend(grader=passes_when_fixed),
                gateway, tmp_path / "home", concurrency=2, seed=3,
                adapter_factory=lambda _: PausingAgent(pause_s), **options,
            )  # fmt: skip
            running = asyncio.create_task(executor.execute())
            if during:
                await during(run_id)
            await running

    asyncio.run(scenario())
    return run_id


def test_a_cancel_request_stops_a_running_run(store, tmp_path, validated):
    pinned = validated("task-a")

    async def cancel_after_first_trial(run_id):
        while store.progress(run_id)[0] == 0:
            await asyncio.sleep(0.05)
        store.request_cancel(run_id)

    run_id = run_paused(store, tmp_path, pinned, 0.4, trials=6, during=cancel_after_first_trial)

    outcomes = sorted(row["outcome"] for row in store.trial_rows(run_id))
    assert store.run_status(run_id) == "cancelled"
    assert "pass" in outcomes and "cancelled" in outcomes
    assert set(outcomes) == {"pass", "cancelled"}  # nothing left queued or in flight


def test_a_dead_workers_trial_is_reclaimed_and_run_again(store, tmp_path, validated):
    pinned = validated("task-a")
    dead = {}

    def die_holding_a_trial(run_id):  # a process that claimed a trial, then was killed
        dead["trial"] = store.claim_next(run_id, "dead:1:0", now=time.time() - 100, lease_s=1).trial

    run_id = run_paused(
        store, tmp_path, pinned, 0.05, trials=3, before=die_holding_a_trial, heartbeat_s=0.1
    )

    assert store.run_status(run_id) == "completed"
    assert {row["outcome"] for row in store.trial_rows(run_id)} == {"pass"}
    attempts = store.get_trial(dead["trial"].trial_id)["attempts"]
    assert [(a["attempt"], a["error_code"]) for a in attempts] == [(1, "WORKER_LOST"), (2, None)]


def test_leases_are_renewed_while_trials_run(store, tmp_path, validated):
    pinned = validated("task-a")

    run_id = run_paused(store, tmp_path, pinned, 1.0, trials=2, lease_s=0.4, heartbeat_s=0.1)

    rows = store.trial_rows(run_id)
    assert {row["outcome"] for row in rows} == {"pass"}
    assert {row["attempt"] for row in rows} == {1}  # never reclaimed from under its worker
