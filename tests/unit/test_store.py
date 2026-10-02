from pathlib import Path

import pytest

from agentoscopy.gateway.session import Usage
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.storage.store import NewTrial, RunSettings, Store
from agentoscopy.worker.trial import AttemptResult

CONFIG = AgentConfig(name="agent", adapter="python", entrypoint="pkg.mod:Agent")


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "agentoscopy.db")
    yield store
    store.close()


def settings(budget_usd=None):
    return RunSettings(
        suite_id=None,
        suite_version=None,
        trials_per_task=1,
        budget_usd=budget_usd,
        seed=1,
        concurrency=1,
        harness_version="test",
    )


def new_trials(*caps):
    return [NewTrial(f"task-{index}", 1, 0, index, cap) for index, cap in enumerate(caps)]


def result(outcome="pass", cost=0.0, error_code=None):
    return AttemptResult(
        outcome=outcome,
        termination="agent_done",
        score=1.0,
        grades=[],
        error_code=error_code,
        error=None,
        cleanup_errors=[],
        usage=Usage(cost_usd=cost),
        duration_s=1.0,
        trajectory_path=Path("t.jsonl"),
        final_state_path=None,
    )


def test_identical_content_reuses_its_version(store, make_task, write_task, tasks_dir):
    first = make_task()
    assert store.save_task_version(first, "img-1", [], ["tests"]) == 1

    write_task(files={"fixtures/app.py": "changed"})
    changed = load_task(tasks_dir, "demo-task")
    assert store.save_task_version(changed, "img-2", [], []) == 2

    (tasks_dir / "demo-task" / "fixtures" / "app.py").unlink()  # revert to the first content
    assert store.save_task_version(load_task(tasks_dir, "demo-task"), "img-1", [], []) == 1
    assert store.task_version("demo-task", first.content_hash).verified_graders == frozenset()


def test_suite_versions_change_only_with_their_tasks(store):
    assert store.save_suite_version("smoke", [("a", 1)]) == 1
    assert store.save_suite_version("smoke", [("a", 1)]) == 1
    assert store.save_suite_version("smoke", [("a", 2)]) == 2


def test_claims_follow_dispatch_order(store):
    run_id = store.create_run(CONFIG, settings(), new_trials(1.0, 1.0, 1.0))

    claimed = [store.claim_next(run_id, "w").trial.task_id for _ in range(3)]

    assert claimed == ["task-0", "task-1", "task-2"]
    assert store.claim_next(run_id, "w").status == "empty"
    assert store.count_in_flight(run_id) == 3


def test_claims_reserve_cost_caps_against_the_run_budget(store):
    run_id = store.create_run(CONFIG, settings(budget_usd=1.0), new_trials(0.6, 0.6))

    first = store.claim_next(run_id, "w")
    assert first.status == "claimed"
    assert store.claim_next(run_id, "w").status == "unaffordable"  # 0.6 reserved + 0.6 > 1.0

    store.finish_attempt(first.trial, "w", result(cost=0.3))
    assert store.claim_next(run_id, "w").status == "claimed"  # 0.3 spent + 0.6 <= 1.0
    assert store.run_spent(run_id) == pytest.approx(0.3)


def test_only_the_holding_worker_can_finish_an_attempt(store):
    run_id = store.create_run(CONFIG, settings(), new_trials(1.0))
    trial = store.claim_next(run_id, "worker-a").trial

    assert store.finish_attempt(trial, "worker-b", result(cost=0.2)) is False
    assert store.trial_rows(run_id)[0]["state"] == "PROVISIONING"
    assert store.finish_attempt(trial, "worker-a", result(cost=0.2)) is True
    row = store.trial_rows(run_id)[0]
    assert (row["state"], row["outcome"], row["reserved_usd"]) == ("COMPLETED", "pass", 0)
    # The refused call still recorded the attempt's spend; it is counted once.
    assert store.run_spent(run_id) == pytest.approx(0.2)


def test_requeued_trials_wait_out_their_backoff(store):
    run_id = store.create_run(CONFIG, settings(), new_trials(1.0))
    trial = store.claim_next(run_id, "w").trial

    store.finish_attempt(
        trial, "w", result("infra_error", 0.1, "SANDBOX_ERROR"), requeue_delay_s=60
    )
    waiting = store.claim_next(run_id, "w")
    retried = store.claim_next(run_id, "w", now=waiting.ready_at + 1)

    assert waiting.status == "waiting"
    assert retried.trial.attempt == 2
    assert store.run_spent(run_id) == pytest.approx(0.1)


def test_skip_and_cancel_close_out_unfinished_trials(store):
    run_id = store.create_run(CONFIG, settings(), new_trials(1.0, 1.0, 1.0))
    store.claim_next(run_id, "w")

    assert store.skip_queued(run_id) == 2
    store.cancel_unfinished(run_id)

    outcomes = sorted(row["outcome"] for row in store.trial_rows(run_id))
    assert outcomes == ["cancelled", "skipped", "skipped"]
    assert store.progress(run_id) == (3, 3)


def test_run_flags_and_summaries_round_trip(store):
    run_id = store.create_run(CONFIG, settings(), new_trials(1.0))
    store.add_run_flag(run_id, "BUDGET_TRUNCATED")
    store.add_run_flag(run_id, "BUDGET_TRUNCATED")
    store.save_summary(run_id, {"macro_pass_rate": 0.5})

    run = store.get_run(run_id)
    assert run["flags"] == ["BUDGET_TRUNCATED"]
    assert run["config_name"] == "agent"
    assert store.get_summary(run_id) == {"macro_pass_rate": 0.5}
