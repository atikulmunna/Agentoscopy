import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from agentoscopy.gateway.session import Usage
from agentoscopy.scheduler.recovery import cancel_run, reclaim_expired
from agentoscopy.spec import AgentConfig
from agentoscopy.storage.store import (
    SCHEMA_VERSION,
    WORKER_LOST,
    NewTrial,
    RunSettings,
    Store,
)
from agentoscopy.testing.fake_sandbox import FakeBackend
from agentoscopy.worker.trial import AttemptResult, attempt_paths, recorded_spend

CONFIG = AgentConfig(name="agent", adapter="python", entrypoint="pkg.mod:Agent")
SETTINGS = RunSettings(
    suite_id=None, suite_version=None, trials_per_task=1, budget_usd=None, seed=1,
    concurrency=1, harness_version="test",
)  # fmt: skip
T0 = 1_000_000.0  # a fixed clock for lease arithmetic


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / "agentoscopy.db")
    yield store
    store.close()


def new_run(store, trials=2):
    return store.create_run(
        CONFIG, SETTINGS, [NewTrial(f"task-{i}", 1, 0, i, 1.0) for i in range(trials)]
    )


def claim(store, run_id, worker="dead:1:0", now=T0, lease_s=30.0):
    return store.claim_next(run_id, worker, now=now, lease_s=lease_s).trial


def result(outcome="pass", cost=0.0):
    return AttemptResult(
        outcome=outcome, termination="agent_done", score=1.0, grades=[], error_code=None,
        error=None, cleanup_errors=[], usage=Usage(cost_usd=cost), duration_s=1.0,
        trajectory_path=Path("t.jsonl"), final_state_path=None,
    )  # fmt: skip


def lease_of(store, trial):
    return store.get_trial(trial.trial_id)["lease_expires_at"]


def test_a_claim_holds_a_lease_that_its_process_renews(store):
    run_id = new_run(store)
    mine = claim(store, run_id, worker="me:1:0")
    theirs = claim(store, run_id, worker="other:2:0")

    assert lease_of(store, mine) == T0 + 30
    assert store.renew_leases(run_id, ["me:1:0", "me:1:1"], until=T0 + 60) == 1
    assert (lease_of(store, mine), lease_of(store, theirs)) == (T0 + 60, T0 + 30)
    assert [t.trial_id for t in store.expired_trials(run_id, now=T0 + 45)] == [theirs.trial_id]
    assert store.expired_trials(run_id, now=T0 + 10) == []


def test_an_abandoned_attempt_keeps_its_spend_and_its_trial_is_retried(store):
    run_id = new_run(store, trials=1)
    trial = claim(store, run_id)

    assert store.abandon_attempt(
        trial, state="QUEUED", cost_usd=0.25, judge_cost_usd=0.05, now=T0 + 31
    )

    row = store.get_trial(trial.trial_id)
    (attempt,) = row["attempts"]
    assert (row["state"], row["worker_id"], row["lease_expires_at"]) == ("QUEUED", None, None)
    assert (attempt["outcome"], attempt["error_code"], attempt["cost_usd"]) == (
        "infra_error",
        WORKER_LOST,
        0.25,
    )
    assert store.run_spent(run_id) == pytest.approx(0.25)
    assert store.run_judge_spent(run_id) == pytest.approx(0.05)
    # The old worker comes back to life: its result is refused and counted only once.
    assert store.finish_attempt(trial, "dead:1:0", result(cost=0.25)) is False
    retry = claim(store, run_id, worker="alive:3:0", now=T0 + 40)
    assert (retry.trial_id, retry.attempt) == (trial.trial_id, 2)
    assert store.finish_attempt(retry, "alive:3:0", result(cost=0.1)) is True
    assert store.run_spent(run_id) == pytest.approx(0.35)
    assert store.get_trial(trial.trial_id)["outcome"] == "pass"


@pytest.mark.parametrize(
    ("state", "outcome"), [("INFRA_ERROR", "infra_error"), ("CANCELLED", "cancelled")]
)
def test_an_abandoned_attempt_can_close_its_trial(store, state, outcome):
    run_id = new_run(store, trials=1)
    trial = claim(store, run_id)

    store.abandon_attempt(trial, state=state, cost_usd=0.0, judge_cost_usd=0.0, now=T0 + 31)

    row = store.get_trial(trial.trial_id)
    assert (row["state"], row["outcome"], row["error_code"]) == (state, outcome, WORKER_LOST)


def test_a_renewed_lease_cannot_be_abandoned(store):
    run_id = new_run(store, trials=1)
    trial = claim(store, run_id)

    assert not store.abandon_attempt(
        trial, state="QUEUED", cost_usd=0.0, judge_cost_usd=0.0, now=T0 + 10
    )
    assert store.get_trial(trial.trial_id)["state"] == "PROVISIONING"
    with pytest.raises(ValueError):
        store.abandon_attempt(trial, state="COMPLETED", cost_usd=0, judge_cost_usd=0, now=T0 + 31)


def test_a_cancel_stops_queued_trials_now_and_waits_for_trials_in_flight(store):
    run_id = new_run(store, trials=3)
    in_flight = claim(store, run_id, worker="me:1:0", now=0, lease_s=1e12)

    assert store.request_cancel(run_id) == "cancelling"
    states = {row["trial_id"]: row["state"] for row in store.trial_rows(run_id)}
    assert sorted(states.values()) == ["CANCELLED", "CANCELLED", "PROVISIONING"]
    assert store.claim_next(run_id, "me:1:1").status == "empty"

    store.finish_attempt(in_flight, "me:1:0", result(outcome="cancelled"))
    assert store.finalize_cancel(run_id)
    assert store.run_status(run_id) == "cancelled"
    assert store.get_trial(in_flight.trial_id)["state"] == "CANCELLED"


def test_a_cancel_with_nothing_in_flight_is_immediate(store):
    run_id = new_run(store)

    assert store.request_cancel(run_id) == "cancelled"
    assert store.get_run(run_id)["finished_at"] is not None
    assert store.request_cancel("no-such-run") is None


def test_finished_runs_cannot_be_cancelled(store):
    run_id = new_run(store)
    store.set_run_status(run_id, "completed", finished=True)

    assert store.request_cancel(run_id) == "completed"


def test_a_retry_queued_after_a_cancel_is_cancelled_instead(store):
    run_id = new_run(store, trials=1)
    trial = claim(store, run_id, worker="me:1:0", now=0, lease_s=1e12)
    store.request_cancel(run_id)

    store.finish_attempt(trial, "me:1:0", result(outcome="infra_error"), requeue_delay_s=0)

    assert store.claim_next(run_id, "me:1:0").status == "empty"
    assert store.get_trial(trial.trial_id)["state"] == "CANCELLED"


def test_a_version_2_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    store = Store(path)
    run_id = new_run(store, trials=1)
    store.close()
    with sqlite3.connect(path) as conn:  # take the database back to how version 2 left it
        conn.execute("DROP INDEX idx_trials_lease")
        for table, column in [
            ("runs", "review_rate"),
            ("trials", "lease_expires_at"),
            ("trial_attempts", "model_calls"),
            ("trial_attempts", "model_latency_s"),
        ]:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        conn.execute("PRAGMA user_version = 2")
    conn.close()

    store = Store(path)

    assert store.get_run(run_id)["review_rate"] is None
    assert store.trial_rows(run_id)[0]["lease_expires_at"] is None
    assert claim(store, run_id) is not None
    store.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    conn.close()


def write_trajectory(tmp_path, run_id, trial, costs, cut_off_tail=False):
    trajectory, artifacts = attempt_paths(tmp_path, run_id, trial.trial_id, trial.attempt)
    trajectory.parent.mkdir(parents=True)
    lines = [
        json.dumps({"seq": i, "type": "model_response", "payload": {"cost_usd": c}})
        for i, c in enumerate(costs)
    ]
    text = "\n".join(lines) + "\n" + ('{"seq": 99, "type": "model_resp' if cut_off_tail else "")
    trajectory.write_text(text, encoding="utf-8")
    artifacts.mkdir(parents=True)
    grading = json.dumps({"seq": 0, "type": "model_response", "payload": {"cost_usd": 0.01}})
    (artifacts / "grading.jsonl").write_text(grading + "\n", encoding="utf-8")


def test_recorded_spend_survives_a_line_cut_off_by_the_crash(store, tmp_path):
    run_id = new_run(store, trials=1)
    trial = claim(store, run_id)
    write_trajectory(tmp_path, run_id, trial, [0.1, 0.2], cut_off_tail=True)

    assert recorded_spend(tmp_path, run_id, trial.trial_id, 1) == pytest.approx((0.3, 0.01))
    assert recorded_spend(tmp_path, run_id, "no-such-trial", 1) == (0.0, 0.0)


def test_reclaiming_removes_the_dead_attempts_sandboxes_and_counts_its_spend(store, tmp_path):
    run_id = new_run(store, trials=2)
    dead = claim(store, run_id, now=T0)
    live = claim(store, run_id, worker="me:1:0", now=T0 + 25)
    write_trajectory(tmp_path, run_id, dead, [0.2])
    backend = FakeBackend()

    async def scenario():
        dead_box = await backend.create("image", None, dead.trial_id)
        live_box = await backend.create("image", None, live.trial_id)
        released = await reclaim_expired(store, backend, tmp_path, run_id, now=T0 + 40)
        return released, dead_box, live_box

    released, dead_box, live_box = asyncio.run(scenario())

    assert released == [dead.trial_id]
    assert dead_box.destroyed and not live_box.destroyed
    assert store.get_trial(dead.trial_id)["state"] == "QUEUED"
    assert store.run_spent(run_id) == pytest.approx(0.2)


def test_a_reclaimed_trial_on_its_last_attempt_is_an_infra_error(store, tmp_path):
    run_id = new_run(store, trials=1)
    first = claim(store, run_id)
    store.abandon_attempt(first, state="QUEUED", cost_usd=0, judge_cost_usd=0, now=T0 + 31)
    second = claim(store, run_id, now=T0 + 40)

    released = asyncio.run(reclaim_expired(store, FakeBackend(), tmp_path, run_id, now=T0 + 80))

    assert released == [second.trial_id]
    assert store.get_trial(second.trial_id)["state"] == "INFRA_ERROR"


def test_cancelling_a_run_whose_process_died_finishes_the_cancel(store, tmp_path):
    run_id = new_run(store, trials=2)
    claim(store, run_id, now=0, lease_s=1)  # its process died long ago

    status = asyncio.run(cancel_run(store, FakeBackend(), tmp_path, run_id))

    assert status == "cancelled"
    assert {row["state"] for row in store.trial_rows(run_id)} == {"CANCELLED"}
    assert store.get_summary(run_id)["counts"]["cancelled"] == 2
