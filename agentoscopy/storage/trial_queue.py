"""The job queue (AD-2) and the spend it reserves against.

Workers claim queued trials with a conditional update and hold them under a lease their
process keeps renewing. An attempt is finalised only by the worker that still holds it
(FR-EXE-01, FR-EXE-05), and a trial whose lease ran out because its process died can be
reclaimed (FR-EXE-06).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import replace
from typing import Any

from agentoscopy.graders.base import GradeResult
from agentoscopy.storage.database import Database, timestamp
from agentoscopy.storage.records import IN_FLIGHT_STATES, Claim, ClaimedTrial
from agentoscopy.worker.trial import AttemptResult

LEASE_S = 30.0  # a worker's process renews its leases well within this (FR-EXE-01)
BUDGET_EPSILON_USD = 1e-9
WORKER_LOST = "WORKER_LOST"
WORKER_LOST_MESSAGE = "the worker stopped renewing its lease: its process died or hung"
HELD = "WHERE trial_id = ? AND attempt = ? AND worker_id = ? AND state IN (?, ?, ?)"


class TrialQueue(Database):
    def claim_next(
        self, run_id: str, worker_id: str, now: float | None = None, lease_s: float = LEASE_S
    ) -> Claim:
        """Claim the next trial in dispatch order if the run budget can reserve its cap."""
        now = time.time() if now is None else now
        with self._transaction() as conn:
            if self.run_status(run_id) == "cancelling":
                cancel_queued(conn, run_id)  # including retries queued after the request
                return Claim("empty")
            ready = conn.execute(
                "SELECT * FROM trials WHERE run_id = ? AND state = 'QUEUED' AND not_before <= ? "
                "ORDER BY dispatch_order LIMIT 1",
                (run_id, now),
            ).fetchone()
            if ready is None:
                next_at = conn.execute(
                    "SELECT MIN(not_before) FROM trials WHERE run_id = ? AND state = 'QUEUED'",
                    (run_id,),
                ).fetchone()[0]
                return Claim("empty") if next_at is None else Claim("waiting", ready_at=next_at)
            if not self._affordable(run_id, ready["cost_cap_usd"]):
                return Claim("unaffordable")
            attempt = ready["attempt"] + 1
            conn.execute(
                "UPDATE trials SET state = 'PROVISIONING', attempt = ?, worker_id = ?, "
                "reserved_usd = cost_cap_usd, lease_expires_at = ? "
                "WHERE trial_id = ? AND state = 'QUEUED'",
                (attempt, worker_id, now + lease_s, ready["trial_id"]),
            )
            conn.execute(
                "INSERT INTO trial_attempts (trial_id, attempt, worker_id, started_at) "
                "VALUES (?, ?, ?, ?)",
                (ready["trial_id"], attempt, worker_id, timestamp()),
            )
        return Claim("claimed", trial=replace(_claimed(ready), attempt=attempt))

    def renew_leases(self, run_id: str, worker_ids: list[str], until: float) -> int:
        """Extend the leases on every trial these workers hold (the heartbeat, FR-EXE-01)."""
        marks = ", ".join("?" * len(worker_ids))
        cursor = self._conn.execute(
            f"UPDATE trials SET lease_expires_at = ? WHERE run_id = ? AND worker_id IN ({marks}) "
            "AND state IN (?, ?, ?)",
            (until, run_id, *worker_ids, *IN_FLIGHT_STATES),
        )
        return cursor.rowcount

    def expired_trials(self, run_id: str, now: float | None = None) -> list[ClaimedTrial]:
        """Trials in flight whose lease ran out: the process holding them died or hung."""
        rows = self._conn.execute(
            "SELECT * FROM trials WHERE run_id = ? AND state IN (?, ?, ?) "
            "AND COALESCE(lease_expires_at, 0) < ? ORDER BY dispatch_order",
            (run_id, *IN_FLIGHT_STATES, time.time() if now is None else now),
        )
        return [_claimed(row) for row in rows]

    def abandon_attempt(
        self,
        trial: ClaimedTrial,
        *,
        state: str,
        cost_usd: float,
        judge_cost_usd: float,
        now: float | None = None,
    ) -> bool:
        """Close an attempt whose lease ran out (WF-04 E5). The attempt becomes an infra error
        (`WORKER_LOST`) that keeps what it spent, and the trial moves to `state`: QUEUED for
        a retry, INFRA_ERROR, or CANCELLED. Changes nothing if the lease was renewed since."""
        if state not in ("QUEUED", "INFRA_ERROR", "CANCELLED"):
            raise ValueError(f"cannot abandon an attempt into state {state}")
        now = time.time() if now is None else now
        outcome = {"INFRA_ERROR": "infra_error", "CANCELLED": "cancelled"}.get(state)
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE trials SET state = ?, outcome = ?, error_code = ?, error = ?, "
                "cost_usd = ?, worker_id = NULL, reserved_usd = 0, lease_expires_at = NULL, "
                "not_before = ? WHERE trial_id = ? AND attempt = ? AND state IN (?, ?, ?) "
                "AND COALESCE(lease_expires_at, 0) < ?",
                (
                    state,
                    outcome,
                    WORKER_LOST,
                    WORKER_LOST_MESSAGE,
                    cost_usd,
                    now,
                    trial.trial_id,
                    trial.attempt,
                    *IN_FLIGHT_STATES,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                return False
            _record_lost_attempt(conn, trial, cost_usd, judge_cost_usd)
            return True

    def set_trial_state(self, trial: ClaimedTrial, worker_id: str, state: str) -> None:
        self._conn.execute(
            "UPDATE trials SET state = ? WHERE trial_id = ? AND attempt = ? AND worker_id = ?",
            (state, trial.trial_id, trial.attempt, worker_id),
        )

    def finish_attempt(
        self,
        trial: ClaimedTrial,
        worker_id: str,
        result: AttemptResult,
        requeue_delay_s: float | None = None,
    ) -> bool:
        """Record an attempt and finalise or re-queue its trial in one transaction.

        Returns False, changing nothing but the attempt's own record, if this worker no longer
        holds the trial (FR-EXE-05). The attempt's spend is recorded either way.
        """
        held = (trial.trial_id, trial.attempt, worker_id, *IN_FLIGHT_STATES)
        with self._transaction() as conn:
            _record_attempt(conn, trial, result)
            if requeue_delay_s is not None:
                return _requeue(conn, held, result, time.time() + requeue_delay_s)
            if not _finalise(conn, held, result):
                return False
            _insert_grades(conn, trial, result.grades)
            return True

    def count_in_flight(self, run_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM trials WHERE run_id = ? AND state IN (?, ?, ?)",
            (run_id, *IN_FLIGHT_STATES),
        ).fetchone()[0]

    def skip_queued(self, run_id: str) -> int:
        cursor = self._conn.execute(
            "UPDATE trials SET state = 'SKIPPED', outcome = 'skipped' "
            "WHERE run_id = ? AND state = 'QUEUED'",
            (run_id,),
        )
        return cursor.rowcount

    def cancel_unfinished(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE trials SET state = 'CANCELLED', outcome = 'cancelled', reserved_usd = 0 "
            "WHERE run_id = ? AND state IN ('QUEUED', ?, ?, ?)",
            (run_id, *IN_FLIGHT_STATES),
        )

    # Spend --------------------------------------------------------------------------------

    def run_spent(self, run_id: str) -> float:
        """Agent spend across every attempt, including ones that ended in infra errors."""
        return self._conn.execute(
            "SELECT COALESCE(SUM(a.cost_usd), 0) FROM trial_attempts a "
            "JOIN trials t USING (trial_id) WHERE t.run_id = ?",
            (run_id,),
        ).fetchone()[0]

    def run_judge_spent(self, run_id: str) -> float:
        """Judge spend, kept apart from agent spend and the run budget (NFR-COST-02)."""
        return self._conn.execute(
            "SELECT COALESCE(SUM(a.judge_cost_usd), 0) FROM trial_attempts a "
            "JOIN trials t USING (trial_id) WHERE t.run_id = ?",
            (run_id,),
        ).fetchone()[0]

    def _affordable(self, run_id: str, cost_cap_usd: float) -> bool:
        budget = self._conn.execute(
            "SELECT budget_usd FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        if budget is None:
            return True
        reserved = self._conn.execute(
            "SELECT COALESCE(SUM(reserved_usd), 0) FROM trials WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        return self.run_spent(run_id) + reserved + cost_cap_usd <= budget + BUDGET_EPSILON_USD


def cancel_queued(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "UPDATE trials SET state = 'CANCELLED', outcome = 'cancelled', reserved_usd = 0 "
        "WHERE run_id = ? AND state = 'QUEUED'",
        (run_id,),
    )


def _claimed(row: sqlite3.Row) -> ClaimedTrial:
    return ClaimedTrial(
        row["trial_id"], row["task_id"], row["task_version"], row["trial_index"], row["attempt"]
    )


def _record_lost_attempt(
    conn: sqlite3.Connection, trial: ClaimedTrial, cost_usd: float, judge_cost_usd: float
) -> None:
    conn.execute(
        "UPDATE trial_attempts SET outcome = 'infra_error', error_code = ?, error = ?, "
        "cost_usd = ?, judge_cost_usd = ?, ended_at = ? WHERE trial_id = ? AND attempt = ?",
        (
            WORKER_LOST,
            WORKER_LOST_MESSAGE,
            cost_usd,
            judge_cost_usd,
            timestamp(),
            trial.trial_id,
            trial.attempt,
        ),
    )


def _record_attempt(conn: sqlite3.Connection, trial: ClaimedTrial, result: AttemptResult) -> None:
    conn.execute(
        "UPDATE trial_attempts SET outcome = ?, error_code = ?, error = ?, cost_usd = ?, "
        "judge_cost_usd = ?, trajectory_uri = ?, ended_at = ?, model_calls = ?, "
        "model_latency_s = ? WHERE trial_id = ? AND attempt = ?",
        (
            result.outcome,
            result.error_code,
            result.error,
            result.usage.cost_usd,
            result.judge_cost_usd,
            str(result.trajectory_path),
            timestamp(),
            result.usage.model_calls,
            result.usage.model_latency_s,
            trial.trial_id,
            trial.attempt,
        ),
    )


def _requeue(
    conn: sqlite3.Connection, held: tuple[Any, ...], result: AttemptResult, not_before: float
) -> bool:
    """Put a held trial back in the queue for a retry; False if this worker lost it."""
    cursor = conn.execute(
        "UPDATE trials SET state = 'QUEUED', worker_id = NULL, reserved_usd = 0, "
        f"lease_expires_at = NULL, not_before = ?, error_code = ?, error = ? {HELD}",
        (not_before, result.error_code, result.error, *held),
    )
    return cursor.rowcount == 1


def _finalise(conn: sqlite3.Connection, held: tuple[Any, ...], result: AttemptResult) -> bool:
    """Write a held trial's final result; False if this worker lost it."""
    cursor = conn.execute(
        "UPDATE trials SET state = ?, outcome = ?, score = ?, termination = ?, "
        "error_code = ?, error = ?, steps = ?, input_tokens = ?, output_tokens = ?, "
        "cache_read_tokens = ?, cache_write_tokens = ?, cost_usd = ?, duration_s = ?, "
        "judge_cost_usd = ?, reserved_usd = 0, lease_expires_at = NULL, "
        f"trajectory_uri = ?, final_state_uri = ? {HELD}",
        (
            _final_state(result),
            result.outcome,
            result.score,
            result.termination,
            result.error_code,
            result.error,
            *_usage_columns(result),
            result.duration_s,
            result.judge_cost_usd,
            str(result.trajectory_path),
            str(result.final_state_path) if result.final_state_path else None,
            *held,
        ),
    )
    return cursor.rowcount == 1


def _insert_grades(
    conn: sqlite3.Connection, trial: ClaimedTrial, grades: list[GradeResult]
) -> None:
    conn.executemany(
        "INSERT INTO grades (trial_id, attempt, grader_name, kind, score, passed, "
        "rationale, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                trial.trial_id,
                trial.attempt,
                grade.name,
                grade.kind,
                grade.score,
                grade.passed,
                grade.rationale,
                json.dumps(grade.metadata),
            )
            for grade in grades
        ],
    )


def _final_state(result: AttemptResult) -> str:
    return {"infra_error": "INFRA_ERROR", "cancelled": "CANCELLED"}.get(result.outcome, "COMPLETED")


def _usage_columns(result: AttemptResult) -> tuple[Any, ...]:
    usage = result.usage
    return (
        usage.steps,
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.cost_usd,
    )
