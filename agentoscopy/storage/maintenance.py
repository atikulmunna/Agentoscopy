"""Queries for keeping the harness tidy and observable: garbage collection and metrics."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from agentoscopy.storage.database import Database
from agentoscopy.storage.records import IN_FLIGHT_STATES, RUN_LISTING, run_from_row


class Maintenance(Database):
    # Garbage collection -------------------------------------------------------------------

    def runs_finished_before(self, cutoff: datetime) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"{RUN_LISTING} WHERE runs.status IN ('completed', 'cancelled', 'failed') "
            "AND runs.finished_at < ? ORDER BY runs.finished_at",
            (cutoff.isoformat(timespec="seconds"),),
        )
        return [run_from_row(row) for row in rows]

    def pending_reviews_for_run(self, run_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM reviews r JOIN trials t USING (trial_id) "
            "WHERE t.run_id = ? AND r.reviewed_at IS NULL",
            (run_id,),
        ).fetchone()[0]

    def live_trial_ids(self, now: float | None = None) -> set[str]:
        """Trials in flight whose lease is still held: their sandboxes are in use."""
        rows = self._conn.execute(
            "SELECT trial_id FROM trials WHERE state IN (?, ?, ?) AND lease_expires_at >= ?",
            (*IN_FLIGHT_STATES, time.time() if now is None else now),
        )
        return {row["trial_id"] for row in rows}

    # Metrics (NFR-OBS-02) -----------------------------------------------------------------

    def metrics(self, duration_buckets: tuple[float, ...]) -> dict[str, Any]:
        """Totals across every run, for the Prometheus endpoint."""
        conn = self._conn
        bucket_sums = ", ".join(f"SUM(duration_s <= {float(le)!r})" for le in duration_buckets)
        durations = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(duration_s), 0), {bucket_sums} FROM trials "
            "WHERE state IN ('COMPLETED', 'INFRA_ERROR', 'CANCELLED') AND duration_s IS NOT NULL"
        ).fetchone()
        attempts = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COALESCE(SUM(judge_cost_usd), 0), "
            "COALESCE(SUM(model_calls), 0), COALESCE(SUM(model_latency_s), 0) FROM trial_attempts"
        ).fetchone()
        return {
            "trial_states": dict(conn.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")),
            "trial_outcomes": dict(
                conn.execute(
                    "SELECT outcome, COUNT(*) FROM trials WHERE outcome IS NOT NULL "
                    "GROUP BY outcome"
                )
            ),
            "attempt_outcomes": dict(
                conn.execute(
                    "SELECT outcome, COUNT(*) FROM trial_attempts WHERE outcome IS NOT NULL "
                    "GROUP BY outcome"
                )
            ),
            "infra_errors": dict(
                conn.execute(
                    "SELECT error_code, COUNT(*) FROM trial_attempts "
                    "WHERE outcome = 'infra_error' GROUP BY error_code"
                )
            ),
            "run_statuses": dict(conn.execute("SELECT status, COUNT(*) FROM runs GROUP BY status")),
            "durations": {
                "count": durations[0],
                "sum": durations[1],
                "buckets": [count or 0 for count in durations[2:]],
            },
            "agent_spend_usd": attempts[0],
            "judge_spend_usd": attempts[1],
            "model_calls": attempts[2],
            "model_latency_s": attempts[3],
        }
