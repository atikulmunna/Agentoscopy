"""Reading results back: trials, a run's pinned tasks and config, summaries, and CI baselines."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agentoscopy.storage.database import Database, timestamp
from agentoscopy.storage.records import TaskVersion, task_version_from_row


class RunReports(Database):
    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        if row is None:
            return None
        trial = dict(row)
        run = self._conn.execute(
            "SELECT mode, labels FROM runs WHERE run_id = ?", (trial["run_id"],)
        ).fetchone()
        trial["run_mode"], trial["run_labels"] = run["mode"], json.loads(run["labels"])
        trial["grades"] = [
            {
                **dict(grade),
                "passed": bool(grade["passed"]),
                "metadata": json.loads(grade["metadata"]),
            }
            for grade in self._conn.execute(
                "SELECT grader_name, kind, score, passed, rationale, metadata FROM grades "
                "WHERE trial_id = ? AND attempt = ? ORDER BY rowid",
                (trial_id, trial["attempt"]),
            )
        ]
        trial["attempts"] = [
            dict(attempt)
            for attempt in self._conn.execute(
                "SELECT attempt, worker_id, outcome, error_code, error, cost_usd, trajectory_uri, "
                "started_at, ended_at FROM trial_attempts WHERE trial_id = ? ORDER BY attempt",
                (trial_id,),
            )
        ]
        trial["failure_tags"] = [
            dict(tag)
            for tag in self._conn.execute(
                "SELECT tag, source, note FROM failure_tags WHERE trial_id = ? ORDER BY tag",
                (trial_id,),
            )
        ]
        return trial

    def trial_rows(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM trials WHERE run_id = ? ORDER BY task_id, trial_index", (run_id,)
        )
        return [dict(row) for row in rows]

    def progress(self, run_id: str) -> tuple[int, int]:
        """(finished trials, total trials)."""
        row = self._conn.execute(
            "SELECT SUM(state IN ('COMPLETED', 'INFRA_ERROR', 'SKIPPED', 'CANCELLED')), COUNT(*) "
            "FROM trials WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row[0] or 0, row[1]

    def tasks_not_passed(self, run_id: str) -> set[str] | None:
        """Tasks with a trial that failed or hit an infra error in the run (None: no run)."""
        if self.run_status(run_id) is None:
            return None
        rows = self._conn.execute(
            "SELECT DISTINCT task_id FROM trials WHERE run_id = ? "
            "AND outcome IN ('fail', 'infra_error')",
            (run_id,),
        )
        return {row["task_id"] for row in rows}

    def historical_mean_cost(self, task_id: str, digest: str) -> float | None:
        return self._conn.execute(
            "SELECT AVG(t.cost_usd) FROM trials t JOIN runs r USING (run_id) "
            "WHERE t.task_id = ? AND r.config_hash = ? AND t.outcome IN ('pass', 'fail') "
            "AND r.mode = 'live'",  # replays cost nothing, which says nothing
            (task_id, digest),
        ).fetchone()[0]

    # What a run pinned --------------------------------------------------------------------

    def task_spec(self, task_id: str, version: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT spec FROM task_versions WHERE task_id = ? AND version = ?", (task_id, version)
        ).fetchone()
        return json.loads(row["spec"]) if row else {}

    def run_task_versions(self, run_id: str) -> list[TaskVersion]:
        """The task versions a run pinned, so it can be resumed exactly as it started."""
        rows = self._conn.execute(
            "SELECT v.* FROM task_versions v WHERE EXISTS (SELECT 1 FROM trials t "
            "WHERE t.run_id = ? AND t.task_id = v.task_id AND t.task_version = v.version) "
            "ORDER BY v.task_id",
            (run_id,),
        )
        return [task_version_from_row(row) for row in rows]

    def task_specs_for_run(self, run_id: str) -> dict[str, dict[str, Any]]:
        """The pinned task spec of every task in a run, by task id."""
        rows = self._conn.execute(
            "SELECT DISTINCT t.task_id, v.spec FROM trials t JOIN task_versions v "
            "ON v.task_id = t.task_id AND v.version = t.task_version WHERE t.run_id = ?",
            (run_id,),
        )
        return {row["task_id"]: json.loads(row["spec"]) for row in rows}

    def config_spec(self, digest: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT spec FROM agent_configs WHERE config_hash = ?", (digest,)
        ).fetchone()
        return json.loads(row["spec"]) if row else {}

    # Summaries ----------------------------------------------------------------------------

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_summaries (run_id, summary, computed_at) VALUES (?, ?, ?)",
            (run_id, json.dumps(summary), timestamp()),
        )

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT summary FROM run_summaries WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    # CI baselines (WF-10) -----------------------------------------------------------------

    def current_suite_version(self, suite_id: str, task_refs: list[tuple[str, int]]) -> int | None:
        """The suite version with exactly these task versions, if it is the latest one."""
        latest = self._conn.execute(
            "SELECT version, task_refs FROM suites WHERE suite_id = ? "
            "ORDER BY version DESC LIMIT 1",
            (suite_id,),
        ).fetchone()
        if latest and latest["task_refs"] == json.dumps(sorted(task_refs)):
            return latest["version"]
        return None

    def find_baseline(
        self, suite_id: str, suite_version: int, branch: str, finished_after: datetime
    ) -> str | None:
        """The newest completed run of this suite version labelled with `branch` that finished
        after the cutoff and ran every planned trial (not BUDGET_TRUNCATED)."""
        rows = self._conn.execute(
            "SELECT run_id, labels, flags FROM runs WHERE suite_id = ? AND suite_version = ? "
            "AND status = 'completed' AND finished_at > ? ORDER BY finished_at DESC",
            (suite_id, suite_version, finished_after.isoformat(timespec="seconds")),
        )
        for row in rows:
            if json.loads(row["labels"]).get("branch") != branch:
                continue
            if "BUDGET_TRUNCATED" not in json.loads(row["flags"]):
                return row["run_id"]
        return None
