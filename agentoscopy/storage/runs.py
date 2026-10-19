"""Task versions, suites, agent configs, and runs: what a run is made of, and its status."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from agentoscopy.spec import AgentConfig, Task, config_hash
from agentoscopy.storage.database import Database, timestamp
from agentoscopy.storage.records import (
    IN_FLIGHT_STATES,
    RUN_LISTING,
    NewTrial,
    RunSettings,
    TaskVersion,
    run_from_row,
    task_version_from_row,
)
from agentoscopy.storage.trial_queue import cancel_queued

ACTIVE_RUN_STATUSES = ("pending", "running", "cancelling")
TOPPED_UP = "TOPPED_UP"  # run flag: the adaptive top-up trials were added


class RunRecords(Database):
    # Task versions ------------------------------------------------------------------------

    def task_version(self, task_id: str, content_hash: str) -> TaskVersion | None:
        row = self._conn.execute(
            "SELECT * FROM task_versions WHERE task_id = ? AND content_hash = ?",
            (task_id, content_hash),
        ).fetchone()
        return task_version_from_row(row) if row else None

    def save_task_version(
        self, task: Task, image_digest: str | None, flags: list[str], verified: list[str]
    ) -> int:
        """Record a passed validation; identical content reuses its existing version."""
        values = (
            image_digest,
            json.dumps(sorted(flags)),
            json.dumps(sorted(verified)),
            timestamp(),
        )
        with self._transaction() as conn:
            existing = self.task_version(task.spec.id, task.content_hash)
            if existing:
                conn.execute(
                    "UPDATE task_versions SET image_digest = ?, flags = ?, verified_graders = ?, "
                    "validated_at = ? WHERE task_id = ? AND version = ?",
                    (*values, task.spec.id, existing.version),
                )
                return existing.version
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM task_versions WHERE task_id = ?",
                (task.spec.id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO task_versions (task_id, version, content_hash, image_digest, spec, "
                "flags, verified_graders, validated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task.spec.id,
                    version,
                    task.content_hash,
                    values[0],
                    task.spec.model_dump_json(),
                    *values[1:],
                ),
            )
            return version

    def add_task_flag(self, task_id: str, version: int, flag: str) -> None:
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT flags FROM task_versions WHERE task_id = ? AND version = ?",
                (task_id, version),
            ).fetchone()
            if row is None:
                return
            flags = sorted(set(json.loads(row["flags"])) | {flag})
            conn.execute(
                "UPDATE task_versions SET flags = ? WHERE task_id = ? AND version = ?",
                (json.dumps(flags), task_id, version),
            )

    # Suites, configs, and runs ------------------------------------------------------------

    def save_suite_version(self, suite_id: str, task_refs: list[tuple[str, int]]) -> int:
        refs = json.dumps(sorted(task_refs))
        with self._transaction() as conn:
            latest = conn.execute(
                "SELECT version, task_refs FROM suites WHERE suite_id = ? "
                "ORDER BY version DESC LIMIT 1",
                (suite_id,),
            ).fetchone()
            if latest and latest["task_refs"] == refs:
                return latest["version"]
            version = latest["version"] + 1 if latest else 1
            conn.execute(
                "INSERT INTO suites (suite_id, version, task_refs, created_at) VALUES (?, ?, ?, ?)",
                (suite_id, version, refs, timestamp()),
            )
            return version

    def save_config(self, config: AgentConfig) -> str:
        """Record an agent config under its hash (FR-AGT-02); returns the hash."""
        return _save_config(self._conn, config)

    def create_run(self, config: AgentConfig, settings: RunSettings, trials: list[NewTrial]) -> str:
        run_id = str(uuid.uuid4())
        with self._transaction() as conn:
            digest = _save_config(conn, config)
            conn.execute(
                "INSERT INTO runs (run_id, suite_id, suite_version, config_hash, trials_per_task, "
                "mode, status, budget_usd, seed, concurrency, labels, harness_version, "
                "judge_model, review_rate, top_up, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    settings.suite_id,
                    settings.suite_version,
                    digest,
                    settings.trials_per_task,
                    settings.mode,
                    settings.budget_usd,
                    settings.seed,
                    settings.concurrency,
                    json.dumps(settings.labels),
                    settings.harness_version,
                    settings.judge_model,
                    settings.review_rate,
                    settings.top_up,
                    timestamp(),
                ),
            )
            _insert_trials(conn, run_id, trials)
        return run_id

    def add_top_up(self, run_id: str, trials: list[NewTrial]) -> None:
        """Queue the top-up trials and mark the run topped up, in one transaction, so a
        crash can never add them twice."""
        with self._transaction() as conn:
            _insert_trials(conn, run_id, trials)
            _add_flag(conn, run_id, TOPPED_UP)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(f"{RUN_LISTING} WHERE runs.run_id = ?", (run_id,)).fetchone()
        return run_from_row(row) if row else None

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Newest first, with progress, spend, and the stored summary's headline numbers."""
        rows = self._conn.execute(
            f"{RUN_LISTING} ORDER BY runs.created_at DESC, runs.rowid DESC LIMIT ?", (limit,)
        )
        return [run_from_row(row) for row in rows]

    def mark_running(self, run_id: str) -> None:
        """A run being executed, or resumed after a crash or a harness failure. A run that is
        being cancelled stays that way."""
        self._conn.execute(
            "UPDATE runs SET status = 'running', finished_at = NULL "
            "WHERE run_id = ? AND status IN ('pending', 'running', 'failed')",
            (run_id,),
        )

    def request_cancel(self, run_id: str) -> str | None:
        """Ask a run to stop (WF-12). Queued trials are cancelled at once; trials in flight
        are stopped by the process running them. Returns the run's status afterwards
        (`cancelled` once nothing is in flight), or None if there is no such run."""
        with self._transaction() as conn:
            status = self.run_status(run_id)
            if status not in ACTIVE_RUN_STATUSES:
                return status
            conn.execute("UPDATE runs SET status = 'cancelling' WHERE run_id = ?", (run_id,))
            cancel_queued(conn, run_id)
        self.finalize_cancel(run_id)
        return self.run_status(run_id)

    def finalize_cancel(self, run_id: str) -> bool:
        """Mark a cancelling run cancelled once none of its trials is in flight."""
        cursor = self._conn.execute(
            "UPDATE runs SET status = 'cancelled', finished_at = ? "
            "WHERE run_id = ? AND status = 'cancelling' AND NOT EXISTS "
            "(SELECT 1 FROM trials WHERE run_id = ? AND state IN (?, ?, ?))",
            (timestamp(), run_id, run_id, *IN_FLIGHT_STATES),
        )
        return cursor.rowcount == 1

    def set_run_status(self, run_id: str, status: str, finished: bool = False) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = CASE WHEN ? THEN ? ELSE finished_at END "
            "WHERE run_id = ?",
            (status, finished, timestamp(), run_id),
        )

    def add_run_flag(self, run_id: str, flag: str) -> None:
        with self._transaction() as conn:
            _add_flag(conn, run_id, flag)


def _add_flag(conn: sqlite3.Connection, run_id: str, flag: str) -> None:
    flags = json.loads(
        conn.execute("SELECT flags FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    )
    conn.execute(
        "UPDATE runs SET flags = ? WHERE run_id = ?",
        (json.dumps(sorted(set(flags) | {flag})), run_id),
    )


def _insert_trials(conn: sqlite3.Connection, run_id: str, trials: list[NewTrial]) -> None:
    conn.executemany(
        "INSERT INTO trials (trial_id, run_id, task_id, task_version, trial_index, "
        "dispatch_order, cost_cap_usd, state, source_trial_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?)",
        [
            (
                str(uuid.uuid4()),
                run_id,
                trial.task_id,
                trial.task_version,
                trial.trial_index,
                trial.dispatch_order,
                trial.cost_cap_usd,
                trial.source_trial_id,
            )
            for trial in trials
        ],
    )


def _save_config(conn: sqlite3.Connection, config: AgentConfig) -> str:
    digest = config_hash(config)
    conn.execute(
        "INSERT OR IGNORE INTO agent_configs "
        "(config_hash, name, adapter, spec, created_at) VALUES (?, ?, ?, ?, ?)",
        (digest, config.name, config.adapter, config.model_dump_json(), timestamp()),
    )
    return digest
