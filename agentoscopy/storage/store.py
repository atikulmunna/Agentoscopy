"""Results database (§14.3, SQLite): task versions, suites, configs, runs, trials, and grades.

The job queue lives here too (AD-2): workers claim queued trials with a conditional update
and hold them under a lease their process keeps renewing. An attempt is finalised only by
the worker that still holds it (FR-EXE-01, FR-EXE-05), and a trial whose lease ran out
because its process died can be reclaimed (FR-EXE-06).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentoscopy.spec import AgentConfig, Task, config_hash
from agentoscopy.worker.trial import AttemptResult

IN_FLIGHT_STATES = ("PROVISIONING", "RUNNING", "GRADING")
ACTIVE_RUN_STATUSES = ("pending", "running", "cancelling")
LEASE_S = 30.0  # a worker's process renews its leases well within this (FR-EXE-01)
SCHEMA_VERSION = 4  # bump whenever SCHEMA changes, and add a migration
REVIEW_PRIORITY = "CASE sample_source WHEN 'low_confidence' THEN 0 WHEN 'random' THEN 1 ELSE 2 END"
RUN_LISTING = """
SELECT runs.*, agent_configs.name AS config_name, run_summaries.summary AS summary,
  (SELECT COUNT(*) FROM trials t WHERE t.run_id = runs.run_id) AS total_trials,
  (SELECT COUNT(*) FROM trials t WHERE t.run_id = runs.run_id
     AND t.state IN ('COMPLETED', 'INFRA_ERROR', 'SKIPPED', 'CANCELLED')) AS finished_trials,
  (SELECT COALESCE(SUM(a.cost_usd), 0) FROM trial_attempts a JOIN trials t USING (trial_id)
     WHERE t.run_id = runs.run_id) AS spent_usd
FROM runs
LEFT JOIN agent_configs USING (config_hash)
LEFT JOIN run_summaries USING (run_id)
"""
BUDGET_EPSILON_USD = 1e-9
WORKER_LOST = "WORKER_LOST"
WORKER_LOST_MESSAGE = "the worker stopped renewing its lease: its process died or hung"

SCHEMA = """
CREATE TABLE IF NOT EXISTS task_versions (
  task_id TEXT NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
  image_digest TEXT, spec TEXT NOT NULL, flags TEXT NOT NULL DEFAULT '[]',
  verified_graders TEXT NOT NULL DEFAULT '[]', validated_at TEXT,
  PRIMARY KEY (task_id, version),
  UNIQUE (task_id, content_hash)
);
CREATE TABLE IF NOT EXISTS suites (
  suite_id TEXT NOT NULL, version INTEGER NOT NULL, task_refs TEXT NOT NULL, created_at TEXT,
  PRIMARY KEY (suite_id, version)
);
CREATE TABLE IF NOT EXISTS agent_configs (
  config_hash TEXT PRIMARY KEY, name TEXT, adapter TEXT, spec TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, suite_id TEXT, suite_version INTEGER,
  config_hash TEXT REFERENCES agent_configs, trials_per_task INTEGER, mode TEXT, status TEXT,
  budget_usd REAL, seed INTEGER, concurrency INTEGER, flags TEXT NOT NULL DEFAULT '[]',
  labels TEXT NOT NULL DEFAULT '{}', harness_version TEXT, judge_model TEXT,
  created_at TEXT, finished_at TEXT, review_rate REAL
);
CREATE TABLE IF NOT EXISTS trials (
  trial_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs,
  task_id TEXT NOT NULL, task_version INTEGER NOT NULL, trial_index INTEGER NOT NULL,
  dispatch_order INTEGER NOT NULL, cost_cap_usd REAL NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL, outcome TEXT, score REAL,
  termination TEXT, error_code TEXT, error TEXT, steps INTEGER, input_tokens INTEGER,
  output_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, cost_usd REAL,
  duration_s REAL, worker_id TEXT, reserved_usd REAL NOT NULL DEFAULT 0,
  not_before REAL NOT NULL DEFAULT 0, trajectory_uri TEXT, final_state_uri TEXT,
  judge_cost_usd REAL NOT NULL DEFAULT 0,
  original_outcome TEXT,  -- set when a reviewer overrides the outcome (FR-REV-06)
  lease_expires_at REAL,  -- epoch seconds; renewed while the trial is in flight
  source_trial_id TEXT,  -- for a replay: the trial it reran (WF-11)
  UNIQUE (run_id, task_id, trial_index)
);
CREATE TABLE IF NOT EXISTS trial_attempts (
  trial_id TEXT NOT NULL REFERENCES trials, attempt INTEGER NOT NULL, worker_id TEXT,
  outcome TEXT, error_code TEXT, error TEXT, cost_usd REAL NOT NULL DEFAULT 0,
  judge_cost_usd REAL NOT NULL DEFAULT 0, trajectory_uri TEXT, started_at TEXT, ended_at TEXT,
  model_calls INTEGER NOT NULL DEFAULT 0, model_latency_s REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (trial_id, attempt)
);
CREATE TABLE IF NOT EXISTS grades (
  trial_id TEXT NOT NULL, attempt INTEGER NOT NULL, grader_name TEXT NOT NULL, kind TEXT,
  score REAL, passed INTEGER, rationale TEXT, metadata TEXT,
  PRIMARY KEY (trial_id, attempt, grader_name)
);
CREATE TABLE IF NOT EXISTS run_summaries (
  run_id TEXT PRIMARY KEY REFERENCES runs, summary TEXT NOT NULL, computed_at TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, trial_id TEXT NOT NULL REFERENCES trials,
  attempt INTEGER NOT NULL,
  grader_name TEXT NOT NULL DEFAULT '',  -- judge grader under review; '' for a trial audit
  sample_source TEXT NOT NULL,  -- random | low_confidence | manual
  queued_at TEXT, reviewer TEXT, passed INTEGER, score REAL, note TEXT,
  override INTEGER NOT NULL DEFAULT 0, reviewed_at TEXT,
  UNIQUE (trial_id, attempt, grader_name, sample_source)
);
CREATE TABLE IF NOT EXISTS failure_tags (
  trial_id TEXT NOT NULL REFERENCES trials, tag TEXT NOT NULL,
  source TEXT NOT NULL,  -- human | auto
  note TEXT, PRIMARY KEY (trial_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_trials_run_state ON trials (run_id, state);
CREATE INDEX IF NOT EXISTS idx_trials_lease ON trials (state, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_reviews_pending ON reviews (reviewed_at, sample_source);
"""
# Upgrades from each older schema version to the next; SCHEMA then adds any new indexes.
MIGRATIONS = {
    2: """
ALTER TABLE runs ADD COLUMN review_rate REAL;
ALTER TABLE trials ADD COLUMN lease_expires_at REAL;
ALTER TABLE trial_attempts ADD COLUMN model_calls INTEGER NOT NULL DEFAULT 0;
ALTER TABLE trial_attempts ADD COLUMN model_latency_s REAL NOT NULL DEFAULT 0;
""",
    3: """
ALTER TABLE trials ADD COLUMN source_trial_id TEXT;
""",
}


class StoreError(Exception):
    """The database cannot be used as is."""


@dataclass(frozen=True)
class TaskVersion:
    task_id: str
    version: int
    content_hash: str
    image_digest: str | None
    flags: list[str]
    verified_graders: frozenset[str]
    validated_at: str | None


@dataclass(frozen=True)
class NewTrial:
    task_id: str
    task_version: int
    trial_index: int
    dispatch_order: int
    cost_cap_usd: float
    source_trial_id: str | None = None  # for a replay


@dataclass(frozen=True)
class ClaimedTrial:
    trial_id: str
    task_id: str
    task_version: int
    trial_index: int
    attempt: int


@dataclass(frozen=True)
class Claim:
    """Result of asking for work: a trial, or why there is none right now."""

    status: str  # claimed | empty | waiting | unaffordable
    trial: ClaimedTrial | None = None
    ready_at: float | None = None  # for `waiting`: when the next retry becomes claimable


@dataclass(frozen=True)
class RunSettings:
    suite_id: str | None
    suite_version: int | None
    trials_per_task: int
    budget_usd: float | None
    seed: int
    concurrency: int
    harness_version: str
    labels: dict[str, str] = field(default_factory=dict)
    judge_model: str | None = None
    review_rate: float | None = None
    mode: str = "live"  # live | replay


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        fresh = not self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table'"
        ).fetchone()
        version = (
            SCHEMA_VERSION if fresh else self._conn.execute("PRAGMA user_version").fetchone()[0]
        )
        if version != SCHEMA_VERSION and not all(
            step in MIGRATIONS for step in range(version, SCHEMA_VERSION)
        ):
            self._conn.close()
            raise StoreError(
                f"{path} uses schema version {version}, but this release needs {SCHEMA_VERSION}; "
                "move it aside to start a fresh database"
            )
        for step in range(version, SCHEMA_VERSION):
            # executescript commits first, so each upgrade is its own transaction.
            self._conn.executescript(
                f"BEGIN; {MIGRATIONS[step]} PRAGMA user_version = {step + 1}; COMMIT;"
            )
        self._conn.executescript(SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # Task versions ------------------------------------------------------------------------

    def task_version(self, task_id: str, content_hash: str) -> TaskVersion | None:
        row = self._conn.execute(
            "SELECT * FROM task_versions WHERE task_id = ? AND content_hash = ?",
            (task_id, content_hash),
        ).fetchone()
        return _task_version(row) if row else None

    def save_task_version(
        self, task: Task, image_digest: str | None, flags: list[str], verified: list[str]
    ) -> int:
        """Record a passed validation; identical content reuses its existing version."""
        values = (image_digest, json.dumps(sorted(flags)), json.dumps(sorted(verified)), _now())
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
                (suite_id, version, refs, _now()),
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
                "judge_model, review_rate, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    _now(),
                ),
            )
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
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(f"{RUN_LISTING} WHERE runs.run_id = ?", (run_id,)).fetchone()
        return _run_dict(row) if row else None

    def list_runs(self, limit: int = 200) -> list[dict[str, Any]]:
        """Newest first, with progress, spend, and the stored summary's headline numbers."""
        rows = self._conn.execute(
            f"{RUN_LISTING} ORDER BY runs.created_at DESC, runs.rowid DESC LIMIT ?", (limit,)
        )
        return [_run_dict(row) for row in rows]

    def run_status(self, run_id: str) -> str | None:
        row = self._conn.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row["status"] if row else None

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
            _cancel_queued(conn, run_id)
        self.finalize_cancel(run_id)
        return self.run_status(run_id)

    def finalize_cancel(self, run_id: str) -> bool:
        """Mark a cancelling run cancelled once none of its trials is in flight."""
        cursor = self._conn.execute(
            "UPDATE runs SET status = 'cancelled', finished_at = ? "
            "WHERE run_id = ? AND status = 'cancelling' AND NOT EXISTS "
            "(SELECT 1 FROM trials WHERE run_id = ? AND state IN (?, ?, ?))",
            (_now(), run_id, run_id, *IN_FLIGHT_STATES),
        )
        return cursor.rowcount == 1

    def set_run_status(self, run_id: str, status: str, finished: bool = False) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = CASE WHEN ? THEN ? ELSE finished_at END "
            "WHERE run_id = ?",
            (status, finished, _now(), run_id),
        )

    def add_run_flag(self, run_id: str, flag: str) -> None:
        with self._transaction() as conn:
            flags = json.loads(
                conn.execute("SELECT flags FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
            )
            conn.execute(
                "UPDATE runs SET flags = ? WHERE run_id = ?",
                (json.dumps(sorted(set(flags) | {flag})), run_id),
            )

    # Queue --------------------------------------------------------------------------------

    def claim_next(
        self, run_id: str, worker_id: str, now: float | None = None, lease_s: float = LEASE_S
    ) -> Claim:
        """Claim the next trial in dispatch order if the run budget can reserve its cap."""
        now = time.time() if now is None else now
        with self._transaction() as conn:
            if self.run_status(run_id) == "cancelling":
                _cancel_queued(conn, run_id)  # including retries queued after the request
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
                (ready["trial_id"], attempt, worker_id, _now()),
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
            conn.execute(
                "UPDATE trial_attempts SET outcome = 'infra_error', error_code = ?, error = ?, "
                "cost_usd = ?, judge_cost_usd = ?, ended_at = ? WHERE trial_id = ? AND attempt = ?",
                (
                    WORKER_LOST,
                    WORKER_LOST_MESSAGE,
                    cost_usd,
                    judge_cost_usd,
                    _now(),
                    trial.trial_id,
                    trial.attempt,
                ),
            )
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
        with self._transaction() as conn:
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
                    _now(),
                    result.usage.model_calls,
                    result.usage.model_latency_s,
                    trial.trial_id,
                    trial.attempt,
                ),
            )
            holder = (trial.trial_id, trial.attempt, worker_id)
            owned = "WHERE trial_id = ? AND attempt = ? AND worker_id = ? AND state IN (?, ?, ?)"
            if requeue_delay_s is not None:
                cursor = conn.execute(
                    "UPDATE trials SET state = 'QUEUED', worker_id = NULL, reserved_usd = 0, "
                    f"lease_expires_at = NULL, not_before = ?, error_code = ?, error = ? {owned}",
                    (
                        time.time() + requeue_delay_s,
                        result.error_code,
                        result.error,
                        *holder,
                        *IN_FLIGHT_STATES,
                    ),
                )
                return cursor.rowcount == 1
            cursor = conn.execute(
                "UPDATE trials SET state = ?, outcome = ?, score = ?, termination = ?, "
                "error_code = ?, error = ?, steps = ?, input_tokens = ?, output_tokens = ?, "
                "cache_read_tokens = ?, cache_write_tokens = ?, cost_usd = ?, duration_s = ?, "
                "judge_cost_usd = ?, reserved_usd = 0, lease_expires_at = NULL, "
                f"trajectory_uri = ?, final_state_uri = ? {owned}",
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
                    *holder,
                    *IN_FLIGHT_STATES,
                ),
            )
            if cursor.rowcount != 1:
                return False
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
                    for grade in result.grades
                ],
            )
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

    # Reporting ----------------------------------------------------------------------------

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        if row is None:
            return None
        trial = dict(row)
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

    def task_spec(self, task_id: str, version: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT spec FROM task_versions WHERE task_id = ? AND version = ?", (task_id, version)
        ).fetchone()
        return json.loads(row["spec"]) if row else {}

    def add_failure_tag(
        self, trial_id: str, tag: str, source: str, note: str | None = None
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO failure_tags (trial_id, tag, source, note) VALUES (?, ?, ?, ?)",
            (trial_id, tag, source, note),
        )

    # Human review (WF-09) ------------------------------------------------------------------

    def queue_review(
        self, trial_id: str, attempt: int, grader_name: str, source: str
    ) -> str | None:
        """Queue a review item; returns its id, or None if that item is already queued."""
        review_id = str(uuid.uuid4())
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO reviews (review_id, trial_id, attempt, grader_name, "
            "sample_source, queued_at) VALUES (?, ?, ?, ?, ?, ?)",
            (review_id, trial_id, attempt, grader_name, source, _now()),
        )
        return review_id if cursor.rowcount == 1 else None

    def review_queue(self, limit: int) -> list[dict[str, Any]]:
        """Pending items: low-confidence judge verdicts first, then random samples (FR-REV-01)."""
        rows = self._conn.execute(
            "SELECT r.*, t.run_id, t.task_id, t.task_version FROM reviews r "
            "JOIN trials t USING (trial_id) WHERE r.reviewed_at IS NULL "
            f"ORDER BY {REVIEW_PRIORITY}, r.queued_at, r.rowid LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def pending_reviews(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE reviewed_at IS NULL"
        ).fetchone()[0]

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT r.*, t.run_id, t.task_id, t.task_version FROM reviews r "
            "JOIN trials t USING (trial_id) WHERE r.review_id = ?",
            (review_id,),
        ).fetchone()
        return dict(row) if row else None

    def submit_review(
        self,
        review_id: str,
        *,
        reviewer: str,
        passed: bool,
        score: float | None,
        note: str | None,
        override: bool,
    ) -> bool:
        """Record a review once. An override replaces the trial's outcome with the reviewer's,
        keeps the original, and drops the run's stored summary so it is recomputed (FR-REV-06)."""
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE reviews SET reviewer = ?, passed = ?, score = ?, note = ?, override = ?, "
                "reviewed_at = ? WHERE review_id = ? AND reviewed_at IS NULL",
                (reviewer, passed, score, note, override, _now(), review_id),
            )
            if cursor.rowcount != 1:
                return False
            if override:
                trial_id, run_id = conn.execute(
                    "SELECT t.trial_id, t.run_id FROM reviews r JOIN trials t USING (trial_id) "
                    "WHERE r.review_id = ?",
                    (review_id,),
                ).fetchone()
                conn.execute(
                    "UPDATE trials SET original_outcome = COALESCE(original_outcome, outcome), "
                    "outcome = ? WHERE trial_id = ? AND outcome IN ('pass', 'fail')",
                    ("pass" if passed else "fail", trial_id),
                )
                conn.execute("DELETE FROM run_summaries WHERE run_id = ?", (run_id,))
            return True

    def grade_for(self, trial_id: str, attempt: int, grader_name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT grader_name, kind, score, passed, rationale, metadata FROM grades "
            "WHERE trial_id = ? AND attempt = ? AND grader_name = ?",
            (trial_id, attempt, grader_name),
        ).fetchone()
        if row is None:
            return None
        return {**dict(row), "passed": bool(row["passed"]), "metadata": json.loads(row["metadata"])}

    def calibration_rows(self) -> list[dict[str, Any]]:
        """Every finished review of a judge grader, paired with the judge's own verdict."""
        rows = self._conn.execute(
            "SELECT r.review_id, r.trial_id, r.grader_name, r.sample_source, "
            "r.passed AS human_passed, "
            "g.passed AS judge_passed, g.metadata, t.task_id FROM reviews r "
            "JOIN grades g ON g.trial_id = r.trial_id AND g.attempt = r.attempt "
            "AND g.grader_name = r.grader_name "
            "JOIN trials t ON t.trial_id = r.trial_id "
            "WHERE r.reviewed_at IS NOT NULL AND r.grader_name != '' AND g.kind = 'judge' "
            "ORDER BY r.reviewed_at",
        )
        return [
            {
                "review_id": row["review_id"],
                "trial_id": row["trial_id"],
                "task_id": row["task_id"],
                "grader_name": row["grader_name"],
                "judge_model": json.loads(row["metadata"]).get("judge_model"),
                "sample_source": row["sample_source"],
                "human_passed": bool(row["human_passed"]),
                "judge_passed": bool(row["judge_passed"]),
            }
            for row in rows
        ]

    # CI baselines (WF-10) ---------------------------------------------------------------------

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

    # Garbage collection -------------------------------------------------------------------

    def runs_finished_before(self, cutoff: datetime) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"{RUN_LISTING} WHERE runs.status IN ('completed', 'cancelled', 'failed') "
            "AND runs.finished_at < ? ORDER BY runs.finished_at",
            (cutoff.isoformat(timespec="seconds"),),
        )
        return [_run_dict(row) for row in rows]

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

    def run_task_versions(self, run_id: str) -> list[TaskVersion]:
        """The task versions a run pinned, so it can be resumed exactly as it started."""
        rows = self._conn.execute(
            "SELECT v.* FROM task_versions v WHERE EXISTS (SELECT 1 FROM trials t "
            "WHERE t.run_id = ? AND t.task_id = v.task_id AND t.task_version = v.version) "
            "ORDER BY v.task_id",
            (run_id,),
        )
        return [_task_version(row) for row in rows]

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

    def trial_rows(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM trials WHERE run_id = ? ORDER BY task_id, trial_index", (run_id,)
        )
        return [dict(row) for row in rows]

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

    def progress(self, run_id: str) -> tuple[int, int]:
        """(finished trials, total trials)."""
        row = self._conn.execute(
            "SELECT SUM(state IN ('COMPLETED', 'INFRA_ERROR', 'SKIPPED', 'CANCELLED')), COUNT(*) "
            "FROM trials WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return row[0] or 0, row[1]

    def historical_mean_cost(self, task_id: str, digest: str) -> float | None:
        return self._conn.execute(
            "SELECT AVG(t.cost_usd) FROM trials t JOIN runs r USING (run_id) "
            "WHERE t.task_id = ? AND r.config_hash = ? AND t.outcome IN ('pass', 'fail') "
            "AND r.mode = 'live'",  # replays cost nothing, which says nothing
            (task_id, digest),
        ).fetchone()[0]

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_summaries (run_id, summary, computed_at) VALUES (?, ?, ?)",
            (run_id, json.dumps(summary), _now()),
        )

    def get_summary(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT summary FROM run_summaries WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

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


def _run_dict(row: sqlite3.Row) -> dict[str, Any]:
    run = dict(row)
    run["flags"], run["labels"] = json.loads(run["flags"]), json.loads(run["labels"])
    summary = json.loads(run.pop("summary")) if run.get("summary") else {}
    run["macro_pass_rate"] = summary.get("macro_pass_rate")
    run["macro_ci_95"] = summary.get("macro_ci_95")
    return run


def _task_version(row: sqlite3.Row) -> TaskVersion:
    return TaskVersion(
        task_id=row["task_id"],
        version=row["version"],
        content_hash=row["content_hash"],
        image_digest=row["image_digest"],
        flags=json.loads(row["flags"]),
        verified_graders=frozenset(json.loads(row["verified_graders"])),
        validated_at=row["validated_at"],
    )


def _final_state(result: AttemptResult) -> str:
    return {"infra_error": "INFRA_ERROR", "cancelled": "CANCELLED"}.get(result.outcome, "COMPLETED")


def _save_config(conn: sqlite3.Connection, config: AgentConfig) -> str:
    digest = config_hash(config)
    conn.execute(
        "INSERT OR IGNORE INTO agent_configs "
        "(config_hash, name, adapter, spec, created_at) VALUES (?, ?, ?, ?, ?)",
        (digest, config.name, config.adapter, config.model_dump_json(), _now()),
    )
    return digest


def _cancel_queued(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        "UPDATE trials SET state = 'CANCELLED', outcome = 'cancelled', reserved_usd = 0 "
        "WHERE run_id = ? AND state = 'QUEUED'",
        (run_id,),
    )


def _claimed(row: sqlite3.Row) -> ClaimedTrial:
    return ClaimedTrial(
        row["trial_id"], row["task_id"], row["task_version"], row["trial_index"], row["attempt"]
    )


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


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
