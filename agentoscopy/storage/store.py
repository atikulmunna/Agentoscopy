"""Results database (§14.3, SQLite): task versions, suites, configs, runs, trials, and grades.

The job queue lives here too (AD-2): workers claim queued trials with a conditional update,
and an attempt is finalised only by the worker that still holds it (FR-EXE-01, FR-EXE-05).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentoscopy.spec import AgentConfig, Task, config_hash
from agentoscopy.worker.trial import AttemptResult

IN_FLIGHT_STATES = ("PROVISIONING", "RUNNING", "GRADING")
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
  labels TEXT NOT NULL DEFAULT '{}', harness_version TEXT, created_at TEXT, finished_at TEXT
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
  UNIQUE (run_id, task_id, trial_index)
);
CREATE TABLE IF NOT EXISTS trial_attempts (
  trial_id TEXT NOT NULL REFERENCES trials, attempt INTEGER NOT NULL, worker_id TEXT,
  outcome TEXT, error_code TEXT, error TEXT, cost_usd REAL NOT NULL DEFAULT 0,
  trajectory_uri TEXT, started_at TEXT, ended_at TEXT,
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
CREATE INDEX IF NOT EXISTS idx_trials_run_state ON trials (run_id, state);
"""


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


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

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

    def create_run(self, config: AgentConfig, settings: RunSettings, trials: list[NewTrial]) -> str:
        run_id = str(uuid.uuid4())
        digest = config_hash(config)
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO agent_configs "
                "(config_hash, name, adapter, spec, created_at) VALUES (?, ?, ?, ?, ?)",
                (digest, config.name, config.adapter, config.model_dump_json(), _now()),
            )
            conn.execute(
                "INSERT INTO runs (run_id, suite_id, suite_version, config_hash, trials_per_task, "
                "mode, status, budget_usd, seed, concurrency, labels, harness_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'live', 'pending', ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    settings.suite_id,
                    settings.suite_version,
                    digest,
                    settings.trials_per_task,
                    settings.budget_usd,
                    settings.seed,
                    settings.concurrency,
                    json.dumps(settings.labels),
                    settings.harness_version,
                    _now(),
                ),
            )
            conn.executemany(
                "INSERT INTO trials (trial_id, run_id, task_id, task_version, trial_index, "
                "dispatch_order, cost_cap_usd, state) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED')",
                [
                    (
                        str(uuid.uuid4()),
                        run_id,
                        trial.task_id,
                        trial.task_version,
                        trial.trial_index,
                        trial.dispatch_order,
                        trial.cost_cap_usd,
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

    def claim_next(self, run_id: str, worker_id: str, now: float | None = None) -> Claim:
        """Claim the next trial in dispatch order if the run budget can reserve its cap."""
        now = time.time() if now is None else now
        with self._transaction() as conn:
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
                "reserved_usd = cost_cap_usd WHERE trial_id = ? AND state = 'QUEUED'",
                (attempt, worker_id, ready["trial_id"]),
            )
            conn.execute(
                "INSERT INTO trial_attempts (trial_id, attempt, worker_id, started_at) "
                "VALUES (?, ?, ?, ?)",
                (ready["trial_id"], attempt, worker_id, _now()),
            )
        trial = ClaimedTrial(
            ready["trial_id"],
            ready["task_id"],
            ready["task_version"],
            ready["trial_index"],
            attempt,
        )
        return Claim("claimed", trial=trial)

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
                "trajectory_uri = ?, ended_at = ? WHERE trial_id = ? AND attempt = ?",
                (
                    result.outcome,
                    result.error_code,
                    result.error,
                    result.usage.cost_usd,
                    str(result.trajectory_path),
                    _now(),
                    trial.trial_id,
                    trial.attempt,
                ),
            )
            holder = (trial.trial_id, trial.attempt, worker_id)
            owned = "WHERE trial_id = ? AND attempt = ? AND worker_id = ? AND state IN (?, ?, ?)"
            if requeue_delay_s is not None:
                cursor = conn.execute(
                    "UPDATE trials SET state = 'QUEUED', worker_id = NULL, reserved_usd = 0, "
                    f"not_before = ?, error_code = ?, error = ? {owned}",
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
                f"reserved_usd = 0, trajectory_uri = ?, final_state_uri = ? {owned}",
                (
                    _final_state(result),
                    result.outcome,
                    result.score,
                    result.termination,
                    result.error_code,
                    result.error,
                    *_usage_columns(result),
                    result.duration_s,
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
                "rationale, metadata) VALUES (?, ?, ?, 'deterministic', ?, ?, ?, ?)",
                [
                    (
                        trial.trial_id,
                        trial.attempt,
                        grade.name,
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
        return trial

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
            "WHERE t.task_id = ? AND r.config_hash = ? AND t.outcome IN ('pass', 'fail')",
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
    return "INFRA_ERROR" if result.outcome == "infra_error" else "COMPLETED"


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
