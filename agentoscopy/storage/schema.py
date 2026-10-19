"""The results database's schema (§14.3, SQLite) and its upgrades from older versions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5  # bump whenever SCHEMA changes, and add a migration

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
  created_at TEXT, finished_at TEXT, review_rate REAL,
  top_up INTEGER NOT NULL DEFAULT 0  -- extra trials per task with mixed results
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
    4: """
ALTER TABLE runs ADD COLUMN top_up INTEGER NOT NULL DEFAULT 0;
""",
}


class StoreError(Exception):
    """The database cannot be used as is."""


def open_database(path: Path) -> sqlite3.Connection:
    """Connect, creating the database or upgrading an older one to SCHEMA_VERSION."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    fresh = not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'").fetchone()
    version = SCHEMA_VERSION if fresh else conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION and not all(
        step in MIGRATIONS for step in range(version, SCHEMA_VERSION)
    ):
        conn.close()
        raise StoreError(
            f"{path} uses schema version {version}, but this release needs {SCHEMA_VERSION}; "
            "move it aside to start a fresh database"
        )
    for step in range(version, SCHEMA_VERSION):
        # executescript commits first, so each upgrade is its own transaction.
        conn.executescript(f"BEGIN; {MIGRATIONS[step]} PRAGMA user_version = {step + 1}; COMMIT;")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return conn
