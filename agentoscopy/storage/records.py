"""The records the results database takes in and hands out, and their conversions from rows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

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
    mode: str = "live"  # live | replay | fork
    top_up: int = 0  # extra trials for each task with mixed results (FR-RUN-10)


def run_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """A RUN_LISTING row, with the stored summary's headline numbers."""
    run = dict(row)
    run["flags"], run["labels"] = json.loads(run["flags"]), json.loads(run["labels"])
    summary = json.loads(run.pop("summary")) if run.get("summary") else {}
    run["macro_pass_rate"] = summary.get("macro_pass_rate")
    run["macro_ci_95"] = summary.get("macro_ci_95")
    return run


def task_version_from_row(row: sqlite3.Row) -> TaskVersion:
    return TaskVersion(
        task_id=row["task_id"],
        version=row["version"],
        content_hash=row["content_hash"],
        image_digest=row["image_digest"],
        flags=json.loads(row["flags"]),
        verified_graders=frozenset(json.loads(row["verified_graders"])),
        validated_at=row["validated_at"],
    )
