"""Run metrics (§9): per-task pass rate, pass@k, pass^k; run-level macro rate with a bootstrap CI.

Only `pass` and `fail` trials are scored; infra errors, skipped, and cancelled trials are
counted separately (FR-AGG-01).
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb
from statistics import fmean
from typing import Any

SCORED_OUTCOMES = frozenset({"pass", "fail"})
UNSCORED_OUTCOMES = ("infra_error", "skipped", "cancelled")
BOOTSTRAP_RESAMPLES = 10_000
SLICE_DIMENSIONS = ("category", "difficulty", "tag")
CI_LEVEL = 0.95


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased estimate that at least one of k sampled trials passes. Requires n >= k."""
    if k <= 0 or n < k:
        return None
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    """Estimate that all k sampled trials pass (reliability). Requires n >= k."""
    if k <= 0 or n < k:
        return None
    return comb(c, k) / comb(n, k)


def bootstrap_ci(
    values: list[float], seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling values (tasks) with replacement."""
    rng = random.Random(seed)
    size = len(values)
    means = sorted(fmean(rng.choices(values, k=size)) for _ in range(resamples))
    tail = (1 - CI_LEVEL) / 2
    return means[int(resamples * tail)], means[int(resamples * (1 - tail)) - 1]


def summarize(
    rows: list[dict[str, Any]],
    *,
    trials_per_task: int,
    seed: int,
    spent_usd: float,
    task_specs: dict[str, dict[str, Any]] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """`rows` are trial records with at least task_id, outcome, cost_usd, steps, duration_s.

    `task_specs` (task id to spec) adds per-slice pass rates by category, difficulty, and tag.
    """
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    tasks = [
        _task_summary(task_id, by_task[task_id], trials_per_task) for task_id in sorted(by_task)
    ]
    scored = [task for task in tasks if task["n"] > 0]
    rates = [task["pass_rate"] for task in scored]
    total_n = sum(task["n"] for task in scored)
    passes = sum(task["passes"] for task in scored)
    flaky = [task["task_id"] for task in scored if 0 < task["passes"] < task["n"]]
    counts = {outcome: 0 for outcome in (*sorted(SCORED_OUTCOMES), *UNSCORED_OUTCOMES)}
    for row in rows:
        if row.get("outcome") in counts:
            counts[row["outcome"]] += 1
    return {
        "k": trials_per_task,
        "macro_pass_rate": fmean(rates) if rates else None,
        "macro_ci_95": list(bootstrap_ci(rates, seed, resamples)) if rates else None,
        "micro_pass_rate": passes / total_n if total_n else None,
        "flaky_tasks": flaky,
        "flakiness": len(flaky) / len(scored) if scored else None,
        "unscored_tasks": [task["task_id"] for task in tasks if task["n"] == 0],
        "counts": counts,
        "total_cost_usd": spent_usd,
        "cost_per_pass_usd": spent_usd / passes if passes else None,
        "slices": slice_rates(scored, "pass_rate", task_specs or {}),
        "tasks": tasks,
    }


def slice_values(spec: dict[str, Any], dimension: str) -> list[str]:
    """The slice labels a task belongs to along one dimension (FR-AGG-04)."""
    if dimension == "tag":
        return list(spec.get("tags") or [])
    value = spec.get(dimension)
    return [str(value)] if value else []


def slice_rates(
    tasks: list[dict[str, Any]], key: str, task_specs: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Mean of `key` over the tasks in each slice, for every slice dimension."""
    slices: dict[str, list[dict[str, Any]]] = {}
    for dimension in SLICE_DIMENSIONS:
        groups: dict[str, list[float]] = defaultdict(list)
        for task in tasks:
            for value in slice_values(task_specs.get(task["task_id"], {}), dimension):
                groups[value].append(task[key])
        slices[dimension] = [
            {"value": value, "tasks": len(values), "mean": fmean(values)}
            for value, values in sorted(groups.items())
        ]
    return slices


def _task_summary(task_id: str, rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    scored = [row for row in rows if row.get("outcome") in SCORED_OUTCOMES]
    n = len(scored)
    passes = sum(row["outcome"] == "pass" for row in scored)
    summary: dict[str, Any] = {
        "task_id": task_id,
        "n": n,
        "passes": passes,
        "pass_rate": passes / n if n else None,
        "pass_at_k": pass_at_k(n, passes, k),
        "pass_hat_k": pass_hat_k(n, passes, k),
        "mean_cost_usd": _mean(scored, "cost_usd"),
        "mean_steps": _mean(scored, "steps"),
        "mean_duration_s": _mean(scored, "duration_s"),
    }
    for outcome in UNSCORED_OUTCOMES:
        summary[outcome] = sum(row.get("outcome") == outcome for row in rows)
    return summary


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return fmean(values) if values else None
