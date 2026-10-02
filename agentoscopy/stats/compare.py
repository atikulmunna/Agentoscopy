"""Run comparison (WF-07): per-task classes, a paired bootstrap of the delta, and a verdict.

Tasks are compared only where both runs scored the same task version. A task is Regressed or
Improved only when a one-sided Fisher's exact test says so (p < 0.05); other differences are
Changed and never affect the verdict. The verdict comes from the bootstrap CI of the mean
per-task delta, plus any critical task classified Regressed.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass
from math import comb
from statistics import fmean
from typing import Any

from agentoscopy.stats.summary import (
    BOOTSTRAP_RESAMPLES,
    SCORED_OUTCOMES,
    bootstrap_ci,
    slice_rates,
)

ALPHA = 0.05
COMPARE_SEED = 0  # the same two runs always compare the same way
FEW_TASKS_WARNING = 10
DELTA_METRICS = ("cost_usd", "tokens", "steps", "duration_s")


class CompareError(Exception):
    """The runs cannot be compared. Messages start with an error code."""


@dataclass(frozen=True)
class RunInputs:
    run: dict[str, Any]
    trials: list[dict[str, Any]]
    task_specs: dict[str, dict[str, Any]]
    config: dict[str, Any]


@dataclass(frozen=True)
class _TaskStats:
    version: int
    n: int
    passes: int
    means: dict[str, float | None]
    trials: list[dict[str, Any]]

    @property
    def rate(self) -> float:
        return self.passes / self.n


def fisher_worse(base_passes: int, base_n: int, cand_passes: int, cand_n: int) -> float:
    """One-sided Fisher's exact p-value that the candidate passes less often than the baseline.

    Under the null hypothesis, the baseline's share of all passes is hypergeometric; the p-value
    is the chance of the baseline holding at least as many passes as it did.
    """
    passes, total = base_passes + cand_passes, base_n + cand_n
    tail = range(base_passes, min(passes, base_n) + 1)
    hits = sum(comb(passes, x) * comb(total - passes, base_n - x) for x in tail)
    return hits / comb(total, base_n)


def compare_runs(
    baseline: RunInputs,
    candidate: RunInputs,
    *,
    allow_version_drift: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    base_tasks, cand_tasks = _task_stats(baseline.trials), _task_stats(candidate.trials)
    shared, excluded, warnings = _pair(base_tasks, cand_tasks, allow_version_drift)
    if not shared:
        raise CompareError("NO_SHARED_TASKS: the runs share no scored task versions")
    specs = {**baseline.task_specs, **candidate.task_specs}
    rows = [
        _task_row(task_id, base_tasks[task_id], cand_tasks[task_id], specs.get(task_id, {}))
        for task_id in shared
    ]
    deltas = [row["delta"] for row in rows]
    ci = bootstrap_ci(deltas, COMPARE_SEED, resamples)
    if len(rows) < FEW_TASKS_WARNING:
        warnings.append(
            f"only {len(rows)} shared tasks: the CI treats tasks as the sampling unit "
            f"and is unreliable below about {FEW_TASKS_WARNING}"
        )
    classes = defaultdict(int)
    for row in rows:
        classes[row["class"]] += 1
    return {
        "verdict": _verdict(ci, rows),
        "baseline": _run_header(baseline, [base_tasks[t] for t in shared]),
        "candidate": _run_header(candidate, [cand_tasks[t] for t in shared]),
        "shared_tasks": len(rows),
        "delta": fmean(deltas),
        "delta_ci_95": list(ci),
        "metric_deltas": {
            metric: _mean_or_none([row["metric_deltas"][metric] for row in rows])
            for metric in DELTA_METRICS
        },
        "classes": dict(sorted(classes.items())),
        "slices": slice_rates(rows, "delta", specs),
        "excluded": excluded,
        "warnings": warnings,
        "config_diff": config_diff(baseline.config, candidate.config),
        "tasks": rows,
    }


def config_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Field-level differences between two agent configs (FR-CMP-07)."""
    flat_base, flat_cand = _flatten(baseline), _flatten(candidate)
    changes = []
    for field in sorted(set(flat_base) | set(flat_cand)):
        before, after = flat_base.get(field), flat_cand.get(field)
        if before == after:
            continue
        change: dict[str, Any] = {"field": field, "baseline": before, "candidate": after}
        if isinstance(before, str) and isinstance(after, str) and "\n" in before + after:
            change["diff"] = list(
                difflib.unified_diff(
                    before.splitlines(), after.splitlines(), "baseline", "candidate", lineterm=""
                )
            )
        changes.append(change)
    return changes


def _task_stats(trials: list[dict[str, Any]]) -> dict[str, _TaskStats]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_task[trial["task_id"]].append(trial)
    stats = {}
    for task_id, task_trials in by_task.items():
        scored = [trial for trial in task_trials if trial.get("outcome") in SCORED_OUTCOMES]
        stats[task_id] = _TaskStats(
            version=task_trials[0]["task_version"],
            n=len(scored),
            passes=sum(trial["outcome"] == "pass" for trial in scored),
            means={
                metric: _mean_or_none([_metric(t, metric) for t in scored])
                for metric in DELTA_METRICS
            },
            trials=[
                {
                    "trial_id": t["trial_id"],
                    "trial_index": t["trial_index"],
                    "outcome": t.get("outcome"),
                }
                for t in sorted(task_trials, key=lambda t: t["trial_index"])
            ],
        )
    return stats


def _pair(
    base: dict[str, _TaskStats], cand: dict[str, _TaskStats], allow_version_drift: bool
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    excluded: dict[str, list[str]] = {
        "only_in_baseline": sorted(set(base) - set(cand)),
        "only_in_candidate": sorted(set(cand) - set(base)),
        "version_drift": [],
        "unscored": [],
    }
    warnings = []
    shared = []
    for task_id in sorted(set(base) & set(cand)):
        if base[task_id].n == 0 or cand[task_id].n == 0:
            excluded["unscored"].append(task_id)
            continue
        if base[task_id].version != cand[task_id].version:
            if not allow_version_drift:
                excluded["version_drift"].append(task_id)
                continue
            warnings.append(
                f"{task_id}: comparing version {base[task_id].version} with "
                f"{cand[task_id].version} (--allow-version-drift)"
            )
        shared.append(task_id)
    return shared, excluded, warnings


def _task_row(
    task_id: str, base: _TaskStats, cand: _TaskStats, spec: dict[str, Any]
) -> dict[str, Any]:
    p_worse = fisher_worse(base.passes, base.n, cand.passes, cand.n)
    p_better = fisher_worse(cand.passes, cand.n, base.passes, base.n)
    return {
        "task_id": task_id,
        "critical": bool(spec.get("critical")),
        "class": _classify(base, cand, p_worse, p_better),
        "baseline": {
            "n": base.n,
            "passes": base.passes,
            "rate": base.rate,
            "version": base.version,
        },
        "candidate": {
            "n": cand.n,
            "passes": cand.passes,
            "rate": cand.rate,
            "version": cand.version,
        },
        "delta": cand.rate - base.rate,
        "p_worse": p_worse,
        "p_better": p_better,
        "metric_deltas": {
            metric: None
            if base.means[metric] is None or cand.means[metric] is None
            else cand.means[metric] - base.means[metric]
            for metric in DELTA_METRICS
        },
        "baseline_trials": base.trials,
        "candidate_trials": cand.trials,
    }


def _classify(base: _TaskStats, cand: _TaskStats, p_worse: float, p_better: float) -> str:
    # Compare rates exactly with integers: cand.passes / cand.n versus base.passes / base.n.
    direction = cand.passes * base.n - base.passes * cand.n
    if direction < 0 and p_worse < ALPHA:
        return "regressed"
    if direction > 0 and p_better < ALPHA:
        return "improved"
    if direction != 0:
        return "changed"
    if 0 < base.passes < base.n or 0 < cand.passes < cand.n:
        return "flaky"
    return "stable_pass" if base.passes == base.n else "stable_fail"


def _verdict(ci: tuple[float, float], rows: list[dict[str, Any]]) -> str:
    if ci[1] < 0 or any(row["critical"] and row["class"] == "regressed" for row in rows):
        return "REGRESSION"
    if ci[0] > 0:
        return "IMPROVEMENT"
    return "NO_SIGNIFICANT_CHANGE"


def _run_header(inputs: RunInputs, shared: list[_TaskStats]) -> dict[str, Any]:
    run = inputs.run
    return {
        "run_id": run["run_id"],
        "config_name": run.get("config_name"),
        "config_hash": run.get("config_hash"),
        "suite_id": run.get("suite_id"),
        "suite_version": run.get("suite_version"),
        "macro_pass_rate": fmean(stats.rate for stats in shared),
    }


def _metric(trial: dict[str, Any], metric: str) -> float | None:
    if metric == "tokens":
        if trial.get("input_tokens") is None:
            return None
        return (trial.get("input_tokens") or 0) + (trial.get("output_tokens") or 0)
    return trial.get(metric)


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        flat: dict[str, Any] = {}
        for key, inner in value.items():
            flat.update(_flatten(inner, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}
