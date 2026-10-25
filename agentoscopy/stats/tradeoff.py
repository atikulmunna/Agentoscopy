"""Cost against quality across runs: which configs no other run beats on both.

Runs are compared on the tasks every one of them scored at the same version, as in WF-07,
so a run on easier tasks cannot look better than it is. For each run: the macro pass rate
over those tasks with its 95% CI (the comparison's t interval over tasks), and the mean
agent cost per scored trial; judge spend is left out, as it is from run budgets. A run is on
the frontier when no other run passes at least as often for no more cost, and is better on
one of the two. That compares point estimates: where intervals overlap, the order may not
hold, and the CIs are reported so a reader can see it.
"""

from __future__ import annotations

from statistics import fmean
from typing import Any

from agentoscopy.stats.compare import (
    FEW_TASKS_WARNING,
    CompareError,
    RunInputs,
    TaskStats,
    mean_ci,
    task_stats,
)
from agentoscopy.stats.summary import SCORED_OUTCOMES

MIN_RUNS = 2
MAX_RUNS = 20


def tradeoff_runs(runs: list[RunInputs]) -> dict[str, Any]:
    check_run_count(len(runs))
    stats = [task_stats(run.trials) for run in runs]
    shared = _shared_tasks(stats)
    if not shared:
        raise CompareError("NO_SHARED_TASKS: no task version was scored in every run")
    points = sorted(
        (_point(run, run_stats, shared) for run, run_stats in zip(runs, stats, strict=True)),
        key=lambda point: (point["cost_per_trial_usd"], -point["pass_rate"]),
    )
    for point in points:
        point["on_frontier"] = not any(_beats(other, point) for other in points)
    warnings = []
    if len(shared) < FEW_TASKS_WARNING:
        warnings.append(
            f"only {len(shared)} shared task{'' if len(shared) == 1 else 's'}: the CIs treat "
            f"tasks as the sampling unit and are unreliable below about {FEW_TASKS_WARNING}"
        )
    return {"shared_tasks": len(shared), "runs": points, "warnings": warnings}


def check_run_count(count: int) -> None:
    """Callers check before loading runs, so a long list is refused without reading it."""
    if not MIN_RUNS <= count <= MAX_RUNS:
        raise CompareError(f"RUN_COUNT: give {MIN_RUNS} to {MAX_RUNS} runs, not {count}")


def _shared_tasks(stats: list[dict[str, TaskStats]]) -> list[str]:
    """Tasks scored in every run, at one version."""
    first, *rest = stats
    return sorted(
        task_id
        for task_id, task in first.items()
        if task.n
        and all(
            task_id in other and other[task_id].n and other[task_id].version == task.version
            for other in rest
        )
    )


def _point(run: RunInputs, stats: dict[str, TaskStats], shared: list[str]) -> dict[str, Any]:
    rates = [stats[task_id].rate for task_id in shared]
    low, high = mean_ci(rates)
    wanted = set(shared)
    scored = [
        trial
        for trial in run.trials
        if trial["task_id"] in wanted and trial.get("outcome") in SCORED_OUTCOMES
    ]
    return {
        "run_id": run.run["run_id"],
        "config_name": run.run.get("config_name"),
        "pass_rate": fmean(rates),
        "pass_rate_ci_95": [max(low, 0.0), min(high, 1.0)],  # a t interval can overshoot
        "cost_per_trial_usd": fmean(trial.get("cost_usd") or 0.0 for trial in scored),
        "trials": len(scored),
        "passes": sum(trial["outcome"] == "pass" for trial in scored),
    }


def _beats(one: dict[str, Any], other: dict[str, Any]) -> bool:
    """`one` passes at least as often for no more cost, and is better on one of the two."""
    no_worse = (
        one["pass_rate"] >= other["pass_rate"]
        and one["cost_per_trial_usd"] <= other["cost_per_trial_usd"]
    )
    better = (
        one["pass_rate"] > other["pass_rate"]
        or one["cost_per_trial_usd"] < other["cost_per_trial_usd"]
    )
    return no_worse and better
