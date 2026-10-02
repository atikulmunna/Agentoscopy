"""Run planning (WF-03 steps 1 to 5): pin validated task versions and lay out the trials."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from agentoscopy.spec import Task, load_task
from agentoscopy.storage.store import NewTrial, Store

MIN_CRITICAL_TRIALS = 6  # with fewer, a one-sided Fisher test cannot flag a regression


class PlanError(Exception):
    """The run cannot be created as requested. Messages start with an error code."""


@dataclass(frozen=True)
class PinnedTask:
    task: Task
    version: int
    image_digest: str | None
    verified_graders: frozenset[str]


def pin_tasks(store: Store, tasks_dir: Path, task_ids: list[str]) -> list[PinnedTask]:
    """Resolve each task to the validated version matching its current content (FR-RUN-01)."""
    pinned, unvalidated = [], []
    for task_id in task_ids:
        task = load_task(tasks_dir, task_id)
        version = store.task_version(task_id, task.content_hash)
        if version is None or version.validated_at is None:
            unvalidated.append(task_id)
            continue
        pinned.append(
            PinnedTask(task, version.version, version.image_digest, version.verified_graders)
        )
    if unvalidated:
        raise PlanError(
            f"UNVALIDATED_TASKS: {', '.join(unvalidated)} "
            "(run `agentoscopy task validate` after any change to a task)"
        )
    return pinned


def check_budget(pinned: list[PinnedTask], budget_usd: float | None) -> None:
    if budget_usd is None:
        return
    largest = max(item.task.spec.budget.max_cost_usd for item in pinned)
    if budget_usd < largest:
        raise PlanError(
            f"BUDGET_BELOW_TRIAL_CAP: run budget ${budget_usd:.2f} is below a trial's "
            f"max_cost_usd of ${largest:.2f}, so that trial could never be dispatched"
        )


def plan_trials(pinned: list[PinnedTask], trials_per_task: int, seed: int) -> list[NewTrial]:
    """Round-robin by trial index (FR-RUN-09): every task's trial 0, then trial 1, and so on.

    Budget truncation then removes trials evenly across tasks instead of whole tasks. Critical
    tasks get at least MIN_CRITICAL_TRIALS trials so a per-task test can reach significance.
    """
    order = list(pinned)
    random.Random(seed).shuffle(order)
    counts = [trials_for(item.task, trials_per_task) for item in order]
    return [
        NewTrial(
            task_id=item.task.spec.id,
            task_version=item.version,
            trial_index=trial_index,
            dispatch_order=trial_index * len(order) + position,
            cost_cap_usd=item.task.spec.budget.max_cost_usd,
        )
        for trial_index in range(max(counts, default=0))
        for position, (item, count) in enumerate(zip(order, counts, strict=True))
        if trial_index < count
    ]


def trials_for(task: Task, trials_per_task: int) -> int:
    return max(trials_per_task, MIN_CRITICAL_TRIALS) if task.spec.critical else trials_per_task


def estimate_cost(
    store: Store, pinned: list[PinnedTask], config_digest: str, trials_per_task: int
) -> float:
    """Historical mean cost per task for this config, else the task's cost cap (FR-RUN-03)."""
    total = 0.0
    for item in pinned:
        mean = store.historical_mean_cost(item.task.spec.id, config_digest)
        total += trials_for(item.task, trials_per_task) * (
            mean if mean is not None else item.task.spec.budget.max_cost_usd
        )
    return total
