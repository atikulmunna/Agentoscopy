"""Run planning (WF-03 steps 1 to 5): pin validated task versions and lay out the trials."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from agentoscopy.gateway.pricing import is_mock, price_for
from agentoscopy.spec import Task, TaskSpec, load_task
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


@dataclass(frozen=True)
class TaskFilter:
    """Narrows a run to some of its tasks (FR-RUN-02). Several values for one field are
    alternatives; the fields themselves must all match."""

    tags: frozenset[str] = field(default_factory=frozenset)
    categories: frozenset[str] = field(default_factory=frozenset)
    difficulties: frozenset[str] = field(default_factory=frozenset)
    failed_in: str | None = None  # a run whose failed tasks to take

    def __bool__(self) -> bool:
        return bool(self.tags or self.categories or self.difficulties or self.failed_in)

    def keeps(self, spec: TaskSpec, failed: set[str] | None) -> bool:
        return (
            (not self.tags or bool(self.tags & set(spec.tags)))
            and (not self.categories or spec.category in self.categories)
            and (not self.difficulties or spec.difficulty in self.difficulties)
            and (failed is None or spec.id in failed)
        )

    def describe(self) -> str:
        parts = [
            f"{name}={'|'.join(sorted(values))}"
            for name, values in (
                ("tag", self.tags),
                ("category", self.categories),
                ("difficulty", self.difficulties),
            )
            if values
        ]
        return " ".join(parts + ([f"failed-in={self.failed_in}"] if self.failed_in else []))


def select_tasks(
    store: Store, tasks_dir: Path, task_ids: list[str], task_filter: TaskFilter
) -> list[str]:
    """The task ids the filter keeps, in their original order."""
    failed = None
    if task_filter.failed_in:
        failed = store.tasks_not_passed(task_filter.failed_in)
        if failed is None:
            raise PlanError(f"NOT_FOUND: no run {task_filter.failed_in}")
    kept = [
        task_id
        for task_id in task_ids
        if task_filter.keeps(load_task(tasks_dir, task_id).spec, failed)
    ]
    if not kept:
        raise PlanError(f"NO_TASKS_SELECTED: no task matches {task_filter.describe()}")
    return kept


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


def check_judge(tasks: list[Task], judge_model: str, has_credentials: bool) -> str | None:
    """The judge model to pin, or None when no task has llm_judge graders. A real judge model
    needs provider credentials, which only the credential proxy holds."""
    if not any(task.spec.has_judges() for task in tasks):
        return None
    if price_for(judge_model) is None:
        raise PlanError(
            f"UNKNOWN_JUDGE_MODEL: no price for {judge_model!r}, so judge spend cannot be tracked"
        )
    if not is_mock(judge_model) and not has_credentials:
        raise PlanError(
            f"JUDGE_NEEDS_CREDENTIALS: judge model {judge_model!r} needs ANTHROPIC_API_KEY "
            "(or pass a mock judge model)"
        )
    return judge_model


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
