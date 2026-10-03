"""Creating, resuming, and executing runs, shared by the CLI and the HTTP API (WF-03, WF-12)."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

from agentoscopy.adapters.python import AdapterLoadError, load_python_adapter
from agentoscopy.gateway.server import Gateway
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.scheduler.plan import (
    PinnedTask,
    PlanError,
    check_budget,
    check_judge,
    pin_tasks,
    plan_trials,
)
from agentoscopy.scheduler.runner import REVIEW_RATE, ProgressCallback, RunExecutor
from agentoscopy.spec import (
    AgentConfig,
    SpecError,
    load_agent_config,
    load_suite,
    load_task,
)
from agentoscopy.storage.store import RunSettings, Store, TaskVersion

DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
DEFAULT_CONCURRENCY = 4
RESUMABLE = ("pending", "running", "cancelling", "failed")


class LaunchError(Exception):
    """A run cannot be created or resumed as asked. Messages start with an error code."""


@dataclass(frozen=True)
class Directories:
    tasks: Path
    suites: Path
    home: Path


@dataclass(frozen=True)
class RunRequest:
    agent: Path
    suite: str | None = None
    tasks: tuple[str, ...] = ()
    trials: int = 3
    budget_usd: float | None = 10.0
    concurrency: int = DEFAULT_CONCURRENCY
    seed: int | None = None
    labels: dict[str, str] = field(default_factory=dict)
    judge_model: str = DEFAULT_JUDGE_MODEL
    review_rate: float = REVIEW_RATE


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    pinned: list[PinnedTask]
    config: AgentConfig
    settings: RunSettings


def create_run(
    store: Store, dirs: Directories, request: RunRequest, *, has_credentials: bool
) -> PreparedRun:
    """Pin the tasks, check the plan, and record the run with its queued trials."""
    try:
        task_ids = load_suite(dirs.suites, request.suite).tasks if request.suite else request.tasks
        pinned = pin_tasks(store, dirs.tasks, list(task_ids))
        check_budget(pinned, request.budget_usd)
        tasks = [item.task for item in pinned]
        judge_model = check_judge(tasks, request.judge_model, has_credentials)
        config = load_agent_config(request.agent)
        load_python_adapter(config.entrypoint)  # fail before the run exists if it cannot load
    except (SpecError, PlanError, AdapterLoadError) as exc:
        raise LaunchError(str(exc)) from exc
    suite_version = None
    if request.suite:
        refs = [(item.task.spec.id, item.version) for item in pinned]
        suite_version = store.save_suite_version(request.suite, refs)
    settings = RunSettings(
        suite_id=request.suite,
        suite_version=suite_version,
        trials_per_task=request.trials,
        budget_usd=request.budget_usd,
        seed=request.seed if request.seed is not None else secrets.randbelow(2**31),
        concurrency=request.concurrency,
        harness_version=version("agentoscopy"),
        labels=dict(request.labels),
        judge_model=judge_model,
        review_rate=request.review_rate,
    )
    run_id = store.create_run(config, settings, plan_trials(pinned, request.trials, settings.seed))
    return PreparedRun(run_id, pinned, config, settings)


def resume_run(
    store: Store,
    dirs: Directories,
    run_id: str,
    *,
    has_credentials: bool,
    concurrency: int | None = None,
) -> PreparedRun:
    """Pick up an unfinished run with the task versions and config it started with (WF-12).
    The tasks on disk must still have the content that was pinned."""
    run = store.get_run(run_id)
    if run is None:
        raise LaunchError(f"NOT_FOUND: no run {run_id}")
    if run["status"] not in RESUMABLE:
        raise LaunchError(f"RUN_FINISHED: run {run_id} is already {run['status']}")
    try:
        pinned = [_repin(dirs.tasks, version) for version in store.run_task_versions(run_id)]
        config = AgentConfig.model_validate(store.config_spec(run["config_hash"]))
        load_python_adapter(config.entrypoint)
        if run["judge_model"]:
            check_judge([item.task for item in pinned], run["judge_model"], has_credentials)
    except (SpecError, PlanError, AdapterLoadError) as exc:
        raise LaunchError(str(exc)) from exc
    settings = RunSettings(
        suite_id=run["suite_id"],
        suite_version=run["suite_version"],
        trials_per_task=run["trials_per_task"],
        budget_usd=run["budget_usd"],
        seed=run["seed"],
        concurrency=concurrency or run["concurrency"],
        harness_version=run["harness_version"],
        labels=run["labels"],
        judge_model=run["judge_model"],
        review_rate=REVIEW_RATE if run["review_rate"] is None else run["review_rate"],
    )
    return PreparedRun(run_id, pinned, config, settings)


async def execute_run(
    store: Store,
    prepared: PreparedRun,
    home: Path,
    backend: SandboxBackend,
    gateway: Gateway,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Run the trials, then store the summary, however the run ended. Returns the status."""
    settings = prepared.settings
    executor = RunExecutor(
        store,
        prepared.run_id,
        prepared.pinned,
        prepared.config,
        backend,
        gateway,
        home,
        concurrency=settings.concurrency,
        seed=settings.seed,
        on_progress=on_progress,
        judge_model=settings.judge_model,
        review_rate=REVIEW_RATE if settings.review_rate is None else settings.review_rate,
    )
    try:
        await executor.execute()
    finally:
        store.save_summary(prepared.run_id, run_summary(store, prepared.run_id, refresh=True))
    return store.run_status(prepared.run_id) or "unknown"


def _repin(tasks_dir: Path, pinned_version: TaskVersion) -> PinnedTask:
    task = load_task(tasks_dir, pinned_version.task_id)
    if task.content_hash != pinned_version.content_hash:
        raise PlanError(
            f"TASK_CHANGED: {pinned_version.task_id} no longer has the content of version "
            f"{pinned_version.version}, which the run pinned; restore it to resume"
        )
    return PinnedTask(
        task, pinned_version.version, pinned_version.image_digest, pinned_version.verified_graders
    )
