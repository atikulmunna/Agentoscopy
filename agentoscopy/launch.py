"""Creating, resuming, and executing runs, shared by the CLI and the HTTP API (WF-03, WF-12)."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

from agentoscopy.adapters.python import AdapterLoadError, load_python_adapter
from agentoscopy.gateway.server import Gateway
from agentoscopy.replay import Playback, load_recording
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.scheduler.plan import (
    PinnedTask,
    PlanError,
    TaskFilter,
    check_budget,
    check_judge,
    pin_tasks,
    plan_trials,
    select_tasks,
)
from agentoscopy.scheduler.runner import REVIEW_RATE, ProgressCallback, RunExecutor
from agentoscopy.spec import (
    AgentConfig,
    SpecError,
    load_agent_config,
    load_suite,
    load_task,
)
from agentoscopy.storage.store import NewTrial, RunSettings, Store, TaskVersion
from agentoscopy.worker.trial import attempt_paths

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
    task_filter: TaskFilter = field(default_factory=TaskFilter)
    top_up: int = 0  # extra trials for each task with mixed results


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    pinned: list[PinnedTask]
    config: AgentConfig
    settings: RunSettings
    playback: Playback | None = None  # for a replay or fork run


def create_run(
    store: Store, dirs: Directories, request: RunRequest, *, has_credentials: bool
) -> PreparedRun:
    """Pin the tasks, check the plan, and record the run with its queued trials.

    A filtered suite run records no suite version: it did not run the whole suite, so it
    must never stand in for one (a CI baseline, say). The filter becomes a label."""
    try:
        task_ids = load_suite(dirs.suites, request.suite).tasks if request.suite else request.tasks
        if request.task_filter:
            task_ids = select_tasks(store, dirs.tasks, list(task_ids), request.task_filter)
        pinned = pin_tasks(store, dirs.tasks, list(task_ids))
        check_budget(pinned, request.budget_usd)
        tasks = [item.task for item in pinned]
        judge_model = check_judge(tasks, request.judge_model, has_credentials)
        config = load_agent_config(request.agent)
        load_python_adapter(config.entrypoint)  # fail before the run exists if it cannot load
    except (SpecError, PlanError, AdapterLoadError) as exc:
        raise LaunchError(str(exc)) from exc
    suite_version = None
    labels = dict(request.labels)
    if request.task_filter:
        labels["filter"] = request.task_filter.describe()
    elif request.suite:
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
        labels=labels,
        judge_model=judge_model,
        review_rate=request.review_rate,
        top_up=request.top_up,
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
    if run["mode"] != "live":
        raise LaunchError(
            f"REPLAY_NOT_RESUMABLE: a {run['mode']} run is not resumed; start it again from "
            "its source trial"
        )
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
        top_up=run["top_up"],
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
        playback=prepared.playback,
        top_up=settings.top_up,
    )
    try:
        await executor.execute()
    finally:
        store.save_summary(prepared.run_id, run_summary(store, prepared.run_id, refresh=True))
    return store.run_status(prepared.run_id) or "unknown"


def create_replay(
    store: Store, dirs: Directories, trial_id: str, *, has_credentials: bool
) -> PreparedRun:
    """A one-trial run that reruns a recorded trial (WF-11): same task version, config,
    and seed, with every model call answered from the trial's final attempt."""
    return _from_recorded_trial(
        store, dirs, trial_id, mode="replay", labels={"replay_of": trial_id},
        has_credentials=has_credentials,
    )  # fmt: skip


def create_fork(
    store: Store,
    dirs: Directories,
    trial_id: str,
    at_step: int,
    *,
    agent: Path | None = None,
    inject: str | None = None,
    has_credentials: bool,
) -> PreparedRun:
    """A one-trial run that replays a recorded trial up to step `at_step`, then continues live
    (WF-11, FR-RPL-03): with another agent config, an injected message, or both."""
    labels = {"fork_of": trial_id, "fork_at": str(at_step)} | (
        {"injected": "yes"} if inject else {}
    )
    return _from_recorded_trial(
        store, dirs, trial_id, mode="fork", labels=labels, has_credentials=has_credentials,
        fork_at=at_step, inject=inject, agent=agent,
    )  # fmt: skip


def _from_recorded_trial(
    store: Store,
    dirs: Directories,
    trial_id: str,
    *,
    mode: str,
    labels: dict[str, str],
    has_credentials: bool,
    fork_at: int | None = None,
    inject: str | None = None,
    agent: Path | None = None,
) -> PreparedRun:
    trial = store.get_trial(trial_id)
    if trial is None:
        raise LaunchError(f"NOT_FOUND: no trial {trial_id}")
    source = store.get_run(trial["run_id"])
    trajectory, artifacts = attempt_paths(dirs.home, source["run_id"], trial_id, trial["attempt"])
    recording = load_recording(trajectory, artifacts) if trajectory.is_file() else []
    if not recording:
        raise LaunchError(f"NO_RECORDING: trial {trial_id} has no recorded model calls")
    if fork_at is not None and not 1 <= fork_at <= len(recording) + 1:
        raise LaunchError(
            f"FORK_STEP_OUT_OF_RANGE: trial {trial_id} recorded {len(recording)} model calls, "
            f"so a fork can start at step 1 to {len(recording) + 1}"
        )
    versions = {v.task_id: v for v in store.run_task_versions(source["run_id"])}
    try:
        pinned = _repin(dirs.tasks, versions[trial["task_id"]])
        if agent is not None:
            config = load_agent_config(agent)
        else:
            config = AgentConfig.model_validate(store.config_spec(source["config_hash"]))
        load_python_adapter(config.entrypoint)
        if source["judge_model"]:
            check_judge([pinned.task], source["judge_model"], has_credentials)
    except (SpecError, PlanError, AdapterLoadError) as exc:
        raise LaunchError(str(exc)) from exc
    settings = RunSettings(
        suite_id=None,
        suite_version=None,
        trials_per_task=1,
        budget_usd=None,  # replayed calls cost nothing; a fork's live calls keep the trial budget
        seed=source["seed"],
        concurrency=1,
        harness_version=version("agentoscopy"),
        labels=labels,
        judge_model=source["judge_model"],
        review_rate=0.0,
        mode=mode,
    )
    new_trial = NewTrial(
        trial["task_id"],
        trial["task_version"],
        trial["trial_index"],  # the same seed key, so a mock judge answers the same
        0,
        pinned.task.spec.budget.max_cost_usd,
        source_trial_id=trial_id,
    )
    run_id = store.create_run(config, settings, [new_trial])
    playback = Playback(tuple(recording), fork_at, inject)
    return PreparedRun(run_id, [pinned], config, settings, playback)


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
