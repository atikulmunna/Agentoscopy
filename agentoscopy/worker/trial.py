"""One trial attempt (WF-04): provision, run the agent through the gateway, snapshot, grade.

The scheduler owns retries and persistence; this module executes an attempt and reports it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from agentoscopy.adapters.base import AgentAdapter, TaskContext
from agentoscopy.gateway.server import Gateway
from agentoscopy.gateway.session import TrialSession, Usage
from agentoscopy.graders.base import GraderError, GradeResult
from agentoscopy.graders.fsdiff import since_baseline
from agentoscopy.graders.judge import GatewayJudge
from agentoscopy.graders.pipeline import grade_snapshot
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.sandbox.base import ManagedSandbox, SandboxBackend, SandboxError
from agentoscopy.sandbox.recording import RecordingSandbox
from agentoscopy.spec import AgentConfig, Task, config_hash

SETUP_TIMEOUT_S = 600
CANCEL_GRACE_S = 30.0  # how long a stopped agent gets to wind down (WF-12)
ERROR_TAIL_CHARS = 2000
RETRYABLE_ERROR_CODES = frozenset({"SANDBOX_ERROR", "PROVIDER_ERROR"})

T = TypeVar("T")
StateCallback = Callable[[str], None]


class SetupError(Exception):
    """A per-trial setup command failed."""


class TaskImageChanged(Exception):
    """The task image no longer matches the digest pinned at validation."""


class ProviderError(Exception):
    """The model provider stayed unavailable after the gateway's retries (WF-04 E3)."""


class AttemptCancelled(Exception):
    """The run is being cancelled, and this attempt had not reached grading (WF-12)."""


INFRA_ERRORS: tuple[tuple[type[Exception], str], ...] = (
    (SetupError, "SETUP_BROKEN"),
    (TaskImageChanged, "TASK_IMAGE_CHANGED"),
    (GraderError, "GRADER_UNSTABLE"),
    (ProviderError, "PROVIDER_ERROR"),
    (SandboxError, "SANDBOX_ERROR"),
)


@dataclass(frozen=True)
class AttemptSpec:
    run_id: str
    trial_id: str
    attempt: int
    task: Task
    config: AgentConfig
    seed_key: str
    image_digest: str | None = None
    verified_graders: frozenset[str] = frozenset()
    judge_model: str | None = None  # pinned per run; used only by llm_judge graders


@dataclass(frozen=True)
class AttemptResult:
    outcome: str  # pass | fail | infra_error | cancelled
    termination: str | None
    score: float
    grades: list[GradeResult]
    error_code: str | None
    error: str | None
    cleanup_errors: list[str]
    usage: Usage
    duration_s: float
    trajectory_path: Path
    final_state_path: Path | None
    vetoed: bool = False  # tamper_check failed: reward hacking (FR-GRD-07)
    judge_cost_usd: float = 0.0  # tracked apart from agent spend (NFR-COST-02)

    @property
    def retryable(self) -> bool:
        return self.error_code in RETRYABLE_ERROR_CODES


@dataclass
class _Resources:
    """What an attempt acquired, so cleanup and usage reporting survive failures."""

    sandboxes: list[ManagedSandbox] = field(default_factory=list)
    session: TrialSession | None = None
    final_state_path: Path | None = None
    judge: GatewayJudge | None = None


@dataclass
class _Outcome:
    outcome: str
    termination: str | None = None
    score: float = 0.0
    grades: list[GradeResult] = field(default_factory=list)
    error_code: str | None = None
    error: str | None = None
    vetoed: bool = False


def attempt_paths(output_root: Path, run_id: str, trial_id: str, attempt: int) -> tuple[Path, Path]:
    """(trajectory file, artifacts dir) for one attempt."""
    trajectory = output_root / "trajectories" / run_id / trial_id / f"attempt-{attempt}.jsonl"
    return trajectory, output_root / "artifacts" / run_id / trial_id / f"attempt-{attempt}"


async def run_attempt(
    spec: AttemptSpec,
    adapter: AgentAdapter,
    backend: SandboxBackend,
    gateway: Gateway,
    output_root: Path,
    on_state: StateCallback | None = None,
    cancel: asyncio.Event | None = None,
) -> AttemptResult:
    """Run one attempt. Setting `cancel` stops it unless it is already grading."""
    trajectory_path, artifacts_dir = attempt_paths(
        output_root, spec.run_id, spec.trial_id, spec.attempt
    )
    recorder = TrajectoryRecorder(trajectory_path, artifacts_dir, spec.trial_id, spec.attempt)
    resources = _Resources()
    started = time.monotonic()
    try:
        recorder.write("trial_start", _start_payload(spec))
        try:
            outcome = await _execute(
                spec,
                adapter,
                backend,
                gateway,
                recorder,
                resources,
                on_state or _ignore_state,
                cancel or asyncio.Event(),
            )
        except AttemptCancelled:
            outcome = _Outcome("cancelled", termination="cancelled")
        except tuple(error_class for error_class, _ in INFRA_ERRORS) as exc:
            outcome = _Outcome("infra_error", error_code=_error_code(exc), error=str(exc))
        finally:
            cleanup_errors = await _destroy(resources.sandboxes, recorder)
        duration_s = round(time.monotonic() - started, 3)
        usage = resources.session.usage if resources.session else Usage()
        judge_cost = resources.judge.cost_usd if resources.judge else 0.0
        recorder.write("trial_end", _end_payload(outcome, usage, duration_s, judge_cost))
    finally:
        recorder.close()
        if resources.judge:
            resources.judge.close()
    return AttemptResult(
        outcome=outcome.outcome,
        termination=outcome.termination,
        score=outcome.score,
        grades=outcome.grades,
        error_code=outcome.error_code,
        error=outcome.error,
        cleanup_errors=cleanup_errors,
        usage=usage,
        duration_s=duration_s,
        trajectory_path=trajectory_path,
        final_state_path=resources.final_state_path,
        vetoed=outcome.vetoed,
        judge_cost_usd=judge_cost,
    )


async def _execute(
    spec: AttemptSpec,
    adapter: AgentAdapter,
    backend: SandboxBackend,
    gateway: Gateway,
    recorder: TrajectoryRecorder,
    resources: _Resources,
    on_state: StateCallback,
    cancel: asyncio.Event,
) -> _Outcome:
    task = spec.task
    _check(cancel)
    image = await backend.prepare_image(task)
    if spec.image_digest and image != spec.image_digest:
        raise TaskImageChanged(
            f"task image is {image} but validation pinned {spec.image_digest}; "
            f"re-run `agentoscopy task validate {task.spec.id}`"
        )
    sandbox = await backend.create(image, task, spec.trial_id)
    resources.sandboxes.append(sandbox)
    await run_setup(sandbox, task)
    baseline = await sandbox.diff() if task.spec.environment.setup else ""
    _check(cancel)
    session = gateway.open_session(spec.trial_id, task.spec.budget, recorder, spec.seed_key)
    resources.session = session
    on_state("RUNNING")
    try:
        termination, final_message = await _run_agent(
            spec, adapter, sandbox, recorder, session, cancel
        )
    finally:
        gateway.close_session(session)
    if session.provider_error:
        raise ProviderError(session.provider_error)
    fs_diff = since_baseline(await sandbox.diff(), baseline)
    recorder.store_artifact("final_state.diff", fs_diff.encode())
    resources.final_state_path = recorder.artifacts_dir / "final_state.diff"
    await sandbox.stop()
    on_state("GRADING")
    resources.judge = _judge(spec, gateway, recorder)
    graded = await grade_snapshot(
        backend,
        await sandbox.snapshot(),
        task,
        spec.trial_id,
        resources.sandboxes,
        verified=spec.verified_graders,
        trajectory=read_events(recorder.path),
        final_message=final_message,
        fs_diff=fs_diff,
        judge=resources.judge,
    )
    outcome = "pass" if graded.passed else "fail"
    return _Outcome(outcome, termination, graded.score, graded.grades, vetoed=graded.vetoed)


def _judge(
    spec: AttemptSpec, gateway: Gateway, recorder: TrajectoryRecorder
) -> GatewayJudge | None:
    """A judge for this attempt's llm_judge graders, logging to its own grading file."""
    if not spec.judge_model or not spec.task.spec.has_judges():
        return None
    grading_log = TrajectoryRecorder(
        recorder.artifacts_dir / "grading.jsonl",
        recorder.artifacts_dir / "grading",
        spec.trial_id,
        spec.attempt,
    )
    return GatewayJudge(gateway, spec.judge_model, grading_log, spec.seed_key)


async def run_setup(sandbox: ManagedSandbox, task: Task) -> None:
    for command in task.spec.environment.setup:
        result = await sandbox.exec(command, timeout_s=SETUP_TIMEOUT_S)
        if result.exit_code != 0 or result.timed_out:
            output = (result.stdout + result.stderr).decode("utf-8", "replace")
            raise SetupError(
                f"SETUP_BROKEN: {command!r} exited {result.exit_code}\n{output[-ERROR_TAIL_CHARS:]}"
            )


async def _run_agent(
    spec: AttemptSpec,
    adapter: AgentAdapter,
    sandbox: ManagedSandbox,
    recorder: TrajectoryRecorder,
    session: TrialSession,
    cancel: asyncio.Event,
) -> tuple[str, str | None]:
    """Run the agent. Budgets, timeouts, and crashes are agent outcomes and still get graded;
    a cancelled run is not, so AttemptCancelled passes through."""
    task = spec.task
    context = TaskContext(task.spec.id, task.spec.instructions, task.spec.environment.workdir)
    agent_sandbox = RecordingSandbox(sandbox, recorder)
    final_message = None
    try:
        await adapter.setup(spec.config)
        result = await _with_deadline(
            adapter.run(
                context, agent_sandbox, session.endpoint, task.spec.budget, recorder.for_adapter()
            ),
            task.spec.budget.timeout_s,
            extension=lambda: session.backoff_s,
            cancel=cancel,
        )
        termination, final_message = "agent_done", result.final_message
    except TimeoutError:
        termination = "timeout"
    except AttemptCancelled:
        raise
    except Exception as exc:  # agent code is arbitrary; a crash counts against the agent
        recorder.write("error", {"source": "agent", "message": f"{type(exc).__name__}: {exc}"})
        termination = "agent_error"
    finally:
        await _teardown(adapter, recorder)
    if session.exhausted and termination != "timeout":
        termination = f"budget_{session.exhausted}"
    return termination, final_message


async def _with_deadline(
    coroutine: Awaitable[T],
    timeout_s: float,
    extension: Callable[[], float],
    cancel: asyncio.Event | None = None,
) -> T:
    """Await `coroutine`, cancelling it after timeout_s plus extension() seconds, or as soon
    as `cancel` is set (raising AttemptCancelled).

    The extension can grow while waiting: provider backoff must not become an agent timeout.
    A stopped coroutine gets CANCEL_GRACE_S to wind down before the caller moves on.
    """
    task = asyncio.ensure_future(coroutine)
    stop = asyncio.ensure_future(cancel.wait()) if cancel else None
    waiters = {task} if stop is None else {task, stop}
    started = time.monotonic()
    try:
        while not task.done():
            if cancel is not None and cancel.is_set():
                raise AttemptCancelled
            remaining = started + timeout_s + extension() - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait(waiters, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        return task.result()
    finally:
        if stop is not None:
            stop.cancel()
        if not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=CANCEL_GRACE_S)


async def _teardown(adapter: AgentAdapter, recorder: TrajectoryRecorder) -> None:
    try:
        await adapter.teardown()
    except Exception as exc:  # recorded, but must not change an already-decided outcome
        message = f"teardown failed: {type(exc).__name__}: {exc}"
        recorder.write("error", {"source": "agent", "message": message})


async def _destroy(sandboxes: list[ManagedSandbox], recorder: TrajectoryRecorder) -> list[str]:
    errors = []
    for sandbox in reversed(sandboxes):
        try:
            await sandbox.destroy()
        except SandboxError as exc:
            errors.append(str(exc))
            recorder.write("error", {"source": "sandbox", "message": str(exc)})
    return errors


def recorded_spend(
    output_root: Path, run_id: str, trial_id: str, attempt: int
) -> tuple[float, float]:
    """(agent, judge) spend recorded by an attempt that never finished, read back from its
    trajectory and grading log: abandoned attempts still count toward the run (WF-04 E5)."""
    trajectory, artifacts = attempt_paths(output_root, run_id, trial_id, attempt)
    return _logged_cost(trajectory), _logged_cost(artifacts / "grading.jsonl")


def _logged_cost(path: Path) -> float:
    if not path.is_file():
        return 0.0
    return sum(
        event["payload"].get("cost_usd") or 0.0
        for event in read_events(path)
        if event["type"] == "model_response"
    )


def _check(cancel: asyncio.Event) -> None:
    if cancel.is_set():
        raise AttemptCancelled


def _error_code(exc: Exception) -> str:
    return next(code for error_class, code in INFRA_ERRORS if isinstance(exc, error_class))


def _ignore_state(state: str) -> None:
    return None


def _start_payload(spec: AttemptSpec) -> dict[str, Any]:
    return {
        "task_id": spec.task.spec.id,
        "content_hash": spec.task.content_hash,
        "config_name": spec.config.name,
        "config_hash": config_hash(spec.config),
        "seed_key": spec.seed_key,
        "budget": spec.task.spec.budget.model_dump(),
    }


def _end_payload(
    outcome: _Outcome, usage: Usage, duration_s: float, judge_cost_usd: float
) -> dict[str, Any]:
    return {
        "termination": outcome.termination,
        "outcome": outcome.outcome,
        "steps": usage.steps,
        "totals": {
            "duration_s": duration_s,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "cost_usd": round(usage.cost_usd, 8),
            "judge_cost_usd": round(judge_cost_usd, 8),
        },
        "vetoed": outcome.vetoed,
        "error_code": outcome.error_code,
        "error": outcome.error,
    }
