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
from agentoscopy.graders.pipeline import grade_snapshot
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.sandbox.base import ManagedSandbox, SandboxBackend, SandboxError
from agentoscopy.sandbox.recording import RecordingSandbox
from agentoscopy.spec import AgentConfig, Task, config_hash

SETUP_TIMEOUT_S = 600
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


@dataclass(frozen=True)
class AttemptResult:
    outcome: str  # pass | fail | infra_error
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

    @property
    def retryable(self) -> bool:
        return self.error_code in RETRYABLE_ERROR_CODES


@dataclass
class _Resources:
    """What an attempt acquired, so cleanup and usage reporting survive failures."""

    sandboxes: list[ManagedSandbox] = field(default_factory=list)
    session: TrialSession | None = None
    final_state_path: Path | None = None


@dataclass
class _Outcome:
    outcome: str
    termination: str | None = None
    score: float = 0.0
    grades: list[GradeResult] = field(default_factory=list)
    error_code: str | None = None
    error: str | None = None


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
) -> AttemptResult:
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
                spec, adapter, backend, gateway, recorder, resources, on_state or _ignore_state
            )
        except tuple(error_class for error_class, _ in INFRA_ERRORS) as exc:
            outcome = _Outcome("infra_error", error_code=_error_code(exc), error=str(exc))
        finally:
            cleanup_errors = await _destroy(resources.sandboxes, recorder)
        duration_s = round(time.monotonic() - started, 3)
        usage = resources.session.usage if resources.session else Usage()
        recorder.write("trial_end", _end_payload(outcome, usage, duration_s))
    finally:
        recorder.close()
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
    )


async def _execute(
    spec: AttemptSpec,
    adapter: AgentAdapter,
    backend: SandboxBackend,
    gateway: Gateway,
    recorder: TrajectoryRecorder,
    resources: _Resources,
    on_state: StateCallback,
) -> _Outcome:
    task = spec.task
    image = await backend.prepare_image(task)
    if spec.image_digest and image != spec.image_digest:
        raise TaskImageChanged(
            f"task image is {image} but validation pinned {spec.image_digest}; "
            f"re-run `agentoscopy task validate {task.spec.id}`"
        )
    sandbox = await backend.create(image, task, spec.trial_id)
    resources.sandboxes.append(sandbox)
    await run_setup(sandbox, task)
    session = gateway.open_session(spec.trial_id, task.spec.budget, recorder, spec.seed_key)
    resources.session = session
    on_state("RUNNING")
    try:
        termination, final_message = await _run_agent(spec, adapter, sandbox, recorder, session)
    finally:
        gateway.close_session(session)
    if session.provider_error:
        raise ProviderError(session.provider_error)
    fs_diff = await sandbox.diff()
    recorder.store_artifact("final_state.diff", fs_diff.encode())
    resources.final_state_path = recorder.artifacts_dir / "final_state.diff"
    await sandbox.stop()
    on_state("GRADING")
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
    )
    return _Outcome("pass" if graded.passed else "fail", termination, graded.score, graded.grades)


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
) -> tuple[str, str | None]:
    """Run the agent. Budgets, timeouts, and crashes are agent outcomes and still get graded."""
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
        )
        termination, final_message = "agent_done", result.final_message
    except TimeoutError:
        termination = "timeout"
    except Exception as exc:  # agent code is arbitrary; a crash counts against the agent
        recorder.write("error", {"source": "agent", "message": f"{type(exc).__name__}: {exc}"})
        termination = "agent_error"
    finally:
        await _teardown(adapter, recorder)
    if session.exhausted and termination != "timeout":
        termination = f"budget_{session.exhausted}"
    return termination, final_message


async def _with_deadline(
    coroutine: Awaitable[T], timeout_s: float, extension: Callable[[], float]
) -> T:
    """Await `coroutine`, cancelling it after timeout_s plus extension() seconds.

    The extension can grow while waiting: provider backoff must not become an agent timeout.
    """
    task = asyncio.ensure_future(coroutine)
    started = time.monotonic()
    try:
        while not task.done():
            remaining = started + timeout_s + extension() - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            await asyncio.wait({task}, timeout=remaining)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


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


def _end_payload(outcome: _Outcome, usage: Usage, duration_s: float) -> dict[str, Any]:
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
        },
        "error_code": outcome.error_code,
        "error": outcome.error,
    }
