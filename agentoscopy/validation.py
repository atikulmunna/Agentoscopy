"""Task validation (WF-01): build the image once, then the null, reference, and isolation checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from agentoscopy.graders.base import GraderError, Judge
from agentoscopy.graders.fsdiff import since_baseline
from agentoscopy.graders.pipeline import GradingOutcome, grade_snapshot
from agentoscopy.sandbox.base import HIDDEN_MOUNT, ManagedSandbox, SandboxBackend, SandboxError
from agentoscopy.spec import Task
from agentoscopy.worker.trial import SetupError, run_setup

CHECK_FAILURE_CODES = frozenset(
    {"GRADER_PASSES_EMPTY_ENV", "REFERENCE_FAILS", "HIDDEN_FILE_EXPOSED", "GRADER_UNSTABLE"}
)
JUDGE_VOTES = 3  # validation asks each judge three times and takes the majority (WF-01 A2)


class ValidationFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ValidationReport:
    task_id: str
    content_hash: str
    image_digest: str | None = None
    error_code: str | None = None
    message: str = ""
    flags: list[str] = field(default_factory=list)
    verified_graders: list[str] = field(default_factory=list)
    cleanup_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error_code is None


async def validate_task(
    task: Task, backend: SandboxBackend, judge: Judge | None = None
) -> ValidationReport:
    """`judge` is required for tasks with llm_judge graders."""
    report = ValidationReport(task.spec.id, task.content_hash)
    sandboxes: list[ManagedSandbox] = []
    grade = _Grader(task, backend, sandboxes, judge)
    try:
        if task.spec.has_judges() and judge is None:
            raise ValidationFailed("JUDGE_UNAVAILABLE", "this task has llm_judge graders")
        report.image_digest = await backend.prepare_image(task)
        await _null_check(task, backend, report.image_digest, grade)
        report.verified_graders = await _reference_check(task, backend, report.image_digest, grade)
        if not task.reference_files():
            report.flags.append("NO_REFERENCE")
        if all(g.type == "llm_judge" for g in task.spec.graders if g.required):
            report.flags.append("SOFT_GRADER_ONLY")
        _host_isolation_check(task)
    except ValidationFailed as exc:
        report.error_code, report.message = exc.code, str(exc)
    except (SetupError, SandboxError) as exc:
        report.error_code, report.message = "SETUP_BROKEN", str(exc)
    finally:
        for sandbox in reversed(sandboxes):
            try:
                await sandbox.destroy()
            except SandboxError as exc:
                report.cleanup_errors.append(str(exc))
    return report


async def _null_check(task: Task, backend: SandboxBackend, image: str, grade: _Grader) -> None:
    sandbox, baseline = await _fresh_sandbox(task, backend, image, grade.sandboxes)
    probe = await sandbox.exec(f"test -e {HIDDEN_MOUNT}")
    if probe.exit_code == 0:
        raise ValidationFailed("HIDDEN_FILE_EXPOSED", f"{HIDDEN_MOUNT} exists in the agent sandbox")
    graded = await grade(sandbox, baseline)
    if graded.passed:
        raise ValidationFailed(
            "GRADER_PASSES_EMPTY_ENV",
            "the aggregated outcome passes on the untouched environment, so the task tests nothing",
        )


async def _reference_check(
    task: Task, backend: SandboxBackend, image: str, grade: _Grader
) -> list[str]:
    """Apply the reference solution and require a pass, tamper check included; returns the
    graders it verified."""
    files = task.reference_files()
    if not files:
        return []
    sandbox, baseline = await _fresh_sandbox(task, backend, image, grade.sandboxes)
    for file in files:
        await sandbox.write_file(file.relative_to(task.reference_dir).as_posix(), file.read_bytes())
    graded = await grade(sandbox, baseline)
    if not graded.passed:
        failing = [grade.name for grade in graded.grades if not grade.passed]
        raise ValidationFailed("REFERENCE_FAILS", f"reference solution fails graders: {failing}")
    return [grade.name for grade in graded.grades]


def _host_isolation_check(task: Task) -> None:
    """Grader-only files must not also ship as fixtures, which are baked into the agent image."""
    fixture_digests = {_digest(path.read_bytes()) for path in task.fixture_files()}
    leaked = [
        path.relative_to(task.path).as_posix()
        for path in task.hidden_files()
        if _digest(path.read_bytes()) in fixture_digests
    ]
    if leaked:
        raise ValidationFailed(
            "HIDDEN_FILE_EXPOSED", f"hidden files also appear in fixtures: {leaked}"
        )


async def _fresh_sandbox(
    task: Task, backend: SandboxBackend, image: str, sandboxes: list[ManagedSandbox]
) -> tuple[ManagedSandbox, str]:
    """A started sandbox after setup, and its post-setup diff baseline."""
    sandbox = await backend.create(image, task, f"validate-{task.spec.id}")
    sandboxes.append(sandbox)
    await run_setup(sandbox, task)
    return sandbox, (await sandbox.diff() if task.spec.environment.setup else "")


@dataclass
class _Grader:
    """Grades a validation sandbox the way trials are graded, with majority-voting judges."""

    task: Task
    backend: SandboxBackend
    sandboxes: list[ManagedSandbox]
    judge: Judge | None

    async def __call__(self, sandbox: ManagedSandbox, baseline: str) -> GradingOutcome:
        fs_diff = since_baseline(await sandbox.diff(), baseline)
        await sandbox.stop()
        try:
            return await grade_snapshot(
                self.backend,
                await sandbox.snapshot(),
                self.task,
                f"validate-{self.task.spec.id}",
                self.sandboxes,
                fs_diff=fs_diff,
                judge=self.judge,
                judge_votes=JUDGE_VOTES,
            )
        except GraderError as exc:
            raise ValidationFailed("GRADER_UNSTABLE", str(exc)) from exc


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
