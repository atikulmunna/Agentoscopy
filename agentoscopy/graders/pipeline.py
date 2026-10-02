"""Grading pipeline (WF-05): grade a stopped sandbox's snapshot in a fresh grading sandbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentoscopy.graders.base import (
    Grader,
    GraderError,
    GradeResult,
    GradingContext,
    Judge,
    JudgeError,
    aggregate,
)
from agentoscopy.graders.command import CommandGrader
from agentoscopy.graders.judge import JudgeGrader
from agentoscopy.graders.tamper import TamperGrader
from agentoscopy.graders.trajectory import (
    ForbiddenCommandGrader,
    MaxStepsGrader,
    MaxToolErrorsGrader,
    MustReadBeforeEditGrader,
)
from agentoscopy.sandbox.base import ManagedSandbox, SandboxBackend
from agentoscopy.spec import GraderSpecBase, Task

GRADERS = {
    "command": CommandGrader,
    "forbidden_command": ForbiddenCommandGrader,
    "must_read_before_edit": MustReadBeforeEditGrader,
    "max_tool_errors": MaxToolErrorsGrader,
    "max_steps": MaxStepsGrader,
    "tamper_check": TamperGrader,
    "llm_judge": JudgeGrader,
}
# tamper_check ignores caches, so they are removed before any grader runs: a planted .pyc
# must not stand in for the source it claims to match (WF-05 step 1).
CACHE_CLEANUP = (
    "find {workdir} /tmp \\( -name __pycache__ -o -name .pytest_cache \\) -prune "
    "-exec rm -rf {{}} + 2>/dev/null; true"
)


@dataclass(frozen=True)
class GradingOutcome:
    grades: list[GradeResult]
    passed: bool
    score: float
    vetoed: bool = False  # tamper_check failed, so the trial fails whatever else passed


def build_grader(spec: GraderSpecBase) -> Grader:
    return GRADERS[spec.type](spec)


async def grade_snapshot(
    backend: SandboxBackend,
    snapshot: str,
    task: Task,
    label: str,
    sandboxes: list[ManagedSandbox],
    *,
    verified: frozenset[str] = frozenset(),
    trajectory: Sequence[dict[str, Any]] = (),
    final_message: str | None = None,
    fs_diff: str = "",
    judge: Judge | None = None,
    judge_votes: int = 1,
) -> GradingOutcome:
    """Grade a snapshot. The grading sandbox joins `sandboxes`; the caller destroys it."""
    grading = await backend.create_grading(snapshot, task, label)
    sandboxes.append(grading)
    await grading.exec(CACHE_CLEANUP.format(workdir=task.spec.environment.workdir))
    context = GradingContext(
        task, grading, list(trajectory), final_message, fs_diff, judge, judge_votes
    )
    grades = [await _grade(spec, context, verified) for spec in task.spec.graders]
    passed, score = aggregate(task.spec.graders, grades)
    vetoed = any(grade.kind == "tamper" and not grade.passed for grade in grades)
    if vetoed:
        passed, score = False, 0.0
    return GradingOutcome(grades, passed, score, vetoed)


async def _grade(
    spec: GraderSpecBase, context: GradingContext, verified: frozenset[str]
) -> GradeResult:
    try:
        return await build_grader(spec).grade(context)
    except JudgeError:
        raise  # a judge failure is never the agent's (WF-05 E2)
    except GraderError as exc:
        if spec.name not in verified:
            raise
        # The grader completed on the reference solution, so this failure is caused by the
        # agent's final state (for example, tests that now hang): the agent fails (WF-05 E1).
        return GradeResult(spec.name, 0.0, False, str(exc), {"grader_error": True})
