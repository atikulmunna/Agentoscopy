"""Grading pipeline (WF-05): grade a stopped sandbox's snapshot in a fresh grading sandbox."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentoscopy.graders.base import GraderError, GradeResult, GradingContext, aggregate
from agentoscopy.graders.command import CommandGrader
from agentoscopy.sandbox.base import ManagedSandbox, SandboxBackend
from agentoscopy.spec import CommandGraderSpec, Task


@dataclass(frozen=True)
class GradingOutcome:
    grades: list[GradeResult]
    passed: bool
    score: float


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
) -> GradingOutcome:
    """Grade a snapshot. The grading sandbox joins `sandboxes`; the caller destroys it."""
    grading = await backend.create_grading(snapshot, task, label)
    sandboxes.append(grading)
    context = GradingContext(task, grading, list(trajectory), final_message, fs_diff)
    grades = [await _grade(spec, context, verified) for spec in task.spec.graders]
    passed, score = aggregate(task.spec.graders, grades)
    return GradingOutcome(grades, passed, score)


async def _grade(
    spec: CommandGraderSpec, context: GradingContext, verified: frozenset[str]
) -> GradeResult:
    try:
        return await CommandGrader(spec).grade(context)
    except GraderError as exc:
        if spec.name not in verified:
            raise
        # The grader completed on the reference solution, so this failure is caused by the
        # agent's final state (for example, tests that now hang): the agent fails (WF-05 E1).
        return GradeResult(spec.name, 0.0, False, str(exc), {"grader_error": True})
