"""The grader contract (§7.4) and the `all_required_pass` aggregation rule (FR-GRD-06)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentoscopy.sandbox.base import Sandbox
from agentoscopy.spec import CommandGraderSpec, Task


@dataclass(frozen=True)
class GradeResult:
    name: str
    score: float  # 0.0 to 1.0
    passed: bool
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradingContext:
    task: Task
    sandbox: Sandbox  # grading sandbox, with hidden/ mounted
    trajectory: list[dict[str, Any]]
    final_message: str | None
    fs_diff: str


class GraderError(Exception):
    """A grader crashed or timed out (WF-05 E1).

    Until graders are verified against reference solutions (M1), this is an infra error.
    """


class Grader(Protocol):
    name: str
    kind: str

    async def grade(self, context: GradingContext) -> GradeResult: ...


def aggregate(
    specs: Sequence[CommandGraderSpec], results: Sequence[GradeResult]
) -> tuple[bool, float]:
    """Pass only if every required grader passes; the score is the weighted mean."""
    pairs = list(zip(specs, results, strict=True))
    passed = all(result.passed for spec, result in pairs if spec.required)
    total_weight = sum(spec.weight for spec, _ in pairs)
    if total_weight == 0:
        return passed, 1.0 if passed else 0.0
    return passed, sum(spec.weight * result.score for spec, result in pairs) / total_weight
