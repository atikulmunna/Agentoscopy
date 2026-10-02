"""The grader contract (§7.4) and the `all_required_pass` aggregation rule (FR-GRD-06)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentoscopy.sandbox.base import Sandbox
from agentoscopy.spec import GraderSpecBase, LlmJudgeSpec, Task


@dataclass(frozen=True)
class GradeResult:
    name: str
    score: float  # 0.0 to 1.0
    passed: bool
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)
    kind: str = "deterministic"  # deterministic | trajectory | tamper | judge


@dataclass(frozen=True)
class JudgeVerdictResult:
    passed: bool
    score: float
    confidence: float
    rationale: str


class Judge(Protocol):
    """Runs a judge model on a prompt; grading code never talks to the gateway directly."""

    model: str

    async def verdict(self, spec: LlmJudgeSpec, prompt: str) -> JudgeVerdictResult: ...


@dataclass(frozen=True)
class GradingContext:
    task: Task
    sandbox: Sandbox  # grading sandbox, with hidden/ mounted
    trajectory: list[dict[str, Any]]
    final_message: str | None
    fs_diff: str
    judge: Judge | None = None
    judge_votes: int = 1  # validation asks judges several times and takes the majority


class GraderError(Exception):
    """A grader crashed or timed out (WF-05 E1).

    A grader verified on the reference solution that fails this way on an agent's final state
    counts against the agent; otherwise it is an infra error.
    """


class JudgeError(GraderError):
    """The judge returned no usable verdict. Never attributed to the agent (WF-05 E2)."""


class Grader(Protocol):
    name: str
    kind: str

    async def grade(self, context: GradingContext) -> GradeResult: ...


def aggregate(
    specs: Sequence[GraderSpecBase], results: Sequence[GradeResult]
) -> tuple[bool, float]:
    """Pass only if every required grader passes; the score is the weighted mean."""
    pairs = list(zip(specs, results, strict=True))
    passed = all(result.passed for spec, result in pairs if spec.required)
    total_weight = sum(spec.weight for spec, _ in pairs)
    if total_weight == 0:
        return passed, 1.0 if passed else 0.0
    return passed, sum(spec.weight * result.score for spec, result in pairs) / total_weight


def check_result(
    name: str, kind: str, passed: bool, rationale: str, **metadata: Any
) -> GradeResult:
    """A pass/fail result scored 1.0 or 0.0."""
    return GradeResult(name, 1.0 if passed else 0.0, passed, rationale, metadata, kind)
