"""`command` grader: passes when a shell command exits 0 in the grading sandbox."""

from __future__ import annotations

from agentoscopy.graders.base import GraderError, GradeResult, GradingContext
from agentoscopy.spec import CommandGraderSpec

RATIONALE_TAIL_CHARS = 2000


class CommandGrader:
    kind = "deterministic"

    def __init__(self, spec: CommandGraderSpec) -> None:
        self.name = spec.name
        self._spec = spec

    async def grade(self, context: GradingContext) -> GradeResult:
        result = await context.sandbox.exec(self._spec.run, timeout_s=self._spec.timeout_s)
        if result.timed_out:
            raise GraderError(f"grader {self.name!r} timed out after {self._spec.timeout_s}s")
        passed = result.exit_code == 0
        output = (result.stdout + result.stderr).decode("utf-8", "replace").strip()
        rationale = f"exit code {result.exit_code}"
        if output:
            rationale += "\n" + output[-RATIONALE_TAIL_CHARS:]
        return GradeResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            rationale=rationale,
            metadata={"exit_code": result.exit_code},
        )
