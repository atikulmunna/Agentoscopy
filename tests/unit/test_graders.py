import asyncio

import pytest

from agentoscopy.graders.base import GraderError, GradeResult, GradingContext, aggregate
from agentoscopy.graders.command import CommandGrader
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import CommandGraderSpec
from agentoscopy.testing.fake_sandbox import FakeSandbox, exit_with


def spec(name="tests", required=True, weight=1.0):
    return CommandGraderSpec(
        name=name, type="command", run="run-tests", required=required, weight=weight
    )


def grade_with(make_task, handler):
    context = GradingContext(make_task(), FakeSandbox(handler=handler), [], None, "")
    return asyncio.run(CommandGrader(spec()).grade(context))


def result(name, passed, score=None):
    return GradeResult(name, (1.0 if passed else 0.0) if score is None else score, passed, "")


def test_command_grader_passes_on_exit_zero(make_task):
    grade = grade_with(make_task, exit_with(0, b"OK"))

    assert (grade.passed, grade.score) == (True, 1.0)
    assert grade.rationale.startswith("exit code 0")
    assert "OK" in grade.rationale


def test_command_grader_fails_on_nonzero_exit(make_task):
    grade = grade_with(make_task, exit_with(1, stderr=b"AssertionError"))

    assert (grade.passed, grade.score) == (False, 0.0)
    assert grade.metadata == {"exit_code": 1}
    assert "AssertionError" in grade.rationale


def test_command_grader_timeout_is_a_grader_error(make_task):
    with pytest.raises(GraderError, match="timed out"):
        grade_with(make_task, lambda cmd: ExecResult(124, b"", b"", timed_out=True))


def test_aggregate_requires_every_required_grader():
    specs = [spec("a"), spec("b")]

    assert aggregate(specs, [result("a", True), result("b", True)]) == (True, 1.0)
    assert aggregate(specs, [result("a", True), result("b", False)]) == (False, 0.5)


def test_optional_grader_affects_score_but_not_outcome():
    specs = [spec("required", weight=0.75), spec("optional", required=False, weight=0.25)]

    passed, score = aggregate(specs, [result("required", True), result("optional", False)])

    assert passed is True
    assert score == pytest.approx(0.75)


def test_zero_total_weight_scores_by_outcome():
    specs = [spec("a", weight=0.0)]

    assert aggregate(specs, [result("a", True)]) == (True, 1.0)
    assert aggregate(specs, [result("a", False)]) == (False, 0.0)


def test_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        aggregate([spec("a"), spec("b")], [result("a", True)])
