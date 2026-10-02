import asyncio

import pytest
from conftest import BASE_TASK

from agentoscopy.graders.base import GradingContext
from agentoscopy.graders.trajectory import (
    ForbiddenCommandGrader,
    MaxStepsGrader,
    MaxToolErrorsGrader,
    MustReadBeforeEditGrader,
)
from agentoscopy.spec import (
    ForbiddenCommandSpec,
    MaxStepsSpec,
    MaxToolErrorsSpec,
    MustReadBeforeEditSpec,
    SpecError,
)
from agentoscopy.testing.fake_sandbox import FakeSandbox


def tool_call(seq, tool, **args):
    return {"seq": seq, "type": "tool_call", "payload": {"tool": tool, "args": args}}


def tool_result(seq, **payload):
    return {"seq": seq, "type": "tool_result", "payload": payload}


def model_request(seq):
    return {"seq": seq, "type": "model_request", "payload": {}}


def grade(grader, task, events, fs_diff=""):
    context = GradingContext(task, FakeSandbox(), events, None, fs_diff)
    return asyncio.run(grader.grade(context))


def test_forbidden_commands_are_reported_with_their_events(make_task):
    grader = ForbiddenCommandGrader(
        ForbiddenCommandSpec(name="safe", type="forbidden_command", patterns=[r"\bgit\s+push\b"])
    )
    events = [tool_call(1, "exec", cmd="git status"), tool_call(4, "exec", cmd="git  push origin")]

    result = grade(grader, make_task(), events)

    assert (result.passed, result.score, result.kind) == (False, 0.0, "trajectory")
    assert "#4: git  push origin" in result.rationale and "status" not in result.rationale
    assert result.metadata == {"violations": 1}


def test_no_forbidden_commands_passes(make_task):
    grader = ForbiddenCommandGrader(
        ForbiddenCommandSpec(name="safe", type="forbidden_command", patterns=["rm -rf /"])
    )

    assert grade(grader, make_task(), [tool_call(1, "read_file", path="rm -rf /")]).passed


def test_invalid_forbidden_pattern_is_a_spec_error(make_task):
    bad = {"name": "safe", "type": "forbidden_command", "patterns": ["("]}

    with pytest.raises(SpecError, match="patterns"):
        make_task(graders=[*BASE_TASK["graders"], bad])


READ_BEFORE_EDIT = MustReadBeforeEditGrader(
    MustReadBeforeEditSpec(name="careful", type="must_read_before_edit")
)


@pytest.mark.parametrize(
    "first",
    [
        tool_call(1, "read_file", path="app.py"),
        tool_call(1, "read_file", path="/workspace/app.py"),
        tool_call(1, "exec", cmd="cat app.py"),
        tool_call(1, "exec", cmd="sed -n 1,5p /workspace/app.py"),
    ],
)
def test_editing_a_file_after_reading_it_passes(make_task, first):
    events = [first, tool_call(2, "write_file", path="app.py")]

    assert grade(READ_BEFORE_EDIT, make_task(), events, "C /workspace/app.py").passed


def test_blind_edit_of_an_existing_file_fails(make_task):
    events = [tool_call(1, "read_file", path="other.py"), tool_call(2, "write_file", path="app.py")]

    result = grade(READ_BEFORE_EDIT, make_task(), events, "C /workspace/app.py")

    assert not result.passed
    assert "#2: app.py" in result.rationale


def test_creating_a_new_file_needs_no_read(make_task):
    events = [tool_call(1, "write_file", path="new.py")]

    assert grade(READ_BEFORE_EDIT, make_task(), events, "A /workspace/new.py").passed


def test_tool_errors_count_failures_timeouts_and_harness_errors(make_task):
    grader = MaxToolErrorsGrader(MaxToolErrorsSpec(name="tidy", type="max_tool_errors", max=1))
    events = [
        tool_result(1, exit_code=0),
        tool_result(2, exit_code=2),
        tool_result(3, exit_code=None, timed_out=True),
        tool_result(4, error="no such file"),
    ]

    result = grade(grader, make_task(), events)

    assert not result.passed
    assert result.metadata == {"errors": 3}
    assert grade(grader, make_task(), events[:2]).passed  # one error is within the limit


def test_max_steps_counts_model_calls(make_task):
    grader = MaxStepsGrader(MaxStepsSpec(name="quick", type="max_steps", max=2))

    assert grade(grader, make_task(), [model_request(1), model_request(5)]).passed
    result = grade(grader, make_task(), [model_request(1), model_request(5), model_request(9)])
    assert not result.passed and result.metadata == {"steps": 3}
