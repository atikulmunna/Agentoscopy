import asyncio

from conftest import BASE_TASK, SetupBackend

from agentoscopy.graders.base import JudgeVerdictResult
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.testing.fake_sandbox import FakeBackend, exit_with
from agentoscopy.validation import validate_task

REFERENCE = {"reference/app.py": "fixed"}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


def validate(task, backend):
    return asyncio.run(validate_task(task, backend))


def test_sound_task_with_reference_validates(make_task):
    backend = FakeBackend(grader=passes_when_fixed)

    report = validate(make_task(files=REFERENCE), backend)

    assert report.ok, report.message
    assert report.verified_graders == ["tests"]
    assert report.image_digest == "fake-image"
    assert report.flags == []
    assert all(box.destroyed for box in backend.agent_sandboxes + backend.grading_sandboxes)


def test_task_without_reference_validates_unverified(make_task):
    report = validate(make_task(), FakeBackend(grader=passes_when_fixed))

    assert report.ok
    assert report.flags == ["NO_REFERENCE"]
    assert report.verified_graders == []


def test_grader_that_passes_untouched_environment_is_rejected(make_task):
    report = validate(make_task(files=REFERENCE), FakeBackend())  # graders always pass

    assert report.error_code == "GRADER_PASSES_EMPTY_ENV"


def test_reference_that_fails_is_rejected(make_task):
    always_fail = lambda cmd, files: ExecResult(1, b"", b"")  # noqa: E731

    report = validate(make_task(files=REFERENCE), FakeBackend(grader=always_fail))

    assert report.error_code == "REFERENCE_FAILS"
    assert "tests" in report.message


def test_hidden_files_copied_into_fixtures_are_exposed(make_task):
    files = {
        **REFERENCE,
        "hidden/test_app.py": "secret test",
        "fixtures/test_app.py": "secret test",
    }

    report = validate(make_task(files=files), FakeBackend(grader=passes_when_fixed))

    assert report.error_code == "HIDDEN_FILE_EXPOSED"


def test_hidden_mount_visible_to_the_agent_is_exposed(make_task):
    report = validate(make_task(files=REFERENCE), FakeBackend(agent_handler=exit_with(0)))

    assert report.error_code == "HIDDEN_FILE_EXPOSED"


def test_broken_setup_is_reported(make_task):
    task = make_task(environment={"base_image": "img@sha256:abc", "setup": ["fetch-data"]})
    backend = FakeBackend(agent_handler=exit_with(1, stderr=b"network down"))

    report = validate(task, backend)

    assert report.error_code == "SETUP_BROKEN"
    assert "network down" in report.message


def test_grader_that_times_out_is_unstable(make_task):
    hangs = lambda cmd, files: ExecResult(124, b"", b"", timed_out=True)  # noqa: E731

    report = validate(make_task(files=REFERENCE), FakeBackend(grader=hangs))

    assert report.error_code == "GRADER_UNSTABLE"


JUDGE = {"name": "quality", "type": "llm_judge", "rubric": "Pass if minimal."}


class FakeJudge:
    """Fails the untouched environment and passes the reference, unless told otherwise."""

    model = "fake-judge"

    def __init__(self, passes=None):
        self.calls = 0
        self._passes = passes

    async def verdict(self, spec, prompt):
        self.calls += 1
        passed = self._passes if self._passes is not None else "fixed" in prompt
        return JudgeVerdictResult(passed, float(passed), 0.9, "judged")


def test_a_task_with_judges_needs_a_judge(make_task):
    task = make_task(files=REFERENCE, graders=[*BASE_TASK["graders"], {**JUDGE, "required": False}])

    report = validate(task, FakeBackend(grader=passes_when_fixed))

    assert report.error_code == "JUDGE_UNAVAILABLE"


def test_judges_vote_three_times_in_each_check(make_task):
    task = make_task(files=REFERENCE, graders=[*BASE_TASK["graders"], {**JUDGE, "required": False}])
    judge = FakeJudge()

    report = asyncio.run(validate_task(task, FakeBackend(grader=passes_when_fixed), judge))

    assert report.ok, report.message
    assert judge.calls == 6  # null check and reference check
    assert sorted(report.verified_graders) == ["quality", "tests"]
    assert report.flags == []


def test_a_task_graded_only_by_judges_is_flagged_soft(make_task):
    task = make_task(files=REFERENCE, graders=[JUDGE])

    report = asyncio.run(validate_task(task, FakeBackend(), FakeJudge()))

    assert report.ok, report.message
    assert report.flags == ["SOFT_GRADER_ONLY"]


def test_a_judge_that_passes_the_untouched_environment_is_rejected(make_task):
    task = make_task(files=REFERENCE, graders=[JUDGE])

    report = asyncio.run(validate_task(task, FakeBackend(), FakeJudge(passes=True)))

    assert report.error_code == "GRADER_PASSES_EMPTY_ENV"


def test_setup_changes_do_not_fail_the_tamper_check(make_task):
    tamper = {"name": "no_tamper", "type": "tamper_check", "weight": 0}
    environment = {**BASE_TASK["environment"], "setup": ["install-deps"]}
    task = make_task(
        files=REFERENCE, environment=environment, graders=[*BASE_TASK["graders"], tamper]
    )

    report = validate(task, SetupBackend(grader=passes_when_fixed))

    assert report.ok, report.message
    assert sorted(report.verified_graders) == ["no_tamper", "tests"]
