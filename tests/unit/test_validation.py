import asyncio

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
