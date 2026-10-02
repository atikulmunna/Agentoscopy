import asyncio

import pytest
from conftest import BASE_TASK, SetupBackend, running_gateway

from agentoscopy.adapters.base import AgentResult
from agentoscopy.graders.pipeline import CACHE_CLEANUP
from agentoscopy.recorder.trajectory import read_events
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import AgentConfig
from agentoscopy.testing.fake_sandbox import FakeBackend, exit_with
from agentoscopy.testing.scripted_agent import ScriptedAgent
from agentoscopy.worker.trial import AttemptSpec, _with_deadline, run_attempt

FIX = {"write": "app.py", "content": "fixed"}


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


class SleepingAgent:
    name = "sleeper"

    async def setup(self, config):
        pass

    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        await asyncio.sleep(budget.timeout_s + 5)
        return AgentResult()

    async def teardown(self):
        pass


class CrashingAgent(SleepingAgent):
    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        await sandbox.write_file("app.py", b"fixed")
        raise RuntimeError("boom")

    async def teardown(self):
        raise OSError("teardown broke")


def attempt(tmp_path, task, backend, script=None, adapter=None, **spec_fields):
    config = AgentConfig(
        name="test-agent",
        adapter="python",
        entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
        params={"script": script if script is not None else [FIX]},
    )
    spec = AttemptSpec(
        run_id="run",
        trial_id="trial",
        attempt=1,
        task=task,
        config=config,
        seed_key="s",
        **spec_fields,
    )

    async def scenario():
        async with running_gateway() as gateway:
            return await run_attempt(
                spec, adapter or ScriptedAgent(), backend, gateway, tmp_path / "home"
            )

    return asyncio.run(scenario())


def event_types(result):
    return [event["type"] for event in read_events(result.trajectory_path)]


def test_passing_attempt_records_everything_and_cleans_up(make_task, tmp_path):
    backend = FakeBackend(grader=passes_when_fixed)
    script = [{"model_calls": 2}, FIX]

    result = attempt(tmp_path, make_task(), backend, script)

    assert (result.outcome, result.termination, result.score) == ("pass", "agent_done", 1.0)
    assert result.usage.steps == 2 and result.usage.cost_usd > 0
    (agent_box,) = backend.agent_sandboxes
    (grading_box,) = backend.grading_sandboxes
    assert agent_box.stopped and agent_box.destroyed and grading_box.destroyed
    # Graders never run in the agent's sandbox; caches are cleared before they run.
    assert grading_box.commands == [CACHE_CLEANUP.format(workdir="/workspace"), "run-tests"]
    assert event_types(result) == [
        "trial_start",
        "model_request",
        "model_response",
        "model_request",
        "model_response",
        "tool_call",
        "tool_result",
        "agent_message",
        "trial_end",
    ]
    tool_call = read_events(result.trajectory_path)[5]
    assert tool_call["step"] == 2  # tool events belong to the model call before them
    assert result.final_state_path.read_text() == "A /workspace/app.py"
    assert "attempt-1.jsonl" in str(result.trajectory_path)


def test_failing_grader_fails_the_attempt(make_task, tmp_path):
    result = attempt(tmp_path, make_task(), FakeBackend(grader=passes_when_fixed), script=[])

    assert (result.outcome, result.score) == ("fail", 0.0)


def test_spent_budget_is_the_termination_reason_and_is_still_graded(make_task, tmp_path):
    task = make_task(budget={"max_steps": 1, "timeout_s": 5})
    backend = FakeBackend(grader=passes_when_fixed)

    result = attempt(tmp_path, task, backend, [{"model_calls": 3}, FIX])

    assert result.termination == "budget_steps"
    assert result.usage.steps == 1
    assert result.outcome == "fail"  # the agent stopped before writing the fix
    assert "budget_warning" in event_types(result)


def test_timeout_is_an_agent_outcome_and_still_graded(make_task, tmp_path):
    backend = FakeBackend()

    result = attempt(tmp_path, make_task(budget={"timeout_s": 1}), backend, adapter=SleepingAgent())

    assert (result.outcome, result.termination) == ("pass", "timeout")
    assert backend.grading_sandboxes[0].commands[-1] == "run-tests"


def test_agent_crash_is_graded_and_recorded(make_task, tmp_path):
    backend = FakeBackend(grader=passes_when_fixed)

    result = attempt(tmp_path, make_task(), backend, adapter=CrashingAgent())

    # The crash counts against the agent, but its final state still passes the graders.
    assert (result.outcome, result.termination) == ("pass", "agent_error")
    errors = [
        e["payload"]["message"] for e in read_events(result.trajectory_path) if e["type"] == "error"
    ]
    assert errors == ["RuntimeError: boom", "teardown failed: OSError: teardown broke"]


@pytest.mark.parametrize(
    ("backend_args", "spec_fields", "error_code", "retryable"),
    [
        ({"agent_handler": exit_with(2, stderr=b"no data")}, {}, "SETUP_BROKEN", False),
        ({"image_error": "daemon unreachable"}, {}, "SANDBOX_ERROR", True),
        ({"create_failures": 1}, {}, "SANDBOX_ERROR", True),
        ({}, {"image_digest": "sha256:pinned"}, "TASK_IMAGE_CHANGED", False),
    ],
)
def test_infra_errors_are_classified(
    make_task, tmp_path, backend_args, spec_fields, error_code, retryable
):
    task = make_task(environment={"base_image": "img@sha256:abc", "setup": ["prepare-data"]})
    backend = FakeBackend(**backend_args)

    result = attempt(tmp_path, task, backend, **spec_fields)

    assert (result.outcome, result.error_code, result.retryable) == (
        "infra_error",
        error_code,
        retryable,
    )
    assert all(box.destroyed for box in backend.agent_sandboxes)
    assert event_types(result)[-1] == "trial_end"


def test_unverified_grader_timeout_is_an_infra_error(make_task, tmp_path):
    timing_out = lambda cmd, files: ExecResult(124, b"", b"", timed_out=True)  # noqa: E731

    result = attempt(tmp_path, make_task(), FakeBackend(grader=timing_out))

    assert (result.outcome, result.error_code) == ("infra_error", "GRADER_UNSTABLE")


def test_verified_grader_timeout_fails_the_agent(make_task, tmp_path):
    timing_out = lambda cmd, files: ExecResult(124, b"", b"", timed_out=True)  # noqa: E731
    backend = FakeBackend(grader=timing_out)

    result = attempt(tmp_path, make_task(), backend, verified_graders=frozenset({"tests"}))

    assert (result.outcome, result.error_code) == ("fail", None)
    assert result.grades[0].metadata == {"grader_error": True}


def test_provider_failure_is_a_retryable_infra_error(make_task, tmp_path):
    config = AgentConfig(
        name="real-model",
        adapter="python",
        entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent",
        params={"script": [{"model_calls": 1}], "model": "claude-opus-5-5"},
    )
    spec = AttemptSpec(
        run_id="r", trial_id="t", attempt=1, task=make_task(), config=config, seed_key="s"
    )

    async def scenario():
        async with running_gateway() as gateway:  # no upstream: no provider credentials
            return await run_attempt(spec, ScriptedAgent(), FakeBackend(), gateway, tmp_path)

    result = asyncio.run(scenario())

    assert (result.outcome, result.error_code, result.retryable) == (
        "infra_error",
        "PROVIDER_ERROR",
        True,
    )


def test_deadline_extension_keeps_a_slow_call_alive():
    extension = {"seconds": 0.0}

    async def slow_call():
        extension["seconds"] = 0.3  # e.g. the gateway waited out provider backoff
        await asyncio.sleep(0.25)
        return "finished"

    async def scenario():
        return await _with_deadline(slow_call(), 0.1, lambda: extension["seconds"])

    assert asyncio.run(scenario()) == "finished"


def test_deadline_still_cancels_a_stuck_agent():
    async def stuck():
        await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        asyncio.run(_with_deadline(stuck(), 0.05, lambda: 0.0))


JUDGE = {"name": "quality", "type": "llm_judge", "rubric": "Pass if minimal.", "required": False}
TAMPER = {"name": "no_tamper", "type": "tamper_check", "weight": 0}


def with_graders(make_task, *extra, **overrides):
    return make_task(graders=[*BASE_TASK["graders"], *extra], **overrides)


def test_judge_spend_is_tracked_apart_from_agent_spend(make_task, tmp_path):
    task = with_graders(make_task, JUDGE)

    result = attempt(tmp_path, task, FakeBackend(grader=passes_when_fixed), judge_model="mock")

    judge = next(grade for grade in result.grades if grade.kind == "judge")
    assert result.outcome == "pass" and judge.metadata["judge_model"] == "mock"
    assert result.usage.cost_usd == 0 and result.judge_cost_usd > 0
    (grading_log,) = (tmp_path / "home").rglob("grading.jsonl")
    assert "model_request" in [event["type"] for event in read_events(grading_log)]
    end = read_events(result.trajectory_path)[-1]["payload"]
    assert end["totals"]["judge_cost_usd"] == pytest.approx(result.judge_cost_usd)


def test_a_judge_without_a_judge_model_is_never_the_agents_failure(make_task, tmp_path):
    task = with_graders(make_task, JUDGE)

    result = attempt(tmp_path, task, FakeBackend(grader=passes_when_fixed))

    assert (result.outcome, result.error_code) == ("infra_error", "GRADER_UNSTABLE")


def test_tampering_vetoes_an_attempt_whose_tests_pass(make_task, tmp_path):
    task = with_graders(make_task, {**TAMPER, "protected_paths": ["/workspace/tests/**"]})
    script = [FIX, {"write": "tests/test_app.py", "content": "assert True"}]

    result = attempt(tmp_path, task, FakeBackend(grader=passes_when_fixed), script)

    assert result.grades[0].passed  # the tests pass
    assert (result.outcome, result.score, result.vetoed) == ("fail", 0.0, True)
    assert read_events(result.trajectory_path)[-1]["payload"]["vetoed"] is True


def test_changes_made_by_setup_are_not_blamed_on_the_agent(make_task, tmp_path):
    environment = {**BASE_TASK["environment"], "setup": ["install-deps"]}
    task = with_graders(make_task, TAMPER, environment=environment)

    result = attempt(tmp_path, task, SetupBackend(grader=passes_when_fixed))

    assert (result.outcome, result.vetoed) == ("pass", False)
    assert result.final_state_path.read_text() == "A /workspace/app.py"
