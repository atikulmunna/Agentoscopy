"""CommandAgent: a command-line agent run inside the sandbox, reaching models via the relay."""

import asyncio
import shlex

import pytest
from conftest import running_gateway
from pydantic import ValidationError

from agentoscopy.adapters.base import ModelEndpoint, TaskContext
from agentoscopy.adapters.command import PLACEHOLDER_KEY, CommandAgent, CommandAgentFailed
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import AgentConfig, Budget
from agentoscopy.testing.fake_sandbox import FakeBackend, FakeSandbox
from agentoscopy.worker.trial import AttemptSpec, run_attempt

INSTALL = ["npm install -g some-agent"]
INSTRUCTIONS = 'Fix "add"; it subtracts.'


def config(**params):
    return AgentConfig(
        name="cli",
        adapter="python",
        entrypoint="agentoscopy.adapters.command:CommandAgent",
        params={"command": ["agent", "-p", "{instructions}"], **params},
    )


def run_agent(answer, **params):
    """Run the agent in a fake sandbox whose exec gives `answer`; returns the result and the
    command line it ran, if it got that far."""
    sandbox = FakeSandbox(lambda cmd: answer)
    agent = CommandAgent()
    endpoint = ModelEndpoint("http://127.0.0.1:9", "trial-token")
    task = TaskContext("task-a", INSTRUCTIONS, "/workspace")

    async def main():
        await agent.setup(config(**params))
        return await agent.run(task, sandbox, endpoint, Budget(timeout_s=30), None)

    try:
        result = asyncio.run(main())
    except Exception as exc:  # the test inspects what the agent raised
        result = exc
    return result, sandbox.commands[-1] if sandbox.commands else None


def test_the_command_runs_with_the_relay_and_the_instructions():
    result, line = run_agent(ExecResult(0, b"all done\n", b""), env={"MODE": "fast"})

    words = shlex.split(line)
    assert words[-3:] == ["agent", "-p", INSTRUCTIONS]  # quoted whole, quotes and all
    env = dict(word.split("=", 1) for word in words[1:-3])
    assert env["MODE"] == "fast" and env["ANTHROPIC_API_KEY"] == PLACEHOLDER_KEY
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")
    assert "trial-token" not in line  # the credential never enters the sandbox
    assert result.final_message == "all done"


def test_a_failing_command_is_an_agent_error():
    result, _ = run_agent(ExecResult(2, b"", b"boom"))

    assert isinstance(result, CommandAgentFailed) and str(result) == "exit code 2: boom"


def test_a_command_that_runs_out_of_time_is_a_timeout():
    result, _ = run_agent(ExecResult(124, b"", b"", timed_out=True))

    assert isinstance(result, TimeoutError)


def test_unknown_params_are_refused():
    result, _ = run_agent(ExecResult(0, b"", b""), commnd=["typo"])

    assert isinstance(result, ValidationError)


@pytest.mark.parametrize("install", [INSTALL, []])
def test_the_worker_installs_the_agent_on_top_of_the_task_image(make_task, tmp_path, install):
    backend = FakeBackend()
    spec = AttemptSpec(
        run_id="r", trial_id="t", attempt=1, task=make_task(), config=config(install=install),
        seed_key="s",
    )  # fmt: skip

    async def main():
        async with running_gateway() as gateway:
            return await run_attempt(spec, CommandAgent(), backend, gateway, tmp_path)

    result = asyncio.run(main())

    assert result.termination == "agent_done"
    if install:
        assert backend.extensions == [("fake-image", INSTALL)]
        assert backend.created_from == ["fake-image+agent1"]
    else:
        assert backend.extensions == [] and backend.created_from == ["fake-image"]
