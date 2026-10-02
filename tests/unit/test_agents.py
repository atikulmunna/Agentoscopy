import asyncio

import pytest
from conftest import running_gateway, running_provider
from pydantic import ValidationError

from agentoscopy.adapters.base import ModelEndpoint, TaskContext
from agentoscopy.adapters.examples.claude_agent import ClaudeAgent
from agentoscopy.adapters.python import AdapterLoadError, load_python_adapter
from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.spec import AgentConfig, Budget
from agentoscopy.testing.fake_sandbox import FakeSandbox
from agentoscopy.testing.scripted_agent import ScriptedAgent

CONTEXT = TaskContext("demo-task", "Fix it.", "/workspace")
UNUSED_ENDPOINT = ModelEndpoint("http://127.0.0.1:9", "unused")  # for agents making no calls


class ListRecorder:
    def __init__(self):
        self.events = []

    def event(self, event_type, **payload):
        self.events.append((event_type, payload))


def config(params, entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent"):
    return AgentConfig(name="agent", adapter="python", entrypoint=entrypoint, params=params)


def run_script(params, sandbox):
    agent = ScriptedAgent()
    recorder = ListRecorder()

    async def scenario():
        await agent.setup(config(params))
        return await agent.run(CONTEXT, sandbox, UNUSED_ENDPOINT, Budget(), recorder)

    return asyncio.run(scenario()), recorder


def test_scripted_agent_performs_actions_in_order():
    sandbox = FakeSandbox(files={"app.py": b"old"})
    params = {
        "final_message": "fixed",
        "script": [
            {"read": "app.py"},
            {"write": "app.py", "content": "new"},
            {"exec": "make test", "timeout_s": 5},
        ],
    }

    result, recorder = run_script(params, sandbox)

    assert sandbox.files["app.py"] == b"new"
    assert sandbox.commands == ["make test"]
    assert result.final_message == "fixed"
    assert recorder.events == [("agent_message", {"text": "fixed"})]


def test_scripted_agent_rejects_unknown_actions():
    with pytest.raises(ValidationError):
        run_script({"script": [{"delete": "app.py"}]}, FakeSandbox())


def test_scripted_agent_requires_setup():
    agent = ScriptedAgent()

    with pytest.raises(RuntimeError, match="setup"):
        asyncio.run(agent.run(CONTEXT, FakeSandbox(), UNUSED_ENDPOINT, Budget(), ListRecorder()))


def test_python_adapter_loads_entrypoint():
    adapter = load_python_adapter("agentoscopy.testing.scripted_agent:ScriptedAgent")

    assert isinstance(adapter, ScriptedAgent)


@pytest.mark.parametrize(
    ("entrypoint", "message"),
    [
        ("agentoscopy.no_such_module:Agent", "cannot import"),
        ("agentoscopy.testing.scripted_agent:NoSuchAgent", "has no attribute"),
        ("agentoscopy.testing.scripted_agent:ScriptParams", "cannot instantiate"),
    ],
)
def test_python_adapter_reports_bad_entrypoints(entrypoint, message):
    with pytest.raises(AdapterLoadError, match=message):
        load_python_adapter(entrypoint)


def assistant(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def run_claude_agent(recorder_file, provider_replies, budget):
    """ClaudeAgent -> gateway -> fake provider, with a fake sandbox for its tools."""
    sandbox = FakeSandbox()
    agent_recorder = ListRecorder()

    async def scenario():
        async with (
            running_provider() as provider,
            running_gateway(upstream_url=provider.base_url, upstream_secret="s") as gateway,
        ):
            for reply in provider_replies:
                provider.reply_message(reply)
            trajectory = TrajectoryRecorder(recorder_file, recorder_file.parent / "a", "t", 1)
            session = gateway.open_session("t", budget, trajectory, "seed")
            agent = ClaudeAgent()
            await agent.setup(config({}, "agentoscopy.adapters.examples.claude_agent:ClaudeAgent"))
            try:
                result = await agent.run(CONTEXT, sandbox, session.endpoint, budget, agent_recorder)
            finally:
                trajectory.close()
            return result, provider

    result, provider = asyncio.run(scenario())
    return result, provider, sandbox, agent_recorder


def test_claude_agent_runs_tools_until_done(tmp_path):
    tool_call = {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}
    replies = [
        assistant([tool_call], "tool_use"),
        assistant([{"type": "text", "text": "All done."}], "end_turn"),
    ]

    result, provider, sandbox, recorder = run_claude_agent(tmp_path / "t.jsonl", replies, Budget())

    assert sandbox.commands == ["ls"]
    assert result.final_message == "All done."
    first, second = (request["body"] for request in provider.requests)
    assert first["model"] == "claude-opus-5-5"
    assert first["output_config"] == {"effort": "high"}
    assert "/workspace" in str(first["system"])
    tool_result = second["messages"][-1]["content"][0]
    assert (tool_result["type"], tool_result["tool_use_id"]) == ("tool_result", "toolu_1")
    assert "exit code 0" in str(tool_result["content"])
    assert recorder.events == [("agent_message", {"text": "All done."})]


def test_claude_agent_stops_cleanly_when_the_budget_is_spent(tmp_path):
    tool_call = {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}
    replies = [assistant([tool_call], "tool_use")]

    result, provider, sandbox, recorder = run_claude_agent(
        tmp_path / "t.jsonl", replies, Budget(max_steps=1)
    )

    assert result.final_message is None
    assert len(provider.requests) == 1  # the second call never left the gateway
    assert recorder.events[0][0] == "agent_message"
    assert "budget" in recorder.events[0][1]["text"]
