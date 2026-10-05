"""`agentoscopy agent check` (WF-02) with fake sandboxes."""

import pytest
import yaml

from agentoscopy.adapters.base import AgentResult
from agentoscopy.cli.main import main
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.spec import config_hash, load_agent_config
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend

REVERSED = b"tset ekoms ypocsotnegA"
SOLVES = [
    {"model_calls": 1},
    {"read": "input.txt"},
    {"write": "output.txt", "content": REVERSED.decode()},
]


class SilentAgent:
    """Calls the model, then does nothing the harness can see."""

    name = "silent"

    async def setup(self, config):
        pass

    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        from anthropic import AsyncAnthropic

        async with AsyncAnthropic(
            base_url=model_endpoint.base_url, api_key=model_endpoint.token
        ) as client:
            await client.messages.create(
                model="mock", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
            )
        return AgentResult()

    async def teardown(self):
        pass


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = FakeBackend(
        files={"input.txt": b"Agentoscopy smoke test"},
        grader=lambda cmd, files: ExecResult(
            0 if files.get("output.txt") == REVERSED else 1, b"", b""
        ),
    )
    monkeypatch.setattr("agentoscopy.cli.agent.DockerBackend", lambda: fake)
    return fake


def config(tmp_path, script=None, entrypoint="agentoscopy.testing.scripted_agent:ScriptedAgent"):
    path = tmp_path / "agent.yaml"
    params = {"script": script} if script is not None else {}
    path.write_text(
        yaml.safe_dump(
            {"name": "candidate", "adapter": "python", "entrypoint": entrypoint, "params": params}
        )
    )
    return path


def check(tmp_path, path):
    return main(["agent", "check", str(path), "--home", str(tmp_path / "home")])


def test_a_working_agent_passes_and_its_config_is_recorded(tmp_path, backend, capsys):
    path = config(tmp_path, SOLVES)

    assert check(tmp_path, path) == 0

    out = capsys.readouterr().out
    assert "model calls through the gateway: 1" in out
    assert "smoke task: pass" in out and "ok: the config is ready to run" in out
    store = Store(tmp_path / "home" / "agentoscopy.db")
    try:
        assert store.config_spec(config_hash(load_agent_config(path)))["name"] == "candidate"
    finally:
        store.close()
    assert all(box.destroyed for box in backend.agent_sandboxes + backend.grading_sandboxes)


def test_an_agent_that_bypasses_the_gateway_fails(tmp_path, backend, capsys):
    assert check(tmp_path, config(tmp_path, SOLVES[1:])) == 1

    out = capsys.readouterr().out
    assert "FAIL GATEWAY_BYPASS" in out
    assert "smoke task: pass" in out  # solving the task is not enough


def test_a_crashing_agent_fails_with_its_error(tmp_path, backend, capsys):
    script = [{"model_calls": 1}, {"read": "no-such-file.txt"}]

    assert check(tmp_path, config(tmp_path, script)) == 1

    out = capsys.readouterr().out
    assert "FAIL AGENT_DID_NOT_FINISH: the agent stopped with termination agent_error" in out
    assert "the agent raised: SandboxError" in out


def test_an_agent_that_leaves_no_trace_fails(tmp_path, backend, capsys):
    assert check(tmp_path, config(tmp_path, entrypoint="test_agent_check:SilentAgent")) == 1

    assert "FAIL NO_AGENT_EVENTS" in capsys.readouterr().out


def test_broken_configs_and_environments(tmp_path, backend, monkeypatch, capsys):
    assert check(tmp_path, tmp_path / "missing.yaml") == 2
    assert check(tmp_path, config(tmp_path, entrypoint="no_such_pkg.mod:Agent")) == 2

    monkeypatch.setattr(
        "agentoscopy.cli.agent.DockerBackend",
        lambda: FakeBackend(image_error="docker is not running"),
    )
    assert check(tmp_path, config(tmp_path, SOLVES)) == 3
    assert "FAIL SANDBOX_ERROR: docker is not running" in capsys.readouterr().out
