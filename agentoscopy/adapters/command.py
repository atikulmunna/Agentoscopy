"""Run a command-line agent, such as Claude Code, inside the sandbox.

The agent's program is installed on top of the task image (`install`), and its `command`
runs in the sandbox's workdir with ANTHROPIC_BASE_URL pointing at the model-call relay on
the sandbox's own loopback address (agentoscopy.relay). The sandbox stays offline and the
agent holds no credential: the harness adds the trial's token as each call leaves for the
gateway, which enforces the trial's budget as for any agent.
"""

from __future__ import annotations

import shlex

from pydantic import BaseModel, ConfigDict, Field

from agentoscopy.adapters.base import AgentResult, Budget, ModelEndpoint, Recorder, TaskContext
from agentoscopy.relay.bridge import RelayBridge
from agentoscopy.sandbox.base import Sandbox
from agentoscopy.spec import AgentConfig

INSTRUCTIONS = "{instructions}"  # replaced by the task's instructions in `command`
PLACEHOLDER_KEY = "agentoscopy-relay"  # SDKs want a key; the relay's calls get the real one
OUTPUT_TAIL_CHARS = 4000


class CommandAgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command: list[str] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    install: list[str] = Field(default_factory=list)  # run as root on top of the task image


class CommandAgentFailed(Exception):
    """The agent's command exited with an error."""


class CommandAgent:
    name = "command"

    def __init__(self) -> None:
        self._params: CommandAgentParams | None = None

    def image_steps(self, config: AgentConfig) -> list[str]:
        return CommandAgentParams.model_validate(config.params).install

    async def setup(self, config: AgentConfig) -> None:
        self._params = CommandAgentParams.model_validate(config.params)

    async def run(
        self,
        task: TaskContext,
        sandbox: Sandbox,
        model_endpoint: ModelEndpoint,
        budget: Budget,
        recorder: Recorder,
    ) -> AgentResult:
        if self._params is None:
            raise RuntimeError("setup() must run before run()")
        async with RelayBridge(sandbox, model_endpoint) as relay:
            env = {**self._params.env, "ANTHROPIC_BASE_URL": relay.url}
            env["ANTHROPIC_API_KEY"] = PLACEHOLDER_KEY
            line = command_line(self._params.command, env, task.instructions)
            result = await sandbox.exec(line, timeout_s=budget.timeout_s)
        if result.timed_out:
            raise TimeoutError
        output = result.stdout.decode("utf-8", "replace").strip()
        if result.exit_code != 0:
            detail = result.stderr.decode("utf-8", "replace").strip() or output
            raise CommandAgentFailed(f"exit code {result.exit_code}: {detail[-OUTPUT_TAIL_CHARS:]}")
        return AgentResult(final_message=output[-OUTPUT_TAIL_CHARS:] or None)

    async def teardown(self) -> None:
        self._params = None


def command_line(command: list[str], env: dict[str, str], instructions: str) -> str:
    """A shell line that runs `command` with `env`, every word quoted."""
    argv = [part.replace(INSTRUCTIONS, instructions) for part in command]
    return shlex.join(["env", *(f"{name}={value}" for name, value in env.items()), *argv])
