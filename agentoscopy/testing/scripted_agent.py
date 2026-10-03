"""Scripted test agent (FR-TST-02): replays a fixed list of actions from its config.

Sandbox actions run as written. `model_calls` actions call the model through the gateway with
the Anthropic SDK, which lets test runs exercise budgets and costs against the mock provider.
The well-behaved, broken, and malicious test agents are all configs for this one class.
"""

from __future__ import annotations

import asyncio

from anthropic import AsyncAnthropic, BadRequestError
from pydantic import BaseModel, ConfigDict, Field

from agentoscopy.adapters.base import (
    BUDGET_EXCEEDED_ERROR,
    AgentResult,
    Budget,
    ModelEndpoint,
    Recorder,
    TaskContext,
)
from agentoscopy.sandbox.base import Sandbox
from agentoscopy.spec import AgentConfig


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecAction(_Model):
    exec: str
    timeout_s: int = 60


class WriteAction(_Model):
    write: str
    content: str


class ReadAction(_Model):
    read: str


class SleepAction(_Model):
    """Keeps the trial in flight, e.g. so tests can cancel or crash a run mid-trial."""

    sleep: float = Field(ge=0)


class ModelCallsAction(_Model):
    model_calls: int = Field(gt=0)
    prompt: str = "Continue with the task."
    max_tokens: int = Field(default=1024, gt=0)
    stream: bool = False


Action = ExecAction | WriteAction | ReadAction | ModelCallsAction | SleepAction


class ScriptParams(_Model):
    script: list[Action]
    model: str = "mock"
    final_message: str = "done"


class ScriptedAgent:
    name = "scripted"

    def __init__(self) -> None:
        self._params: ScriptParams | None = None

    async def setup(self, config: AgentConfig) -> None:
        self._params = ScriptParams.model_validate(config.params)

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
        client = AsyncAnthropic(base_url=model_endpoint.base_url, api_key=model_endpoint.token)
        try:
            for action in self._params.script:
                if isinstance(action, SleepAction):
                    await asyncio.sleep(action.sleep)
                elif isinstance(action, ModelCallsAction):
                    if not await _call_model(client, self._params.model, action):
                        break  # budget spent: stop, as a real agent would
                else:
                    await _perform(action, sandbox)
        finally:
            await client.close()
        recorder.event("agent_message", text=self._params.final_message)
        return AgentResult(final_message=self._params.final_message)

    async def teardown(self) -> None:
        self._params = None


async def _call_model(client: AsyncAnthropic, model: str, action: ModelCallsAction) -> bool:
    """Make the calls; returns False once the gateway reports the trial budget is spent."""
    request = {
        "model": model,
        "max_tokens": action.max_tokens,
        "messages": [{"role": "user", "content": action.prompt}],
    }
    for _ in range(action.model_calls):
        try:
            if action.stream:
                async with client.messages.stream(**request) as stream:
                    await stream.get_final_message()
            else:
                await client.messages.create(**request)
        except BadRequestError as exc:
            if exc.type != BUDGET_EXCEEDED_ERROR:
                raise
            return False
    return True


async def _perform(action: ExecAction | WriteAction | ReadAction, sandbox: Sandbox) -> None:
    if isinstance(action, ExecAction):
        await sandbox.exec(action.exec, timeout_s=action.timeout_s)
    elif isinstance(action, WriteAction):
        await sandbox.write_file(action.write, action.content.encode())
    else:
        await sandbox.read_file(action.read)
