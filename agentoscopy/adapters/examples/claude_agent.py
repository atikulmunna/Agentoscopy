"""Example agent: Claude with a shell tool and a file-writing tool, on the SDK's tool runner.

Every model call goes through the Agentoscopy gateway (`model_endpoint`), which enforces the
trial budget and records each call. Refusal fallbacks are deliberately not enabled: a fallback
serves a turn with a different model than the config names, mixing models inside one
evaluated config.
"""

from __future__ import annotations

from typing import Literal

from anthropic import AsyncAnthropic, BadRequestError, beta_async_tool
from pydantic import BaseModel, ConfigDict

from agentoscopy.adapters.base import (
    BUDGET_EXCEEDED_ERROR,
    AgentResult,
    Budget,
    ModelEndpoint,
    Recorder,
    TaskContext,
)
from agentoscopy.sandbox.base import Sandbox, SandboxError
from agentoscopy.spec import AgentConfig

DEFAULT_MODEL = "claude-opus-5-5"
OUTPUT_TAIL_CHARS = 20_000

SYSTEM_PROMPT = """\
You are an autonomous software engineer working in a repository at {workdir}.
Use the bash tool to inspect files and run commands, and the write_file tool to create or
replace files. Verify your change by running the relevant tests before you finish, then reply
with a short summary of what you changed."""


class ClaudeAgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = DEFAULT_MODEL
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "high"
    max_tokens: int = 16000
    command_timeout_s: int = 120


class ClaudeAgent:
    name = "claude"

    def __init__(self) -> None:
        self._params: ClaudeAgentParams | None = None

    async def setup(self, config: AgentConfig) -> None:
        self._params = ClaudeAgentParams.model_validate(config.params)

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
        final_text = None
        try:
            final_text = await self._loop(client, task, sandbox)
        except BadRequestError as exc:
            if exc.type != BUDGET_EXCEEDED_ERROR:
                raise
            recorder.event("agent_message", text=f"stopped: {exc.message}")
        finally:
            await client.close()
        if final_text:
            recorder.event("agent_message", text=final_text)
        return AgentResult(final_message=final_text)

    async def teardown(self) -> None:
        self._params = None

    async def _loop(
        self, client: AsyncAnthropic, task: TaskContext, sandbox: Sandbox
    ) -> str | None:
        params = self._params
        extra = {"output_config": {"effort": params.effort}} if params.effort else {}
        runner = client.beta.messages.tool_runner(
            model=params.model,
            max_tokens=params.max_tokens,
            system=SYSTEM_PROMPT.format(workdir=task.workdir),
            tools=_sandbox_tools(sandbox, params.command_timeout_s),
            messages=[{"role": "user", "content": task.instructions}],
            **extra,
        )
        final_text = None
        async for message in runner:
            texts = [block.text for block in message.content if block.type == "text"]
            if texts:
                final_text = "\n".join(texts)
        return final_text


def _sandbox_tools(sandbox: Sandbox, command_timeout_s: int) -> list:
    @beta_async_tool
    async def bash(command: str) -> str:
        """Run a shell command in the task workspace and return its exit code and output.

        Args:
            command: The shell command to run, for example "python -m unittest".
        """
        try:
            result = await sandbox.exec(command, timeout_s=command_timeout_s)
        except SandboxError as exc:
            return f"error: {exc}"
        output = (result.stdout + result.stderr).decode("utf-8", "replace")[-OUTPUT_TAIL_CHARS:]
        status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
        return f"{status}\n{output}"

    @beta_async_tool
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the workspace with the complete new contents.

        Args:
            path: File path, relative to the workspace.
            content: The complete file contents to write.
        """
        try:
            await sandbox.write_file(path, content.encode())
        except SandboxError as exc:
            return f"error: {exc}"
        return f"wrote {len(content)} characters to {path}"

    return [bash, write_file]
