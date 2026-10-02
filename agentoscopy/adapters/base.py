"""The agent adapter contract (§7.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentoscopy.sandbox.base import Sandbox
from agentoscopy.spec import AgentConfig, Budget

__all__ = [
    "BUDGET_EXCEEDED_ERROR",
    "AgentAdapter",
    "AgentResult",
    "Budget",
    "ModelEndpoint",
    "Recorder",
    "TaskContext",
]

# Error type the gateway returns (HTTP 400) once a trial's step, token, or cost budget is spent.
# With the Anthropic SDK, catch `anthropic.BadRequestError` and compare its `.type` to this.
BUDGET_EXCEEDED_ERROR = "budget_exceeded_error"


@dataclass(frozen=True)
class TaskContext:
    """What the agent may know about the task: never grader details."""

    task_id: str
    instructions: str
    workdir: str


@dataclass(frozen=True)
class ModelEndpoint:
    """Gateway base URL plus a per-trial token; use them as the SDK's base_url and api_key."""

    base_url: str
    token: str


@dataclass(frozen=True)
class AgentResult:
    final_message: str | None = None


class Recorder(Protocol):
    def event(self, event_type: str, **payload: Any) -> None: ...


class AgentAdapter(Protocol):
    name: str

    async def setup(self, config: AgentConfig) -> None: ...

    async def run(
        self,
        task: TaskContext,
        sandbox: Sandbox,
        model_endpoint: ModelEndpoint,
        budget: Budget,
        recorder: Recorder,
    ) -> AgentResult:
        """Drive the agent to completion. All LLM calls must use model_endpoint."""
        ...

    async def teardown(self) -> None: ...
