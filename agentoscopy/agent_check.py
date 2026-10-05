"""`agentoscopy agent check` (WF-02): try an agent config on the built-in smoke task.

The check passes when the agent (a) called a model through the gateway, (b) acted in the
sandbox or reported a message, and (c) finished on its own: no timeout, crash, or spent
budget. Whether it solved the smoke task is reported but does not decide the check.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from agentoscopy.adapters.base import AgentAdapter
from agentoscopy.gateway.server import Gateway
from agentoscopy.recorder.trajectory import read_events
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.spec import AgentConfig, load_task
from agentoscopy.worker.trial import AttemptResult, AttemptSpec, run_attempt

SMOKE_TASKS_DIR = Path(__file__).resolve().parent / "smoke"
SMOKE_TASK_ID = "echo-file"
AGENT_EVENTS = ("tool_call", "agent_message")


@dataclass(frozen=True)
class CheckReport:
    result: AttemptResult
    model_calls: int
    agent_events: int
    agent_errors: list[str]
    problems: list[tuple[str, str]]  # (error code, explanation)

    @property
    def ok(self) -> bool:
        return not self.problems


async def check_agent(
    config: AgentConfig,
    adapter: AgentAdapter,
    backend: SandboxBackend,
    gateway: Gateway,
    output_root: Path,
) -> CheckReport:
    spec = AttemptSpec(
        run_id="agent-check",
        trial_id=f"check-{uuid.uuid4().hex[:12]}",
        attempt=1,
        task=load_task(SMOKE_TASKS_DIR, SMOKE_TASK_ID),
        config=config,
        seed_key="agent-check",
    )
    result = await run_attempt(spec, adapter, backend, gateway, output_root)
    events = read_events(result.trajectory_path)
    model_calls = sum(event["type"] == "model_request" for event in events)
    agent_events = sum(event["type"] in AGENT_EVENTS for event in events)
    agent_errors = [
        event["payload"]["message"]
        for event in events
        if event["type"] == "error" and event["payload"].get("source") == "agent"
    ]
    return CheckReport(
        result,
        model_calls,
        agent_events,
        agent_errors,
        _problems(result, model_calls, agent_events),
    )


def _problems(result: AttemptResult, model_calls: int, agent_events: int) -> list[tuple[str, str]]:
    if result.outcome == "infra_error":
        return [(result.error_code or "INFRA_ERROR", result.error or "the harness failed")]
    problems = []
    if model_calls == 0:
        problems.append(
            (
                "GATEWAY_BYPASS",
                "the agent made no model calls through the gateway; it must send them to the "
                "base_url, with the token as api_key, that it is given in its model endpoint",
            )
        )
    if agent_events == 0:
        problems.append(
            ("NO_AGENT_EVENTS", "the agent neither used the sandbox nor reported a message")
        )
    if result.termination != "agent_done":
        problems.append(
            ("AGENT_DID_NOT_FINISH", f"the agent stopped with termination {result.termination}")
        )
    return problems
