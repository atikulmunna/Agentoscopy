"""Per-trial gateway state: the trial token, budget accounting, and signals for the worker."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from agentoscopy.adapters.base import ModelEndpoint
from agentoscopy.gateway.pricing import Price, cost_usd
from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.replay import RecordedCall
from agentoscopy.spec import Budget


class BudgetExhausted(Exception):
    def __init__(self, dimension: str, used: float, limit: float) -> None:
        super().__init__(f"{dimension} budget exhausted ({used} of {limit} used)")
        self.dimension = dimension
        self.used = used
        self.limit = limit


@dataclass(frozen=True)
class Usage:
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    model_calls: int = 0  # answered calls, for gateway latency (NFR-OBS-02)
    model_latency_s: float = 0.0


@dataclass
class TrialSession:
    trial_id: str
    token: str
    base_url: str
    budget: Budget
    recorder: TrajectoryRecorder
    seed_key: str
    usage: Usage = Usage()
    backoff_s: float = 0.0  # extends the trial deadline (FR-EXE-03)
    exhausted: str | None = None  # budget dimension that stopped the agent, if any
    provider_error: str | None = None  # set when the provider stays unavailable (WF-04 E3)
    replay: list[RecordedCall] | None = field(default=None, repr=False)  # replay mode
    replay_divergence: str | None = None  # where a replay stopped matching (FR-RPL-02)

    @property
    def endpoint(self) -> ModelEndpoint:
        return ModelEndpoint(base_url=self.base_url, token=self.token)

    def admit(self, price: Price, requested_max_tokens: int, estimated_input_tokens: int) -> int:
        """The max_tokens this call may use; raises BudgetExhausted when nothing is left.

        A call's cost is only known after it returns, so the output is capped at what the
        remaining cost and token budgets can pay for (FR-GW-04).
        """
        budget, usage = self.budget, self.usage
        if usage.steps >= budget.max_steps:
            raise BudgetExhausted("steps", usage.steps, budget.max_steps)
        used_tokens = (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
        )
        token_room = budget.max_tokens - used_tokens - estimated_input_tokens
        if token_room <= 0:
            raise BudgetExhausted("tokens", used_tokens, budget.max_tokens)
        input_cost = estimated_input_tokens * price.input / 1_000_000
        cost_room = int(
            (budget.max_cost_usd - usage.cost_usd - input_cost) * 1_000_000 / price.output
        )
        if cost_room <= 0:
            raise BudgetExhausted("cost", round(usage.cost_usd, 6), budget.max_cost_usd)
        return min(requested_max_tokens, token_room, cost_room)

    def start_call(self) -> int:
        """Count the call as a step; tool events that follow belong to it (§1.3)."""
        self.usage = replace(self.usage, steps=self.usage.steps + 1)
        self.recorder.step = self.usage.steps
        return self.usage.steps

    def add_usage(self, price: Price, usage: dict[str, Any], latency_s: float = 0.0) -> float:
        call_cost = cost_usd(price, usage)
        current = self.usage
        self.usage = replace(
            current,
            input_tokens=current.input_tokens + int(usage.get("input_tokens") or 0),
            output_tokens=current.output_tokens + int(usage.get("output_tokens") or 0),
            cache_read_tokens=current.cache_read_tokens
            + int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=current.cache_write_tokens
            + int(usage.get("cache_creation_input_tokens") or 0),
            cost_usd=current.cost_usd + call_cost,
            model_calls=current.model_calls + 1,
            model_latency_s=current.model_latency_s + latency_s,
        )
        return call_cost
