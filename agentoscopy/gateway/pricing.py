"""Per-model prices for cost accounting (FR-GW-07). USD per million tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MOCK_MODEL_PREFIX = "mock"
CACHE_WRITE_5M_MULTIPLIER = 1.25  # cache writes are priced relative to base input
CACHE_WRITE_1H_MULTIPLIER = 2.0


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float


PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(input=10.0, output=50.0, cache_read=0.25),
    "claude-opus-5-5": Price(input=4.0, output=20.0, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.0, output=10.0, cache_read=0.20),
    "claude-haiku-4-5": Price(input=1.0, output=5.0, cache_read=0.10),
    MOCK_MODEL_PREFIX: Price(input=1.0, output=5.0, cache_read=0.10),
}


def is_mock(model: str) -> bool:
    return model.startswith(MOCK_MODEL_PREFIX)


def price_for(model: str) -> Price | None:
    return PRICES[MOCK_MODEL_PREFIX] if is_mock(model) else PRICES.get(model)


def cost_usd(price: Price, usage: dict[str, Any]) -> float:
    cache_written = int(usage.get("cache_creation_input_tokens") or 0)
    breakdown = usage.get("cache_creation") or {}
    written_1h = int(breakdown.get("ephemeral_1h_input_tokens") or 0)
    written_5m = cache_written - written_1h  # without a breakdown, assume the default 5m TTL
    per_million = (
        int(usage.get("input_tokens") or 0) * price.input
        + int(usage.get("output_tokens") or 0) * price.output
        + int(usage.get("cache_read_input_tokens") or 0) * price.cache_read
        + written_5m * price.input * CACHE_WRITE_5M_MULTIPLIER
        + written_1h * price.input * CACHE_WRITE_1H_MULTIPLIER
    )
    return per_million / 1_000_000
