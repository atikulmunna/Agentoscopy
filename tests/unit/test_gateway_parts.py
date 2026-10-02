import json
from dataclasses import replace

import pytest

from agentoscopy.gateway.mock import estimate_input_tokens, mock_message, sse_events
from agentoscopy.gateway.pricing import PRICES, Price, cost_usd, is_mock, price_for
from agentoscopy.gateway.session import BudgetExhausted, TrialSession
from agentoscopy.gateway.sse import MessageAccumulator
from agentoscopy.spec import Budget

BODY = {"model": "mock", "max_tokens": 1000, "messages": [{"role": "user", "content": "hi"}]}


def test_cost_covers_every_usage_field():
    price = Price(input=4.0, output=20.0, cache_read=0.2)
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 3_000_000,
        "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000},
    }

    # 4 input + 20 output + 0.2 cache read + 2M x 5m writes at 1.25x + 1M x 1h writes at 2x
    assert cost_usd(price, usage) == pytest.approx(4 + 20 + 0.2 + 2 * 4 * 1.25 + 4 * 2)


def test_cache_writes_default_to_five_minute_price():
    price = Price(input=4.0, output=20.0, cache_read=0.2)

    assert cost_usd(price, {"cache_creation_input_tokens": 1_000_000}) == pytest.approx(5.0)


def test_mock_models_share_one_price_and_unknown_models_have_none():
    assert is_mock("mock-fast") and price_for("mock-fast") == PRICES["mock"]
    assert price_for("claude-opus-5-5") == PRICES["claude-opus-5-5"]
    assert price_for("some-unpriced-model") is None


def test_mock_is_deterministic_per_seed_and_respects_max_tokens():
    first, again = mock_message(BODY, "seed-1"), mock_message(BODY, "seed-1")
    other = mock_message(BODY, "seed-2")
    capped = mock_message({**BODY, "max_tokens": 10}, "seed-1")

    assert first == again
    assert first["content"] != other["content"]
    assert capped["usage"]["output_tokens"] == 10
    assert capped["stop_reason"] == "max_tokens"
    assert first["usage"]["input_tokens"] == estimate_input_tokens(BODY)


@pytest.mark.parametrize("chunk_size", [1, 7, 10_000])
def test_accumulator_rebuilds_streamed_message(chunk_size):
    message = mock_message(BODY, "seed")
    stream = b"".join(sse_events(message))
    accumulator = MessageAccumulator()

    for offset in range(0, len(stream), chunk_size):
        accumulator.feed(stream[offset : offset + chunk_size])

    assert accumulator.message["content"] == message["content"]
    assert accumulator.message["usage"] == message["usage"]
    assert accumulator.message["stop_reason"] == message["stop_reason"]


def test_accumulator_assembles_tool_input_and_survives_bad_events():
    events = [
        {"type": "message_start", "message": {"id": "m", "content": [], "usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "bash", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"command": "l'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": 's"}'},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    accumulator = MessageAccumulator()
    for event in events:
        accumulator.feed(f"data: {json.dumps(event)}\n\n".encode())
    accumulator.feed(b"data: {not json\n\n")

    assert accumulator.message["content"][0]["input"] == {"command": "ls"}
    assert accumulator.malformed_events == 1


def session(recorder, **budget):
    return TrialSession("t", "token", "http://gw", Budget(**budget), recorder, "seed")


def test_admit_caps_output_by_remaining_cost(recorder):
    trial = session(recorder, max_cost_usd=0.001)
    price = Price(input=1.0, output=5.0, cache_read=0.1)

    # $0.001 minus 100 input tokens at $1/MTok leaves room for 180 output tokens at $5/MTok.
    assert trial.admit(price, requested_max_tokens=10_000, estimated_input_tokens=100) == 180


@pytest.mark.parametrize(
    ("budget", "usage_change", "dimension"),
    [
        ({"max_steps": 1}, {"steps": 1}, "steps"),
        ({"max_tokens": 100}, {"input_tokens": 100}, "tokens"),
        ({"max_cost_usd": 0.01}, {"cost_usd": 0.01}, "cost"),
    ],
)
def test_admit_refuses_exhausted_budgets(recorder, budget, usage_change, dimension):
    trial = session(recorder, **budget)
    trial.usage = replace(trial.usage, **usage_change)

    with pytest.raises(BudgetExhausted) as raised:
        trial.admit(PRICES["mock"], requested_max_tokens=10, estimated_input_tokens=1)

    assert raised.value.dimension == dimension


def test_start_call_advances_the_recorder_step(recorder):
    trial = session(recorder)

    assert trial.start_call() == 1
    assert recorder.step == 1
