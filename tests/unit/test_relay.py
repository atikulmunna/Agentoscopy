"""The model-call relay: an agent in an offline sandbox reaches the gateway through it."""

import asyncio
import json

import aiohttp
import pytest
from conftest import running_gateway

from agentoscopy.adapters.base import ModelEndpoint
from agentoscopy.recorder.trajectory import TrajectoryRecorder, read_events
from agentoscopy.relay import bridge
from agentoscopy.relay.bridge import RelayBridge, RelayError
from agentoscopy.spec import Budget
from agentoscopy.testing.fake_sandbox import FakeSandbox

REQUEST = {"model": "mock", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
# What a sandboxed agent sends: it holds no real credential, only a placeholder.
AGENT_HEADERS = {"x-api-key": "placeholder", "anthropic-version": "2023-06-01"}


def through_relay(tmp_path, scenario):
    """Run `scenario(relay_url, http)` against a relay started in a fake sandbox, for a trial
    on a real gateway with the mock model; returns its result and the trial's events."""

    async def main():
        async with running_gateway() as gateway:
            recorder = TrajectoryRecorder(tmp_path / "t.jsonl", tmp_path / "a", "t", 1)
            session = gateway.open_session("t", Budget(), recorder, "seed")
            try:
                async with RelayBridge(FakeSandbox(), session.endpoint) as relay:
                    async with aiohttp.ClientSession() as http:
                        return await scenario(relay.url, http)
            finally:
                gateway.close_session(session)
                recorder.close()

    result = asyncio.run(main())
    return result, read_events(tmp_path / "t.jsonl")


async def post(http, url, body):
    async with http.post(f"{url}/v1/messages", json=body, headers=AGENT_HEADERS) as response:
        return response.status, response.content_type, await response.read()


def test_a_model_call_through_the_relay_reaches_the_gateway_with_the_trial_token(tmp_path):
    (status, _, body), events = through_relay(tmp_path, lambda url, http: post(http, url, REQUEST))

    assert status == 200
    assert json.loads(body)["content"][0]["type"] == "text"
    # The gateway accepted and recorded it, so the bridge swapped in the trial's token.
    assert [e["type"] for e in events if e["type"].startswith("model_")] == [
        "model_request",
        "model_response",
    ]


def test_a_streamed_answer_streams_through(tmp_path):
    (status, content_type, body), _ = through_relay(
        tmp_path, lambda url, http: post(http, url, {**REQUEST, "stream": True})
    )

    assert (status, content_type) == (200, "text/event-stream")
    assert b"event: message_start" in body and b"event: message_stop" in body


def test_the_gateways_refusals_pass_through_unchanged(tmp_path):
    (status, _, body), _ = through_relay(
        tmp_path, lambda url, http: post(http, url, {**REQUEST, "model": "unpriced"})
    )

    assert status == 400
    assert "has no price" in json.loads(body)["error"]["message"]


def test_concurrent_calls_are_each_answered(tmp_path):
    async def scenario(url, http):
        return await asyncio.gather(*(post(http, url, REQUEST) for _ in range(3)))

    answers, events = through_relay(tmp_path, scenario)

    assert [status for status, _, _ in answers] == [200, 200, 200]
    assert sum(e["type"] == "model_response" for e in events) == 3


def test_an_unreachable_gateway_answers_502():
    async def main():
        closed = ModelEndpoint("http://127.0.0.1:9", "token")  # nothing listens there
        async with RelayBridge(FakeSandbox(), closed) as relay, aiohttp.ClientSession() as http:
            return await post(http, relay.url, REQUEST)

    status, _, body = asyncio.run(main())

    assert status == 502
    assert "gateway unreachable" in json.loads(body)["error"]["message"]


def test_a_relay_that_cannot_start_is_reported(monkeypatch):
    monkeypatch.setattr(bridge, "INSIDE_SOURCE", "import sys")  # exits without listening

    async def main():
        async with RelayBridge(FakeSandbox(), ModelEndpoint("http://127.0.0.1:9", "t")):
            pass

    with pytest.raises(RelayError, match="needs python3"):
        asyncio.run(main())
