"""The gateway exercised through the official Anthropic SDK, as agents use it."""

import asyncio

import anthropic
import pytest
from conftest import running_gateway, running_provider

from agentoscopy.adapters.base import BUDGET_EXCEEDED_ERROR
from agentoscopy.recorder.trajectory import read_events
from agentoscopy.spec import Budget

PROMPT = [{"role": "user", "content": "hello"}]


def client_for(session):
    endpoint = session.endpoint
    return anthropic.AsyncAnthropic(
        base_url=endpoint.base_url, api_key=endpoint.token, max_retries=0
    )


def events(recorder, event_type):
    return [e["payload"] for e in read_events(recorder.path) if e["type"] == event_type]


def run(scenario):
    return asyncio.run(scenario())


def test_mock_call_is_answered_costed_and_recorded(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                message = await client.messages.create(
                    model="mock", max_tokens=200, messages=PROMPT
                )
            return session, message

    session, message = run(scenario)

    assert message.content[0].type == "text"
    assert session.usage.steps == 1
    assert session.usage.cost_usd > 0
    (request,) = events(recorder, "model_request")
    (response,) = events(recorder, "model_response")
    assert request["call_index"] == 1 and request["params"]["max_tokens"] == 200
    assert (recorder.artifacts_dir / request["messages_ref"]).is_file()
    assert response["output_tokens"] == message.usage.output_tokens
    assert response["cost_usd"] == pytest.approx(session.usage.cost_usd)


def test_streaming_matches_the_non_streaming_response(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            plain_session = gateway.open_session("a", Budget(), recorder, "same-seed")
            streamed_session = gateway.open_session("b", Budget(), recorder, "same-seed")
            async with client_for(plain_session) as client:
                plain = await client.messages.create(model="mock", max_tokens=300, messages=PROMPT)
            async with client_for(streamed_session) as client:
                async with client.messages.stream(
                    model="mock", max_tokens=300, messages=PROMPT
                ) as stream:
                    streamed = await stream.get_final_message()
            return plain, streamed, streamed_session

    plain, streamed, streamed_session = run(scenario)

    assert streamed.content[0].text == plain.content[0].text
    assert streamed.usage.output_tokens == plain.usage.output_tokens
    assert streamed_session.usage.output_tokens == plain.usage.output_tokens


def test_rejects_unknown_tokens_and_unpriced_models(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with anthropic.AsyncAnthropic(
                base_url=gateway.base_url, api_key="wrong", max_retries=0
            ) as stranger:
                with pytest.raises(anthropic.AuthenticationError):
                    await stranger.messages.create(model="mock", max_tokens=10, messages=PROMPT)
            async with client_for(session) as client:
                with pytest.raises(anthropic.BadRequestError) as raised:
                    await client.messages.create(model="unpriced", max_tokens=10, messages=PROMPT)
            return session, raised.value

    session, error = run(scenario)

    assert error.type == "invalid_request_error"
    assert session.usage.steps == 0


def test_step_budget_stops_the_agent_with_a_typed_error(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(max_steps=2), recorder, "seed")
            async with client_for(session) as client:
                for _ in range(2):
                    await client.messages.create(model="mock", max_tokens=10, messages=PROMPT)
                with pytest.raises(anthropic.BadRequestError) as raised:
                    await client.messages.create(model="mock", max_tokens=10, messages=PROMPT)
            return session, raised.value

    session, error = run(scenario)

    assert error.type == BUDGET_EXCEEDED_ERROR
    assert (session.exhausted, session.usage.steps) == ("steps", 2)
    assert events(recorder, "budget_warning") == [{"dimension": "steps", "used": 2, "limit": 2}]


def test_cost_budget_caps_output_and_never_overspends(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(max_cost_usd=0.003), recorder, "seed")
            async with client_for(session) as client:
                with pytest.raises(anthropic.BadRequestError):
                    for _ in range(100):
                        await client.messages.create(model="mock", max_tokens=4000, messages=PROMPT)
            return session

    session = run(scenario)

    assert session.exhausted == "cost"
    assert session.usage.cost_usd <= 0.003
    sent = [request["params"]["max_tokens"] for request in events(recorder, "model_request")]
    assert min(sent) < 4000  # the last calls were capped to what the budget could pay for


def test_provider_overloads_are_retried_without_extra_steps(recorder):
    async def scenario():
        async with (
            running_provider() as provider,
            running_gateway(
                upstream_url=provider.base_url, upstream_secret="s", backoff_base_s=0.01
            ) as gateway,
        ):
            provider.reply_status(529, "overloaded_error")
            provider.reply_status(503)
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                await client.messages.create(
                    model="claude-opus-5-5", max_tokens=50, messages=PROMPT
                )
            return session, provider

    session, provider = run(scenario)

    assert len(provider.requests) == 3
    assert session.usage.steps == 1
    assert session.backoff_s == pytest.approx(0.01 + 0.02)
    assert session.provider_error is None
    assert events(recorder, "model_response")[0]["backoff_ms"] == 30


def test_persistent_provider_failure_marks_the_trial(recorder):
    async def scenario():
        async with (
            running_provider() as provider,
            running_gateway(
                upstream_url=provider.base_url, upstream_secret="s", backoff_base_s=0.001
            ) as gateway,
        ):
            for _ in range(10):
                provider.reply_status(500)
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                for _ in range(2):  # the second call fails fast without reaching the provider
                    with pytest.raises(anthropic.APIStatusError):
                        await client.messages.create(
                            model="claude-opus-5-5", max_tokens=50, messages=PROMPT
                        )
            return session, provider

    session, provider = run(scenario)

    assert "4 attempts failed" in session.provider_error
    assert len(provider.requests) == 4


def test_request_errors_pass_through_without_blaming_the_provider(recorder):
    async def scenario():
        async with (
            running_provider() as provider,
            running_gateway(upstream_url=provider.base_url, upstream_secret="s") as gateway,
        ):
            provider.reply_status(400, "invalid_request_error")
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                with pytest.raises(anthropic.BadRequestError) as raised:
                    await client.messages.create(
                        model="claude-opus-5-5", max_tokens=50, messages=PROMPT
                    )
            return session, raised.value

    session, error = run(scenario)

    assert error.type == "invalid_request_error"
    assert session.provider_error is None
    assert events(recorder, "model_response")[0]["status"] == 400


def test_upstream_stream_is_relayed_and_costed(recorder):
    async def scenario():
        async with (
            running_provider() as provider,
            running_gateway(
                upstream_url=provider.base_url, upstream_secret="proxy-secret"
            ) as gateway,
        ):
            provider.reply_stream()
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                async with client.messages.stream(
                    model="claude-opus-5-5", max_tokens=300, messages=PROMPT
                ) as stream:
                    message = await stream.get_final_message()
            return session, provider, message

    session, provider, message = run(scenario)

    headers = provider.requests[0]["headers"]
    assert headers["x-agentoscopy-proxy-secret"] == "proxy-secret"
    assert session.token not in str(headers)  # the trial token never leaves the gateway
    assert session.usage.output_tokens == message.usage.output_tokens
    # claude-opus-5-5: $4 / $20 per million tokens
    expected = (message.usage.input_tokens * 4 + message.usage.output_tokens * 20) / 1e6
    assert session.usage.cost_usd == pytest.approx(expected)


def test_without_credentials_real_models_fail_as_provider_errors(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(), recorder, "seed")
            async with client_for(session) as client:
                with pytest.raises(anthropic.APIStatusError):
                    await client.messages.create(
                        model="claude-opus-5-5", max_tokens=10, messages=PROMPT
                    )
            return session

    assert "ANTHROPIC_API_KEY" in run(scenario).provider_error


def test_secrets_are_redacted_from_trajectory_and_artifacts(recorder):
    async def scenario():
        async with running_gateway() as gateway:
            session = gateway.open_session("t", Budget(), recorder, "seed")
            leak = f"my token is {session.token} and key sk-ant-api03-abcdefghijklmnop"
            async with client_for(session) as client:
                await client.messages.create(
                    model="mock", max_tokens=10, messages=[{"role": "user", "content": leak}]
                )
            return session

    session = run(scenario)
    recorder.close()

    written = recorder.path.read_text() + "".join(
        path.read_text() for path in recorder.artifacts_dir.rglob("*.json")
    )
    assert session.token not in written
    assert "sk-ant-api03" not in written
    assert "[REDACTED]" in written
