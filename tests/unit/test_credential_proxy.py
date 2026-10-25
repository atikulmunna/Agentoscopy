"""The credential proxy runs as a real subprocess in front of a fake provider."""

import asyncio
import os

import aiohttp
import anthropic
from conftest import running_gateway, running_provider

from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.pricing import Price
from agentoscopy.gateway.providers import ANTHROPIC_ONLY, Provider, Providers
from agentoscopy.spec import Budget

API_KEY = "sk-ant-api03-only-the-proxy-sees-this"
HOSTED_KEY = "sk-or-only-the-proxy-sees-this-too"
PROMPT = [{"role": "user", "content": "hello"}]


def test_without_a_key_no_proxy_starts():
    keyless = Providers(
        {"local": Provider("http://localhost:11434", None, {})}, frozenset({"local"})
    )

    assert CredentialProxy.start(Providers()) is None
    assert CredentialProxy.start(keyless) is None  # the gateway calls keyless servers itself


async def ask(proxy, providers, recorder, model):
    """One model call through a gateway that forwards to `proxy`; returns the answer."""
    async with running_gateway(
        upstream_url=proxy.url, upstream_secret=proxy.secret, providers=providers
    ) as gateway:
        session = gateway.open_session("t", Budget(), recorder, "seed")
        async with anthropic.AsyncAnthropic(
            base_url=session.endpoint.base_url, api_key=session.endpoint.token, max_retries=0
        ) as client:
            return await client.messages.create(model=model, max_tokens=50, messages=PROMPT)


def test_proxy_adds_the_key_that_only_it_holds(recorder, monkeypatch):
    async def scenario():
        async with running_provider() as provider:
            monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
            monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "also-a-credential")
            monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url)
            proxy = CredentialProxy.start(ANTHROPIC_ONLY)
            try:
                stripped = "ANTHROPIC_API_KEY" not in os.environ
                stripped &= "ANTHROPIC_AUTH_TOKEN" not in os.environ
                async with aiohttp.ClientSession() as http:
                    async with http.post(f"{proxy.url}/v1/messages", json={}) as unauthorized:
                        forbidden_status = unauthorized.status
                message = await ask(proxy, ANTHROPIC_ONLY, recorder, "claude-opus-5-5")
            finally:
                proxy.stop()
            return provider, stripped, forbidden_status, message

    provider, stripped, forbidden_status, message = asyncio.run(scenario())

    assert stripped, "the CLI process must drop provider credentials once the proxy holds them"
    assert forbidden_status == 403
    assert message.content[0].type == "text"
    headers = {name.lower(): value for name, value in provider.requests[-1]["headers"].items()}
    assert headers["x-api-key"] == API_KEY
    assert "x-agentoscopy-proxy-secret" not in headers
    recorder.close()
    assert API_KEY not in recorder.path.read_text()


def test_a_listed_provider_gets_its_own_key_and_the_model_without_its_prefix(recorder, monkeypatch):
    async def scenario():
        async with running_provider() as provider:
            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            monkeypatch.setenv("HOSTED_KEY", HOSTED_KEY)
            hosted = Provider(provider.base_url, "HOSTED_KEY", {"qwen/qwen3": Price(1, 2, 0)})
            providers = Providers({"hosted": hosted}, frozenset({"hosted"}))
            proxy = CredentialProxy.start(providers)
            try:
                stripped = "HOSTED_KEY" not in os.environ
                message = await ask(proxy, providers, recorder, "hosted/qwen/qwen3")
            finally:
                proxy.stop()
            return provider, stripped, message

    provider, stripped, message = asyncio.run(scenario())

    assert stripped
    assert message.content[0].type == "text"
    sent = provider.requests[-1]
    headers = {name.lower(): value for name, value in sent["headers"].items()}
    assert headers["authorization"] == f"Bearer {HOSTED_KEY}"
    assert headers["x-api-key"] == HOSTED_KEY
    assert sent["body"]["model"] == "qwen/qwen3"
    recorder.close()
    assert HOSTED_KEY not in recorder.path.read_text()
