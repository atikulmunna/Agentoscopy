"""The credential proxy runs as a real subprocess in front of a fake provider."""

import asyncio
import os

import aiohttp
import anthropic
from conftest import running_gateway, running_provider

from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.spec import Budget

API_KEY = "sk-ant-api03-only-the-proxy-sees-this"
PROMPT = [{"role": "user", "content": "hello"}]


def test_without_a_key_no_proxy_starts(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert CredentialProxy.start_if_configured() is None


def test_proxy_adds_the_key_that_only_it_holds(recorder, monkeypatch):
    async def scenario():
        async with running_provider() as provider:
            monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)
            monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "also-a-credential")
            monkeypatch.setenv("ANTHROPIC_BASE_URL", provider.base_url)
            proxy = CredentialProxy.start_if_configured()
            try:
                stripped = "ANTHROPIC_API_KEY" not in os.environ
                stripped &= "ANTHROPIC_AUTH_TOKEN" not in os.environ
                async with aiohttp.ClientSession() as http:
                    async with http.post(f"{proxy.url}/v1/messages", json={}) as unauthorized:
                        forbidden_status = unauthorized.status
                async with running_gateway(
                    upstream_url=proxy.url, upstream_secret=proxy.secret
                ) as gateway:
                    session = gateway.open_session("t", Budget(), recorder, "seed")
                    async with anthropic.AsyncAnthropic(
                        base_url=session.endpoint.base_url,
                        api_key=session.endpoint.token,
                        max_retries=0,
                    ) as client:
                        message = await client.messages.create(
                            model="claude-opus-5-5", max_tokens=50, messages=PROMPT
                        )
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
