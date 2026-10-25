"""Model providers besides Anthropic: the providers file, routing, and keyless servers."""

import asyncio
from pathlib import Path

import anthropic
import pytest
import yaml
from conftest import running_gateway, running_provider

from agentoscopy.gateway.pricing import Price
from agentoscopy.gateway.providers import (
    ANTHROPIC,
    MOCK,
    Provider,
    Providers,
    Route,
    load_providers,
)
from agentoscopy.recorder.trajectory import read_events
from agentoscopy.spec import Budget, SpecError

REPO = Path(__file__).resolve().parents[2]
PROMPT = [{"role": "user", "content": "hello"}]
FREE = Price(0, 0, 0)
FILE = {
    "providers": {
        "ollama": {
            "base_url": "http://localhost:11434/",
            "prices": {"*": {"input": 0, "output": 0}},
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api",
            "api_key_env": "OPENROUTER_API_KEY",
            "prices": {"qwen/qwen3-coder": {"input": 0.2, "output": 0.8}},
        },
    }
}
HOSTED = Provider("https://example.com", "HOSTED_KEY", {"qwen3": Price(1, 2, 0)})


def write(tmp_path, data):
    path = tmp_path / "providers.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_models_are_routed_by_provider_prefix(tmp_path):
    providers = load_providers(write(tmp_path, FILE), {})

    assert providers.route("ollama/qwen3:8b") == Route("ollama", "qwen3:8b", FREE)
    assert providers.route("openrouter/qwen/qwen3-coder") == Route(
        "openrouter", "qwen/qwen3-coder", Price(0.2, 0.8, 0.2)
    )
    assert providers.route("openrouter/qwen/unlisted") is None  # no "*" price there
    assert providers.route("claude-haiku-5-5").provider == ANTHROPIC
    assert providers.route("mock").provider == MOCK
    assert providers.route("elsewhere/model") is None
    assert providers.configured["ollama"].base_url == "http://localhost:11434"


def test_cache_reads_cost_the_input_price_unless_the_file_lists_one(tmp_path):
    prices = {
        "discounted": {"input": 1, "output": 4, "cache_read": 0.1},
        "undiscounted": {"input": 1, "output": 4},
    }
    hosted = {"base_url": "https://example.com", "prices": prices}
    providers = load_providers(write(tmp_path, {"providers": {"hosted": hosted}}), {})

    assert providers.route("hosted/discounted").price == Price(1, 4, 0.1)
    assert providers.route("hosted/undiscounted").price == Price(1, 4, 1)


@pytest.mark.parametrize(
    ("environ", "reachable"),
    [
        ({}, {"ollama"}),  # a server that needs no key is always reachable
        ({"OPENROUTER_API_KEY": "k"}, {"ollama", "openrouter"}),
        ({"ANTHROPIC_API_KEY": "k"}, {"ollama", ANTHROPIC}),
    ],
)
def test_providers_are_reachable_when_their_key_is_set(tmp_path, environ, reachable):
    providers = load_providers(write(tmp_path, FILE), environ)

    assert providers.reachable == reachable


@pytest.mark.parametrize(
    ("name", "provider", "problem"),
    [
        ("Ollama", {"base_url": "http://localhost:1"}, "lowercase letters"),
        ("anthropic", {"base_url": "http://localhost:1"}, "reserved"),
        ("mock-server", {"base_url": "http://localhost:1"}, "reserved"),
        ("local", {"base_url": "localhost:11434"}, "http or https URL"),
        ("lan", {"base_url": "http://gpu-box:8000", "api_key_env": "KEY"}, "unencrypted"),
        ("local", {"base_url": "http://localhost:1", "api_key": "sk-secret"}, "api_key"),
        ("local", {"base_url": "http://x", "prices": {"*": {"input": -1, "output": 0}}}, "input"),
    ],
)
def test_a_bad_providers_file_is_refused_with_the_field(tmp_path, name, provider, problem):
    with pytest.raises(SpecError, match=problem):
        load_providers(write(tmp_path, {"providers": {name: provider}}), {})


def test_a_named_file_must_exist_but_the_default_need_not(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert load_providers(None, {}) == Providers()
    with pytest.raises(SpecError, match="cannot read file"):
        load_providers(tmp_path / "missing.yaml", {})


def test_the_shipped_providers_file_makes_local_servers_free_and_keyless():
    providers = load_providers(REPO / "providers.yaml", {})

    local = {"ollama", "llama-cpp", "lm-studio", "vllm"}
    assert local <= providers.reachable
    assert all(providers.route(f"{name}/any-model").price == FREE for name in local)
    assert "openrouter" not in providers.reachable  # until OPENROUTER_API_KEY is set


async def ask(providers, recorder, model):
    """One model call through a gateway with no credential proxy: the answer or the error."""
    async with running_gateway(providers=providers) as gateway:
        session = gateway.open_session("t", Budget(), recorder, "seed")
        async with anthropic.AsyncAnthropic(
            base_url=session.endpoint.base_url, api_key=session.endpoint.token, max_retries=0
        ) as client:
            try:
                return await client.messages.create(model=model, max_tokens=50, messages=PROMPT)
            except anthropic.APIStatusError as exc:
                return exc


def test_a_keyless_server_is_called_directly_without_a_key_and_costs_nothing(recorder):
    async def scenario():
        async with running_provider() as server:
            local = Provider(server.base_url, None, {"*": FREE})
            providers = Providers({"local": local}, frozenset({"local"}))
            return await ask(providers, recorder, "local/qwen3:8b"), server.requests

    message, requests = asyncio.run(scenario())

    assert message.content[0].type == "text"
    (sent,) = requests
    headers = {name.lower() for name in sent["headers"]}
    assert sent["body"]["model"] == "qwen3:8b"
    assert not headers & {"x-api-key", "authorization", "x-agentoscopy-proxy-secret"}
    recorder.close()
    (response,) = [e for e in read_events(recorder.path) if e["type"] == "model_response"]
    assert response["payload"]["cost_usd"] == 0


def test_a_provider_without_its_key_or_a_price_is_refused(recorder):
    providers = Providers({"hosted": HOSTED}, frozenset())  # HOSTED_KEY is not set

    no_key = asyncio.run(ask(providers, recorder, "hosted/qwen3"))
    no_price = asyncio.run(ask(providers, recorder, "hosted/unlisted"))

    assert no_key.status_code == 502
    assert "no credentials for hosted: set HOSTED_KEY" in no_key.message
    assert no_price.status_code == 400
    assert "<provider>/<model>" in no_price.message
