"""Model providers besides Anthropic (FR-GW-09): which server answers a model, at what price.

Agents and judges name a provider's model as `<provider>/<model>`, for example
`ollama/qwen3-coder`; Claude models keep their plain names and go to Anthropic. Providers are
listed in a YAML file (`providers.yaml` by default), each with the base URL of a server that
speaks the Anthropic Messages API, the environment variable that holds its key, and its prices.
A server that needs no key, such as a local one, is called by the gateway directly; keys stay
with the credential proxy (NFR-SEC-01). Every call is budgeted, so a model without a price is
refused; the price under "*" covers every model a provider serves that is not listed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentoscopy.gateway.pricing import MOCK_MODEL_PREFIX, Price, is_mock, price_for
from agentoscopy.spec import load_spec

ANTHROPIC = "anthropic"
MOCK = "mock"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_FILE = Path("providers.yaml")
ANY_MODEL = "*"
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]*$"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class _PriceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input: float = Field(ge=0)  # USD per million tokens, like pricing.PRICES
    output: float = Field(ge=0)
    # Left out, cache reads cost the same as input: OpenRouter bills them that way when a
    # model's providers give no discount, and a budget should never undercount.
    cache_read: float | None = Field(default=None, ge=0)

    def price(self) -> Price:
        cache_read = self.input if self.cache_read is None else self.cache_read
        return Price(self.input, self.output, cache_read)


class _ProviderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]*$")
    prices: dict[str, _PriceSpec] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be an http or https URL")
        return url.rstrip("/")

    @model_validator(mode="after")
    def _no_key_in_the_clear(self) -> _ProviderSpec:
        parts = urlsplit(self.base_url)
        if self.api_key_env and parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
            raise ValueError("its key would travel unencrypted: use https for a remote server")
        return self


class _ProvidersFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    providers: dict[str, _ProviderSpec] = Field(default_factory=dict)

    @field_validator("providers")
    @classmethod
    def _usable_names(cls, providers: dict[str, _ProviderSpec]) -> dict[str, _ProviderSpec]:
        for name in providers:
            if not re.fullmatch(NAME_PATTERN, name):
                raise ValueError(f"{name!r}: names are lowercase letters, digits, and hyphens")
            if name == ANTHROPIC or name.startswith(MOCK_MODEL_PREFIX):
                raise ValueError(f"{name!r} is reserved")
        return providers


@dataclass(frozen=True)
class Provider:
    base_url: str
    api_key_env: str | None  # None: the server needs no key, so the gateway calls it directly
    prices: Mapping[str, Price]


@dataclass(frozen=True)
class Route:
    """Where one model's calls go: the provider, the name it serves the model under, and the
    price the call is budgeted at."""

    provider: str  # ANTHROPIC, MOCK, or a provider from the file
    model: str
    price: Price


@dataclass(frozen=True)
class Providers:
    """The providers this process knows, and those it can reach: Anthropic when its key is
    set, and each listed provider whose key is set or that needs none."""

    configured: Mapping[str, Provider] = field(default_factory=dict)
    reachable: frozenset[str] = frozenset()

    def route(self, model: str) -> Route | None:
        """None when the model is not priced, so its calls cannot be budgeted."""
        if is_mock(model):
            return Route(MOCK, model, price_for(model))
        name, _, served_as = model.partition("/")
        provider = self.configured.get(name) if served_as else None
        if provider is None:
            price = price_for(model)
            return Route(ANTHROPIC, model, price) if price else None
        price = provider.prices.get(served_as) or provider.prices.get(ANY_MODEL)
        return Route(name, served_as, price) if price else None

    def missing_key(self, route: Route) -> str | None:
        """The environment variable that was not set for this route's provider, if any."""
        if route.provider == MOCK or route.provider in self.reachable:
            return None
        if route.provider == ANTHROPIC:
            return ANTHROPIC_KEY_ENV
        return self.configured[route.provider].api_key_env

    def direct_url(self, route: Route) -> str | None:
        """The base URL of a provider that needs no key; None means calls go through the
        credential proxy."""
        provider = self.configured.get(route.provider)
        return provider.base_url if provider and provider.api_key_env is None else None

    def keyed(self) -> dict[str, Provider]:
        """The reachable listed providers whose keys the credential proxy must hold."""
        return {
            name: provider
            for name, provider in self.configured.items()
            if provider.api_key_env and name in self.reachable
        }

    def key_envs(self) -> list[str]:
        return [
            provider.api_key_env for provider in self.configured.values() if provider.api_key_env
        ]


ANTHROPIC_ONLY = Providers(reachable=frozenset({ANTHROPIC}))


def load_providers(path: Path | None, environ: Mapping[str, str] = os.environ) -> Providers:
    """The providers in `path`, or in `providers.yaml` when no path is given and that file
    exists. A SpecError names the file and each bad field."""
    if path is None and not DEFAULT_FILE.is_file():
        spec = _ProvidersFile()
    else:
        spec = load_spec(_ProvidersFile, path or DEFAULT_FILE)
    configured = {
        name: Provider(
            provider.base_url,
            provider.api_key_env,
            {model: price.price() for model, price in provider.prices.items()},
        )
        for name, provider in spec.providers.items()
    }
    reachable = {
        name
        for name, provider in configured.items()
        if provider.api_key_env is None or environ.get(provider.api_key_env)
    }
    if environ.get(ANTHROPIC_KEY_ENV):
        reachable.add(ANTHROPIC)
    return Providers(configured, frozenset(reachable))
