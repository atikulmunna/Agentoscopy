"""How a process reaches models: the providers it knows, and the credential proxy that holds
their keys (None when no key is set: mock models and keyless servers need none)."""

from __future__ import annotations

from dataclasses import dataclass

from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.providers import Providers
from agentoscopy.gateway.server import Gateway


@dataclass(frozen=True)
class ModelAccess:
    providers: Providers
    proxy: CredentialProxy | None = None

    @classmethod
    def start(cls, providers: Providers) -> ModelAccess:
        """Start the credential proxy first: it takes the keys out of this process before any
        agent code is imported (NFR-SEC-01)."""
        return cls(providers, CredentialProxy.start(providers))

    def gateway(self) -> Gateway:
        if self.proxy is None:
            return Gateway(providers=self.providers)
        return Gateway(self.proxy.url, self.proxy.secret, providers=self.providers)

    def stop(self) -> None:
        if self.proxy:
            self.proxy.stop()


OFFLINE = ModelAccess(Providers())  # mock models only
