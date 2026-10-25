"""Credential proxy: the only process that holds provider API keys (NFR-SEC-01).

The gateway forwards provider-bound requests here with a shared secret; this process adds the
provider's key and relays its response, streaming included. Anthropic's calls arrive at
`/v1/messages`, and those of a provider from the providers file at
`/providers/<name>/v1/messages`. `agentoscopy run` starts it before any agent code is imported
and then removes the keys from its own environment, so in-process agent code never sees them.
Providers that need no key, such as local servers, are not routed through here.

Run as `python -m agentoscopy.gateway.credential_proxy`; it prints `PORT <n>` once listening.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass

import aiohttp
from aiohttp import web

from agentoscopy.gateway.providers import ANTHROPIC, ANTHROPIC_KEY_ENV, Providers

SECRET_HEADER = "x-agentoscopy-proxy-secret"
SECRET_ENV = "AGENTOSCOPY_PROXY_SECRET"
PROVIDERS_ENV = "AGENTOSCOPY_PROXY_PROVIDERS"
CREDENTIAL_ENVS = (ANTHROPIC_KEY_ENV, "ANTHROPIC_AUTH_TOKEN")
BASE_URL_ENV = "ANTHROPIC_BASE_URL"
DEFAULT_BASE_URL = "https://api.anthropic.com"
FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta", "content-type")
RELAYED_RESPONSE_HEADERS = ("retry-after", "request-id")
STOP_TIMEOUT_S = 5


@dataclass
class CredentialProxy:
    process: subprocess.Popen
    url: str
    secret: str

    @classmethod
    def start(cls, providers: Providers) -> CredentialProxy | None:
        """Start the proxy when any reachable provider has a key, then drop every provider
        credential from this process's environment. Returns None when there is no key to
        hold (mock models and keyless servers only)."""
        keyed = providers.keyed()
        if ANTHROPIC not in providers.reachable and not keyed:
            return None
        secret = secrets.token_urlsafe(32)
        listed = {name: [p.base_url, p.api_key_env] for name, p in keyed.items()}
        process = subprocess.Popen(
            [sys.executable, "-m", "agentoscopy.gateway.credential_proxy"],
            env={**os.environ, SECRET_ENV: secret, PROVIDERS_ENV: json.dumps(listed)},
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = process.stdout.readline() if process.stdout else ""
        if not ready.startswith("PORT "):
            process.kill()
            raise RuntimeError("credential proxy failed to start (see its output above)")
        for name in (*CREDENTIAL_ENVS, *providers.key_envs()):
            os.environ.pop(name, None)
        return cls(process, f"http://127.0.0.1:{int(ready.split()[1])}", secret)

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()


@dataclass(frozen=True)
class Upstream:
    base_url: str
    auth: dict[str, str]  # the headers that carry the key


def build_app(secret: str, upstreams: dict[str, Upstream]) -> web.Application:
    async def client_context(app: web.Application):
        app["client"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        yield
        await app["client"].close()

    async def forward(request: web.Request) -> web.StreamResponse:
        if not hmac.compare_digest(request.headers.get(SECRET_HEADER, ""), secret):
            return web.json_response({"error": "forbidden"}, status=403)
        name = request.match_info.get("provider", ANTHROPIC)
        upstream = upstreams.get(name)
        if upstream is None:
            return _error(404, "not_found_error", f"the credential proxy holds no key for {name}")
        headers = {
            header: request.headers[header]
            for header in FORWARDED_HEADERS
            if header in request.headers
        }
        headers.update(upstream.auth)
        url = f"{upstream.base_url}/v1/messages"
        if request.query_string:
            url += f"?{request.query_string}"
        try:
            async with request.app["client"].post(
                url, data=await request.read(), headers=headers
            ) as response:
                return await _relay(request, response)
        except aiohttp.ClientError as exc:
            return _error(502, "api_error", f"provider unreachable: {type(exc).__name__}")

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.cleanup_ctx.append(client_context)
    app.router.add_post("/v1/messages", forward)
    app.router.add_post("/providers/{provider}/v1/messages", forward)
    return app


def _error(status: int, error_type: str, message: str) -> web.Response:
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    return web.json_response(body, status=status)


async def _relay(request: web.Request, upstream: aiohttp.ClientResponse) -> web.StreamResponse:
    response = web.StreamResponse(status=upstream.status)
    response.content_type = upstream.content_type
    for name in RELAYED_RESPONSE_HEADERS:
        if name in upstream.headers:
            response.headers[name] = upstream.headers[name]
    await response.prepare(request)
    async for chunk in upstream.content.iter_any():
        await response.write(chunk)
    await response.write_eof()
    return response


def upstreams_from(environ: dict[str, str]) -> dict[str, Upstream]:
    """The servers this proxy may call, with the keys it found for them."""
    upstreams = {}
    if environ.get(ANTHROPIC_KEY_ENV):
        base_url = environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")
        upstreams[ANTHROPIC] = Upstream(base_url, {"x-api-key": environ[ANTHROPIC_KEY_ENV]})
    for name, (base_url, key_env) in json.loads(environ.get(PROVIDERS_ENV) or "{}").items():
        key = environ[key_env]
        # Anthropic-compatible servers differ in where they read a key: OpenRouter takes a
        # bearer token, llama.cpp's server an x-api-key header. Both go to the same server.
        upstreams[name] = Upstream(base_url, {"authorization": f"Bearer {key}", "x-api-key": key})
    return upstreams


async def _serve() -> None:
    runner = web.AppRunner(build_app(os.environ[SECRET_ENV], upstreams_from(dict(os.environ))))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    print(f"PORT {runner.addresses[0][1]}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_serve())
