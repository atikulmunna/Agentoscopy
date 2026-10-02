"""Credential proxy: the only process that holds the provider API key (NFR-SEC-01).

The gateway forwards provider-bound requests here with a shared secret; this process adds the
API key and relays the provider's response, streaming included. `agentoscopy run` starts it
before any agent code is imported and then removes the key from its own environment, so
in-process agent code never sees it.

Run as `python -m agentoscopy.gateway.credential_proxy`; it prints `PORT <n>` once listening.
"""

from __future__ import annotations

import asyncio
import hmac
import os
import secrets
import subprocess
import sys
from dataclasses import dataclass

import aiohttp
from aiohttp import web

SECRET_HEADER = "x-agentoscopy-proxy-secret"
SECRET_ENV = "AGENTOSCOPY_PROXY_SECRET"
KEY_ENV = "ANTHROPIC_API_KEY"
CREDENTIAL_ENVS = (KEY_ENV, "ANTHROPIC_AUTH_TOKEN")
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
    def start_if_configured(cls) -> CredentialProxy | None:
        """Start the proxy when ANTHROPIC_API_KEY is set, then drop provider credentials from
        this process's environment. Returns None when no key is configured (mock models only)."""
        if not os.environ.get(KEY_ENV):
            return None
        secret = secrets.token_urlsafe(32)
        process = subprocess.Popen(
            [sys.executable, "-m", "agentoscopy.gateway.credential_proxy"],
            env={**os.environ, SECRET_ENV: secret},
            stdout=subprocess.PIPE,
            text=True,
        )
        ready = process.stdout.readline() if process.stdout else ""
        if not ready.startswith("PORT "):
            process.kill()
            raise RuntimeError("credential proxy failed to start (see its output above)")
        for name in CREDENTIAL_ENVS:
            os.environ.pop(name, None)
        return cls(process, f"http://127.0.0.1:{int(ready.split()[1])}", secret)

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()


def build_app(api_key: str, secret: str, base_url: str) -> web.Application:
    async def client_context(app: web.Application):
        app["client"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        yield
        await app["client"].close()

    async def forward(request: web.Request) -> web.StreamResponse:
        if not hmac.compare_digest(request.headers.get(SECRET_HEADER, ""), secret):
            return web.json_response({"error": "forbidden"}, status=403)
        headers = {
            name: request.headers[name] for name in FORWARDED_HEADERS if name in request.headers
        }
        headers["x-api-key"] = api_key
        url = f"{base_url}/v1/messages"
        if request.query_string:
            url += f"?{request.query_string}"
        try:
            async with request.app["client"].post(
                url, data=await request.read(), headers=headers
            ) as upstream:
                return await _relay(request, upstream)
        except aiohttp.ClientError as exc:
            message = f"provider unreachable: {type(exc).__name__}"
            return web.json_response(
                {"type": "error", "error": {"type": "api_error", "message": message}}, status=502
            )

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.cleanup_ctx.append(client_context)
    app.router.add_post("/v1/messages", forward)
    return app


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


async def _serve() -> None:
    api_key, secret = os.environ[KEY_ENV], os.environ[SECRET_ENV]
    base_url = os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")
    runner = web.AppRunner(build_app(api_key, secret, base_url))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    print(f"PORT {runner.addresses[0][1]}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_serve())
