"""Model gateway (§3.2): an Anthropic Messages API proxy that every agent model call goes through.

It authenticates each trial by its own token, enforces the trial's step, token, and cost budgets,
records model events into the trial's trajectory, retries provider overloads without charging the
agent, and serves the mock provider in-process. Real provider calls go through the credential
proxy, a separate process that alone holds the API key (NFR-SEC-01).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from typing import Any

import aiohttp
from aiohttp import web

from agentoscopy.adapters.base import BUDGET_EXCEEDED_ERROR
from agentoscopy.gateway.credential_proxy import SECRET_HEADER
from agentoscopy.gateway.mock import estimate_input_tokens, mock_message, sse_events
from agentoscopy.gateway.pricing import Price, is_mock, price_for
from agentoscopy.gateway.session import BudgetExhausted, TrialSession
from agentoscopy.gateway.sse import MessageAccumulator
from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.spec import Budget

UPSTREAM_ATTEMPTS = 4
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
PROVIDER_FAILURE_TYPES = frozenset({"api_error", "overloaded_error"})
FORWARDED_HEADERS = ("anthropic-version", "anthropic-beta")
RECORDED_PARAMS = (
    "max_tokens",
    "stream",
    "temperature",
    "thinking",
    "output_config",
    "tool_choice",
)
MAX_UPSTREAM_CONCURRENCY = 16  # per-provider limit (FR-EXE-02)
MAX_REQUEST_BYTES = 64 * 1024 * 1024


class ProviderUnavailable(Exception):
    """The provider kept failing after every retry."""


class Gateway:
    def __init__(
        self,
        upstream_url: str | None = None,
        upstream_secret: str | None = None,
        *,
        backoff_base_s: float = BACKOFF_BASE_S,
    ) -> None:
        self.base_url = ""
        self._sessions: dict[str, TrialSession] = {}
        self._upstream_url = upstream_url
        self._upstream_secret = upstream_secret
        self._backoff_base_s = backoff_base_s
        self._upstream_slots = asyncio.Semaphore(MAX_UPSTREAM_CONCURRENCY)
        self._runner: web.AppRunner | None = None
        self._client: aiohttp.ClientSession | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES)
        app.router.add_post("/v1/messages", self._messages)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, host, port).start()
        self.base_url = f"http://{host}:{self._runner.addresses[0][1]}"
        self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
        if self._runner:
            await self._runner.cleanup()

    def open_session(
        self, trial_id: str, budget: Budget, recorder: TrajectoryRecorder, seed_key: str
    ) -> TrialSession:
        token = "agt-" + secrets.token_urlsafe(32)
        recorder.add_secret(token)
        session = TrialSession(trial_id, token, self.base_url, budget, recorder, seed_key)
        self._sessions[token] = session
        return session

    def close_session(self, session: TrialSession) -> None:
        self._sessions.pop(session.token, None)

    async def _messages(self, request: web.Request) -> web.StreamResponse:
        session = self._sessions.get(_token(request))
        if session is None:
            return _error(401, "authentication_error", "unknown or expired Agentoscopy trial token")
        if session.provider_error:
            return _error(502, "api_error", f"provider unavailable: {session.provider_error}")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "invalid_request_error", "request body must be JSON")
        problem = _check_body(body)
        if problem:
            return _error(400, "invalid_request_error", problem)
        price = price_for(body["model"])
        if price is None:
            return _error(
                400,
                "invalid_request_error",
                f"model {body['model']!r} has no price in "
                "agentoscopy.gateway.pricing.PRICES, so its cost cannot be budgeted",
            )
        requested = body["max_tokens"]
        try:
            body["max_tokens"] = session.admit(price, requested, estimate_input_tokens(body))
        except BudgetExhausted as exc:
            session.exhausted = exc.dimension
            session.recorder.write(
                "budget_warning", {"dimension": exc.dimension, "used": exc.used, "limit": exc.limit}
            )
            return _error(400, BUDGET_EXCEEDED_ERROR, f"Agentoscopy trial budget: {exc}")
        call_index = session.start_call()
        _record_request(session, call_index, body, requested)
        if is_mock(body["model"]):
            return await self._serve_mock(request, session, call_index, body, price)
        return await self._serve_upstream(request, session, call_index, body, price)

    async def _serve_mock(
        self,
        request: web.Request,
        session: TrialSession,
        call_index: int,
        body: dict[str, Any],
        price: Price,
    ) -> web.StreamResponse:
        message = mock_message(body, f"{session.seed_key}:{call_index}")
        _record_response(session, call_index, message, price, time.monotonic(), 0.0)
        if not body.get("stream"):
            return web.json_response(message)
        response = await _start_event_stream(request)
        for event in sse_events(message):
            await response.write(event)
        await response.write_eof()
        return response

    async def _serve_upstream(
        self,
        request: web.Request,
        session: TrialSession,
        call_index: int,
        body: dict[str, Any],
        price: Price,
    ) -> web.StreamResponse:
        if self._upstream_url is None:
            session.provider_error = (
                "no provider credentials: set ANTHROPIC_API_KEY for agentoscopy run"
            )
            return _error(502, "api_error", session.provider_error)
        headers = {
            name: request.headers[name] for name in FORWARDED_HEADERS if name in request.headers
        }
        headers.update(
            {"content-type": "application/json", SECRET_HEADER: self._upstream_secret or ""}
        )
        url = f"{self._upstream_url}/v1/messages"
        if request.query_string:
            url += f"?{request.query_string}"
        started, backoff_before = time.monotonic(), session.backoff_s
        async with self._upstream_slots:
            try:
                upstream = await self._post_with_retries(
                    session, url, json.dumps(body).encode(), headers
                )
            except ProviderUnavailable as exc:
                session.provider_error = str(exc)
                session.recorder.write("error", {"source": "gateway", "message": str(exc)})
                return _error(502, "api_error", f"provider unavailable: {exc}")
            try:
                backoff_s = session.backoff_s - backoff_before
                if upstream.status != 200:  # e.g. context too long: the agent's error (WF-04 E7)
                    return await _relay_error(session, call_index, upstream)
                if body.get("stream"):
                    return await _relay_stream(
                        request, session, call_index, upstream, price, started, backoff_s
                    )
                message = await upstream.json()
                _record_response(session, call_index, message, price, started, backoff_s)
                return web.json_response(message)
            finally:
                upstream.release()

    async def _post_with_retries(
        self, session: TrialSession, url: str, payload: bytes, headers: dict[str, str]
    ) -> aiohttp.ClientResponse:
        """Retry 429/5xx and connection errors; retries cost no steps but extend the deadline."""
        failure = ""
        for attempt in range(UPSTREAM_ATTEMPTS):
            retry_after = None
            try:
                response = await self._client.post(url, data=payload, headers=headers)
            except aiohttp.ClientError as exc:
                failure = f"{type(exc).__name__}: {exc}"
            else:
                if response.status not in RETRYABLE_STATUSES:
                    return response
                failure = f"HTTP {response.status}"
                retry_after = response.headers.get("retry-after")
                response.release()
            if attempt == UPSTREAM_ATTEMPTS - 1:
                break
            delay = _backoff_delay(attempt, retry_after, self._backoff_base_s)
            session.backoff_s += delay  # counted before the wait so the deadline moves first
            await asyncio.sleep(delay)
        raise ProviderUnavailable(f"{UPSTREAM_ATTEMPTS} attempts failed; last: {failure}")


async def _relay_stream(
    request: web.Request,
    session: TrialSession,
    call_index: int,
    upstream: aiohttp.ClientResponse,
    price: Price,
    started: float,
    backoff_s: float,
) -> web.StreamResponse:
    response = await _start_event_stream(request)
    accumulator = MessageAccumulator()
    try:
        async for chunk in upstream.content.iter_any():
            accumulator.feed(chunk)
            await response.write(chunk)
    except aiohttp.ClientError as exc:
        session.provider_error = f"stream interrupted: {exc}"
        session.recorder.write("error", {"source": "gateway", "message": session.provider_error})
    if accumulator.error and accumulator.error.get("type") in PROVIDER_FAILURE_TYPES:
        session.provider_error = f"stream error: {accumulator.error}"
        session.recorder.write("error", {"source": "gateway", "message": session.provider_error})
    if accumulator.message:
        _record_response(session, call_index, accumulator.message, price, started, backoff_s)
    await response.write_eof()
    return response


async def _relay_error(
    session: TrialSession, call_index: int, upstream: aiohttp.ClientResponse
) -> web.Response:
    data = await upstream.read()
    session.recorder.write(
        "model_response",
        {
            "call_index": call_index,
            "status": upstream.status,
            "error": data.decode("utf-8", "replace")[:2000],
        },
    )
    return web.Response(status=upstream.status, body=data, content_type="application/json")


async def _start_event_stream(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await response.prepare(request)
    return response


def _record_request(
    session: TrialSession, call_index: int, body: dict[str, Any], requested_max_tokens: int
) -> None:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    session.recorder.write(
        "model_request",
        {
            "call_index": call_index,
            "model": body["model"],
            "messages_ref": session.recorder.store_artifact(
                f"requests/{call_index}.json", canonical
            ),
            "params": {key: body[key] for key in RECORDED_PARAMS if key in body},
            "requested_max_tokens": requested_max_tokens,
            "tools": [tool.get("name") for tool in body.get("tools") or []],
            "request_hash": hashlib.sha256(canonical).hexdigest(),
        },
    )


def _record_response(
    session: TrialSession,
    call_index: int,
    message: dict[str, Any],
    request_price: Price,
    started: float,
    backoff_s: float,
) -> None:
    usage = message.get("usage") or {}
    # A refusal fallback can serve the turn with another model; bill the model that answered.
    price = price_for(str(message.get("model", ""))) or request_price
    latency_s = time.monotonic() - started
    call_cost = session.add_usage(price, usage, latency_s)
    session.recorder.write(
        "model_response",
        {
            "call_index": call_index,
            "model": message.get("model"),
            "content": message.get("content"),
            "stop_reason": message.get("stop_reason"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
            "cache_write_tokens": usage.get("cache_creation_input_tokens"),
            "cost_usd": round(call_cost, 8),
            "latency_ms": round(latency_s * 1000),
            "backoff_ms": round(backoff_s * 1000),
        },
    )


def _check_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return "request body must be a JSON object"
    if not isinstance(body.get("model"), str):
        return "model: required string"
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        return "max_tokens: required positive integer"
    return None


def _token(request: web.Request) -> str:
    bearer = request.headers.get("authorization", "")
    return request.headers.get("x-api-key") or bearer.removeprefix("Bearer ").strip()


def _backoff_delay(attempt: int, retry_after: str | None, base_s: float) -> float:
    try:
        delay = float(retry_after) if retry_after else base_s * 2**attempt
    except ValueError:
        delay = base_s * 2**attempt
    return min(max(delay, 0.0), BACKOFF_CAP_S)


def _error(status: int, error_type: str, message: str) -> web.Response:
    body = {"type": "error", "error": {"type": error_type, "message": message}}
    return web.json_response(body, status=status)
