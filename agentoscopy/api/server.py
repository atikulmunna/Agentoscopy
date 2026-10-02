"""REST API (§7.2) and web UI host (§7.5): runs, trials, comparisons, and human review.

Every API route needs the server's token (`Authorization: Bearer <token>`; `/events` also
accepts `?token=`, since browsers cannot set headers on an EventSource). Requests must name a
loopback host in their Host header, which blocks DNS-rebinding attacks from web pages. The UI
shell itself is public: it holds no data and reads the token from the URL fragment.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from aiohttp import web
from yarl import URL

from agentoscopy.api import reviews
from agentoscopy.api.common import (
    ALLOWED_HOSTS,
    HOME,
    STORE,
    TOKEN,
    ApiError,
    int_query,
)
from agentoscopy.recorder.trajectory import read_events
from agentoscopy.reporting import NotFound, comparison, run_summary, safe_artifact
from agentoscopy.stats.compare import CompareError
from agentoscopy.storage.store import Store

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_PAGE = 500
MAX_PAGE = 5000
EVENT_POLL_S = 1.0
EVENT_HEARTBEAT_S = 15.0

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def create_app(
    store: Store, home: Path, token: str, extra_hosts: frozenset[str] = frozenset()
) -> web.Application:
    app = web.Application(middlewares=[_guard])
    app[STORE], app[HOME], app[TOKEN] = store, home.resolve(), token
    app[ALLOWED_HOSTS] = LOOPBACK_HOSTS | extra_hosts
    app.router.add_get("/", _index)
    app.router.add_static("/ui/", UI_DIR)
    app.router.add_get("/runs", _list_runs)
    app.router.add_get("/runs/{run_id}", _get_run)
    app.router.add_get("/runs/{run_id}/summary", _get_summary)
    app.router.add_get("/runs/{run_id}/trials", _get_trials)
    app.router.add_get("/trials/{trial_id}", _get_trial)
    app.router.add_get("/trials/{trial_id}/trajectory", _get_trajectory)
    app.router.add_get("/trials/{trial_id}/diff", _get_diff)
    app.router.add_get("/compare", _compare)
    app.router.add_get("/events", _events)
    reviews.add_routes(app)
    return app


@web.middleware
async def _guard(request: web.Request, handler: Handler) -> web.StreamResponse:
    try:
        if URL(f"http://{request.host}").host not in request.app[ALLOWED_HOSTS]:
            raise ApiError(403, "HOST_NOT_ALLOWED", f"host {request.host!r} is not allowed")
        if request.path != "/" and not request.path.startswith("/ui/"):
            _check_token(request)
        return await handler(request)
    except ApiError as exc:
        return _error(exc.status, exc.code, str(exc))
    except NotFound as exc:
        return _error(404, "NOT_FOUND", str(exc))
    except CompareError as exc:
        code, _, message = str(exc).partition(": ")
        return _error(422, code, message)


def _check_token(request: web.Request) -> None:
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not supplied and request.path == "/events":
        supplied = request.query.get("token", "")
    if not hmac.compare_digest(supplied.encode(), request.app[TOKEN].encode()):
        raise ApiError(401, "UNAUTHORIZED", "missing or wrong API token")


async def _index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(UI_DIR / "index.html")


async def _list_runs(request: web.Request) -> web.Response:
    runs = request.app[STORE].list_runs()
    filters = {
        "status": request.query.get("status"),
        "suite_id": request.query.get("suite"),
        "config_name": request.query.get("config"),
    }
    runs = [run for run in runs if all(not v or run.get(k) == v for k, v in filters.items())]
    if "label" in request.query:
        key, _, value = request.query["label"].partition("=")
        runs = [run for run in runs if run["labels"].get(key) == value]
    return web.json_response({"runs": runs})


async def _get_run(request: web.Request) -> web.Response:
    return web.json_response(_run(request))


async def _get_summary(request: web.Request) -> web.Response:
    return web.json_response(run_summary(request.app[STORE], request.match_info["run_id"]))


async def _get_trials(request: web.Request) -> web.Response:
    run = _run(request)
    return web.json_response({"trials": request.app[STORE].trial_rows(run["run_id"])})


async def _get_trial(request: web.Request) -> web.Response:
    return web.json_response(_trial(request))


async def _get_trajectory(request: web.Request) -> web.Response:
    trial = _trial(request)
    attempt = int_query(request, "attempt", trial["attempt"])
    offset = int_query(request, "offset", 0)
    limit = min(int_query(request, "limit", DEFAULT_PAGE), MAX_PAGE)
    record = next((a for a in trial["attempts"] if a["attempt"] == attempt), None)
    if record is None or not record["trajectory_uri"]:
        raise ApiError(404, "NOT_FOUND", f"trial has no trajectory for attempt {attempt}")
    events = read_events(_artifact(request, record["trajectory_uri"]))
    return web.json_response(
        {
            "attempt": attempt,
            "offset": offset,
            "limit": limit,
            "total": len(events),
            "events": events[offset : offset + limit],
        }
    )


async def _get_diff(request: web.Request) -> web.Response:
    trial = _trial(request)
    if not trial["final_state_uri"]:
        raise ApiError(404, "NOT_FOUND", "trial has no final state diff")
    path = _artifact(request, trial["final_state_uri"])
    return web.json_response({"diff": path.read_text(encoding="utf-8")})


async def _compare(request: web.Request) -> web.Response:
    base, cand = request.query.get("base"), request.query.get("cand")
    if not base or not cand:
        raise ApiError(400, "BAD_REQUEST", "base and cand run ids are required")
    drift = request.query.get("allow_version_drift", "").lower() in ("1", "true", "yes")
    result = comparison(request.app[STORE], base, cand, allow_version_drift=drift)
    return web.json_response(result)


async def _events(request: web.Request) -> web.StreamResponse:
    """Server-sent `runs` events whenever any run's status, progress, or spend changes."""
    response = web.StreamResponse(
        headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
    )
    await response.prepare(request)
    last, idle = None, 0.0
    try:
        while True:
            snapshot = [
                {key: run[key] for key in ("run_id", "status", "finished_trials", "total_trials")}
                | {"spent_usd": round(run["spent_usd"], 6)}
                for run in request.app[STORE].list_runs(limit=50)
            ]
            if snapshot != last:
                await response.write(f"event: runs\ndata: {json.dumps(snapshot)}\n\n".encode())
                last, idle = snapshot, 0.0
            elif idle >= EVENT_HEARTBEAT_S:
                await response.write(b": keep-alive\n\n")
                idle = 0.0
            await asyncio.sleep(EVENT_POLL_S)
            idle += EVENT_POLL_S
    except (ConnectionResetError, asyncio.CancelledError):
        return response


def _run(request: web.Request) -> dict[str, Any]:
    run = request.app[STORE].get_run(request.match_info["run_id"])
    if run is None:
        raise NotFound(f"no run {request.match_info['run_id']}")
    return run


def _trial(request: web.Request) -> dict[str, Any]:
    trial = request.app[STORE].get_trial(request.match_info["trial_id"])
    if trial is None:
        raise NotFound(f"no trial {request.match_info['trial_id']}")
    return trial


def _artifact(request: web.Request, uri: str) -> Path:
    path = safe_artifact(request.app[HOME], uri)
    if path is None:
        raise ApiError(404, "NOT_FOUND", "artifact is missing or outside the Agentoscopy home")
    return path


def _error(status: int, code: str, message: str) -> web.Response:
    body = {"error": {"code": code, "message": message, "details": {}}}
    return web.json_response(body, status=status)
