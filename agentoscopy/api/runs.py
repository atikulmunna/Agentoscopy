"""Starting and cancelling runs over HTTP: `POST /runs` and `DELETE /runs/{run_id}`.

A run started here executes in this server process, on its event loop. Agent configs are
named, not uploaded: only files in the server's configs directory can run, so an API caller
cannot point the harness at arbitrary code.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web

from agentoscopy.api.common import HOME, STORE, ApiError, json_body
from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.server import Gateway
from agentoscopy.launch import (
    DEFAULT_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
    Directories,
    LaunchError,
    PreparedRun,
    RunRequest,
    create_run,
    execute_run,
)
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.scheduler.plan import TaskFilter
from agentoscopy.scheduler.recovery import cancel_run
from agentoscopy.scheduler.runner import REVIEW_RATE
from agentoscopy.spec import SLUG_PATTERN
from agentoscopy.storage.store import Store

SLUG = re.compile(SLUG_PATTERN)
ERROR_CODE = re.compile(r"^([A-Z_]+): (.*)$", re.DOTALL)
FIELDS = frozenset(
    {
        "agent",
        "suite",
        "tasks",
        "trials",
        "budget_usd",
        "concurrency",
        "seed",
        "labels",
        "judge_model",
        "review_rate",
        "filter",
    }
)
MAX_TRIALS = 1000
MAX_CONCURRENCY = 64
MAX_TASKS = 1000
MAX_LABELS = 20
MAX_TEXT = 200


@dataclass
class RunService:
    """What this server needs to execute runs, and the runs it is executing."""

    dirs: Directories
    configs_dir: Path
    backend: SandboxBackend
    proxy: CredentialProxy | None = None
    gateway: Gateway | None = None
    running: set[asyncio.Task] = field(default_factory=set)
    _gateway_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def start(self, store: Store, prepared: PreparedRun, home: Path) -> None:
        async with self._gateway_lock:  # one gateway for every run this server executes
            if self.gateway is None:
                proxy = self.proxy
                gateway = Gateway(proxy.url if proxy else None, proxy.secret if proxy else None)
                await gateway.start()
                self.gateway = gateway
        task = asyncio.create_task(self._execute(store, prepared, home, self.gateway))
        self.running.add(task)
        task.add_done_callback(self.running.discard)

    async def close(self) -> None:
        """Server shutdown: runs still executing here are cancelled, as Ctrl-C would."""
        for task in self.running:
            task.cancel()
        await asyncio.gather(*self.running, return_exceptions=True)
        if self.gateway is not None:
            await self.gateway.stop()

    async def _execute(
        self, store: Store, prepared: PreparedRun, home: Path, gateway: Gateway
    ) -> None:
        try:
            await execute_run(store, prepared, home, self.backend, gateway)
        except Exception as exc:  # the run is marked failed; say why on the server's console
            print(f"run {prepared.run_id} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


RUNS = web.AppKey("runs", RunService)


def add_routes(app: web.Application, service: RunService) -> None:
    app[RUNS] = service
    app.router.add_post("/runs", _create)
    app.router.add_delete("/runs/{run_id}", _cancel)
    app.on_cleanup.append(_shutdown)


async def _create(request: web.Request) -> web.Response:
    service, store = request.app[RUNS], request.app[STORE]
    run_request = parse_request(await json_body(request), service.configs_dir)
    try:
        prepared = create_run(
            store, service.dirs, run_request, has_credentials=service.proxy is not None
        )
    except LaunchError as exc:
        match = ERROR_CODE.match(str(exc))
        code, message = match.groups() if match else ("INVALID_INPUT", str(exc))
        raise ApiError(422, code, message) from exc
    await service.start(store, prepared, request.app[HOME])
    return web.json_response({"run_id": prepared.run_id, "status": "pending"}, status=201)


async def _cancel(request: web.Request) -> web.Response:
    service, store, home = request.app[RUNS], request.app[STORE], request.app[HOME]
    run_id = request.match_info["run_id"]
    status = await cancel_run(store, service.backend, home, run_id)
    if status is None:
        raise ApiError(404, "NOT_FOUND", f"no run {run_id}")
    if status not in ("cancelling", "cancelled"):
        raise ApiError(409, "RUN_FINISHED", f"run {run_id} already finished ({status})")
    return web.json_response(
        {"run_id": run_id, "status": status}, status=202 if status == "cancelling" else 200
    )


async def _shutdown(app: web.Application) -> None:
    await app[RUNS].close()


def parse_request(body: dict[str, Any], configs_dir: Path) -> RunRequest:
    """A run request from a JSON body; every field is checked before anything runs."""
    unknown = sorted(set(body) - FIELDS)
    if unknown:
        raise _bad(f"unknown fields: {', '.join(unknown)}")
    suite, tasks = body.get("suite"), body.get("tasks")
    if (suite is None) == (tasks is None):
        raise _bad("name either a suite or a list of tasks")
    if suite is not None and not _is_slug(suite):
        raise _bad("suite must be a suite name")
    if tasks is not None and not (
        isinstance(tasks, list) and 0 < len(tasks) <= MAX_TASKS and all(map(_is_slug, tasks))
    ):
        raise _bad("tasks must be a non-empty list of task ids")
    budget = body.get("budget_usd", 10.0)
    if budget is not None and not (_is_number(budget) and budget > 0):
        raise _bad("budget_usd must be a positive number, or null for no ceiling")
    seed = body.get("seed")
    if seed is not None and not _is_int(seed):
        raise _bad("seed must be an integer")
    review_rate = body.get("review_rate", REVIEW_RATE)
    if not (_is_number(review_rate) and 0 <= review_rate <= 1):
        raise _bad("review_rate must be a number from 0 to 1")
    judge_model = body.get("judge_model", DEFAULT_JUDGE_MODEL)
    if not (isinstance(judge_model, str) and 0 < len(judge_model) <= MAX_TEXT):
        raise _bad("judge_model must be a model name")
    return RunRequest(
        agent=_agent_path(body.get("agent"), configs_dir),
        suite=suite,
        tasks=tuple(tasks or ()),
        trials=_bounded_int(body, "trials", 3, MAX_TRIALS),
        budget_usd=None if budget is None else float(budget),
        concurrency=_bounded_int(body, "concurrency", DEFAULT_CONCURRENCY, MAX_CONCURRENCY),
        seed=seed,
        labels=_labels(body.get("labels", {})),
        judge_model=judge_model,
        review_rate=float(review_rate),
        task_filter=_task_filter(body.get("filter", {})),
    )


def _agent_path(agent: Any, configs_dir: Path) -> Path:
    if not _is_slug(agent):
        raise _bad("agent must name a config in the configs directory, without .yaml")
    path = configs_dir / f"{agent}.yaml"
    if not path.is_file():
        raise ApiError(422, "UNKNOWN_AGENT", f"no agent config {agent!r}")
    return path


def _task_filter(spec: Any) -> TaskFilter:
    """`filter`: lists of `tags`, `categories`, and `difficulties`, and a `failed_in` run id."""
    lists = ("tags", "categories", "difficulties")
    if not isinstance(spec, dict) or set(spec) - {*lists, "failed_in"}:
        raise _bad("filter may hold tags, categories, difficulties, and failed_in")
    values = {}
    for name in lists:
        items = spec.get(name, [])
        if not (isinstance(items, list) and len(items) <= MAX_LABELS):
            raise _bad(f"filter.{name} must be a list of up to {MAX_LABELS} names")
        if not all(isinstance(item, str) and 0 < len(item) <= MAX_TEXT for item in items):
            raise _bad(f"filter.{name} must be a list of up to {MAX_LABELS} names")
        values[name] = frozenset(items)
    failed_in = spec.get("failed_in")
    if failed_in is not None and not (
        isinstance(failed_in, str) and 0 < len(failed_in) <= MAX_TEXT
    ):
        raise _bad("filter.failed_in must be a run id")
    return TaskFilter(values["tags"], values["categories"], values["difficulties"], failed_in)


def _bounded_int(body: dict[str, Any], name: str, default: int, maximum: int) -> int:
    value = body.get(name, default)
    if not (_is_int(value) and 1 <= value <= maximum):
        raise _bad(f"{name} must be an integer from 1 to {maximum}")
    return value


def _labels(labels: Any) -> dict[str, str]:
    valid = (
        isinstance(labels, dict)
        and len(labels) <= MAX_LABELS
        and all(
            isinstance(key, str)
            and isinstance(value, str)
            and 0 < len(key) <= MAX_TEXT
            and len(value) <= MAX_TEXT
            for key, value in labels.items()
        )
    )
    if not valid:
        raise _bad(f"labels must map up to {MAX_LABELS} names to text")
    return dict(labels)


def _is_slug(value: Any) -> bool:
    return isinstance(value, str) and bool(SLUG.match(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _bad(message: str) -> ApiError:
    return ApiError(400, "BAD_REQUEST", message)
