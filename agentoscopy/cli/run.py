"""`agentoscopy run`: create or resume a run, execute it, and print the summary."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_ENVIRONMENT,
    EXIT_INTERRUPTED,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.server import Gateway
from agentoscopy.launch import (
    DEFAULT_CONCURRENCY,
    Directories,
    LaunchError,
    PreparedRun,
    RunRequest,
    create_run,
    execute_run,
    resume_run,
)
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.plan import estimate_cost
from agentoscopy.scheduler.runner import TrialProgress
from agentoscopy.spec import config_hash
from agentoscopy.stats.report import render
from agentoscopy.storage.store import Store

EXIT_BY_STATUS = {"completed": EXIT_OK, "cancelled": EXIT_INTERRUPTED}


def run_command(args: argparse.Namespace) -> int:
    proxy, failure = start_proxy()
    if failure is not None:
        return failure
    store = Store(args.home / DB_NAME)
    try:
        return _run(args, store, proxy)
    finally:
        store.close()
        if proxy:
            proxy.stop()


def start_proxy() -> tuple[CredentialProxy | None, int | None]:
    """Start the credential proxy first: it takes the API key out of this process before any
    agent code is imported (NFR-SEC-01). Returns (proxy, exit code on failure)."""
    try:
        return CredentialProxy.start_if_configured(), None
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None, EXIT_ENVIRONMENT


def _run(args: argparse.Namespace, store: Store, proxy: CredentialProxy | None) -> int:
    dirs = Directories(args.tasks_dir, args.suites_dir, args.home)
    try:
        if args.resume:
            prepared = resume_run(
                store,
                dirs,
                args.resume,
                has_credentials=proxy is not None,
                concurrency=args.concurrency,
            )
        else:
            prepared = create_run(store, dirs, run_request(args), has_credentials=proxy is not None)
    except (LaunchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    status = execute_with_progress(store, prepared, args.home, proxy, resumed=bool(args.resume))
    if status is None:
        return EXIT_ENVIRONMENT
    print()
    print(render(run_summary(store, prepared.run_id), store.get_run(prepared.run_id), "table"))
    return EXIT_BY_STATUS.get(status, EXIT_ENVIRONMENT)


def run_request(args: argparse.Namespace, extra_labels: dict[str, str] | None = None) -> RunRequest:
    if not args.agent:
        raise ValueError("--agent is required unless you pass --resume")
    return RunRequest(
        agent=args.agent,
        suite=args.suite,
        tasks=tuple(args.tasks or ()),
        trials=args.trials,
        budget_usd=args.budget_usd,
        concurrency=args.concurrency or DEFAULT_CONCURRENCY,
        seed=args.seed,
        labels={**labels(args.label), **(extra_labels or {})},
        judge_model=args.judge_model,
        review_rate=args.review_rate,
    )


def execute_with_progress(
    store: Store,
    prepared: PreparedRun,
    home: Path,
    proxy: CredentialProxy | None,
    *,
    resumed: bool = False,
) -> str | None:
    """Execute a prepared run with live progress. Returns its final status, or None after a
    harness failure (reported on stderr)."""
    settings = prepared.settings
    if not resumed:
        estimate = estimate_cost(
            store, prepared.pinned, config_hash(prepared.config), settings.trials_per_task
        )
        budget = settings.budget_usd
        truncation = " (likely truncated)" if budget is not None and estimate > budget else ""
        budget_text = "none" if budget is None else f"${budget:.2f}"
        print(f"estimated cost ${estimate:.2f}, budget {budget_text}{truncation}")
    finished, total = store.progress(prepared.run_id)
    action = f"resuming at {finished} of {total} trials" if resumed else f"{total} trials"
    print(
        f"run {prepared.run_id}: {action}, concurrency {settings.concurrency}, "
        f"seed {settings.seed}",
        flush=True,
    )
    try:
        return asyncio.run(_execute(store, prepared, home, proxy))
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return "cancelled"
    except Exception as exc:  # report any harness failure with its cause, then fail the command
        print(f"error: run failed: {_describe(exc)}", file=sys.stderr)
        return None


async def _execute(
    store: Store, prepared: PreparedRun, home: Path, proxy: CredentialProxy | None
) -> str:
    gateway = Gateway(proxy.url if proxy else None, proxy.secret if proxy else None)
    await gateway.start()
    try:
        return await execute_run(
            store, prepared, home, DockerBackend(), gateway, on_progress=_ProgressPrinter()
        )
    finally:
        await gateway.stop()


class _ProgressPrinter:
    """One line per finished attempt, with running pass rate, spend, and ETA (FR-RUN-06)."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._scored = 0
        self._passed = 0

    def __call__(self, progress: TrialProgress) -> None:
        result = progress.result
        if result.outcome in ("pass", "fail") and not progress.retrying:
            self._scored += 1
            self._passed += result.outcome == "pass"
        status = "retrying" if progress.retrying else result.outcome
        detail = result.error_code or result.termination or ""
        rate = f"{self._passed / self._scored:.0%}" if self._scored else "-"
        elapsed = time.monotonic() - self._started
        remaining = progress.total - progress.finished
        eta = elapsed / progress.finished * remaining if progress.finished else 0.0
        print(
            f"[{progress.finished}/{progress.total}] {progress.task_id} #{progress.trial_index} "
            f"(attempt {progress.attempt}): {status} {detail}  ${result.usage.cost_usd:.4f}  "
            f"| pass rate {rate}, spent ${progress.spent_usd:.4f}, eta {eta:.0f}s",
            flush=True,
        )


def labels(pairs: list[str]) -> dict[str, str]:
    parsed = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"label {pair!r} must look like KEY=VALUE")
        parsed[key] = value
    return parsed


def _describe(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(inner) for inner in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"
