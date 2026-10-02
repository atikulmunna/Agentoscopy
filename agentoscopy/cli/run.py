"""`agentoscopy run`: pin tasks, create the run, start the gateway, run, and print the summary."""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from agentoscopy.adapters.python import AdapterLoadError, load_python_adapter
from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_ENVIRONMENT,
    EXIT_INTERRUPTED,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.server import Gateway
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.plan import (
    PinnedTask,
    PlanError,
    check_budget,
    estimate_cost,
    pin_tasks,
    plan_trials,
)
from agentoscopy.scheduler.runner import RunExecutor, TrialProgress
from agentoscopy.spec import AgentConfig, SpecError, config_hash, load_agent_config, load_suite
from agentoscopy.stats.report import render
from agentoscopy.storage.store import RunSettings, Store


@dataclass(frozen=True)
class _Prepared:
    pinned: list[PinnedTask]
    config: AgentConfig
    settings: RunSettings


def run_command(args: argparse.Namespace) -> int:
    # Start the credential proxy first: it takes the API key out of this process before any
    # agent code is imported (NFR-SEC-01).
    try:
        proxy = CredentialProxy.start_if_configured()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    store = Store(args.home / DB_NAME)
    try:
        return _run(args, store, proxy)
    finally:
        store.close()
        if proxy:
            proxy.stop()


def _run(args: argparse.Namespace, store: Store, proxy: CredentialProxy | None) -> int:
    try:
        prepared = _prepare(args, store)
    except (SpecError, PlanError, AdapterLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    settings = prepared.settings
    estimate = estimate_cost(
        store, prepared.pinned, config_hash(prepared.config), settings.trials_per_task
    )
    truncation = " (likely truncated)" if estimate > settings.budget_usd else ""
    print(f"estimated cost ${estimate:.2f}, budget ${settings.budget_usd:.2f}{truncation}")
    trials = plan_trials(prepared.pinned, settings.trials_per_task, settings.seed)
    run_id = store.create_run(prepared.config, settings, trials)
    print(
        f"run {run_id}: {len(trials)} trials, concurrency {settings.concurrency}, "
        f"seed {settings.seed}",
        flush=True,
    )
    try:
        asyncio.run(_execute(store, run_id, prepared, args.home, proxy))
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:  # report any harness failure with its cause, then fail the command
        print(f"error: run failed: {_describe(exc)}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    run = store.get_run(run_id)
    summary = run_summary(store, run_id, refresh=True)
    store.save_summary(run_id, summary)
    print()
    print(render(summary, run, "table"))
    return EXIT_OK


def _prepare(args: argparse.Namespace, store: Store) -> _Prepared:
    task_ids = load_suite(args.suites_dir, args.suite).tasks if args.suite else args.tasks
    pinned = pin_tasks(store, args.tasks_dir, task_ids)
    check_budget(pinned, args.budget_usd)
    config = load_agent_config(args.agent)
    load_python_adapter(config.entrypoint)  # fail before the run exists if it cannot load
    suite_version = None
    if args.suite:
        refs = [(item.task.spec.id, item.version) for item in pinned]
        suite_version = store.save_suite_version(args.suite, refs)
    settings = RunSettings(
        suite_id=args.suite,
        suite_version=suite_version,
        trials_per_task=args.trials,
        budget_usd=args.budget_usd,
        seed=args.seed if args.seed is not None else secrets.randbelow(2**31),
        concurrency=args.concurrency,
        harness_version=version("agentoscopy"),
        labels=_labels(args.label),
    )
    return _Prepared(pinned, config, settings)


async def _execute(
    store: Store, run_id: str, prepared: _Prepared, home: Path, proxy: CredentialProxy | None
) -> None:
    gateway = Gateway(proxy.url if proxy else None, proxy.secret if proxy else None)
    await gateway.start()
    try:
        executor = RunExecutor(
            store,
            run_id,
            prepared.pinned,
            prepared.config,
            DockerBackend(),
            gateway,
            home,
            concurrency=prepared.settings.concurrency,
            seed=prepared.settings.seed,
            on_progress=_ProgressPrinter(),
        )
        await executor.execute()
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


def _labels(pairs: list[str]) -> dict[str, str]:
    labels = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"label {pair!r} must look like KEY=VALUE")
        labels[key] = value
    return labels


def _describe(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(inner) for inner in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"
