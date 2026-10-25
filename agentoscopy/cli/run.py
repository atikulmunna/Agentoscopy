"""`agentoscopy run`: create or resume a run, execute it, and print the summary."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_CHECK_FAILED,
    EXIT_ENVIRONMENT,
    EXIT_INTERRUPTED,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.gateway.access import ModelAccess
from agentoscopy.gateway.providers import Providers, load_providers
from agentoscopy.launch import (
    DEFAULT_CONCURRENCY,
    Directories,
    LaunchError,
    PreparedRun,
    RunRequest,
    create_fork,
    create_replay,
    create_run,
    execute_run,
    resume_run,
)
from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.plan import TaskFilter, estimate_cost
from agentoscopy.scheduler.runner import TrialProgress
from agentoscopy.spec import SpecError, config_hash
from agentoscopy.stats.report import render
from agentoscopy.storage.store import Store

EXIT_BY_STATUS = {"completed": EXIT_OK, "cancelled": EXIT_INTERRUPTED}


def run_command(args: argparse.Namespace) -> int:
    models, failure = connect_models(args)
    if models is None:
        return failure
    store = Store(args.home / DB_NAME)
    try:
        return _run(args, store, models)
    finally:
        store.close()
        models.stop()


def connect_models(args: argparse.Namespace) -> tuple[ModelAccess | None, int]:
    """Load the providers and start the credential proxy first: it takes the keys out of this
    process before any agent code is imported (NFR-SEC-01). Returns the access, or None and
    the exit code to fail with."""
    try:
        providers = load_providers(args.providers)
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None, EXIT_INVALID_INPUT
    try:
        return ModelAccess.start(providers), EXIT_OK
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None, EXIT_ENVIRONMENT


def _run(args: argparse.Namespace, store: Store, models: ModelAccess) -> int:
    dirs = Directories(args.tasks_dir, args.suites_dir, args.home)
    try:
        if args.resume and (args.tag or args.category or args.difficulty or args.failed_in):
            raise ValueError("a resumed run keeps its tasks; it takes no task filters")
        if args.resume:
            prepared = resume_run(
                store,
                dirs,
                args.resume,
                providers=models.providers,
                concurrency=args.concurrency,
            )
        else:
            prepared = create_run(store, dirs, run_request(args), providers=models.providers)
    except (LaunchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    status = execute_with_progress(store, prepared, args.home, models, resumed=bool(args.resume))
    if status is None:
        return EXIT_ENVIRONMENT
    print()
    print(render(run_summary(store, prepared.run_id), store.get_run(prepared.run_id), "table"))
    return EXIT_BY_STATUS.get(status, EXIT_ENVIRONMENT)


def replay_command(args: argparse.Namespace) -> int:
    """`agentoscopy replay <trial_id>`: rerun a trial against its recorded model calls."""
    return _from_recorded_trial(
        args,
        lambda store, dirs, providers: create_replay(
            store, dirs, args.trial_id, providers=providers
        ),
        _replay_verdict,
    )


def fork_command(args: argparse.Namespace) -> int:
    """`agentoscopy fork <trial_id> --at-step N`: replay up to step N, then go live."""
    return _from_recorded_trial(
        args,
        lambda store, dirs, providers: create_fork(
            store,
            dirs,
            args.trial_id,
            args.at_step,
            agent=args.agent,
            inject=args.inject,
            providers=providers,
        ),
        _fork_verdict,
    )


def _from_recorded_trial(
    args: argparse.Namespace,
    create: Callable[[Store, Directories, Providers], PreparedRun],
    verdict: Callable[[dict[str, Any], dict[str, Any]], int],
) -> int:
    models, failure = connect_models(args)  # judges grade live, and a fork calls models live
    if models is None:
        return failure
    store = Store(args.home / DB_NAME)
    try:
        dirs = Directories(args.tasks_dir, args.suites_dir, args.home)
        try:
            prepared = create(store, dirs, models.providers)
        except LaunchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INVALID_INPUT
        status = execute_with_progress(store, prepared, args.home, models)
        if status != "completed":
            return EXIT_BY_STATUS.get(status, EXIT_ENVIRONMENT)
        (rerun,) = store.trial_rows(prepared.run_id)
        if rerun["error_code"] == "REPLAY_DIVERGED":
            print(f"FAIL {rerun['error']}; see trial {rerun['trial_id']}")
            return EXIT_CHECK_FAILED
        return verdict(rerun, store.get_trial(args.trial_id))
    finally:
        store.close()
        models.stop()


def _replay_verdict(replayed: dict[str, Any], source: dict[str, Any]) -> int:
    print(
        f"replay {replayed['trial_id']}: {replayed['outcome']} after {replayed['steps']} model "
        f"calls; original {source['trial_id']}: {source['outcome']} after {source['steps']}"
    )
    if (replayed["outcome"], replayed["steps"]) == (source["outcome"], source["steps"]):
        print("reproduced: same outcome from the same model calls")
        return EXIT_OK
    print("FAIL not reproduced: the replay ended differently")
    return EXIT_CHECK_FAILED


def _fork_verdict(fork: dict[str, Any], source: dict[str, Any]) -> int:
    print(
        f"fork {fork['trial_id']}: {fork['outcome']} after {fork['steps']} model calls; "
        f"original {source['trial_id']}: {source['outcome']} after {source['steps']}"
    )
    print(f"side by side in the web UI: #/side/{source['trial_id']}/{fork['trial_id']}")
    return EXIT_OK


def run_request(args: argparse.Namespace, extra_labels: dict[str, str] | None = None) -> RunRequest:
    if not args.agent:
        raise ValueError("--agent is required unless you pass --resume")
    task_filter = TaskFilter(
        tags=frozenset(getattr(args, "tag", None) or ()),
        categories=frozenset(getattr(args, "category", None) or ()),
        difficulties=frozenset(getattr(args, "difficulty", None) or ()),
        failed_in=getattr(args, "failed_in", None),
    )
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
        task_filter=task_filter,
        top_up=getattr(args, "top_up", 0) or 0,
    )


def execute_with_progress(
    store: Store,
    prepared: PreparedRun,
    home: Path,
    models: ModelAccess,
    *,
    resumed: bool = False,
) -> str | None:
    """Execute a prepared run with live progress. Returns its final status, or None after a
    harness failure (reported on stderr)."""
    settings = prepared.settings
    if not resumed and prepared.playback is None:
        estimate = estimate_cost(
            store, prepared.pinned, config_hash(prepared.config), settings.trials_per_task
        )
        budget = settings.budget_usd
        truncation = " (likely truncated)" if budget is not None and estimate > budget else ""
        budget_text = "none" if budget is None else f"${budget:.2f}"
        print(f"estimated cost ${estimate:.2f}, budget {budget_text}{truncation}")
    finished, total = store.progress(prepared.run_id)
    action = f"resuming at {finished} of {total} trials" if resumed else f"{total} trials"
    playback = prepared.playback
    if playback is not None and playback.fork_at is None:
        action = f"replaying {len(playback.recording)} recorded model calls"
    elif playback is not None:
        action = f"replaying {playback.fork_at - 1} recorded model calls, then live"
    print(
        f"run {prepared.run_id}: {action}, concurrency {settings.concurrency}, "
        f"seed {settings.seed}",
        flush=True,
    )
    try:
        return asyncio.run(_execute(store, prepared, home, models))
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return "cancelled"
    except Exception as exc:  # report any harness failure with its cause, then fail the command
        print(f"error: run failed: {_describe(exc)}", file=sys.stderr)
        return None


async def _execute(store: Store, prepared: PreparedRun, home: Path, models: ModelAccess) -> str:
    gateway = models.gateway()
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
