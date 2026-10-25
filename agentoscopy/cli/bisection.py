"""`agentoscopy bisect <trial_id> --agent CONFIG`: find the step where a failed trial went
wrong, by forking it with a config that passes the task (see agentoscopy.bisection)."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable

from agentoscopy.bisection import Bisection, Probe, bisect_steps
from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_CHECK_FAILED,
    EXIT_ENVIRONMENT,
    EXIT_INTERRUPTED,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.cli.run import connect_models, execute_with_progress
from agentoscopy.gateway.access import ModelAccess
from agentoscopy.launch import Directories, LaunchError, create_fork, trial_recording
from agentoscopy.storage.store import Store


class _Aborted(Exception):
    """A probe's run was cancelled or the harness failed; the search cannot go on."""

    def __init__(self, exit_code: int) -> None:
        super().__init__(exit_code)
        self.exit_code = exit_code


def bisect_command(args: argparse.Namespace) -> int:
    models, failure = connect_models(args)  # the rescue config calls models live
    if models is None:
        return failure
    store = Store(args.home / DB_NAME)
    try:
        return _bisect(args, store, models)
    finally:
        store.close()
        models.stop()


def _bisect(args: argparse.Namespace, store: Store, models: ModelAccess) -> int:
    dirs = Directories(args.tasks_dir, args.suites_dir, args.home)
    run_ids: list[str] = []
    try:
        recorded_calls = _failed_trial_calls(store, dirs, args.trial_id, args.budget_usd)
        result = bisect_steps(recorded_calls, _prober(args, store, dirs, models, run_ids))
    except LaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except _Aborted as exc:
        return exc.exit_code
    _report(store, args.trial_id, result, run_ids)
    return EXIT_OK if result.culprit_step is not None else EXIT_CHECK_FAILED


def _failed_trial_calls(store: Store, dirs: Directories, trial_id: str, budget_usd: float) -> int:
    """The trial's recorded model calls, once it is known to be a failed trial that the
    budget can fork at all."""
    trial = store.get_trial(trial_id)
    if trial is None:
        raise LaunchError(f"NOT_FOUND: no trial {trial_id}")
    if trial["outcome"] != "fail":
        raise LaunchError(
            f"NOT_FAILED: trial {trial_id} ended {trial['outcome']}; bisect looks for the step "
            "where a failed trial went wrong"
        )
    cap = store.task_spec(trial["task_id"], trial["task_version"])["budget"]["max_cost_usd"]
    if budget_usd < cap:
        raise LaunchError(
            f"BUDGET_BELOW_TRIAL_CAP: --budget-usd ${budget_usd:.2f} is below the task's "
            f"per-trial cap of ${cap:.2f}, which each fork reserves before it starts"
        )
    return len(trial_recording(dirs.home, trial))


def _prober(
    args: argparse.Namespace,
    store: Store,
    dirs: Directories,
    models: ModelAccess,
    run_ids: list[str],
) -> Callable[[int], Probe]:
    """Forks the trial at a step with the rescue config, within what is left of the budget."""
    search = uuid.uuid4().hex[:8]

    def probe(step: int) -> Probe:
        remaining = args.budget_usd - sum(store.run_spent(run_id) for run_id in run_ids)
        prepared = create_fork(
            store, dirs, args.trial_id, step, agent=args.agent, providers=models.providers,
            trials=args.trials, budget_usd=max(remaining, 0.0), extra_labels={"bisect": search},
        )  # fmt: skip
        run_ids.append(prepared.run_id)
        status = execute_with_progress(store, prepared, args.home, models)
        if status == "cancelled":
            raise _Aborted(EXIT_INTERRUPTED)
        if status != "completed":
            raise _Aborted(EXIT_ENVIRONMENT)
        return _probe_result(store, step, prepared.run_id)

    return probe


def _probe_result(store: Store, step: int, run_id: str) -> Probe:
    rows = store.trial_rows(run_id)
    passes = sum(row["outcome"] == "pass" for row in rows)
    unscored = [row for row in rows if row["outcome"] not in ("pass", "fail")]
    problem = None
    if any(row["outcome"] == "skipped" for row in unscored):
        problem = "were skipped: the bisection budget ran out (raise --budget-usd)"
    elif unscored:
        problem = f"did not finish: {unscored[0]['error_code']} {unscored[0]['error'] or ''}"
    return Probe(step, passes, len(rows), run_id, problem.strip() if problem else None)


def _report(store: Store, trial_id: str, result: Bisection, run_ids: list[str]) -> None:
    print(f"\nbisect of failed trial {trial_id}:")
    for probe in sorted(result.probes, key=lambda probe: probe.step):
        verdict = "rescued" if probe.rescued else "not rescued"
        print(
            f"  fork at step {probe.step}: {probe.passes} of {probe.trials} passed, {verdict} "
            f"(run {probe.run_id})"
        )
    spent = sum(store.run_spent(run_id) for run_id in run_ids)
    print(f"live model spend ${spent:.4f}")
    if result.culprit_step is None:
        print(f"no step found: {result.stopped}")
        return
    step = result.culprit_step
    print(
        f"step {step}: the trial can still be rescued by taking over at step {step}, but not "
        f"once its own recorded answer at step {step} is kept. That answer sealed the failure."
    )
    rescued = next(probe for probe in result.probes if probe.step == step)
    passing = next(row for row in store.trial_rows(rescued.run_id) if row["outcome"] == "pass")
    print(f"side by side in the web UI: #/side/{trial_id}/{passing['trial_id']}")
