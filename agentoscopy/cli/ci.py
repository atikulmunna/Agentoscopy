"""`agentoscopy ci`: run a suite, compare it with a baseline run, report on the pull request.

Exit codes (FR-CI-04): 0 no regression, 1 regression, 2 harness error (including invalid
input), 3 the candidate run hit its budget.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime

from agentoscopy.cli.common import DB_NAME
from agentoscopy.cli.run import connect_models, execute_with_progress, run_request
from agentoscopy.gateway.access import ModelAccess
from agentoscopy.github import GitHubError, pull_request_from_env, upsert_comment
from agentoscopy.launch import Directories, LaunchError, create_run
from agentoscopy.reporting import NotFound, comparison
from agentoscopy.scheduler.plan import PlanError, pin_tasks
from agentoscopy.spec import SpecError, load_suite, suite_task_ids
from agentoscopy.stats.compare import CompareError
from agentoscopy.stats.report import render_ci_comment, render_comparison
from agentoscopy.storage.store import Store

CI_OK = 0
CI_REGRESSION = 1
CI_HARNESS_ERROR = 2
CI_BUDGET_EXCEEDED = 3
OVERRIDE_LABEL = "eval-override"


def ci_command(args: argparse.Namespace) -> int:
    models, _ = connect_models(args)
    if models is None:
        return CI_HARNESS_ERROR
    store = Store(args.home / DB_NAME)
    try:
        return _ci(args, store, models)
    finally:
        store.close()
        models.stop()


def _ci(args: argparse.Namespace, store: Store, models: ModelAccess) -> int:
    dirs = Directories(args.tasks_dir, args.suites_dir, args.home)
    try:
        baseline_id = _find_baseline(store, dirs, args) or _run_baseline(store, dirs, args, models)
        if baseline_id is None:
            return CI_HARNESS_ERROR
        ci_labels = {"ci": "candidate"} | ({"override": OVERRIDE_LABEL} if args.override else {})
        candidate = create_run(
            store, dirs, run_request(args, ci_labels), providers=models.providers
        )
    except (LaunchError, PlanError, SpecError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CI_HARNESS_ERROR
    if execute_with_progress(store, candidate, args.home, models) != "completed":
        print("error: the candidate run did not complete", file=sys.stderr)
        return CI_HARNESS_ERROR
    try:
        result = comparison(store, baseline_id, candidate.run_id)
    except (NotFound, CompareError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return CI_HARNESS_ERROR
    truncated = "BUDGET_TRUNCATED" in store.get_run(candidate.run_id)["flags"]
    regression = result["verdict"] == "REGRESSION"
    print()
    print(render_comparison(result, "table"))
    comment = render_ci_comment(
        result, overridden=regression and args.override, truncated=truncated
    )
    if not _publish(comment, args):
        return CI_HARNESS_ERROR
    if truncated:
        return CI_BUDGET_EXCEEDED
    return CI_REGRESSION if regression and not args.override else CI_OK


def _find_baseline(store: Store, dirs: Directories, args: argparse.Namespace) -> str | None:
    """The newest recent completed run of the suite's current version on the baseline branch."""
    task_ids = suite_task_ids(dirs.tasks, load_suite(dirs.suites, args.suite))
    pinned = pin_tasks(store, dirs.tasks, task_ids)
    refs = [(item.task.spec.id, item.version) for item in pinned]
    version = store.current_suite_version(args.suite, refs)
    if version is None:
        return None  # this suite content has never run, so no run can serve as its baseline
    since = datetime.now(UTC) - args.max_baseline_age
    found = store.find_baseline(args.suite, version, args.baseline, since)
    if found:
        print(f"baseline: run {found} (branch {args.baseline})")
    return found


def _run_baseline(
    store: Store, dirs: Directories, args: argparse.Namespace, models: ModelAccess
) -> str | None:
    """No usable baseline exists, so run one with the baseline branch's agent config."""
    if args.baseline_agent is None:
        print(
            f"error: BASELINE_MISSING: no completed run of suite {args.suite} at its current "
            f"version on branch {args.baseline} finished recently enough; pass --baseline-agent "
            "to run one",
            file=sys.stderr,
        )
        return None
    request = replace(
        run_request(args),
        agent=args.baseline_agent,
        labels={"branch": args.baseline, "ci": "baseline"},
    )
    prepared = create_run(store, dirs, request, providers=models.providers)
    print(f"no recent baseline on branch {args.baseline}; running one first")
    if execute_with_progress(store, prepared, args.home, models) != "completed":
        print("error: the baseline run did not complete", file=sys.stderr)
        return None
    return prepared.run_id


def _publish(comment: str, args: argparse.Namespace) -> bool:
    if args.comment_file:
        args.comment_file.write_text(comment, encoding="utf-8")
    if not args.github:
        return True
    try:
        url = upsert_comment(pull_request_from_env(os.environ, args.pr), comment)
    except GitHubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return False
    print(f"pull request comment: {url}")
    return True
