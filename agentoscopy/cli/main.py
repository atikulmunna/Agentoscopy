"""`agentoscopy` command-line interface.

Exit codes: 0 success; 1 a task failed a validation check; 2 invalid input (spec, config, or
plan); 3 environment or harness failure; 130 the run was cancelled. `agentoscopy ci` has
its own codes (see agentoscopy.cli.ci).
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from datetime import timedelta
from pathlib import Path

from aiohttp import web

from agentoscopy.api.runs import RunService
from agentoscopy.api.server import create_app
from agentoscopy.cli.agent import agent_check_command
from agentoscopy.cli.ci import ci_command
from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_INVALID_INPUT,
    EXIT_OK,
    all_task_ids,
)
from agentoscopy.cli.manage import cancel_command, gc_command
from agentoscopy.cli.run import replay_command, run_command, start_proxy
from agentoscopy.cli.validate import validate_command
from agentoscopy.launch import DEFAULT_JUDGE_MODEL, Directories
from agentoscopy.reporting import NotFound, comparison, run_summary
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.runner import REVIEW_RATE
from agentoscopy.spec import SpecError, load_task
from agentoscopy.stats.compare import CompareError
from agentoscopy.stats.report import FORMATS, render, render_comparison
from agentoscopy.storage.store import Store

TOKEN_ENV = "AGENTOSCOPY_API_TOKEN"  # a fixed API token, e.g. for a Prometheus scraper
DURATION_PATTERN = re.compile(r"^(\d+)([smhd])$")
DURATION_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Absolute, so stored artifact paths stay valid when commands run from other directories.
    args.home = args.home.resolve()
    return args.handler(args)


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--home",
        type=Path,
        default=Path(".agentoscopy"),
        help="results database, trajectories, and artifacts",
    )
    common.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    common.add_argument("--suites-dir", type=Path, default=Path("suites"))
    run_options = _run_options()

    parser = argparse.ArgumentParser(
        prog="agentoscopy", description="Evaluation harness for LLM agents."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    task = commands.add_parser("task", help="validate and list tasks")
    task_commands = task.add_subparsers(dest="task_command", required=True)
    validate = task_commands.add_parser("validate", parents=[common], help="run WF-01 checks")
    validate.add_argument("task_ids", nargs="*", metavar="TASK_ID")
    validate.add_argument("--all", action="store_true", help="validate every task")
    validate.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="pinned model for llm_judge graders"
    )
    validate.set_defaults(handler=validate_command)
    listing = task_commands.add_parser("list", parents=[common], help="tasks and validation status")
    listing.set_defaults(handler=_list_command)

    agent = commands.add_parser("agent", help="check agent configs")
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    check = agent_commands.add_parser(
        "check", parents=[common], help="try a config on the built-in smoke task (WF-02)"
    )
    check.add_argument("config", type=Path, help="agent config YAML file")
    check.set_defaults(handler=agent_check_command)

    run = commands.add_parser(
        "run", parents=[common, run_options], help="run a suite or tasks with an agent"
    )
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite", help="suite name (suites/<name>.yaml)")
    selection.add_argument(
        "--task",
        action="append",
        dest="tasks",
        metavar="TASK_ID",
        help="task id; repeat for several",
    )
    selection.add_argument(
        "--resume", metavar="RUN_ID", help="continue an unfinished run, e.g. after a crash"
    )
    selecting = run.add_argument_group("choosing tasks (FR-RUN-02); repeat an option for any of")
    selecting.add_argument("--tag", action="append", metavar="TAG")
    selecting.add_argument("--category", action="append", metavar="CATEGORY")
    selecting.add_argument("--difficulty", action="append", choices=("easy", "medium", "hard"))
    selecting.add_argument(
        "--failed-in", metavar="RUN_ID", help="only tasks that failed in this run"
    )
    run.set_defaults(handler=run_command)

    replay = commands.add_parser(
        "replay", parents=[common], help="rerun a trial from its recorded model calls (WF-11)"
    )
    replay.add_argument("trial_id")
    replay.set_defaults(handler=replay_command)

    cancel = commands.add_parser("cancel", parents=[common], help="cancel a run (WF-12)")
    cancel.add_argument("run_id")
    cancel.set_defaults(handler=cancel_command)

    gc = commands.add_parser(
        "gc", parents=[common], help="delete old run outputs and orphaned sandboxes"
    )
    gc.add_argument(
        "--older-than",
        type=_duration,
        default=_duration("30d"),
        metavar="AGE",
        help="age of finished runs whose outputs go, like 30d or 12h (default 30d)",
    )
    gc.add_argument("--dry-run", action="store_true", help="report without deleting")
    gc.set_defaults(handler=gc_command)

    ci = commands.add_parser(
        "ci", parents=[common, run_options], help="run, compare with a baseline, report (WF-10)"
    )
    ci.add_argument("--suite", required=True, help="suite name (suites/<name>.yaml)")
    ci.add_argument(
        "--baseline",
        required=True,
        metavar="BRANCH",
        help="compare with the latest completed run labelled branch=BRANCH",
    )
    ci.add_argument(
        "--baseline-agent",
        type=Path,
        help="agent config for running the baseline when no recent one exists",
    )
    ci.add_argument(
        "--max-baseline-age",
        type=_duration,
        default=_duration("7d"),
        metavar="AGE",
        help="an older baseline is stale and is run again (default 7d)",
    )
    ci.add_argument(
        "--override",
        action="store_true",
        help="report a regression without failing (the eval-override pull request label)",
    )
    ci.add_argument("--comment-file", type=Path, help="also write the comment here")
    ci.add_argument("--github", action="store_true", help="post the comment on the pull request")
    ci.add_argument("--pr", type=_positive_int, help="pull request number (default: this event's)")
    ci.set_defaults(handler=ci_command, tasks=None, resume=None)

    report = commands.add_parser("report", parents=[common], help="print a run summary")
    report.add_argument("run_id")
    report.add_argument("--format", choices=FORMATS, default="table")
    report.set_defaults(handler=_report_command)

    compare = commands.add_parser("compare", parents=[common], help="compare two runs (WF-07)")
    compare.add_argument("baseline", help="baseline run id")
    compare.add_argument("candidate", help="candidate run id")
    compare.add_argument("--format", choices=FORMATS, default="table")
    compare.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="also compare tasks whose versions differ between the runs",
    )
    compare.set_defaults(handler=_compare_command)

    serve = commands.add_parser("serve", parents=[common], help="REST API and web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8321)
    serve.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="agent configs that POST /runs may name",
    )
    serve.set_defaults(handler=_serve_command)
    return parser


def _run_options() -> argparse.ArgumentParser:
    """Options shared by `run` and `ci`."""
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--agent", type=Path, help="agent config YAML file")
    options.add_argument("--trials", type=_positive_int, default=3, help="trials per task")
    options.add_argument("--budget-usd", type=float, default=10.0, help="run-level cost ceiling")
    options.add_argument(
        "--concurrency", type=_positive_int, help="trials at once (default 4, or as before)"
    )
    options.add_argument(
        "--seed", type=int, help="dispatch order and mock responses (default random)"
    )
    options.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    options.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="pinned model for llm_judge graders"
    )
    options.add_argument(
        "--review-rate",
        type=_fraction,
        default=REVIEW_RATE,
        help="fraction of trials sampled for human review",
    )
    return options


def _list_command(args: argparse.Namespace) -> int:
    store = Store(args.home / DB_NAME)
    try:
        for task_id in all_task_ids(args.tasks_dir):
            print(_task_status(store, args.tasks_dir, task_id))
    finally:
        store.close()
    return EXIT_OK


def _task_status(store: Store, tasks_dir: Path, task_id: str) -> str:
    try:
        task = load_task(tasks_dir, task_id)
    except SpecError:
        return f"{task_id}  invalid spec"
    version = store.task_version(task_id, task.content_hash)
    if version is None or version.validated_at is None:
        return f"{task_id}  not validated (changed or new)"
    flags = f"  {', '.join(version.flags)}" if version.flags else ""
    return f"{task_id}  version {version.version}, validated {version.validated_at}{flags}"


def _report_command(args: argparse.Namespace) -> int:
    store = Store(args.home / DB_NAME)
    try:
        run = store.get_run(args.run_id)
        if run is None:
            print(f"error: no run {args.run_id}", file=sys.stderr)
            return EXIT_INVALID_INPUT
        print(render(run_summary(store, args.run_id), run, args.format))
    finally:
        store.close()
    return EXIT_OK


def _compare_command(args: argparse.Namespace) -> int:
    store = Store(args.home / DB_NAME)
    try:
        result = comparison(
            store, args.baseline, args.candidate, allow_version_drift=args.allow_version_drift
        )
    except (NotFound, CompareError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    finally:
        store.close()
    print(render_comparison(result, args.format))
    return EXIT_OK


def _serve_command(args: argparse.Namespace) -> int:
    proxy, failure = start_proxy()  # runs started over the API may call real models
    if failure is not None:
        return failure
    store = Store(args.home / DB_NAME)
    token = os.environ.get(TOKEN_ENV) or secrets.token_urlsafe(24)
    runs = RunService(
        Directories(args.tasks_dir.resolve(), args.suites_dir.resolve(), args.home),
        args.configs_dir.resolve(),
        DockerBackend(),
        proxy,
    )
    app = create_app(store, args.home, token, frozenset({args.host}), runs=runs)
    print(f"Agentoscopy UI: http://{args.host}:{args.port}/#token={token}", flush=True)
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        store.close()
        if proxy:
            proxy.stop()
    return EXIT_OK


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _duration(text: str) -> timedelta:
    match = DURATION_PATTERN.match(text.strip())
    if not match:
        raise argparse.ArgumentTypeError("must be a number and a unit: s, m, h, or d (e.g. 30d)")
    return timedelta(**{DURATION_UNITS[match.group(2)]: int(match.group(1))})


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value
