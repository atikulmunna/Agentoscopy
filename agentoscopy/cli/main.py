"""`agentoscopy` command-line interface.

Exit codes: 0 success; 1 a task failed a validation check; 2 invalid input (spec, config, or
plan); 3 environment or harness failure.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
from pathlib import Path

from aiohttp import web

from agentoscopy.api.server import create_app
from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_CHECK_FAILED,
    EXIT_ENVIRONMENT,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.cli.run import run_command
from agentoscopy.reporting import NotFound, comparison, run_summary
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.spec import SpecError, load_task
from agentoscopy.stats.compare import CompareError
from agentoscopy.stats.report import FORMATS, render, render_comparison
from agentoscopy.storage.store import Store
from agentoscopy.validation import CHECK_FAILURE_CODES, validate_task


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

    parser = argparse.ArgumentParser(
        prog="agentoscopy", description="Evaluation harness for LLM agents."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    task = commands.add_parser("task", help="validate and list tasks")
    task_commands = task.add_subparsers(dest="task_command", required=True)
    validate = task_commands.add_parser("validate", parents=[common], help="run WF-01 checks")
    validate.add_argument("task_ids", nargs="*", metavar="TASK_ID")
    validate.add_argument("--all", action="store_true", help="validate every task")
    validate.set_defaults(handler=_validate_command)
    listing = task_commands.add_parser("list", parents=[common], help="tasks and validation status")
    listing.set_defaults(handler=_list_command)

    run = commands.add_parser("run", parents=[common], help="run a suite or tasks with an agent")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite", help="suite name (suites/<name>.yaml)")
    selection.add_argument(
        "--task",
        action="append",
        dest="tasks",
        metavar="TASK_ID",
        help="task id; repeat for several",
    )
    run.add_argument("--agent", required=True, type=Path, help="agent config YAML file")
    run.add_argument("--trials", type=_positive_int, default=3, help="trials per task")
    run.add_argument("--budget-usd", type=float, default=10.0, help="run-level cost ceiling")
    run.add_argument("--concurrency", type=_positive_int, default=4)
    run.add_argument("--seed", type=int, help="dispatch order and mock responses (default random)")
    run.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    run.set_defaults(handler=run_command)

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
    serve.set_defaults(handler=_serve_command)
    return parser


def _validate_command(args: argparse.Namespace) -> int:
    task_ids = _all_task_ids(args.tasks_dir) if args.all else args.task_ids
    if not task_ids:
        print("error: name one or more task ids, or pass --all", file=sys.stderr)
        return EXIT_INVALID_INPUT
    store = Store(args.home / DB_NAME)
    worst = EXIT_OK
    try:
        for task_id in task_ids:
            worst = max(worst, _validate_one(store, args.tasks_dir, task_id))
    finally:
        store.close()
    return worst


def _validate_one(store: Store, tasks_dir: Path, task_id: str) -> int:
    try:
        task = load_task(tasks_dir, task_id)
    except SpecError as exc:
        print(f"FAIL  {task_id}: SCHEMA_INVALID {exc}")
        return EXIT_INVALID_INPUT
    report = asyncio.run(validate_task(task, DockerBackend()))
    for problem in report.cleanup_errors:
        print(f"warning: {problem}", file=sys.stderr)
    if not report.ok:
        print(f"FAIL  {task_id}: {report.error_code} {report.message}")
        return EXIT_CHECK_FAILED if report.error_code in CHECK_FAILURE_CODES else EXIT_ENVIRONMENT
    version = store.save_task_version(
        task, report.image_digest, report.flags, report.verified_graders
    )
    notes = f" ({', '.join(report.flags)})" if report.flags else ""
    print(f"ok    {task_id}: version {version}, verified graders {report.verified_graders}{notes}")
    return EXIT_OK


def _list_command(args: argparse.Namespace) -> int:
    store = Store(args.home / DB_NAME)
    try:
        for task_id in _all_task_ids(args.tasks_dir):
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
    store = Store(args.home / DB_NAME)
    token = secrets.token_urlsafe(24)
    app = create_app(store, args.home, token, frozenset({args.host}))
    print(f"Agentoscopy UI: http://{args.host}:{args.port}/#token={token}", flush=True)
    try:
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        store.close()
    return EXIT_OK


def _all_task_ids(tasks_dir: Path) -> list[str]:
    if not tasks_dir.is_dir():
        return []
    return sorted(path.name for path in tasks_dir.iterdir() if (path / "task.yaml").is_file())


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value
