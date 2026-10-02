"""`agentoscopy task validate`: run the WF-01 checks and record validated task versions."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_CHECK_FAILED,
    EXIT_ENVIRONMENT,
    EXIT_INVALID_INPUT,
    EXIT_OK,
    all_task_ids,
)
from agentoscopy.gateway.credential_proxy import CredentialProxy
from agentoscopy.gateway.server import Gateway
from agentoscopy.graders.judge import GatewayJudge
from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.plan import PlanError, check_judge
from agentoscopy.spec import SpecError, Task, load_task
from agentoscopy.storage.store import Store
from agentoscopy.validation import CHECK_FAILURE_CODES, ValidationReport, validate_task


def validate_command(args: argparse.Namespace) -> int:
    task_ids = all_task_ids(args.tasks_dir) if args.all else args.task_ids
    if not task_ids:
        print("error: name one or more task ids, or pass --all", file=sys.stderr)
        return EXIT_INVALID_INPUT
    try:
        proxy = CredentialProxy.start_if_configured()  # only judges call a real model here
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    store = Store(args.home / DB_NAME)
    worst = EXIT_OK
    try:
        for task_id in task_ids:
            worst = max(worst, _validate_one(store, args, task_id, proxy))
    finally:
        store.close()
        if proxy:
            proxy.stop()
    return worst


def _validate_one(
    store: Store, args: argparse.Namespace, task_id: str, proxy: CredentialProxy | None
) -> int:
    try:
        task = load_task(args.tasks_dir, task_id)
        judge_model = check_judge([task], args.judge_model, has_credentials=proxy is not None)
    except (SpecError, PlanError) as exc:
        code = "" if isinstance(exc, PlanError) else "SCHEMA_INVALID "
        print(f"FAIL  {task_id}: {code}{exc}")
        return EXIT_INVALID_INPUT
    report, judge_cost = asyncio.run(_validate(task, judge_model, proxy, args.home))
    for problem in report.cleanup_errors:
        print(f"warning: {problem}", file=sys.stderr)
    if not report.ok:
        print(f"FAIL  {task_id}: {report.error_code} {report.message}")
        return EXIT_CHECK_FAILED if report.error_code in CHECK_FAILURE_CODES else EXIT_ENVIRONMENT
    version = store.save_task_version(
        task, report.image_digest, report.flags, report.verified_graders
    )
    notes = f" ({', '.join(report.flags)})" if report.flags else ""
    spend = f", judge spend ${judge_cost:.4f}" if judge_model else ""
    print(
        f"ok    {task_id}: version {version}, verified graders {report.verified_graders}"
        f"{notes}{spend}"
    )
    return EXIT_OK


async def _validate(
    task: Task, judge_model: str | None, proxy: CredentialProxy | None, home: Path
) -> tuple[ValidationReport, float]:
    """The report and the judge spend. Judge calls go through a gateway like a run's do."""
    if judge_model is None:
        return await validate_task(task, DockerBackend()), 0.0
    gateway = Gateway(proxy.url if proxy else None, proxy.secret if proxy else None)
    await gateway.start()
    log_dir = home / "validation" / task.spec.id / task.content_hash[:12]
    recorder = TrajectoryRecorder(
        log_dir / "grading.jsonl", log_dir / "grading", f"validate-{task.spec.id}", 1
    )
    judge = GatewayJudge(gateway, judge_model, recorder, f"validate:{task.content_hash}")
    try:
        return await validate_task(task, DockerBackend(), judge), judge.cost_usd
    finally:
        judge.close()
        await gateway.stop()
