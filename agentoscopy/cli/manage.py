"""`agentoscopy cancel` and `agentoscopy gc`."""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from agentoscopy.cli.common import DB_NAME, EXIT_ENVIRONMENT, EXIT_INVALID_INPUT, EXIT_OK
from agentoscopy.housekeeping import collect_garbage
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.scheduler.recovery import cancel_run
from agentoscopy.storage.store import Store


def cancel_command(args: argparse.Namespace) -> int:
    store = Store(args.home / DB_NAME)
    try:
        status = asyncio.run(cancel_run(store, DockerBackend(), args.home, args.run_id))
    finally:
        store.close()
    if status is None:
        print(f"error: no run {args.run_id}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    if status == "cancelling":
        print(f"run {args.run_id}: cancelling; the process running it stops its trials shortly")
        return EXIT_OK
    if status == "cancelled":
        print(f"run {args.run_id}: cancelled")
        return EXIT_OK
    print(f"error: run {args.run_id} already finished ({status})", file=sys.stderr)
    return EXIT_INVALID_INPUT


def gc_command(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    store = Store(args.home / DB_NAME)
    try:
        report = asyncio.run(
            collect_garbage(
                store, DockerBackend(), args.home, args.older_than, dry_run=args.dry_run, now=now
            )
        )
    finally:
        store.close()
    verb = "would remove" if args.dry_run else "removed"
    megabytes = report.freed_bytes / 1_000_000
    print(
        f"{verb} the trajectories and artifacts of {len(report.cleared_runs)} runs that "
        f"finished before {now - args.older_than:%Y-%m-%d %H:%M} UTC ({megabytes:.1f} MB)"
    )
    for run_id, reason in report.kept_runs:
        print(f"kept run {run_id}: {reason}")
    print(f"{verb} {len(report.removed_sandboxes)} sandboxes that no running trial holds")
    for error in report.errors:
        print(f"warning: {error}", file=sys.stderr)
    return EXIT_ENVIRONMENT if report.errors else EXIT_OK
