"""`agentoscopy task swebench ROWS INSTANCE_ID...`: tasks from SWE-bench rows (see
agentoscopy.swebench_import)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from agentoscopy.cli.common import EXIT_ENVIRONMENT, EXIT_INVALID_INPUT, EXIT_OK
from agentoscopy.swebench_import import (
    SwebenchImportError,
    import_instance,
    read_rows,
    registry_digest,
)


def swebench_command(args: argparse.Namespace) -> int:
    try:
        rows = read_rows(args.rows)
        wanted = _wanted(args, rows)
    except SwebenchImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    written: list[str] = []
    failed = 0
    for instance_id in wanted:
        try:
            imported = import_instance(
                rows[instance_id], args.tasks_dir, registry_digest, tuple(args.tag)
            )
        except SwebenchImportError as exc:
            print(f"error: {instance_id}: {exc}", file=sys.stderr)
            failed += 1
            continue
        except OSError as exc:
            print(f"error: {instance_id}: cannot write the task: {exc}", file=sys.stderr)
            return EXIT_ENVIRONMENT
        written.append(imported.path.name)
        print(
            f"wrote {imported.path}: {imported.fail_to_pass} FAIL_TO_PASS and "
            f"{imported.pass_to_pass} PASS_TO_PASS tests"
        )
    if written:
        print(f"next: agentoscopy task validate {' '.join(written)}")
    return EXIT_INVALID_INPUT if failed else EXIT_OK


def _wanted(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> list[str]:
    if bool(args.instance_ids) == args.all:
        raise SwebenchImportError("name the instances to import, or pass --all")
    wanted = list(rows) if args.all else list(dict.fromkeys(args.instance_ids))
    unknown = [instance_id for instance_id in wanted if instance_id not in rows]
    if unknown:
        raise SwebenchImportError(f"not in {args.rows}: {', '.join(unknown)}")
    return wanted
