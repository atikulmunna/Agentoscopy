"""`agentoscopy task import <repo> <commit> --id ID`: a task from a fix in a repository's
history (see agentoscopy.task_import)."""

from __future__ import annotations

import argparse
import sys

from agentoscopy.cli.common import EXIT_ENVIRONMENT, EXIT_INVALID_INPUT, EXIT_OK
from agentoscopy.task_import import ImportOptions, TaskImportError, import_task


def import_command(args: argparse.Namespace) -> int:
    options = ImportOptions(
        task_id=args.id,
        instructions=args.instructions,
        base_image=args.base_image,
        runner=args.runner,
        build=tuple(args.build),
        tags=tuple(args.tag),
    )
    try:
        imported = import_task(args.repo, args.commit, args.tasks_dir, options)
    except TaskImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except OSError as exc:
        print(f"error: cannot write the task: {exc}", file=sys.stderr)
        return EXIT_ENVIRONMENT
    fix = imported.fix
    print(f"wrote {imported.path} from commit {fix.commit[:12]}")
    print(f"  starting code: {imported.fixture_files} files from its parent {fix.parent[:12]}")
    print(f"  hidden tests: {', '.join(fix.tests)}")
    print(f"  reference fix: {', '.join(fix.code)}")
    if imported.skipped:
        print(f"  skipped, not regular files: {', '.join(imported.skipped)}")
    print(f"next: agentoscopy task validate {args.id}")
    return EXIT_OK
