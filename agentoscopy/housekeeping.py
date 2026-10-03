"""Garbage collection (`agentoscopy gc`): outputs of old runs, and sandboxes no trial is using."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agentoscopy.sandbox.base import SandboxBackend, SandboxError, SandboxRecord
from agentoscopy.storage.store import Store

ARTIFACTS_DELETED = "ARTIFACTS_DELETED"  # run flag: trajectories and artifacts were removed
# Younger sandboxes are left alone: one may belong to a trial being set up, or to a validation.
SANDBOX_GRACE = timedelta(minutes=10)
OUTPUT_DIRS = ("trajectories", "artifacts")


@dataclass
class GcReport:
    cleared_runs: list[str] = field(default_factory=list)
    kept_runs: list[tuple[str, str]] = field(default_factory=list)  # (run id, reason)
    freed_bytes: int = 0
    removed_sandboxes: list[SandboxRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def collect_garbage(
    store: Store,
    backend: SandboxBackend,
    home: Path,
    older_than: timedelta,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> GcReport:
    """Delete the trajectories and artifacts of runs that finished before the cutoff (their
    results stay in the database), and remove sandboxes that no live trial holds."""
    now = now or datetime.now(UTC)
    report = GcReport()
    for run in store.runs_finished_before(now - older_than):
        run_id = run["run_id"]
        if ARTIFACTS_DELETED in run["flags"]:
            continue
        if store.pending_reviews_for_run(run_id):
            report.kept_runs.append((run_id, "it has trials waiting for review"))
            continue
        for directory in run_output_dirs(home, run_id):
            report.freed_bytes += sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
            if not dry_run:
                shutil.rmtree(directory)
        if not dry_run:
            store.add_run_flag(run_id, ARTIFACTS_DELETED)
        report.cleared_runs.append(run_id)
    await _remove_orphans(store, backend, report, dry_run, now)
    return report


def run_output_dirs(home: Path, run_id: str) -> list[Path]:
    """A run's existing output directories; a malformed run id can never point elsewhere."""
    found = []
    for name in OUTPUT_DIRS:
        parent = (home / name).resolve()
        directory = (parent / run_id).resolve()
        if directory.parent == parent and directory.is_dir():
            found.append(directory)
    return found


async def _remove_orphans(
    store: Store, backend: SandboxBackend, report: GcReport, dry_run: bool, now: datetime
) -> None:
    live = store.live_trial_ids(now.timestamp())
    try:
        records = await backend.list_sandboxes()
    except SandboxError as exc:
        report.errors.append(f"cannot list sandboxes: {exc}")
        return
    for record in records:
        if record.trial_id in live or record.created > now - SANDBOX_GRACE:
            continue
        try:
            if not dry_run:
                await backend.remove_sandbox(record)
            report.removed_sandboxes.append(record)
        except SandboxError as exc:
            report.errors.append(str(exc))
