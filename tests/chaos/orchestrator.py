"""The orchestrator process that the crash-injection test kills mid-run.

Usage: python orchestrator.py HOME TASKS_DIR RUN_ID LEASE_S
It executes the run with fake sandboxes and short leases, as `agentoscopy run --resume` would.
"""

import asyncio
import sys
from pathlib import Path

from agentoscopy.gateway.server import Gateway
from agentoscopy.launch import Directories, resume_run
from agentoscopy.sandbox.base import ExecResult
from agentoscopy.scheduler.runner import RunExecutor
from agentoscopy.storage.store import Store
from agentoscopy.testing.fake_sandbox import FakeBackend


def passes_when_fixed(cmd, files):
    return ExecResult(0 if files.get("app.py") == b"fixed" else 1, b"", b"")


async def execute(store: Store, home: Path, tasks_dir: Path, run_id: str, lease_s: float) -> None:
    prepared = resume_run(
        store, Directories(tasks_dir, tasks_dir, home), run_id, has_credentials=False
    )
    gateway = Gateway()
    await gateway.start()
    try:
        await RunExecutor(
            store,
            run_id,
            prepared.pinned,
            prepared.config,
            FakeBackend(grader=passes_when_fixed),
            gateway,
            home,
            concurrency=prepared.settings.concurrency,
            seed=prepared.settings.seed,
            lease_s=lease_s,
            heartbeat_s=lease_s / 4,
        ).execute()
    finally:
        await gateway.stop()


if __name__ == "__main__":
    home, tasks_dir, run_id, lease_s = sys.argv[1:]
    store = Store(Path(home) / "agentoscopy.db")
    asyncio.run(execute(store, Path(home), Path(tasks_dir), run_id, float(lease_s)))
