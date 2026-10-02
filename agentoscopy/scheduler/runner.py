"""Run execution (WF-03 steps 6 to 8): concurrent workers drain the run's queue.

Each worker claims the next trial in dispatch order while the run budget can reserve that
trial's cost cap, runs one attempt, and records it. Retryable infra errors are re-queued with
backoff; when nothing is affordable and nothing is in flight, the rest is skipped.
"""

from __future__ import annotations

import asyncio
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentoscopy.adapters.base import AgentAdapter
from agentoscopy.adapters.python import load_python_adapter
from agentoscopy.gateway.server import Gateway
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.scheduler.plan import PinnedTask
from agentoscopy.spec import AgentConfig
from agentoscopy.storage.store import ClaimedTrial, Store
from agentoscopy.worker.trial import AttemptResult, AttemptSpec, run_attempt

MAX_ATTEMPTS = 2  # infra retries (FR-EXE-07)
RETRY_BACKOFF_S = 5.0
IDLE_POLL_S = 1.0  # upper bound on how long an idle worker sleeps before re-checking
TASK_FLAG_CODES = frozenset({"SETUP_BROKEN", "GRADER_UNSTABLE"})


@dataclass(frozen=True)
class TrialProgress:
    task_id: str
    trial_index: int
    attempt: int
    result: AttemptResult
    retrying: bool
    finished: int
    total: int
    spent_usd: float


ProgressCallback = Callable[[TrialProgress], None]
AdapterFactory = Callable[[str], AgentAdapter]


class RunExecutor:
    def __init__(
        self,
        store: Store,
        run_id: str,
        tasks: list[PinnedTask],
        config: AgentConfig,
        backend: SandboxBackend,
        gateway: Gateway,
        output_root: Path,
        *,
        concurrency: int,
        seed: int,
        adapter_factory: AdapterFactory = load_python_adapter,
        on_progress: ProgressCallback | None = None,
        retry_backoff_s: float = RETRY_BACKOFF_S,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._tasks = {item.task.spec.id: item for item in tasks}
        self._config = config
        self._backend = backend
        self._gateway = gateway
        self._output_root = output_root
        self._concurrency = concurrency
        self._seed = seed
        self._adapter_factory = adapter_factory
        self._on_progress = on_progress
        self._retry_backoff_s = retry_backoff_s
        self._changed = asyncio.Condition()

    async def execute(self) -> None:
        self._store.set_run_status(self._run_id, "running")
        try:
            async with asyncio.TaskGroup() as group:
                for index in range(self._concurrency):
                    group.create_task(self._worker(index))
        except asyncio.CancelledError:
            self._store.cancel_unfinished(self._run_id)
            self._store.set_run_status(self._run_id, "cancelled", finished=True)
            raise
        except BaseException:
            self._store.set_run_status(self._run_id, "failed", finished=True)
            raise
        self._store.set_run_status(self._run_id, "completed", finished=True)

    async def _worker(self, index: int) -> None:
        worker_id = f"{socket.gethostname()}:{os.getpid()}:{index}"
        while (trial := await self._next_trial(worker_id)) is not None:
            await self._run_trial(trial, worker_id)
            async with self._changed:
                self._changed.notify_all()

    async def _next_trial(self, worker_id: str) -> ClaimedTrial | None:
        while True:
            claim = self._store.claim_next(self._run_id, worker_id)
            if claim.trial:
                return claim.trial
            idle = self._store.count_in_flight(self._run_id) == 0
            if idle and claim.status == "empty":
                return None
            if idle and claim.status == "unaffordable":
                if self._store.skip_queued(self._run_id):
                    self._store.add_run_flag(self._run_id, "BUDGET_TRUNCATED")
                return None
            # Wait for a running trial to finish (freeing its reservation or re-queueing it)
            # or for a retry's backoff to expire.
            timeout = IDLE_POLL_S
            if claim.status == "waiting" and claim.ready_at is not None:
                timeout = min(timeout, max(claim.ready_at - time.time(), 0.0))
            async with self._changed:
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout)
                except TimeoutError:
                    pass  # re-check the queue

    async def _run_trial(self, trial: ClaimedTrial, worker_id: str) -> None:
        pinned = self._tasks[trial.task_id]
        spec = AttemptSpec(
            run_id=self._run_id,
            trial_id=trial.trial_id,
            attempt=trial.attempt,
            task=pinned.task,
            config=self._config,
            seed_key=f"{self._seed}:{trial.task_id}:{trial.trial_index}",
            image_digest=pinned.image_digest,
            verified_graders=pinned.verified_graders,
        )
        result = await run_attempt(
            spec,
            self._adapter_factory(self._config.entrypoint),
            self._backend,
            self._gateway,
            self._output_root,
            on_state=lambda state: self._store.set_trial_state(trial, worker_id, state),
        )
        if result.error_code in TASK_FLAG_CODES:
            self._store.add_task_flag(trial.task_id, trial.task_version, result.error_code)
        retrying = (
            result.outcome == "infra_error" and result.retryable and trial.attempt < MAX_ATTEMPTS
        )
        delay = self._retry_backoff_s * trial.attempt if retrying else None
        self._store.finish_attempt(trial, worker_id, result, requeue_delay_s=delay)
        if self._on_progress:
            finished, total = self._store.progress(self._run_id)
            self._on_progress(
                TrialProgress(
                    trial.task_id,
                    trial.trial_index,
                    trial.attempt,
                    result,
                    retrying,
                    finished,
                    total,
                    self._store.run_spent(self._run_id),
                )
            )
