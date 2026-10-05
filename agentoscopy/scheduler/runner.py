"""Run execution (WF-03 steps 6 to 8): concurrent workers drain the run's queue.

Each worker claims the next trial in dispatch order while the run budget can reserve that
trial's cost cap, runs one attempt, and records it. Retryable infra errors are re-queued with
backoff; when nothing is affordable and nothing is in flight, the rest is skipped.

A control loop beside the workers renews this process's leases, reclaims trials whose
process died (which is also how a resumed run picks up after a crash), and stops the
workers' trials when the run is cancelled (WF-12).
"""

from __future__ import annotations

import asyncio
import os
import random
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentoscopy.adapters.base import AgentAdapter
from agentoscopy.adapters.python import load_python_adapter
from agentoscopy.gateway.server import Gateway
from agentoscopy.replay import RecordedCall
from agentoscopy.sandbox.base import SandboxBackend
from agentoscopy.scheduler.plan import PinnedTask
from agentoscopy.scheduler.recovery import MAX_ATTEMPTS, reclaim_expired
from agentoscopy.spec import AgentConfig
from agentoscopy.storage.store import LEASE_S, ClaimedTrial, Store
from agentoscopy.worker.trial import AttemptResult, AttemptSpec, run_attempt

RETRY_BACKOFF_S = 5.0
HEARTBEAT_S = 10.0  # lease renewal and reclaiming, well within LEASE_S
CONTROL_POLL_S = 1.0  # how soon a cancel request is noticed
IDLE_POLL_S = 1.0  # upper bound on how long an idle worker sleeps before re-checking
TASK_FLAG_CODES = frozenset({"SETUP_BROKEN", "GRADER_UNSTABLE"})
REVIEW_RATE = 0.05  # share of finished trials sampled at random for human review (FR-GRD-08)
LOW_CONFIDENCE = 0.7  # judge verdicts below this confidence always go to review


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
        judge_model: str | None = None,
        review_rate: float = REVIEW_RATE,
        lease_s: float = LEASE_S,
        heartbeat_s: float = HEARTBEAT_S,
        replay: tuple[RecordedCall, ...] | None = None,
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
        self._judge_model = judge_model
        self._review_rate = review_rate
        self._lease_s = lease_s
        self._replay = replay
        self._heartbeat_s = heartbeat_s
        self._changed = asyncio.Condition()
        self._cancel = asyncio.Event()
        process = f"{socket.gethostname()}:{os.getpid()}"
        self._worker_ids = [f"{process}:{index}" for index in range(concurrency)]

    async def execute(self) -> None:
        self._store.mark_running(self._run_id)
        try:
            async with asyncio.TaskGroup() as group:
                control = group.create_task(self._control())
                workers = [group.create_task(self._worker(id_)) for id_ in self._worker_ids]
                await asyncio.wait(workers)  # a failing worker cancels this wait
                control.cancel()
        except asyncio.CancelledError:
            self._store.cancel_unfinished(self._run_id)
            self._store.set_run_status(self._run_id, "cancelled", finished=True)
            raise
        except BaseException:
            self._store.set_run_status(self._run_id, "failed", finished=True)
            raise
        if not self._store.finalize_cancel(self._run_id):
            self._store.set_run_status(self._run_id, "completed", finished=True)

    async def _control(self) -> None:
        last_beat = -self._heartbeat_s
        while True:
            if not self._cancel.is_set() and self._store.run_status(self._run_id) == "cancelling":
                self._cancel.set()  # attempts not yet grading stop (WF-12)
                await self._wake_workers()
            if time.monotonic() - last_beat >= self._heartbeat_s:
                last_beat = time.monotonic()
                until = time.time() + self._lease_s
                self._store.renew_leases(self._run_id, self._worker_ids, until)
                released = await reclaim_expired(
                    self._store, self._backend, self._output_root, self._run_id
                )
                if released:
                    await self._wake_workers()
            await asyncio.sleep(min(CONTROL_POLL_S, self._heartbeat_s))

    async def _wake_workers(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    async def _worker(self, worker_id: str) -> None:
        while (trial := await self._next_trial(worker_id)) is not None:
            await self._run_trial(trial, worker_id)
            await self._wake_workers()

    async def _next_trial(self, worker_id: str) -> ClaimedTrial | None:
        while True:
            claim = self._store.claim_next(self._run_id, worker_id, lease_s=self._lease_s)
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
            judge_model=self._judge_model,
            replay=self._replay,
        )
        result = await run_attempt(
            spec,
            self._adapter_factory(self._config.entrypoint),
            self._backend,
            self._gateway,
            self._output_root,
            on_state=lambda state: self._store.set_trial_state(trial, worker_id, state),
            cancel=self._cancel,
        )
        if result.error_code in TASK_FLAG_CODES:
            self._store.add_task_flag(trial.task_id, trial.task_version, result.error_code)
        retrying = (
            result.outcome == "infra_error" and result.retryable and trial.attempt < MAX_ATTEMPTS
        )
        delay = self._retry_backoff_s * trial.attempt if retrying else None
        finished = self._store.finish_attempt(trial, worker_id, result, requeue_delay_s=delay)
        if finished and result.outcome in ("pass", "fail"):
            self._after_grading(trial, result)
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

    def _after_grading(self, trial: ClaimedTrial, result: AttemptResult) -> None:
        if result.vetoed:
            self._store.add_failure_tag(
                trial.trial_id, "reward_hacking", "auto", "tamper_check vetoed the trial"
            )
        judges = [grade for grade in result.grades if grade.kind == "judge"]
        for grade in judges:
            if grade.metadata.get("confidence", 1.0) < LOW_CONFIDENCE:
                self._store.queue_review(
                    trial.trial_id, trial.attempt, grade.name, "low_confidence"
                )
        # Seeded per trial, so the same run samples the same trials.
        if random.Random(f"{self._seed}:review:{trial.trial_id}").random() < self._review_rate:
            for name in [grade.name for grade in judges] or [""]:
                self._store.queue_review(trial.trial_id, trial.attempt, name, "random")
