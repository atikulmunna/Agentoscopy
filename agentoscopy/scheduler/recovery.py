"""Crash recovery and cancellation (WF-12).

A trial in flight is held under a lease that its process renews. When the process dies, the
lease runs out, and any process working on the run can reclaim the trial: remove what the
dead attempt left in the sandbox backend, keep what it spent, and retry the trial.
"""

from __future__ import annotations

from pathlib import Path

from agentoscopy.reporting import run_summary
from agentoscopy.sandbox.base import SandboxBackend, SandboxError
from agentoscopy.storage.store import Store
from agentoscopy.worker.trial import recorded_spend

MAX_ATTEMPTS = 2  # infra retries (FR-EXE-07)


async def reclaim_expired(
    store: Store,
    backend: SandboxBackend,
    output_root: Path,
    run_id: str,
    now: float | None = None,
) -> list[str]:
    """Release the run's trials whose lease ran out (FR-EXE-06). Each is retried, closed as an
    infra error after its last attempt, or cancelled if the run is being cancelled. Returns the
    trial ids released."""
    cancelling = store.run_status(run_id) == "cancelling"
    released = []
    for trial in store.expired_trials(run_id, now):
        agent_usd, judge_usd = recorded_spend(output_root, run_id, trial.trial_id, trial.attempt)
        if cancelling:
            state = "CANCELLED"
        else:
            state = "QUEUED" if trial.attempt < MAX_ATTEMPTS else "INFRA_ERROR"
        abandoned = store.abandon_attempt(
            trial, state=state, cost_usd=agent_usd, judge_cost_usd=judge_usd, now=now
        )
        if not abandoned:
            continue  # its process renewed the lease after all
        released.append(trial.trial_id)
        try:
            await backend.remove_trial_sandboxes(trial.trial_id)
        except SandboxError:
            pass  # `agentoscopy gc` removes what is left
    return released


async def cancel_run(
    store: Store, backend: SandboxBackend, output_root: Path, run_id: str
) -> str | None:
    """Ask a run to stop. Trials whose process has died are cancelled here, so a run with no
    process left still ends; live processes stop their own trials. Returns the run's status
    (None if there is no such run)."""
    status = store.request_cancel(run_id)
    if status == "cancelling":
        await reclaim_expired(store, backend, output_root, run_id)
        store.finalize_cancel(run_id)
        status = store.run_status(run_id)
    if status == "cancelled" and store.get_summary(run_id) is None:
        store.save_summary(run_id, run_summary(store, run_id, refresh=True))
    return status
