"""Sandbox interfaces shared by the worker, adapters, graders, and backends."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from agentoscopy.spec import Task

HIDDEN_MOUNT = "/hidden"  # where grader-only files appear, in grading sandboxes only


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False


@dataclass(frozen=True)
class SandboxRecord:
    """A container or snapshot image that belongs to a trial, as the backend reports it."""

    kind: str  # container | image
    ref: str
    trial_id: str
    created: datetime


class SandboxError(Exception):
    """A sandbox operation failed: the container runtime itself, or a file read or write."""


class Sandbox(Protocol):
    """What adapters and graders may do inside a sandbox (§7.3)."""

    @property
    def workdir(self) -> str: ...

    async def exec(self, cmd: str, timeout_s: int = 60) -> ExecResult: ...

    async def read_file(self, path: str) -> bytes: ...

    async def write_file(self, path: str, data: bytes) -> None: ...


class ManagedSandbox(Sandbox, Protocol):
    """A sandbox plus the lifecycle operations only the worker uses."""

    async def diff(self) -> str:
        """Whole-container filesystem changes since the sandbox started (FR-SBX-06)."""
        ...

    async def stop(self) -> None: ...

    async def snapshot(self) -> str:
        """Capture the stopped sandbox's final state; returns a backend-specific reference."""
        ...

    async def destroy(self) -> None: ...


class SandboxBackend(Protocol):
    async def prepare_image(self, task: Task) -> str:
        """Return the task image for this task version, building it once if needed (AD-6)."""
        ...

    async def create(self, image: str, task: Task, trial_id: str) -> ManagedSandbox:
        """Start a fresh agent sandbox from the task image."""
        ...

    async def create_grading(self, snapshot: str, task: Task, trial_id: str) -> ManagedSandbox:
        """Start a grading sandbox from a snapshot, with hidden/ mounted read-only (AD-4)."""
        ...

    async def remove_trial_sandboxes(self, trial_id: str) -> None:
        """Remove everything a trial left behind after its process died (FR-EXE-06)."""
        ...

    async def list_sandboxes(self) -> list[SandboxRecord]:
        """Every sandbox and snapshot that belongs to some trial, for garbage collection."""
        ...

    async def remove_sandbox(self, record: SandboxRecord) -> None: ...
