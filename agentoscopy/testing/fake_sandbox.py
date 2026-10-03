"""In-memory sandbox and backend for testing adapters, the worker, and the scheduler without Docker.

Files live in a dict. A grading sandbox starts with a copy of the snapshotted sandbox's files,
and its exec answers come from a `GradingHandler` that can inspect them, so a fake grader can
pass or fail depending on what the agent wrote.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from datetime import UTC, datetime

from agentoscopy.sandbox.base import ExecResult, SandboxError, SandboxRecord
from agentoscopy.spec import Task

ExecHandler = Callable[[str], ExecResult]
GradingHandler = Callable[[str, dict[str, bytes]], ExecResult]


def exit_with(code: int, stdout: bytes = b"", stderr: bytes = b"") -> ExecHandler:
    return lambda cmd: ExecResult(code, stdout, stderr)


class FakeSandbox:
    """Calls are kept for assertions. Without a handler, exec succeeds, except that
    `test -e PATH` reports whether PATH exists among the files."""

    def __init__(
        self,
        handler: ExecHandler | None = None,
        files: dict[str, bytes] | None = None,
        workdir: str = "/workspace",
        on_destroy: Callable[[], None] | None = None,
    ) -> None:
        self.files = dict(files or {})
        self._initial = dict(self.files)
        self.commands: list[str] = []
        self.stopped = False
        self.destroyed = False
        self.destroy_error: str | None = None
        self.snapshot_id = f"fake-snapshot-{id(self)}"
        self.created = datetime.now(UTC)
        self._handler = handler
        self._workdir = workdir
        self._on_destroy = on_destroy

    @property
    def workdir(self) -> str:
        return self._workdir

    async def exec(self, cmd: str, timeout_s: int = 60) -> ExecResult:
        self.commands.append(cmd)
        if self._handler:
            return self._handler(cmd)
        if cmd.startswith("test -e "):
            path = shlex.split(cmd)[2]
            exists = any(name == path or name.startswith(path + "/") for name in self.files)
            return ExecResult(0 if exists else 1, b"", b"")
        return ExecResult(0, b"", b"")

    async def read_file(self, path: str) -> bytes:
        prefix = self._workdir + "/"
        relative = path[len(prefix) :] if path.startswith(prefix) else path
        for name in (path, relative):
            if name in self.files:
                return self.files[name]
        raise SandboxError(f"cannot read {path}: no such file")

    async def write_file(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def diff(self) -> str:
        """Like `docker diff`: absolute paths of files added, changed, or deleted since the
        sandbox was created. Relative paths resolve against the workdir."""
        lines = []
        for name in sorted(set(self.files) | set(self._initial)):
            if name not in self._initial:
                lines.append(f"A {self._absolute(name)}")
            elif name not in self.files:
                lines.append(f"D {self._absolute(name)}")
            elif self.files[name] != self._initial[name]:
                lines.append(f"C {self._absolute(name)}")
        return "\n".join(lines)

    def _absolute(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._workdir}/{path}"

    async def stop(self) -> None:
        self.stopped = True

    async def snapshot(self) -> str:
        return self.snapshot_id

    async def destroy(self) -> None:
        self.destroyed = True
        if self._on_destroy:
            self._on_destroy()
        if self.destroy_error:
            raise SandboxError(self.destroy_error)


class FakeBackend:
    def __init__(
        self,
        *,
        agent_handler: ExecHandler | None = None,
        grader: GradingHandler | None = None,
        files: dict[str, bytes] | None = None,
        image: str = "fake-image",
        image_error: str | None = None,
        create_failures: int = 0,
    ) -> None:
        self.agent_sandboxes: list[FakeSandbox] = []
        self.grading_sandboxes: list[FakeSandbox] = []
        self.trial_of: dict[int, str] = {}  # id(sandbox) -> trial id, as a label would
        self.max_live_agents = 0
        self._live_agents = 0
        self._agent_handler = agent_handler
        self._grader = grader or (lambda cmd, files: ExecResult(0, b"", b""))
        self._files = files or {}
        self._image = image
        self._image_error = image_error
        self._create_failures = create_failures

    async def prepare_image(self, task: Task) -> str:
        if self._image_error:
            raise SandboxError(self._image_error)
        return self._image

    async def create(self, image: str, task: Task, trial_id: str) -> FakeSandbox:
        if self._create_failures > 0:
            self._create_failures -= 1
            raise SandboxError("injected sandbox failure")
        sandbox = FakeSandbox(self._agent_handler, self._files, on_destroy=self._release_agent)
        self.agent_sandboxes.append(sandbox)
        self.trial_of[id(sandbox)] = trial_id
        self._live_agents += 1
        self.max_live_agents = max(self.max_live_agents, self._live_agents)
        return sandbox

    async def create_grading(self, snapshot: str, task: Task, trial_id: str) -> FakeSandbox:
        source = next(box for box in self.agent_sandboxes if box.snapshot_id == snapshot)
        files = dict(source.files)
        grading = FakeSandbox(lambda cmd: self._grader(cmd, files), files)
        self.grading_sandboxes.append(grading)
        self.trial_of[id(grading)] = trial_id
        return grading

    async def remove_trial_sandboxes(self, trial_id: str) -> None:
        for sandbox in self._live():
            if self.trial_of[id(sandbox)] == trial_id:
                await sandbox.destroy()

    async def list_sandboxes(self) -> list[SandboxRecord]:
        return [
            SandboxRecord(
                "container", str(id(sandbox)), self.trial_of[id(sandbox)], sandbox.created
            )
            for sandbox in self._live()
        ]

    async def remove_sandbox(self, record: SandboxRecord) -> None:
        for sandbox in self._live():
            if str(id(sandbox)) == record.ref:
                await sandbox.destroy()

    def _live(self) -> list[FakeSandbox]:
        return [s for s in self.agent_sandboxes + self.grading_sandboxes if not s.destroyed]

    def _release_agent(self) -> None:
        self._live_agents -= 1
