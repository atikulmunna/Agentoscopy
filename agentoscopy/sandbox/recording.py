"""Agent-facing sandbox that records every operation as trajectory events (FR-TRJ-07).

Adapters only ever receive this wrapper, so trajectory graders do not depend on adapters
reporting their own tool calls.
"""

from __future__ import annotations

import time

from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.sandbox.base import ExecResult, Sandbox, SandboxError


class RecordingSandbox:
    def __init__(self, inner: Sandbox, recorder: TrajectoryRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def workdir(self) -> str:
        return self._inner.workdir

    async def exec(self, cmd: str, timeout_s: int = 60) -> ExecResult:
        started = self._record_call("exec", {"cmd": cmd, "timeout_s": timeout_s})
        try:
            result = await self._inner.exec(cmd, timeout_s)
        except SandboxError as exc:
            self._record_result(started, error=str(exc))
            raise
        self._record_result(
            started,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return result

    async def read_file(self, path: str) -> bytes:
        started = self._record_call("read_file", {"path": path})
        try:
            data = await self._inner.read_file(path)
        except SandboxError as exc:
            self._record_result(started, error=str(exc))
            raise
        self._record_result(started, exit_code=0, stdout=data)
        return data

    async def write_file(self, path: str, data: bytes) -> None:
        args = {"path": path, "bytes": len(data), **self._recorder.inline_blobs(content=data)}
        started = self._record_call("write_file", args)
        try:
            await self._inner.write_file(path, data)
        except SandboxError as exc:
            self._record_result(started, error=str(exc))
            raise
        self._record_result(started, exit_code=0)

    def _record_call(self, tool: str, args: dict) -> float:
        self._recorder.write("tool_call", {"tool": tool, "args": args})
        return time.monotonic()

    def _record_result(
        self,
        started: float,
        *,
        exit_code: int | None = None,
        timed_out: bool = False,
        error: str | None = None,
        **blobs: bytes,
    ) -> None:
        payload = {
            "exit_code": exit_code,
            **self._recorder.inline_blobs(**blobs),
            "duration_ms": round((time.monotonic() - started) * 1000),
            "timed_out": timed_out,
            "error": error,
        }
        self._recorder.write("tool_result", payload)
