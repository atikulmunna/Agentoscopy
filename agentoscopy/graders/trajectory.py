"""Trajectory graders (FR-GRD-04): rules over the events the harness recorded.

Tool events come from the sandbox handle and model events from the gateway, never from the
agent, so an agent cannot hide an action from these graders by not reporting it.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterator
from typing import Any

from agentoscopy.graders.base import GradeResult, GradingContext, check_result
from agentoscopy.graders.fsdiff import diff_entries
from agentoscopy.spec import (
    ForbiddenCommandSpec,
    MaxStepsSpec,
    MaxToolErrorsSpec,
    MustReadBeforeEditSpec,
)

KIND = "trajectory"


class ForbiddenCommandGrader:
    kind = KIND

    def __init__(self, spec: ForbiddenCommandSpec) -> None:
        self.name = spec.name
        self._patterns = [re.compile(pattern) for pattern in spec.patterns]

    async def grade(self, context: GradingContext) -> GradeResult:
        hits = [
            f"#{event['seq']}: {command}"
            for event, command in commands(context.trajectory)
            if any(pattern.search(command) for pattern in self._patterns)
        ]
        if not hits:
            return check_result(self.name, KIND, True, "No forbidden commands.")
        return check_result(
            self.name, KIND, False, "Forbidden commands:\n" + "\n".join(hits), violations=len(hits)
        )


class MustReadBeforeEditGrader:
    """An edit is a write to a file that existed before the trial: the container diff shows it
    as changed, not added. Reading means a read_file of that path, or a command naming it."""

    kind = KIND

    def __init__(self, spec: MustReadBeforeEditSpec) -> None:
        self.name = spec.name

    async def grade(self, context: GradingContext) -> GradeResult:
        workdir = context.task.spec.environment.workdir
        existing = {path for kind, path in diff_entries(context.fs_diff) if kind == "C"}
        read_paths: set[str] = set()
        ran: list[str] = []
        blind = []
        for event in context.trajectory:
            if event["type"] != "tool_call":
                continue
            tool, args = event["payload"]["tool"], event["payload"].get("args", {})
            if tool == "read_file":
                read_paths.add(_absolute(args["path"], workdir))
            elif tool == "exec":
                ran.append(args["cmd"])
            elif tool == "write_file":
                path = _absolute(args["path"], workdir)
                seen = path in read_paths or any(_names(cmd, path, workdir) for cmd in ran)
                if path in existing and not seen:
                    blind.append(f"#{event['seq']}: {args['path']}")
        if not blind:
            return check_result(self.name, KIND, True, "Every edited file was read first.")
        rationale = "Edited without reading first:\n" + "\n".join(blind)
        return check_result(self.name, KIND, False, rationale, violations=len(blind))


class MaxToolErrorsGrader:
    kind = KIND

    def __init__(self, spec: MaxToolErrorsSpec) -> None:
        self.name = spec.name
        self._max = spec.max

    async def grade(self, context: GradingContext) -> GradeResult:
        errors = [
            event["seq"]
            for event in context.trajectory
            if event["type"] == "tool_result" and _failed(event["payload"])
        ]
        passed = len(errors) <= self._max
        rationale = f"{len(errors)} failed tool calls (limit {self._max})."
        return check_result(self.name, KIND, passed, rationale, errors=len(errors))


class MaxStepsGrader:
    kind = KIND

    def __init__(self, spec: MaxStepsSpec) -> None:
        self.name = spec.name
        self._max = spec.max

    async def grade(self, context: GradingContext) -> GradeResult:
        steps = sum(event["type"] == "model_request" for event in context.trajectory)
        rationale = f"{steps} model calls (limit {self._max})."
        return check_result(self.name, KIND, steps <= self._max, rationale, steps=steps)


def commands(events: list[dict[str, Any]]) -> Iterator[tuple[dict[str, Any], str]]:
    """Every shell command the agent ran, with its event."""
    for event in events:
        payload = event["payload"]
        if event["type"] == "tool_call" and payload.get("tool") == "exec":
            yield event, payload["args"]["cmd"]


def _failed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("error") or result.get("timed_out") or result.get("exit_code") not in (0, None)
    )


def _absolute(path: str, workdir: str) -> str:
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(workdir, path))


def _names(command: str, path: str, workdir: str) -> bool:
    return path in command or posixpath.relpath(path, workdir) in command
