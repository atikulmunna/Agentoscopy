"""Append-only JSONL trajectory for one trial attempt (FR-TRJ-01 to FR-TRJ-05, FR-TRJ-07)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ADAPTER_EVENT_TYPES = frozenset({"agent_message", "error"})
INLINE_OUTPUT_LIMIT = 4096  # bytes kept inline per blob; the full blob goes to an artifact
REDACTED = b"[REDACTED]"
# Anthropic API keys and admin keys; registered secrets (gateway tokens) are matched exactly.
KEY_PATTERN = re.compile(rb"sk-ant-[A-Za-z0-9_\-]{8,}")


class TrajectoryRecorder:
    """Writes events for one trial attempt, flushing after every event.

    Every event and artifact passes a secret-redaction filter before it reaches disk.
    """

    def __init__(self, path: Path, artifacts_dir: Path, trial_id: str, attempt: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.artifacts_dir = artifacts_dir
        # A step is one model call through the gateway (§1.3); the gateway advances it.
        self.step = 0
        self._trial_id = trial_id
        self._attempt = attempt
        self._seq = 0
        self._secrets: set[bytes] = set()
        self._file = path.open("ab")

    def add_secret(self, secret: str) -> None:
        if secret:
            self._secrets.add(secret.encode())

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "seq": self._seq,
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "trial_id": self._trial_id,
            "attempt": self._attempt,
            "step": self.step,
            "type": event_type,
            "payload": payload,
        }
        line = json.dumps(event, ensure_ascii=False).encode() + b"\n"
        self._file.write(self._redact(line))
        self._file.flush()
        self._seq += 1

    def inline_blobs(self, **blobs: bytes) -> dict[str, Any]:
        """Truncated text for each blob, plus `output_ref` to the full blobs if any was cut."""
        fields: dict[str, Any] = {name: _truncate(data) for name, data in blobs.items()}
        fields["output_ref"] = None
        if any(len(data) > INLINE_OUTPUT_LIMIT for data in blobs.values()):
            full = {name: data.decode("utf-8", "replace") for name, data in blobs.items()}
            name = f"outputs/{self._seq}.json"
            fields["output_ref"] = self.store_artifact(name, json.dumps(full).encode())
        return fields

    def store_artifact(self, name: str, data: bytes) -> str:
        """Store a blob for this attempt; returns its path relative to the artifacts dir."""
        target = self.artifacts_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._redact(data))
        return name

    def for_adapter(self) -> AdapterRecorder:
        return AdapterRecorder(self)

    def close(self) -> None:
        self._file.close()

    def _redact(self, data: bytes) -> bytes:
        for secret in self._secrets:
            data = data.replace(secret, REDACTED)
        return KEY_PATTERN.sub(REDACTED, data)


class AdapterRecorder:
    """The recorder handed to adapters. Model and tool events come from the harness instead."""

    def __init__(self, recorder: TrajectoryRecorder) -> None:
        self._recorder = recorder

    def event(self, event_type: str, **payload: Any) -> None:
        if event_type not in ADAPTER_EVENT_TYPES:
            allowed = ", ".join(sorted(ADAPTER_EVENT_TYPES))
            raise ValueError(f"adapters may only write {allowed} events, not {event_type!r}")
        self._recorder.write(event_type, payload)


def read_events(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _truncate(data: bytes) -> str:
    text = data[:INLINE_OUTPUT_LIMIT].decode("utf-8", "replace")
    return text + "\n[truncated]" if len(data) > INLINE_OUTPUT_LIMIT else text
