"""Rebuilds the final message from a Messages API event stream, for recording and costing."""

from __future__ import annotations

import json
from typing import Any


class MessageAccumulator:
    def __init__(self) -> None:
        self.message: dict[str, Any] | None = None
        self.error: dict[str, Any] | None = None
        self.malformed_events = 0  # relayed to the agent untouched, but not understood here
        self._buffer = b""
        self._partial_json: dict[int, str] = {}

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk.replace(b"\r\n", b"\n")
        while b"\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split(b"\n\n", 1)
            for line in raw_event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                try:
                    self._apply(json.loads(line[5:].strip()))
                except (ValueError, KeyError, IndexError, TypeError):
                    self.malformed_events += 1

    def _apply(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start":
            self.message = {**event["message"], "content": []}
        elif kind == "error":
            self.error = event.get("error")
        elif self.message is None:
            return
        elif kind == "content_block_start":
            self._block(event["index"], event["content_block"])
        elif kind == "content_block_delta":
            self._delta(event["index"], event["delta"])
        elif kind == "content_block_stop":
            self._finish_block(event["index"])
        elif kind == "message_delta":
            self.message.update(event.get("delta") or {})
            self.message["usage"] = {**self.message.get("usage", {}), **(event.get("usage") or {})}

    def _block(self, index: int, block: dict[str, Any]) -> None:
        content = self.message["content"]
        while len(content) <= index:
            content.append({})
        content[index] = dict(block)

    def _delta(self, index: int, delta: dict[str, Any]) -> None:
        block = self.message["content"][index]
        kind = delta.get("type")
        if kind == "text_delta":
            block["text"] = block.get("text", "") + delta["text"]
        elif kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta["thinking"]
        elif kind == "signature_delta":
            block["signature"] = delta["signature"]
        elif kind == "input_json_delta":
            self._partial_json[index] = self._partial_json.get(index, "") + delta["partial_json"]

    def _finish_block(self, index: int) -> None:
        partial = self._partial_json.pop(index, None)
        if not partial:
            return
        try:
            self.message["content"][index]["input"] = json.loads(partial)
        except ValueError:  # keep what arrived; the trajectory should show it as-is
            self.message["content"][index]["input_raw"] = partial
