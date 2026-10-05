"""Replay (WF-11, FR-RPL-01, FR-RPL-02): rerun a trial against the model responses it recorded.

The gateway answers the k-th model call of the replay with the response recorded at call k of
the source attempt, but only if the agent sent the same request. Requests are compared as the
agent sent them, before the gateway fitted max_tokens to the remaining budget, so a replay's
budget state cannot cause a false divergence. Tool calls run live in a fresh sandbox.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentoscopy.recorder.trajectory import read_events

REPLAY_DIVERGED_ERROR = "replay_diverged_error"


@dataclass(frozen=True)
class RecordedCall:
    request_hash: str  # of the request as the agent sent it
    status: int  # 200, or the error status the provider answered with
    response: Any  # the response message, or the error body as text


def request_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_recording(trajectory: Path, artifacts_dir: Path) -> list[RecordedCall]:
    """The model calls an attempt recorded, in order, up to the first that never got an answer."""
    events = read_events(trajectory)
    requests = {
        e["payload"]["call_index"]: e["payload"] for e in events if e["type"] == "model_request"
    }
    responses = {
        e["payload"]["call_index"]: e["payload"] for e in events if e["type"] == "model_response"
    }
    calls = []
    for index in sorted(requests):
        if index not in responses:
            break  # the attempt ended during this call, so there is no answer to replay
        sent = json.loads((artifacts_dir / requests[index]["messages_ref"]).read_text("utf-8"))
        sent["max_tokens"] = requests[index]["requested_max_tokens"]
        answer = responses[index]
        if "status" in answer:
            calls.append(RecordedCall(request_hash(sent), answer["status"], answer["error"]))
        else:
            calls.append(RecordedCall(request_hash(sent), 200, _message(index, answer)))
    return calls


def _message(index: int, answer: dict[str, Any]) -> dict[str, Any]:
    """A Messages API response rebuilt from what the trajectory kept of it."""
    return {
        "id": f"msg_replay_{index}",
        "type": "message",
        "role": "assistant",
        "model": answer["model"],
        "content": answer["content"],
        "stop_reason": answer["stop_reason"],
        "stop_sequence": None,
        "usage": {
            "input_tokens": answer.get("input_tokens") or 0,
            "output_tokens": answer.get("output_tokens") or 0,
            "cache_read_input_tokens": answer.get("cache_read_tokens") or 0,
            "cache_creation_input_tokens": answer.get("cache_write_tokens") or 0,
        },
    }
