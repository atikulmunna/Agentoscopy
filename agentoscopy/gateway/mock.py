"""Mock provider (FR-TST-01): seeded-random Messages API responses with real-looking usage.

The same seed key always produces the same response, so runs against the mock are repeatable.
"""

from __future__ import annotations

import json
import random
from typing import Any

WORDS = ("alpha", "beta", "gamma", "delta", "patch", "test", "file", "check", "done", "next")
OUTPUT_TOKENS_RANGE = (50, 500)
STREAM_CHUNK_WORDS = 8


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Rough count (4 characters per token) of everything the model would read."""
    readable = {key: body.get(key) for key in ("system", "messages", "tools")}
    return len(json.dumps(readable)) // 4 + 1


def mock_message(body: dict[str, Any], seed_key: str) -> dict[str, Any]:
    rng = random.Random(seed_key)
    wanted = rng.randint(*OUTPUT_TOKENS_RANGE)
    output_tokens = min(wanted, int(body["max_tokens"]))
    text = " ".join(rng.choice(WORDS) for _ in range(output_tokens))
    return {
        "id": f"msg_mock_{rng.getrandbits(64):016x}",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn" if output_tokens == wanted else "max_tokens",
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_input_tokens(body),
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def sse_events(message: dict[str, Any]) -> list[bytes]:
    """The Messages API streaming event sequence for a text-only message."""
    usage = message["usage"]
    start = {**message, "content": [], "stop_reason": None, "usage": {**usage, "output_tokens": 1}}
    events = [_event("message_start", {"type": "message_start", "message": start})]
    for index, block in enumerate(message["content"]):
        events.append(
            _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
        words = block["text"].split(" ")
        for offset in range(0, len(words), STREAM_CHUNK_WORDS):
            piece = " ".join(words[offset : offset + STREAM_CHUNK_WORDS])
            text = piece if offset == 0 else " " + piece
            events.append(
                _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            )
        events.append(_event("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(
        _event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
    )
    events.append(_event("message_stop", {"type": "message_stop"}))
    return events


def _event(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
