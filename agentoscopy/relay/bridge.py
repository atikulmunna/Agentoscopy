"""The harness side of the in-sandbox model-call relay (see relay/inside.py).

RelayBridge starts the relay in a sandbox and answers every request it passes along by
calling the trial's gateway with the trial's token, streaming the answer back. The agent in
the sandbox only ever reaches the relay, on the sandbox's own loopback address.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp

from agentoscopy.adapters.base import ModelEndpoint
from agentoscopy.sandbox.base import Sandbox

INSIDE_SOURCE = (Path(__file__).parent / "inside.py").read_text(encoding="utf-8")
START_TIMEOUT_S = 30.0
STOP_TIMEOUT_S = 5.0
FORWARDED_HEADERS = ("content-type", "accept", "anthropic-version", "anthropic-beta")
RETURNED_HEADERS = ("content-type", "retry-after", "request-id")


class RelayError(Exception):
    """The relay could not be started in the sandbox."""


class RelayBridge:
    def __init__(self, sandbox: Sandbox, endpoint: ModelEndpoint) -> None:
        self._sandbox = sandbox
        self._endpoint = endpoint
        self._lock = asyncio.Lock()
        self._requests: set[asyncio.Task[None]] = set()
        self.url = ""  # the relay's address inside the sandbox, once started

    async def __aenter__(self) -> RelayBridge:
        self._process = await self._sandbox.start("python3", "-u", "-c", INSIDE_SOURCE)
        try:
            ready = await asyncio.wait_for(self._process.stdout.readline(), START_TIMEOUT_S)
            port = int(json.loads(ready)["port"])
        except (TimeoutError, ValueError, KeyError, TypeError) as exc:
            await self._stop()
            raise RelayError(
                "the model-call relay did not start in the sandbox; the image needs python3"
            ) from exc
        self.url = f"http://127.0.0.1:{port}"
        self._client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        self._pump = asyncio.create_task(self._serve())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        tasks = [self._pump, *self._requests]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._client.close()
        await self._stop()

    async def _serve(self) -> None:
        async for line in _lines(self._process.stdout):
            frame = json.loads(line)
            if frame.get("type") == "request":
                task = asyncio.create_task(self._forward(frame))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)

    async def _forward(self, frame: dict[str, Any]) -> None:
        request_id = frame["id"]
        headers = {
            name: value for name, value in frame["headers"].items() if name in FORWARDED_HEADERS
        }
        headers["x-api-key"] = self._endpoint.token
        url = self._endpoint.base_url.rstrip("/") + frame["path"]
        body = base64.b64decode(frame["body"])
        answered = False
        try:
            async with self._client.request(
                frame["method"], url, data=body, headers=headers
            ) as response:
                kept = {
                    name: response.headers[name]
                    for name in RETURNED_HEADERS
                    if name in response.headers
                }
                await self._send(
                    {"type": "head", "id": request_id, "status": response.status, "headers": kept}
                )
                answered = True
                async for chunk in response.content.iter_any():
                    await self._send({"type": "chunk", "id": request_id, "data": _b64(chunk)})
        except aiohttp.ClientError as exc:
            if not answered:
                await self._send_error(request_id, f"gateway unreachable: {type(exc).__name__}")
        await self._send({"type": "end", "id": request_id})

    async def _send_error(self, request_id: int, message: str) -> None:
        error = {"type": "error", "error": {"type": "api_error", "message": message}}
        head = {"content-type": "application/json"}
        await self._send({"type": "head", "id": request_id, "status": 502, "headers": head})
        await self._send(
            {"type": "chunk", "id": request_id, "data": _b64(json.dumps(error).encode())}
        )

    async def _send(self, frame: dict[str, Any]) -> None:
        async with self._lock:
            self._process.stdin.write(json.dumps(frame).encode() + b"\n")
            await self._process.stdin.drain()

    async def _stop(self) -> None:
        """Closing stdin ends the relay; a relay that hangs is killed."""
        if self._process.stdin and not self._process.stdin.is_closing():
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), STOP_TIMEOUT_S)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    """Lines of any length. StreamReader's own readline stops at 64 KiB, and one model
    request from a real agent is a single line far longer than that."""
    buffer = bytearray()
    while chunk := await stream.read(1 << 16):
        searched = len(buffer)  # only new bytes can hold the end of the current line
        buffer += chunk
        while (end := buffer.find(b"\n", searched)) >= 0:
            yield bytes(buffer[:end])
            del buffer[: end + 1]
            searched = 0


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
