"""Model-call relay that runs inside an offline sandbox (started by relay.bridge.RelayBridge).

An agent in the sandbox sends HTTP requests to this relay on the container's own loopback
address. Each request goes to the harness as one JSON line on stdout; the harness sends it
to the trial's gateway and streams the answer back as JSON lines on stdin. So the sandbox
keeps no network, and no credential ever enters it.

Standard library only, and Python 3.8 or later, so it runs in any image with python3.
"""

import asyncio
import base64
import json
import sys
import threading

DROPPED_HEADERS = {"connection", "content-length", "transfer-encoding", "keep-alive"}


class Relay:
    def __init__(self, loop):
        self.loop = loop
        self.pending = {}  # request id -> queue of answer frames
        self.next_id = 0
        self.stopped = asyncio.Event()
        self._stdout = threading.Lock()

    def send(self, frame):
        line = (json.dumps(frame) + "\n").encode("utf-8")
        with self._stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    def deliver(self, frame):
        queue = self.pending.get(frame.get("id"))
        if queue is not None:
            queue.put_nowait(frame)

    async def handle(self, reader, writer):
        try:
            method, path, headers, body = await read_request(reader)
        except (ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            writer.close()
            return
        self.next_id += 1
        request_id = self.next_id
        self.pending[request_id] = asyncio.Queue()
        self.send(
            {
                "type": "request",
                "id": request_id,
                "method": method,
                "path": path,
                "headers": headers,
                "body": base64.b64encode(body).decode("ascii"),
            }
        )
        try:
            await respond(self.pending[request_id], writer)
        finally:
            self.pending.pop(request_id, None)
            writer.close()


async def respond(queue, writer):
    """Write the answer frames as a chunked HTTP response, so streamed answers stream."""
    while True:
        frame = await queue.get()
        if frame["type"] == "head":
            lines = [f"HTTP/1.1 {frame['status']} Relayed"]
            lines += [
                f"{name}: {value}"
                for name, value in frame["headers"].items()
                if name.lower() not in DROPPED_HEADERS
            ]
            lines += ["Transfer-Encoding: chunked", "Connection: close", "", ""]
            writer.write("\r\n".join(lines).encode("latin-1"))
        elif frame["type"] == "chunk":
            data = base64.b64decode(frame["data"])
            if data:
                writer.write(b"%x\r\n" % len(data) + data + b"\r\n")
        else:  # end
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            return
        await writer.drain()


async def read_request(reader):
    head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
    method, path, _version = head[0].split(" ", 2)
    headers = {}
    for line in head[1:]:
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    if headers.get("transfer-encoding", "").lower() == "chunked":
        body = await read_chunked(reader)
    else:
        body = await reader.readexactly(int(headers.get("content-length") or 0))
    return method, path, headers, body


async def read_chunked(reader):
    body = b""
    while True:
        size = int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)
        if size == 0:
            await reader.readuntil(b"\r\n")
            return body
        body += await reader.readexactly(size)
        await reader.readexactly(2)


def read_answers(relay):
    """Runs in a thread: stdin carries the harness's answer frames until it closes."""
    for line in sys.stdin.buffer:
        relay.loop.call_soon_threadsafe(relay.deliver, json.loads(line))
    relay.loop.call_soon_threadsafe(relay.stopped.set)


async def main():
    relay = Relay(asyncio.get_running_loop())
    server = await asyncio.start_server(relay.handle, "127.0.0.1", 0)
    relay.send({"type": "listening", "port": server.sockets[0].getsockname()[1]})
    threading.Thread(target=read_answers, args=(relay,), daemon=True).start()
    await relay.stopped.wait()
    server.close()


if __name__ == "__main__":
    asyncio.run(main())
