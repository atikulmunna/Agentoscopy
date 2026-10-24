import subprocess
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml
from aiohttp import web

from agentoscopy.gateway.mock import mock_message, sse_events
from agentoscopy.gateway.server import Gateway
from agentoscopy.recorder.trajectory import TrajectoryRecorder
from agentoscopy.spec import Task, load_task
from agentoscopy.testing.fake_sandbox import FakeBackend

FAKE_DIGEST = "sha256:" + "0" * 64

BASE_TASK: dict[str, Any] = {
    "id": "demo-task",
    "instructions": "Make the tests pass.",
    "environment": {"base_image": f"python:3.12-slim@{FAKE_DIGEST}"},
    "budget": {"timeout_s": 5},
    "graders": [{"name": "tests", "type": "command", "run": "run-tests"}],
}


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    path = tmp_path / "tasks"
    path.mkdir()
    return path


@pytest.fixture
def write_task(tasks_dir: Path) -> Callable[..., Path]:
    """Write a task directory; returns its path. `files` maps task-relative paths to text."""

    def _write(spec: dict[str, Any] | None = None, files: dict[str, str] | None = None) -> Path:
        spec = BASE_TASK if spec is None else spec
        task_dir = tasks_dir / spec.get("id", "demo-task")
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
        for relative, content in (files or {}).items():
            target = task_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return task_dir

    return _write


@pytest.fixture
def make_task(write_task: Callable[..., Path], tasks_dir: Path) -> Callable[..., Task]:
    """Write a task (BASE_TASK merged with overrides) and load it."""

    def _make(files: dict[str, str] | None = None, **overrides: Any) -> Task:
        spec = {**BASE_TASK, **overrides}
        write_task(spec, files)
        return load_task(tasks_dir, spec["id"])

    return _make


@pytest.fixture
def recorder(tmp_path: Path):
    recorder = TrajectoryRecorder(tmp_path / "attempt-1.jsonl", tmp_path / "artifacts", "trial", 1)
    yield recorder
    recorder.close()


@asynccontextmanager
async def running_gateway(**kwargs: Any) -> AsyncIterator[Gateway]:
    gateway = Gateway(**kwargs)
    await gateway.start()
    try:
        yield gateway
    finally:
        await gateway.stop()


class FakeProvider:
    """Stands in for the Anthropic API. Each request pops the next queued reply; with none
    queued it answers with a mock message. Requests are kept for assertions."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.replies: list[tuple[str, int, Any]] = []
        self.base_url = ""
        self._runner: web.AppRunner | None = None

    def reply_status(self, status: int, error_type: str = "api_error") -> None:
        body = {"type": "error", "error": {"type": error_type, "message": f"fake {status}"}}
        self.replies.append(("json", status, body))

    def reply_message(self, message: dict[str, Any]) -> None:
        self.replies.append(("json", 200, message))

    def reply_stream(self) -> None:
        self.replies.append(("sse", 200, None))

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/v1/messages", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        self.base_url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(
            {"headers": dict(request.headers), "body": body, "query": request.query_string}
        )
        kind, status, payload = self.replies.pop(0) if self.replies else ("json", 200, None)
        if kind == "sse":
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for event in sse_events(mock_message(body, "provider")):
                await response.write(event)
            await response.write_eof()
            return response
        return web.json_response(payload or mock_message(body, "provider"), status=status)


@asynccontextmanager
async def running_provider() -> AsyncIterator[FakeProvider]:
    provider = FakeProvider()
    await provider.start()
    try:
        yield provider
    finally:
        await provider.stop()


class SetupBackend(FakeBackend):
    """Its setup command installs a package outside the workdir, as `pip install` would."""

    async def create(self, image, task, trial_id):
        sandbox = await super().create(image, task, trial_id)
        run = sandbox.exec

        async def exec(cmd, timeout_s=60):
            if cmd == "install-deps":
                sandbox.files["/usr/lib/python3/site-packages/dep.py"] = b"dep"
            return await run(cmd, timeout_s)

        sandbox.exec = exec
        return sandbox


class GitRepo:
    """A throwaway git repository for import tests, with a test-only commit identity."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True)
        self.path = path
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        identity = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
        command = ["git", "-C", str(self.path), *identity, "-c", "commit.gpgsign=false", *args]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

    def commit(self, message: str, files: dict[str, str] | None = None, delete=()) -> str:
        for name, content in (files or {}).items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        for name in delete:
            self.git("rm", "-q", name)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


BUGGY_CALC = "def add(a, b):\n    return a - b\n"
FIXED_CALC = "def add(a, b):\n    return a + b\n"
OLD_CALC_TEST = (
    "import unittest\n\nfrom calc import add\n\n\n"
    "class AddTest(unittest.TestCase):\n"
    "    def test_zero(self):\n        self.assertEqual(add(0, 0), 0)\n"
)
NEW_CALC_TEST = (
    OLD_CALC_TEST + "\n    def test_sum(self):\n        self.assertEqual(add(2, 3), 5)\n"
)
OTHER_TEST = (
    "import unittest\n\n\nclass OtherTest(unittest.TestCase):\n"
    "    def test_unrelated(self):\n        self.assertTrue(True)\n"
)


def calc_repo_with_fix(path: Path) -> tuple[GitRepo, str]:
    """A repository whose second commit fixes add() and adds the test that catches it; a
    second test file is left alone by the fix."""
    repo = GitRepo(path)
    repo.commit("Add calc", {"calc.py": BUGGY_CALC, "tests/test_calc.py": OLD_CALC_TEST,
                             "tests/test_other.py": OTHER_TEST, "README.md": "calc\n"})  # fmt: skip
    fix = repo.commit(
        "add() subtracts instead of adding",
        {"calc.py": FIXED_CALC, "tests/test_calc.py": NEW_CALC_TEST},
    )
    return repo, fix
