"""Docker sandbox backend, driven through the docker CLI."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from agentoscopy.sandbox.base import HIDDEN_MOUNT, ExecResult, SandboxError, SandboxRecord
from agentoscopy.spec import Task

AGENT_USER = "1000:1000"  # non-root (FR-SBX-04); the task image chowns the workdir to it
PIDS_LIMIT = 512
TRIAL_LABEL = "agentoscopy.trial_id"
INSPECT_FORMAT = '{{.Id}} {{index .Config.Labels "' + TRIAL_LABEL + '"}} {{.Created}}'
KILL_GRACE_S = 5  # time between SIGTERM and SIGKILL when an exec times out
FILE_OP_TIMEOUT_S = 60
TIMEOUT_EXIT_CODES = (124, 137)  # coreutils `timeout`: SIGTERM worked / SIGKILL was needed
DAEMON_ERROR_MARKER = b"Error response from daemon"


class DockerBackend:
    async def prepare_image(self, task: Task) -> str:
        tag = task_image_tag(task)
        existing = await run_docker("image", "inspect", "--format", "{{.Id}}", tag)
        if existing.exit_code == 0:
            return existing.stdout.decode().strip()
        with tempfile.TemporaryDirectory(prefix="agentoscopy-build-") as context:
            _write_build_context(Path(context), task)
            await _docker_checked("build", "--quiet", "--tag", tag, context)
        return await _docker_checked("image", "inspect", "--format", "{{.Id}}", tag)

    async def create(self, image: str, task: Task, trial_id: str) -> DockerSandbox:
        container = await _docker_checked(*container_run_args(image, task, trial_id))
        return DockerSandbox(container, task.spec.environment.workdir)

    async def create_grading(self, snapshot: str, task: Task, trial_id: str) -> DockerSandbox:
        mounts = hidden_mount(task.hidden_dir) if task.hidden_dir.is_dir() else []
        try:
            container = await _docker_checked(*container_run_args(snapshot, task, trial_id, mounts))
        except SandboxError:
            await run_docker("image", "rm", "--force", snapshot)
            raise
        return DockerSandbox(container, task.spec.environment.workdir, owned_image=snapshot)

    async def remove_trial_sandboxes(self, trial_id: str) -> None:
        label = f"label={TRIAL_LABEL}={trial_id}"
        containers = (await _docker_checked("ps", "--all", "--quiet", "--filter", label)).split()
        if containers:
            await _docker_checked("rm", "--force", "--volumes", *containers)
        images = sorted(
            set((await _docker_checked("images", "--quiet", "--filter", label)).split())
        )
        if images:
            await _docker_checked("image", "rm", "--force", *images)

    async def list_sandboxes(self) -> list[SandboxRecord]:
        label = f"label={TRIAL_LABEL}"
        found = {
            "container": await _docker_checked("ps", "--all", "--quiet", "--filter", label),
            "image": await _docker_checked("images", "--quiet", "--filter", label),
        }
        records = []
        for kind, listing in found.items():
            refs = sorted(set(listing.split()))
            if not refs:
                continue
            inspected = await _docker_checked(
                "inspect", "--type", kind, "--format", INSPECT_FORMAT, *refs
            )
            for line in inspected.splitlines():
                ref, trial_id, created = line.split(" ", 2)
                records.append(SandboxRecord(kind, ref, trial_id, _docker_time(created)))
        return records

    async def remove_sandbox(self, record: SandboxRecord) -> None:
        if record.kind == "container":
            await _docker_checked("rm", "--force", "--volumes", record.ref)
        else:
            await _docker_checked("image", "rm", "--force", record.ref)


class DockerSandbox:
    def __init__(self, container_id: str, workdir: str, owned_image: str | None = None) -> None:
        self.container_id = container_id
        self._workdir = workdir
        self._owned_image = owned_image

    @property
    def workdir(self) -> str:
        return self._workdir

    async def exec(self, cmd: str, timeout_s: int = 60) -> ExecResult:
        # Killing the docker client does not stop the process in the container, so the
        # deadline is enforced inside it with coreutils `timeout`.
        started = time.monotonic()
        result = await self._exec_raw(
            "timeout",
            "--kill-after",
            str(KILL_GRACE_S),
            str(timeout_s),
            "sh",
            "-c",
            cmd,
            timeout_s=timeout_s + 2 * KILL_GRACE_S,
        )
        hit_deadline = time.monotonic() - started >= timeout_s
        timed_out = result.timed_out or (result.exit_code in TIMEOUT_EXIT_CODES and hit_deadline)
        return ExecResult(result.exit_code, result.stdout, result.stderr, timed_out)

    async def read_file(self, path: str) -> bytes:
        result = await self._exec_raw("cat", "--", path, timeout_s=FILE_OP_TIMEOUT_S)
        if result.exit_code != 0 or result.timed_out:
            raise SandboxError(f"cannot read {path}: {_text(result.stderr) or 'timed out'}")
        return result.stdout

    async def write_file(self, path: str, data: bytes) -> None:
        # The path is passed as $1, never interpolated into the script.
        script = 'mkdir -p "$(dirname -- "$1")" && cat > "$1"'
        result = await self._exec_raw(
            "sh", "-c", script, "sh", path, stdin=data, timeout_s=FILE_OP_TIMEOUT_S
        )
        if result.exit_code != 0 or result.timed_out:
            raise SandboxError(f"cannot write {path}: {_text(result.stderr) or 'timed out'}")

    async def diff(self) -> str:
        return await _docker_checked("diff", self.container_id)

    async def stop(self) -> None:
        await _docker_checked("stop", "-t", "0", self.container_id)

    async def snapshot(self) -> str:
        return await _docker_checked("commit", self.container_id)

    async def destroy(self) -> None:
        removed = await run_docker("rm", "--force", "--volumes", self.container_id)
        problems = [] if removed.exit_code == 0 else [_text(removed.stderr)]
        if self._owned_image:
            image = await run_docker("image", "rm", "--force", self._owned_image)
            if image.exit_code != 0:
                problems.append(_text(image.stderr))
        if problems:
            raise SandboxError(f"cleanup of {self.container_id[:12]} failed: {'; '.join(problems)}")

    async def _exec_raw(
        self, *command: str, timeout_s: float, stdin: bytes | None = None
    ) -> ExecResult:
        interactive = ["--interactive"] if stdin is not None else []
        result = await run_docker(
            "exec",
            *interactive,
            "--user",
            AGENT_USER,
            "--workdir",
            self._workdir,
            self.container_id,
            *command,
            stdin=stdin,
            timeout_s=timeout_s,
        )
        # The docker CLI exits 1 when the container itself is gone, the same code a failing
        # command uses; its error banner tells them apart, so a dead sandbox is an infra error
        # rather than a failed agent command or grader.
        if result.exit_code != 0 and DAEMON_ERROR_MARKER in result.stderr:
            raise SandboxError(f"sandbox {self.container_id[:12]} failed: {_text(result.stderr)}")
        return result


def task_image_tag(task: Task) -> str:
    return f"agentoscopy-task/{task.spec.id}:{task.content_hash[:16]}"


def render_dockerfile(task: Task) -> str:
    environment = task.spec.environment
    lines = [f"FROM {environment.base_image}", f"WORKDIR {environment.workdir}"]
    if task.fixture_files():
        lines.append(f"COPY fixtures/ {environment.workdir}/")
    # JSON exec form keeps multi-line and quoted build steps intact.
    lines += [f"RUN {json.dumps(['/bin/sh', '-c', step])}" for step in environment.build]
    lines.append(f"RUN chown -R {AGENT_USER} {environment.workdir}")
    return "\n".join(lines) + "\n"


def container_run_args(
    image: str, task: Task, trial_id: str, mounts: list[str] | None = None
) -> list[str]:
    environment = task.spec.environment
    return [
        "run",
        "--detach",
        "--label",
        f"{TRIAL_LABEL}={trial_id}",
        "--user",
        AGENT_USER,
        "--workdir",
        environment.workdir,
        "--env",
        "HOME=/tmp",
        "--network",
        "none",  # deny-all egress (FR-SBX-03); the gateway route arrives in M1
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(PIDS_LIMIT),
        "--cpus",
        str(environment.resources.cpu),
        "--memory",
        f"{environment.resources.memory_mb}m",
        *(mounts or []),
        image,
        "sleep",
        "infinity",
    ]


def hidden_mount(hidden_dir: Path) -> list[str]:
    # docker parses --mount as CSV, so the source is quoted to survive commas in the path.
    source = str(hidden_dir.resolve()).replace('"', '""')
    return ["--mount", f'type=bind,"source={source}",target={HIDDEN_MOUNT},readonly']


async def run_docker(
    *args: str, stdin: bytes | None = None, timeout_s: float | None = None
) -> ExecResult:
    """Run one docker CLI command; the client is killed if it times out or is cancelled."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SandboxError("docker CLI not found on PATH") from exc
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        return ExecResult(exit_code=-1, stdout=b"", stderr=b"", timed_out=True)
    except asyncio.CancelledError:
        process.kill()
        raise
    exit_code = process.returncode if process.returncode is not None else -1
    return ExecResult(exit_code, stdout, stderr)


async def _docker_checked(*args: str) -> str:
    result = await run_docker(*args)
    if result.exit_code != 0:
        raise SandboxError(f"docker {args[0]} failed: {_text(result.stderr)[-2000:]}")
    return _text(result.stdout)


def _write_build_context(context: Path, task: Task) -> None:
    (context / "Dockerfile").write_text(render_dockerfile(task), encoding="utf-8", newline="\n")
    for file in task.fixture_files():
        target = context / "fixtures" / file.relative_to(task.fixtures_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)


def _docker_time(text: str) -> datetime:
    """Docker prints RFC 3339 times with nanoseconds, more digits than datetime accepts."""
    trimmed = re.sub(r"(\.\d{6})\d+", r"\1", text.strip()).replace("Z", "+00:00")
    return datetime.fromisoformat(trimmed)


def _text(data: bytes) -> str:
    return data.decode("utf-8", "replace").strip()
