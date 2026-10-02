import asyncio
import json
from pathlib import Path

import pytest

from agentoscopy.sandbox import docker
from agentoscopy.sandbox.base import ExecResult, SandboxError
from agentoscopy.sandbox.docker import (
    AGENT_USER,
    DockerSandbox,
    container_run_args,
    hidden_mount,
    render_dockerfile,
    task_image_tag,
)


def test_dockerfile_bakes_fixtures_and_build_steps(make_task):
    step = "echo 'quoted' && printf \"%s\\n\" multi\nline"
    task = make_task(
        files={"fixtures/app.py": "x = 1\n"},
        environment={"base_image": "img@sha256:abc", "build": [step]},
    )

    lines = render_dockerfile(task).splitlines()

    assert lines[0] == "FROM img@sha256:abc"
    assert "WORKDIR /workspace" in lines
    assert "COPY fixtures/ /workspace/" in lines
    run_line = next(line for line in lines if line.startswith("RUN [")).removeprefix("RUN ")
    assert json.loads(run_line) == ["/bin/sh", "-c", step]
    assert lines[-1] == f"RUN chown -R {AGENT_USER} /workspace"


def test_dockerfile_skips_copy_without_fixtures(make_task):
    assert "COPY" not in render_dockerfile(make_task())


def test_image_tag_is_keyed_by_content_hash(make_task):
    task = make_task()

    assert task_image_tag(task) == f"agentoscopy-task/demo-task:{task.content_hash[:16]}"


def test_agent_container_is_locked_down(make_task):
    task = make_task(
        environment={"base_image": "img@sha256:abc", "resources": {"cpu": 2, "memory_mb": 512}}
    )

    args = container_run_args("image-id", task, "trial-1")

    for flag, value in [
        ("--network", "none"),
        ("--user", AGENT_USER),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--label", "agentoscopy.trial_id=trial-1"),
        ("--cpus", "2.0"),
        ("--memory", "512m"),
    ]:
        assert args[args.index(flag) + 1] == value, flag
    assert "--mount" not in args
    assert args[-3:] == ["image-id", "sleep", "infinity"]


def test_hidden_mount_is_read_only_and_quotes_awkward_paths(tmp_path):
    hidden = tmp_path / "dir with space, and comma" / "hidden"

    flag, value = hidden_mount(hidden)

    assert flag == "--mount"
    assert value.startswith("type=bind,")
    assert f'"source={Path(hidden).resolve()}"' in value
    assert value.endswith(",target=/hidden,readonly")


def fake_docker(monkeypatch, exit_code, stderr):
    async def run_docker(*args, **kwargs):
        return ExecResult(exit_code, b"", stderr)

    monkeypatch.setattr(docker, "run_docker", run_docker)


def test_a_dead_container_is_a_sandbox_error_not_a_failed_command(monkeypatch):
    fake_docker(monkeypatch, 1, b"Error response from daemon: No such container: abc123")

    with pytest.raises(SandboxError, match="No such container"):
        asyncio.run(DockerSandbox("abc123", "/workspace").exec("pytest"))


def test_a_failing_command_is_just_a_result(monkeypatch):
    fake_docker(monkeypatch, 1, b"AssertionError: page 2 repeats an item")

    result = asyncio.run(DockerSandbox("abc123", "/workspace").exec("pytest"))

    assert (result.exit_code, result.timed_out) == (1, False)
