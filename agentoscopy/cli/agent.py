"""`agentoscopy agent check <config>`: check an agent config before a real run (WF-02)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentoscopy.adapters.python import AdapterLoadError, load_python_adapter
from agentoscopy.agent_check import CheckReport, check_agent
from agentoscopy.cli.common import (
    DB_NAME,
    EXIT_CHECK_FAILED,
    EXIT_ENVIRONMENT,
    EXIT_INVALID_INPUT,
    EXIT_OK,
)
from agentoscopy.cli.run import connect_models
from agentoscopy.gateway.access import ModelAccess
from agentoscopy.sandbox.docker import DockerBackend
from agentoscopy.spec import AgentConfig, SpecError, load_agent_config
from agentoscopy.storage.store import Store


def agent_check_command(args: argparse.Namespace) -> int:
    try:
        config = load_agent_config(args.config)  # only YAML; no agent code runs yet
    except SpecError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    # The proxy takes the API key out of this process before the agent's code is imported.
    models, failure = connect_models(args)
    if models is None:
        return failure
    store = Store(args.home / DB_NAME)
    try:
        try:
            adapter = load_python_adapter(config.entrypoint)
        except AdapterLoadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_INVALID_INPUT
        digest = store.save_config(config)
        print(f"checking agent {config.name} (config {digest[:12]}) on the smoke task", flush=True)
        return _print(asyncio.run(_check(config, adapter, args.home, models)))
    finally:
        store.close()
        models.stop()


async def _check(config: AgentConfig, adapter, home: Path, models: ModelAccess) -> CheckReport:
    gateway = models.gateway()
    await gateway.start()
    try:
        return await check_agent(config, adapter, DockerBackend(), gateway, home / "checks")
    finally:
        await gateway.stop()


def _print(report: CheckReport) -> int:
    result = report.result
    print(f"  model calls through the gateway: {report.model_calls}")
    print(f"  sandbox actions and messages: {report.agent_events}")
    print(
        f"  ended: {result.termination or result.outcome} after {result.duration_s:.1f} s, "
        f"${result.usage.cost_usd:.4f}"
    )
    for error in report.agent_errors:
        print(f"  the agent raised: {error}")
    if result.outcome in ("pass", "fail"):
        print(f"  smoke task: {result.outcome} (reported only; it does not decide the check)")
    print(f"  trajectory: {result.trajectory_path}")
    if report.ok:
        print("ok: the config is ready to run")
        return EXIT_OK
    for code, message in report.problems:
        print(f"FAIL {code}: {message}")
    infra = result.outcome == "infra_error"
    return EXIT_ENVIRONMENT if infra else EXIT_CHECK_FAILED
