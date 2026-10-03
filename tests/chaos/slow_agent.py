"""A test agent that spends money through the gateway, pauses, then applies the fix, so a
crash lands while trials are in flight and their spend is on record."""

import asyncio

from anthropic import AsyncAnthropic

from agentoscopy.adapters.base import AgentResult


class SlowAgent:
    name = "slow"

    def __init__(self) -> None:
        self.pause_s = 0.0

    async def setup(self, config):
        self.pause_s = float(config.params["pause_s"])

    async def run(self, task, sandbox, model_endpoint, budget, recorder):
        async with AsyncAnthropic(
            base_url=model_endpoint.base_url, api_key=model_endpoint.token
        ) as client:
            await client.messages.create(
                model="mock", max_tokens=64, messages=[{"role": "user", "content": "plan"}]
            )
        await asyncio.sleep(self.pause_s)
        await sandbox.write_file("app.py", b"fixed")
        return AgentResult(final_message="fixed")

    async def teardown(self):
        pass
