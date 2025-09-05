from __future__ import annotations

import asyncio
import os

from agents import Agent, AzureProvider, RunConfig, Runner, function_tool, set_tracing_disabled

API_KEY = os.getenv("AZURE_API_KEY") or ""
API_VERSION = os.getenv("AZURE_API_VERSION") or ""
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT") or ""
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT") or ""

if not all([API_KEY, API_VERSION, AZURE_ENDPOINT, AZURE_DEPLOYMENT]):
    raise ValueError(
        "Please set AZURE_API_KEY, AZURE_API_VERSION, AZURE_ENDPOINT, and AZURE_DEPLOYMENT "
        "via env var or code."
    )


# This example shows how to use the AzureProvider to connect to an Azure OpenAI deployment.
# Note that in this example, we disable tracing under the assumption that you don't have an API key
# from platform.openai.com. If you do have one, you can either set the `OPENAI_API_KEY` env var
# or call set_tracing_export_api_key() to set a tracing specific key.
set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    """Gets the weather for a given city."""
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main() -> None:
    azure_provider = AzureProvider(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
    )

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
        model_provider=azure_provider,
    )

    result = await Runner.run(
        agent,
        "What's the weather in Tokyo?",
        run_config=RunConfig(model_provider=azure_provider),
    )
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
