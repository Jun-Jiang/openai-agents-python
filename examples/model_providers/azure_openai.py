from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

"""This example demonstrates how to use Azure OpenAI models with the OpenAI Agents SDK.

You can run it like this:
uv run examples/model_providers/azure_openai.py --api-base https://your-resource.openai.azure.com/ --api-version 2023-05-15 --deployment-name your-deployment-name

Azure OpenAI requires these configuration parameters:
- api_base: Azure endpoint URL (e.g., "https://your-resource.openai.azure.com/")
- api_version: Azure API version (e.g., "2023-05-15") 
- api_key: Azure API key
- deployment_name: Your Azure OpenAI deployment name

Learn more about Azure OpenAI: https://docs.litellm.ai/docs/providers/azure
"""

set_tracing_disabled(disabled=True)


@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main(api_base: str, api_version: str, deployment_name: str, api_key: str):
    # The model name format for Azure OpenAI is "azure/<deployment-name>"
    model_name = f"azure/{deployment_name}"
    
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(
            model=model_name,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key
        ),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    # Get configuration from args or environment variables
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", type=str, 
                        default=os.environ.get("AZURE_API_BASE"))
    parser.add_argument("--api-version", type=str, 
                        default=os.environ.get("AZURE_API_VERSION"))
    parser.add_argument("--deployment-name", type=str, 
                        default=os.environ.get("AZURE_DEPLOYMENT_NAME"))
    parser.add_argument("--api-key", type=str, 
                        default=os.environ.get("AZURE_API_KEY"))
    args = parser.parse_args()

    api_base = args.api_base
    if not api_base:
        api_base = input("Enter your Azure API base (e.g., https://your-resource.openai.azure.com/): ")

    api_version = args.api_version
    if not api_version:
        api_version = input("Enter your Azure API version (e.g., 2023-05-15): ")
        
    deployment_name = args.deployment_name
    if not deployment_name:
        deployment_name = input("Enter your Azure deployment name: ")

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your Azure API key: ")

    asyncio.run(main(api_base, api_version, deployment_name, api_key))
