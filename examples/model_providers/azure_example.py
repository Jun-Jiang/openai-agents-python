# 1. Using AsyncOpenAI client with Azure configuration
# 2. Using LiteLLM integration with Azure models

import asyncio
import os
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
from agents.extensions.models.litellm_model import LitellmModel

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2023-05-15")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "")

# Example function tool
@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

async def azure_openai_client_example():
    """Example using AsyncOpenAI client with Azure configuration"""
    print("\n=== Using AsyncOpenAI client with Azure configuration ===")
    
    # Create Azure OpenAI client
    azure_client = AsyncOpenAI(
        api_key=AZURE_API_KEY,
        base_url=f"{AZURE_API_BASE}/openai/deployments/{AZURE_DEPLOYMENT_NAME}",
        api_version=AZURE_API_VERSION,
    )
    
    # Set as default client
    set_default_openai_client(client=azure_client, use_for_tracing=False)
    
    # Most Azure deployments use Chat Completions API
    set_default_openai_api("chat_completions")
    
    # Disable tracing for this example
    set_tracing_disabled(disabled=True)
    
    # Create agent using the Azure deployment
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that provides concise responses.",
        tools=[get_weather],
    )
    
    # Run the agent
    result = await Runner.run(agent, "What's the weather in Seattle?")
    print(result.final_output)

async def azure_litellm_example():
    """Example using LiteLLM integration with Azure models"""
    print("\n=== Using LiteLLM integration with Azure models ===")
    
    # Disable tracing for this example
    set_tracing_disabled(disabled=True)
    
    # Create agent using LiteLLM with Azure model
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant that provides concise responses.",
        model=LitellmModel(
            model=f"azure/{AZURE_DEPLOYMENT_NAME}",
            api_key=AZURE_API_KEY,
            api_base=AZURE_API_BASE,
            api_version=AZURE_API_VERSION,
        ),
        tools=[get_weather],
    )
    
    # Run the agent
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)

async def main():
    # Check if Azure configuration is set
    if not AZURE_API_KEY or not AZURE_API_BASE or not AZURE_DEPLOYMENT_NAME:
        print("Please set AZURE_API_KEY, AZURE_API_BASE, and AZURE_DEPLOYMENT_NAME environment variables.")
        print("Example: export AZURE_API_KEY=your-key")
        print("Example: export AZURE_API_BASE=https://your-resource.openai.azure.com")
        print("Example: export AZURE_DEPLOYMENT_NAME=your-deployment-name")
        return
    
    # Run both examples
    await azure_openai_client_example()
    await azure_litellm_example()

if __name__ == "__main__":
    asyncio.run(main())
