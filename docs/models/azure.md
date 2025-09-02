
This guide explains how to use Azure OpenAI with the Agents SDK.

## Prerequisites

- An Azure account with access to Azure OpenAI Service
- An Azure OpenAI deployment
- The following information:
  - Azure API Key
  - Azure API Base URL
  - Azure Deployment Name
  - Azure API Version (default: "2023-05-15")

## Integration Methods

There are two main ways to use Azure OpenAI with the Agents SDK:

### Method 1: Using AsyncOpenAI client with Azure configuration

This method configures the SDK to use Azure OpenAI globally by setting up a custom AsyncOpenAI client:

```python
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api

# Create Azure OpenAI client
azure_client = AsyncOpenAI(
    api_key="your-azure-api-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment-name",
    api_version="2023-05-15",
)

# Set as default client
set_default_openai_client(client=azure_client)

# Most Azure deployments use Chat Completions API
set_default_openai_api("chat_completions")
```

### Method 2: Using LiteLLM integration with Azure models

This method uses the LiteLLM integration to access Azure OpenAI models on a per-agent basis:

```python
from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel

agent = Agent(
    model=LitellmModel(
        model="azure/your-deployment-name",
        api_key="your-azure-api-key",
        api_base="https://your-resource.openai.azure.com",
        api_version="2023-05-15",
    ),
    # ... other agent parameters
)
```

## Authentication Options

### API Key Authentication

The examples above use API key authentication, which is the simplest method.

### Microsoft Entra ID (formerly Azure AD) Authentication

For Entra ID authentication, you can use the Azure Identity library with LiteLLM:

```python
from azure.identity import DefaultAzureCredential
from agents.extensions.models.litellm_model import LitellmModel

# Get token using DefaultAzureCredential
default_credential = DefaultAzureCredential()
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

agent = Agent(
    model=LitellmModel(
        model="azure/your-deployment-name",
        api_base="https://your-resource.openai.azure.com",
        api_version="2023-05-15",
        azure_ad_token=token.token,
    ),
    # ... other agent parameters
)
```

## Full Example

See the [azure_example.py](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/azure_example.py) for a complete working example.
 EOF