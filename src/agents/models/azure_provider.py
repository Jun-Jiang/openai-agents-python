from __future__ import annotations

import httpx
from openai import AsyncAzureOpenAI, DefaultAsyncHttpxClient

from . import _openai_shared
from .interface import Model, ModelProvider
from .openai_chatcompletions import OpenAIChatCompletionsModel
from .openai_responses import OpenAIResponsesModel

DEFAULT_MODEL: str = "gpt-4o"


_http_client: httpx.AsyncClient | None = None


# If we create a new httpx client for each request, that would mean no sharing of connection pools,
# which would mean worse latency and resource usage. So, we share the client across requests.
def shared_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = DefaultAsyncHttpxClient()
    return _http_client


class AzureProvider(ModelProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_version: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        azure_client: AsyncAzureOpenAI | None = None,
        use_responses: bool | None = None,
    ) -> None:
        if azure_client is not None:
            assert (
                api_key is None
                and api_version is None
                and azure_endpoint is None
                and azure_deployment is None
            ), "Don't provide Azure credentials if you provide azure_client"
            self._client: AsyncAzureOpenAI | None = azure_client
        else:
            self._client = None
            self._stored_api_key = api_key
            self._stored_api_version = api_version
            self._stored_azure_endpoint = azure_endpoint
            self._stored_azure_deployment = azure_deployment

        if use_responses is not None:
            self._use_responses = use_responses
        else:
            self._use_responses = _openai_shared.get_use_responses_by_default()

    def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            self._client = AsyncAzureOpenAI(
                api_key=self._stored_api_key or _openai_shared.get_default_openai_key(),
                api_version=self._stored_api_version,
                azure_endpoint=self._stored_azure_endpoint,
                azure_deployment=self._stored_azure_deployment,
                http_client=shared_http_client(),
            )
        return self._client

    def get_model(self, model_name: str | None) -> Model:
        if model_name is None:
            model_name = DEFAULT_MODEL

        client = self._get_client()

        return (
            OpenAIResponsesModel(model=model_name, openai_client=client)
            if self._use_responses
            else OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        )
