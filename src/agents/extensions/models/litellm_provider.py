from typing import Optional

from ...models.interface import Model, ModelProvider
from .litellm_model import LitellmModel

DEFAULT_MODEL: str = "gpt-4.1"


class LitellmProvider(ModelProvider):
    """A ModelProvider that uses LiteLLM to route to any model provider. You can use it via:
    ```python
    Runner.run(agent, input, run_config=RunConfig(model_provider=LitellmProvider()))
    ```
    See supported models here: [litellm models](https://docs.litellm.ai/docs/providers).

    If you're using models that require additional configuration like Azure OpenAI, you can
    provide the necessary parameters when initializing the provider:

    ```python
    # For Azure OpenAI
    provider = LitellmProvider(
        api_base="https://your-resource.openai.azure.com/",
        api_version="2023-05-15",
        api_key="your-api-key"
    )
    ```

    Alternatively, you can use the dedicated factory method for Azure:
    ```python
    provider = LitellmProvider.for_azure(
        deployment_name="your-deployment-name",  # Optional default deployment
        api_base="https://your-resource.openai.azure.com/",
        api_version="2023-05-15",
        api_key="your-api-key"
    )
    ```

    For other configurations, you can pass additional parameters that will be forwarded
    to the LitellmModel:
    ```python
    provider = LitellmProvider(base_url="https://your-custom-endpoint")
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize a LitellmProvider with optional configuration parameters.

        Args:
            api_key: API key for the model provider
            api_base: API base URL (especially useful for Azure OpenAI)
            api_version: API version (especially useful for Azure OpenAI)
            base_url: Alternative base URL for the API
            **kwargs: Additional parameters to pass to LitellmModel
        """
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.base_url = base_url
        self.kwargs = kwargs

    @classmethod
    def for_azure(
        cls,
        api_base: str,
        api_version: str,
        api_key: str,
        deployment_name: Optional[str] = None,
        **kwargs,
    ) -> "LitellmProvider":
        """Create a LitellmProvider configured for Azure OpenAI.

        Args:
            api_base: Azure endpoint URL (e.g., "https://example.openai.azure.com/")
            api_version: Azure API version (e.g., "2023-05-15")
            api_key: Azure API key
            deployment_name: Optional default Azure deployment name to use when no model is specified
            **kwargs: Additional parameters to pass to the provider

        Returns:
            A LitellmProvider instance configured for Azure OpenAI
        """
        provider = cls(api_base=api_base, api_version=api_version, api_key=api_key, **kwargs)
        provider.azure_deployment_name = deployment_name
        return provider

    def get_model(self, model_name: str | None) -> Model:
        """Get a model instance from this provider.

        Args:
            model_name: The name of the model to use, or None to use the default

        Returns:
            A Model instance configured with the specified parameters
        """
        if not model_name:
            model_name = DEFAULT_MODEL

        # Special handling for Azure when using the for_azure factory method
        if hasattr(self, "azure_deployment_name") and not model_name.startswith("azure/"):
            if model_name == DEFAULT_MODEL and self.azure_deployment_name:
                # Use the deployment name specified in for_azure if no specific model is requested
                model_name = f"azure/{self.azure_deployment_name}"

        return LitellmModel(
            model=model_name,
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
            base_url=self.base_url,
            **self.kwargs,
        )
