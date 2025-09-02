from unittest.mock import AsyncMock

import litellm
import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.models.litellm_provider import LitellmProvider
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing


@pytest.fixture
def mock_litellm_response():
    """Return a mock response for litellm.acompletion."""
    msg = Message(role="assistant", content="Azure OpenAI response")
    choice = Choices(index=0, message=msg)
    return ModelResponse(choices=[choice], usage=Usage(0, 0, 0))


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_azure_helper_method(monkeypatch, mock_litellm_response):
    """Test the create_azure_model helper method in LitellmModel."""
    mock_acompletion = AsyncMock(return_value=mock_litellm_response)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    # Test the Azure factory method
    model = LitellmModel.create_azure_model(
        deployment_name="test-deployment",
        api_base="https://test.openai.azure.com/",
        api_version="2023-05-15",
        api_key="test-key",
    )

    # Verify model parameters
    assert model.model == "azure/test-deployment"
    assert model.api_base == "https://test.openai.azure.com/"
    assert model.api_version == "2023-05-15"
    assert model.api_key == "test-key"

    # Test the get_response method to ensure it works with Azure configuration
    result = await model.get_response(
        system_instructions="System instructions",
        input=[{"role": "user", "content": "Hello"}],
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
    )

    # Verify the result
    assert result.output == "Azure OpenAI response"

    # Verify litellm was called with correct parameters
    mock_acompletion.assert_called_once()
    _, kwargs = mock_acompletion.call_args

    assert kwargs["model"] == "azure/test-deployment"
    assert kwargs["api_base"] == "https://test.openai.azure.com/"
    assert kwargs["api_version"] == "2023-05-15"
    assert kwargs["api_key"] == "test-key"


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_litellm_provider_azure_support(monkeypatch, mock_litellm_response):
    """Test the Azure support in LitellmProvider."""
    mock_acompletion = AsyncMock(return_value=mock_litellm_response)
    monkeypatch.setattr(litellm, "acompletion", mock_acompletion)

    # Test the standard initialization with Azure parameters
    provider = LitellmProvider(
        api_base="https://test.openai.azure.com/", api_version="2023-05-15", api_key="test-key"
    )

    # Get a model and verify it passes the Azure parameters
    model = provider.get_model("azure/test-deployment")
    assert model.api_base == "https://test.openai.azure.com/"
    assert model.api_version == "2023-05-15"
    assert model.api_key == "test-key"

    # Test the get_response method
    result = await model.get_response(
        system_instructions="System instructions",
        input=[{"role": "user", "content": "Hello"}],
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
    )

    # Verify the result
    assert result.output == "Azure OpenAI response"

    # Reset the mock for the next test
    mock_acompletion.reset_mock()

    # Test the for_azure factory method
    azure_provider = LitellmProvider.for_azure(
        deployment_name="default-deployment",
        api_base="https://test.openai.azure.com/",
        api_version="2023-05-15",
        api_key="test-key",
    )

    # Test getting the default model (should use the deployment name)
    model = azure_provider.get_model(None)
    assert model.model == "azure/default-deployment"

    # Test the get_response method with the default model
    result = await model.get_response(
        system_instructions="System instructions",
        input=[{"role": "user", "content": "Hello"}],
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
    )

    # Verify the result and parameters
    assert result.output == "Azure OpenAI response"

    # Verify litellm was called with correct parameters
    mock_acompletion.assert_called_once()
    _, kwargs = mock_acompletion.call_args

    assert kwargs["model"] == "azure/default-deployment"
    assert kwargs["api_base"] == "https://test.openai.azure.com/"
    assert kwargs["api_version"] == "2023-05-15"
    assert kwargs["api_key"] == "test-key"


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
async def test_azure_parameters_in_fetch_response(monkeypatch):
    """Test that Azure parameters are correctly passed to litellm.acompletion."""
    captured_kwargs = {}

    async def capture_kwargs(model, messages=None, **kwargs):
        captured_kwargs.update(kwargs)
        msg = Message(role="assistant", content="ok")
        choice = Choices(index=0, message=msg)
        return ModelResponse(choices=[choice], usage=Usage(0, 0, 0))

    monkeypatch.setattr(litellm, "acompletion", capture_kwargs)

    # Create model with Azure configuration
    model = LitellmModel(
        model="azure/test-deployment",
        api_base="https://test.openai.azure.com/",
        api_version="2023-05-15",
        api_key="test-key",
        extra_param="should be included",  # Test that extra parameters are passed through
    )

    # Call get_response
    await model.get_response(
        system_instructions="Test",
        input=[{"role": "user", "content": "Hello"}],
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
    )

    # Verify that all parameters were passed
    assert captured_kwargs["api_base"] == "https://test.openai.azure.com/"
    assert captured_kwargs["api_version"] == "2023-05-15"
    assert captured_kwargs["api_key"] == "test-key"
    assert captured_kwargs["extra_param"] == "should be included"

    # Verify model name
    assert captured_kwargs["model"] == "azure/test-deployment"
