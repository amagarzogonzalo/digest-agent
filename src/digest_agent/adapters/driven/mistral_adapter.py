"""Mistral adapter - implements LLMProviderPort for Mistral's API."""

from digest_agent.adapters.driven.openai_compat import OpenAICompatAdapter
from digest_agent.ports import ModelId


class MistralAdapter(OpenAICompatAdapter):
    """Adapter for Mistral (api.mistral.ai) - implements LLMProviderPort."""

    endpoint = "https://api.mistral.ai/v1"
    candidate_models: list[ModelId] = [
        "mistral-small-latest",
        "open-mistral-7b",
    ]