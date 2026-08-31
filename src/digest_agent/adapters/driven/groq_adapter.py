"""Groq adapter - implements LLMProviderPort for Groq's OpenAI-compatible API."""

from digest_agent.adapters.driven.openai_compat import OpenAICompatAdapter
from digest_agent.ports import ModelId


class GroqAdapter(OpenAICompatAdapter):
    """Adapter for Groq (api.groq.com) - implements LLMProviderPort."""

    endpoint = "https://api.groq.com/openai/v1"
    candidate_models: list[ModelId] = [
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-prompt-guard-2-86m",
    ]