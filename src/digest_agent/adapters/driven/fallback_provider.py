"""Fallback provider - tries multiple LLM providers in order through the ports.

Routes model IDs to whichever underlying provider owns them, so the
evaluator-optimizer loop in the orchestrator tests Groq models first,
then Mistral, then OpenRouter, and picks the first that works.

Each underlying provider only exposes free models, so the chain stays free:
- Groq / Mistral: fixed no-cost candidate lists
- OpenRouter: /models catalog filtered to a price of 0
"""
import time

from digest_agent.ports import (
    LLMProviderPort,
    ModelId,
    TestResult,
    CompletionError,
)


class FallbackLLMProvider(LLMProviderPort):
    """Chains several providers; routes each model to its owner provider."""

    def __init__(self, providers: list[LLMProviderPort]):
        self.providers = providers
        self._owner: dict[ModelId, LLMProviderPort] = {}
        self._models_cache: list[ModelId] | None = None
        self._models_cache_time: float = 0

    def list_free_models(self) -> list[ModelId]:
        """Return all free models, providers in the order they were given."""
        if self._models_cache is None or time.time() - self._models_cache_time > 1800:
            pool: list[ModelId] = []
            for provider in self.providers:
                try:
                    for model in provider.list_free_models():
                        pool.append(model)
                        self._owner.setdefault(model, provider)
                except Exception:
                    # A provider without valid creds should not break the chain.
                    continue
            self._models_cache = pool
            self._models_cache_time = time.time()
        return self._models_cache

    def _provider_for(self, model_id: ModelId) -> LLMProviderPort:
        """Look up which provider owns a model."""
        if model_id in self._owner:
            return self._owner[model_id]
        for provider in self.providers:
            try:
                if model_id in provider.list_free_models():
                    self._owner[model_id] = provider
                    return provider
            except Exception:
                continue
        raise CompletionError(f"No configured provider exposes model {model_id}")

    def test(self, model_id: ModelId) -> TestResult:
        return self._provider_for(model_id).test(model_id)

    def complete(self, model_id: ModelId, prompt: str) -> str:
        return self._provider_for(model_id).complete(model_id, prompt)