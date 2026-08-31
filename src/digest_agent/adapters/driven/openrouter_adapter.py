"""OpenRouter adapter - implements LLMProviderPort for the hexagonal architecture.

This adapter encapsulates all OpenRouter-specific logic: model catalog fetching,
test calls, completion calls, and rate-limit handling. The core depends only
on the port interface, not on requests or OpenRouter SDKs directly."""
import requests
from typing import List

from digest_agent.ports import LLMProviderPort, ModelId, TestResult, CompletionError


MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
CHAT_COMPLETIONS_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAdapter(LLMProviderPort):
    """Adapter for OpenRouter API - implements LLMProviderPort."""

    def __init__(self, api_key: str, model: str | None = None, timeout: int = 15):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._free_models_cache: list[ModelId] | None = None
        self._free_models_cache_time: float = 0

    def _fetch_free_models(self) -> list[ModelId]:
        """Fetch free models from OpenRouter's model catalog."""
        resp = requests.get(MODELS_ENDPOINT, timeout=self.timeout)
        resp.raise_for_status()
        models = resp.json().get("data", [])

        free = []
        for m in models:
            pricing = m.get("pricing", {})
            try:
                prompt_price = float(pricing.get("prompt", "1"))
                completion_price = float(pricing.get("completion", "1"))
            except (TypeError, ValueError):
                continue

            if prompt_price != 0 or completion_price != 0:
                continue

            # Skip modalities that won't do plain text well
            modality = m.get("architecture", {}).get("modality", "")
            if modality and "text->text" not in modality:
                continue

            free.append(m["id"])

        # Sort by context length (best first)
        free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)

        return [m for m in free]

    def list_free_models(self) -> list[ModelId]:
        """Return a ranked list of currently-free model IDs."""
        import time
        # Cache for 30 minutes to avoid hitting rate limits
        if self._free_models_cache is None or time.time() - self._free_models_cache_time > 1800:
            self._free_models_cache = self._fetch_free_models()
            self._free_models_cache_time = time.time()
        return self._free_models_cache

    def test(self, model_id: ModelId) -> TestResult:
        """Test a model with a cheap, fixed prompt.

        The prompt is designed to:
        1. Be minimal cost (just "Hi" or simple question)
        2. Check basic response quality
        3. Return passed/failed based on both connectivity AND response quality

        The test prompt asks the model to respond with a brief, well-formed sentence.
        Models that respond with error messages, empty responses, or very short
        responses are flagged as failed so the evaluator-optimizer loop tries the
        next candidate rather than accepting a model that won't produce good output.
        """
        try:
            test_prompt = """You are a helpful assistant. Respond with a brief, well-formed sentence about the topic. 
            Do not repeat the prompt, do not include error messages, and make at least 5 words."""

            resp = requests.post(
                CHAT_COMPLETIONS_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": test_prompt}],
                },
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                return TestResult(model_id=model_id, passed=False, reason="rate_limited")

            resp.raise_for_status()
            body = resp.json()

            if "choices" in body and len(body["choices"]) > 0:
                content = body["choices"][0].get("message", {}).get("content", "").strip()

                # Quality checks
                if not content:
                    return TestResult(model_id=model_id, passed=False, reason="empty_response")

                if len(content) < 10:
                    return TestResult(model_id=model_id, passed=False, reason="response_too_short")

                if content.startswith("Error") or content.startswith("Error:"):
                    return TestResult(model_id=model_id, passed=False, reason="error_masked")

                # Passed all checks
                return TestResult(model_id=model_id, passed=True, reason="ok")

            return TestResult(model_id=model_id, passed=False, reason="no_choices")

        except requests.exceptions.Timeout:
            return TestResult(model_id=model_id, passed=False, reason="timeout")
        except Exception as e:
            if "429" in str(e).lower() or "rate limit" in str(e).lower():
                return TestResult(model_id=model_id, passed=False, reason="rate_limited")
            return TestResult(model_id=model_id, passed=False, reason="error")

    def complete(self, model_id: ModelId, prompt: str) -> str:
        """Complete a prompt using the specified model."""
        try:
            resp = requests.post(
                CHAT_COMPLETIONS_ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            if resp.status_code == 429:
                raise CompletionError("Rate limited during completion")
            resp.raise_for_status()
            body = resp.json()
            if "choices" in body and len(body["choices"]) > 0:
                return body["choices"][0]["message"]["content"]
            raise CompletionError("No response choices from OpenRouter")
        except requests.exceptions.Timeout:
            raise CompletionError("Timeout during completion")
        except Exception as e:
            raise CompletionError(f"Completion failed: {e}")