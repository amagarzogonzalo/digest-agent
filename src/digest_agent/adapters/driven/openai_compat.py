"""Base adapter for OpenAI-compatible chat-completion providers.

Groq and Mistral expose OpenAI-style endpoints, so they share test/complete
logic. Subclasses only set the endpoint and the candidate model list. The
core depends only on the port, never on this base or on `requests` directly.
"""
import time
import requests

from digest_agent.ports import LLMProviderPort, ModelId, TestResult, CompletionError


TEST_PROMPT = (
    "You are a helpful assistant. Respond with a brief, well-formed sentence "
    "about the topic. Do not repeat the prompt, do not include error messages, "
    "and make at least 5 words."
)


class RateLimitError(Exception):
    """Raised when the provider returns HTTP 429."""


class OpenAICompatAdapter(LLMProviderPort):
    """OpenAI-compatible chat completions provider adapter."""

    endpoint: str = ""
    candidate_models: list[ModelId] = []

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self._free_models_cache: list[ModelId] | None = None
        self._free_models_cache_time: float = 0

    # -- shared HTTP plumbing --

    def _chat(self, model_id: ModelId, messages: list[dict], timeout: int) -> str:
        resp = requests.post(
            f"{self.endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model_id, "messages": messages},
            timeout=timeout,
        )
        if resp.status_code == 429:
            raise RateLimitError("rate_limited")
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices") or []
        if not choices:
            raise CompletionError("No response choices from provider")
        return choices[0].get("message", {}).get("content", "")

    # -- LLMProviderPort --

    def list_free_models(self) -> list[ModelId]:
        """Return the provider's candidate models (cached)."""
        if self._free_models_cache is None or time.time() - self._free_models_cache_time > 1800:
            self._free_models_cache = list(self.candidate_models)
            self._free_models_cache_time = time.time()
        return self._free_models_cache

    def test(self, model_id: ModelId) -> TestResult:
        """Cheap health-check prompt against the model."""
        try:
            content = self._chat(
                model_id,
                [{"role": "user", "content": TEST_PROMPT}],
                timeout=self.timeout,
            ).strip()
            if not content:
                return TestResult(model_id=model_id, passed=False, reason="empty_response")
            if len(content) < 10:
                return TestResult(model_id=model_id, passed=False, reason="response_too_short")
            if content.lower().startswith("error"):
                return TestResult(model_id=model_id, passed=False, reason="error_masked")
            return TestResult(model_id=model_id, passed=True, reason="ok")
        except requests.exceptions.Timeout:
            return TestResult(model_id=model_id, passed=False, reason="timeout")
        except RateLimitError:
            return TestResult(model_id=model_id, passed=False, reason="rate_limited")
        except Exception as e:
            if "429" in str(e).lower() or "rate limit" in str(e).lower():
                return TestResult(model_id=model_id, passed=False, reason="rate_limited")
            return TestResult(model_id=model_id, passed=False, reason="error")

    def complete(self, model_id: ModelId, prompt: str) -> str:
        """Complete a prompt using the specified model."""
        try:
            return self._chat(model_id, [{"role": "user", "content": prompt}], timeout=60).strip()
        except requests.exceptions.Timeout:
            raise CompletionError("Timeout during completion")
        except RateLimitError:
            raise CompletionError("Rate limited during completion")
        except Exception as e:
            raise CompletionError(f"Completion failed: {e}")