"""Gemini adapter - implements LLMProviderPort for Google AI Studio (Gemini).

Uses the REST ``models:generateContent`` endpoint directly via `requests`
(no SDK dependency). Suitable free candidate models should be ordered best
first. Configure with GEMINI_API_KEY (AI Studio key).
"""
import time
import requests

from digest_agent.ports import LLMProviderPort, ModelId, TestResult, CompletionError
from digest_agent.core.text_utils import strip_reasoning_prefix


class GeminiRateLimit(Exception):
    """Raised on HTTP 429 from the Gemini API."""


class GeminiAdapter(LLMProviderPort):
    """Adapter for Google Gemini (AI Studio) - implements LLMProviderPort."""

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, candidate_models: list[ModelId] | None = None):
        self.api_key = api_key
        self.candidate_models = candidate_models or [
            "gemini-3.6-flash",
            "gemini-flash-lite-latest",
        ]
        self._cache: list[ModelId] | None = None
        self._cache_time: float = 0

    def _generate(self, model_id: ModelId, prompt: str, max_tokens: int, temperature: float = 0.4) -> str:
        if not self.api_key:
            raise CompletionError("GEMINI_API_KEY is not set")
        resp = requests.post(
            f"{self.BASE}/{model_id}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
            },
            timeout=120,
        )
        if resp.status_code == 429:
            raise GeminiRateLimit("quota_exceeded")
        if resp.status_code != 200:
            raise CompletionError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            # Possible safety block or empty response.
            reason = body.get("promptFeedback", {}).get("blockReason", "empty")
            raise CompletionError(f"Gemini returned no content ({reason})")
        return text

    # -- LLMProviderPort --

    def list_free_models(self) -> list[ModelId]:
        if self._cache is None or time.time() - self._cache_time > 1800:
            self._cache = list(self.candidate_models)
            self._cache_time = time.time()
        return self._cache

    def test(self, model_id: ModelId) -> TestResult:
        """Cheap health-check prompt against the model."""
        try:
            content = strip_reasoning_prefix(
                self._generate(model_id, "Reply with a short, well-formed sentence (at least 5 words).", 40, 0.0)
            ).strip()
            if not content:
                return TestResult(model_id, False, "empty_response")
            if len(content) < 10:
                return TestResult(model_id, False, "response_too_short")
            return TestResult(model_id, True, "ok")
        except GeminiRateLimit:
            return TestResult(model_id, False, "rate_limited")
        except requests.exceptions.Timeout:
            return TestResult(model_id, False, "timeout")
        except Exception:
            return TestResult(model_id, False, "error")

    def complete(self, model_id: ModelId, prompt: str) -> str:
        try:
            return strip_reasoning_prefix(self._generate(model_id, prompt, 3000, 0.4))
        except GeminiRateLimit as e:
            raise CompletionError(f"Gemini rate limited: {e}")
        except requests.exceptions.Timeout as e:
            raise CompletionError(f"Gemini timeout: {e}")