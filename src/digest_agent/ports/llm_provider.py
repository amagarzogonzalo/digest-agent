"""LLM provider port - lists free models, tests them, and completes prompts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


ModelId = str


@dataclass(frozen=True)
class TestResult:
    """Result of testing a candidate model."""
    model_id: ModelId
    passed: bool
    reason: str  # e.g., "ok", "timeout", "rate_limited", "empty_response", "malformed"


class LLMProviderPort(ABC):
    """Port for interacting with an LLM provider."""

    @abstractmethod
    def list_free_models(self) -> list[ModelId]:
        """Return a ranked list of currently-free model IDs.

        Returns:
            List of model IDs, best (largest context, most capable) first.
            Must not include paid models.
        """
        pass

    @abstractmethod
    def test(self, model_id: ModelId) -> TestResult:
        """Test a model with a cheap, fixed prompt.

        This is a lightweight health check, NOT the real summarization prompt.
        Keep it minimal to conserve free-tier quota.

        Args:
            model_id: The model to test.

        Returns:
            TestResult indicating pass/fail and reason.
        """
        pass

    @abstractmethod
    def complete(self, model_id: ModelId, prompt: str) -> str:
        """Complete a prompt using the specified model.

        Args:
            model_id: The model to use (must have passed test()).
            prompt: The full prompt to send.

        Returns:
            The model's completion text.

        Raises:
            CompletionError: If the provider returns an error or malformed response.
        """
        pass


class CompletionError(Exception):
    """Raised when completing a prompt fails."""
    pass