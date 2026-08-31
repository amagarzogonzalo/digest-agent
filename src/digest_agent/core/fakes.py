"""In-memory fake implementations of ports for testing the core."""

from digest_agent.ports import (
    ActivitySourcePort,
    RepoActivity,
    Commit,
    PullRequest,
    LLMProviderPort,
    ModelId,
    TestResult,
    DocumentRendererPort,
    NotifierPort,
)
from digest_agent.core.state import DigestState


class FakeActivitySource(ActivitySourcePort):
    """Fake activity source for testing."""

    def __init__(self, activity: list[RepoActivity] | None = None):
        self._activity = activity or []
        self.called_with: tuple[list[str], str] | None = None

    def fetch_activity(self, repos: list[str], since: str) -> list[RepoActivity]:
        self.called_with = (repos, since)
        return self._activity


class FakeLLMProvider(LLMProviderPort):
    """Fake LLM provider for testing model selection logic."""

    def __init__(
        self,
        free_models: list[ModelId] | None = None,
        test_results: dict[ModelId, TestResult] | None = None,
        completions: dict[ModelId, str] | None = None,
    ):
        self._free_models = free_models or ["model-a", "model-b", "model-c"]
        self._test_results = test_results or {}
        self._completions = completions or {}
        self.list_free_models_called = 0
        self.test_called_with: list[ModelId] = []
        self.complete_called_with: list[tuple[ModelId, str]] = []

    def list_free_models(self) -> list[ModelId]:
        self.list_free_models_called += 1
        return self._free_models

    def test(self, model_id: ModelId) -> TestResult:
        self.test_called_with.append(model_id)
        if model_id in self._test_results:
            return self._test_results[model_id]
        # Default: pass
        return TestResult(model_id=model_id, passed=True, reason="ok")

    def complete(self, model_id: ModelId, prompt: str) -> str:
        self.complete_called_with.append((model_id, prompt))
        if "VERDICT:" in prompt:
            # Reviewer prompt -> pass by default so tests don't loop.
            return "VERDICT: PASS"
        if model_id in self._completions:
            return self._completions[model_id]
        return f"Summary from {model_id}"


class FakeDocumentRenderer(DocumentRendererPort):
    """Fake document renderer for testing."""

    def __init__(self):
        self.rendered_summaries: list[str] = []
        self.output_path = "/tmp/fake_digest.pdf"

    def render(self, summary: str) -> str:
        self.rendered_summaries.append(summary)
        return self.output_path


class FakeNotifier(NotifierPort):
    """Fake notifier for testing."""

    def __init__(self):
        self.sent_notifications: list[tuple[str, str, str]] = []

    def send(self, recipient: str, subject: str, attachment_path: str) -> None:
        self.sent_notifications.append((recipient, subject, attachment_path))


def create_test_state(
    repos: list[str] | None = None,
    activity: list[RepoActivity] | None = None,
    candidate_pool: list[ModelId] | None = None,
    max_attempts: int = 5,
) -> DigestState:
    """Create a DigestState pre-populated for testing."""
    state = DigestState(
        repos=repos or ["owner/repo1"],
        lookback_days=1,
        max_attempts=max_attempts,
    )
    state.compute_since()
    if activity:
        state.activity = activity
    if candidate_pool:
        state.candidate_pool = candidate_pool
    return state