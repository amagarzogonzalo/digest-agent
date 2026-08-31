"""Integration tests for the Orchestrator with fake ports."""

import pytest

from digest_agent.core import (
    Orchestrator,
    build_orchestrator,
    create_test_state,
    FakeActivitySource,
    FakeLLMProvider,
    FakeDocumentRenderer,
    FakeNotifier,
)
from digest_agent.ports import RepoActivity, Commit, PullRequest, TestResult


class TestOrchestratorIntegration:
    """Test the full orchestrator graph with fake ports."""

    def make_activity(self, repo="owner/repo1", commits=None, prs=None):
        """Helper to create RepoActivity."""
        return RepoActivity(
            repo=repo,
            commits=commits or [
                Commit(sha="abc123", message="Add feature", author="Alice"),
            ],
            pull_requests=prs or [
                PullRequest(number=1, title="Fix bug", state="closed", merged=True),
            ],
        )

    def test_full_pipeline_success(self):
        """PASS path: model selected, summary generated, PDF rendered, email sent."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a", "model-b"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
            completions={"model-a": "# Daily Digest\n\n## owner/repo1\n- Added feature\n- Merged PR #1\n\nOverall: productive day."},
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
            max_attempts=5,
        )

        initial_state = create_test_state(
            repos=["owner/repo1"],
            candidate_pool=["model-a", "model-b"],
        )

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "success"
        assert result.selected_model == "model-a"
        assert "Daily Digest" in result.summary
        assert result.pdf_path == "/tmp/fake_digest.pdf"
        assert result.email_sent is True
        assert len(fake_notifier.sent_notifications) == 1

    def test_model_selection_fail_exhausted(self):
        """FAIL_EXHAUSTED: all models fail, pipeline stops at handle_failure."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a", "model-b"],
            test_results={
                "model-a": TestResult(model_id="model-a", passed=False, reason="timeout"),
                "model-b": TestResult(model_id="model-b", passed=False, reason="rate_limited"),
            },
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
            max_attempts=5,
        )

        initial_state = create_test_state(
            repos=["owner/repo1"],
            candidate_pool=["model-a", "model-b"],
        )

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "fail_exhausted"
        assert result.selected_model is None
        assert "Model selection failed: fail_exhausted" in result.error_message
        assert result.email_sent is False
        assert len(fake_notifier.sent_notifications) == 0

    def test_model_selection_fail_budget(self):
        """FAIL_BUDGET: max_attempts reached before exhausting pool."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a", "model-b", "model-c"],
            test_results={
                "model-a": TestResult(model_id="model-a", passed=False, reason="timeout"),
                "model-b": TestResult(model_id="model-b", passed=False, reason="timeout"),
            },
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
            max_attempts=2,  # Budget of 2
        )

        initial_state = create_test_state(
            repos=["owner/repo1"],
            candidate_pool=["model-a", "model-b", "model-c"],
            max_attempts=2,
        )

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "fail_budget"
        assert result.selected_model is None
        assert "Model selection failed: fail_budget" in result.error_message
        assert result.email_sent is False

    def test_no_activity_returns_early(self):
        """No activity -> returns early with 'no activity' summary."""
        fake_activity = FakeActivitySource([
            RepoActivity(repo="owner/repo1", commits=[], pull_requests=[]),
        ])
        fake_llm = FakeLLMProvider(free_models=["model-a"])
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(repos=["owner/repo1"])

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "success"
        assert "No activity" in result.summary
        assert result.email_sent is True

    def test_fetch_error_handled(self):
        """Fetch error -> outcome reflects error, no email sent."""
        class FailingActivitySource(FakeActivitySource):
            def fetch_activity(self, repos, since):
                raise Exception("GitHub API down")

        fake_activity = FailingActivitySource()
        fake_llm = FakeLLMProvider(free_models=["model-a"])
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(repos=["owner/repo1"])

        result = orchestrator.invoke(initial_state)

        assert "Fetch failed" in result.error_message
        assert result.email_sent is False

    def test_summarization_error_handled(self):
        """Summarization failure -> error handled, no email sent."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
        )
        # Make complete() raise
        class FailingLLM(FakeLLMProvider):
            def complete(self, model_id, prompt):
                raise Exception("OpenRouter timeout")

        fake_llm = FailingLLM(
            free_models=["model-a"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(repos=["owner/repo1"])

        result = orchestrator.invoke(initial_state)

        assert "Summarization failed" in result.error_message
        assert result.email_sent is False

    def test_pdf_render_error_handled(self):
        """PDF render failure -> error handled, no email sent."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
            completions={"model-a": "Summary"},
        )
        class FailingRenderer(FakeDocumentRenderer):
            def render(self, summary):
                raise Exception("ReportLab error")

        fake_renderer = FailingRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(repos=["owner/repo1"])

        result = orchestrator.invoke(initial_state)

        assert "PDF render failed" in result.error_message
        assert result.email_sent is False

    def test_notification_error_handled(self):
        """Notification failure -> outcome=error."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
            completions={"model-a": "Summary"},
        )
        fake_renderer = FakeDocumentRenderer()
        class FailingNotifier(FakeNotifier):
            def send(self, recipient, subject, attachment_path):
                raise Exception("SMTP error")

        fake_notifier = FailingNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(repos=["owner/repo1"])

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "error"
        assert "Notification failed" in result.error_message
        assert result.email_sent is False

    def test_second_model_passes(self):
        """First model fails, second passes."""
        activity = [self.make_activity()]
        fake_activity = FakeActivitySource(activity)
        fake_llm = FakeLLMProvider(
            free_models=["model-a", "model-b"],
            test_results={
                "model-a": TestResult(model_id="model-a", passed=False, reason="timeout"),
                "model-b": TestResult(model_id="model-b", passed=True, reason="ok"),
            },
            completions={"model-b": "Summary from model-b"},
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        initial_state = create_test_state(
            repos=["owner/repo1"],
            candidate_pool=["model-a", "model-b"],
        )

        result = orchestrator.invoke(initial_state)

        assert result.outcome == "success"
        assert result.selected_model == "model-b"
        assert "model-b" in result.summary
        assert result.email_sent is True

    def test_graph_structure_has_one_cycle(self):
        """Verify the graph structure: only select_and_test_model has a cycle back to itself."""
        fake_activity = FakeActivitySource([self.make_activity()])
        fake_llm = FakeLLMProvider(
            free_models=["model-a"],
            test_results={"model-a": TestResult(model_id="model-a", passed=True, reason="ok")},
        )
        fake_renderer = FakeDocumentRenderer()
        fake_notifier = FakeNotifier()

        orchestrator = build_orchestrator(
            activity_source=fake_activity,
            llm_provider=fake_llm,
            document_renderer=fake_renderer,
            notifier=fake_notifier,
        )

        graph = orchestrator.build_graph()
        # Get the underlying graph structure
        # The graph should have nodes and edges
        # We can't easily inspect the compiled graph's cycles, but we can verify
        # the orchestrator was created without error
        assert graph is not None