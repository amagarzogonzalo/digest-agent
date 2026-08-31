"""Tests for the summary review/refine loop and verdict parsing."""

import pytest

from digest_agent.core import (
    build_orchestrator,
    create_test_state,
    FakeActivitySource,
    FakeLLMProvider,
    FakeDocumentRenderer,
    FakeNotifier,
)
from digest_agent.core.text_utils import parse_review_verdict
from digest_agent.ports import RepoActivity, Commit, PullRequest, TestResult


class RevisingLLM(FakeLLMProvider):
    """Fake provider that asks for one revision, then approves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_calls = 0

    def complete(self, model_id, prompt):
        if "VERDICT:" in prompt:
            self.review_calls += 1
            if self.review_calls == 1:
                return "VERDICT: REVISE\n- be more specific about PR numbers\n- remove the duplicated title"
            return "VERDICT: PASS"
        if "improve" in prompt and "Current digest" in prompt:
            return "# owner/repo1\n\n## owner/repo1\n- Fixed per reviewer: added PR #1 detail."
        return super().complete(model_id, prompt)


def make_activity():
    return [RepoActivity(
        repo="owner/repo1",
        commits=[Commit(sha="abc123", message="Add feature", author="Alice")],
        pull_requests=[PullRequest(number=1, title="Fix bug", state="closed", merged=True)],
    )]


class TestReviewLoop:
    def test_accept_path_skips_refine(self):
        fake_llm = FakeLLMProvider(
            free_models=["model-a"],
            test_results={"model-a": TestResult("model-a", True, "ok")},
            completions={"model-a": "# Daily Digest\n\n## owner/repo1\n- item"},
        )
        orchestrator = build_orchestrator(
            activity_source=FakeActivitySource(make_activity()),
            llm_provider=fake_llm,
            document_renderer=FakeDocumentRenderer(),
            notifier=FakeNotifier(),
        )
        result = orchestrator.invoke(create_test_state(repos=["owner/repo1"]))
        assert "item" in result.summary

    def test_revise_then_accept_refines_summary(self):
        fake_llm = RevisingLLM(
            free_models=["model-a"],
            test_results={"model-a": TestResult("model-a", True, "ok")},
            completions={"model-a": "# Draft summary\n\n- generic content"},
        )
        orchestrator = build_orchestrator(
            activity_source=FakeActivitySource(make_activity()),
            llm_provider=fake_llm,
            document_renderer=FakeDocumentRenderer(),
            notifier=FakeNotifier(),
        )
        result = orchestrator.invoke(create_test_state(repos=["owner/repo1"]))
        assert fake_llm.review_calls >= 2
        assert "PR #1" in result.summary


class TestParseReviewVerdict:
    def test_pass(self):
        accepted, critique = parse_review_verdict("VERDICT: PASS")
        assert accepted is True
        assert critique == ""

    def test_revise(self):
        accepted, critique = parse_review_verdict("VERDICT: REVISE\n- add details")
        assert accepted is False
        assert "add details" in critique

    def test_fallback_without_flag(self):
        accepted, _ = parse_review_verdict("Looks good, no issues here.")
        assert accepted is True