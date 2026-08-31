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
        if "VERDICT:" in prompt or "RUBRIC" in prompt:
            self.review_calls += 1
            if self.review_calls == 1:
                return "SCORE: 5\nVERDICT: REVISE\nCRITIQUE:\n- add PR #1 detail\n- add a code snippet"
            return "SCORE: 9\nVERDICT: PASS\nCRITIQUE: none"
        if "REWRITE the digest" in prompt or ("CURRENT DIGEST" in prompt and "REVIEWER CRITIQUE" in prompt):
            return "# owner/repo1\n\n## owner/repo1\n- Fixed per reviewer: added PR #1 detail.\n```python\nx=1\n```"
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
        accepted, critique, score = parse_review_verdict("SCORE: 9\nVERDICT: PASS")
        assert accepted is True
        assert critique == ""
        assert score == 9

    def test_revise(self):
        accepted, critique, score = parse_review_verdict("SCORE: 5\nVERDICT: REVISE\nCRITIQUE:\n- add details")
        assert accepted is False
        assert "add details" in critique
        assert score == 5

    def test_high_score_passes_even_without_flag(self):
        accepted, _, score = parse_review_verdict("SCORE: 8.5\nVERDICT: PASS")
        assert accepted is True
        assert score == 8.5