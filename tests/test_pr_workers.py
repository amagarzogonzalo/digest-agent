"""Tests for per-PR deep-dive sub-agents and report metadata."""

from digest_agent.core import (
    build_orchestrator,
    create_test_state,
    FakeActivitySource,
    FakeLLMProvider,
    FakeDocumentRenderer,
    FakeNotifier,
)
from digest_agent.ports import RepoActivity, Commit, PullRequest, FileChange, TestResult


def make_activity():
    return [RepoActivity(
        repo="owner/repo1",
        commits=[Commit(sha="abc123", message="Add feature", author="Alice")],
        pull_requests=[
            PullRequest(
                number=42, title="BigCode harness", state="closed", merged=True,
                files=[FileChange("eval/runner.py", "modified", additions=10, deletions=2,
                                  patch="@@ -1,3 +1,5 @@\n-def foo():\n+def foo(x):\n+    return x")],
                additions=10, deletions=2, changed_files=1,
            )
        ],
    )]


class FakeDeepLLM(FakeLLMProvider):
    """Returns a detailed dive for PR prompts."""

    def complete(self, model_id, prompt):
        if "PR TITLE" in prompt:
            return "## PR #42 -- BigCode harness\n\nDetailed dive body with code."
        if "engineering lead" in prompt:
            return "Overall narrative paragraph."
        return super().complete(model_id, prompt)


def make_orchestrator(fake_llm):
    return build_orchestrator(
        activity_source=FakeActivitySource(make_activity()),
        llm_provider=fake_llm,
        document_renderer=FakeDocumentRenderer(),
        notifier=FakeNotifier(),
    )


def test_pr_deep_dive_workers_populate_summary():
    fake = FakeDeepLLM(
        free_models=["model-a"],
        test_results={"model-a": TestResult("model-a", True, "ok")},
    )
    result = make_orchestrator(fake).invoke(create_test_state(repos=["owner/repo1"]))

    assert result.outcome == "success"
    assert "## PR #42 -- BigCode harness" in result.summary
    assert "Detailed dive body" in result.summary
    # The detailed dive should dominate; overall narrative also present.
    assert "Overall" in result.summary


def test_report_metadata_populated():
    fake = FakeLLMProvider(
        free_models=["model-a"],
        test_results={"model-a": TestResult("model-a", True, "ok")},
    )
    fake._completions = {"model-a": "# x"}
    result = make_orchestrator(fake).invoke(create_test_state(repos=["owner/repo1"]))

    assert result.report_uid
    assert result.report_version == "1.0.0"
    assert result.report_generated_at