"""Tests for the deterministic (git-only) summary and diff-aware activity."""

from digest_agent.core.deterministic import build_deterministic_summary
from digest_agent.ports import RepoActivity, Commit, PullRequest, FileChange


def make_activity():
    return [RepoActivity(
        repo="owner/repo1",
        commits=[Commit(sha="a1b2c3", message="feat: add sandbox", author="Alice")],
        pull_requests=[
            PullRequest(
                number=42, title="BigCode harness", state="closed", merged=True,
                files=[
                    FileChange("eval/runner.py", "modified", additions=20, deletions=5),
                    FileChange("eval/harness.py", "added", additions=60, deletions=0),
                ],
                additions=80, deletions=5, changed_files=2,
            )
        ],
    )]


def test_deterministic_summary_contains_totals():
    text = build_deterministic_summary(make_activity())
    assert "Executive Snapshot" in text
    assert "Commits" in text and "1" in text
    assert "Pull requests" in text
    assert "+80" in text and "-5" in text
    assert "eval/runner.py" in text
    assert "PR #42" in text
    assert "no AI involved" in text


def test_deterministic_empty_activity():
    text = build_deterministic_summary([RepoActivity(repo="r", commits=[], pull_requests=[])])
    assert "Commits" in text
    assert "none" in text


def test_deterministic_short_refs():
    activity = [RepoActivity(
        repo="r", commits=[
            Commit(sha=f"{i:07x}", message=f"m{i}", author="x") for i in range(6)
        ], pull_requests=[])]
    text = build_deterministic_summary(activity)
    assert "+1 more" in text