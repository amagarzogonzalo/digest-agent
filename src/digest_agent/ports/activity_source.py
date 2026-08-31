"""Activity source port - fetches repository activity (commits, PRs)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Commit:
    sha: str
    message: str
    author: str


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    state: str
    merged: bool


@dataclass(frozen=True)
class RepoActivity:
    repo: str
    commits: list[Commit]
    pull_requests: list[PullRequest]


class ActivitySourcePort(ABC):
    """Port for fetching recent activity from a git hosting provider."""

    @abstractmethod
    def fetch_activity(self, repos: list[str], since: str) -> list[RepoActivity]:
        """Fetch commits and PRs for the given repos since the given ISO timestamp.

        Args:
            repos: List of repository identifiers (e.g., "owner/repo").
            since: ISO 8601 timestamp to filter activity from.

        Returns:
            List of RepoActivity, one per repo (may be empty if no activity).

        Raises:
            ActivityFetchError: If the provider API returns an error or auth fails.
        """
        pass


class ActivityFetchError(Exception):
    """Raised when fetching activity from the source fails."""
    pass