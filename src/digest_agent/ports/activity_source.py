"""Activity source port - fetches repository activity (commits, PRs, diffs)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FileChange:
    """A file touched by a commit or PR, with diffstat and optional patch."""
    filename: str
    status: str = "modified"   # "added" | "modified" | "removed" | "renamed"
    additions: int = 0
    deletions: int = 0
    patch: str = ""           # unified diff hunks (may be truncated)


@dataclass(frozen=True)
class Commit:
    sha: str
    message: str
    author: str
    files: list[FileChange] = field(default_factory=list)


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    state: str
    merged: bool
    files: list[FileChange] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


@dataclass(frozen=True)
class RepoActivity:
    repo: str
    commits: list[Commit]
    pull_requests: list[PullRequest]

    @property
    def files_changed(self) -> int:
        return sum(pr.changed_files for pr in self.pull_requests)

    @property
    def additions(self) -> int:
        return sum(pr.additions for pr in self.pull_requests)

    @property
    def deletions(self) -> int:
        return sum(pr.deletions for pr in self.pull_requests)

    @property
    def changed_files_set(self) -> tuple[str, ...]:
        files: set[str] = set()
        for pr in self.pull_requests:
            for f in pr.files:
                files.add(f.filename)
        return tuple(sorted(files))


class ActivitySourcePort(ABC):
    """Port for fetching recent activity from a git hosting provider."""

    @abstractmethod
    def fetch_activity(self, repos: list[str], since: str) -> list[RepoActivity]:
        """Fetch commits, PRs, and diff details for the given repos.

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