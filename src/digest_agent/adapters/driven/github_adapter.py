"""GitHub adapter - implements ActivitySourcePort for the hexagonal architecture.

This adapter encapsulates all GitHub-specific logic: API calls, pagination,
authentication, and response parsing. The core depends only on the port
interface, not on requests or GitHub SDKs directly."""
import requests
from datetime import datetime, timedelta, timezone

from digest_agent.ports import ActivitySourcePort, RepoActivity, Commit, PullRequest, ActivityFetchError


GITHUB_API = "https://api.github.com"


class GitHubAdapter(ActivitySourcePort):
    """Adapter for GitHub API - implements ActivitySourcePort."""

    def __init__(self, token: str, repos: str, lookback_days: int = 1):
        self.token = token
        self.repos = [r.strip() for r in repos.split(",") if r.strip()]
        self.lookback_days = lookback_days

    def fetch_activity(self, repos: list[str] | None = None, since: str | None = None) -> list[RepoActivity]:
        """Fetch commits and PRs for the given repos since the given ISO timestamp."""
        token = self.token
        repo_list = self.repos if repos is None else repos
        lookback = self.lookback_days if since is None else None

        if lookback:
            since_dt = datetime.now(timezone.utc) - timedelta(days=lookback)
            since = since_dt.isoformat()

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

        activity_by_repo: dict[str, dict] = {}
        for repo in repo_list:
            try:
                commits_resp = requests.get(
                    f"{GITHUB_API}/repos/{repo}/commits",
                    headers=headers,
                    params={"since": since, "per_page": 50},
                    timeout=30,
                )
                commits_resp.raise_for_status()
                commits = [
                    {
                        "sha": c["sha"][:7],
                        "message": c["commit"]["message"].split("\n")[0],
                        "author": c["commit"]["author"]["name"],
                    }
                    for c in commits_resp.json()
                ]

                prs_resp = requests.get(
                    f"{GITHUB_API}/repos/{repo}/pulls",
                    headers=headers,
                    params={"state": "all", "sort": "updated", "direction": "desc", "per_page": 20},
                    timeout=30,
                )
                prs_resp.raise_for_status()
                cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
                prs = [
                    {
                        "number": p["number"],
                        "title": p["title"],
                        "state": p["state"],
                        "merged": bool(p.get("merged_at")),
                    }
                    for p in prs_resp.json()
                    if datetime.fromisoformat(p["updated_at"].replace("Z", "+00:00")) >= cutoff
                ]

                activity_by_repo[repo] = {"commits": commits, "pull_requests": prs}
            except Exception as e:
                raise ActivityFetchError(f"GitHub API error for {repo}: {e}")

        result = []
        for repo, data in activity_by_repo.items():
            result.append(RepoActivity(
                repo=repo,
                commits=[Commit(**c) for c in data["commits"]],
                pull_requests=[PullRequest(**pr) for pr in data["pull_requests"]],
            ))

        return result