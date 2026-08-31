"""GitHub adapter - implements ActivitySourcePort for the hexagonal architecture.

This adapter encapsulates all GitHub-specific logic: API calls, pagination,
authentication, and response parsing - including diff/file detail so the
orchestrator can produce detailed digests. The core depends only on the port
interface, not on requests or GitHub SDKs directly."""
import requests
from datetime import datetime, timedelta, timezone

from digest_agent.ports import (
    ActivitySourcePort,
    RepoActivity,
    Commit,
    PullRequest,
    FileChange,
    ActivityFetchError,
)


GITHUB_API = "https://api.github.com"

MAX_DETAIL_COMMITS = 12     # per repo, cap commit-detail calls
MAX_PR_FILES = 100
MAX_DETAIL_PATCH_LEN = 3000  # max patch chars per file for the LLM
MAX_DETAIL_FILES = 40        # max files with patch text to attach


class GitHubAdapter(ActivitySourcePort):
    """Adapter for GitHub API - implements ActivitySourcePort."""

    def __init__(self, token, repos, lookback_days=1, include_diffs=True):
        self.token = token
        self.repos = [r.strip() for r in repos.split(",") if r.strip()]
        self.lookback_days = lookback_days
        self.include_diffs = include_diffs

    @staticmethod
    def _headers(token):
        return {
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
        }

    @staticmethod
    def _file_change(file, include_patch):
        patch = ""
        if include_patch and file.get("patch"):
            patch = file["patch"][:MAX_DETAIL_PATCH_LEN]
        return FileChange(
            filename=file.get("filename", ""),
            status=file.get("status", "modified"),
            additions=int(file.get("additions", 0) or 0),
            deletions=int(file.get("deletions", 0) or 0),
            patch=patch,
        )

    def _commit_files(self, token, repo, sha):
        """Fetch a single commit's changed files (diffstat + patches)."""
        try:
            resp = requests.get(
                "{GITHUB_API}/repos/{repo}/commits/{sha}".format(
                    GITHUB_API=GITHUB_API, repo=repo, sha=sha),
                headers=self._headers(token),
                timeout=30,
            )
            resp.raise_for_status()
            return [self._file_change(f, self.include_diffs)
                    for f in resp.json().get("files", [])][:MAX_DETAIL_FILES]
        except Exception:
            return []

    def _pull_files(self, token, repo, number):
        """Fetch a pull request's files and aggregate diffstat."""
        try:
            resp = requests.get(
                "{GITHUB_API}/repos/{repo}/pulls/{number}/files".format(
                    GITHUB_API=GITHUB_API, repo=repo, number=number),
                headers=self._headers(token),
                params={"per_page": MAX_PR_FILES},
                timeout=30,
            )
            resp.raise_for_status()
            changes = [self._file_change(f, self.include_diffs) for f in resp.json()]
            return changes, sum(c.additions for c in changes), sum(c.deletions for c in changes)
        except Exception:
            return [], 0, 0

    def fetch_activity(self, repos=None, since=None):
        """Fetch commits, PRs, and (optionally) diffs since the given timestamp."""
        token = self.token
        repo_list = self.repos if repos is None else repos
        lookback = self.lookback_days if since is None else None

        if lookback:
            since = (datetime.now(timezone.utc) - timedelta(days=lookback)).isoformat()

        headers = self._headers(token)
        activity_by_repo = {}

        for repo in repo_list:
            try:
                commits_resp = requests.get(
                    "{GITHUB_API}/repos/{repo}/commits".format(
                        GITHUB_API=GITHUB_API, repo=repo),
                    headers=headers,
                    params={"since": since, "per_page": 50},
                    timeout=30,
                )
                commits_resp.raise_for_status()
                commit_dicts = commits_resp.json()

                detailed_commits = []
                for i, c in enumerate(commit_dicts[:MAX_DETAIL_COMMITS]):
                    rec = {
                        "sha": c["sha"][:7],
                        "message": c["commit"]["message"].split("\n")[0],
                        "author": c["commit"]["author"]["name"],
                    }
                    if self.include_diffs:
                        rec["files"] = self._commit_files(token, repo, c["sha"])
                    detailed_commits.append(rec)
                for c in commit_dicts[MAX_DETAIL_COMMITS:]:
                    detailed_commits.append({
                        "sha": c["sha"][:7],
                        "message": c["commit"]["message"].split("\n")[0],
                        "author": c["commit"]["author"]["name"],
                    })

                prs_resp = requests.get(
                    "{GITHUB_API}/repos/{repo}/pulls".format(
                        GITHUB_API=GITHUB_API, repo=repo),
                    headers=headers,
                    params={"state": "all", "sort": "updated", "direction": "desc", "per_page": 20},
                    timeout=30,
                )
                prs_resp.raise_for_status()
                cutoff = datetime.fromisoformat(since.replace("Z", "+00:00"))
                prs = []
                for p in prs_resp.json():
                    if datetime.fromisoformat(p["updated_at"].replace("Z", "+00:00")) < cutoff:
                        continue
                    pr_rec = {
                        "number": p["number"],
                        "title": p["title"],
                        "state": p["state"],
                        "merged": bool(p.get("merged_at")),
                    }
                    if self.include_diffs:
                        files, adds, dels = self._pull_files(token, repo, p["number"])
                        pr_rec["files"] = files
                        pr_rec["additions"] = adds
                        pr_rec["deletions"] = dels
                        pr_rec["changed_files"] = len(files)
                    prs.append(pr_rec)

                activity_by_repo[repo] = {"commits": detailed_commits, "pull_requests": prs}
            except Exception as e:
                raise ActivityFetchError("GitHub API error for {}: {}".format(repo, e))

        result = []
        for repo, data in activity_by_repo.items():
            result.append(RepoActivity(
                repo=repo,
                commits=[Commit(**c) for c in data["commits"]],
                pull_requests=[PullRequest(**pr) for pr in data["pull_requests"]],
            ))

        return result