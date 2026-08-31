"""Deterministic (programmatic) summary built purely from fetched git data.

Two sections, both requiring no LLM:
  1. Executive Snapshot - totals + per-PR diffstat.
  2. Latest PR deep-dives - the most recent merged PRs, each listing every
     changed file with additions/deletions and the actual diff hunks fetched
     from git, so real integrated code is shown. Rich enough that the LLM
     summary builds directly on it.
"""
from datetime import datetime, timezone


def _short_refs(commits) -> str:
    refs = [c.sha for c in commits]
    if not refs:
        return "none"
    if len(refs) <= 5:
        return ", ".join(refs)
    return ", ".join(refs[:5]) + " (+" + str(len(refs) - 5) + " more)"


def _sort_prs(activity):
    """Flatten all PRs across repos, merged first, most recent first (by number)."""
    prs = [p for ra in activity for p in ra.pull_requests]
    prs.sort(key=lambda p: (-int(p.merged), -p.number))
    return prs


def latest_merged_prs(activity, n: int = 3) -> list:
    """Return the n most-recent MERGED pull requests across all repos, newest first."""
    merged = [p for ra in activity for p in ra.pull_requests if p.merged]
    merged.sort(key=lambda p: p.number, reverse=True)
    return merged[:max(0, n)]


def build_deterministic_summary(activity, deep_prs: int = 3) -> str:
    """Return a markdown snapshot including deep dives on the latest PRs."""
    total_commits = sum(len(ra.commits) for ra in activity)
    total_prs = sum(len(ra.pull_requests) for ra in activity)
    merged_prs = sum(1 for ra in activity for p in ra.pull_requests if p.merged)
    total_adds = sum(ra.additions for ra in activity)
    total_dels = sum(ra.deletions for ra in activity)
    all_files: set[str] = set()
    for ra in activity:
        all_files.update(ra.changed_files_set)

    files_txt = "none"
    if all_files:
        files_txt = ", ".join(sorted(all_files)[:12])
        if len(all_files) > 12:
            files_txt += " (+" + str(len(all_files) - 12) + " more)"

    sorted_prs = _sort_prs(activity)
    recent = [p for p in sorted_prs if p.merged][:deep_prs]

    lines = [
        "## Executive Snapshot (from git data)",
        "",
        "- **Repos**: " + ", ".join(ra.repo for ra in activity),
        "- **Commits**: " + str(total_commits) + "  (refs: " + _short_refs([c for ra in activity for c in ra.commits]) + ")",
        "- **Pull requests**: " + str(total_prs) + " total, " + str(merged_prs) + " merged",
        "- **Lines changed**: +" + str(total_adds) + " / \u2212" + str(total_dels),
        "- **Files touched** (" + str(len(all_files)) + "): " + files_txt,
    ]

    # Latest merged PR deep-dives with complete file lists + diff patches.
    lines.append("")
    lines.append("### \u2193 Latest " + str(len(recent)) + " merged PRs — full diff detail")
    for p in recent:
        repo = next((ra.repo for ra in activity for x in ra.pull_requests if x.number == p.number), "?")
        lines.append("")
        lines.append("#### PR #" + str(p.number) + " — " + p.title)
        lines.append(
            "Merged PR on `" + repo + "` - " + str(p.changed_files) + " file(s), "
            "+" + str(p.additions) + "/-" + str(p.deletions) + " lines.")
        if not p.files:
            lines.append("- files: not available")
        for f in p.files:
            status = f.status
            lines.append("- `" + f.filename + "`  (" + status + ", +" + str(f.additions) + "/-" + str(f.deletions) + ")")
            if f.patch:
                lines.append("  ```diff")
                lines.append("  " + f.patch.replace("\n", "\n  "))
                lines.append("  ```")

    # Compact per-PR stat for everything else (older PRs).
    older = [p for p in sorted_prs if p.merged and p not in recent]
    if older:
        lines.append("")
        lines.append("### Other merged PRs")
        for p in older:
            stat = ("  [" + str(p.changed_files) + " file(s), +" + str(p.additions) + "/-" + str(p.deletions) + "]" if p.changed_files else "")
            lines.append("- PR #" + str(p.number) + ": " + p.title + stat)

    lines.append("")
    lines.append("_Produced from git data; no AI involved. Diff hunks are verbatim from the repository._")
    return "\n".join(lines)