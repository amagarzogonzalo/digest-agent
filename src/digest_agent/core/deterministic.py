"""Deterministic (programmatic) summary built purely from fetched git data.

This is the ground-truth section shown ABOVE the LLM summary. It requires no
LLM call, so it is always accurate: totals, per-PR diffstats, and the most
changed files. The LLM section builds on top of it for narrative.
"""


def _short_refs(commits) -> str:
    refs = [c.sha for c in commits]
    if not refs:
        return "none"
    if len(refs) <= 5:
        return ", ".join(refs)
    return ", ".join(refs[:5]) + " (+" + str(len(refs) - 5) + " more)"


def build_deterministic_summary(activity) -> str:
    """Return a markdown 'executive snapshot' derived only from git data."""
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

    lines = [
        "## Executive Snapshot (from git data)",
        "",
        "- **Repos**: " + ", ".join(ra.repo for ra in activity),
        "- **Commits**: " + str(total_commits) + "  (refs: " + _short_refs([c for ra in activity for c in ra.commits]) + ")",
        "- **Pull requests**: " + str(total_prs) + " total, " + str(merged_prs) + " merged",
        "- **Lines changed**: +" + str(total_adds) + " / \u2212" + str(total_dels),
        "- **Files touched** (" + str(len(all_files)) + "): " + files_txt,
    ]

    for ra in activity:
        if not ra.pull_requests:
            continue
        lines.append("")
        lines.append("### PR detail for " + ra.repo)
        for p in ra.pull_requests:
            merged = "merged " if p.merged else p.state + " "
            files_desc = ", ".join(f.filename for f in p.files[:5])
            if len(p.files) > 5:
                files_desc += " (+" + str(len(p.files) - 5) + " more)"
            stat = (
                "  [" + str(p.changed_files) + " file(s), +" + str(p.additions) + "/-" + str(p.deletions) + "]"
                if p.changed_files else ""
            )
            lines.append("- PR #" + str(p.number) + " (" + merged.strip() + "): " + p.title + stat)
            if files_desc:
                lines.append("    files: " + files_desc)

    lines.append("")
    lines.append("_Produced from git data; no AI involved._")
    return "\n".join(lines)