"""Text helpers for cleaning LLM output and parsing review verdicts."""

import re


def strip_reasoning_prefix(text: str) -> str:
    """Remove a leading reasoning/thinking block some models emit.

    Returns the visible answer only. If no reasoning marker is found the
    text is returned unchanged.
    """
    lowered = text.lower()
    markers = ("<thinking", "<|start|>", "thinking\n", "here's a thinking")
    idx = -1
    for m in markers:
        j = lowered.find(m)
        if j != -1 and (idx == -1 or j < idx):
            idx = j
    if idx == -1:
        return text
    rest = text[idx:]
    # A closing tag clearly ends the block.
    for close in ("</thinking>", "<|end|>"):
        c = rest.lower().find(close)
        if c != -1:
            return rest[c + len(close):].strip()
    # No closing tag: skip reasoning-style lines until real content appears.
    lines = rest.splitlines()
    keep = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not _is_reasoning_line(stripped):
            keep = i
            break
    return "\n".join(lines[keep:]).strip()


def _is_reasoning_line(line: str) -> bool:
    """Heuristic: is this line part of a model's reasoning preamble?"""
    lowered = line.lower()
    if lowered.startswith(("thinking", "<", "here's", "to ", "i will", "first", "next", "the user", "we need")):
        return True
    if lowered.startswith(("* ", "- ", "# ", "**", "`")) and len(line) < 80:
        return False
    if lowered in ("analyze", "analyse", "understand", "identify", "decide", "final", "conclusion"):
        return True
    return bool(line.startswith("   ") or line.startswith("\t") or line.startswith("    "))


def parse_review_verdict(response: str) -> tuple[bool, str]:
    """Parse a reviewer response into (accepted, critique).

    The reviewer is instructed to begin with ``VERDICT: PASS`` or
    ``VERDICT: REVISE`` on its own line, optionally followed by a critique.
    """
    text = strip_reasoning_prefix(response)
    verdict_match = re.search(r"VERDICT\s*:\s*(PASS|REVISE)", text, re.IGNORECASE)
    if not verdict_match:
        # Fallback: anything overwhelmingly positive counts as a pass.
        low = text.lower()
        accepted = any(k in low for k in ("looks good", "meets criteria", "no issues", "well written", "acceptable"))
        return accepted, text.strip()
    accepted = verdict_match.group(1).upper() == "PASS"
    critique = text[verdict_match.end():].strip().lstrip(":-").strip()
    return accepted, critique