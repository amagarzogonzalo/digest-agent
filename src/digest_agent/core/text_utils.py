"""Text helpers for cleaning LLM output and parsing review verdicts."""

import re


def strip_reasoning_prefix(text: str) -> str:
    """Remove a leading reasoning/thinking block some models emit.

    Returns the visible answer only. If no reasoning marker is found the
    text is returned unchanged.
    """
    lowered = text.lower()
    markers = ("<thinking", "<|start|>", "thinking\n", "here's a thinking", "tool used")
    idx = -1
    for m in markers:
        j = lowered.find(m)
        if j != -1 and (idx == -1 or j < idx):
            idx = j
    if idx == -1:
        return text
    rest = text[idx:]
    for close in ("</thinking>", "<|end|>", "<end_of_turn>"):
        c = rest.lower().find(close)
        if c != -1:
            return rest[c + len(close):].strip()
    lines = rest.splitlines()
    keep = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not _is_reasoning_line(stripped):
            keep = i
            break
    return "\n".join(lines[keep:]).strip()


def _is_reasoning_line(line: str) -> bool:
    lowered = line.lower()
    if lowered.startswith(("thinking", "<", "here's", "to answer", "to start", "i'll", "let me", "the user", "we need")):
        return True
    if lowered.startswith(("* ", "- ", "# ", "**", "`")) and len(line) < 80:
        return False
    return bool(line.startswith("   ") or line.startswith("\t") or line.startswith("    "))


def parse_review_verdict(response: str) -> tuple[bool, str, float]:
    """Parse a reviewer response into (accepted, critique, score).

    Expected format: a SCORE: <0-10> line + VERDICT: PASS|REVISE + CRITIQUE.
    Falls back gracefully when the verdict flag is missing.
    """
    text = strip_reasoning_prefix(response)

    # Score: best-effort parse of any "SCORE:" / "Score:" / "score" number.
    score = 0.0
    m = re.search(r"score\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        try:
            score = min(10.0, float(m.group(1)))
        except ValueError:
            score = 0.0

    verdict_match = re.search(r"VERDICT\s*:\s*(PASS|REVISE)", text, re.IGNORECASE)
    if not verdict_match:
        low = text.lower()
        accepted = any(k in low for k in ("looks good", "meets criteria", "no issues", "well written", "acceptable", "pass"))
        return accepted, text.strip(), score

    accepted = verdict_match.group(1).upper() == "PASS"
    crit = text[verdict_match.end():]
    # Drop any trailing CRITIQUE label tail into critique.
    crit_match = re.search(r"CRITIQUE\s*[:]?\s*(.*)", crit, re.IGNORECASE | re.DOTALL)
    if crit_match and crit_match.group(1).strip():
        critique = crit_match.group(1).strip()
    else:
        critique = crit.strip().lstrip(":-").strip()
    return accepted, critique, score