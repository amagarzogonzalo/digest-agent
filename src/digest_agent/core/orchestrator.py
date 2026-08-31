"""Orchestrator - LangGraph StateGraph wiring the digest pipeline.

The graph composes several bounded, reviewable loops:
  * model selection  (select_and_test_model -> itself)   Evaluator-Optimizer
  * summary review    (committee reviewers -> refine -> reviewers)  complexity
This follows Anthropic's agent patterns: Prompt Chaining for the outer
pipeline, Parallelization (reviewer committee) for critique, and
Evaluator-Optimizer for both model selection and summary refinement.
"""

from functools import partial

from langgraph.graph import StateGraph, START, END

from digest_agent.core.state import DigestState
from digest_agent.core.policy import ModelSelectionPolicy, SelectionOutcome
from digest_agent.core.text_utils import strip_reasoning_prefix, parse_review_verdict
from digest_agent.core.deterministic import build_deterministic_summary
from digest_agent.ports import (
    ActivitySourcePort,
    LLMProviderPort,
    DocumentRendererPort,
    NotifierPort,
)


REVIEWER_FOCUS = {
    0: ("detail_code", "detail & code substance"),
    1: ("accuracy", "accuracy & groundedness"),
    2: ("structure_tone", "structure, coverage & tone"),
}


class Orchestrator:
    """Builds and runs the digest pipeline graph."""

    def __init__(
        self,
        activity_source: ActivitySourcePort,
        llm_provider: LLMProviderPort,
        document_renderer: DocumentRendererPort,
        notifier: NotifierPort,
        max_attempts: int = 6,
        max_reviews: int = 4,
    ):
        self.activity_source = activity_source
        self.llm_provider = llm_provider
        self.document_renderer = document_renderer
        self.notifier = notifier
        self.max_attempts = max_attempts
        self.max_reviews = max_reviews
        self.policy = ModelSelectionPolicy()
        self._graph = None

    # ---------------------------------------------------------------- graph

    def build_graph(self) -> StateGraph:
        if self._graph is not None:
            return self._graph

        graph = StateGraph(DigestState)

        graph.add_node("fetch_github", self._fetch_github_node)
        graph.add_node("select_and_test_model", self._select_and_test_model_node)
        graph.add_node("summarize", self._summarize_node)

        # Reviewer committee: 3 parallel critics fan into one aggregator.
        graph.add_node("review_0", partial(self._review_node, focus_key=0))
        graph.add_node("review_1", partial(self._review_node, focus_key=1))
        graph.add_node("review_2", partial(self._review_node, focus_key=2))
        graph.add_node("aggregate_review", self._aggregate_review_node)
        graph.add_node("refine", self._refine_node)

        graph.add_node("finalize", self._finalize_node)
        graph.add_node("render_pdf", self._render_pdf_node)
        graph.add_node("send_notification", self._send_notification_node)
        graph.add_node("handle_failure", self._handle_failure_node)

        graph.add_edge(START, "fetch_github")
        graph.add_edge("fetch_github", "select_and_test_model")

        graph.add_conditional_edges(
            "select_and_test_model",
            self._route_after_selection,
            {
                "pass": "summarize",
                "continue": "select_and_test_model",
                "fail_exhausted": "handle_failure",
                "fail_budget": "handle_failure",
            },
        )

        # Fan-out to committee, fan-in to aggregator.
        graph.add_edge("summarize", "review_0")
        graph.add_edge("summarize", "review_1")
        graph.add_edge("summarize", "review_2")
        graph.add_edge("review_0", "aggregate_review")
        graph.add_edge("review_1", "aggregate_review")
        graph.add_edge("review_2", "aggregate_review")

        graph.add_conditional_edges(
            "aggregate_review",
            self._route_after_review,
            {
                "accept": "finalize",
                "refine": "refine",
                "give_up": "finalize",
            },
        )

        # Refine loops back to the committee.
        graph.add_edge("refine", "review_0")
        graph.add_edge("refine", "review_1")
        graph.add_edge("refine", "review_2")

        graph.add_edge("finalize", "render_pdf")
        graph.add_edge("render_pdf", "send_notification")
        graph.add_edge("send_notification", END)
        graph.add_edge("handle_failure", END)

        self._graph = graph.compile()
        return self._graph

    def invoke(self, initial_state: DigestState | None = None) -> DigestState:
        graph = self.build_graph()
        state = initial_state or DigestState(
            repos=[],
            lookback_days=1,
            max_attempts=self.max_attempts,
            max_reviews=self.max_reviews,
        )
        state.max_attempts = self.max_attempts
        state.max_reviews = self.max_reviews
        raw_result = graph.invoke(state)
        return DigestState(**{k: v for k, v in raw_result.items()
                              if k in DigestState.__dataclass_fields__})

    # ------------------------------------------------------- node helpers

    def _ensure_state(self, state) -> DigestState:
        if isinstance(state, dict):
            return DigestState(**{k: v for k, v in state.items()
                                  if k in DigestState.__dataclass_fields__})
        return state

    def _fetch_github_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        try:
            if not state.since:
                state.compute_since()
            activity = self.activity_source.fetch_activity(state.repos, state.since)
            deterministic = build_deterministic_summary(activity)
            return {"activity": activity, "deterministic_summary": deterministic}
        except Exception as e:
            return {"error_message": f"Fetch failed: {e}"}

    def _select_and_test_model_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        if not state.candidate_pool:
            state.candidate_pool = self.llm_provider.list_free_models()
        if not state.candidate_pool:
            return {"outcome": "fail_exhausted"}

        untried = [m for m in state.candidate_pool if m not in state.tried_models]
        if not untried:
            return {"outcome": "fail_exhausted"}

        candidate = untried[0]
        result = self.llm_provider.test(candidate)
        outcome = self.policy.on_test_result(state, result)

        updates = {
            "candidate_pool": state.candidate_pool,
            "tried_models": state.tried_models,
            "attempt_count": state.attempt_count,
            "last_test_result": result,
        }
        if outcome.action == "pass":
            updates["selected_model"] = outcome.selected_model
            updates["outcome"] = "pass"
        elif outcome.action in ("fail_exhausted", "fail_budget"):
            updates["outcome"] = outcome.action
        return updates

    def _route_after_selection(self, state: DigestState) -> str:
        return state.outcome or "continue"

    def _summarize_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}
        if not state.has_activity:
            return {"summary": "No activity across your repos in this period. Nothing to report today."}

        prompt = self._build_summarization_prompt(state.activity)
        try:
            summary = strip_reasoning_prefix(self.llm_provider.complete(state.selected_model, prompt))
            return {"summary": summary, "review_count": 0}
        except Exception as e:
            return {"outcome": "error", "error_message": f"Summarization failed: {e}"}

    # ------------------------------------------------- committee review

    def _review_node(self, state: DigestState, focus_key: int) -> dict:
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}

        focus = REVIEWER_FOCUS[focus_key][0]
        label = REVIEWER_FOCUS[focus_key][1]
        prompt = self._build_review_prompt(state.summary, state.activity, focus, label)
        try:
            response = self.llm_provider.complete(state.selected_model, prompt)
            accepted, critique, score = parse_review_verdict(response)
            ret = {f"reviewer_critique_{focus_key}": critique, f"reviewer_score_{focus_key}": score}
            if not accepted:
                ret[f"reviewer_critique_{focus_key}"] = (critique + f" [focused on: {label}]").strip()
            return ret
        except Exception as e:
            # A single reviewer failing must not block the committee.
            return {f"reviewer_score_{focus_key}": 0.0, f"reviewer_critique_{focus_key}": "[reviewer failed]"}

    def _aggregate_review_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        scores = [state.reviewer_score_0, state.reviewer_score_1, state.reviewer_score_2]
        critiques = [c for c in (state.reviewer_critique_0, state.reviewer_critique_1, state.reviewer_critique_2)
                     if c and "[reviewer failed]" not in c]
        avg = sum(scores) / max(1, sum(1 for s in scores if s))
        passes = sum(1 for s in scores if s >= state.min_quality_score)
        merged = "\n".join("- " + c if not c.strip().startswith("-") else c for c in critiques) if critiques else ""

        next_count = state.review_count + 1
        got_real_review = sum(1 for s in scores if s > 0)
        decision = "refine"
        if state.error_message:
            decision = "give_up"
        elif got_real_review == 0:
            # No reviewer could run (rate-limited). Keep current draft.
            decision = "accept"
        elif next_count >= state.max_reviews:
            # Budget exhausted: accept the best draft we have.
            decision = "accept"
        else:
            # Consensus: accept only when BOTH average is high AND a majority
            # of reviewers individually pass. Otherwise keep refining.
            if avg >= state.min_quality_score and passes >= 2:
                decision = "accept"
            else:
                decision = "refine"
        return {
            "review_count": next_count,
            "review_score": round(avg, 2),
            "last_critique": merged,
            "review_decision": decision,
        }

    def _refine_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}
        prompt = self._build_refine_prompt(state.summary, state.last_critique, state.activity)
        try:
            improved = strip_reasoning_prefix(self.llm_provider.complete(state.selected_model, prompt))
            # Only keep the improvement if it is non-empty; otherwise retain current.
            if improved and improved.strip():
                return {"summary": improved}
            return {"review_decision": "accept"}
        except Exception as e:
            # Best-effort refinement: never discard an already-good summary.
            # Accept the current draft and do not set a blocking error.
            return {"review_decision": "accept"}

    def _route_after_review(self, state: DigestState) -> str:
        return getattr(state, "review_decision", None) or "accept"

    def _finalize_node(self, state: DigestState) -> dict:
        """Prepend the deterministic snapshot above the LLM narrative."""
        state = self._ensure_state(state)
        if state.deterministic_summary and state.summary and not state.summary.startswith(state.deterministic_summary[:30]):
            final = state.deterministic_summary.strip() + "\n\n# Daily Digest (AI summary)\n\n" + state.summary.strip()
            return {"summary": final}
        return {}

    # ---------------------------------------------------- render + notify

    def _render_pdf_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        try:
            return {"pdf_path": self.document_renderer.render(state.summary)}
        except Exception as e:
            return {"outcome": "error", "error_message": f"PDF render failed: {e}"}

    def _send_notification_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        if state.error_message:
            return {"outcome": "error", "error_message": f"Prior error: {state.error_message}", "email_sent": False}
        try:
            recipient = getattr(state, "recipient", "") or "user@example.com"
            subject = f"GitHub Daily Digest — {(state.since[:10] if state.since else 'today')}"
            self.notifier.send(recipient, subject, state.pdf_path)
            return {"email_sent": True, "outcome": "success"}
        except Exception as e:
            return {"outcome": "error", "error_message": f"Notification failed: {e}", "email_sent": False}

    def _handle_failure_node(self, state: DigestState) -> dict:
        state = self._ensure_state(state)
        reason = state.error_message or state.outcome
        return {
            "outcome": state.outcome,
            "error_message": f"Model selection failed: {reason}",
            "summary": f"Failed to generate digest: {reason}",
            "email_sent": False,
        }

    # ------------------------------------------------------- prompt builders

    def _group_activity(self, activity: list) -> str:
        """Render activity (with diffs where present) into a prompt block."""
        lines = []
        for repo_activity in activity:
            lines.append(f"### repo {repo_activity.repo}")
            for pr in repo_activity.pull_requests:
                status = "merged/closed" if pr.merged else pr.state
                stat = f" [+{pr.additions}/-{pr.deletions}, {pr.changed_files} files]" if pr.changed_files else ""
                lines.append(f"- PR #{pr.number} ({status}): {pr.title}{stat}")
                for f in pr.files:
                    lines.append(f"    FILE {f.filename} ({f.additions}+/-{f.deletions})")
                    if f.patch:
                        for hunk in f.patch.splitlines()[:24]:
                            lines.append("      | " + hunk)
            for commit in repo_activity.commits:
                lines.append(f"- commit {commit.sha} by {commit.author}: {commit.message}")
                for f in commit.files[:6]:
                    lines.append(f"    FILE {f.filename} ({f.additions}+/-{f.deletions})")
        return "\n".join(lines)

    def _build_summarization_prompt(self, activity: list) -> str:
        raw = self._group_activity(activity)
        return (
            "You are a meticulous senior engineer writing a DETAILED daily engineering "
            "digest from raw git activity INCLUDING file diffs. The reader is a competent "
            "engineer who needs full context - NOT a fluffy executive summary.\n\n"
            "MANDATORY:\n"
            "- Start with '# <repo name>' title only.\n"
            "- Organize into '## ' sections by area (Platform, Evaluation, Training, "
            "Frontend, Docs, CI/CD, Runtime...).\n"
            "- GROUPED, substantive bullets across all areas. Each bullet: intent, affected "
            "files/paths, key APIs/functions, PR/commit refs, the diffscope (e.g. additions/"
            "deletions), tradeoff/risk, and ONE honest limitation or follow-up.\n"
            "- Use the ACTUAL code seen in the diff hunks (## +## -## lines) to include real "
            "short ```code``` snippets when a change is significant.\n"
            "- Reference concrete identifiers verbatim: PR #N, sha refs, file paths, function "
            "names, config keys, CLI flags, diff stats.\n"
            "- Prefer depth: 4-8 bullets per area.\n"
            "- Do NOT invent facts; if a detail is absent from the diff, skip it and do not "
            "invent placeholder code.\n"
            "- End with a 2-3 sentence overall summary.\n"
            "- Output only markdown; no preamble.\n\n"
            "ACTIVITY WITH DIFFS:\n" + raw
        )

    def _build_review_prompt(self, summary: str, activity: list, focus: str, label: str) -> str:
        return (
            "You are a demanding reviewer on a digest-review committee. Your specialty: {label}.\n\n"
            "Grade the digest below on YOUR dimension only, then give a verdict.\n\n"
            "RUBRIC (0-10), focus: {label}:\n".format(label=label) +
            _focus_criteria(focus) +
            "\nDIGEST:\n" + summary + "\n\n"
            "Respond EXACTLY:\n"
            "SCORE: <0-10 overall for " + label + ">\n"
            "VERDICT: PASS|REVISE\n"
            "CRITIQUE:\n- <specific, actionable bullets focused on " + label + ">\n\n"
            "PASS only if your dimension is well met (>=8). Otherwise REVISE with concrete "
            "guidance naming what to add (files, PRs, code, structure)."
        )

    def _build_refine_prompt(self, summary: str, critique: str, activity: list) -> str:
        return (
            "You are a senior engineering editor. REWRITE the digest to fully address ALL "
            "reviewer critiques and make it DETAILED, accurate, and technically rich.\n\n"
            "REVIEWER CRITIQUES:\n" + critique + "\n\n"
            "HARD REQUIREMENTS:\n"
            "- Keep one '# ' title and '## ' sections.\n"
            "- ADD the identifiers named in critiques: files, functions, PR numbers, config keys.\n"
            "- EXPAND each area: intent, files, APIs, diffscope, tradeoffs, follow-up.\n"
            "- Use real ```code``` from the diff context; label anything inferred "
            "'// illustrative — verify against source'.\n"
            "- Ground every claim in the RAW ACTIVITY below; do not invent.\n"
            "- Output only the rewritten markdown digest.\n\n"
            "CURRENT DIGEST:\n" + summary + "\n\n"
            "RAW ACTIVITY (ground truth):\n" + self._group_activity(activity)
        )


def _focus_criteria(focus: str) -> str:
    if focus == "accuracy":
        return (
            "1. Every fact is grounded in the diff/activity (no invented code or claims).\n"
            "2. PR/commit refs and file paths are correct.\n"
            "3. Placeholder/illustrative code is clearly labeled.\n"
        )
    if focus == "structure_tone":
        return (
            "1. One '# ' title; '## ' sections; markdown bullets; no preamble.\n"
            "2. All areas covered (platform, eval, training, frontend, docs, CI).\n"
            "3. Tone professional, concrete, no hype/filler.\n"
        )
    return (
        "1. Names files, functions, PRs, config keys, CLI flags (HIGH impact).\n"
        "2. Includes real ```code``` snippets for key changes.\n"
        "3. Gives diffscope (+/- lines) and per-area depth.\n"
    )


def build_orchestrator(
    activity_source: ActivitySourcePort,
    llm_provider: LLMProviderPort,
    document_renderer: DocumentRendererPort,
    notifier: NotifierPort,
    max_attempts: int = 6,
    max_reviews: int = 4,
) -> Orchestrator:
    """Factory function."""
    return Orchestrator(
        activity_source=activity_source,
        llm_provider=llm_provider,
        document_renderer=document_renderer,
        notifier=notifier,
        max_attempts=max_attempts,
        max_reviews=max_reviews,
    )