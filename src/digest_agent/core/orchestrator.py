"""Orchestrator - LangGraph StateGraph wiring the pipeline nodes.

The graph implements:
- Prompt Chaining for the outer pipeline
- Evaluator-Optimizer loop for model selection (the one cycle in the graph)
"""

from langgraph.graph import StateGraph, START, END

from digest_agent.core.state import DigestState
from digest_agent.core.policy import ModelSelectionPolicy, SelectionOutcome
from digest_agent.core.text_utils import strip_reasoning_prefix, parse_review_verdict
from digest_agent.ports import (
    ActivitySourcePort,
    LLMProviderPort,
    DocumentRendererPort,
    NotifierPort,
)


class Orchestrator:
    """Builds and runs the digest pipeline graph.

    The graph has two bounded cycles:
    - model selection (select_and_test_model -> itself)
    - summary review (review <-> refine, capped by max_reviews)
    All other paths are acyclic.
    """

    def __init__(
        self,
        activity_source: ActivitySourcePort,
        llm_provider: LLMProviderPort,
        document_renderer: DocumentRendererPort,
        notifier: NotifierPort,
        max_attempts: int = 5,
        max_reviews: int = 2,
    ):
        self.activity_source = activity_source
        self.llm_provider = llm_provider
        self.document_renderer = document_renderer
        self.notifier = notifier
        self.max_attempts = max_attempts
        self.max_reviews = max_reviews
        self.policy = ModelSelectionPolicy()
        self._graph = None

    def build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph."""
        if self._graph is not None:
            return self._graph

        graph = StateGraph(DigestState)

        # Add nodes
        graph.add_node("fetch_github", self._fetch_github_node)
        graph.add_node("select_and_test_model", self._select_and_test_model_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("review", self._review_node)
        graph.add_node("refine", self._refine_node)
        graph.add_node("render_pdf", self._render_pdf_node)
        graph.add_node("send_notification", self._send_notification_node)
        graph.add_node("handle_failure", self._handle_failure_node)

        # Define edges - linear pipeline with two bounded cycles
        graph.add_edge(START, "fetch_github")
        graph.add_edge("fetch_github", "select_and_test_model")

        # The evaluator-optimizer loop: conditional edge from select_and_test_model
        graph.add_conditional_edges(
            "select_and_test_model",
            self._route_after_selection,
            {
                "pass": "summarize",
                "continue": "select_and_test_model",  # THE MODEL-SELECTION CYCLE
                "fail_exhausted": "handle_failure",
                "fail_budget": "handle_failure",
            },
        )

        # Summary quality loop: generate -> review -> (refine | render)
        graph.add_edge("summarize", "review")
        graph.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "accept": "render_pdf",
                "refine": "refine",
                "give_up": "render_pdf",
            },
        )
        graph.add_edge("refine", "review")

        graph.add_edge("render_pdf", "send_notification")
        graph.add_edge("send_notification", END)
        graph.add_edge("handle_failure", END)

        self._graph = graph.compile()
        return self._graph

    def invoke(self, initial_state: DigestState | None = None) -> DigestState:
        """Run the graph with optional initial state."""
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
        # Convert result dict back to DigestState for convenient attribute access
        return DigestState(**{k: v for k, v in raw_result.items()
                              if k in DigestState.__dataclass_fields__})

    # ---- Node implementations ----

    def _ensure_state(self, state) -> DigestState:
        """Normalize state to DigestState if LangGraph passed a dict."""
        if isinstance(state, dict):
            # Reconstruct DigestState from dict keys that exist
            return DigestState(**{k: v for k, v in state.items()
                                  if k in DigestState.__dataclass_fields__})
        return state

    def _fetch_github_node(self, state: DigestState) -> dict:
        """Fetch activity from the activity source."""
        state = self._ensure_state(state)
        try:
            if not state.since:
                state.compute_since()
            activity = self.activity_source.fetch_activity(state.repos, state.since)
            return {"activity": activity}
        except Exception as e:
            return {"error_message": f"Fetch failed: {e}"}

    def _select_and_test_model_node(self, state: DigestState) -> dict:
        """Evaluator-Optimizer loop node: test the next candidate model."""
        state = self._ensure_state(state)
        if not state.candidate_pool:
            candidates = self.llm_provider.list_free_models()
            state.candidate_pool = candidates

        # Check initial condition
        if not state.candidate_pool:
            return {"outcome": "fail_exhausted"}

        # Get next untried candidate
        untried = [m for m in state.candidate_pool if m not in state.tried_models]
        if not untried:
            return {"outcome": "fail_exhausted"}

        candidate = untried[0]
        result = self.llm_provider.test(candidate)

        # Apply policy decision
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
        # For "continue", no outcome set - will loop back

        return updates

    def _route_after_selection(self, state: DigestState) -> str:
        """Route based on selection outcome."""
        return state.outcome or "continue"

    def _summarize_node(self, state: DigestState) -> dict:
        """Summarize activity using the selected model."""
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}

        if not state.has_activity:
            return {"summary": "No activity across your repos in this period. Nothing to report today."}

        # Build prompt from activity
        prompt = self._build_summarization_prompt(state.activity)

        try:
            summary = self.llm_provider.complete(state.selected_model, prompt)
            summary = strip_reasoning_prefix(summary)
            return {"summary": summary, "review_count": 0}
        except Exception as e:
            return {"outcome": "error", "error_message": f"Summarization failed: {e}"}

    def _review_node(self, state: DigestState) -> dict:
        """Review the summary against a rubric; decide accept/refine."""
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}

        prompt = self._build_review_prompt(state.summary, state.activity, state.review_count, state.max_reviews)
        try:
            response = self.llm_provider.complete(state.selected_model, prompt)
            accepted, critique, score = parse_review_verdict(response)
            next_count = state.review_count + 1
            updates: dict = {
                "review_count": next_count,
                "last_critique": critique,
                "review_score": score or 0.0,
            }
            # Accept only if the reviewer PASSES AND the draft meets the quality bar,
            # OR we have exhausted the review budget (degrade gracefully).
            if (accepted and score >= state.min_quality_score) or next_count >= state.max_reviews:
                updates["review_decision"] = "accept"
            elif accepted:
                # passed but under the quality bar -> push one more refine if budget remains
                updates["review_decision"] = "refine"
            else:
                updates["review_decision"] = "refine"
            return updates
        except Exception as e:
            # If review tooling fails, do not block the pipeline - accept what we have.
            return {
                "review_count": state.review_count + 1,
                "review_decision": "accept",
                "last_critique": "",
                "review_score": 0.0,
            }

    def _refine_node(self, state: DigestState) -> dict:
        """Rewrite the summary addressing the reviewer's critique."""
        state = self._ensure_state(state)
        if not state.selected_model:
            return {"outcome": "error", "error_message": "No model selected"}

        prompt = self._build_refine_prompt(state.summary, state.last_critique, state.activity)
        try:
            improved = self.llm_provider.complete(state.selected_model, prompt)
            improved = strip_reasoning_prefix(improved)
            return {"summary": improved}
        except Exception as e:
            return {"review_decision": "accept", "error_message": f"Refinement failed: {e}"}

    def _route_after_review(self, state: DigestState) -> str:
        """Route after review: accept or refine (with bounded revisions)."""
        return getattr(state, "review_decision", None) or "accept"

    def _render_pdf_node(self, state: DigestState) -> dict:
        """Render summary to PDF."""
        state = self._ensure_state(state)
        try:
            pdf_path = self.document_renderer.render(state.summary)
            return {"pdf_path": pdf_path}
        except Exception as e:
            return {"outcome": "error", "error_message": f"PDF render failed: {e}"}

    def _send_notification_node(self, state: DigestState) -> dict:
        """Send notification with PDF attachment."""
        state = self._ensure_state(state)
        # Don't send email if there was a prior error
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
        """Handle terminal failure from model selection."""
        state = self._ensure_state(state)
        reason = state.error_message or state.outcome
        return {
            "outcome": state.outcome,
            "error_message": f"Model selection failed: {reason}",
            "summary": f"Failed to generate digest: {reason}",
            "email_sent": False,
        }

    def _build_summarization_prompt(self, activity: list) -> str:
        """Build a prompt that demands a DEEP, technical, code-bearing digest."""
        grouped = self._group_activity(activity)
        raw_activity = "\n".join(grouped)

        return (
            "You are a meticulous senior engineer writing a DETAILED daily engineering "
            "digest from raw git activity. The reader is a competent engineer who needs "
            "full context — NOT a fluffy executive summary.\n\n"
            "MANDATORY requirements:\n"
            "- Start with '# ' + repo name (one title only).\n"
            "- Organize into '## ' sections by area (e.g. '## Platform', '## Evaluation', "
            "'## Training', '## Frontend', '## Docs', '## CI/CD').\n"
            "- Under each area, write GROUPED, substantive bullets. Each bullet: state the "
            "intent, the affected files/paths, the key APIs/functions, the PR or commit ref, "
            "the tradeoff/risk, and ONE honest limitation or follow-up.\n"
            "- For any significant new capability, include a SHORT real code or config "
            "snippet inside a ``` block showing the core call, signature, or flag. "
            "If you did NOT see the exact code in the activity, mark the snippet as "
            "an illustrative excerpt (label it '// illustrative — verify against source'). "
            "NEVER present inferred code as if it were the verbatim repo source.\n"
            "- Reference concrete identifiers verbatim: PR #N, sha/refs, file paths, "
            "function names, config keys, CLI flags.\n"
            "- Prefer depth and specificity over terseness. 4-8 bullets per area is fine.\n"
            "- Do not invent facts. If a detail is unknown, say 'not specified'.\n"
            "- End with a 2-3 sentence overall summary capturing direction.\n"
            "- Output only the markdown digest; no preamble.\n\n"
            "RAW ACTIVITY:\n"
            f"{raw_activity}"
        )

    def _group_activity(self, activity: list) -> list[str]:
        """Render activity into one string per repo (already grouped by repo)."""
        lines = []
        for repo_activity in activity:
            lines.append(f"### repo {repo_activity.repo}")
            for pr in repo_activity.pull_requests:
                status = "merged/closed" if pr.merged else pr.state
                lines.append(f"- PR #{pr.number} ({status}): {pr.title}")
            for commit in repo_activity.commits:
                lines.append(f"- commit {commit.sha} by {commit.author}: {commit.message}")
        return lines

    def _build_review_prompt(self, summary: str, activity: list, review_count: int, max_reviews: int) -> str:
        """Build a prompt that SCORES the digest on a strict detail rubric."""
        return (
            "You are a demanding engineering editor. Grade the following digest against this "
            "rubric (0-10 each), then give a verdict.\n\n"
            "RUBRIC:\n"
            "1. Detail & specificity (does it name files, functions, PRs, config keys, CLI flags?) "
            "HIGH impact.\n"
            "2. Code substance (does it include real ```code snippets``` for key changes?)."
            "3. Coverage (does it address ALL areas: platform, eval, training, frontend, docs, CI?)."
            "4. Technical correctness & groundedness (no invented facts).\n"
            "5. Structure (one '# ' title, '## ' sections, markdown bullets).\n"
            "6. Tone: professional, concrete, no hype/filler.\n\n"
            f"DIGEST:\n{summary}\n\n"
            "Respond with EXACTLY this format:\n"
            "SCORE: <0-10 overall>\n"
            "VERDICT: PASS|REVISE\n"
            "CRITIQUE:\n- <actionable bullets: what specific detail/code/coverage to add>\n\n"
            "PASS only if score >= 8 AND the digest is genuinely detailed with code and "
            "concrete identifiers. Otherwise REVISE with precise, actionable guidance.\n\n"
            f"(review {review_count+1}/{max_reviews})"
        )

    def _build_refine_prompt(self, summary: str, critique: str, activity: list) -> str:
        """Build a prompt that rewrites the digest to add depth, specificity and code."""
        return (
            "You are a senior engineering editor. REWRITE the digest below to fully address "
            "the reviewer's critique and make it DETAILED and technically rich.\n\n"
            f"REVIEWER CRITIQUE:\n{critique}\n\n"
            "While rewriting, HARD REQUIREMENTS:\n"
            "- Keep one '# ' title, '## ' area sections, markdown bullets.\n"
            "- ADD the specific identifiers named in the critique: file paths, function "
            "names, PR numbers, config keys, CLI flags.\n"
            "- EXPAND each area with more substance — intent, affected files, key APIs, "
            "tradeoffs/risks, and a follow-up.\n"
            "- INCLUDE short real ```code``` blocks for any new capability (core call, "
            "flag, or config excerpt). If the exact code is not in the activity, label it "
            "'// illustrative — verify against source' and do not claim it is verbatim.\n"
            "- Cover ALL areas present in the raw activity.\n"
            "- Be concrete, grounded, and professional; no fluff.\n"
            "- Output only the rewritten markdown digest.\n\n"
            f"CURRENT DIGEST:\n{summary}\n\n"
            "RAW ACTIVITY (use this — it is the source of ground truth):\n"
            f"{self._group_activity(activity)}"
        )


def build_orchestrator(
    activity_source: ActivitySourcePort,
    llm_provider: LLMProviderPort,
    document_renderer: DocumentRendererPort,
    notifier: NotifierPort,
    max_attempts: int = 5,
    max_reviews: int = 2,
) -> Orchestrator:
    """Factory function to create an Orchestrator with the given adapters."""
    return Orchestrator(
        activity_source=activity_source,
        llm_provider=llm_provider,
        document_renderer=document_renderer,
        notifier=notifier,
        max_attempts=max_attempts,
        max_reviews=max_reviews,
    )