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
            accepted, critique = parse_review_verdict(response)
            updates: dict = {"review_count": state.review_count + 1, "last_critique": critique}
            if accepted or state.review_count >= state.max_reviews:
                updates["review_decision"] = "accept" if accepted else "give_up"
            else:
                updates["review_decision"] = "refine"
            return updates
        except Exception as e:
            # If review tooling fails, do not block the pipeline - accept what we have.
            return {"review_count": state.review_count + 1, "review_decision": "accept", "last_critique": ""}

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
        """Build the summarization prompt from activity data."""
        lines = []
        for repo_activity in activity:
            lines.append(f"### {repo_activity.repo}")
            for commit in repo_activity.commits:
                lines.append(f"- commit {commit.sha} by {commit.author}: {commit.message}")
            for pr in repo_activity.pull_requests:
                status = "merged" if pr.merged else pr.state
                lines.append(f"- PR #{pr.number} ({status}): {pr.title}")
        raw_activity = "\n".join(lines)

        return (
            "You are a senior technical editor producing a concise daily digest "
            "of engineering activity. Below is raw git activity (commits and pull "
            "requests) across one or more repos for the lookback window.\n\n"
            "Rules:\n"
            "- Use '#' for the single main title (the repo name).\n"
            "- Use '##' for each repo section, then '-' bullet points.\n"
            "- For each repo with activity, give a one-line headline of what "
            "happened, then 2-4 specific bullet points in plain, concrete English "
            "(name the repo, PR number, or commit topic; do NOT just restate raw "
            "commit messages verbatim).\n"
            "- Skip repos with no activity.\n"
            "- Be specific, factual, and jargon-lean. No hype, no fluff, no "
            "'Overall a busy day' filler.\n"
            "- End with a single one-sentence overall summary.\n"
            "- No preamble or explanation; output the markdown digest only.\n\n"
            f"{raw_activity}"
        )

    def _build_review_prompt(self, summary: str, activity: list, review_count: int, max_reviews: int) -> str:
        """Build a prompt that critiques the digest against a rubric."""
        return (
            "You are a meticulous chief reviewer. Assess the following daily digest "
            "against these criteria and give a verdict.\n\n"
            "Criteria:\n"
            "1. Accuracy: everything is grounded in the activity; no invented facts.\n"
            "2. Specificity: names repos, PR numbers, or concrete changes (not "
            "vague filler).\n"
            "3. Clarity & structure: single '#' title, '##' per repo, '-' bullets,\n"
            "4. Tone: professional, factual, no hype, no 'busy day' filler.\n"
            "5. Formatting: short, skimmable, no preamble, no duplicated title.\n\n"
            f"Digest:\n{summary}\n\n"
            "If it already meets these criteria well, respond with exactly:\n"
            "VERDICT: PASS\n\n"
            "Otherwise respond with:\n"
            "VERDICT: REVISE\n"
            "Followed by a short, actionable critique (2-4 bullets) listing the "
            "specific things to fix.\n\n"
            f"(review {review_count+1}/{max_reviews})"
        )

    def _build_refine_prompt(self, summary: str, critique: str, activity: list) -> str:
        """Build a prompt that rewrites the digest to address critique."""
        return (
            "You are a senior technical editor. Rewrite the following daily digest "
            "to address the reviewer's feedback.\n\n"
            f"Reviewer feedback:\n{critique}\n\n"
            "Rules (keep these strictly):\n"
            "- '#' for the main title (repo name).\n"
            "- '##' per repo section, '-' bullet points.\n"
            "- Specific, factual, plain English; no hype or filler.\n"
            "- One one-sentence overall summary at the end.\n"
            "- Output only the rewritten digest, no preamble.\n\n"
            f"Current digest to improve:\n{summary}"
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