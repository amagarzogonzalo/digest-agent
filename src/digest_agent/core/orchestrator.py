"""Orchestrator - LangGraph StateGraph wiring the digest pipeline.

The graph composes two bounded, reviewable loops:
  * model selection   (select_and_test_model -> itself) Evaluator-Optimizer loop.
  * summary review    (review committee -> refine -> review committee) loop.

Per-agent model selection is re-performed in a loop: each worker sub-agent
(deep-dive, chair, reviewers, refine) asks the provider for a working model
via _select_agent_model rather than blindly reusing one hardcoded model.

Node flow: fetch -> select -> [per-PR deep dives] -> assemble -> chair ->
review committee -> [refine loop] -> finalize -> render -> notify -> END,
plus a handle_failure branch.
"""

from functools import partial
from uuid import uuid4
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END

from digest_agent.core.state import DigestState
from digest_agent.core.policy import ModelSelectionPolicy
from digest_agent.core.text_utils import strip_reasoning_prefix, parse_review_verdict
from digest_agent.core.deterministic import (
    build_deterministic_summary,
    latest_merged_prs,
)
from digest_agent.ports import (
    ActivitySourcePort,
    LLMProviderPort,
    DocumentRendererPort,
    NotifierPort,
    ModelId,
)

REVIEWER_FOCUS = {
    0: ("detail_code", "detail & code substance"),
    1: ("accuracy", "accuracy & groundedness"),
    2: ("structure_tone", "structure, coverage & tone"),
}

PR_DEPTH = 3  # number of latest merged PRs to deep-dive


def _dataclass_fields(cls):
    return set(getattr(cls, "__dataclass_fields__", {}))


def _normalize_markdown(text: str) -> str:
    """Lightly clean LLM markdown so it matches the PDF renderer expectations.

    - strips any leading reasoning prefix
    - ensures no leading/trailing blank lines
    - collapses a leading single '# ' title so the deterministic prefix
      (added in finalize) isn't duplicated
    - keeps '## ' section headings and ``` code fences intact
    """
    text = strip_reasoning_prefix(text).strip()
    lines = text.splitlines()
    out = []
    dropped_title = False
    for ln in lines:
        s = ln.strip()
        if not dropped_title and s.startswith("# ") and not s.startswith("## "):
            dropped_title = True
            continue
        out.append(ln)
    return "\n".join(out).strip()


class Orchestrator:
    """Builds and runs the digest pipeline graph."""

    def __init__(
        self,
        activity_source: ActivitySourcePort,
        llm_provider: LLMProviderPort,
        document_renderer: DocumentRendererPort,
        notifier: NotifierPort,
        max_attempts: int = 6,
        max_reviews: int = 6,
        pr_depth: int = PR_DEPTH,
        min_quality_score: float = 8.0,
    ):
        self.activity_source = activity_source
        self.llm_provider = llm_provider
        self.document_renderer = document_renderer
        self.notifier = notifier
        self.max_attempts = max_attempts
        self.max_reviews = max_reviews
        self.pr_depth = pr_depth
        self.min_quality_score = min_quality_score
        self.policy = ModelSelectionPolicy()
        self._graph = None

    def build_graph(self) -> StateGraph:
        if self._graph is not None:
            return self._graph

        graph = StateGraph(DigestState)

        graph.add_node("fetch_github", self._fetch_github_node)
        graph.add_node("set_metadata", self._set_metadata_node)
        graph.add_node("select_and_test_model", self._select_and_test_model_node)

        # Orchestrator-Workers: one deep-dive sub-agent per latest merged PR.
        for i in range(self.pr_depth):
            graph.add_node(f"pr_dive_{i}", partial(self._pr_worker_node, pr_index=i))
        graph.add_node("assemble", self._assemble_node)
        # Final consolidated summary written by the digest "chair" agent.
        graph.add_node("chair", self._chair_node)

        for k in range(3):
            graph.add_node(f"review_{k}", partial(self._review_node, focus_key=k))
        graph.add_node("aggregate_review", self._aggregate_review_node)
        graph.add_node("refine", self._refine_node)

        graph.add_node("finalize", self._finalize_node)
        graph.add_node("render_pdf", self._render_pdf_node)
        graph.add_node("send_notification", self._send_notification_node)
        graph.add_node("handle_failure", self._handle_failure_node)

        graph.add_edge(START, "fetch_github")

        graph.add_conditional_edges(
            "fetch_github",
            self._route_after_fetch,
            {"ok": "set_metadata", "error": "handle_failure"},
        )
        graph.add_edge("set_metadata", "select_and_test_model")

        graph.add_conditional_edges(
            "select_and_test_model",
            self._route_after_selection,
            {
                "pass": "pr_dive_0",
                "continue": "select_and_test_model",
                "fail_exhausted": "handle_failure",
                "fail_budget": "handle_failure",
            },
        )

        # Workers fan out over the latest PRs, then assemble merges results.
        for i in range(self.pr_depth):
            if i + 1 < self.pr_depth:
                graph.add_edge(f"pr_dive_{i}", f"pr_dive_{i + 1}")
            else:
                graph.add_edge(f"pr_dive_{i}", "assemble")
        graph.add_edge("assemble", "chair")
        graph.add_edge("chair", "review_0")
        graph.add_edge("review_0", "review_1")
        graph.add_edge("review_1", "review_2")
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

        graph.add_edge("refine", "review_0")

        graph.add_edge("finalize", "render_pdf")
        graph.add_edge("render_pdf", "send_notification")
        graph.add_edge("send_notification", END)
        graph.add_edge("handle_failure", END)

        self._graph = graph.compile()
        return self._graph

    def invoke(self, initial_state: DigestState | None = None) -> DigestState:
        graph = self.build_graph()
        state = initial_state or DigestState(repos=[], lookback_days=1)
        state.max_attempts = self.max_attempts
        state.max_reviews = self.max_reviews
        state.min_quality_score = self.min_quality_score
        raw = graph.invoke(self._to_fields(state))
        return self._from_fields(raw)

    def _to_fields(self, state: DigestState) -> dict:
        return {k: getattr(state, k) for k in _dataclass_fields(DigestState)}

    def _from_fields(self, data: dict) -> DigestState:
        fields = _dataclass_fields(DigestState)
        return DigestState(**{k: v for k, v in data.items() if k in fields})

    def _ensure_state(self, state) -> DigestState:
        if isinstance(state, dict):
            return self._from_fields(state)
        return state

    def _route_after_fetch(self, state) -> str:
        state = self._ensure_state(state)
        return "error" if state.error_message else "ok"

    def _set_metadata_node(self, state) -> dict:
        return {
            "report_uid": getattr(state, "report_uid", "") or str(uuid4())[:8].upper(),
            "report_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "report_version": getattr(state, "report_version", "") or "1.0.0",
        }

    def _route_after_selection(self, state) -> str:
        state = self._ensure_state(state)
        return state.outcome or "continue"

    def _route_after_review(self, state) -> str:
        state = self._ensure_state(state)
        return state.review_decision or "accept"

    def _select_agent_model(self, state) -> ModelId | None:
        """Return a WORKING model for the current agent call.

        Reuses state.selected_model if it still passes a quick test; otherwise
        iterates the free-model pool and tests sequentially, stopping at the
        first pass. Returns the last-resort selected_model (may be None).
        """
        if state.selected_model:
            try:
                if self.llm_provider.test(state.selected_model).passed:
                    return state.selected_model
            except Exception:
                pass

        pool = state.candidate_pool or self.llm_provider.list_free_models()
        if not state.candidate_pool:
            state.candidate_pool = list(pool)
        for m in pool:
            if m == state.selected_model:
                continue
            try:
                if self.llm_provider.test(m).passed:
                    state.selected_model = m
                    return m
            except Exception:
                continue
        return state.selected_model

    def _fetch_github_node(self, state) -> dict:
        state = self._ensure_state(state)
        try:
            if not state.since:
                state.compute_since()
            activity = self.activity_source.fetch_activity(state.repos, state.since)
            deterministic = build_deterministic_summary(activity, deep_prs=self.pr_depth)
            return {"activity": activity, "deterministic_summary": deterministic}
        except Exception as e:
            return {"error_message": f"Fetch failed: {e}", "outcome": "error"}

    def _select_and_test_model_node(self, state) -> dict:
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

    def _pr_worker_node(self, state, pr_index: int) -> dict:
        """Orchestrator-worker: deep-dive ONE recent merged PR into detailed prose."""
        state = self._ensure_state(state)
        model = self._select_agent_model(state)
        if not model:
            return {
                f"pr_dive_{pr_index}": "<!-- deep-dive failed -->",
                "outcome": "error",
                "error_message": "No working model available",
            }
        updates = {f"agent_model_{pr_index}": model}
        recent = latest_merged_prs(state.activity, self.pr_depth)
        if pr_index < len(recent):
            pr = recent[pr_index]
            try:
                dive = _normalize_markdown(
                    self.llm_provider.complete(
                        model,
                        self._build_pr_dive_prompt(pr, state.activity),
                    )
                )
                updates[f"pr_dive_{pr_index}"] = dive
                return updates
            except Exception as e:
                updates[f"pr_dive_{pr_index}"] = "<!-- deep-dive failed -->"
                updates["error_message"] = f"Summarization failed: {e}"
                return updates
        updates[f"pr_dive_{pr_index}"] = "<!-- no PR -->"
        return updates

    def _assemble_node(self, state) -> dict:
        """Assemble per-PR deep dives into a detailed digest + a concise overall."""
        state = self._ensure_state(state)
        if not state.has_activity:
            return {
                "summary": "No activity across your repos in this period. Nothing to report today.",
                "review_count": 0,
            }
        dives = [
            getattr(state, f"pr_dive_{i}", "") for i in range(self.pr_depth)
        ]
        dives = [
            d for d in dives
            if d and "<!-- no PR -->" not in d and "deep-dive failed" not in d
        ]
        dive_block = "\n\n".join(dives).strip()

        body = "# Detailed PR Analysis\n\n"
        if dive_block:
            body += dive_block + "\n\n"
        else:
            body += "No mergeable PR detail available for this window.\n\n"

        overall = self._overall_narrative(state)
        if overall:
            body += "## Overall\n\n" + overall + "\n"

        return {"summary": body.strip(), "review_count": 0}

    def _chair_node(self, state) -> dict:
        """Final reviewer agent: synthesize the assembled deep dives into one
        polished, cohesive final summary."""
        state = self._ensure_state(state)
        model = self._select_agent_model(state)
        if not model:
            return {}
        try:
            polished = _normalize_markdown(
                self.llm_provider.complete(
                    model,
                    self._build_chair_prompt(state.summary, state.activity),
                )
            )
            if polished and polished.strip():
                return {"final_summary": polished}
        except Exception:
            pass
        return {}

    def _overall_narrative(self, state) -> str:
        """A short (3-5 sentence) non-bullet narrative tying the PRs together."""
        model = self._select_agent_model(state)
        if not model:
            return ""
        try:
            return _normalize_markdown(
                self.llm_provider.complete(
                    model,
                    self._build_overall_prompt(state.activity),
                )
            )
        except Exception:
            return ""

    def _review_node(self, state, focus_key: int) -> dict:
        state = self._ensure_state(state)
        model = self._select_agent_model(state)
        if not model:
            return {"outcome": "error", "error_message": "No model selected"}
        focus, label = REVIEWER_FOCUS[focus_key]
        content = state.final_summary if state.final_summary else state.summary
        prompt = self._build_review_prompt(content, state.activity, focus, label)
        try:
            accepted, critique, score = parse_review_verdict(
                self.llm_provider.complete(model, prompt))
            ret = {
                f"reviewer_critique_{focus_key}": critique,
                f"reviewer_score_{focus_key}": score,
            }
            if not accepted:
                ret[f"reviewer_critique_{focus_key}"] = (
                    critique + f" [focused on: {label}]").strip()
            return ret
        except Exception:
            return {
                f"reviewer_score_{focus_key}": 0.0,
                f"reviewer_critique_{focus_key}": "[reviewer failed]",
            }

    def _aggregate_review_node(self, state) -> dict:
        state = self._ensure_state(state)
        scores = [state.reviewer_score_0, state.reviewer_score_1, state.reviewer_score_2]
        real = [s for s in scores if s > 0]
        critiques = [
            c for c in (state.reviewer_critique_0, state.reviewer_critique_1,
                        state.reviewer_critique_2)
            if c and "[reviewer failed]" not in c
        ]
        avg = round(sum(real) / len(real), 2) if real else 0.0
        merged = "\n".join("- " + c if not c.strip().startswith("-") else c
                           for c in critiques) if critiques else ""

        next_count = state.review_count + 1
        if state.error_message:
            decision = "give_up"
        elif not real:
            decision = "accept"
        elif next_count >= state.max_reviews:
            decision = "accept"
        else:
            all_pass = all(s >= state.min_quality_score for s in real)
            decision = "accept" if all_pass else "refine"
        return {
            "review_count": next_count,
            "review_score": avg,
            "last_critique": merged,
            "review_decision": decision,
        }

    def _refine_node(self, state) -> dict:
        """The chair re-writes the final summary to address reviewer critique."""
        state = self._ensure_state(state)
        model = self._select_agent_model(state)
        if not model:
            return {"review_decision": "accept"}
        try:
            improved = _normalize_markdown(
                self.llm_provider.complete(
                    model,
                    self._build_refine_prompt(
                        state.final_summary or state.summary,
                        state.last_critique,
                        state.activity,
                    ),
                )
            )
            if improved and improved.strip():
                return {"final_summary": improved}
            return {"review_decision": "accept"}
        except Exception:
            return {"review_decision": "accept"}

    def _finalize_node(self, state) -> dict:
        state = self._ensure_state(state)
        body = state.summary.strip() if state.summary else ""
        polished = state.final_summary.strip() if state.final_summary else ""
        if not body and not polished:
            return {}
        merged = "\n\n".join(p for p in (body, polished) if p)
        if (state.deterministic_summary
                and not merged.startswith(state.deterministic_summary[:30])):
            final = (
                state.deterministic_summary.strip()
                + "\n\n# Daily Digest (AI summary)\n\n"
                + merged
            )
            return {"summary": final}
        return {"summary": merged}

    def _render_pdf_node(self, state) -> dict:
        state = self._ensure_state(state)
        try:
            try:
                pdf_path = self.document_renderer.render(
                    state.summary,
                    extra_meta={
                        "repos": ", ".join(state.repos),
                        "uid": state.report_uid,
                        "version": state.report_version,
                        "generated_at": state.report_generated_at,
                    },
                )
            except TypeError:
                # Fall back if the renderer only accepts a single argument.
                pdf_path = self.document_renderer.render(state.summary)
            return {"pdf_path": pdf_path}
        except Exception as e:
            return {
                "outcome": "error",
                "error_message": f"PDF render failed: {e}",
            }

    def _send_notification_node(self, state) -> dict:
        state = self._ensure_state(state)
        if state.error_message:
            return {"outcome": "error", "email_sent": False}
        try:
            recipient = state.recipient or "user@example.com"
            date_part = state.since[:10] if state.since else "today"
            subject = (f"GitHub Daily Digest {date_part} · "
                       f"{state.report_uid} (v{state.report_version})")
            self.notifier.send(recipient, subject, state.pdf_path)
            return {"email_sent": True, "outcome": "success"}
        except Exception as e:
            return {
                "outcome": "error",
                "error_message": f"Notification failed: {e}",
                "email_sent": False,
            }

    def _handle_failure_node(self, state) -> dict:
        state = self._ensure_state(state)
        reason = state.error_message or state.outcome or "unknown failure"
        return {
            "outcome": state.outcome or "error",
            "error_message": f"Model selection failed: {reason}",
            "summary": f"Failed to generate digest: {reason}",
            "email_sent": False,
        }

    def _group_activity(self, activity: list) -> str:
        lines = []
        for ra in activity:
            lines.append(f"### repo {ra.repo}")
            for pr in ra.pull_requests:
                status = "merged/closed" if pr.merged else pr.state
                stat = ""
                if pr.changed_files:
                    stat = f" [+{pr.additions}/-{pr.deletions}, {pr.changed_files} files]"
                lines.append(f"- PR #{pr.number} ({status}): {pr.title}{stat}")
                for f in pr.files:
                    lines.append(f"    FILE {f.filename} ({f.additions}+/-{f.deletions})")
                    if f.patch:
                        for hunk in f.patch.splitlines()[:24]:
                            lines.append("      | " + hunk)
            for commit in ra.commits:
                lines.append(f"- commit {commit.sha} by {commit.author}: {commit.message}")
                for f in commit.files[:6]:
                    lines.append(f"    FILE {f.filename} ({f.additions}+/-{f.deletions})")
        return "\n".join(lines)

    def _build_pr_dive_prompt(self, pr, activity) -> str:
        file_lines = []
        for f in pr.files[:20]:
            fl = f"- `{f.filename}` ({f.status}, +{f.additions}/-{f.deletions})"
            if f.patch:
                fl += "\n" + "\n".join("    | " + l for l in f.patch.splitlines()[:60])
            file_lines.append(fl)
        files_txt = "\n".join(file_lines) if file_lines else "files: not available"

        return (
            "You are a detail-oriented senior engineer reviewing a SINGLE merged pull "
            "request. Below is its raw git diff detail.\n\n"
            "PR TITLE: " + pr.title + "\nPR NUMBER: " + str(pr.number) + "\n\n"
            "FILES & PATCHES:\n" + files_txt + "\n\n"
            "Write a DETAILED technical section with header '## PR #" + str(pr.number)
            + " — " + pr.title + "'. It must delve deep, not just be a bullet list:\n"
            "- A 1-2 sentence overview of what this PR does and why (the intent).\n"
            "- Analyse each significant file: what changed, the key functions/classes/"
            "identifiers, and the architectural impact, grounded ONLY in the diff.\n"
            "- Quote the most meaningful ACTUAL code lines from the patch hunks as "
            "verbatim ```code``` blocks.\n"
            "- Note risks, tradeoffs and integration touch-points.\n"
            "Be specific, technical and detailed. Do not invent facts not in the diff."
        )

    def _build_overall_prompt(self, activity) -> str:
        return (
            "You are a seasoned engineering lead. Based on the repo activity below, write "
            "ONE concise 'Overall' paragraph (3-5 sentences, not a bullet list) that ties "
            "the recent merged pull requests into a coherent technical narrative, "
            "highlighting the major capabilities added and the next risks/watch-outs.\n\n"
            "ACTIVITY:\n" + self._group_activity(activity)
        )

    def _build_chair_prompt(self, body, activity) -> str:
        return (
            "You are the digest chair. Below are per-PR deep dives assembled from recent "
            "merged pull requests. Review them and write the FINAL polished "
            "'Daily Digest (AI summary)': a cohesive, well-structured markdown summary "
            "that synthesizes these deep dives into one clear overview. Do not repeat the "
            "per-PR sections verbatim; condense into an executive-style summary while "
            "keeping the key technical detail, risks and watch-outs.\n\n"
            "DEEP DIVES:\n" + body + "\n\n"
            "ACTIVITY:\n" + self._group_activity(activity)
        )

    def _build_review_prompt(self, summary, activity, focus, label) -> str:
        return (
            "You are a demanding reviewer on a digest committee. Specialty: " + label + ".\n\n"
            "Rubric (0-10):\n" + _focus_criteria(focus) + "\n"
            "DIGEST:\n" + summary + "\n\n"
            "Respond exactly:\n"
            "SCORE: <0-10>\n"
            "VERDICT: PASS|REVISE\n"
            "CRITIQUE:\n- <actionable>\n\n"
            "PASS only if the dimension is well met (>=8)."
        )

    def _build_refine_prompt(self, summary, critique, activity) -> str:
        return (
            "You are a senior editor. REWRITE the digest to address all reviewer critique "
            "and make it detailed and accurate.\n\n"
            "Critique:\n" + critique + "\n\n"
            "Digest to improve:\n" + summary + "\n\n"
            "RAW ACTIVITY:\n" + self._group_activity(activity)
        )


def _focus_criteria(focus):
    if focus == "accuracy":
        return "1. Grounded in diff. 2. refs correct. 3. inferred code labeled.\n"
    if focus == "structure_tone":
        return "1. one title, sections. 2. all areas. 3. no hype.\n"
    return "1. names files/functions/PRs/keys. 2. real code snippets. 3. diffscope depth.\n"


def build_orchestrator(
    activity_source: ActivitySourcePort,
    llm_provider: LLMProviderPort,
    document_renderer: DocumentRendererPort,
    notifier: NotifierPort,
    max_attempts: int = 6,
    max_reviews: int = 6,
    pr_depth: int = PR_DEPTH,
    min_quality_score: float = 8.0,
) -> Orchestrator:
    return Orchestrator(
        activity_source=activity_source,
        llm_provider=llm_provider,
        document_renderer=document_renderer,
        notifier=notifier,
        max_attempts=max_attempts,
        max_reviews=max_reviews,
        pr_depth=pr_depth,
        min_quality_score=min_quality_score,
    )