"""Core state schema for the digest agent."""

from dataclasses import dataclass, field
from datetime import datetime

from digest_agent.ports import RepoActivity, ModelId, TestResult


@dataclass
class DigestState:
    """Shared state passed through the orchestrator graph."""
    # Input configuration
    repos: list[str] = field(default_factory=list)
    lookback_days: int = 1
    since: str = ""  # ISO timestamp, computed at runtime
    recipient: str = ""  # where to send the notification

    # Fetched activity
    activity: list[RepoActivity] = field(default_factory=list)

    # Model selection loop state
    candidate_pool: list[ModelId] = field(default_factory=list)
    tried_models: set[ModelId] = field(default_factory=set)
    selected_model: ModelId | None = None
    attempt_count: int = 0
    max_attempts: int = 5
    last_test_result: TestResult | None = None

    # Output
    summary: str = ""
    pdf_path: str = ""
    email_sent: bool = False

    # Report identity (UID, version, timestamp) shown in PDF + email subject
    report_uid: str = ""
    report_version: str = "1.0.0"
    report_generated_at: str = ""

    # Deterministic (git-only) snapshot shown above the LLM summary
    deterministic_summary: str = ""

    # Per-PR deep-dive worker outputs (orchestrator-workers on the latest PRs)
    pr_dive_0: str = ""
    pr_dive_1: str = ""
    pr_dive_2: str = ""
    overall_narrative: str = ""

    # Final polished summary produced by the digest "chair" agent (last summary)
    final_summary: str = ""

    # Working model chosen per-PR worker via _select_agent_model (informational)
    agent_model_0: str = ""
    agent_model_1: str = ""
    agent_model_2: str = ""

    # Review committee state (3 parallel reviewers use distinct keys to avoid reducers)
    reviewer_critique_0: str = ""
    reviewer_critique_1: str = ""
    reviewer_critique_2: str = ""
    reviewer_score_0: float = 0.0
    reviewer_score_1: float = 0.0
    reviewer_score_2: float = 0.0

    # Review / refine loop state
    review_count: int = 0
    max_reviews: int = 6
    last_critique: str = ""
    review_decision: str = ""  # "accept" | "refine" | "give_up"
    review_score: float = 0.0  # 0-10 rubric score (committee average)
    min_quality_score: float = 8.0  # below this, refine (until budget)

    # Terminal state
    outcome: str = ""  # "success", "fail_exhausted", "fail_budget", "error"
    error_message: str = ""

    def compute_since(self) -> str:
        """Compute the ISO timestamp for the lookback window."""
        from datetime import timedelta, timezone
        since_dt = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        self.since = since_dt.isoformat()
        return self.since

    @property
    def has_activity(self) -> bool:
        return any(a.commits or a.pull_requests for a in self.activity)

    @property
    def untried_candidates(self) -> list[ModelId]:
        return [m for m in self.candidate_pool if m not in self.tried_models]

    @property
    def loop_exhausted(self) -> bool:
        return len(self.untried_candidates) == 0

    @property
    def loop_budget_exceeded(self) -> bool:
        return self.attempt_count >= self.max_attempts