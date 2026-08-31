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

    # Review / refine loop state
    review_count: int = 0
    max_reviews: int = 2
    last_critique: str = ""
    review_decision: str = ""  # "accept" | "refine" | "give_up"

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