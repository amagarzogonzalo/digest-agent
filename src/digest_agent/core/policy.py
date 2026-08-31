"""Model selection policy - pure logic for the evaluator-optimizer loop."""

from dataclasses import dataclass

from digest_agent.ports import ModelId, TestResult
from digest_agent.core.state import DigestState


@dataclass(frozen=True)
class SelectionOutcome:
    """Result of a selection decision."""
    action: str  # "pass", "fail_exhausted", "fail_budget", "continue"
    selected_model: ModelId | None = None
    reason: str = ""


class ModelSelectionPolicy:
    """Pure logic for deciding model selection loop behavior.

    No network calls, no I/O. Given state and a test result, decides
    whether to accept the model, try the next one, or terminate.
    """

    def on_test_result(self, state: DigestState, result: TestResult) -> SelectionOutcome:
        """Process a test result and decide the next action.

        Args:
            state: Current digest state (mutated in place).
            result: Result of testing the current candidate.

        Returns:
            SelectionOutcome with action and any selected model.
        """
        state.last_test_result = result
        state.tried_models.add(result.model_id)
        state.attempt_count += 1

        if result.passed:
            state.selected_model = result.model_id
            return SelectionOutcome(
                action="pass",
                selected_model=result.model_id,
                reason=f"Model {result.model_id} passed test",
            )

        # Test failed - check termination conditions
        if state.loop_exhausted:
            return SelectionOutcome(
                action="fail_exhausted",
                reason=f"Exhausted all {len(state.candidate_pool)} candidates after {state.attempt_count} attempts",
            )

        if state.loop_budget_exceeded:
            return SelectionOutcome(
                action="fail_budget",
                reason=f"Hit max_attempts ({state.max_attempts}) with {len(state.untried_candidates)} candidates remaining",
            )

        return SelectionOutcome(
            action="continue",
            reason=f"Model {result.model_id} failed: {result.reason}; trying next",
        )

    def initial_outcome(self, state: DigestState) -> SelectionOutcome:
        """Check initial conditions before any test."""
        if not state.candidate_pool:
            return SelectionOutcome(
                action="fail_exhausted",
                reason="No free models available from provider",
            )
        return SelectionOutcome(action="continue", reason="Starting model selection")

    def summarize_decision(self, state: DigestState) -> str:
        """Produce a human-readable summary of the selection process."""
        tried = len(state.tried_models)
        total = len(state.candidate_pool)
        selected = state.selected_model or "none"
        return f"Model selection: tried {tried}/{total}, selected={selected}, outcome={state.outcome}"