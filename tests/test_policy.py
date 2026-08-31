"""Unit tests for ModelSelectionPolicy - testing all three loop exit paths."""

import pytest

from digest_agent.core import ModelSelectionPolicy, create_test_state
from digest_agent.ports import TestResult, ModelId


class TestModelSelectionPolicy:
    """Tests for the evaluator-optimizer model selection policy."""

    def test_pass_path_first_model_works(self):
        """PASS path: first candidate passes test."""
        state = create_test_state(candidate_pool=["model-a", "model-b"])
        policy = ModelSelectionPolicy()

        # First model passes
        result = TestResult(model_id="model-a", passed=True, reason="ok")
        outcome = policy.on_test_result(state, result)

        assert outcome.action == "pass"
        assert outcome.selected_model == "model-a"
        assert state.selected_model == "model-a"
        assert state.attempt_count == 1
        assert state.tried_models == {"model-a"}

    def test_pass_path_second_model_works(self):
        """PASS path: first fails, second passes."""
        state = create_test_state(candidate_pool=["model-a", "model-b", "model-c"])
        policy = ModelSelectionPolicy()

        # First fails
        result1 = TestResult(model_id="model-a", passed=False, reason="timeout")
        outcome1 = policy.on_test_result(state, result1)
        assert outcome1.action == "continue"
        assert state.attempt_count == 1

        # Second passes
        result2 = TestResult(model_id="model-b", passed=True, reason="ok")
        outcome2 = policy.on_test_result(state, result2)
        assert outcome2.action == "pass"
        assert outcome2.selected_model == "model-b"
        assert state.selected_model == "model-b"
        assert state.attempt_count == 2
        assert state.tried_models == {"model-a", "model-b"}

    def test_fail_exhausted_all_candidates_fail(self):
        """FAIL_EXHAUSTED: all candidates fail, pool exhausted."""
        state = create_test_state(candidate_pool=["model-a", "model-b"], max_attempts=10)
        policy = ModelSelectionPolicy()

        # First fails
        result1 = TestResult(model_id="model-a", passed=False, reason="rate_limited")
        policy.on_test_result(state, result1)

        # Second fails - pool exhausted
        result2 = TestResult(model_id="model-b", passed=False, reason="empty_response")
        outcome = policy.on_test_result(state, result2)

        assert outcome.action == "fail_exhausted"
        assert state.selected_model is None
        assert state.attempt_count == 2
        assert state.tried_models == {"model-a", "model-b"}
        assert "Exhausted all 2 candidates" in outcome.reason

    def test_fail_budget_max_attempts_reached_before_exhaustion(self):
        """FAIL_BUDGET: max_attempts hit before pool exhausted."""
        state = create_test_state(candidate_pool=["model-a", "model-b", "model-c"], max_attempts=2)
        policy = ModelSelectionPolicy()

        # First fails
        result1 = TestResult(model_id="model-a", passed=False, reason="timeout")
        policy.on_test_result(state, result1)

        # Second fails - budget exceeded (max_attempts=2)
        result2 = TestResult(model_id="model-b", passed=False, reason="timeout")
        outcome = policy.on_test_result(state, result2)

        assert outcome.action == "fail_budget"
        assert state.selected_model is None
        assert state.attempt_count == 2
        assert state.tried_models == {"model-a", "model-b"}
        assert "Hit max_attempts (2)" in outcome.reason
        assert "1 candidates remaining" in outcome.reason

    def test_initial_no_candidates(self):
        """Initial check: empty candidate pool -> fail_exhausted."""
        state = create_test_state(candidate_pool=[])
        policy = ModelSelectionPolicy()

        outcome = policy.initial_outcome(state)
        assert outcome.action == "fail_exhausted"
        assert "No free models available" in outcome.reason

    def test_initial_has_candidates(self):
        """Initial check: has candidates -> continue."""
        state = create_test_state(candidate_pool=["model-a"])
        policy = ModelSelectionPolicy()

        outcome = policy.initial_outcome(state)
        assert outcome.action == "continue"

    def test_summarize_decision(self):
        """Test the human-readable summary."""
        state = create_test_state(candidate_pool=["model-a", "model-b"])
        policy = ModelSelectionPolicy()

        result = TestResult(model_id="model-a", passed=True, reason="ok")
        policy.on_test_result(state, result)

        summary = policy.summarize_decision(state)
        assert "tried 1/2" in summary
        assert "selected=model-a" in summary
        assert "outcome=" in summary

    def test_rate_limited_not_treated_as_broken(self):
        """Rate limit failures are logged separately but still count as failures."""
        state = create_test_state(candidate_pool=["model-a", "model-b"], max_attempts=5)
        policy = ModelSelectionPolicy()

        # Rate limited - not "broken" but still fails this attempt
        result = TestResult(model_id="model-a", passed=False, reason="rate_limited")
        outcome = policy.on_test_result(state, result)

        assert outcome.action == "continue"
        assert state.attempt_count == 1
        assert "rate_limited" in outcome.reason

    def test_malformed_response_fails(self):
        """Malformed response is a failure."""
        state = create_test_state(candidate_pool=["model-a", "model-b"])
        policy = ModelSelectionPolicy()

        result = TestResult(model_id="model-a", passed=False, reason="malformed")
        outcome = policy.on_test_result(state, result)

        assert outcome.action == "continue"
        assert state.attempt_count == 1

    def test_empty_candidate_pool_after_some_tried(self):
        """When untried candidates empty but not all tried (edge case)."""
        state = create_test_state(candidate_pool=["model-a"])
        policy = ModelSelectionPolicy()

        result = TestResult(model_id="model-a", passed=False, reason="timeout")
        outcome = policy.on_test_result(state, result)

        assert outcome.action == "fail_exhausted"
        assert state.attempt_count == 1