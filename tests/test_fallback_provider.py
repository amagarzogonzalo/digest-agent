"""Tests for the fallback (multi-provider) LLM provider and reasoning stripping."""

import pytest

from digest_agent.adapters.driven.fallback_provider import FallbackLLMProvider
from digest_agent.adapters.driven.openai_compat import strip_reasoning_prefix
from digest_agent.ports import LLMProviderPort, ModelId, TestResult, CompletionError


class _Stub(LLMProviderPort):
    """Minimal in-memory provider for testing the fallback chain."""

    def __init__(self, label, models, tests=None, completions=None):
        self.label = label
        self.models = models
        self.tests = tests or {}
        self.completions = completions or {}
        self.calls = []

    def list_free_models(self):
        return list(self.models)

    def test(self, model_id):
        self.calls.append(("test", model_id))
        return self.tests.get(model_id) or TestResult(model_id=model_id, passed=True, reason="ok")

    def complete(self, model_id, prompt):
        self.calls.append(("complete", model_id, prompt))
        return self.completions.get(model_id, f"answer from {model_id}")


class TestFallbackProvider:
    def test_order_groq_then_mistral_then_openrouter(self):
        groq = _Stub("groq", ["groq-m"], tests={"groq-m": TestResult("groq-m", False, "err")})
        mistral = _Stub("mistral", ["mistral-m"])
        openrouter = _Stub("openrouter", ["or-m"])
        fb = FallbackLLMProvider([groq, mistral, openrouter])

        models = fb.list_free_models()
        assert models == ["groq-m", "mistral-m", "or-m"]

        # Testing groq-m fails; mistral-m is the next candidate -> should pass.
        assert fb.test("groq-m").passed is False
        assert fb.test("mistral-m").passed is True
        assert fb.complete("or-m", "hi") == "answer from or-m"
        assert ("complete", "or-m", "hi") in openrouter.calls

    def test_skips_provider_that_raises_on_list(self):
        bad = _Stub("bad", [])
        bad.list_free_models = lambda: (_ for _ in ()).throw(RuntimeError("auth"))
        good = _Stub("good", ["good-m"], tests={"good-m": TestResult("good-m", False, "err")})
        fb = FallbackLLMProvider([bad, good])

        assert "good-m" in fb.list_free_models()

    def test_route_unknown_model_raises(self):
        fb = FallbackLLMProvider([_Stub("a", ["a-m"])])
        with pytest.raises(CompletionError):
            fb.complete("ghost", "hi")


class TestStripReasoning:
    def test_strips_thinking_block(self):
        text = "thinking\n1. analyze\n2. decide\n\nHere is the answer."
        out = strip_reasoning_prefix(text)
        assert "thinking" not in out.lower()

    def test_passthrough_clean(self):
        clean = "# Daily Digest\n\n- item"
        assert strip_reasoning_prefix(clean) == clean

    def test_strips_thinking_tags(self):
        text = "<thinking>reasoning here</thinking>Final answer"
        assert strip_reasoning_prefix(text) == "Final answer"