"""Tests for the Gemini (AI Studio) adapter."""

import pytest

from digest_agent.adapters.driven.gemini_adapter import GeminiAdapter, GeminiRateLimit
from digest_agent.core.text_utils import strip_reasoning_prefix

import digest_agent.adapters.driven.gemini_adapter as gemini_mod


class FakeResp:
    def __init__(self, status_code, body, text=""):
        self.status_code = status_code
        self._json = body
        self.text = text

    def json(self):
        return self._json


def test_complete_returns_text(monkeypatch):
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["model"] = url.split("/models/")[1].replace(":generateContent", "")
        captured["prompt"] = json["contents"][0]["parts"][0]["text"]
        return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "Hello world."}]}}]})

    monkeypatch.setattr(gemini_mod.requests, "post", fake_post)
    a = GeminiAdapter(api_key="k", candidate_models=["gemini-x"])
    assert a.complete("gemini-x", "hi") == "Hello world."
    assert captured["model"] == "gemini-x"
    assert "hi" in captured["prompt"]


def test_complete_raises_on_api_error(monkeypatch):
    def fake_post(url, params=None, json=None, timeout=None):
        return FakeResp(404, {"error": {"message": "not found"}}, text='{"error":{}}')

    monkeypatch.setattr(gemini_mod.requests, "post", fake_post)
    a = GeminiAdapter(api_key="k")
    with pytest.raises(Exception):
        a.complete("gemini-x", "hi")


def test_test_returns_failed_on_rate_limit(monkeypatch):
    def fake_post(url, params=None, json=None, timeout=None):
        return FakeResp(429, {"error": {}})

    monkeypatch.setattr(gemini_mod.requests, "post", fake_post)
    a = GeminiAdapter(api_key="k")
    res = a.test("gemini-x")
    assert res.passed is False
    assert res.reason == "rate_limited"


def test_list_free_models_caches():
    a = GeminiAdapter(api_key="k", candidate_models=["m1", "m2"])
    assert a.list_free_models() == ["m1", "m2"]
    a.candidate_models.append("m3")
    # Cached (within 30 min) so still m1, m2.
    assert a.list_free_models() == ["m1", "m2"]


def test_strip_gemini_thinking_tags():
    text = "<thinking>internal reasoning</thinking>Final answer"
    assert strip_reasoning_prefix(text) == "Final answer"