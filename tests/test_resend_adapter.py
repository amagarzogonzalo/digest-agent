"""Tests for the Resend notifier adapter."""

import base64

from digest_agent.adapters.driven.resend_adapter import ResendAdapter
from digest_agent.ports import NotificationError

import digest_agent.adapters.driven.resend_adapter as resend_mod


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_send_posts_correct_payload(monkeypatch, tmp_path):
    pdf = tmp_path / "digest.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse(200, "")

    monkeypatch.setattr(resend_mod.requests, "post", fake_post)

    adapter = ResendAdapter(api_key="re_testkey", from_address="onboarding@resend.dev")
    adapter.send("amagarzo@gmail.com", "GitHub Daily Digest", str(pdf))

    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["headers"]["Authorization"] == "Bearer re_testkey"
    body = captured["json"]
    assert body["from"] == "onboarding@resend.dev"
    assert "amagarzo@gmail.com" in body["to"]
    assert body["subject"] == "GitHub Daily Digest"
    # Attachment is base64-encoded PDF bytes.
    assert body["attachments"][0]["content"] == base64.b64encode(b"%PDF-1.4 fake").decode()

    # Restore for any later tests in the module.
    monkeypatch.setattr(resend_mod.requests, "post", resend_mod.requests.post)


def test_send_missing_key_raises():
    adapter = ResendAdapter(api_key="")
    import pytest  # noqa: PLC0415
    with pytest.raises(NotificationError):
        adapter.send("a@b.com", "subject", "/tmp/nope.pdf")


def test_send_api_error_raises(monkeypatch, tmp_path):
    pdf = tmp_path / "d.pdf"
    pdf.write_bytes(b"x")

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(401, '{"message":"authentication error"}')

    monkeypatch.setattr(resend_mod.requests, "post", fake_post)

    adapter = ResendAdapter(api_key="re_bad")
    import pytest  # noqa: PLC0415
    with pytest.raises(NotificationError) as exc_info:
        adapter.send("a@b.com", "s", str(pdf))
    assert "401" in str(exc_info.value)

    monkeypatch.setattr(resend_mod.requests, "post", resend_mod.requests.post)