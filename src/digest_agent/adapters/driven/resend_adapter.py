"""Resend adapter - implements NotifierPort for Resend's HTTP API.

Sends emails via the Resend REST API using the `requests` HTTP client (no SDK
dependency). Configure with RESEND_API_KEY. Recipient(s) come from the digest
state's recipient, or from RECIPIENT_EMAIL env var.

Docs: https://resend.com/docs/api-reference/emails/send-email
"""
import base64
import os
import requests

from digest_agent.ports import NotifierPort, NotificationError


class ResendAdapter(NotifierPort):
    """Adapter for Resend - implements NotifierPort."""

    SEND_ENDPOINT = "https://api.resend.com/emails"

    def __init__(self, api_key: str | None = None, from_address: str | None = None):
        self.api_key = api_key or os.environ.get("RESEND_API_KEY", "")
        self.from_address = (
            from_address
            or os.environ.get("RESEND_FROM", "onboarding@resend.dev")
        )

    def send(self, recipient: str, subject: str, attachment_path: str) -> None:
        """Send the digest PDF via Resend."""
        if not self.api_key:
            raise NotificationError(
                "RESEND_API_KEY is not set. Create a key at https://resend.com/api-keys"
            )

        # Read the PDF so we can attach it as base64.
        with open(attachment_path, "rb") as f:
            pdf_bytes = f.read()

        payload = {
            "from": self.from_address,
            "to": [r.strip() for r in recipient.split(",") if r.strip()],
            "subject": subject,
            "text": "Attached: your automated daily project digest.",
            "attachments": [
                {
                    "filename": os.path.basename(attachment_path),
                    "content": base64.b64encode(pdf_bytes).decode("ascii"),
                }
            ],
        }

        resp = requests.post(
            self.SEND_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            raise NotificationError(f"Resend API error {resp.status_code}: {resp.text}")