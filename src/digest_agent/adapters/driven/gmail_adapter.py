"""SMTP adapter - implements NotifierPort for any SMTP host (Gmail, Ethereal, ...).

Credentials come from environment variables:
- SMTP_HOST / SMTP_PORT / SMTP_STARTTLS (defaults: smtp.gmail.com / 465 / no STARTTLS)
- SMTP_USER / SMTP_PASSWORD (fall back to GMAIL_ADDRESS / GMAIL_APP_PASSWORD)

The core depends only on the port interface, not on smtplib or email SDKs.
"""
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from digest_agent.ports import NotifierPort


class SMTPAdapter(NotifierPort):
    """Adapter for SMTP - implements NotifierPort."""

    def __init__(self):
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "465"))
        self.starttls = os.environ.get("SMTP_STARTTLS", "").strip().lower() in ("1", "true", "yes")
        self.user = os.environ.get("SMTP_USER") or os.environ.get("GMAIL_ADDRESS", "")
        self.password = os.environ.get("SMTP_PASSWORD") or os.environ.get("GMAIL_APP_PASSWORD", "")

    def send(self, recipient: str, subject: str, attachment_path: str) -> None:
        """Send a notification with the given attachment."""
        if not self.user or not self.password:
            raise PermissionError("SMTP user/password not configured (SMTP_USER/GMAIL_ADDRESS, SMTP_PASSWORD/GMAIL_APP_PASSWORD)")

        msg = MIMEMultipart()
        msg["From"] = self.user
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText("Attached: your automated daily project digest.", "plain"))

        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(attachment_path))
            msg.attach(part)

        if self.smtp_port == 465:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port) as server:
                server.login(self.user, self.password)
                server.sendmail(self.user, recipient, msg.as_string())
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.sendmail(self.user, recipient, msg.as_string())


# Backwards-compatible alias for callers that import GmailAdapter by name.
GmailAdapter = SMTPAdapter