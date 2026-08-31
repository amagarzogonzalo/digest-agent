"""Notifier port - sends notifications with attachments."""

from abc import ABC, abstractmethod


class NotifierPort(ABC):
    """Port for sending a notification with an attachment."""

    @abstractmethod
    def send(self, recipient: str, subject: str, attachment_path: str) -> None:
        """Send a notification with the given attachment.

        Args:
            recipient: The recipient address/identifier.
            subject: The notification subject line.
            attachment_path: Path to the file to attach.
        """
        pass


class NotificationError(Exception):
    """Raised when sending the notification fails."""
    pass