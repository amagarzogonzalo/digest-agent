"""Document renderer port - renders summary text to a file."""

from abc import ABC, abstractmethod


class DocumentRendererPort(ABC):
    """Port for rendering a digest summary to a document file."""

    @abstractmethod
    def render(self, summary: str) -> str:
        """Render the summary to a document and return the file path.

        Args:
            summary: The digest summary text (markdown-ish format).

        Returns:
            Path to the generated document file.
        """
        pass


class RenderError(Exception):
    """Raised when rendering the document fails."""
    pass