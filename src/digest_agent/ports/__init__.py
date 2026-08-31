"""Port interfaces for the GitHub digest agent.

These abstract contracts define what the application core depends on.
Adapters implement these ports. The core never imports adapters or vendor SDKs.
"""

from .activity_source import (
    ActivitySourcePort,
    RepoActivity,
    Commit,
    PullRequest,
    FileChange,
    ActivityFetchError,
)
from .llm_provider import LLMProviderPort, ModelId, TestResult, CompletionError
from .document_renderer import DocumentRendererPort, RenderError
from .notifier import NotifierPort, NotificationError

__all__ = [
    "ActivitySourcePort",
    "RepoActivity",
    "Commit",
    "PullRequest",
    "FileChange",
    "ActivityFetchError",
    "LLMProviderPort",
    "ModelId",
    "TestResult",
    "CompletionError",
    "DocumentRendererPort",
    "RenderError",
    "NotifierPort",
    "NotificationError",
]