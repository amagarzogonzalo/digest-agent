"""Core application components - no vendor dependencies."""

from .state import DigestState
from .policy import ModelSelectionPolicy, SelectionOutcome
from .orchestrator import Orchestrator, build_orchestrator
from .fakes import (
    FakeActivitySource,
    FakeLLMProvider,
    FakeDocumentRenderer,
    FakeNotifier,
    create_test_state,
)

__all__ = [
    "DigestState",
    "ModelSelectionPolicy",
    "SelectionOutcome",
    "Orchestrator",
    "build_orchestrator",
    "FakeActivitySource",
    "FakeLLMProvider",
    "FakeDocumentRenderer",
    "FakeNotifier",
    "create_test_state",
]