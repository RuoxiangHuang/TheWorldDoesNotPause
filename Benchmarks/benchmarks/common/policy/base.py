"""Policy client interface for Real-Time benchmarks (in-process or remote)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PolicyClient(Protocol):
    """Process-agnostic policy API: sim sends obs, receives an action chunk."""

    benchmark: str

    def predict(self, obs: Any, instruction: str, **kwargs: Any) -> Any:
        """Return an action chunk (numpy array or list of actions)."""

    def close(self) -> None:
        """Release remote / GPU resources if any."""
