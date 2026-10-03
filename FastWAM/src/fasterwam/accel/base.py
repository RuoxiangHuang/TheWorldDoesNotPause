"""Shared plumbing for the acceleration controllers.

Every controller is an instance-level monkeypatch: it swaps a bound method on a
live model and can restore it exactly. Nothing in `fasterwam.models` needs to
know that acceleration exists, so an accelerated and a reference model can be
compared in the same process.
"""

from __future__ import annotations

import abc
from typing import Any, Callable


class Controller(abc.ABC):
    """A reversible, instance-level patch on a model."""

    name: str = "controller"
    # Attributes the model must expose for this controller to be installable.
    # Not every model in the package has the action-denoising path, so the
    # stack skips controllers whose hooks are absent rather than crashing.
    REQUIRED_ATTRS: tuple[str, ...] = ()

    @classmethod
    def missing_requirements(cls, model: Any, config: Any = None) -> list[str]:
        return [attr for attr in cls.REQUIRED_ATTRS if not hasattr(model, attr)]

    def __init__(self, model: Any):
        self.model = model
        # (owner, attr, attr_was_in_instance_dict, previous_instance_value)
        self._patches: list[tuple[Any, str, bool, Any]] = []

    @property
    def installed(self) -> bool:
        return bool(self._patches)

    def _patch(self, owner: Any, attr: str, factory: Callable[[Callable], Callable]) -> None:
        """Replace `owner.attr` with `factory(original_bound_method)`.

        The original is captured as a bound method so the wrapper can call
        through without re-entering the patch.
        """
        original = getattr(owner, attr)
        had_own = attr in vars(owner)
        self._patches.append((owner, attr, had_own, vars(owner).get(attr)))
        setattr(owner, attr, factory(original))

    def install(self) -> "Controller":
        if self.installed:
            return self
        self._install()
        return self

    def uninstall(self) -> None:
        for owner, attr, had_own, previous in reversed(self._patches):
            if had_own:
                setattr(owner, attr, previous)
            else:
                # Drop the instance attribute so lookups fall back to the class.
                try:
                    delattr(owner, attr)
                except AttributeError:
                    pass
        self._patches.clear()
        self._on_uninstall()

    @abc.abstractmethod
    def _install(self) -> None:
        ...

    def _on_uninstall(self) -> None:
        """Hook for controllers that hold cached state."""

    def reset(self) -> None:
        """Drop cached state that must not leak across episodes.

        Cross-replan caches are intentional *inside* an episode. Episode
        boundaries (new seed, new task, post-latency-warmup) have to start
        cold. Default is a no-op; stateful controllers override.
        """

    def stats(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} installed={self.installed}>"
