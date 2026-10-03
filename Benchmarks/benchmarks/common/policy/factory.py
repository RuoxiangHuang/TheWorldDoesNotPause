"""Factory helpers for PolicyClient instances."""

from __future__ import annotations

from typing import Any

from .base import PolicyClient
from .http_client import HttpPolicyClient
from .inprocess import LiberoInProcessPolicy, RoboCasaInProcessPolicy, RoboTwinInProcessPolicy


def make_inprocess_policy(benchmark: str, ctx: Any, *, backend: str = "fastwam") -> PolicyClient:
    del backend  # reserved; FastWAM is the only in-process backend
    benchmark = benchmark.lower()
    if benchmark == "libero":
        return LiberoInProcessPolicy(ctx=ctx)
    if benchmark == "robotwin":
        return RoboTwinInProcessPolicy(ctx=ctx)
    if benchmark == "robocasa":
        return RoboCasaInProcessPolicy(ctx=ctx)
    raise ValueError(f"unknown benchmark {benchmark!r}")


def make_policy_client(
    benchmark: str,
    *,
    policy_url: str | None = None,
    ctx: Any | None = None,
    backend: str = "fastwam",
    codec: str | None = None,
) -> PolicyClient:
    if policy_url:
        # codec=None → HttpPolicyClient reads POLICY_HTTP_CODEC (default binary).
        kwargs = {"url": policy_url, "benchmark": benchmark.lower()}
        if codec is not None:
            kwargs["codec"] = codec  # type: ignore[assignment]
        return HttpPolicyClient(**kwargs)
    if ctx is None:
        raise ValueError("either policy_url or ctx is required")
    return make_inprocess_policy(benchmark, ctx, backend=backend)
