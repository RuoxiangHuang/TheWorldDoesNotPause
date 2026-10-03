"""Per-replan cache counters for FastWAM and π₀.₅ stacks.

Closed modules stay absent. Callers mark them N/A. Missing counters stay
unknown rather than zero.
"""

from __future__ import annotations

from typing import Any


def _plain(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return str(value)


def _controller_stats(controller: Any) -> dict[str, Any] | None:
    stats = getattr(controller, "stats", None)
    if callable(stats):
        stats = stats()
    if isinstance(stats, dict):
        return {str(k): _plain(v) for k, v in stats.items()}
    return None


def snapshot_model(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stack = getattr(model, "_fasterwam_accel", None)
    if stack is not None and hasattr(stack, "stats"):
        raw = stack.stats()
        if isinstance(raw, dict):
            out["fasterwam"] = {str(k): _plain(v) for k, v in raw.items()}
    controllers = getattr(model, "_fasterpi_controllers", None) or []
    pi: dict[str, Any] = {}
    for controller in controllers:
        name = getattr(controller, "name", None) or type(controller).__name__
        stats = _controller_stats(controller)
        if stats is not None:
            pi[str(name)] = stats
    if pi:
        out["fasterpi"] = pi
    return out


def snapshot_policy(policy: Any) -> dict[str, Any]:
    fetcher = getattr(policy, "cache_stats", None)
    if callable(fetcher):
        try:
            payload = fetcher()
        except Exception as exc:  # noqa: BLE001
            return {"status": "unknown", "error": str(exc)}
        if isinstance(payload, dict):
            return payload
        return {"status": "unknown"}
    model = None
    ctx = getattr(policy, "ctx", None)
    if ctx is not None:
        model = getattr(ctx, "model", None)
    inner = getattr(policy, "policy", None)
    if model is None and inner is not None:
        model = getattr(inner, "_model", None)
    if model is None:
        model = getattr(policy, "model", None)
    if model is None:
        return {"status": "unknown"}
    snap = snapshot_model(model)
    if not snap:
        return {"status": "unknown"}
    return snap


_RATE_KEYS = {"hit_rate", "skip_rate"}


def _numeric_delta(before: Any, after: Any) -> Any:
    if isinstance(after, bool) or isinstance(before, bool):
        return after
    if isinstance(after, dict) and isinstance(before, dict):
        keys = set(before) | set(after)
        out = {}
        for key in sorted(keys):
            if key in _RATE_KEYS:
                continue
            out[key] = _numeric_delta(before.get(key), after.get(key))
        return _recompute_rates(out)
    if isinstance(after, dict) and before is None:
        out = {
            k: _numeric_delta(0, v) if isinstance(v, (int, float)) and k not in _RATE_KEYS else v
            for k, v in after.items()
            if k not in _RATE_KEYS
        }
        return _recompute_rates(out)
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return after - before
    return after


def _recompute_rates(node: dict[str, Any]) -> dict[str, Any]:
    """Rates are ratios, not counters. Rebuild them from the count delta."""
    hits, misses = node.get("hits"), node.get("misses")
    if isinstance(hits, (int, float)) and isinstance(misses, (int, float)):
        total = hits + misses
        node["hit_rate"] = (hits / total) if total else None
    skipped, computed = node.get("skipped"), node.get("computed")
    if isinstance(skipped, (int, float)) and isinstance(computed, (int, float)):
        total = skipped + computed
        node["skip_rate"] = (skipped / total) if total else None
    return node


def diff_snapshots(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if not after or after.get("status") == "unknown":
        return {"status": "unknown"}
    if not before or before.get("status") == "unknown":
        return {"status": "unknown"}
    return _numeric_delta(before, after)
