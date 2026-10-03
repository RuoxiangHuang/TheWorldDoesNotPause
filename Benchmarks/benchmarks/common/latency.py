"""Per-replan latency coupling: measured / replay / constant modes.

Paper protocol uses continuously valued
    n_delay^(c) = L_c / Δt = L_c * control_hz
rather than a single median freeze injected for the whole episode.

Modes
-----
constant
    Legacy / synthetic SR(L): one ``n_freeze`` for every replan.
measured
    Time each ``policy.predict`` (CUDA sync + perf_counter); that L_c advances
    the world and is appended to a :class:`LatencyTrace`.
replay
    Still call ``predict`` for actions, but advance the world with L_c taken
    from a previously recorded trace (indexed by replan). Enables deterministic
    pairing while preserving per-replan variance.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .protocol import DEFAULT_DEADLINE_MS

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class LatencyMode(str, Enum):
    CONSTANT = "constant"
    MEASURED = "measured"
    REPLAY = "replay"


def parse_latency_mode(value: str | LatencyMode | None) -> LatencyMode:
    if value is None:
        return LatencyMode.MEASURED
    if isinstance(value, LatencyMode):
        return value
    key = str(value).strip().lower()
    try:
        return LatencyMode(key)
    except ValueError as e:
        raise ValueError(
            f"latency_mode must be one of {[m.value for m in LatencyMode]}, got {value!r}"
        ) from e


def n_delay_from_latency(latency_s: float, control_hz: float) -> float:
    """Map wall-clock seconds to fractional control ticks (no rounding)."""
    lat = float(latency_s)
    hz = float(control_hz)
    if lat < 0.0:
        raise ValueError(f"latency_s must be >= 0, got {lat}")
    if hz <= 0.0:
        raise ValueError(f"control_hz must be > 0, got {hz}")
    if not math.isfinite(lat):
        return float("inf")
    return max(0.0, lat * hz)


def _cuda_sync() -> None:
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_policy_predict(policy: Any, obs: Any, instruction: str, **kwargs: Any) -> tuple[Any, float]:
    """Return ``(chunk, latency_s)`` with end-to-end wall-clock timing."""
    _cuda_sync()
    t0 = time.perf_counter()
    chunk = policy.predict(obs, instruction, **kwargs)
    _cuda_sync()
    return chunk, float(time.perf_counter() - t0)


@dataclass
class ReplanLatencyRecord:
    """One replan's timing + observation-age samples (ticks).

    Sequential measured-delay protocol (not overlapping GPU+sim threads):
    - ``L_model``: wall-clock of ``policy.predict`` (CUDA synchronized).
    - ``delta``: ``τ_ready - τ_obs``. Predict is blocking, so this equals ``L_model``.
    - ``L_act``: ``τ_exec - τ_obs``, world time until the new chunk's first action.
      After thinking injection this is ``n_delay / control_hz`` (equals ``L_model``
      when the coupler uses the same measured seconds).
    """

    replan_idx: int
    latency_s: float
    n_delay: float
    action_ages_ticks: List[float] = field(default_factory=list)
    L_model_s: float | None = None
    delta_s: float | None = None
    L_act_s: float | None = None
    # Extra delivery wait after compute. World coupling uses L_model + this
    # wait; ``latency_s`` / ``L_model_s`` stay the measured compute time.
    delivery_delay_s: float = 0.0
    t_obs_s: float | None = None
    t_ready_s: float | None = None
    t_apply_s: float | None = None
    cache: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LatencyTrace:
    """Ordered per-replan latency records for one episode."""

    control_hz: float
    deadline_ms: float = DEFAULT_DEADLINE_MS
    mode: str = LatencyMode.MEASURED.value
    records: List[ReplanLatencyRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def append(
        self,
        replan_idx: int,
        latency_s: float,
        n_delay: float | None = None,
        *,
        delivery_delay_s: float = 0.0,
        t_obs_s: float | None = None,
        cache: dict[str, Any] | None = None,
    ) -> ReplanLatencyRecord:
        """Record one replan.

        ``latency_s`` is measured compute time. ``delivery_delay_s`` is an extra
        wait after the action is ready and before the executor installs it.
        World advance is ``compute + delivery`` unless ``n_delay`` is passed.
        """
        compute = float(latency_s)
        delivery = float(delivery_delay_s)
        if delivery < 0.0:
            raise ValueError(f"delivery_delay_s must be >= 0, got {delivery}")
        world_s = compute + delivery
        n = (
            float(n_delay)
            if n_delay is not None
            else n_delay_from_latency(world_s, self.control_hz)
        )
        rec = ReplanLatencyRecord(
            replan_idx=int(replan_idx),
            latency_s=compute,
            n_delay=n,
            delivery_delay_s=delivery,
            cache=None if cache is None else dict(cache),
        )
        rec.L_model_s = compute
        rec.delta_s = compute
        if math.isfinite(n) and float(self.control_hz) > 0.0:
            rec.L_act_s = float(n) / float(self.control_hz)
        else:
            rec.L_act_s = float("inf")
        if t_obs_s is not None and math.isfinite(float(t_obs_s)):
            t_obs = float(t_obs_s)
            rec.t_obs_s = t_obs
            rec.t_ready_s = t_obs + compute
            rec.t_apply_s = t_obs + compute + delivery
        self.records.append(rec)
        return rec

    def n_delay_at(self, replan_idx: int, *, pad_last: bool = True) -> float:
        """Lookup n_delay for replan index (0-based)."""
        idx = int(replan_idx)
        if not self.records:
            raise ValueError("LatencyTrace is empty")
        for rec in self.records:
            if rec.replan_idx == idx:
                return float(rec.n_delay)
        if pad_last:
            return float(self.records[-1].n_delay)
        raise KeyError(f"replan_idx={idx} not in latency trace")

    def latency_s_at(self, replan_idx: int, *, pad_last: bool = True) -> float:
        idx = int(replan_idx)
        if not self.records:
            raise ValueError("LatencyTrace is empty")
        for rec in self.records:
            if rec.replan_idx == idx:
                return float(rec.latency_s)
        if pad_last:
            return float(self.records[-1].latency_s)
        raise KeyError(f"replan_idx={idx} not in latency trace")

    def note_action_age(self, replan_idx: int, action_j: int, n_delay: float) -> float:
        """Observation age in ticks when action ``j`` of this chunk is applied."""
        age = float(n_delay) + float(action_j)
        for rec in self.records:
            if rec.replan_idx == int(replan_idx):
                rec.action_ages_ticks.append(age)
                return age
        # Constant mode may not pre-append; create a stub record.
        rec = self.append(replan_idx, latency_s=float(n_delay) / max(self.control_hz, 1e-9), n_delay=n_delay)
        rec.action_ages_ticks.append(age)
        return age

    def all_ages(self) -> list[float]:
        out: list[float] = []
        for rec in self.records:
            out.extend(float(a) for a in rec.action_ages_ticks)
        return out

    def summary(self) -> dict[str, Any]:
        lats = [float(r.latency_s) for r in self.records if math.isfinite(r.latency_s)]
        nds = [float(r.n_delay) for r in self.records if math.isfinite(r.n_delay)]
        l_model = [
            float(r.L_model_s)
            for r in self.records
            if r.L_model_s is not None and math.isfinite(float(r.L_model_s))
        ]
        deltas = [
            float(r.delta_s)
            for r in self.records
            if r.delta_s is not None and math.isfinite(float(r.delta_s))
        ]
        l_act = [
            float(r.L_act_s)
            for r in self.records
            if r.L_act_s is not None and math.isfinite(float(r.L_act_s))
        ]
        ages = self.all_ages()
        deadline_s = float(self.deadline_ms) / 1000.0
        miss = None
        if lats:
            miss = float(np.mean([1.0 if x > deadline_s else 0.0 for x in lats]))

        def _pct(xs: Sequence[float], q: float) -> float | None:
            if not xs:
                return None
            return float(np.percentile(np.asarray(xs, dtype=np.float64), q))

        return {
            "mode": self.mode,
            "control_hz": float(self.control_hz),
            "deadline_ms": float(self.deadline_ms),
            "n_replans_timed": len(self.records),
            "latency_p50_ms": None if not lats else 1000.0 * _pct(lats, 50),  # type: ignore[operator]
            "latency_p95_ms": None if not lats else 1000.0 * _pct(lats, 95),  # type: ignore[operator]
            "latency_mean_ms": None if not lats else 1000.0 * float(np.mean(lats)),
            "L_model_mean_s": None if not l_model else float(np.mean(l_model)),
            "L_model_p50_ms": None if not l_model else 1000.0 * _pct(l_model, 50),  # type: ignore[operator]
            "delta_mean_s": None if not deltas else float(np.mean(deltas)),
            "delta_p50_ms": None if not deltas else 1000.0 * _pct(deltas, 50),  # type: ignore[operator]
            "L_act_mean_s": None if not l_act else float(np.mean(l_act)),
            "L_act_p50_ms": None if not l_act else 1000.0 * _pct(l_act, 50),  # type: ignore[operator]
            "n_delay_p50": _pct(nds, 50),
            "n_delay_p95": _pct(nds, 95),
            "n_delay_mean": None if not nds else float(np.mean(nds)),
            "deadline_miss_rate": miss,
            "obs_age_ticks_p50": _pct(ages, 50),
            "obs_age_ticks_p95": _pct(ages, 95),
            "obs_age_ticks_mean": None if not ages else float(np.mean(ages)),
            "obs_age_ms_p50": None
            if not ages
            else 1000.0 * float(_pct(ages, 50)) / float(self.control_hz),  # type: ignore[arg-type]
            "obs_age_ms_p95": None
            if not ages
            else 1000.0 * float(_pct(ages, 95)) / float(self.control_hz),  # type: ignore[arg-type]
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_hz": float(self.control_hz),
            "deadline_ms": float(self.deadline_ms),
            "mode": self.mode,
            "meta": dict(self.meta),
            "records": [r.as_dict() for r in self.records],
            "summary": self.summary(),
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LatencyTrace":
        data = json.loads(Path(path).read_text())
        tr = cls(
            control_hz=float(data.get("control_hz", 20.0)),
            deadline_ms=float(data.get("deadline_ms", DEFAULT_DEADLINE_MS)),
            mode=str(data.get("mode", LatencyMode.REPLAY.value)),
            meta=dict(data.get("meta") or {}),
        )
        for raw in data.get("records") or []:
            tr.records.append(
                ReplanLatencyRecord(
                    replan_idx=int(raw["replan_idx"]),
                    latency_s=float(raw["latency_s"]),
                    n_delay=float(raw["n_delay"]),
                    action_ages_ticks=[float(x) for x in raw.get("action_ages_ticks") or []],
                    L_model_s=(
                        None if raw.get("L_model_s") is None else float(raw["L_model_s"])
                    ),
                    delta_s=None if raw.get("delta_s") is None else float(raw["delta_s"]),
                    L_act_s=None if raw.get("L_act_s") is None else float(raw["L_act_s"]),
                    delivery_delay_s=float(raw.get("delivery_delay_s") or 0.0),
                    t_obs_s=None if raw.get("t_obs_s") is None else float(raw["t_obs_s"]),
                    t_ready_s=None if raw.get("t_ready_s") is None else float(raw["t_ready_s"]),
                    t_apply_s=None if raw.get("t_apply_s") is None else float(raw["t_apply_s"]),
                    cache=None if raw.get("cache") is None else dict(raw["cache"]),
                )
            )
        return tr


@dataclass
class LatencyCoupler:
    """Resolves per-replan ``n_freeze`` / ``n_delay`` under the selected mode."""

    mode: LatencyMode
    control_hz: float
    deadline_ms: float = DEFAULT_DEADLINE_MS
    constant_n_freeze: float = 0.0
    replay_trace: Optional[LatencyTrace] = None
    trace: LatencyTrace = field(init=False)

    def __post_init__(self) -> None:
        self.mode = parse_latency_mode(self.mode)
        self.trace = LatencyTrace(
            control_hz=float(self.control_hz),
            deadline_ms=float(self.deadline_ms),
            mode=self.mode.value,
        )
        if self.mode == LatencyMode.REPLAY:
            if self.replay_trace is None or not self.replay_trace.records:
                raise ValueError("latency_mode=replay requires a non-empty replay_trace")

    def resolve_and_record(
        self,
        replan_idx: int,
        *,
        measured_latency_s: float | None = None,
        delivery_delay_s: float = 0.0,
        t_obs_s: float | None = None,
        cache: dict[str, Any] | None = None,
    ) -> float:
        """Return world-advance ticks for this replan and append the trace.

        ``delivery_delay_s`` is a post-compute wait. It lengthens ``n_delay``
        and ``L_act`` while ``latency_s`` / ``L_model`` stay the CUDA-timed
        compute. Replay traces already contain their world delay.
        """
        idx = int(replan_idx)
        delivery = float(delivery_delay_s)
        if delivery < 0.0:
            raise ValueError(f"delivery_delay_s must be >= 0, got {delivery}")
        extra = {
            "delivery_delay_s": delivery,
            "t_obs_s": t_obs_s,
            "cache": cache,
        }
        if self.mode == LatencyMode.CONSTANT:
            n_compute = float(self.constant_n_freeze)
            lat = (
                float(n_compute) / float(self.control_hz)
                if math.isfinite(n_compute)
                else float("inf")
            )
            n = n_compute
            if delivery > 0.0 and math.isfinite(n_compute):
                n = n_compute + n_delay_from_latency(delivery, self.control_hz)
            self.trace.append(idx, latency_s=lat, n_delay=n, **extra)
            return n
        if self.mode == LatencyMode.MEASURED:
            if measured_latency_s is None:
                raise ValueError("measured mode requires measured_latency_s")
            compute = float(measured_latency_s)
            n = n_delay_from_latency(compute + delivery, self.control_hz)
            self.trace.append(idx, latency_s=compute, n_delay=n, **extra)
            return n
        assert self.replay_trace is not None
        n = float(self.replay_trace.n_delay_at(idx))
        lat = float(self.replay_trace.latency_s_at(idx))
        extra["delivery_delay_s"] = 0.0
        self.trace.append(idx, latency_s=lat, n_delay=n, **extra)
        return n

    def on_action_applied(self, replan_idx: int, action_j: int, n_delay: float) -> float:
        return self.trace.note_action_age(replan_idx, action_j, n_delay)


def episode_latency_fields(coupler: LatencyCoupler) -> dict[str, Any]:
    """Flatten coupler.trace summary into episode JSON fields."""
    summary = coupler.trace.summary()
    # Keep a short alias list used by aggregators / paper tables.
    return {
        "latency_mode": coupler.mode.value,
        "latency_trace": coupler.trace.to_dict(),
        "latency_p50_ms": summary["latency_p50_ms"],
        "latency_p95_ms": summary["latency_p95_ms"],
        "L_model_mean_s": summary["L_model_mean_s"],
        "L_model_p50_ms": summary["L_model_p50_ms"],
        "delta_mean_s": summary["delta_mean_s"],
        "delta_p50_ms": summary["delta_p50_ms"],
        "L_act_mean_s": summary["L_act_mean_s"],
        "L_act_p50_ms": summary["L_act_p50_ms"],
        "n_delay_mean": summary["n_delay_mean"],
        "n_delay_p50": summary["n_delay_p50"],
        "n_delay_p95": summary["n_delay_p95"],
        "deadline_miss_rate": summary["deadline_miss_rate"],
        "obs_age_ticks_mean": summary["obs_age_ticks_mean"],
        "obs_age_ticks_p50": summary["obs_age_ticks_p50"],
        "obs_age_ticks_p95": summary["obs_age_ticks_p95"],
        "obs_age_ms_p50": summary["obs_age_ms_p50"],
        "obs_age_ms_p95": summary["obs_age_ms_p95"],
        # Representative scalar for legacy consumers (mean n_delay).
        "n_freeze": summary["n_delay_mean"],
    }
