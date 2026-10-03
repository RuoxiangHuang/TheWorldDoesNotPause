"""Numpy ↔ HTTP helpers for Real-Time policy transport.

Two wire formats share the same logical payload ``{obs, instruction, kwargs}``:

* **JSON** (legacy): ndarrays as ``{__ndarray__, data: list, dtype, shape}``.
* **Binary** (default): JSON header + raw C-contiguous bytes.
  Content-Type: ``application/x-realtime-numpy``.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_BIN = "application/x-realtime-numpy"
BIN_MAGIC = b"RTBIN1\0"
_BIN_HEADER = struct.Struct("<I")  # header byte length (LE uint32)


def encode_obs(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): encode_obs(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_obs(x) for x in obj]
    return obj


def decode_obs(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            arr = np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"])
        return {k: decode_obs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_obs(x) for x in obj]
    return obj


def encode_chunk(chunk: Any) -> list:
    arr = np.asarray(chunk)
    return arr.tolist()


def decode_chunk(data: list) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _is_ndarray_placeholder(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("__ndarray__") is True and "data" not in obj and "offset" in obj


def _collect_arrays(obj: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        idx = len(arrays)
        arrays.append(arr)
        return {
            "__ndarray__": True,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "offset": idx,
        }
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _collect_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_collect_arrays(x, arrays) for x in obj]
    return obj


def _restore_arrays(obj: Any, arrays: list[np.ndarray]) -> Any:
    if _is_ndarray_placeholder(obj):
        arr = arrays[int(obj["offset"])]
        return arr.reshape(tuple(obj["shape"]))
    if isinstance(obj, dict):
        if obj.get("__ndarray__") and "data" in obj:
            arr = np.asarray(obj["data"], dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"])
        return {k: _restore_arrays(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(x, arrays) for x in obj]
    return obj


def pack_binary(payload: dict[str, Any]) -> bytes:
    arrays: list[np.ndarray] = []
    meta = _collect_arrays(payload, arrays)
    header = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    parts = [BIN_MAGIC, _BIN_HEADER.pack(len(header)), header]
    for arr in arrays:
        parts.append(arr.tobytes(order="C"))
    return b"".join(parts)


def unpack_binary(raw: bytes) -> dict[str, Any]:
    if not raw.startswith(BIN_MAGIC):
        raise ValueError(f"bad binary magic: {raw[:8]!r}")
    off = len(BIN_MAGIC)
    (hlen,) = _BIN_HEADER.unpack_from(raw, off)
    off += _BIN_HEADER.size
    meta = json.loads(raw[off : off + hlen].decode("utf-8"))
    off += hlen

    placeholders: list[dict[str, Any]] = []

    def _find(obj: Any) -> None:
        if _is_ndarray_placeholder(obj):
            placeholders.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _find(v)
        elif isinstance(obj, list):
            for x in obj:
                _find(x)

    _find(meta)
    by_offset: dict[int, dict[str, Any]] = {}
    for ph in placeholders:
        by_offset[int(ph["offset"])] = ph
    arrays: list[np.ndarray] = []
    for idx in sorted(by_offset):
        ph = by_offset[idx]
        if idx != len(arrays):
            raise ValueError(f"non-contiguous array offsets: expected {len(arrays)} got {idx}")
        dtype = np.dtype(ph["dtype"])
        shape = tuple(ph["shape"])
        nbytes = int(np.prod(shape)) * int(dtype.itemsize)
        buf = raw[off : off + nbytes]
        if len(buf) != nbytes:
            raise ValueError(f"truncated array bytes for offset={idx}: need {nbytes} got {len(buf)}")
        arrays.append(np.frombuffer(buf, dtype=dtype).reshape(shape).copy())
        off += nbytes
    out = _restore_arrays(meta, arrays)
    if not isinstance(out, dict):
        raise TypeError(f"binary payload must be a dict, got {type(out)}")
    return out


def pack_predict_request(
    *,
    obs: Any,
    instruction: str,
    benchmark: str = "",
    kwargs: dict | None = None,
) -> bytes:
    return pack_binary(
        {
            "benchmark": benchmark,
            "instruction": instruction,
            "obs": obs,
            "kwargs": dict(kwargs or {}),
        }
    )


def unpack_predict_request(raw: bytes) -> dict[str, Any]:
    data = unpack_binary(raw)
    return {
        "benchmark": data.get("benchmark", ""),
        "instruction": str(data.get("instruction", "")),
        "obs": data.get("obs"),
        "kwargs": dict(data.get("kwargs") or {}),
    }


def pack_predict_response(chunk: Any) -> bytes:
    arr = np.ascontiguousarray(np.asarray(chunk, dtype=np.float32))
    return pack_binary({"chunk": arr})


def unpack_predict_response(raw: bytes) -> np.ndarray:
    data = unpack_binary(raw)
    chunk = data.get("chunk")
    if chunk is None:
        raise KeyError("binary response missing 'chunk'")
    return np.asarray(chunk, dtype=np.float32)
