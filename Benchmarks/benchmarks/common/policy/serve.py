#!/usr/bin/env python3
"""Launch an HTTP policy server for Real-Time benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys
from pathlib import Path
for _anc in Path(__file__).resolve().parents:
    _b = _anc / "Benchmarks"
    if (_b / "bootstrap_paths.py").is_file():
        sys.path.insert(0, str(_b))
        break
from bootstrap_paths import setup_sys_path

ROOT = setup_sys_path()

from benchmarks.common.policy.http_server import create_policy_server, load_server_policy


def main() -> None:
    p = argparse.ArgumentParser(description="Real-Time policy HTTP server")
    p.add_argument("--benchmark", choices=["libero", "robotwin", "robocasa"], required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--stats", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--accel", type=str, default="off")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-inference-steps", type=int, default=4)
    # libero
    p.add_argument("--task-suite", type=str, default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-name", type=str, default="PickPlaceCounterToCabinet", help="RoboCasa env class name")
    p.add_argument(
        "--allow-libero-shim",
        action="store_true",
        help="RoboCasa only: allow LIBERO 7-D→12-D pad (policy_compatible=false)",
    )
    p.add_argument(
        "--action-layout",
        type=str,
        default=None,
        choices=["native_12d", "libero_pad"],
        help="RoboCasa action layout override",
    )
    args = p.parse_args()

    load_kwargs: dict = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "accel": args.accel,
        "seed": args.seed,
    }
    if args.stats:
        load_kwargs["stats"] = args.stats
    if args.task:
        load_kwargs["task"] = args.task
    if args.benchmark == "libero":
        load_kwargs["task_suite_name"] = args.task_suite
        load_kwargs["task_id"] = args.task_id
    if args.benchmark == "robocasa":
        load_kwargs["task_name"] = args.task_name
        load_kwargs["allow_libero_shim"] = bool(args.allow_libero_shim)
        if args.action_layout:
            load_kwargs["action_layout"] = args.action_layout

    policy = load_server_policy(args.benchmark, **load_kwargs)
    httpd = create_policy_server(policy, host=args.host, port=args.port)
    host, port = httpd.server_address
    print(f"Serving {args.benchmark} policy on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        policy.close()
        httpd.shutdown()


if __name__ == "__main__":
    main()
