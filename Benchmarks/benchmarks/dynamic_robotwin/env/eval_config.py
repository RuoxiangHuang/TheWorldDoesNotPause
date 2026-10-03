"""Load Dynamic-RoboTwin YAML defaults so CLI and configs/default.yaml stay aligned."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def default_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_yaml_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else default_yaml_path()
    if not p.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def dynamics_defaults_from_yaml(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Return dynamics kwargs used by argparse defaults / rollout resolution."""
    cfg = load_yaml_config(path)
    dyn = dict(cfg.get("dynamics") or {})
    modes = dyn.get("modes", ["contact_trigger", "occlusion", "contact_chain"])
    if isinstance(modes, str):
        modes_str = modes
    else:
        modes_str = ",".join(str(m) for m in modes)
    secondary = dyn.get("secondary_rails", "auto")
    if isinstance(secondary, bool):
        secondary = "on" if secondary else "off"
    chain = dyn.get("contact_chain", True)
    if isinstance(chain, bool):
        chain = "on" if chain else "off"
    occ = dyn.get("occlusion", True)
    if isinstance(occ, bool):
        occ = "on" if occ else "off"
    return {
        "modes": modes_str,
        "contact_switch": str(dyn.get("contact_switch", "medium")),
        "occlusion": str(occ),
        "occlusion_duty": float(dyn.get("occlusion_duty", 0.35)),
        "occlusion_period_ticks": float(dyn.get("occlusion_period_ticks", 40)),
        "contact_chain": str(chain),
        "contact_chain_n_aux": int(dyn.get("contact_chain_n_aux", 3)),
        "contact_chain_impulse": float(dyn.get("contact_chain_impulse", 0.35)),
        "secondary_rails": str(secondary),
    }


def add_dynamics_cli_args(parser, *, defaults: Optional[Dict[str, Any]] = None) -> None:
    """Attach shared Dynamic-RoboTwin CLI flags; defaults come from YAML."""
    d = defaults if defaults is not None else dynamics_defaults_from_yaml()
    parser.add_argument(
        "--dynamic-modes",
        type=str,
        default=d["modes"],
        help="Comma-separated modes, or 'none'/'all' (default from configs/default.yaml)",
    )
    parser.add_argument(
        "--contact-trigger",
        type=str,
        default=d["contact_switch"],
        choices=["off", "mild", "medium", "strong"],
        help="Contact-triggered trajectory switch intensity",
    )
    parser.add_argument(
        "--occlusion",
        type=str,
        default=d["occlusion"],
        choices=["on", "off"],
    )
    parser.add_argument("--occlusion-duty", type=float, default=d["occlusion_duty"])
    parser.add_argument(
        "--contact-chain",
        type=str,
        default=d["contact_chain"],
        choices=["on", "off"],
        help="Multi-object contact-chain propagation",
    )
    parser.add_argument(
        "--secondary-rails",
        type=str,
        default=d["secondary_rails"],
        choices=["auto", "on", "off"],
        help="Rail receive-side placement target; 'auto' enables for handoff+secondary_attr",
    )
