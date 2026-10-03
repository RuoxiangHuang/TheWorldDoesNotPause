"""Shared language modes for Real-Time Dynamic benchmarks.

Tracks
------
- ``keep`` (primary): training-aligned instruction; motion is unannounced.
- ``motion`` (B track): append a light motion suffix; success goal unchanged.
- ``scene``: rewrite *stale locatives* to match the live scene. Spatial
  carrier becomes "bowl on the moving platform"; the place clause (BDDL) is
  unchanged. Object-suite soup→basket strings are left alone.

Appearance noun rewrite (LIBERO toy-car) stays separate (``mover-language``).
"""

from __future__ import annotations

from typing import Optional

LANGUAGE_MODES = ("keep", "motion", "scene")

# Unified B-track suffix. ``{noun}`` is the visible pick target (not the full goal).
DEFAULT_MOTION_HINT_TEMPLATE = (
    "The {noun} is moving; grasp it before it leaves reach."
)

# Spatial carrier: locatives like "between the plate and the ramekin" / "in the
# top drawer" are false after the bowl is seated on the flatbed. Goal clause
# stays original LIBERO ("place it on the plate").
SPATIAL_CARRIER_SCENE_INSTRUCTION = (
    "pick up the black bowl on the moving platform and place it on the plate"
)


def parse_language_mode(value: str | None) -> str:
    v = (value or "keep").strip().lower()
    if v not in LANGUAGE_MODES:
        raise ValueError(f"language-mode must be one of {LANGUAGE_MODES}, got {value!r}")
    return v


def humanize_noun(raw: str) -> str:
    """``alphabet_soup`` / ``AlphabetSoup`` → ``alphabet soup``."""
    s = str(raw).strip().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s.lower() if s else "target"


def motion_hint(*, noun: str | None = None, template: str = DEFAULT_MOTION_HINT_TEMPLATE) -> str:
    n = humanize_noun(noun) if noun else "target"
    return template.format(noun=n)


def append_motion_hint(
    description: str,
    *,
    noun: str | None = None,
    template: str = DEFAULT_MOTION_HINT_TEMPLATE,
) -> str:
    """Append B-track motion suffix; does not rewrite the original goal clause."""
    text = (description or "").rstrip()
    hint = motion_hint(noun=noun, template=template)
    if not text:
        return hint
    # Avoid double-appending if caller already applied B-track.
    if hint.lower() in text.lower():
        return text
    if text.endswith((".", "!", "?")):
        return f"{text} {hint}"
    return f"{text}. {hint}"


def is_spatial_bowl_on_plate(text: str) -> bool:
    t = (text or "").lower()
    return "black bowl" in t and "place it on the plate" in t


def apply_scene_language(
    description: str,
    *,
    motive: str | None = None,
) -> str:
    """Replace outdated spatial locatives; do not change the BDDL goal clause."""
    text = description or ""
    if str(motive or "").strip().lower() == "carrier" or is_spatial_bowl_on_plate(text):
        return SPATIAL_CARRIER_SCENE_INSTRUCTION
    return text


def apply_language_mode(
    description: str,
    mode: str | None = "keep",
    *,
    noun: str | None = None,
    template: str = DEFAULT_MOTION_HINT_TEMPLATE,
    motive: str | None = None,
) -> str:
    """Apply ``keep`` / ``motion`` / ``scene`` to an instruction string."""
    m = parse_language_mode(mode)
    text = description or ""
    if m == "motion":
        return append_motion_hint(text, noun=noun, template=template)
    if m == "scene":
        return apply_scene_language(text, motive=motive)
    return text


def add_language_mode_arg(parser, *, default: str = "keep") -> None:
    parser.add_argument(
        "--language-mode",
        type=str,
        default=default,
        choices=list(LANGUAGE_MODES),
        help="keep=training instruction; motion=B-track suffix; "
        "scene=rewrite stale spatial locatives (carrier platform).",
    )
