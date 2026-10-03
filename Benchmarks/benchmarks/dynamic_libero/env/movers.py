"""Dynamic-LIBERO v2: swap rails targets with plausible self-moving toy-car assets.

Patches ``OBJECTS_DICT`` in-process (no LIBERO install mutation) and provides a
per-task ``MoverSpec`` for deterministic target selection + language rewrites.
"""

from __future__ import annotations

import copy
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "movers"

# Phrase replacements for language rewrite (longest first within each category).
_CATEGORY_PHRASES: dict[str, tuple[str, ...]] = {
    "akita_black_bowl": ("black bowl", "bowl"),
    "alphabet_soup": ("alphabet soup",),
    "bbq_sauce": ("bbq sauce",),
    "butter": ("butter",),
    "chocolate_pudding": ("chocolate pudding",),
    "cream_cheese": ("cream cheese box", "cream cheese"),
    "ketchup": ("ketchup",),
    "milk": ("milk",),
    "orange_juice": ("orange juice",),
    "salad_dressing": ("salad dressing",),
    "tomato_sauce": ("tomato sauce",),
    "plate": ("plate",),
    "wine_bottle": ("wine bottle",),
    "moka_pot": ("moka pot", "moka pots"),
    "black_book": ("book",),
    "porcelain_mug": ("white mug", "porcelain mug"),
    "white_yellow_mug": ("yellow and white mug",),
}


@dataclass(frozen=True)
class MoverSpec:
    """Per-task rails target and visual swap metadata."""

    target: str  # instance name, e.g. alphabet_soup_1
    category: str  # BDDL object category swapped to toy-car MJCF
    noun: str  # replacement noun for language rewrite
    motion: str = "drive"  # drive | roll
    height_offset: float = 0.0  # baked at begin_policy; 0 avoids start z-pop
    roll_radius: float = 0.025  # used when motion=="roll"


def _spec(
    target: str,
    category: str,
    noun: str = "toy car",
    *,
    motion: str = "drive",
    height_offset: float = 0.0,
    roll_radius: float = 0.025,
) -> MoverSpec:
    return MoverSpec(
        target=target,
        category=category,
        noun=noun,
        motion=motion,
        height_offset=height_offset,
        roll_radius=roll_radius,
    )


# fmt: off
MOVERS: dict[str, dict[int, MoverSpec]] = {
    "libero_spatial": {
        i: _spec("akita_black_bowl_1", "akita_black_bowl")
        for i in range(10)
    },
    "libero_object": {
        0: _spec("alphabet_soup_1", "alphabet_soup"),
        1: _spec("cream_cheese_1", "cream_cheese"),
        2: _spec("salad_dressing_1", "salad_dressing"),
        3: _spec("bbq_sauce_1", "bbq_sauce"),
        4: _spec("ketchup_1", "ketchup"),
        5: _spec("tomato_sauce_1", "tomato_sauce"),
        6: _spec("butter_1", "butter"),
        7: _spec("milk_1", "milk"),
        8: _spec("chocolate_pudding_1", "chocolate_pudding"),
        9: _spec("orange_juice_1", "orange_juice"),
    },
    "libero_goal": {
        0: _spec("akita_black_bowl_1", "akita_black_bowl"),  # drawer task: disturbance car
        1: _spec("akita_black_bowl_1", "akita_black_bowl"),
        2: _spec("wine_bottle_1", "wine_bottle"),
        3: _spec("akita_black_bowl_1", "akita_black_bowl"),
        4: _spec("akita_black_bowl_1", "akita_black_bowl"),
        5: _spec("plate_1", "plate"),
        6: _spec("cream_cheese_1", "cream_cheese"),
        7: _spec("akita_black_bowl_1", "akita_black_bowl"),  # stove task: disturbance car
        8: _spec("akita_black_bowl_1", "akita_black_bowl"),
        9: _spec("wine_bottle_1", "wine_bottle"),
    },
    "libero_10": {
        0: _spec("alphabet_soup_1", "alphabet_soup"),
        1: _spec("cream_cheese_1", "cream_cheese"),
        2: _spec("moka_pot_1", "moka_pot"),
        3: _spec("akita_black_bowl_1", "akita_black_bowl"),
        4: _spec("porcelain_mug_1", "porcelain_mug", noun="white toy car"),
        5: _spec("black_book_1", "black_book"),
        6: _spec("porcelain_mug_1", "porcelain_mug", noun="white toy car"),
        7: _spec("alphabet_soup_1", "alphabet_soup"),
        8: _spec("moka_pot_1", "moka_pot"),
        9: _spec("white_yellow_mug_1", "white_yellow_mug", noun="yellow and white toy car"),
    },
}
# fmt: on


def get_mover_spec(suite: str, task_id: int) -> MoverSpec:
    suite = suite.strip()
    if suite not in MOVERS:
        raise KeyError(f"No mover table for suite {suite!r}")
    if task_id not in MOVERS[suite]:
        raise KeyError(f"No mover spec for {suite} task_id={task_id}")
    return MOVERS[suite][task_id]


def all_mover_specs() -> list[tuple[str, int, MoverSpec]]:
    out: list[tuple[str, int, MoverSpec]] = []
    for suite, tasks in MOVERS.items():
        for tid, spec in sorted(tasks.items()):
            out.append((suite, tid, spec))
    return out


def rewrite_language(description: str, spec: MoverSpec) -> str:
    """Replace original object phrases with ``spec.noun`` (case-insensitive)."""
    text = description
    phrases = _CATEGORY_PHRASES.get(spec.category, (spec.category.replace("_", " "),))
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(spec.noun, text)
    return text


def mover_xml_path(category: str) -> Path:
    path = _ASSETS / category / f"{category}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Mover MJCF missing: {path}")
    return path


def _make_mover_class(category: str, xml_path: Path, prototype):
    from robosuite.models.objects import MujocoXMLObject

    class _MoverObject(MujocoXMLObject):
        def __init__(self, name, joints=None, obj_type="all", duplicate_collision_geoms=False):
            super().__init__(
                str(xml_path),
                name=name,
                joints=joints if joints is not None else [dict(type="free", damping="0.0005")],
                obj_type=obj_type,
                duplicate_collision_geoms=duplicate_collision_geoms,
            )
            self.category_name = prototype.category_name
            self.rotation = copy.deepcopy(prototype.rotation)
            self.rotation_axis = copy.deepcopy(prototype.rotation_axis)
            self.object_properties = copy.deepcopy(prototype.object_properties)

    _MoverObject.__name__ = f"Mover{category.title().replace('_', '')}"
    return _MoverObject


_PROTOTYPES: dict[str, object] = {}


def _prototype(category: str):
    if category not in _PROTOTYPES:
        from libero.libero.envs.base_object import OBJECTS_DICT

        cls = OBJECTS_DICT[category.lower()]
        _PROTOTYPES[category] = cls(name=f"__mover_proto_{category}__")
    return _PROTOTYPES[category]


@contextmanager
def mover_assets(
    suite: str,
    task_id: int,
    *,
    enabled: bool = True,
) -> Iterator[Optional[MoverSpec]]:
    """Optionally patch ``OBJECTS_DICT``; always resolves the task ``MoverSpec``.

    When ``enabled=False``, yields ``None`` (caller should use ``get_mover_spec``
    separately if only the target registry is needed). When ``enabled=True``,
    patches the category class and yields the spec.
    """
    if not enabled:
        yield None
        return

    spec = get_mover_spec(suite, task_id)
    from libero.libero.envs.base_object import OBJECTS_DICT

    category = spec.category.lower()
    if category not in OBJECTS_DICT:
        raise KeyError(f"Unknown LIBERO object category {category!r}")

    orig_cls = OBJECTS_DICT[category]
    xml_path = mover_xml_path(category)
    proto = _prototype(category)
    patched_cls = _make_mover_class(category, xml_path, proto)
    OBJECTS_DICT[category] = patched_cls
    try:
        yield spec
    finally:
        OBJECTS_DICT[category] = orig_cls


def resolve_rails_spec(
    suite: str,
    task_id: int,
    *,
    movers_enabled: bool,
    target_registry: bool = True,
) -> Optional[MoverSpec]:
    """Spec passed to ``RealtimeDriver``.

    Official runs keep ``target_registry=True`` so rails targets stay
    deterministic even when asset swap is off.
    """
    if movers_enabled or target_registry:
        return get_mover_spec(suite, task_id)
    return None


def parse_movers_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        # Official primary track default: original appearance.
        return False
    v = str(value).strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"movers must be on/off, got {value!r}")


def parse_mover_language(value: str | None) -> str:
    v = (value or "keep").strip().lower()
    if v not in ("rewrite", "keep"):
        raise ValueError(f"mover-language must be rewrite/keep, got {value!r}")
    return v
