#!/usr/bin/env python3
"""Generate toy-car mover MJCF assets that fit LIBERO object bounding boxes.

Reads each source object's ``bottom_site`` / ``top_site`` / ``horizontal_radius_site``
from the installed LIBERO asset pack and writes a parametric toy-car XML under
``dynamic_libero/assets/movers/<category>/<category>.xml``.

Sites are copied verbatim so placement samplers remain valid.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Demo toycar reference half-extents (collision box in alphabet_soup demo pack).
_REF_HALF = (0.042, 0.028, 0.018)

COLOR_PRESETS: dict[str, dict[str, str]] = {
    "default": {
        "car_body": "0.85 0.12 0.10 1",
        "car_cabin": "0.15 0.35 0.75 1",
        "car_wheel": "0.08 0.08 0.08 1",
    },
    "white": {
        "car_body": "0.92 0.92 0.94 1",
        "car_cabin": "0.75 0.78 0.82 1",
        "car_wheel": "0.08 0.08 0.08 1",
    },
    "yellow_white": {
        "car_body": "0.95 0.82 0.15 1",
        "car_cabin": "0.92 0.92 0.94 1",
        "car_wheel": "0.08 0.08 0.08 1",
    },
}

# Categories swapped in Dynamic-LIBERO v2 (unique asset types).
MOVER_CATEGORIES: dict[str, str] = {
    # stable_hope_objects
    "alphabet_soup": "stable_hope_objects",
    "bbq_sauce": "stable_hope_objects",
    "butter": "stable_hope_objects",
    "chocolate_pudding": "stable_hope_objects",
    "cream_cheese": "stable_hope_objects",
    "ketchup": "stable_hope_objects",
    "milk": "stable_hope_objects",
    "orange_juice": "stable_hope_objects",
    "salad_dressing": "stable_hope_objects",
    "tomato_sauce": "stable_hope_objects",
    # stable_scanned_objects
    "akita_black_bowl": "stable_scanned_objects",
    "plate": "stable_scanned_objects",
    # turbosquid_objects
    "black_book": "turbosquid_objects",
    "moka_pot": "turbosquid_objects",
    "porcelain_mug": "turbosquid_objects",
    "white_yellow_mug": "turbosquid_objects",
    "wine_bottle": "turbosquid_objects",
}

# Optional colour variant per category (defaults to ``default``).
CATEGORY_COLOR: dict[str, str] = {
    "porcelain_mug": "white",
    "white_yellow_mug": "yellow_white",
}


@dataclass(frozen=True)
class SiteBox:
    bottom_z: float
    top_z: float
    hr_x: float
    hr_y: float

    @property
    def half_x(self) -> float:
        return max(abs(self.hr_x), 0.015)

    @property
    def half_y(self) -> float:
        return max(abs(self.hr_y), 0.012)

    @property
    def half_z(self) -> float:
        return max((self.top_z - self.bottom_z) * 0.5, 0.012)


def _libero_assets_root() -> Path:
    try:
        import libero.libero.envs.objects.hope_objects as hope

        return Path(hope.absolute_path) / "assets"
    except Exception:
        pass
    env = os.environ.get("LIBERO_ROOT")
    if env:
        p = Path(env) / "libero" / "libero" / "assets"
        if p.is_dir():
            return p
    fallback = Path("/DATA/YuanZhen/LIBERO/libero/libero/assets")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError("Cannot locate LIBERO assets root")


def _parse_sites(xml_path: Path) -> SiteBox:
    root = ET.parse(xml_path).getroot()
    sites = {}
    for site in root.iter("site"):
        name = site.get("name")
        if name in ("bottom_site", "top_site", "horizontal_radius_site"):
            pos = [float(x) for x in site.get("pos", "0 0 0").split()]
            sites[name] = pos
    missing = {"bottom_site", "top_site", "horizontal_radius_site"} - set(sites)
    if missing:
        raise ValueError(f"{xml_path}: missing sites {missing}")
    return SiteBox(
        bottom_z=float(sites["bottom_site"][2]),
        top_z=float(sites["top_site"][2]),
        hr_x=float(sites["horizontal_radius_site"][0]),
        hr_y=float(sites["horizontal_radius_site"][1]),
    )


def _fmt_vec3(x: float, y: float, z: float) -> str:
    return f"{x:.5f} {y:.5f} {z:.5f}"


def _scale_geom(base_size: tuple[float, float, float], base_pos: tuple[float, float, float], sx: float, sy: float, sz: float):
    return (
        base_size[0] * sx,
        base_size[1] * sy,
        base_size[2] * sz,
    ), (
        base_pos[0] * sx,
        base_pos[1] * sy,
        base_pos[2] * sz,
    )


def render_toycar_xml(category: str, box: SiteBox, color_key: str = "default") -> str:
    colors = COLOR_PRESETS[color_key]
    sx = min(box.half_x / _REF_HALF[0], 1.25)
    sy = min(box.half_y / _REF_HALF[1], 1.25)
    sz = min(box.half_z / _REF_HALF[2], 1.25)
    sx = sy = sz = min(sx, sy, sz)  # uniform scale to stay inside AABB

    def g(name, gtype, size, pos, euler=None, mat="car_body", group="1"):
        if gtype == "cylinder":
            size_s = f"{size[0]:.5f} {size[1]:.5f}"
        else:
            size_s = _fmt_vec3(*size)
        pos_s = _fmt_vec3(*pos)
        euler_attr = f' euler="{euler}"' if euler else ""
        return (
            f'        <geom name="{name}" type="{gtype}" size="{size_s}" pos="{pos_s}"{euler_attr}\n'
            f'              material="{mat}" group="{group}" contype="0" conaffinity="0"\n'
            f'              solimp="0.998 0.998 0.001" solref="0.001 1" density="100" friction="0.95 0.3 0.1"/>'
        )

    # Reference primitive sizes from demo_toycar/alphabet_soup.xml
    parts = [
        g("chassis", "box", *_scale_geom((0.040, 0.022, 0.010), (0, 0, 0.002), sx, sy, sz), mat="car_body"),
        g("cabin", "box", *_scale_geom((0.018, 0.018, 0.010), (-0.005, 0, 0.020), sx, sy, sz), mat="car_cabin"),
        g("hood", "box", *_scale_geom((0.012, 0.020, 0.006), (0.022, 0, 0.012), sx, sy, sz), mat="car_body"),
    ]
    wheel_r, wheel_half = 0.008 * sx, 0.004 * sy
    for wname, wpos in (
        ("w_fl", (0.022, 0.024, -0.004)),
        ("w_fr", (0.022, -0.024, -0.004)),
        ("w_rl", (-0.022, 0.024, -0.004)),
        ("w_rr", (-0.022, -0.024, -0.004)),
    ):
        _, pos = _scale_geom((wheel_r, wheel_r, wheel_half), wpos, sx, sy, sz)
        parts.append(
            g(wname, "cylinder", (wheel_r, wheel_half), pos, euler="1.5708 0 0", mat="car_wheel")
        )

    col_size, col_pos = _scale_geom((0.042, 0.028, 0.018), (0, 0, 0.005), sx, sy, sz)
    col_size_s = _fmt_vec3(*col_size)
    col_pos_s = _fmt_vec3(*col_pos)
    collision = (
        f'        <geom type="box" size="{col_size_s}" pos="{col_pos_s}" group="0"\n'
        f'              rgba="0.8 0.2 0.2 0.0" solimp="0.998 0.998 0.001" solref="0.001 1"\n'
        f'              density="100" friction="0.95 0.3 0.1"/>'
    )

    root_sites = ""
    # Preserve original site positions exactly.
    for sname, z_or_pos in (
        ("bottom_site", (0, 0, box.bottom_z)),
        ("top_site", (0, 0, box.top_z)),
        ("horizontal_radius_site", (box.hr_x, box.hr_y, 0)),
    ):
        root_sites += (
            f'      <site rgba="0 0 0 0" size="0.005" pos="{_fmt_vec3(*z_or_pos)}" name="{sname}" />\n'
        )

    mats = "\n".join(
        f'    <material name="{k}" rgba="{v}" specular="0.4" reflectance="0.3"/>'
        for k, v in colors.items()
    )
    body = "\n".join(parts)
    return f"""<mujoco model="{category}_mover">
  <compiler angle="radian"/>
  <asset>
{mats}
  </asset>
  <worldbody>
    <body>
      <body name="object">
{body}
{collision}
      </body>
{root_sites}    </body>
  </worldbody>
</mujoco>
"""


def generate_one(category: str, assets_root: Path, out_root: Path) -> Path:
    subdir = MOVER_CATEGORIES[category]
    src = assets_root / subdir / category / f"{category}.xml"
    if not src.is_file():
        raise FileNotFoundError(src)
    box = _parse_sites(src)
    color_key = CATEGORY_COLOR.get(category, "default")
    xml = render_toycar_xml(category, box, color_key=color_key)
    out_dir = out_root / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{category}.xml"
    out_path.write_text(xml)
    return out_path


def generate_all(categories: Iterable[str] | None = None) -> list[Path]:
    assets_root = _libero_assets_root()
    out_root = Path(__file__).resolve().parents[1] / "assets" / "movers"
    cats = list(categories or MOVER_CATEGORIES.keys())
    written: list[Path] = []
    for cat in cats:
        written.append(generate_one(cat, assets_root, out_root))
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LIBERO toy-car mover MJCF assets")
    p.add_argument(
        "--categories",
        type=str,
        default="all",
        help="Comma-separated category names or 'all'",
    )
    args = p.parse_args()
    if args.categories.strip().lower() == "all":
        cats = None
    else:
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    paths = generate_all(cats)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
