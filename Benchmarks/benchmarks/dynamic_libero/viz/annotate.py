"""Frame annotation helpers (shared with integrations/libero_viz)."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(sz: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def annotate(frame_rgb, lines, ok=None):
    """Draw caption bar; optional SUCCESS/FAIL border."""
    im = Image.fromarray(frame_rgb).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    fnt = _font(max(14, W // 34))
    bar_h = int(fnt.size * (len(lines) + 0.6))
    draw.rectangle([0, 0, W, bar_h], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        draw.text((8, 4 + i * fnt.size), ln, fill=(255, 255, 255), font=fnt)
    if ok is not None:
        col = (40, 200, 60) if ok else (220, 50, 50)
        b = max(4, W // 90)
        draw.rectangle([0, 0, W - 1, H - 1], outline=col, width=b)
        tag = "SUCCESS" if ok else "FAIL"
        tf = _font(max(18, W // 20))
        tw = draw.textlength(tag, font=tf)
        draw.rectangle([W - tw - 20, H - tf.size - 14, W, H], fill=col)
        draw.text((W - tw - 12, H - tf.size - 10), tag, fill=(255, 255, 255), font=tf)
    return np.asarray(im)


def render_agent(env):
    o = env.env._get_observations(force_update=True)
    return np.ascontiguousarray(o["agentview_image"][::-1, ::-1])
