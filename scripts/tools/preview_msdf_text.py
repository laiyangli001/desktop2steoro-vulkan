"""Render MSDF text locally from the JSON atlas contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from project_paths import load_project_paths

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def median_alpha(rgb: np.ndarray) -> Image.Image:
    """Convert an RGB MSDF crop into a grayscale coverage mask."""
    distance = np.median(rgb.astype(np.float32) / 255.0, axis=2)
    coverage = np.clip((distance - 0.5) * 8.0 + 0.5, 0.0, 1.0)
    return Image.fromarray(np.asarray(coverage * 255.0, dtype=np.uint8), "L")


def render_preview(root: Path, text: str, output: Path, scale: int) -> None:
    metadata = json.loads(
        (root / "d2s_overlay_msdf.json").read_text(encoding="utf-8")
    )
    glyphs = {int(item["id"]): item for item in metadata["chars"]}
    pages = [Image.open(root / name).convert("RGB") for name in metadata["pages"]]
    canvas = Image.new("RGBA", (max(1000, len(text) * 80), 260), (18, 20, 26, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("consola.ttf", 14)
    except OSError:
        label_font = ImageFont.load_default()

    cursor_x = 30
    baseline = 95
    for character in text:
        if character == "\n":
            baseline += 90
            cursor_x = 30
            continue
        glyph = glyphs.get(ord(character)) or glyphs.get(ord("?"))
        if glyph is None:
            continue
        page = pages[int(glyph["page"])]
        x, y = int(glyph["x"]), int(glyph["y"])
        width, height = int(glyph["width"]), int(glyph["height"])
        crop = np.asarray(page.crop((x, y, x + width, y + height)))
        mask = median_alpha(crop).resize((width * scale, height * scale), Image.Resampling.LANCZOS)
        glyph_image = Image.new("RGBA", mask.size, (0, 220, 235, 255))
        glyph_image.putalpha(mask)
        dest_x = int(cursor_x + float(glyph["xoffset"]) * scale)
        dest_y = int(baseline + float(glyph["yoffset"]) * scale)
        canvas.alpha_composite(glyph_image, (dest_x, dest_y))
        draw.rectangle(
            (dest_x, dest_y, dest_x + width * scale, dest_y + height * scale),
            outline=(255, 120, 80, 180),
            width=1,
        )
        draw.text(
            (dest_x, dest_y + height * scale + 3),
            f"U+{ord(character):04X} p{glyph['page']} ({x},{y})",
            fill=(220, 220, 220, 255),
            font=label_font,
        )
        cursor_x += float(glyph["xadvance"]) * scale

    draw.text((30, 20), f"MSDF JSON preview: {text!r}", fill=(255, 255, 255, 255), font=label_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--text", default="Size 1920 x 1080 m  Dist 2.50 m  预设 性能")
    parser.add_argument("--output", type=Path, default=Path(".tmp/msdf_text_preview.png"))
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    paths = load_project_paths(REPO_ROOT)
    root = args.root or (paths.app_dir / "xr_viewer/fonts")
    render_preview(root, args.text, args.output, max(1, args.scale))


if __name__ == "__main__":
    main()
