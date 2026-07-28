"""Load the offline MSDF atlas and build lightweight glyph instances."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MsdfGlyph:
    codepoint: int
    page: int
    x: float
    y: float
    width: float
    height: float
    xoffset: float
    yoffset: float
    xadvance: float


@dataclass(frozen=True)
class MsdfGlyphInstance:
    page: int
    position: tuple[float, float]
    size: tuple[float, float]
    uv_min: tuple[float, float]
    uv_max: tuple[float, float]


class MsdfFontAtlas:
    """Immutable atlas metadata shared by all GPU text overlays."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parent / "fonts"
        metadata_path = self.root / "d2s_overlay_msdf.json"
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

        common = metadata.get("common") or {}
        self.page_width = int(common.get("scaleW", 0))
        self.page_height = int(common.get("scaleH", 0))
        self.line_height = float(common.get("lineHeight", 0.0))
        self.pages = tuple(str(page) for page in metadata.get("pages", ()))
        if self.page_width <= 0 or self.page_height <= 0 or not self.pages:
            raise ValueError("MSDF atlas metadata has invalid page dimensions")
        if any(not (self.root / page).is_file() for page in self.pages):
            raise FileNotFoundError("MSDF atlas page is missing")

        glyphs: dict[int, MsdfGlyph] = {}
        for item in metadata.get("chars", ()):
            codepoint = int(item["id"])
            if codepoint in glyphs:
                raise ValueError(f"duplicate MSDF glyph codepoint: {codepoint}")
            glyphs[codepoint] = MsdfGlyph(
                codepoint=codepoint,
                page=int(item.get("page", 0)),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
                xoffset=float(item.get("xoffset", 0.0)),
                yoffset=float(item.get("yoffset", 0.0)),
                xadvance=float(item.get("xadvance", item["width"])),
            )
        if not glyphs:
            raise ValueError("MSDF atlas contains no glyphs")
        self.glyphs = glyphs
        self.fallback = glyphs.get(ord("?")) or next(iter(glyphs.values()))

    def layout(
        self,
        text: str,
        *,
        origin: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        line_gap: float = 0.0,
    ) -> tuple[MsdfGlyphInstance, ...]:
        """Return glyph quads; no rasterization or texture allocation occurs."""
        cursor_x, cursor_y = float(origin[0]), float(origin[1])
        instances: list[MsdfGlyphInstance] = []
        line_step = (self.line_height + float(line_gap)) * float(scale)
        for character in text:
            if character == "\n":
                cursor_x = float(origin[0])
                cursor_y -= line_step
                continue
            glyph = self.glyphs.get(ord(character), self.fallback)
            x = cursor_x + glyph.xoffset * scale
            y = cursor_y + glyph.yoffset * scale
            width = glyph.width * scale
            height = glyph.height * scale
            instances.append(
                MsdfGlyphInstance(
                    page=glyph.page,
                    position=(x, y),
                    size=(width, height),
                    uv_min=(glyph.x / self.page_width, glyph.y / self.page_height),
                    uv_max=(
                        (glyph.x + glyph.width) / self.page_width,
                        (glyph.y + glyph.height) / self.page_height,
                    ),
                )
            )
            cursor_x += glyph.xadvance * scale
        return tuple(instances)

    def page_path(self, page: int) -> Path:
        if page < 0 or page >= len(self.pages):
            raise IndexError(f"MSDF atlas page out of range: {page}")
        return self.root / self.pages[page]

    def page_rgba(self, page: int) -> np.ndarray:
        """Decode one atlas page once before handing it to the GPU Bridge."""
        from PIL import Image

        with Image.open(self.page_path(page)) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()

    def text_advance(self, text: str, *, scale: float = 1.0) -> float:
        """Return the atlas-layout advance used to center a text run."""
        return sum(
            (self.glyphs.get(ord(character), self.fallback).xadvance * float(scale))
            for character in text
            if character != "\n"
        )

    def build_geometry(
        self,
        text: str,
        *,
        transform: np.ndarray,
        pixel_scale: tuple[float, float],
        origin: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        line_gap: float = 0.0,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Pack glyph quads into world-space native Bridge buffers.

        The transform is column-major 4x4. Layout coordinates use the atlas
        top-left convention; the Y axis is inverted for the Filament plane.
        """
        matrix = np.asarray(transform, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError("MSDF text transform must be a 4x4 matrix")
        sx, sy = float(pixel_scale[0]), float(pixel_scale[1])
        if sx <= 0.0 or sy <= 0.0:
            raise ValueError("MSDF pixel scale must be positive")
        instances = self.layout(
            text, origin=origin, scale=scale, line_gap=line_gap
        )
        grouped: dict[int, list[MsdfGlyphInstance]] = {}
        for instance in instances:
            grouped.setdefault(instance.page, []).append(instance)
        rgba = np.asarray(color, dtype=np.float32)
        if rgba.shape != (4,):
            raise ValueError("MSDF text color must contain four components")
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for page, page_instances in grouped.items():
            vertices = np.zeros((len(page_instances) * 4, 9), dtype=np.float32)
            indices = np.zeros(len(page_instances) * 6, dtype=np.uint16)
            for glyph_index, instance in enumerate(page_instances):
                x, y = instance.position
                width, height = instance.size
                local = np.asarray(
                    (
                        (x * sx, -y * sy, 0.0, 1.0),
                        ((x + width) * sx, -y * sy, 0.0, 1.0),
                        (x * sx, -(y + height) * sy, 0.0, 1.0),
                        ((x + width) * sx, -(y + height) * sy, 0.0, 1.0),
                    ),
                    dtype=np.float32,
                )
                base = glyph_index * 4
                vertices[base : base + 4, :3] = (matrix @ local.T).T[:, :3]
                vertices[base : base + 4, 3:5] = (
                    (instance.uv_min[0], instance.uv_min[1]),
                    (instance.uv_max[0], instance.uv_min[1]),
                    (instance.uv_min[0], instance.uv_max[1]),
                    (instance.uv_max[0], instance.uv_max[1]),
                )
                vertices[base : base + 4, 5:9] = rgba
                offset = glyph_index * 6
                indices[offset : offset + 6] = (
                    base, base + 1, base + 2,
                    base + 1, base + 3, base + 2,
                )
            result[page] = (vertices, indices)
        return result


def load_msdf_font_atlas(root: str | Path | None = None) -> MsdfFontAtlas:
    return MsdfFontAtlas(Path(root) if root is not None else None)
