"""Repack an MSDF atlas into a deterministic charset-order grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def repack(source: Path, output: Path, *, cell_size: int = 64) -> None:
    metadata_path = source.with_name("d2s_overlay_msdf.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pages = [source.with_name(name) for name in metadata["pages"]]
    images = [Image.open(path).convert("RGBA") for path in pages]
    source_width = int(metadata["common"]["scaleW"])
    source_height = int(metadata["common"]["scaleH"])
    if (source_width, source_height) not in ((2042, 2032), (2048, 2048)):
        raise ValueError("unexpected source atlas dimensions")
    page_width = 2048
    page_height = 2048
    if cell_size <= 0 or page_width // cell_size == 0:
        raise ValueError("invalid grid cell size")

    charset = [char for char in metadata["info"]["charset"] if ord(char) != 0xFEFF]
    by_codepoint = {
        int(item["id"]): item
        for item in metadata["chars"]
        if int(item["id"]) != 0xFEFF
    }
    ordered = [by_codepoint[ord(char)] for char in charset if ord(char) in by_codepoint]
    if len(ordered) != len(by_codepoint):
        raise ValueError("atlas glyphs do not match charset order")

    columns = page_width // cell_size
    rows = page_height // cell_size
    capacity = columns * rows
    page_count = (len(ordered) + capacity - 1) // capacity
    if page_count > 4:
        raise ValueError("ordered MSDF grid exceeds the native four-page capacity")

    output.mkdir(parents=True, exist_ok=True)
    destination_images = [
        Image.new("RGBA", (page_width, page_height), (0, 0, 0, 255))
        for _ in range(page_count)
    ]
    for order, glyph in enumerate(ordered):
        source_page = images[int(glyph["page"])]
        width = int(glyph["width"])
        height = int(glyph["height"])
        x = int(glyph["x"])
        y = int(glyph["y"])
        crop = source_page.crop((x, y, x + width, y + height))
        page_index, slot = divmod(order, capacity)
        row, col = divmod(slot, columns)
        cell_x = col * cell_size
        cell_y = row * cell_size
        paste_x = cell_x + (cell_size - width) // 2
        paste_y = cell_y + (cell_size - height) // 2
        destination_images[page_index].paste(crop, (paste_x, paste_y), crop)
        glyph["x"] = paste_x
        glyph["y"] = paste_y
        glyph["page"] = page_index
        glyph["index"] = order

    page_names = list(metadata["pages"])
    while len(page_names) < page_count:
        page_names.append(f"d2s_overlay_msdf.{len(page_names)}.png")
    for index, image in enumerate(destination_images):
        name = page_names[index]
        image.save(output / name, format="PNG")
    metadata["pages"] = page_names[:page_count]
    metadata["common"]["scaleW"] = page_width
    metadata["common"]["scaleH"] = page_height
    metadata["common"]["pages"] = page_count
    metadata["chars"] = ordered
    (output / metadata_path.name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-size", type=int, default=64)
    args = parser.parse_args()
    repack(args.source, args.output, cell_size=args.cell_size)


if __name__ == "__main__":
    main()
