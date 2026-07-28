# GPU MSDF Text Rendering

## Asset contract

The OpenXR FPS panel, operation guide, and screen adjustment OSD use one
shared MSDF atlas. The source charset is `src/xr_viewer/fonts/overlay_charset.txt`,
copied from `E:\VAM2\UI字符中文字库\3500.txt` and currently contains 3,958
unique characters. The atlas is generated offline; the runtime
does not load a system font and does not rasterize text with Pillow on the
presenter thread.

The source font must be a distributable CJK OpenType font with glyph coverage
for this charset. The generator intentionally fails when the font file is
missing or invalid; it must not silently fall back to a partial system font.

The generated files are:

- `src/xr_viewer/fonts/d2s_overlay_msdf.*.png`: RGB MSDF atlas pages
- `src/xr_viewer/fonts/d2s_overlay_msdf.json`: glyph metrics and page layout

The checked-in atlas is repacked by `scripts/tools/repack_msdf_grid.py` after
MSDF generation. It uses fixed 64x64 cells and the source charset order from
`overlay_charset.txt`, filling each page left-to-right and top-to-bottom. This
is intentionally different from MaxRects size optimization: the atlas is
human-inspectable and deterministic, while runtime lookup still uses Unicode
glyph IDs. The RGB color pattern is MSDF distance data and is not intended to
look like normal black-and-white text.

The texture is sampled as linear data. The fragment shader reconstructs the
glyph coverage from the median of the RGB channels, then applies the overlay
color and alpha. Text layout and glyph instance updates remain lightweight
metadata updates; the atlas is uploaded once and shared by all overlays.
The JSON atlas uses top-left image coordinates; the Python geometry adapter
flips only the V coordinate at the Filament boundary, where V=0 is bottom.

Use the local JSON-coordinate preview before testing OpenXR:

```powershell
$env:PYTHONPATH = "src"
& .\src\python3\python.exe scripts/tools/preview_msdf_text.py `
  --output .tmp/msdf_text_preview.png
```

The preview reconstructs glyph coverage from JSON coordinates and labels each
glyph with its Unicode codepoint, page, and atlas position. It separates
charset/UV metadata problems from Filament shader and Vulkan presentation
problems.

## Reproducible generation

Install the pinned generator once from the repository root:

```powershell
npm install --no-audit --no-fund msdf-bmfont-xml@2.8.0 --prefix .tools/msdf
```

Generate the atlas:

```powershell
./scripts/tools/generate_msdf_overlay_font.ps1

python scripts/tools/repack_msdf_grid.py \
  src/xr_viewer/fonts/d2s_overlay_msdf.0.png \
  src/xr_viewer/fonts/grid-repacked
```

The production font must be distributed under a license compatible with the
application. The Windows system font is not a production dependency and is
only a fallback for the legacy Pillow path.

## Runtime migration

The native Bridge will expose a stable C ABI for one shared atlas and dynamic
glyph quads. Python retains the text content, layout, visibility, and update
policy. The Presenter thread owns all Bridge calls. Until the rebuilt Bridge
reports the MSDF ABI, the existing Quad Layer bitmap path remains the explicit
fallback.

The first native ABI uses one GPU texture and fixed-capacity geometry per atlas
page. Python submits packed world-space glyph vertices containing position,
UV, and RGBA color; Filament reconstructs coverage from the MSDF median and
`fwidth` in the fragment shader. Atlas pages use linear `RGBA8`, because MSDF
channels contain distance data rather than display colors. A Bridge binary
without these optional symbols continues to use the legacy Quad Layer path.

For small headset overlays, the shader keeps the derivative-based MSDF
transition neutral: the minimum screen range is 1.0 pixel and the
edge-sharpness factor is 1.0. The OSD layout scale remains 0.5, matching the
legacy bitmap path. This avoids enlarging glyphs or merging small CJK counters
when the text is rasterized at the actual OpenXR projection resolution.

When the optional ABI is present, the OpenXR Presenter decodes and uploads all
atlas pages once during Filament initialization. Text content and pose updates
remain separate from this upload; this prevents page decoding or texture
allocation from entering the per-frame render loop.

The screen size/distance and preset OSD now use the first mixed path: the
cached Quad Layer contains only the rounded background, while its colored text
is submitted as MSDF glyph geometry. FPS, keyboard, controller help, and laser
cursor overlays remain on the legacy path until their individual background
and interaction update policies are migrated.
