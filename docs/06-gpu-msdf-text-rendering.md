# GPU MSDF Text Rendering

## Asset contract

The MSDF text path uses one shared MSDF atlas. The source charset is
`src/desktop2stereo/xr_viewer/fonts/overlay_charset.txt`,
copied from `E:\VAM2\UI字符中文字库\3500.txt` and currently contains 3,958
unique characters. The atlas is generated offline; the runtime
does not load a system font and does not rasterize text with Pillow on the
presenter thread.

The source font must be a distributable CJK OpenType font with glyph coverage
for this charset. The generator intentionally fails when the font file is
missing or invalid; it must not silently fall back to a partial system font.

The generated files are:

- `src/desktop2stereo/xr_viewer/fonts/d2s_overlay_msdf.*.png`: RGB MSDF atlas pages
- `src/desktop2stereo/xr_viewer/fonts/d2s_overlay_msdf.json`: glyph metrics and page layout

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
  src/desktop2stereo/xr_viewer/fonts/d2s_overlay_msdf.0.png \
  src/desktop2stereo/xr_viewer/fonts/grid-repacked
```

The production font must be distributed under a license compatible with the
application. The Windows system font is not a production dependency and is
only a fallback for the legacy Pillow path.

## Runtime path

The screen size/distance OSD, preset OSD, FPS panel, screen operation guide,
and controller operation guide use the same Vulkan MSDF compute path. The
atlas pages are concatenated and uploaded once to a resident GPU `rgba8`
storage image. When an overlay changes, Python submits only glyph metrics and
colors to a storage buffer; the compute shader samples the atlas, computes
`median(R,G,B)`, applies the atlas `distanceRange=4` coverage equation, and
writes the rounded background and glyphs to a Quad-sized intermediate storage
image. The existing Vulkan image copy then writes that image into the acquired
OpenXR Quad swapchain image.

The OSD background and MSDF glyphs therefore remain one Quad Layer. No
world-space MSDF geometry is submitted to the Projection Layer, and no second
Filament Engine is created. Keyboard and laser cursor overlays continue to use
their existing non-text Quad paths. If the device cannot use the fixed RGBA8
storage-image format, the compatibility path is logged and the CPU MSDF
decoder is used only for that capability failure.

The OSD canvas is content-sized: its width is the measured MSDF advance plus
horizontal padding, and its height is the atlas line height plus vertical
padding. The Quad world size uses the same canvas aspect ratio and scales with
the current screen width, so changing a screen preset does not leave a fixed
oversized background or compress the text into a narrow panel.

The screen-side operation guide is the exception to the content-sized world
height rule: its Quad height is always the current virtual screen height. To
keep the guide readable at every screen preset, its MSDF text keeps the same
canvas proportion as the guide background while the Quad follows the full
screen height. This prevents a large screen from leaving only a tiny centered
block of text inside the operation-guide panel.
