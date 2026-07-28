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

The texture is sampled as linear data. The fragment shader reconstructs the
glyph coverage from the median of the RGB channels, then applies the overlay
color and alpha. Text layout and glyph instance updates remain lightweight
metadata updates; the atlas is uploaded once and shared by all overlays.

## Reproducible generation

Install the pinned generator once from the repository root:

```powershell
npm install --no-audit --no-fund msdf-bmfont-xml@2.8.0 --prefix .tools/msdf
```

Generate the atlas:

```powershell
./scripts/tools/generate_msdf_overlay_font.ps1
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

When the optional ABI is present, the OpenXR Presenter decodes and uploads all
atlas pages once during Filament initialization. Text content and pose updates
remain separate from this upload; this prevents page decoding or texture
allocation from entering the per-frame render loop.

The screen size/distance and preset OSD now use the first mixed path: the
cached Quad Layer contains only the rounded background, while its colored text
is submitted as MSDF glyph geometry. FPS, keyboard, controller help, and laser
cursor overlays remain on the legacy path until their individual background
and interaction update policies are migrated.
