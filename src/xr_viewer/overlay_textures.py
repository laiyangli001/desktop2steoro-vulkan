# Desktop2Stereo OpenXR viewer: shared overlay RGBA texture builders.

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .keyboard_layout import _KB_ROWS, _KB_TEX_H, _KB_TEX_W, _KB_UNITS_WIDE, _KeyEntry
from .laser_params import CURSOR_RING_INNER_RATIO
from viewer.controller_help import get_controller_help_rows


def load_overlay_font(size, font_type=None, *, prefer_cjk=False):
    candidates = []
    if prefer_cjk:
        candidates.append(r"C:\Windows\Fonts\msyh.ttc")
    candidates.extend((
        r"C:\Windows\Fonts\seguisym.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        font_type,
    ))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def build_keyboard_rgba(show_shifted, keyboard_width, keyboard_height, font_type=None, *, hover_indices=(), held_indices=(), hover_points=()):
    """Build the validated keyboard texture content for any renderer."""
    hover_indices = set(i for i in hover_indices if i is not None)
    held_indices = set(i for i in held_indices if i is not None)
    hover_points = tuple(p for p in hover_points if p is not None)
    tw, th = _KB_TEX_W, _KB_TEX_H
    row_h = th / len(_KB_ROWS)
    unit_w = tw / float(_KB_UNITS_WIDE)
    unit_m = float(keyboard_width) / float(_KB_UNITS_WIDE)
    pad = 3

    img = Image.new("RGBA", (tw, th), (30, 30, 35, 230))
    draw = ImageDraw.Draw(img)
    font = load_overlay_font(16, font_type)

    keys = []
    kw_half = float(keyboard_width) / 2.0
    kh_half = float(keyboard_height) / 2.0
    row_h_m = float(keyboard_height) / len(_KB_ROWS)

    for row_i, row in enumerate(_KB_ROWS):
        py0 = int(row_i * row_h)
        py1 = int((row_i + 1) * row_h)
        ly1 = kh_half - row_i * row_h_m
        ly0 = ly1 - row_h_m
        px = 0.0
        lx = -kw_half
        for label, vk_normal, shifted_label, vk_shifted, width_units in row:
            px_end = px + width_units * unit_w
            lx_end = lx + width_units * unit_m

            if vk_normal == -1:
                px = px_end
                lx = lx_end
                continue

            key_index = len(keys)
            is_held = key_index in held_indices
            is_hover = key_index in hover_indices
            draw.rectangle(
                [px + pad, py0 + pad, px_end - pad, py1 - pad],
                fill=(92, 122, 170, 255) if is_held else (72, 92, 125, 255) if is_hover else (60, 62, 70, 255),
                outline=(245, 248, 255, 255) if is_held or is_hover else (130, 132, 140, 255),
            )

            display_label = shifted_label if show_shifted and shifted_label is not None else label
            if font:
                draw.text(
                    ((px + px_end) / 2.0, (py0 + py1) / 2.0),
                    display_label,
                    font=font,
                    fill=(220, 220, 225, 255),
                    anchor="mm",
                )
            else:
                draw.text((int(px + pad + 2), int(py0 + pad + 2)), display_label, fill=(220, 220, 225, 255))

            keys.append(
                _KeyEntry(
                    label=label,
                    shifted_label=shifted_label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=(px / tw, py0 / th, px_end / tw, py1 / th),
                    rect_local=(lx, ly0, lx_end, ly1),
                )
            )

            px = px_end
            lx = lx_end

    for lx, ly in hover_points:
        cx = int((float(lx) + kw_half) / max(float(keyboard_width), 1e-6) * tw)
        cy = int((kh_half - float(ly)) / max(float(keyboard_height), 1e-6) * th)
        if 0 <= cx < tw and 0 <= cy < th:
            r_outer = 11
            r_inner = int(round(r_outer * CURSOR_RING_INNER_RATIO))
            draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=(80, 180, 255, 235))
            draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=(255, 255, 255, 220))

    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8)), keys


def build_cursor_rgba(size=64):
    size = max(8, int(size))
    yy, xx = np.ogrid[:size, :size]
    c = (size - 1) * 0.5
    d = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    outer = size * 0.45
    inner = outer * CURSOR_RING_INNER_RATIO
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[d <= outer] = (80, 180, 255, 235)
    rgba[d <= inner] = (255, 255, 255, 235)
    return np.ascontiguousarray(rgba)


def build_short_osd_rgba(lines, font_type=None, *, width=768, height=96):
    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, int(width) - 1, int(height) - 1], radius=14, fill=(32, 32, 36, 210))
    draw.text(
        (18, 16),
        "  ".join(str(line) for line in lines[:2]),
        font=load_overlay_font(24, font_type, prefer_cjk=True),
        fill=(220, 235, 255, 255),
    )
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def _text_width(draw, text, font):
    try:
        return int(draw.textlength(text, font=font))
    except AttributeError:
        return int(font.getsize(text)[0]) if hasattr(font, "getsize") else len(str(text)) * 10


def _draw_status_row(draw, y, label, value, *, label_font, value_font, label_color, value_color, x, val_x):
    def _ascent(font):
        try:
            return font.getmetrics()[0]
        except Exception:
            return 0

    label_dy = max(0, _ascent(value_font) - _ascent(label_font))
    value_dy = max(0, _ascent(label_font) - _ascent(value_font))
    draw.text((x, y + label_dy), label, font=label_font, fill=label_color)
    draw.text((val_x, y + value_dy), value, font=value_font, fill=value_color)


def build_fps_overlay_rgba(
    *,
    actual_fps,
    sbs_fps,
    latency_ms,
    screen_width,
    screen_height,
    screen_distance,
    depth_strength,
    vr_res,
    sbs_res,
    controller_brand,
    environment_visible,
    font_type=None,
    size=(768, 224),
):
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, ow - 1, oh - 1], radius=14, fill=(32, 32, 36, 210))

    font = load_overlay_font(24, font_type, prefer_cjk=True)
    label_font = load_overlay_font(24, font_type, prefer_cjk=True)
    c_label = (150, 158, 185, 255)
    c_green = (0, 230, 90, 255)
    c_cyan = (0, 210, 230, 255)
    c_amber = (255, 190, 40, 255)
    pad = 14
    labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Controller]", "[Environment]"]
    val_x = pad + max(_text_width(draw, label, label_font) for label in labels) + 10

    lat_str = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "N/A"
    fps_str = f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   Latency {lat_str}"
    _draw_status_row(
        draw,
        22,
        "[Performance]",
        fps_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_green,
        x=pad,
        val_x=val_x,
    )
    scr_str = (
        f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
        f"  @  {float(screen_distance):.2f} m"
        f"   Depth Strength {float(depth_strength):.2f}"
    )
    _draw_status_row(
        draw,
        56,
        "[3D Display]",
        scr_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    vw, vh = vr_res
    sw, sh = sbs_res
    _draw_status_row(
        draw,
        90,
        "[Resolution]",
        f"XR {int(vw)}x{int(vh)}/eye   Screen {int(sw)}x{int(sh)}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_amber,
        x=pad,
        val_x=val_x,
    )
    if controller_brand:
        _draw_status_row(
            draw,
            124,
            "[Controller]",
            f"Model: {controller_brand}",
            label_font=label_font,
            value_font=font,
            label_color=c_label,
            value_color=c_cyan,
            x=pad,
            val_x=val_x,
        )
    _draw_status_row(
        draw,
        158,
        "[Environment]",
        "ON" if environment_visible else "OFF",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_team_status_rgba(
    *,
    actual_fps,
    sbs_fps,
    latency_ms,
    screen_width,
    screen_height,
    screen_distance,
    depth_strength,
    vr_res,
    sbs_res,
    environment_name,
    controller_brand,
    shortcuts_visible,
    font_type=None,
    size=(768, 224),
):
    ow, oh = int(size[0]), int(size[1])
    img = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, ow - 1, oh - 1], radius=14, fill=(32, 32, 36, 210))

    font = load_overlay_font(24, font_type, prefer_cjk=True)
    label_font = load_overlay_font(24, font_type, prefer_cjk=True)
    c_label = (150, 158, 185, 255)
    c_green = (0, 230, 90, 255)
    c_cyan = (0, 210, 230, 255)
    c_amber = (255, 190, 40, 255)
    pad = 14
    labels = ["[Performance]", "[3D Display]", "[Resolution]", "[Show Shortcuts]", "[Models]"]
    val_x = pad + max(_text_width(draw, label, label_font) for label in labels) + 10

    lat_str = f"{float(latency_ms):.0f}ms" if float(latency_ms or 0.0) > 0 else "--"
    _draw_status_row(
        draw,
        22,
        "[Performance]",
        f"XR {float(actual_fps):.0f} FPS   SBS {float(sbs_fps):.0f} FPS   Latency {lat_str}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_green,
        x=pad,
        val_x=val_x,
    )
    _draw_status_row(
        draw,
        56,
        "[3D Display]",
        (
            f"{float(screen_width):.2f} x {float(screen_height):.2f} m"
            f"  @  {float(screen_distance):.2f} m   Depth Strength {float(depth_strength):.2f}"
        ),
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )
    vw, vh = vr_res
    sw, sh = sbs_res
    _draw_status_row(
        draw,
        90,
        "[Resolution]",
        f"XR {int(vw)}x{int(vh)}/eye   Screen {int(sw)}x{int(sh)}",
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_amber,
        x=pad,
        val_x=val_x,
    )
    model_str = f"Environment: {environment_name or 'Default'}"
    if controller_brand:
        model_str += f"   Controller: {controller_brand}"
    _draw_status_row(
        draw,
        124,
        "[Models]",
        model_str,
        label_font=label_font,
        value_font=font,
        label_color=c_label,
        value_color=c_cyan,
        x=pad,
        val_x=val_x,
    )

    draw.text((pad, 158), "[Show Shortcuts]", font=label_font, fill=c_label)
    sw_w, sw_h = 52, 26
    sw_x = val_x
    sw_y = 158 + (34 - sw_h) // 2
    track_col = (0, 200, 80, 255) if shortcuts_visible else (80, 84, 100, 255)
    draw.rounded_rectangle([sw_x, sw_y, sw_x + sw_w, sw_y + sw_h], radius=sw_h // 2, fill=track_col)
    kr = sw_h // 2 - 2
    kx = (sw_x + sw_w - kr - 3) if shortcuts_visible else (sw_x + kr + 3)
    ky = sw_y + sw_h // 2
    draw.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=(255, 255, 255, 255))
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def build_help_rgba(*, environment_mode=False, font_type=None, lang="EN"):
    rows, env_rows = get_controller_help_rows(lang)
    rows = env_rows if environment_mode else rows
    return _build_help_rows_rgba(rows, font_type=font_type, two_columns=True)


def build_team_help_rgba(*, font_type=None, lang="EN"):
    rows, _env_rows = get_controller_help_rows(lang)
    return _build_help_rows_rgba(rows, font_type=font_type, two_columns=False)


def _build_help_rows_rgba(rows, *, font_type=None, two_columns):
    font_size = 16 if two_columns else 21
    title_size = 18 if two_columns else 21
    font = load_overlay_font(font_size, font_type, prefer_cjk=True)
    title_font = load_overlay_font(title_size, font_type, prefer_cjk=True)
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    col_w = [0, 0, 0]
    for row in rows:
        is_title = bool(row[3]) if len(row) >= 4 else False
        for ci in range(3):
            col_w[ci] = max(col_w[ci], _text_width(draw, row[ci], title_font if is_title else font))

    gap = 20
    mid_gap = 50
    pad_x = 30
    pad_y = 20
    line_h = font_size + 6
    inner_w = col_w[0] + gap + col_w[1] + gap + col_w[2]
    if two_columns:
        title_indices = [i for i, row in enumerate(rows) if len(row) >= 4 and row[3]]
        mid_idx = title_indices[4] if len(title_indices) > 4 else len(rows)
        left_rows = rows[:mid_idx]
        right_rows = rows[mid_idx:]
        tw = inner_w * 2 + mid_gap + pad_x * 2
        th = max(len(left_rows), len(right_rows)) * line_h + pad_y * 2
    else:
        left_rows = rows
        right_rows = []
        tw = inner_w + pad_x * 2
        th = len(rows) * line_h + pad_y * 2

    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, tw - 1, th - 1], radius=14, fill=(18, 18, 28, 210))
    col_x = [pad_x, pad_x + col_w[0] + gap, pad_x + col_w[0] + gap + col_w[1] + gap]
    col_x2 = [pad_x + inner_w + mid_gap, pad_x + inner_w + mid_gap + col_w[0] + gap, pad_x + inner_w + mid_gap + col_w[0] + gap + col_w[1] + gap]

    def _draw_rows(group_rows, xs):
        for ri, row in enumerate(group_rows):
            is_title = bool(row[3]) if len(row) >= 4 else False
            y = pad_y + ri * line_h
            row_font = title_font if is_title else font
            color = (90, 190, 255, 255) if is_title else (200, 210, 235, 255)
            for ci in range(3):
                if row[ci]:
                    draw.text((xs[ci], y), row[ci], font=row_font, fill=color)

    _draw_rows(left_rows, col_x)
    if two_columns:
        _draw_rows(right_rows, col_x2)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
