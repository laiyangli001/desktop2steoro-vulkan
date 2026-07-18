from stereo_runtime.render_size import RenderSizePolicy
from viewer.settings import resolve_viewer_settings


BASE_SETTINGS = {
    "Monitor Index": 1,
    "Display Mode": "Half-SBS",
    "Stereo Output": None,
    "Processing Resolution": "Auto",
    "Capture Mode": "Monitor",
    "Window Title": "",
    "Target FPS": 60,
    "Language": "EN",
    "Show FPS": False,
    "Depth Strength": 2.0,
    "Convergence": 0.0,
    "Fill 16:9": True,
    "Upscaler": "Off",
    "Upscaler Sharpness": 0.35,
    "Controller Model": "PICO",
    "Environment Model": "None",
    "XR Preview Window": False,
}


def test_resolve_viewer_settings_reads_vsync_key():
    settings = dict(BASE_SETTINGS, VSync=False)

    resolved = resolve_viewer_settings(settings)

    assert resolved.local_vsync is False
    assert not hasattr(resolved, "ipd")


def test_resolve_viewer_settings_requires_vsync_key():
    settings = dict(BASE_SETTINGS)

    try:
        resolve_viewer_settings(settings)
    except KeyError as exc:
        assert exc.args == ("VSync",)
    else:
        raise AssertionError("resolve_viewer_settings should require VSync")


def test_resolve_viewer_settings_resolves_xr_headset_screen_preset():
    settings = dict(BASE_SETTINGS, VSync=False, **{"XR Headset Model": "XREAL Air / Air 2 / Pro"})

    resolved = resolve_viewer_settings(settings)

    assert resolved.xr_headset_model == "XREAL Air / Air 2 / Pro"
    assert resolved.openxr_screen_distance == 4.0
    assert resolved.openxr_screen_width == 4.62


def test_resolve_viewer_settings_reads_render_size_config():
    settings = dict(
        BASE_SETTINGS,
        VSync=False,
        **{
            "Render Size Policy": "scaled",
            "Render Scale": "1K / 50%",
            "Render Fixed Width": 1600,
            "Render Fixed Height": 900,
            "Render Max Pixels": 2073600,
            "Render Min Dimension": 540,
            "Render Align": 8,
        },
    )

    resolved = resolve_viewer_settings(settings)

    assert resolved.render_size_config.policy is RenderSizePolicy.SCALED
    assert resolved.render_size_config.scale_factor == "1K / 50%"
    assert resolved.render_size_config.fixed_width == 1600
    assert resolved.render_size_config.fixed_height == 900
    assert resolved.render_size_config.max_pixels == 2073600
    assert resolved.render_size_config.min_dimension == 540
    assert resolved.render_size_config.align == 8

