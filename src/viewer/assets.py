import platform


OS_NAME = platform.system()


def get_font_type(os_name=OS_NAME):
    if os_name == "Darwin":
        return "Verdana.ttf"
    if os_name == "Windows":
        return "verdana.ttf"
    if os_name == "Linux":
        return "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    return "Verdana.ttf"


def crop_icon(icon_img, os_name=OS_NAME):
    """Crop icon padding on Windows so the tray/window icon appears larger."""
    if os_name == "Windows":
        icon_img = icon_img.convert("RGBA")
        bbox = icon_img.getbbox()
        icon_img = icon_img.crop(bbox)
    return icon_img
