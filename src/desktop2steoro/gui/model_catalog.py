"""Lightweight depth-model metadata for GUI startup."""

from stereo_runtime.model_registry import ModelRegistry


def _resolutions_for_model(name):
    if name.startswith("InfiniDepth-"):
        return [192, 240, 304, 336, 384, 448, 512]
    if name.startswith("DA3"):
        return [182, 224, 280, 322, 378, 434, 504]
    if name == "DepthPro-Large":
        return [1536]
    return [196, 238, 294, 336, 392, 448, 518]


GUI_MODEL_CATALOG = {
    spec.name: {"resolutions": _resolutions_for_model(spec.name)}
    for spec in ModelRegistry.default().list()
}
