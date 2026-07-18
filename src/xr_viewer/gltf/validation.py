"""glTF extension diagnostics and fail-fast validation helpers."""

SUPPORTED_REQUIRED_EXTENSIONS = {
    "KHR_lights_punctual",
    "KHR_materials_unlit",
    "KHR_texture_transform",
}

SUPPORTED_OPTIONAL_EXTENSIONS = SUPPORTED_REQUIRED_EXTENSIONS | {
    "KHR_materials_emissive_strength",
    "KHR_materials_pbrSpecularGlossiness",
}

UNSUPPORTED_REQUIRED_EXTENSION_HINTS = {
    "KHR_draco_mesh_compression": "Transcode the mesh to plain glTF geometry or add Draco decoder support.",
    "EXT_meshopt_compression": "Transcode the mesh to plain glTF geometry or add Meshopt decoder support.",
    "EXT_mesh_gpu_instancing": "Bake GPU instances into ordinary scene nodes before loading.",
}


def _extension_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def audit_gltf_extensions(gltf):
    used = set(_extension_list(gltf.get("extensionsUsed")))
    required = set(_extension_list(gltf.get("extensionsRequired")))
    material_extensions = set()
    primitive_extensions = set()

    for material in gltf.get("materials") or []:
        if isinstance(material, dict) and isinstance(material.get("extensions"), dict):
            material_extensions.update(str(key) for key in material["extensions"].keys())
    for mesh in gltf.get("meshes") or []:
        if not isinstance(mesh, dict):
            continue
        for primitive in mesh.get("primitives") or []:
            if isinstance(primitive, dict) and isinstance(primitive.get("extensions"), dict):
                primitive_extensions.update(str(key) for key in primitive["extensions"].keys())

    used.update(material_extensions)
    used.update(primitive_extensions)
    unsupported_required = sorted(required - SUPPORTED_REQUIRED_EXTENSIONS)
    unsupported_optional = sorted((used - required) - SUPPORTED_OPTIONAL_EXTENSIONS)
    return {
        "extensionsUsed": sorted(used),
        "extensionsRequired": sorted(required),
        "unsupportedRequired": unsupported_required,
        "unsupportedRequiredHints": {
            extension: UNSUPPORTED_REQUIRED_EXTENSION_HINTS.get(
                extension,
                "Convert the asset or add renderer support before loading it.",
            )
            for extension in unsupported_required
        },
        "unsupportedOptional": unsupported_optional,
        "materialExtensions": sorted(material_extensions),
        "primitiveExtensions": sorted(primitive_extensions),
    }


def raise_unsupported_required_extensions(gltf, path):
    diagnostics = audit_gltf_extensions(gltf)
    unsupported = diagnostics["unsupportedRequired"]
    if unsupported:
        hints = diagnostics.get("unsupportedRequiredHints") or {}
        hint_text = " ".join(
            f"{extension}: {hints.get(extension)}"
            for extension in unsupported
            if hints.get(extension)
        )
        raise ValueError(
            f"Unsupported required glTF extensions for {path}: {', '.join(unsupported)}. "
            f"{hint_text or 'Convert the asset or add decoder/material support before loading it.'}"
        )
    return diagnostics


__all__ = [
    "SUPPORTED_OPTIONAL_EXTENSIONS",
    "SUPPORTED_REQUIRED_EXTENSIONS",
    "UNSUPPORTED_REQUIRED_EXTENSION_HINTS",
    "audit_gltf_extensions",
    "raise_unsupported_required_extensions",
]
