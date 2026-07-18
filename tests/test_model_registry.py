from pathlib import Path

from stereo_runtime import DepthRuntimeConfig, ModelRegistry, resolve_model_dir
from stereo_runtime.adapter import depth_provider_config_from_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_default_registry_contains_d2s_models():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("Distill-Any-Depth-Base") == "lc700x/Distill-Any-Depth-Base-hf"
    assert registry.resolve_model_id("DepthPro-Large") == "apple/DepthPro-hf"
    assert registry.resolve_model_id("depth-anything/DA3-BASE") == "depth-anything/DA3-BASE"
    assert registry.resolve_model_id("depth-anything-indoor-large") == "lc700x/depth-anything-indoor-large-hf"
    assert registry.resolve_model_id("depth-anything-outdoor-large") == "lc700x/depth-anything-outdoor-large-hf"
    assert len(registry.names()) >= 42


def test_depth_runtime_config_resolves_model_dir_from_model_name():
    config = DepthRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="./models")

    assert config.resolved_model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(config.model_path).replace("\\", "/").endswith("models/models--lc700x--Distill-Any-Depth-Base-hf")
    assert config.onnx_path.name == "model_fp16_294x518.onnx"
    assert config.fp32_onnx_path.name == "model_fp32_294x518.onnx"
    assert config.trt_engine_path.name == "model_fp16_294x518.trt"


def test_resolve_model_dir_uses_huggingface_cache_name():
    path = resolve_model_dir("owner/repo-name", "cache-root")

    assert str(path).replace("\\", "/") == "cache-root/models--owner--repo-name"


def test_da3_registry_uses_correct_huggingface_repo_revisions():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("DA3-SMALL") == "depth-anything/DA3-SMALL"
    assert registry.resolve_model_id("DA3-BASE") == "depth-anything/DA3-BASE"
    assert registry.resolve_model_id("DA3-LARGE") == "depth-anything/DA3-LARGE-1.1"
    assert registry.resolve_model_id("DA3-GIANT") == "depth-anything/DA3-GIANT-1.1"
    assert registry.resolve_model_id("DA3NESTED-GIANT-LARGE") == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert registry.resolve_model_id("DA3MONO-LARGE") == "depth-anything/DA3MONO-LARGE"


def test_da3_model_dir_uses_resolved_repo_id():
    config = DepthRuntimeConfig(model_id="DA3-LARGE", cache_dir="./models")

    assert config.resolved_model_id == "depth-anything/DA3-LARGE-1.1"
    assert str(config.model_path).replace("\\", "/").endswith("models/models--depth-anything--DA3-LARGE-1.1")


def test_depth_runtime_config_passes_trt_build_options():
    config = DepthRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        depth_backend="tensorrt_native",
        build_trt_engine=True,
        force_rebuild_trt=True,
        use_cuda_graph=True,
    )

    provider_config = depth_provider_config_from_runtime(config)

    assert provider_config.build_engine is True
    assert provider_config.force_rebuild is True
    assert provider_config.use_cuda_graph is True


def test_model_registry_is_d2s_model_mapping_source():
    names = set(ModelRegistry.default().names())

    assert "Distill-Any-Depth-Base" in names
    assert "Depth-Anything-V2-Large" in names
    assert "DA3-LARGE" in names
    assert "depth-anything-indoor-large" in names
    assert "depth-anything-outdoor-large" in names
    assert "DepthPro-Large" in names


def test_utils_model_mapping_delegates_to_runtime_registry():
    utils_source = (ROOT / "src" / "utils" / "__init__.py").read_text(encoding="utf-8")
    runtime_exports_source = (ROOT / "src" / "utils" / "runtime_exports.py").read_text(encoding="utf-8")
    capabilities_source = (ROOT / "src" / "stereo_runtime" / "model_capabilities.py").read_text(encoding="utf-8")

    assert "model_name_mapping" in runtime_exports_source
    assert "from .model_registry import ModelRegistry" in capabilities_source
    assert "spec.name: spec.model_id" in capabilities_source
    assert '"Depth-Anything-V2-Small":' not in utils_source
    assert '"Depth-Anything-V2-Small":' not in runtime_exports_source
