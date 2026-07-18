import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.depth_provider import (
    DepthProviderConfig,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    GenericAutoDepthProvider,
    TorchDepthProvider,
    create_depth_provider,
    estimate_depth,
    _normalize_depth,
)
from stereo_runtime.depth_onnx_provider import DistillPreprocessor, ModelOnnxPreprocessor, _preprocess_distill_rgb, estimate_depth_onnx_cuda
from stereo_runtime.providers.nvidia.tensorrt_ort import estimate_depth_nvidia_chain


class FakeTorchProvider:
    def __init__(self, **kwargs):
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="Distill-Any-Depth-Base",
            model_id="fake",
            depth_resolution=518,
            cache_dir=str(kwargs.get("cache_dir") or ""),
            load_mode="test",
            depth_backend="pytorch_cpu",
            runtime="fake",
        )

    def predict(self, rgb):
        return torch.zeros(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])


def test_normalize_depth_uses_beta_percentile_bounds():
    values = torch.full((100,), 0.5)
    values[0] = 0.0
    values[1] = 0.1
    values[2] = 0.2
    values[97] = 0.8
    values[98] = 0.9
    values[99] = 1.0
    values = values.view(1, 1, 10, 10)

    normalized = _normalize_depth(values)

    mid = normalized[0, 0, 5, 0]
    assert 0.45 < float(mid) < 0.55
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_nvidia_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_ort as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_nvidia_chain(
        rgb,
        device="cpu",
        onnx_path=missing,
        prefer_onnx=True,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_nvidia_provider_requires_tensorrt_when_requested(tmp_path):
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    try:
        estimate_depth_nvidia_chain(
            rgb,
            device="cpu",
            onnx_path=missing,
            require_tensorrt=True,
        )
    except RuntimeError as exc:
        assert "tensorrt" in str(exc)
    else:
        raise AssertionError("expected TensorRT requirement to fail")


def test_onnx_cuda_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_runtime.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "TorchDepthProvider", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_depth_onnx_cuda(
        rgb,
        device="cpu",
        onnx_path=missing,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_create_depth_provider_falls_back_to_pytorch_when_ort_missing(monkeypatch, tmp_path):
    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "_onnxruntime_available", lambda: False)

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="onnx_cuda",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBase518)
    assert isinstance(provider, TorchDepthProvider)
    assert provider.info.depth_backend == "pytorch_cpu"


def test_create_depth_provider_requires_ort_when_fallback_disabled(monkeypatch, tmp_path):
    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "_onnxruntime_available", lambda: False)

    try:
        create_depth_provider(
            DepthProviderConfig(
                backend="onnx_cuda",
                device="cpu",
                cache_dir=tmp_path,
                local_files_only=True,
                allow_pytorch_fallback=False,
            )
        )
    except RuntimeError as exc:
        assert "ONNX Runtime is not installed" in str(exc)
    else:
        raise AssertionError("expected missing ONNX Runtime to fail when fallback is disabled")

def test_create_depth_provider_supports_persistent_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch_cuda",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, DistillAnyDepthBase518)
    assert isinstance(provider, TorchDepthProvider)
    assert provider.info.depth_backend == "pytorch_cpu"


def test_create_depth_provider_supports_generic_pytorch_provider(tmp_path):
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="pytorch",
            model_id="Intel/dpt-large",
            model_name="dpt-large",
            device="cpu",
            cache_dir=tmp_path,
            local_files_only=True,
        )
    )

    assert isinstance(provider, GenericAutoDepthProvider)
    assert provider.info.model_id == "Intel/dpt-large"
    assert provider.info.runtime == "transformers-generic"


def test_generic_provider_predict_profile_with_fake_transformers_model(monkeypatch, tmp_path):
    class Output:
        predicted_depth = torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            return Output()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeModel()

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForDepthEstimation", FakeAutoModel)
    provider = GenericAutoDepthProvider(
        model_id="Intel/dpt-large",
        model_name="dpt-large",
        device="cpu",
        cache_dir=tmp_path,
        local_files_only=True,
        depth_resolution=8,
        patch_size=1,
    )

    result = provider.predict_profile(torch.zeros(1, 3, 8, 8))

    assert result.depth.shape == (1, 1, 8, 8)
    assert float(result.depth.min()) >= 0.0
    assert float(result.depth.max()) <= 1.0


def test_create_depth_provider_supports_native_tensorrt(monkeypatch, tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    class Paths:
        trt_fp16_path = tmp_path / "model.trt"

        def onnx_path_for_dtype(self, dtype_name):
            return tmp_path / "model.onnx"

        def trt_path_for_dtype(self, dtype_name):
            return self.trt_fp16_path

    class Artifacts:
        selected_onnx_path = tmp_path / "model.onnx"
        paths = Paths()

    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    monkeypatch.setattr(native_module, "_prepare_accelerated_artifacts", lambda *args, **kwargs: Artifacts())
    engine_path = tmp_path / "model.trt"
    provider = create_depth_provider(
        DepthProviderConfig(
            backend="tensorrt_native",
            device="cuda",
            cache_dir=tmp_path,
            build_engine=True,
        )
    )

    assert isinstance(provider, NativeTensorRtDepthProvider)
    assert provider.info.depth_backend == "tensorrt_native"
    assert provider.onnx_path.name == "model_fp16_294x518.onnx"
    assert provider.engine_path.name == "model_fp16_294x518.trt"
    assert provider.build_engine is True

    provider._ensure_artifacts_for_input(768, 1024)
    assert provider.onnx_path == tmp_path / "model.onnx"
    assert provider.engine_path == engine_path


def test_native_tensorrt_infers_large_metadata_from_model_path(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    onnx_path = model_dir / "model_fp16_294x518.onnx"
    engine_path = model_dir / "model_fp16_294x518.trt"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=onnx_path,
        engine_path=engine_path,
    )

    assert provider.info.model_id == "xingyang1/Distill-Any-Depth-Large-hf"
    assert provider.info.model_name == "Distill-Any-Depth-Large"


def test_native_tensorrt_keeps_explicit_metadata(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    model_dir = tmp_path / "models--xingyang1--Distill-Any-Depth-Large-hf"
    provider = NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=model_dir / "model_fp16_294x518.onnx",
        engine_path=model_dir / "model_fp16_294x518.trt",
        model_id="custom/model",
        model_name="Custom Model",
    )

    assert provider.info.model_id == "custom/model"
    assert provider.info.model_name == "Custom Model"


def test_estimate_depth_uses_configured_provider(monkeypatch):
    class Provider:
        info = DepthProviderInfo(
            provider="fake",
            model_name="fake",
            model_id="fake",
            depth_resolution=518,
            cache_dir="",
            load_mode="test",
            depth_backend="fake",
            runtime="test",
        )

        def predict(self, rgb):
            return torch.ones(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])

    import stereo_runtime.depth_provider as provider_module

    monkeypatch.setattr(provider_module, "create_depth_provider", lambda config=None: Provider())
    depth, info = estimate_depth(torch.zeros(1, 3, 4, 5), {"backend": "fake"})

    assert depth.shape == (1, 1, 4, 5)
    assert info.depth_backend == "fake"


def test_distill_preprocessor_matches_reference():
    rgb = torch.linspace(0, 1, steps=3 * 12 * 16, dtype=torch.float32).view(1, 3, 12, 16)
    device = torch.device("cpu")
    dtype = torch.float32

    expected = _preprocess_distill_rgb(rgb, device=device, dtype=dtype)
    actual = DistillPreprocessor(device=device, dtype=dtype)(rgb)

    assert torch.equal(actual, expected)


def test_infinidepth_onnx_preprocessor_uses_patch_16_without_normalization():
    rgb = torch.linspace(0, 1, steps=3 * 12 * 18, dtype=torch.float32).view(1, 3, 12, 18)
    preprocessor = ModelOnnxPreprocessor(
        model_id="lc700x/InfiniDepth-Base",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    actual = preprocessor(rgb)

    assert actual.shape[-2] % 16 == 0
    assert actual.shape[-1] % 16 == 0
    assert float(actual.min()) >= 0.0
    assert float(actual.max()) <= 1.0

def test_onnx_provider_defers_load_until_frame_artifact_size_is_known(tmp_path):
    from stereo_runtime.depth_onnx_provider import OnnxCudaDepthProvider

    provider = OnnxCudaDepthProvider(
        device="cpu",
        cache_dir=tmp_path,
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_name="Distill-Any-Depth-Base",
        onnx_dtype="fp32",
    )

    assert provider.load() is None
    assert provider.onnx_path.name.startswith("model_fp16_")
    assert provider._session is None


def test_onnx_provider_prepares_artifact_for_first_frame_shape(monkeypatch, tmp_path):
    from stereo_runtime.depth_onnx_provider import OnnxCudaDepthProvider

    calls = {}
    selected = tmp_path / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_392x518.onnx"

    class Artifacts:
        selected_onnx_path = selected

    def fake_prepare(*args, **kwargs):
        calls.update(kwargs)
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(b"onnx")
        return Artifacts()

    import stereo_runtime.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "_prepare_accelerated_artifacts", fake_prepare)
    provider = OnnxCudaDepthProvider(
        device="cpu",
        cache_dir=tmp_path,
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_name="Distill-Any-Depth-Base",
    )

    provider._ensure_artifacts_for_input(768, 1024)

    assert calls["input_size"] == (392, 518)
    assert provider.onnx_path == selected
    assert provider._preprocessor.fixed_input_size == (392, 518)


def test_distill_preprocessor_can_use_fixed_tensorrt_input_size():
    rgb = torch.zeros(1, 3, 2160, 1920)
    preprocessor = DistillPreprocessor(
        device=torch.device("cpu"),
        dtype=torch.float32,
        fixed_input_size=(294, 518),
    )

    tensor = preprocessor(rgb)

    assert tensor.shape == (1, 3, 294, 518)
    assert preprocessor.input_size(2160, 1920) == (294, 518)


def test_native_tensorrt_defers_load_until_frame_artifact_size_is_known(tmp_path):
    from stereo_runtime.providers.nvidia.tensorrt_native import NativeTensorRtDepthProvider

    provider = NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        onnx_dtype="fp32",
    )

    assert provider.load() is None
    assert provider.onnx_path.name.startswith("model_fp16_")
    assert provider._engine is None


def test_native_tensorrt_load_prints_compact_engine_path(monkeypatch, tmp_path, capsys):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    class FakeEngine:
        input_image_size = (294, 518)

        def __init__(self, engine_path, *, device, dtype):
            self.engine_path = engine_path

    engine_path = tmp_path / "model.trt"
    onnx_path = tmp_path / "model.onnx"
    engine_path.write_bytes(b"trt")
    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        onnx_path=onnx_path,
        engine_path=engine_path,
    )

    monkeypatch.setattr(native_module, "ensure_tensorrt_dll_path", lambda: [tmp_path / "dlls"])
    monkeypatch.setattr(native_module, "NativeTensorRtEngine", FakeEngine)

    provider.load()

    output = capsys.readouterr().out.strip()
    assert output == f"[TensorRT] native provider loaded: engine={engine_path}"
    assert " onnx=" not in output
    assert " dll_dirs=" not in output


def test_native_tensorrt_provider_uses_engine_static_input_size(monkeypatch, tmp_path):
    import stereo_runtime.providers.nvidia.tensorrt_native as native_module

    captured = {}

    class FakeEngine:
        input_image_size = (294, 518)

        def __call__(self, tensor):
            captured["tensor_shape"] = tuple(tensor.shape)
            return torch.zeros(1, 1, 294, 518, dtype=torch.float32)

    class FakePreprocessor:
        fixed_input_size = None

        def __call__(self, rgb):
            assert self.fixed_input_size == (294, 518)
            return torch.zeros(1, 3, *self.fixed_input_size, dtype=torch.float32)

    provider = native_module.NativeTensorRtDepthProvider(
        device="cuda",
        cache_dir=tmp_path,
        engine_path=tmp_path / "model.trt",
    )
    provider._engine = FakeEngine()
    provider._preprocessor = FakePreprocessor()
    provider._artifact_input_size = (294, 518)
    monkeypatch.setattr(provider, "_ensure_artifacts_for_input", lambda height, width: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = provider.predict_profile(torch.zeros(1, 3, 2160, 1920))

    assert captured["tensor_shape"] == (1, 3, 294, 518)
    assert result.depth.shape == (1, 1, 2160, 1920)
