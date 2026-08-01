from __future__ import annotations

from types import SimpleNamespace

from app_runtime.runtime_output import CudaVulkanOutputAdapter


def _adapter_with_backend(backend):
    adapter = object.__new__(CudaVulkanOutputAdapter)
    adapter.presenter = SimpleNamespace(
        _filament_glow_mode="glow",
        _filament_glow_environment_enabled=True,
        _frosted_glow_lod=5.4,
        filament_bridge=SimpleNamespace(glow_vulkan_image_abi_available=True),
        vulkan=object(),
    )
    adapter._glow_gpu_backend = backend
    adapter._glow_gpu_last_submit = 0.0
    adapter._glow_gpu_status = None
    adapter._glow_gpu_submission_disabled = False
    return adapter


def test_cuda_adapter_does_not_submit_glow_for_glb_environment() -> None:
    class Backend:
        def poll(self) -> None:
            raise AssertionError("GLB environments must not poll the Glow backend")

    adapter = _adapter_with_backend(Backend())
    adapter.presenter._filament_glow_environment_enabled = False

    assert adapter._update_glow_gpu_source("cuda-source", frame_id=1) == {}
    assert adapter._glow_cpu_metadata() == {}


def test_cuda_adapter_reuses_completed_glow_while_new_dispatch_runs() -> None:
    resource = SimpleNamespace(width=320, height=180)

    class Backend:
        def __init__(self) -> None:
            self.submits = 0
            self.polls = 0

        def poll(self) -> None:
            self.polls += 1

        def submit(self, source, *, mode, frosted_lod) -> bool:
            self.submits += 1
            assert source == "cuda-source"
            assert mode == "glow"
            assert frosted_lod == 5.4
            return True

        def acquire(self, frame_id: int):
            return {
                "glow_vulkan_image": resource,
                "glow_vulkan_serial": 7,
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    backend = Backend()
    adapter = _adapter_with_backend(backend)

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=12)

    assert backend.polls == 1
    assert backend.submits == 1
    assert metadata["glow_vulkan_image"] is resource
    assert metadata["glow_source_size"] == (320, 180)


def test_cuda_adapter_keeps_last_glow_image_after_submit_failure() -> None:
    resource = SimpleNamespace(width=320, height=180)

    class Backend:
        def poll(self) -> None:
            return None

        def submit(self, _source, *, mode, frosted_lod) -> bool:
            raise RuntimeError("dispatch failed")

        def acquire(self, frame_id: int):
            return {
                "glow_vulkan_image": resource,
                "glow_vulkan_serial": 3,
                "_vulkan_glow_release": lambda _frame_id: None,
            }

    adapter = _adapter_with_backend(Backend())

    metadata = adapter._update_glow_gpu_source("cuda-source", frame_id=9)

    assert metadata["glow_vulkan_image"] is resource
    assert adapter._glow_gpu_submission_disabled is True
