from stereo_runtime.providers.nvidia import tensorrt_native
from stereo_runtime.providers.nvidia.tensorrt_native import _attach_tensorrt_progress_monitor


class _FakeProgressMonitorBase:
    def __init__(self) -> None:
        pass


class _FakeTrt:
    IProgressMonitor = _FakeProgressMonitorBase


class _FakeConfig:
    def __init__(self) -> None:
        self.progress_monitor = None


def test_attach_tensorrt_progress_monitor_emits_phase_progress(capsys):
    config = _FakeConfig()

    monitor = _attach_tensorrt_progress_monitor(_FakeTrt, config, "Building TensorRT engine: model.trt")

    assert monitor is config.progress_monitor
    monitor.phase_start("Build engine", None, 2)
    assert monitor.step_complete("Build engine", 0) is True
    assert monitor.step_complete("Build engine", 1) is True
    monitor.phase_finish("Build engine")

    out = capsys.readouterr().out
    assert "[D2S_PROGRESS]" in out
    assert "Building TensorRT engine: model.trt: Build engine" in out
    assert '"percent":100.0' in out
    assert '"downloaded":"2 steps"' in out
    assert '"unit":"steps"' in out


def test_attach_tensorrt_progress_monitor_is_optional():
    class _NoProgressTrt:
        pass

    assert _attach_tensorrt_progress_monitor(_NoProgressTrt, _FakeConfig(), "build") is None


def test_tensorrt_progress_monitor_throttles_phase_spam(monkeypatch, capsys):
    now = [100.0]
    monkeypatch.setattr(tensorrt_native.time, "perf_counter", lambda: now[0])
    config = _FakeConfig()
    monitor = _attach_tensorrt_progress_monitor(_FakeTrt, config, "Building TensorRT engine: model.trt")

    for index in range(5):
        monitor.phase_start(f"Phase {index}", None, 1)
        monitor.step_complete(f"Phase {index}", 0)
        monitor.phase_finish(f"Phase {index}")

    out = capsys.readouterr().out
    assert out.count("[D2S_PROGRESS]") == 1

    now[0] += 1.1
    monitor.phase_start("Phase later", None, 1)
    monitor.step_complete("Phase later", 0)
    assert capsys.readouterr().out.count("[D2S_PROGRESS]") == 1
