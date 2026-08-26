from app_runtime.probe import hardware_regression_matrix


def test_hardware_matrix_covers_required_platforms_and_has_diagnostics():
    entries = hardware_regression_matrix()
    pairs = {(item["os"], item["hardware"]) for item in entries}
    assert ("Windows", "NVIDIA") in pairs
    assert ("Windows", "AMD") in pairs
    assert ("Windows", "Intel") in pairs
    assert ("Linux", "NVIDIA") in pairs
    assert ("Linux", "AMD") in pairs
    assert ("macOS", "Apple Silicon") in pairs
    assert ("macOS", "Intel Mac") in pairs
    assert all(item["status"] == "待硬件验证" for item in entries)
    assert all(item["runtime_log_prefix"] == "[D2S_BACKEND_STATUS]" for item in entries)
    assert all("zero_copy" in item["diagnostic_log_keys"] for item in entries)
    assert all("resource_kind" in item["diagnostic_log_keys"] for item in entries)
    assert all("directml_resource_mode" in item["diagnostic_log_keys"] for item in entries)
    assert all("pytest" in item["test_entry"] for item in entries)


def test_nvidia_matrix_keeps_windows_and_linux_fallbacks_distinct():
    entries = hardware_regression_matrix()
    by_key = {(item["os"], item["hardware"]): item for item in entries}
    assert by_key[("Windows", "NVIDIA")]["expected_inference_order"].endswith(
        "DirectML -> CPU"
    )
    assert "DirectML" not in by_key[("Linux", "NVIDIA")]["expected_inference_order"]
