def get_device(index=0):
    import torch

    try:
        try:
            import torch_directml

            if torch_directml.is_available():
                return (
                    torch_directml.device(index),
                    f"Using DirectML device: {torch_directml.device_name(index)}",
                )
        except ImportError:
            pass
        if torch.backends.mps.is_available() and index == 0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        if torch.xpu.is_available():
            return torch.device("xpu"), f"Using XPU device: {torch.xpu.get_device_name(index)}"
        return torch.device("cpu"), "Using CPU device"
    except Exception:
        return torch.device("cpu"), "Using CPU device"
