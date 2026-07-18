import logging

from utils import OS_NAME

logger = logging.getLogger(__name__)


def get_devices():
    """
    Returns (devices_dict, is_rocm).
    devices_dict: {0: {"name": str, "Computing Device": torch.device}, ...}
    """
    is_rocm = False
    devices = {}
    count = 0
    try:
        import torch_directml
        if torch_directml.is_available():
            for i in range(torch_directml.device_count()):
                dev_name = torch_directml.device_name(i).strip().rstrip('\x00')
                devices[count] = {
                    "name": f"DirectML{i}: {dev_name}",
                    "Computing Device": torch_directml.device(i),
                }
                count += 1
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if torch.version.hip is not None:
                    is_rocm = True
                devices[count] = {"name": f"CUDA {i}: {name}", "Computing Device": torch.device(f"cuda:{i}")}
                count += 1
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices[count] = {"name": "MPS: Apple Silicon", "Computing Device": torch.device("mps")}
            count += 1
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                name = torch.xpu.get_device_name(i)
                devices[count] = {"name": f"XPU {i}: {name}", "Computing Device": torch.device(f"xpu:{i}")}
                count += 1
        devices[count] = {"name": "CPU", "Computing Device": torch.device("cpu")}
    except ImportError:
        raise ImportError("PyTorch Not Found! Make sure you have deployed the Python environment in '.env'.")
    return devices, is_rocm


class _LazyDevices(dict):
    """Lazy hardware detection, only runs get_devices() on first dict API access."""

    def __init__(self):
        self._loaded = False

    def _ensure(self):
        if not self._loaded:
            self._loaded = True
            try:
                detected_devices, is_rocm = get_devices()
                self.update(detected_devices)
                global IS_ROCM
                IS_ROCM = is_rocm
            except Exception as e:
                logger.warning("Hardware detection failed, fallback to CPU: %s", e)
                self.update({0: {"name": "CPU", "Computing Device": None}})

    def __getitem__(self, k):
        self._ensure()
        return super().__getitem__(k)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __len__(self):
        self._ensure()
        return super().__len__()

    def values(self):
        self._ensure()
        return super().values()

    def keys(self):
        self._ensure()
        return super().keys()

    def items(self):
        self._ensure()
        return super().items()

    def get(self, k, d=None):
        self._ensure()
        return super().get(k, d)

    def __contains__(self, k):
        self._ensure()
        return super().__contains__(k)


DEVICES = _LazyDevices()
IS_ROCM = False
