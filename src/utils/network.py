import os
import socket
from urllib.request import Request, urlopen


def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # The remote address does not need to be reachable for getsockname().
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


_HF_ENDPOINT_MIRROR = "https://hf-mirror.com"
_HF_ENDPOINT_OFFICIAL = "https://huggingface.co"
_HF_ENDPOINT_DEFAULT = _HF_ENDPOINT_MIRROR


def _can_connect(url: str, timeout: float = 5.0) -> bool:
    try:
        request = Request(url, headers={"User-Agent": "Desktop2Stereo/1.0"})
        with urlopen(request, timeout=timeout):
            return True
    except Exception:
        return False


def is_cn_ip(checker=None) -> bool:
    """Return True when Google and Hugging Face are not both reachable."""
    check = checker or _can_connect
    google_ok = bool(check("https://www.google.com", timeout=5))
    hf_ok = bool(check(_HF_ENDPOINT_OFFICIAL, timeout=5))
    return not (google_ok and hf_ok)


def huggingface_endpoint_candidates(async_probe=True) -> tuple[str, str]:
    if async_probe and not is_cn_ip():
        return _HF_ENDPOINT_OFFICIAL, _HF_ENDPOINT_MIRROR
    return _HF_ENDPOINT_MIRROR, _HF_ENDPOINT_OFFICIAL


def configure_huggingface_endpoint(async_probe=True):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        return endpoint
    endpoint = huggingface_endpoint_candidates(async_probe=async_probe)[0]
    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint
