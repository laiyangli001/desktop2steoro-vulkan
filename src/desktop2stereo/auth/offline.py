"""Local storage and verification for server-signed offline entitlements."""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from pathlib import Path
from typing import Any

from .device import device_hash


from .public_keys import PUBLIC_KEYS


class OfflineEntitlementError(RuntimeError):
    pass


def _decode_bytes(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _decode(value: str) -> str:
    return _decode_bytes(value).decode("utf-8")


class OfflineEntitlementStore:
    def __init__(self, app_dir: str | os.PathLike | None = None):
        root = Path(app_dir or Path.home() / ".desktop2stereo")
        self.path = root / "offline-entitlement.jws"

    def load(self) -> str | None:
        try:
            return self.path.read_text(encoding="ascii").strip() or None
        except OSError:
            return None

    def save(self, jws: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(jws, encoding="ascii")
        if os.name != "nt":
            temporary.chmod(0o600)
        os.replace(temporary, self.path)

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def verify_entitlement(jws: str, *, now: int | None = None, expected_device_hash: str | None = None) -> dict[str, Any]:
    try:
        encoded_header, encoded_payload, encoded_signature = jws.split(".")
        header = json.loads(_decode(encoded_header))
        claims = json.loads(_decode(encoded_payload))
        signature = _decode_bytes(encoded_signature)
    except (ValueError, TypeError, UnicodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise OfflineEntitlementError("离线授权凭证格式无效") from exc
    if header.get("alg") != "ES256" or header.get("typ") != "JWT":
        raise OfflineEntitlementError("离线授权签名算法无效")
    key_data = PUBLIC_KEYS.get(str(header.get("kid")))
    if not key_data:
        raise OfflineEntitlementError("离线授权公钥不可用")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

        if len(signature) != 64:
            raise OfflineEntitlementError("离线授权签名长度无效")
        der_signature = encode_dss_signature(int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big"))
        public_key = serialization.load_pem_public_key(key_data)
        public_key.verify(der_signature, f"{encoded_header}.{encoded_payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    except OfflineEntitlementError:
        raise
    except Exception as exc:
        raise OfflineEntitlementError("离线授权签名验证失败") from exc
    current = int(time.time() if now is None else now)
    required = {"version", "key_id", "entitlement_id", "license_id", "product", "device_hash", "mode", "features", "issued_at", "not_before", "expires_at", "trial", "offline_period_days"}
    if not required.issubset(claims) or claims.get("version") != 1 or claims.get("product") != "desktop2stereo" or claims.get("mode") != "offline":
        raise OfflineEntitlementError("离线授权字段无效")
    if expected_device_hash is not None and claims.get("device_hash") != expected_device_hash:
        raise OfflineEntitlementError("离线授权设备不匹配")
    try:
        not_before = int(claims["not_before"])
        expires_at = int(claims["expires_at"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise OfflineEntitlementError("离线授权时间字段无效") from exc
    if not_before > current + 60 or expires_at <= current:
        raise OfflineEntitlementError("离线授权已过期或尚未生效")
    return claims
