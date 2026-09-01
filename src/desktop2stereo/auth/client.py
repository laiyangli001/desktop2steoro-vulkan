"""HTTP client for the versioned Desktop2Stereo authentication API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


API_ROOT = "https://d2s.site/api/v1"


class AuthError(RuntimeError):
    def __init__(self, message: str, code: str = "auth_error", request_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id


@dataclass
class AuthSession:
    access_token: str
    refresh_token: str | None
    user: dict[str, Any]
    licenses: list[dict[str, Any]]
    selected_license_id: str | None = None
    server_time: int | None = None


@dataclass
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


class AuthClient:
    def __init__(self, base_url: str = API_ROOT, timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def login(self, email: str, password: str, turnstile_token: str | None = None) -> AuthSession:
        if not email.strip() or not password:
            raise AuthError("邮箱和密码不能为空", "invalid_input")
        payload = {"email": email.strip(), "password": password}
        if turnstile_token and turnstile_token.strip():
            payload["turnstile_token"] = turnstile_token.strip()
        try:
            response = httpx.post(
                f"{self.base_url}/auth/login",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_response(response)

    def authorize_device(self, client_name: str = "Desktop2Stereo") -> DeviceAuthorization:
        try:
            response = httpx.post(f"{self.base_url}/device/authorize", json={"client_name": client_name}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
            return DeviceAuthorization(str(data["device_code"]), str(data["user_code"]), str(data["verification_uri"]), int(data["expires_in"]), int(data.get("interval", 5)))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthError("授权服务器返回了无效设备码", "invalid_response") from exc

    def device_token(self, device_code: str) -> AuthSession:
        try:
            response = httpx.post(f"{self.base_url}/device/token", json={"device_code": device_code}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_response(response)

    def refresh(self, refresh_token: str) -> AuthSession:
        try:
            response = httpx.post(f"{self.base_url}/auth/refresh", json={"refresh_token": refresh_token}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        return self._session_from_response(response)

    def change_password(self, access_token: str, current_password: str, new_password: str) -> dict[str, Any]:
        if len(new_password) < 8:
            raise AuthError("新密码至少需要 8 位", "invalid_input")
        try:
            response = httpx.post(
                f"{self.base_url}/auth/password-change",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"current_password": current_password, "new_password": new_password},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def cancel_device(self, device_code: str) -> None:
        try:
            httpx.post(f"{self.base_url}/device/cancel", json={"device_code": device_code}, timeout=self.timeout)
        except httpx.HTTPError:
            pass

    def status(self, access_token: str) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.base_url}/license/status",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def license_list(self, access_token: str) -> list[dict[str, Any]]:
        try:
            response = httpx.get(f"{self.base_url}/license/list", headers={"Authorization": f"Bearer {access_token}"}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        licenses = data.get("licenses") if isinstance(data, dict) else None
        return licenses if isinstance(licenses, list) else []

    def activate_license(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        try:
            response = httpx.post(f"{self.base_url}/license/activate", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def switch_license(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        """Select/bind an account license without coupling the launcher to GUI code."""
        try:
            response = httpx.post(
                f"{self.base_url}/license/switch",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def change_license_mode(
        self,
        access_token: str,
        license_id: str,
        device_hash: str,
        mode: str,
        confirmation: str | None = None,
    ) -> dict[str, Any]:
        """Request a server-side license mode transition."""
        payload: dict[str, Any] = {"license_id": license_id, "device_hash": device_hash, "mode": mode}
        if confirmation is not None:
            payload["confirmation"] = confirmation
        try:
            response = httpx.post(
                f"{self.base_url}/license/change-mode",
                headers={"Authorization": f"Bearer {access_token}"},
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def revoke_free(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        try:
            response = httpx.post(f"{self.base_url}/license/revoke/free", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def renew_offline(self, access_token: str, license_id: str, device_hash: str, offline_period_days: int) -> dict[str, Any]:
        """Renew one bound license; the server calculates the new expiry."""
        try:
            response = httpx.post(
                f"{self.base_url}/license/renew",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "offline_period_days": offline_period_days},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def create_offline_extension_order(self, access_token: str, license_id: str, device_hash: str, channel: str) -> dict[str, Any]:
        """Create a server-priced +30 day offline extension order."""
        try:
            response = httpx.post(
                f"{self.base_url}/license/offline/extend",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "channel": channel},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def confirm_permanent(self, access_token: str, license_id: str, device_hash: str) -> dict[str, Any]:
        """Permanently bind one already-selected license after explicit confirmation."""
        try:
            response = httpx.post(
                f"{self.base_url}/license/permanent/confirm",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "confirmation": "PERMANENT"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def revoke_paid(self, access_token: str, license_id: str, device_hash: str, channel: str = "") -> dict[str, Any]:
        """Create a server-priced paid-revoke order; payment settlement is separate."""
        try:
            response = httpx.post(
                f"{self.base_url}/license/revoke/paid",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"license_id": license_id, "device_hash": device_hash, "channel": channel},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def request_manual_unbind(self, access_token: str, license_id: str, reason: str, purchase_proof_ref: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"license_id": license_id, "reason": reason}
        if purchase_proof_ref:
            payload["purchase_proof_ref"] = purchase_proof_ref
        try:
            response = httpx.post(f"{self.base_url}/license/manual-unbind", headers={"Authorization": f"Bearer {access_token}"}, json=payload, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def online_heartbeat(self, access_token: str, license_id: str, device_hash: str, lease_token: str | None = None) -> dict[str, Any]:
        try:
            response = httpx.post(f"{self.base_url}/license/online/heartbeat", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash, "lease_token": lease_token}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        return data if isinstance(data, dict) else {}

    def online_logout(self, access_token: str, lease_token: str) -> None:
        try:
            response = httpx.post(f"{self.base_url}/license/online/logout", headers={"Authorization": f"Bearer {access_token}"}, json={"lease_token": lease_token}, timeout=self.timeout)
        except httpx.HTTPError:
            return
        if response.status_code >= 400:
            self._raise_response(response)

    def issue_offline_entitlement(self, access_token: str, license_id: str, device_hash: str, offline_period_days: int) -> str:
        try:
            response = httpx.post(f"{self.base_url}/license/offline/issue", headers={"Authorization": f"Bearer {access_token}"}, json={"license_id": license_id, "device_hash": device_hash, "offline_period_days": offline_period_days}, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise AuthError(f"无法连接授权服务器：{exc}", "network_error") from exc
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
            entitlement = data.get("entitlement") if isinstance(data, dict) else None
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        if not entitlement:
            raise AuthError("服务器未返回离线授权凭证", "invalid_response")
        return str(entitlement)

    def logout(self, access_token: str) -> None:
        try:
            response = httpx.post(
                f"{self.base_url}/auth/logout",
                headers={"Authorization": f"Bearer {access_token}"},
                json={},
                timeout=self.timeout,
            )
        except httpx.HTTPError:
            return
        if response.status_code >= 400 and response.status_code != 401:
            self._raise_response(response)

    def _session_from_response(self, response: httpx.Response) -> AuthSession:
        if response.status_code >= 400:
            self._raise_response(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise AuthError("授权服务器返回了无效响应", "invalid_response") from exc
        if not isinstance(data, dict):
            raise AuthError("授权服务器返回了无效响应", "invalid_response")
        access_token = str(data.get("access_token", ""))
        if not access_token:
            raise AuthError("授权服务器未返回登录令牌", "invalid_response")
        try:
            server_time = int(data["server_time"]) if data.get("server_time") is not None else None
        except (TypeError, ValueError, OverflowError) as exc:
            raise AuthError("授权服务器时间响应无效", "invalid_response") from exc
        return AuthSession(
            access_token=access_token,
            refresh_token=data.get("refresh_token"),
            user=data.get("user") if isinstance(data.get("user"), dict) else {},
            licenses=data.get("licenses") if isinstance(data.get("licenses"), list) else [],
            server_time=server_time,
        )

    @staticmethod
    def _raise_response(response: httpx.Response) -> None:
        try:
            data = response.json()
        except ValueError:
            data = {}
        error = data.get("error") if isinstance(data, dict) else None
        message = data.get("message") if isinstance(data, dict) else None
        request_id = data.get("request_id") if isinstance(data, dict) else None
        raise AuthError(str(message or "授权验证失败"), str(error or "auth_error"), str(request_id) if request_id else None)
