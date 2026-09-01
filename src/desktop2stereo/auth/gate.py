"""Launcher authentication gate kept separate from GUI1 and GUI2."""

from __future__ import annotations

from .client import AuthClient, AuthError, AuthSession
from .clock import ClockSuspectError, TrustedClock
from .device import device_hash
from .instance import InstanceLock
from .gui import LoginLauncher
from .offline import OfflineEntitlementError, OfflineEntitlementStore, verify_entitlement
from .storage import TokenStore


def require_authentication() -> AuthSession:
    """Return a valid session or raise without importing a runtime GUI."""
    lock = InstanceLock()
    try:
        lock.acquire()
    except Exception as exc:
        raise AuthError(str(exc), "already_running") from exc
    client = AuthClient()
    store = TokenStore()
    trusted_clock = TrustedClock()
    saved = store.load()
    offline = _load_offline_session(saved, trusted_clock)
    if not saved and offline:
        return offline
    if saved and saved.get("access_token"):
        try:
            access_token = str(saved["access_token"])
            status = client.status(access_token)
            _observe_server_time(status, trusted_clock)
            if status.get("valid") is not True:
                raise AuthError("账号没有可用授权", "license_unavailable")
            licenses = status.get("licenses") if isinstance(status.get("licenses"), list) else []
            selected = str(saved.get("selected_license_id")) if saved.get("selected_license_id") else (str(licenses[0].get("id")) if len(licenses) == 1 and licenses[0].get("id") else None)
            if not selected or not any(isinstance(item, dict) and str(item.get("id")) == selected for item in licenses):
                raise AuthError("当前需要选择有效授权", "license_selection_required")
            client.activate_license(access_token, selected, device_hash())
            return AuthSession(
                access_token=str(saved["access_token"]),
                refresh_token=saved.get("refresh_token"),
                user=saved.get("user") if isinstance(saved.get("user"), dict) else {},
                licenses=licenses,
                selected_license_id=selected,
            )
        except AuthError as error:
            refresh_token = saved.get("refresh_token")
            if refresh_token:
                try:
                    session = client.refresh(str(refresh_token))
                    _observe_session_time(session, trusted_clock)
                    selected = str(saved.get("selected_license_id")) if saved.get("selected_license_id") else (str(session.licenses[0].get("id")) if len(session.licenses) == 1 and session.licenses[0].get("id") else None)
                    if not selected or not any(str(item.get("id")) == selected for item in session.licenses):
                        raise AuthError("当前需要选择有效授权", "license_selection_required")
                    client.activate_license(session.access_token, selected, device_hash())
                    session.selected_license_id = selected
                    store.save({"access_token": session.access_token, "refresh_token": session.refresh_token, "user": session.user, "licenses": session.licenses, "selected_license_id": selected})
                    return session
                except AuthError as refresh_error:
                    if refresh_error.code == "network_error" and offline:
                        return offline
            store.clear()
    session = LoginLauncher(client=client, store=store).run()
    if session is None:
        raise AuthError("未完成登录，已阻止启动运行界面", "login_required")
    _observe_session_time(session, trusted_clock)
    if session.selected_license_id:
        client.activate_license(session.access_token, session.selected_license_id, device_hash())
    return session


def _load_offline_session(saved: dict | None, trusted_clock: TrustedClock | None = None) -> AuthSession | None:
    try:
        (trusted_clock or TrustedClock()).check()
        claims = verify_entitlement(OfflineEntitlementStore().load() or "", expected_device_hash=device_hash())
    except (OfflineEntitlementError, ClockSuspectError):
        return None
    return AuthSession("", None, saved.get("user", {}) if isinstance(saved, dict) else {}, [claims], str(claims["license_id"]))


def validate_saved_authentication() -> AuthSession:
    """Validate saved credentials for a Runtime child without opening Flet."""
    client = AuthClient()
    store = TokenStore()
    saved = store.load()
    trusted_clock = TrustedClock()
    offline = _load_offline_session(saved, trusted_clock)
    if not saved or not saved.get("access_token"):
        if offline:
            return offline
        raise AuthError("未找到已保存的登录状态，请先登录", "login_required")
    access_token = str(saved["access_token"])
    try:
        status = client.status(access_token)
        _observe_server_time(status, trusted_clock)
        if status.get("valid") is not True:
            raise AuthError("账号没有可用授权", "license_unavailable")
        session = AuthSession(access_token, saved.get("refresh_token"), saved.get("user") if isinstance(saved.get("user"), dict) else {}, status.get("licenses") if isinstance(status.get("licenses"), list) else [])
    except AuthError as error:
        refresh_token = saved.get("refresh_token")
        if not refresh_token:
            if error.code == "network_error" and offline:
                return offline
            store.clear()
            raise
        try:
            session = client.refresh(str(refresh_token))
            _observe_session_time(session, trusted_clock)
        except AuthError as refresh_error:
            if refresh_error.code == "network_error" and offline:
                return offline
            store.clear()
            raise
        store.save({"access_token": session.access_token, "refresh_token": session.refresh_token, "user": session.user, "licenses": session.licenses})
    selected = str(saved.get("selected_license_id")) if saved.get("selected_license_id") else (str(session.licenses[0].get("id")) if len(session.licenses) == 1 and session.licenses[0].get("id") else None)
    if not selected or not any(str(item.get("id")) == selected for item in session.licenses):
        raise AuthError("当前需要选择有效授权", "license_selection_required")
    session.selected_license_id = selected
    client.activate_license(session.access_token, selected, device_hash())
    return session


def _observe_server_time(payload: dict, clock: TrustedClock) -> None:
    value = payload.get("server_time")
    if value is not None:
        try:
            clock.observe(int(value))
        except (TypeError, ValueError) as exc:
            raise AuthError("授权服务器时间响应无效", "invalid_response") from exc
        except ClockSuspectError as exc:
            raise AuthError(str(exc), "clock_suspect") from exc


def _observe_session_time(session: AuthSession, clock: TrustedClock) -> None:
    if session.server_time is not None:
        try:
            clock.observe(session.server_time)
        except ClockSuspectError as exc:
            raise AuthError(str(exc), "clock_suspect") from exc
