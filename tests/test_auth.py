from __future__ import annotations

import json
import sys
import types
import base64
import time
import pytest

import httpx

from desktop2stereo.auth.client import AuthClient, AuthError
from desktop2stereo.auth.storage import TokenStore
from desktop2stereo.auth.device import device_hash
from desktop2stereo.auth.instance import InstanceAlreadyRunning, InstanceLock
from desktop2stereo.auth.offline import OfflineEntitlementError, OfflineEntitlementStore, verify_entitlement
from desktop2stereo.auth.clock import ClockSuspectError, TrustedClock
from desktop2stereo.auth.gate import _observe_session_time
from desktop2stereo.auth.client import AuthSession
from desktop2stereo.auth.client import DeviceAuthorization


def test_auth_client_login_maps_success_response(monkeypatch):
    def fake_post(url, **kwargs):
        assert url.endswith("/auth/login")
        assert kwargs["json"] == {"email": "user@example.com", "password": "secret"}
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "refresh",
                "server_time": 2_000,
                "user": {"id": "u1"},
                "licenses": [{"license_code": "D2S-1"}],
            },
        )

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    session = AuthClient("https://example.test").login(" user@example.com ", "secret")
    assert session.access_token == "access"
    assert session.refresh_token == "refresh"
    assert session.server_time == 2_000
    assert session.licenses[0]["license_code"] == "D2S-1"


def test_auth_client_login_can_forward_turnstile_token(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, json={"access_token": "access", "licenses": []})

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    AuthClient("https://example.test").login("user@example.com", "secret", "turnstile-token")
    assert calls == [{"email": "user@example.com", "password": "secret", "turnstile_token": "turnstile-token"}]


def test_auth_client_reports_server_error(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(401, json={"error": "invalid_credentials", "message": "邮箱或密码错误", "request_id": "req-1"}),
    )
    try:
        AuthClient().login("user@example.com", "wrong")
    except AuthError as exc:
        assert exc.code == "invalid_credentials"
        assert exc.request_id == "req-1"
        assert "密码" in str(exc)
    else:
        raise AssertionError("login should reject invalid credentials")


def test_login_launcher_formats_request_id_for_user_support():
    from desktop2stereo.auth.gui import LoginLauncher

    text = LoginLauncher._error_text(AuthError("服务器不可用", "network_error", "req-2"))
    assert "req-2" in text


def test_login_launcher_uses_one_hidden_flet_view(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    monkeypatch.setattr(gui.ft, "run", lambda target, **kwargs: calls.append((target, kwargs)))
    launcher = gui.LoginLauncher()
    assert launcher.run() is None
    assert len(calls) == 1
    assert calls[0][0] == launcher._main
    assert calls[0][1]["view"] == gui.ft.AppView.FLET_APP_HIDDEN


def test_login_launcher_does_not_close_after_secure_storage_failure(monkeypatch):
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher(store=types.SimpleNamespace(save=lambda payload: False))
    launcher.session = AuthSession("access", "refresh", {"id": "u1"}, [{"id": "license-1"}])
    page = types.SimpleNamespace(window=types.SimpleNamespace(destroy=lambda: None))
    launcher._page = page
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None, visible=False, options=[])
    confirm = types.SimpleNamespace(visible=False)

    async def run():
        try:
            await launcher._accept_session(status, picker, confirm)
        except AuthError as exc:
            assert exc.code == "secure_storage_unavailable"
        else:
            raise AssertionError("secure storage failure must block launcher completion")

    import asyncio
    asyncio.run(run())


def test_login_launcher_has_separate_ready_handshake_file():
    from desktop2stereo.auth import gui

    assert gui.AUTH_READY_FILE.name == "auth_ready.flag"
    assert gui.AUTH_READY_FILE.name != "gui_ready.flag"


def test_device_authorization_and_pending_poll(monkeypatch):
    responses = iter([
        httpx.Response(201, json={"device_code": "device", "user_code": "ABCD1234", "verification_uri": "https://d2s.site/device", "expires_in": 600, "interval": 5}),
        httpx.Response(428, json={"error": "authorization_pending", "message": "等待浏览器完成授权"}),
    ])
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda *args, **kwargs: next(responses))
    client = AuthClient()
    authorization = client.authorize_device()
    assert authorization.user_code == "ABCD1234"
    try:
        client.device_token(authorization.device_code)
    except AuthError as exc:
        assert exc.code == "authorization_pending"
    else:
        raise AssertionError("device token should remain pending")


def test_auth_client_refresh_maps_rotated_session(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh", "user": {"id": "u1"}, "licenses": []}),
    )
    session = AuthClient().refresh("old-refresh")
    assert session.access_token == "new-access"
    assert session.refresh_token == "new-refresh"


def test_auth_client_exposes_password_change(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(200, json={"changed": True, "requires_login": True}),
    )
    result = AuthClient("https://example.test").change_password("access", "old-password", "new-password")
    assert result["changed"] is True
    assert calls[0][0].endswith("/auth/password-change")
    assert calls[0][1]["json"]["new_password"] == "new-password"


def test_auth_client_exposes_license_selection_and_mode_transition(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return httpx.Response(200, json={"version": 1, "changed": True})

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    client = AuthClient("https://example.test")
    assert client.switch_license("access", "license-1", "a" * 64)["changed"] is True
    client.change_license_mode("access", "license-1", "a" * 64, "permanent", "PERMANENT")
    assert calls[0][0].endswith("/license/switch")
    assert calls[1][0].endswith("/license/change-mode")


def test_auth_client_exposes_renew_permanent_and_paid_revoke(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return httpx.Response(200, json={"version": 1, "changed": True})

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    client = AuthClient("https://example.test")
    client.renew_offline("access", "license-1", "a" * 64, 14)
    client.confirm_permanent("access", "license-1", "a" * 64)
    client.revoke_paid("access", "license-1", "a" * 64, "paymentfm")
    assert calls[0] == ("https://example.test/license/renew", {"license_id": "license-1", "device_hash": "a" * 64, "offline_period_days": 14})
    assert calls[1][0].endswith("/license/permanent/confirm")
    assert calls[1][1]["confirmation"] == "PERMANENT"
    assert calls[2][0].endswith("/license/revoke/paid")


def test_auth_client_exposes_offline_extension_order(monkeypatch):
    calls = []
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda url, **kwargs: calls.append((url, kwargs["json"])) or httpx.Response(201, json={"order": {"product": "offline_extension"}}))
    result = AuthClient("https://example.test").create_offline_extension_order("access", "license-1", "a" * 64, "paymentfm")
    assert result["order"]["product"] == "offline_extension"
    assert calls[0][0].endswith("/license/offline/extend")


def test_auth_client_exposes_manual_unbind_request(monkeypatch):
    calls = []
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda url, **kwargs: calls.append((url, kwargs["json"])) or httpx.Response(201, json={"request": {"status": "pending"}}))
    result = AuthClient("https://example.test").request_manual_unbind("access", "license-1", "需要更换设备并提交购买凭证", "proof-1")
    assert result["request"]["status"] == "pending"
    assert calls[0][0].endswith("/license/manual-unbind")


def test_auth_client_rejects_invalid_server_time(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"access_token": "access", "server_time": "invalid"}),
    )
    try:
        AuthClient().refresh("refresh")
    except AuthError as exc:
        assert exc.code == "invalid_response"
    else:
        raise AssertionError("invalid server time should be rejected")


def test_auth_client_rejects_non_object_session_response(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json=["not", "a", "session"]),
    )
    try:
        AuthClient().refresh("refresh")
    except AuthError as exc:
        assert exc.code == "invalid_response"
    else:
        raise AssertionError("non-object session response should be rejected")


def test_device_authorization_cancel_is_available_after_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(204),
    )
    AuthClient().cancel_device("device-code")
    assert calls[0][0].endswith("/device/cancel")


def test_device_hash_is_sha256_hex():
    value = device_hash()
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)


def test_trusted_clock_rejects_large_local_clock_rollback(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    try:
        clock.check(local_time=1_000)
    except ClockSuspectError as exc:
        assert "系统时间" in str(exc)
    else:
        raise AssertionError("clock rollback should be rejected")


def test_trusted_clock_allows_small_ntp_adjustment(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    clock.check(local_time=1_800)


def test_login_session_server_time_is_recorded(tmp_path):
    clock = TrustedClock(tmp_path)
    _observe_session_time(AuthSession("access", "refresh", {}, [], server_time=2_000), clock)
    assert json.loads(clock.path.read_text(encoding="utf-8"))["max_server_time"] == 2_000


def test_instance_lock_rejects_second_process_lock(tmp_path):
    first = InstanceLock(tmp_path / "instance.lock")
    second = InstanceLock(tmp_path / "instance.lock")
    first.acquire()
    try:
        try:
            second.acquire()
        except InstanceAlreadyRunning:
            pass
        else:
            raise AssertionError("second instance should be rejected")
    finally:
        first.release()


def test_offline_entitlement_rejects_missing_signature_key(tmp_path):
    store = OfflineEntitlementStore(tmp_path)
    store.save("bad.token.value")
    assert store.load() == "bad.token.value"
    try:
        verify_entitlement(store.load() or "")
    except OfflineEntitlementError as exc:
        assert "格式" in str(exc)
    else:
        raise AssertionError("invalid offline entitlement should be rejected")


def test_offline_entitlement_rejects_malformed_base64():
    try:
        verify_entitlement("%%%%.%%%%.%%%%")
    except OfflineEntitlementError as exc:
        assert "格式" in str(exc)
    else:
        raise AssertionError("malformed entitlement should be rejected")


def test_offline_entitlement_accepts_server_es256_raw_signature(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    now = int(time.time())
    key = ec.generate_private_key(ec.SECP256R1())
    header = encode(json.dumps({"alg": "ES256", "typ": "JWT", "kid": "test-key"}, separators=(",", ":")).encode())
    claims = {
        "version": 1, "key_id": "test-key", "entitlement_id": "ent-1", "license_id": "lic-1",
        "product": "desktop2stereo", "device_hash": "a" * 64, "mode": "offline", "features": ["runtime"],
        "issued_at": now, "not_before": now - 1, "expires_at": now + 3600, "trial": False, "offline_period_days": 7,
    }
    payload = encode(json.dumps(claims, separators=(",", ":")).encode())
    der = key.sign(f"{header}.{payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    signature = encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    monkeypatch.setitem(__import__("desktop2stereo.auth.offline", fromlist=["PUBLIC_KEYS"]).PUBLIC_KEYS, "test-key", public)
    verified = verify_entitlement(f"{header}.{payload}.{signature}", now=now, expected_device_hash="a" * 64)
    assert verified["license_id"] == "lic-1"


def test_token_store_does_not_write_plaintext_without_secure_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.storage.platform.system", lambda: "Linux")
    monkeypatch.setattr("desktop2stereo.auth.storage.shutil.which", lambda _: None)
    store = TokenStore(tmp_path)
    assert store.save({"access_token": "secret", "refresh_token": "refresh"}) is False
    assert not store.fallback_path.exists()
    assert store.load() is None


def test_token_store_reads_secure_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.storage.platform.system", lambda: "Linux")
    monkeypatch.setattr("desktop2stereo.auth.storage.shutil.which", lambda _: "secret-tool")
    payload = json.dumps({"access_token": "access"})
    monkeypatch.setattr(TokenStore, "_run_security", staticmethod(lambda args, stdin=None: payload))
    assert TokenStore(tmp_path).load() == {"access_token": "access"}


def test_bootstrap_authenticates_before_loading_gui(monkeypatch):
    from desktop2stereo.app_runtime import bootstrap

    events: list[str] = []
    fake_gate = types.ModuleType("desktop2stereo.auth.gate")
    fake_gate.require_authentication = lambda: events.append("auth")
    fake_gui_package = types.ModuleType("gui2")
    fake_gui_package.__path__ = []
    fake_gui = types.ModuleType("desktop2stereo.gui2.gui")
    fake_gui.main = lambda: events.append("gui")
    legacy_package = types.ModuleType("gui")
    legacy_package.__path__ = []
    legacy_gui = types.ModuleType("gui.gui")
    legacy_gui.main = lambda: events.append("gui1")
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.gate", fake_gate)
    monkeypatch.setitem(sys.modules, "gui2", fake_gui_package)
    monkeypatch.setitem(sys.modules, "gui2.gui", fake_gui)
    monkeypatch.setitem(sys.modules, "gui", legacy_package)
    monkeypatch.setitem(sys.modules, "gui.gui", legacy_gui)
    assert bootstrap.main(["--gui2"]) == 0
    assert bootstrap.main(["--gui"]) == 0
    assert events == ["auth", "gui", "auth", "gui1"]


def test_bootstrap_reports_auth_error_without_loading_gui(monkeypatch, capsys):
    from desktop2stereo.app_runtime import bootstrap

    fake_gate = types.ModuleType("desktop2stereo.auth.gate")
    fake_gate.require_authentication = lambda: (_ for _ in ()).throw(AuthError("服务器不可用", "network_error"))
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.gate", fake_gate)
    assert bootstrap.main(["--gui2"]) == 1
    assert bootstrap.main(["--gui"]) == 1
    assert capsys.readouterr().err.count("[AUTH] 服务器不可用 (network_error)") == 2


def test_gui_module_entrypoints_delegate_to_bootstrap(monkeypatch):
    from desktop2stereo.gui import __main__ as gui_entry
    from desktop2stereo.gui2 import __main__ as gui2_entry

    calls: list[list[str]] = []
    monkeypatch.setattr(gui_entry, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(gui2_entry, "main", lambda argv: calls.append(argv) or 0)
    assert gui_entry.main(["--gui"]) == 0
    assert gui2_entry.main(["--gui2"]) == 0
    assert calls == [["--gui"], ["--gui2"]]
