"""Standalone Flet login window for the Desktop2Stereo launcher."""

from __future__ import annotations

import asyncio
import time
import webbrowser
from pathlib import Path
import flet as ft

from .client import AuthClient, AuthError, AuthSession, DeviceAuthorization
from .storage import TokenStore


AUTH_READY_FILE = Path(__file__).resolve().parents[1] / "logs" / "auth_ready.flag"


def _write_auth_ready_flag() -> None:
    AUTH_READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_READY_FILE.write_text("ready\n", encoding="utf-8")


class LoginLauncher:
    """Authenticate before importing either existing runtime GUI."""

    def __init__(self, client: AuthClient | None = None, store: TokenStore | None = None):
        self.client = client or AuthClient()
        self.store = store or TokenStore()
        self.session: AuthSession | None = None
        self._page: ft.Page | None = None
        self._busy = False

    @staticmethod
    def _error_text(error: AuthError) -> str:
        suffix = f"，请求 ID：{error.request_id}" if error.request_id else ""
        return f"{error} ({error.code}){suffix}"

    def run(self) -> AuthSession | None:
        ft.run(self._main, view=ft.AppView.FLET_APP_HIDDEN)
        return self.session

    async def _main(self, page: ft.Page):
        self._page = page
        page.title = "Desktop2Stereo 登录验证"
        page.window.width = 440
        page.window.height = 360
        page.window.resizable = False
        page.padding = 28
        page.theme = ft.Theme(color_scheme_seed="blue")

        email = ft.TextField(label="邮箱 / Email", autofocus=True)
        password = ft.TextField(label="密码 / Password", password=True, can_reveal_password=True)
        status = ft.Text(color=ft.Colors.RED, selectable=True)
        license_picker = ft.Dropdown(label="选择授权", visible=False, options=[])
        confirm = ft.Button(content="确认授权", visible=False, on_click=lambda e: self._confirm_selection(status, license_picker, confirm))
        login = ft.Button(content="登录 / Sign in", on_click=lambda e: self._login(e, email, password, status, license_picker, confirm))
        device_login = ft.OutlinedButton(content="浏览器授权登录", on_click=lambda e: self._device_login(status, license_picker, confirm))
        logout = ft.TextButton(content="退出登录 / Clear saved login", on_click=lambda e: self._logout_saved(status))

        page.add(ft.Column([
            ft.Text("Desktop2Stereo", size=26, weight=ft.FontWeight.BOLD),
            ft.Text("登录后验证授权，验证成功才会启动运行界面。"),
            email,
            password,
            license_picker,
            confirm,
            ft.Row([login, device_login], alignment=ft.MainAxisAlignment.END),
            status,
            logout,
        ], spacing=14))
        page.update()
        await page.window.wait_until_ready_to_show()
        await page.window.center()
        page.window.visible = True
        page.update()
        _write_auth_ready_flag()

    async def _login(self, _event, email: ft.TextField, password: ft.TextField, status: ft.Text, license_picker: ft.Dropdown, confirm: ft.Button):
        if self._busy:
            return
        self._busy = True
        status.value = "正在验证授权... / Validating..."
        status.color = ft.Colors.BLUE
        self._page.update()
        try:
            self.session = await asyncio.to_thread(self.client.login, email.value or "", password.value or "")
            if not self.session.access_token:
                raise AuthError("授权服务器未返回登录令牌", "invalid_response")
            await self._accept_session(status, license_picker, confirm)
        except AuthError as exc:
            status.value = self._error_text(exc)
            status.color = ft.Colors.RED
            self._page.update()
        finally:
            self._busy = False

    async def _device_login(self, status: ft.Text, license_picker: ft.Dropdown, confirm: ft.Button):
        if self._busy:
            return
        self._busy = True
        authorization: DeviceAuthorization | None = None
        try:
            authorization = await asyncio.to_thread(self.client.authorize_device)
            webbrowser.open(authorization.verification_uri)
            status.value = f"请在浏览器确认授权，用户码：{authorization.user_code}"
            status.color = ft.Colors.BLUE
            self._page.update()
            deadline = time.monotonic() + authorization.expires_in
            while time.monotonic() < deadline:
                try:
                    self.session = await asyncio.to_thread(self.client.device_token, authorization.device_code)
                    await self._accept_session(status, license_picker, confirm)
                    return
                except AuthError as exc:
                    if exc.code != "authorization_pending":
                        raise
                await asyncio.sleep(authorization.interval)
            raise AuthError("设备授权已超时，请重新尝试", "device_code_expired")
        except AuthError as exc:
            if authorization is not None:
                await asyncio.to_thread(self.client.cancel_device, authorization.device_code)
            status.value = self._error_text(exc)
            status.color = ft.Colors.RED
            self._page.update()
        finally:
            self._busy = False

    async def _accept_session(self, status: ft.Text, license_picker: ft.Dropdown, confirm: ft.Button):
        if not self.session or not self.session.licenses:
            raise AuthError("账号没有可用授权", "license_unavailable")
        if len(self.session.licenses) > 1 and not license_picker.value:
            license_picker.options = [ft.DropdownOption(key=str(item.get("id")), text=f"{item.get('license_code', item.get('id'))} · {item.get('mode', '')}") for item in self.session.licenses if item.get("id")]
            license_picker.visible = True
            confirm.visible = True
            status.value = "请选择要绑定当前设备的授权。"
            status.color = ft.Colors.BLUE
            self._page.update()
            return
        selected = license_picker.value or str(self.session.licenses[0].get("id", ""))
        if not selected or not any(str(item.get("id")) == selected for item in self.session.licenses):
            raise AuthError("请选择有效授权", "license_selection_required")
        self.session.selected_license_id = selected
        license_picker.visible = False
        confirm.visible = False
        if not self.session or not self.session.access_token:
            raise AuthError("授权服务器未返回登录令牌", "invalid_response")
        saved = self.store.save({"access_token": self.session.access_token, "refresh_token": self.session.refresh_token, "user": self.session.user, "licenses": self.session.licenses, "selected_license_id": selected})
        if not saved:
            raise AuthError("无法使用系统安全凭据保存登录状态，请先配置 Windows DPAPI、macOS Keychain 或 Linux Secret Service", "secure_storage_unavailable")
        await self._page.window.destroy()

    async def _confirm_selection(self, status: ft.Text, license_picker: ft.Dropdown, confirm: ft.Button):
        try:
            await self._accept_session(status, license_picker, confirm)
        except AuthError as exc:
            status.value = self._error_text(exc)
            status.color = ft.Colors.RED
            self._page.update()

    async def _logout_saved(self, status: ft.Text):
        saved = self.store.load()
        if saved and saved.get("access_token"):
            await asyncio.to_thread(self.client.logout, str(saved["access_token"]))
        self.store.clear()
        status.value = "已退出登录并清除本地登录状态。"
        status.color = ft.Colors.BLUE
        self._page.update()


def authenticate() -> AuthSession | None:
    return LoginLauncher().run()
