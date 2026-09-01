"""Online authorization lease held by the protected Runtime child."""

from __future__ import annotations

import threading

from .client import AuthClient, AuthError, AuthSession
from .device import device_hash


class RuntimeLease:
    def __init__(self, session: AuthSession, client: AuthClient | None = None):
        if not session.selected_license_id:
            raise AuthError("当前需要选择有效授权", "license_selection_required")
        self.session = session
        self.client = client or AuthClient()
        self.license_id = session.selected_license_id
        self.device = device_hash()
        self.lease_token: str | None = None
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        result = self.client.online_heartbeat(self.session.access_token, self.license_id, self.device)
        self.lease_token = str(result.get("lease_token", "")) or None
        if not self.lease_token:
            raise AuthError("服务器未返回运行租约", "invalid_response")
        self._thread = threading.Thread(target=self._run, name="D2SOnlineLease", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(300):
            try:
                result = self.client.online_heartbeat(self.session.access_token, self.license_id, self.device, self.lease_token)
                self.lease_token = str(result.get("lease_token", self.lease_token or "")) or None
            except AuthError:
                self.lost.set()
                return

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.lease_token:
            self.client.online_logout(self.session.access_token, self.lease_token)
