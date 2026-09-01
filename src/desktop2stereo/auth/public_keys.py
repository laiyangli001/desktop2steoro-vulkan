"""Release-time ES256 public keys for offline entitlements.

Populate this file from the server's public ``license-key.json`` during a
release. Never put a private key in the client repository.
"""

PUBLIC_KEYS: dict[str, bytes] = {
    # "d2s-2026-01": b"-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n",
}
