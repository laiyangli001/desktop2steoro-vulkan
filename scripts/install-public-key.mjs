import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const targetPath = process.argv[2];
const rawManifest = process.env.D2S_LICENSE_PUBLIC_KEY_JSON || "";
if (!targetPath || !rawManifest.trim()) {
  console.error("Usage: D2S_LICENSE_PUBLIC_KEY_JSON='<manifest>' node scripts/install-public-key.mjs <client-public_keys.py>");
  process.exit(2);
}

let manifest;
try {
  manifest = JSON.parse(rawManifest);
} catch (error) {
  throw new Error(`Invalid public key manifest JSON: ${error.message}`);
}
const keyId = String(manifest.key_id || "").trim();
const publicKey = String(manifest.public_key_pem || "");
if (!/^[A-Za-z0-9._-]{1,80}$/.test(keyId) || !publicKey.includes("BEGIN PUBLIC KEY") || publicKey.includes("PRIVATE KEY")) {
  throw new Error("Manifest must contain one valid public key and no private key");
}
const output = `"""Release-time ES256 public keys for offline entitlements.\n\nGenerated from the server public key manifest. Never put a private key here.\n"""\n\nPUBLIC_KEYS: dict[str, bytes] = {\n    ${JSON.stringify(keyId)}: ${JSON.stringify(publicKey)}.encode("ascii"),\n}\n`;
await writeFile(resolve(targetPath), output, "utf8");
console.log(`Installed public key ${keyId} into ${resolve(targetPath)}`);
