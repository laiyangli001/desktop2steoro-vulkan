import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import { join, relative, resolve } from "node:path";

const packageRoot = resolve(process.argv[2] || "dist/Desktop2Stereo");
const output = resolve(process.argv[3] || join(packageRoot, "release-manifest.json"));

async function filesUnder(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await filesUnder(path));
    else if (entry.isFile() && path !== output) result.push(path);
  }
  return result;
}

const files = [];
for (const path of (await filesUnder(packageRoot)).sort()) {
  const bytes = await readFile(path);
  files.push({ path: relative(packageRoot, path).replaceAll("\\", "/"), sha256: createHash("sha256").update(bytes).digest("hex"), size: bytes.length });
}
let commit = "unknown";
try { commit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(); } catch {}
await writeFile(output, JSON.stringify({ version: 1, product: "desktop2stereo", commit, generated_at: new Date().toISOString(), files }, null, 2) + "\n", "utf8");
