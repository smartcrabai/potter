// Fetches the pinned official VRM / glTF JSON schemas into .cache/schema/ and verifies
// every file's sha256 against schema/schemas.lock.json. Mismatch is a hard failure.
// Only missing files are fetched; the lock is never updated by this script.
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
export const schemaDir = join(root, ".cache", "schema");
const lockPath = join(root, "schema", "schemas.lock.json");

const HEX64 = /^[0-9a-f]{64}$/;
const HEX40 = /^[0-9a-f]{40}$/;

export async function loadLock() {
  const lock = JSON.parse(await readFile(lockPath, "utf8"));
  const names = new Set();
  for (const source of lock.sources) {
    if (!HEX40.test(source.commit)) throw new Error(`lock: bad commit for ${source.repo}`);
    const prefix = `https://raw.githubusercontent.com/${source.repo}/${source.commit}/`;
    for (const file of source.files) {
      if (!file.url.startsWith(prefix)) throw new Error(`lock: ${file.url} is not under ${prefix}`);
      if (!/^[A-Za-z0-9._-]+\.json$/.test(file.path)) throw new Error(`lock: bad local path ${file.path}`);
      if (names.has(file.path)) throw new Error(`lock: duplicate local path ${file.path}`);
      if (!HEX64.test(file.sha256)) throw new Error(`lock: bad sha256 for ${file.path}`);
      names.add(file.path);
    }
  }
  return lock.sources.flatMap((source) => source.files);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function checkSha(bytes, file, where) {
  const actual = sha256(bytes);
  if (actual !== file.sha256) {
    throw new Error(`sha256 mismatch for ${where} (${file.url}): lock ${file.sha256}, got ${actual}`);
  }
}

// Ensures every schema in the lock exists in .cache/schema/ and matches its pinned sha256.
// Returns the local path of each schema keyed by its file name.
export async function ensureSchemas() {
  const files = await loadLock();
  await mkdir(schemaDir, { recursive: true });
  const paths = {};
  let fetched = 0;
  for (const file of files) {
    const dest = join(schemaDir, file.path);
    paths[file.path] = dest;
    if (existsSync(dest)) {
      checkSha(await readFile(dest), file, dest);
      continue;
    }
    const res = await fetch(file.url, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${file.url}`);
    const bytes = Buffer.from(await res.arrayBuffer());
    checkSha(bytes, file, file.path);
    await writeFile(dest, bytes);
    fetched++;
  }
  return { paths, fetched, total: files.length };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { fetched, total } = await ensureSchemas();
  console.log(`schemas: ${total} verified (${fetched} fetched) in ${schemaDir}`);
}
