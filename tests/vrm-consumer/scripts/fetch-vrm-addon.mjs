// Downloads the pinned VRM Add-on for Blender extension zip into .cache/vrm-addon/ and verifies its sha256
// against blender/vrm-addon.lock.json. A cached file that does not match is a hard failure: delete it to refetch.
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
export const addonCacheDir = join(root, '.cache', 'vrm-addon');
const lockPath = join(root, 'blender', 'vrm-addon.lock.json');

export class AddonIntegrityError extends Error {}

export async function loadAddonLock() {
  const lock = JSON.parse(await readFile(lockPath, 'utf8'));
  if (!/^[0-9a-f]{64}$/.test(lock.sha256)) throw new Error(`lock: bad sha256 for ${lock.url}`);
  if (!lock.url.startsWith('https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/')) {
    throw new Error(`lock: ${lock.url} is not a release asset of saturday06/VRM-Addon-for-Blender`);
  }
  return lock;
}

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function verify(bytes, lock, where) {
  const actual = sha256(bytes);
  if (actual !== lock.sha256) {
    throw new AddonIntegrityError(`sha256 mismatch for ${where}: lock ${lock.sha256}, got ${actual}`);
  }
}

// Returns the local path of the verified add-on zip, fetching it when it is not cached.
export async function ensureVrmAddon() {
  const lock = await loadAddonLock();
  const dest = join(addonCacheDir, basename(new URL(lock.url).pathname));
  if (existsSync(dest)) {
    verify(await readFile(dest), lock, dest);
    return dest;
  }
  const res = await fetch(lock.url, { signal: AbortSignal.timeout(120_000) });
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${lock.url}`);
  const bytes = Buffer.from(await res.arrayBuffer());
  verify(bytes, lock, lock.url);
  await mkdir(addonCacheDir, { recursive: true });
  await writeFile(`${dest}.part`, bytes);
  await rename(`${dest}.part`, dest);
  return dest;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(await ensureVrmAddon());
}
