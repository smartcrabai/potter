// Concurrent first fetches of the VRM Add-on zip (parallel test files do this) must each return a complete,
// verified file and leave no temp files. A shared temp path let one fetch rename a zip another was still
// writing; Blender then extracted a truncated archive and the install failed without a clear error.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { ensureVrmAddon } from '../scripts/fetch-vrm-addon.mjs';

test('concurrent first fetches each return a complete, verified zip', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'vrm-addon-fetch-'));
  const bytes = Buffer.alloc(4 << 20, 7);
  const lockFile = path.join(directory, 'lock.json');
  await writeFile(
    lockFile,
    JSON.stringify({
      url: 'https://github.com/saturday06/VRM-Addon-for-Blender/releases/download/test/VRM_Addon_for_Blender-test.zip',
      sha256: createHash('sha256').update(bytes).digest('hex'),
    }),
  );
  const cacheDir = path.join(directory, 'cache');
  const fetchBytes = async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
    return bytes;
  };
  const paths = await Promise.all(
    Array.from({ length: 16 }, () => ensureVrmAddon({ cacheDir, lockFile, fetchBytes })),
  );
  assert.equal(new Set(paths).size, 1, 'every caller gets the same cached file');
  assert.deepEqual(await readFile(paths[0]), bytes);
  assert.deepEqual(
    (await readdir(cacheDir)).filter((name) => name.endsWith('.part')),
    [],
    'no temp file is left behind',
  );
});
