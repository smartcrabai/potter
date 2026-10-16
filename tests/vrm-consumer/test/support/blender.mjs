// Blender side of the consumer harness: finds a Blender 5.2.x (POTTER_BLENDER, then PATH, then the macOS app bundle,
// the same order as tests/common/blender_discovery.rs), installs the pinned VRM Add-on into an isolated user-resources
// directory, and runs blender/probe.py on one .vrm file.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AddonIntegrityError, ensureVrmAddon } from '../../scripts/fetch-vrm-addon.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const probePath = path.join(root, 'blender', 'probe.py');
const PREFIX = 'POTTER_VRM_FACTS ';
const BLENDER_SERIES = '5.2.';

function blenderCandidates() {
  if (process.env.POTTER_BLENDER) return [process.env.POTTER_BLENDER];
  const fromPath = (process.env.PATH ?? '').split(path.delimiter).map((dir) => path.join(dir, 'blender'));
  return [...fromPath, '/Applications/Blender.app/Contents/MacOS/Blender'];
}

// Returns { path, version } for the first usable Blender 5.2.x, or null.
export function discoverBlender() {
  for (const candidate of blenderCandidates()) {
    if (!existsSync(candidate)) continue;
    const r = spawnSync(candidate, ['--version'], { encoding: 'utf8', timeout: 60_000 });
    const m = /^Blender (\d+\.\d+\.\d+)/m.exec(r.stdout ?? '');
    if (r.status === 0 && m && m[1].startsWith(BLENDER_SERIES)) return { path: candidate, version: m[1] };
  }
  return null;
}

// Resolves { blender, userResources } or { reason } when Blender or the add-on cannot be used here.
// Integrity failures (sha256 mismatch) throw; they are never a skip.
export async function setupBlender() {
  const blender = discoverBlender();
  if (!blender) {
    const tried = process.env.POTTER_BLENDER ? `POTTER_BLENDER=${process.env.POTTER_BLENDER}` : 'PATH and the macOS app bundle';
    return { reason: `no usable Blender ${BLENDER_SERIES}x found (tried ${tried})` };
  }
  let zip;
  try {
    zip = await ensureVrmAddon();
  } catch (error) {
    if (error instanceof AddonIntegrityError) throw error;
    return { reason: `VRM Add-on download failed: ${error.message}` };
  }
  // Isolated user resources: the add-on never touches the real Blender config or extensions.
  const userResources = await mkdtemp(path.join(tmpdir(), 'potter-blender-'));
  const env = { ...process.env, BLENDER_USER_RESOURCES: userResources };
  const install = spawnSync(
    blender.path,
    ['-b', '--factory-startup', '--command', 'extension', 'install-file', '-r', 'user_default', '-e', zip],
    { env, encoding: 'utf8', timeout: 300_000 },
  );
  const output = `${install.stdout ?? ''}${install.stderr ?? ''}`;
  // The installed tree is the result. Concurrent Blender installs can exit 0 without printing
  // `STATUS Installed "vrm"` (the line is lost while the tree is complete), so the status line is not required.
  // A real failure (for example a truncated zip) is reported by Blender as an `ERROR` line, also with exit 0.
  const installed = path.join(userResources, 'extensions', 'user_default', 'vrm', 'blender_manifest.toml');
  if (install.status !== 0 || /ERROR/.test(output) || !existsSync(installed)) {
    await rm(userResources, { recursive: true, force: true });
    const errors = output.split('\n').filter((line) => line.includes('ERROR')).join('\n');
    return { reason: `VRM Add-on install failed (exit ${install.status}): ${errors || output.slice(-1000)}` };
  }
  return { blender, userResources };
}

// Imports one .vrm with the add-on. Returns { facts, stderr }; throws when the import is not FINISHED.
// acceptLicensePrompt: only for the VRM 0.x fixture, whose otherPermissionUrl makes the add-on return CANCELLED
// unless its license prompt is accepted. BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION is that switch
// (add-on importer/import_scene.py:166). The prompt itself is still logged and asserted by the test.
export function probeVrm(setup, file, { acceptLicensePrompt = false } = {}) {
  const env = { ...process.env, BLENDER_USER_RESOURCES: setup.userResources };
  delete env.BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION;
  if (acceptLicensePrompt) env.BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION = 'true';
  const r = spawnSync(
    setup.blender.path,
    ['-b', '--factory-startup', '--python-exit-code', '1', '--python', probePath, '--', file],
    { env, encoding: 'utf8', maxBuffer: 256 << 20, timeout: 300_000 },
  );
  const line = (r.stdout ?? '').split('\n').find((l) => l.startsWith(PREFIX));
  if (r.status !== 0 || !line) {
    throw new Error(`Blender probe failed for ${path.basename(file)} (exit ${r.status}):\n${(r.stderr ?? '').slice(-4000)}`);
  }
  return { facts: JSON.parse(line.slice(PREFIX.length)), stderr: r.stderr ?? '' };
}
