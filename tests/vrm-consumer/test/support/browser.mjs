import { chromium } from 'playwright';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..', '..');
const origin = 'http://potter.test';

const mounts = [
  ['/harness/', path.join(root, 'test', 'harness')],
  ['/node_modules/', path.join(root, 'node_modules')],
  ['/fixture/', path.join(root, '.cache', 'fixture')],
];

const types = {
  '.html': 'text/html; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.glb': 'model/gltf-binary',
  '.vrm': 'model/gltf-binary',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
};

function resolveFile(pathname) {
  for (const [prefix, dir] of mounts) {
    if (!pathname.startsWith(prefix)) continue;
    const file = path.resolve(dir, decodeURIComponent(pathname.slice(prefix.length)));
    if (file !== dir && !file.startsWith(dir + path.sep)) return null;
    return file;
  }
  return null;
}

async function serve(route) {
  const url = new URL(route.request().url());
  const file = resolveFile(url.pathname);
  if (!file || !existsSync(file)) {
    return route.fulfill({ status: 404, body: `not found: ${url.pathname}` });
  }
  const body = await readFile(file);
  return route.fulfill({
    status: 200,
    contentType: types[path.extname(file)] ?? 'application/octet-stream',
    body,
  });
}

// Launches headless Chromium on a fake origin serving the harness, node_modules and fixture.
// Resolves to { page, close }; the harness page is loaded and `window.potterReady` is awaited.
export async function openHarness() {
  // Headless Linux runners have no GPU; SwiftShader is the software WebGL backend. POTTER_SWIFTSHADER=1 forces it elsewhere.
  const swiftshader = process.platform === 'linux' || process.env.POTTER_SWIFTSHADER === '1';
  const args = swiftshader ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : [];
  const browser = await chromium.launch({ headless: true, args });
  try {
    const page = await browser.newPage();
    await page.route(`${origin}/**`, serve);
    await page.goto(`${origin}/harness/index.html`);
    await page.waitForFunction(() => window.potterReady === true || window.potterError, null, { timeout: 60_000 });
    const error = await page.evaluate(() => window.potterError ?? null);
    if (error) throw new Error(`harness failed to start: ${error}`);
    return {
      page,
      async close() {
        await browser.close();
      },
    };
  } catch (err) {
    await browser.close();
    throw err;
  }
}
