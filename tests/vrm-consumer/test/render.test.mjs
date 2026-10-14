// Render checks: real three.js WebGL in headless Chromium on fixture/v1-full.vrm.
// Expectations come from fixture/spec.mjs; render geometry and camera mapping live in test/harness/render.mjs.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { FILES, MESHES, MATERIALS, EYES, HEAD_MESH, RIBBON, TUBES } from '../fixture/spec.mjs';
import { openHarness } from './support/browser.mjs';

const SPEC = JSON.parse(JSON.stringify({ FILES, MESHES, MATERIALS, EYES, HEAD_MESH, RIBBON, TUBES }));
const j = (v) => JSON.stringify(v);

let h;
before(async () => {
  h = await openHarness();
});
after(async () => {
  await h?.close();
});

const run = async (t, name) => {
  const r = await h.page.evaluate(
    async ([n, s]) => (await import('/harness/render.mjs')).check(n, s),
    [name, SPEC],
  );
  t.diagnostic(j(r));
  return r;
};

const sameRgb = (a, b) => a !== null && b !== null && a.length === b.length && a.every((v, k) => v === b[k]);

test('1 smooth head shading is more continuous than the flat-normal control', async (t) => {
  const r = await run(t, 'smoothShading');
  assert.ok(r.smoothTV.pairs > 0, `no head pixels in view: ${j(r)}`);
  assert.ok(r.smoothTV.sum < r.flatTV.sum, `smooth TV ${j(r.smoothTV)} must be < flat TV ${j(r.flatTV)}`);
});

test('2 outline: black band of outline width outside the body silhouette, none when width is zero', async (t) => {
  const r = await run(t, 'outline');
  assert.ok(r.outlineMaterials > 0, `no outline materials on body: ${j(r)}`);
  assert.ok(r.innerPixels > 0 && r.innerOutline === r.innerPixels, `inner band outline pixels ${r.innerOutline}/${r.innerPixels}: ${j(r)}`);
  assert.equal(r.beyondOutline, 0, `outline leaks beyond width: ${j(r)}`);
  assert.equal(r.zeroWidthDarkBand, 0, `dark pixels in band with outline width 0: ${j(r)}`);
  assert.ok(r.headRingPixels > 0 && r.headRingDark === 0, `head (no outline) has dark ring pixels: ${j(r)}`);
});

test('3 MASK material: alpha < cutoff leaves holes, alpha >= cutoff is drawn', async (t) => {
  const r = await run(t, 'cutoutFront');
  assert.ok(r.hole > 0, `no hole texels in view: ${j(r)}`);
  assert.equal(r.visibleHole, 0, `holes drawn: ${j(r)}`);
  assert.ok(r.solid > 0 && r.visibleSolid === r.solid, `solid texels visible ${r.visibleSolid}/${r.solid}: ${j(r)}`);
  assert.equal(r.frontSolid, r.solid, `winding: front-facing raycast hits ${r.frontSolid}/${r.solid}: ${j(r)}`);
});

test('4a double-sided BLEND ribbon is visible from the back; alpha 0 texels are not', async (t) => {
  const r = await run(t, 'ribbonBackDoubleSided');
  assert.ok(r.solid > 0 && r.visibleSolid === r.solid, `ribbon back faces visible ${r.visibleSolid}/${r.solid}: ${j(r)}`);
  assert.equal(r.visibleHole, 0, `alpha-0 texels drawn: ${j(r)}`);
});

test('4b single-sided MASK mirror is not visible from the back', async (t) => {
  const r = await run(t, 'cutoutBack');
  assert.ok(r.solid + r.hole > 0, `mirror not in view from the back camera: ${j(r)}`);
  assert.equal(r.frontSolid + r.frontHole, 0, `winding faces away from camera but raycast says front: ${j(r)}`);
  assert.equal(r.visibleTotal, 0, `single-sided back face drawn: ${j(r)}`);
});

test('5 normal map changes only pixels that hit skin faces', async (t) => {
  const r = await run(t, 'normalMap');
  assert.ok(r.skinNormalMap, `skin material has no normalMap: ${j(r)}`);
  assert.equal(r.outsideDiff, 0, `normal map changed pixels outside the head: ${j(r)}`);
  assert.equal(r.diffByMaterial.eye ?? 0, 0, `normal map changed eye-face pixels: ${j(r)}`);
  assert.equal(r.diffByMaterial.none ?? 0, 0, `normal map changed pixels hitting nothing: ${j(r)}`);
  assert.ok((r.diffByMaterial.skin ?? 0) > 0, `normal map changed no skin pixels: ${j(r)}`);
});

test('6 UV animation: mask 0 pixels stay still; mask > 0 pixels move as predicted by speed', async (t) => {
  const r = await run(t, 'uvAnimation');
  assert.equal(r.offsetDelta, r.expectedOffsetDelta, `scroll offset delta ${r.offsetDelta} != speed*dt ${r.expectedOffsetDelta}`);
  assert.equal(r.mismatch, 0, `pixels disagree with predicted motion: ${j(r)}`);
  assert.ok(r.predictedChanged > 0, `no predicted moving pixels: ${j(r)}`);
  assert.ok(r.maskZeroDetermined > 0, `no determined mask-0 pixels: ${j(r)}`);
});

test('7 per-face material: eye faces render with the eye material, skin faces with the skin material', async (t) => {
  const r = await run(t, 'eyeSkinColour');
  assert.ok(r.eyeMaterialPresent, `no material named "eye" on the head; eye faces use: ${j(r.eye_l?.hitMaterial)}`);
  assert.deepEqual(r.eyeLoadedColour.map((v) => +v.toFixed(6)), r.eyeSpecColour, `loaded eye colour != spec`);
  assert.deepEqual(r.skinLoadedColour.map((v) => +v.toFixed(6)), r.skinSpecColour, `loaded skin colour != spec`);
  for (const side of ['eye_l', 'eye_r']) {
    assert.ok(r[side].hitInsideEye, `${side}: sample ray does not hit the eye sphere: ${j(r[side])}`);
    assert.equal(r[side].hitMaterial, 'eye', `${side} front face material: ${j(r[side])}`);
    assert.ok(sameRgb(r[side].actual, r[side].reference), `${side} pixel ${j(r[side].actual)} != eye-material reference ${j(r[side].reference)}`);
  }
  assert.equal(r.skin.hitMaterial, 'skin', `skin sample front face material: ${j(r.skin)}`);
  assert.ok(sameRgb(r.skin.actual, r.skin.reference), `skin pixel ${j(r.skin.actual)} != skin-material reference ${j(r.skin.reference)}`);
});
