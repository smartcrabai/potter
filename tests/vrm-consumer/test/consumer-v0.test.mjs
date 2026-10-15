// Check 10: VRM 0.x (after VRMUtils.rotateVRM0) matches VRM 1.0 morph outputs; the thumbnail decodes at spec size.
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { openHarness } from './support/browser.mjs';
import {
  api,
  readReference,
  readGlbJson,
  fixtureUrl,
  FILES,
  MESHES,
  TOL,
  correspondence,
  triples,
  assertVecClose,
  approx,
} from './support/fixture.mjs';
import { EXPRESSIONS, IMAGES, META } from '../fixture/spec.mjs';

const reference = await readReference();
const V1 = 'parity1';
const V0 = 'parity0';

let h;
let potter;
before(async () => {
  h = await openHarness();
  potter = api(h.page);
  await potter.load(V1, fixtureUrl(FILES.v1Full));
  await potter.load(V0, fixtureUrl(FILES.v0));
});
after(async () => {
  await h?.close();
});

describe('VRM 0.x vs VRM 1.0 parity', () => {
  for (const [expr, morph] of [
    ['blink', 'blink'],
    ['aa', 'aa'],
    ['lookLeft', 'look_left'],
  ]) {
    it(`morph ${morph} at weight 0.5 matches VRM 1.0`, async () => {
      assert.deepEqual(EXPRESSIONS[expr].morphs, [[morph, 1]]);
      // Earlier tests leave a pose and expression weights on the shared models; restore rest first.
      await potter.resetPose(V1);
      await potter.resetPose(V0);
      for (const m of [V1, V0]) {
        await potter.setExpression(m, 'blink', 0);
        await potter.setExpression(m, 'aa', 0);
        await potter.setExpression(m, 'lookLeft', 0);
        await potter.update(m, 0);
      }
      const restGltf = reference.meshes[MESHES.head].restGltf;
      const rest1 = await potter.vertexPositions(V1, MESHES.head);
      const rest0 = await potter.vertexPositions(V0, MESHES.head);
      await potter.setExpression(V1, expr, 0.5);
      await potter.setExpression(V0, expr, 0.5);
      await potter.update(V1, 0);
      await potter.update(V0, 0);
      approx((await potter.morphWeights(V0, MESHES.head))[morph], 0.5, TOL.morph, `VRM 0.x ${morph} influence`);
      const p1 = triples(await potter.vertexPositions(V1, MESHES.head));
      const p0 = triples(await potter.vertexPositions(V0, MESHES.head));
      const k1 = correspondence(rest1, restGltf, `VRM 1.0 ${MESHES.head}`);
      const k0 = correspondence(rest0, restGltf, `VRM 0.x ${MESHES.head}`);
      restGltf.forEach((_, i) => assertVecClose(p0[k0[i]], p1[k1[i]], TOL.morph, `${morph} reference vertex ${i}`));
    });
  }

  it('VRM 0.x thumbnail (meta.texture) decodes at the spec thumbnail size', async () => {
    const json = await readGlbJson(FILES.v0);
    const thumb = json.extensions.VRM.meta.texture;
    const img = (await potter.imageInfo(V0))[thumb];
    assert.ok(img, `no decoded image at meta.texture ${thumb}`);
    assert.deepEqual([img.width, img.height], [IMAGES[META.thumbnail].width, IMAGES[META.thumbnail].height]);
  });
});
