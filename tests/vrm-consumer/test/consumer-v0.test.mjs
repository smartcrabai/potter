// Check 10: VRM 0.x (after VRMUtils.rotateVRM0) matches VRM 1.0 for rest positions, a skinning sample and morph
// outputs, within the same tolerances as the reference checks. Poses use the frame-invariant delta transfer.
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
  vrmPoseWorlds,
  captureRestWorlds,
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
  it('rest positions of every reference vertex match in both models', async () => {
    for (const [id, mesh] of Object.entries(reference.meshes)) {
      correspondence(await potter.vertexPositions(V1, id), mesh.restGltf, `VRM 1.0 ${id}`);
      correspondence(await potter.vertexPositions(V0, id), mesh.restGltf, `VRM 0.x ${id}`);
    }
  });

  it('skinning sample (elbow and knee blended vertices) matches VRM 1.0 after the same delta-transferred pose', async () => {
    const rest1 = {};
    const rest0 = {};
    for (const id of Object.keys(reference.blended)) {
      rest1[id] = await potter.vertexPositions(V1, id);
      rest0[id] = await potter.vertexPositions(V0, id);
    }
    const world1 = vrmPoseWorlds('v1Full', reference, await captureRestWorlds(potter, V1, 'v1Full', reference));
    const world0 = vrmPoseWorlds('v0', reference, await captureRestWorlds(potter, V0, 'v0', reference));
    await potter.setWorldPose(V1, world1);
    await potter.setWorldPose(V0, world0);
    let checked = 0;
    for (const [id, entries] of Object.entries(reference.blended)) {
      const posed1 = triples(await potter.vertexPositions(V1, id));
      const posed0 = triples(await potter.vertexPositions(V0, id));
      for (const e of entries) {
        const k1 = correspondence(rest1[id], [e.restGltf], id)[0];
        const k0 = correspondence(rest0[id], [e.restGltf], id)[0];
        assertVecClose(posed0[k0], posed1[k1], TOL.posedPosition, `${id} potter vertex ${e.index}`);
        checked++;
      }
    }
    assert.ok(checked > 0, 'no blended vertices checked');
  });

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
