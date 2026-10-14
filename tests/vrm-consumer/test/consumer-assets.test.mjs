// Check 8 (bone-parented hat follows head rotation) and check 9 (embedded images decode with spec dimensions).
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { openHarness } from './support/browser.mjs';
import { api, fixtureUrl, readGlbJson, FILES, MESHES, TOL, toGltf, triples, assertVecClose, approx } from './support/fixture.mjs';
import { HAT, IMAGES, META } from '../fixture/spec.mjs';

const M = 'assets';

let h;
let potter;
before(async () => {
  h = await openHarness();
  potter = api(h.page);
});
after(async () => {
  await h?.close();
});

const vec = (a) => new THREE.Vector3().fromArray(a);
const quat = (a) => new THREE.Quaternion().fromArray(a);

describe('check 8: bone-parented hat follows the head', () => {
  it('hat is the spec disc at rest: centre, glTF Y extent = depth, radial extent = radius', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const p = triples(await potter.vertexPositions(M, MESHES.hat));
    const min = [0, 1, 2].map((a) => Math.min(...p.map((v) => v[a])));
    const max = [0, 1, 2].map((a) => Math.max(...p.map((v) => v[a])));
    const centre = toGltf(HAT.center);
    assertVecClose(min.map((v, a) => (v + max[a]) / 2), centre, TOL.position, 'hat centre');
    approx(max[1] - min[1], HAT.depth, TOL.position, 'hat depth (glTF Y)');
    const radial = Math.max(...p.map((v) => Math.hypot(v[0] - centre[0], v[2] - centre[2])));
    approx(radial, HAT.radius, TOL.position, 'hat radius');
  });

  it('hat vertices move with the head pivot rotation', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const headRest = await potter.nodeWorld(M, 'head');
    const restHat = triples(await potter.vertexPositions(M, MESHES.hat));
    const turn = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, 0.5, -0.2));
    await potter.setPose(M, [{ bone: 'head', quaternion: turn.toArray() }]);
    const headPosed = await potter.nodeWorld(M, 'head');
    // The head rotates by R = q_posed * q_rest^-1 about its rest position.
    const rot = quat(headPosed.quaternion).multiply(quat(headRest.quaternion).invert());
    const pivot = vec(headRest.position);
    const expected = restHat.map((v) => vec(headPosed.position).add(vec(v).sub(pivot).applyQuaternion(rot)).toArray());
    const posed = triples(await potter.vertexPositions(M, MESHES.hat));
    expected.forEach((e, i) => assertVecClose(posed[i], e, TOL.posedPosition, `hat vertex ${i}`));
  });
});

describe('check 9: embedded images decode with the spec dimensions', () => {
  const bySize = (a, b) => a[0] - b[0] || a[1] - b[1];

  it('every glTF image decodes from a bufferView with the spec size', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const images = await potter.imageInfo(M);
    const got = images.map((i) => [i.width, i.height]).sort(bySize);
    const want = Object.values(IMAGES)
      .map((i) => [i.width, i.height])
      .sort(bySize);
    assert.deepEqual(got, want);
    for (const img of images) assert.notEqual(img.bufferView, null, `image ${img.index} is not embedded in a bufferView`);
  });

  it('the VRM thumbnail is the spec thumbnail image', async () => {
    const json = await readGlbJson(FILES.v1Full);
    const thumb = json.extensions.VRMC_vrm.meta.thumbnailImage;
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const img = (await potter.imageInfo(M))[thumb];
    assert.ok(img, `no decoded image at thumbnailImage ${thumb}`);
    assert.deepEqual([img.width, img.height], [IMAGES[META.thumbnail].width, IMAGES[META.thumbnail].height]);
  });
});
