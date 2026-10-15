// Shading and face checks: smooth NORMAL, per-face material slots, winding of negative-determinant copies.
// Expectations come from spec.mjs only.
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { openHarness } from './support/browser.mjs';
import { api, fixtureUrl, FILES, MESHES, TOL, toGltf, triples, readGlbJson } from './support/fixture.mjs';
import { readGlb, smoothNormalRecords } from './support/glb.mjs';
import { EYES, GRAPH_MATERIALS, HEAD_MESH, RIBBON } from '../fixture/spec.mjs';

const M = 'shading';
// Normal tolerance (per component, unit vectors). Lead decision; not part of spec.mjs TOL.
const NORMAL_TOL = 1e-4;

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
const angleDeg = (a, b) => THREE.MathUtils.radToDeg(Math.acos(Math.min(1, Math.max(-1, a.dot(b)))));

describe('smooth NORMAL export (shade_smooth)', () => {
  it('head_mesh NORMAL attribute has one unit vector per glTF vertex', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const positions = triples(await potter.vertexPositions(M, MESHES.head));
    const normals = triples(await potter.restNormals(M, MESHES.head));
    assert.equal(normals.length, positions.length, 'NORMAL count differs from POSITION count');
    normals.forEach((n, i) => {
      assert.ok(Math.abs(vec(n).length() - 1) <= TOL.weight, `vertex ${i}: normal length ${vec(n).length()}`);
    });
  });

  it('head_mesh vertices lie on the spec sphere (HEAD_MESH centre, radius) within TOL.position', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const centre = vec(toGltf(HEAD_MESH.center));
    triples(await potter.vertexPositions(M, MESHES.head)).forEach((p, i) => {
      const r = vec(p).distanceTo(centre);
      assert.ok(Math.abs(r - HEAD_MESH.radius) <= TOL.position, `vertex ${i}: radius ${r} vs spec ${HEAD_MESH.radius}`);
    });
  });

  it('head_mesh NORMAL equals the corner-angle-weighted face normals of the exported glTF (position-merged, one fan per mesh)', async (t) => {
    const glb = await readGlb(FILES.v1Full);
    const meshIndex = glb.json.nodes.find((n) => n.name === MESHES.head).mesh;
    const { records, degenerate, unreferenced, primitives, vertices } = smoothNormalRecords(glb, meshIndex);
    t.diagnostic(`${primitives} primitives share one fan; ${vertices} glTF vertices, ${records.length} referenced, ${unreferenced} unreferenced, ${degenerate} degenerate triangles`);
    const badLength = records.filter((r) => Math.abs(r.exportedLen - 1) > TOL.weight);
    const bad = records.filter((r) => r.maxComp > NORMAL_TOL);
    const worst = [...records].sort((a, b) => b.angleDeg - a.angleDeg)[0];
    t.diagnostic(`checked ${records.length} vertices; mismatches ${bad.length}; max angle ${worst.angleDeg.toFixed(4)} deg at vertex ${worst.vertex}`);
    assert.equal(badLength.length, 0, `non-unit NORMAL at ${badLength.length} vertices, first ${badLength.slice(0, 5).map((r) => r.vertex)}`);
    const list = bad.slice(0, 8).map((r) => `v${r.vertex}: ${r.angleDeg.toFixed(4)} deg (exported [${r.exported.map((x) => x.toFixed(6))}] expected [${r.expected.map((x) => x.toFixed(6))}])`);
    assert.equal(bad.length, 0, `${bad.length} of ${records.length} NORMAL vertices differ from corner-angle expectation (tol ${NORMAL_TOL}):\n  ${list.join('\n  ')}`);
  });
});

describe('per-face material slots (Face.material_index)', () => {
  it('faces whose centroid lies within the eye radius are in the eye primitive, all others in skin', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const primitives = await potter.primitiveTriangles(M, MESHES.head);
    const eyes = [EYES.l, EYES.r].map((e) => ({ c: vec(toGltf(e.center)), r: e.radius }));
    const eyeCount = [0, 0];
    for (const prim of primitives) {
      triples(prim.centroids).forEach((c, t) => {
        const hit = eyes.findIndex(({ c: ec, r }) => vec(c).distanceTo(ec) <= r);
        const expected = hit >= 0 ? 'eye' : 'skin';
        if (hit >= 0) eyeCount[hit]++;
        assert.equal(prim.material, expected, `head_mesh triangle ${t} centroid [${c}] in material ${prim.material}, expected ${expected}`);
      });
    }
    assert.ok(eyeCount[0] > 0 && eyeCount[1] > 0, `no eye faces found (counts ${eyeCount})`);
  });
});

describe('winding of the negative-determinant copy', () => {
  it('ribbon_mirror triangles face the mirror of the ribbon normal (viewer rule: sign(det) times raw cross)', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const ribbonNormals = (await potter.primitiveTriangles(M, MESHES.ribbon)).flatMap((p) => triples(p.normals));
    assert.ok(ribbonNormals.length > 0, 'ribbon has no triangles');
    const n = ribbonNormals.reduce((acc, v) => acc.add(vec(v)), new THREE.Vector3()).normalize();
    const expected = n.clone().multiply(vec(RIBBON.mirrorScale)).normalize();
    const mirror = (await potter.primitiveTriangles(M, MESHES.ribbonMirror)).flatMap((p) => triples(p.normals));
    assert.ok(mirror.length > 0, 'ribbon_mirror has no triangles');
    mirror.forEach((t, i) => {
      const err = angleDeg(vec(t), expected);
      assert.ok(err <= TOL.angleDeg, `ribbon_mirror triangle ${i} faces ${err} deg away from the mirrored ribbon normal`);
    });
  });
});

describe('graph material export', () => {
  it('graph_avatar (shader graph) exports as MASK with a normalTexture in the v1 GLB', async () => {
    const json = await readGlbJson(FILES.v1Full);
    const name = Object.keys(GRAPH_MATERIALS)[0];
    const material = json.materials.find((m) => m.name === name);
    assert.ok(material, `glTF material ${name} is missing`);
    assert.equal(material.alphaMode, 'MASK', `${name} alphaMode`);
    assert.ok(material.normalTexture, `${name} has no normalTexture`);
  });
});
