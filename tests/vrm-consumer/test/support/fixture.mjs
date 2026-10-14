// Node-side helpers shared by the consumer checks: fixture paths, reference data, basis conversion,
// vertex correspondence, the frame-invariant pose transfer, and a proxy that forwards calls to window.potter.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FILES, MESHES, HUMANOID, TOL, toGltf } from '../../fixture/spec.mjs';

export { FILES, MESHES, HUMANOID, TOL, toGltf };

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const fixtureDir = path.join(root, '.cache', 'fixture');

export const fixtureUrl = (file) => `/fixture/${file}`;
export const fixturePath = (file) => path.join(fixtureDir, file);

export async function readReference() {
  return JSON.parse(await readFile(fixturePath(FILES.reference), 'utf8'));
}

// Parses the JSON chunk of a GLB/VRM file (extension data the harness does not expose).
export async function readGlbJson(file) {
  const buf = await readFile(fixturePath(file));
  assert.equal(buf.readUInt32LE(0), 0x46546c67, `${file} is not a GLB`);
  const length = buf.readUInt32LE(12);
  assert.equal(buf.readUInt32LE(16), 0x4e4f534a, `${file}: first chunk is not JSON`);
  return JSON.parse(buf.subarray(20, 20 + length).toString('utf8'));
}

// Calls window.potter[method](...args) in the harness page; results are JSON from the page.
export function api(page) {
  return new Proxy(
    {},
    {
      get: (_, method) => (...args) => page.evaluate(([m, a]) => window.potter[m](...a), [method, args]),
    },
  );
}

// Basis change Potter -> glTF: gltf = [x, z, -y]. Row-major input; C is a rotation so C^-1 = C^T.
const C = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
const Ct = C.clone().transpose();

// Column-major Potter world matrix -> column-major glTF world matrix (frame change B M B^-1).
export const toGltfMatrix = (elements) => new THREE.Matrix4().fromArray(elements).premultiply(C).multiply(Ct).toArray();

// Translation column of a column-major matrix.
export const translationOf = (elements) => [elements[12], elements[13], elements[14]];

// Potter humanoid name -> humanoid name for a variant (inverse of the HUMANOID mapping).
export function humanoidNameOf(variant) {
  const out = {};
  for (const [human, potter] of Object.entries(HUMANOID[variant])) out[potter] = human;
  return out;
}

// Frame-invariant pose transfer. Per pose bone: D_potter = posedWorld * restWorld^-1 (Potter frame),
// D_g = B D B^-1 (glTF frame), and the VRM bone's posed world = D_g * restWorld_vrm, where restWorld_vrm is
// the VRM bone's world matrix at load (glTF). No assumption that VRM bone axes equal Potter's.
// restWorldVrm: humanoid name -> column-major glTF world matrix captured before any pose.
// Returns [{ bone: humanoid name, world: column-major glTF world matrix }] for setWorldPose.
export function vrmPoseWorlds(variant, reference, restWorldVrm) {
  const names = humanoidNameOf(variant);
  return reference.pose.bones.map(({ bone, restWorldPotter, posedWorldPotter }) => {
    const rest = new THREE.Matrix4().fromArray(restWorldPotter);
    const posed = new THREE.Matrix4().fromArray(posedWorldPotter);
    const dPotter = posed.multiply(rest.invert());
    const dGltf = C.clone().multiply(dPotter).multiply(Ct);
    const human = names[bone];
    const world = dGltf.multiply(new THREE.Matrix4().fromArray(restWorldVrm[human]));
    return { bone: human, world: world.toArray() };
  });
}

// Captures the VRM rest world matrix of every pose bone (call before any pose is applied).
export async function captureRestWorlds(potter, name, variant, reference) {
  const names = humanoidNameOf(variant);
  const out = {};
  for (const { bone } of reference.pose.bones) out[names[bone]] = await potter.humanoidBoneMatrixWorld(name, names[bone]);
  return out;
}

// flat: harness readback [x,y,z,...] at rest (glTF space). For each reference vertex i (restGltf), the index
// of the glTF vertex at the same rest position within TOL.position.
export function correspondence(flat, restGltf, label) {
  const glTF = triples(flat);
  return restGltf.map((p, i) => {
    let best = -1;
    let bestD = Infinity;
    for (let k = 0; k < glTF.length; k++) {
      const d = Math.max(Math.abs(glTF[k][0] - p[0]), Math.abs(glTF[k][1] - p[1]), Math.abs(glTF[k][2] - p[2]));
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
    assert.ok(bestD <= TOL.position, `${label}: potter vertex ${i} has no glTF vertex within ${TOL.position} (nearest ${bestD})`);
    return best;
  });
}

export function triples(flat) {
  const out = [];
  for (let k = 0; k < flat.length; k += 3) out.push(flat.slice(k, k + 3));
  return out;
}

const maxAbsDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

// Asserts each component of actual matches expected within tol; label names the value in failures.
export function assertVecClose(actual, expected, tol, label) {
  assert.equal(actual.length, expected.length, `${label}: length`);
  const d = maxAbsDiff(actual, expected);
  assert.ok(d <= tol, `${label}: max abs diff ${d} > ${tol}; actual [${actual}] expected [${expected}]`);
}

export function approx(actual, expected, tol, label) {
  assert.ok(Math.abs(actual - expected) <= tol, `${label}: actual ${actual} expected ${expected} (tol ${tol})`);
}

export const saturate = (x) => Math.min(1, Math.max(0, x));
