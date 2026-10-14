// Reads the exported glTF (GLB/VRM) directly, without three.js, and recomputes expected vertex normals from the
// exported triangles. Used to check the exporter's NORMAL attribute against its own geometry.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { fixturePath } from './fixture.mjs';

const TYPES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
// componentType -> [DataView getter, byte size]
const COMPONENTS = { 5123: ['getUint16', 2], 5125: ['getUint32', 4], 5126: ['getFloat32', 4] };

export async function readGlb(file) {
  const buf = await readFile(fixturePath(file));
  assert.equal(buf.readUInt32LE(0), 0x46546c67, `${file} is not a GLB`);
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
  const binStart = 20 + jsonLen;
  const binLen = buf.readUInt32LE(binStart);
  const bin = buf.subarray(binStart + 8, binStart + 8 + binLen);
  return { json, dv: new DataView(bin.buffer, bin.byteOffset, bin.byteLength) };
}

// Accessor data as a flat Float64Array with n components per element.
export function accessor({ json, dv }, index) {
  const a = json.accessors[index];
  const view = json.bufferViews[a.bufferView];
  const n = TYPES[a.type];
  const [getter, size] = COMPONENTS[a.componentType];
  const stride = view.byteStride ?? n * size;
  const base = (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
  const data = new Float64Array(a.count * n);
  for (let i = 0; i < a.count; i++) {
    for (let c = 0; c < n; c++) data[i * n + c] = dv[getter](base + i * stride + c * size, true);
  }
  return { data, n, count: a.count };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const angleBetween = (u, v) => Math.acos(Math.min(1, Math.max(-1, dot(u, v) / (len(u) * len(v)))));

// Normals for one mesh. All primitives of the mesh must read the same POSITION and NORMAL accessors (asserted):
// the exported NORMAL is then one value per mesh vertex, so the smoothing fan spans every primitive of the mesh.
// Vertices are merged by POSITION (UV/normal seams split them in the file). Each merged vertex sums its incident
// triangles' unit face normals, weighted by the triangle's corner angle at that vertex; the sum is normalized.
// Returns one record per referenced vertex: { vertex, exported, expected, maxComp, angleDeg, exportedLen }, plus
// counts (degenerate triangles, unreferenced vertices, primitives).
export function smoothNormalRecords(glb, meshIndex) {
  const mesh = glb.json.meshes[meshIndex];
  const posIdx = mesh.primitives[0].attributes.POSITION;
  const nrmIdx = mesh.primitives[0].attributes.NORMAL;
  for (const p of mesh.primitives) {
    assert.equal(p.attributes.POSITION, posIdx, 'primitives do not share the POSITION accessor');
    assert.equal(p.attributes.NORMAL, nrmIdx, 'primitives do not share the NORMAL accessor');
  }
  const pos = accessor(glb, posIdx);
  const nrm = accessor(glb, nrmIdx);
  const P = (v) => [pos.data[v * 3], pos.data[v * 3 + 1], pos.data[v * 3 + 2]];
  const key = (v) => P(v).map((c) => Math.round(c * 1e6)).join(',');
  const sums = new Map();
  const referenced = new Set();
  let degenerate = 0;
  for (const prim of mesh.primitives) {
    const idx = prim.indices !== undefined ? accessor(glb, prim.indices).data : Float64Array.from({ length: pos.count }, (_, i) => i);
    for (let t = 0; t < idx.length; t += 3) {
      const v = [idx[t], idx[t + 1], idx[t + 2]];
      v.forEach((x) => referenced.add(x));
      const [a, b, c] = v.map(P);
      const fn = cross(sub(b, a), sub(c, a));
      if (len(fn) < 1e-12) {
        degenerate++;
        continue;
      }
      const unit = scale(fn, 1 / len(fn));
      const corners = [
        [v[0], angleBetween(sub(b, a), sub(c, a))],
        [v[1], angleBetween(sub(c, b), sub(a, b))],
        [v[2], angleBetween(sub(a, c), sub(b, c))],
      ];
      for (const [x, theta] of corners) {
        const k = key(x);
        const acc = sums.get(k) ?? [0, 0, 0];
        sums.set(k, [acc[0] + theta * unit[0], acc[1] + theta * unit[1], acc[2] + theta * unit[2]]);
      }
    }
  }
  const records = [];
  for (const x of referenced) {
    const s = sums.get(key(x));
    const expected = s ? scale(s, 1 / len(s)) : [NaN, NaN, NaN];
    const exported = [nrm.data[x * 3], nrm.data[x * 3 + 1], nrm.data[x * 3 + 2]];
    records.push({
      vertex: x,
      exported,
      expected,
      exportedLen: len(exported),
      maxComp: Math.max(...exported.map((e, i) => Math.abs(e - expected[i]))),
      angleDeg: (angleBetween(exported, expected) * 180) / Math.PI,
    });
  }
  let unreferenced = 0;
  for (let x = 0; x < pos.count; x++) if (!referenced.has(x)) unreferenced++;
  return { records, degenerate, unreferenced, primitives: mesh.primitives.length, vertices: pos.count };
}
