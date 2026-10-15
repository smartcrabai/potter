// Browser-side render checks for the VRM consumer fixture (v1-full.vrm).
// Real three.js WebGL renderer, fixed white directional light plus ambient, fixed orthographic cameras,
// antialiasing off, 0.5 mm/px. All geometry is in glTF space (Y-up, front = +Z); spec Potter values are
// converted with toGltf = [x, z, -y] (spec.mjs), so the render never mixes frames.
// Camera: orthographic, looks along -Z (front) or +Z (back), up = +Y. Screen right = s*X with s=+1 front, -1 back.
// Pixel (i, j), row 0 at top: x = cx + s*((i+0.5)/S*2-1)*half, y = cy + (1-(j+0.5)/S*2)*half.
import * as THREE from 'three';
import { load, models } from './main.mjs';

const G = ([x, y, z]) => [x, z, -y]; // Potter -> glTF, same as spec.mjs toGltf
const BACKGROUND = [255, 255, 255];
const DARK = 8; // outline colour is black: max channel of an outline pixel <= 8/255

let ctx = null;
const texCache = new Map();

function targets(spec) {
  return {
    head: { center: G(spec.HEAD_MESH.center), half: 0.15, size: 600, side: 'front' },
    body: { center: G(spec.TUBES[spec.MESHES.body].center), half: 0.18, size: 720, side: 'front' },
    mirrorFront: { center: G(spec.RIBBON.mirrorCenter), half: 0.2, size: 800, side: 'front' },
    mirrorBack: { center: G(spec.RIBBON.mirrorCenter), half: 0.2, size: 800, side: 'back' },
    ribbonBack: { center: G(spec.RIBBON.center), half: 0.2, size: 800, side: 'back' },
  };
}

async function init(spec) {
  if (ctx) return ctx;
  if (!models.has('v1')) await load('v1', `/fixture/${spec.FILES.v1Full}`);
  const root = models.get('v1').gltf.scene;
  const renderer = new THREE.WebGLRenderer({ canvas: document.createElement('canvas'), antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setClearColor(new THREE.Color(1, 1, 1), 1);
  const scene = new THREE.Scene();
  scene.add(root);
  const sun = new THREE.DirectionalLight(0xffffff, 1);
  sun.position.set(0.5, 3, 4); // glTF space, from front-up-right
  sun.target.position.set(0, 1.5, 0);
  scene.add(sun, sun.target, new THREE.AmbientLight(0xffffff, 0.3));
  ctx = { spec, root, renderer, scene, ray: new THREE.Raycaster(), targets: targets(spec) };
  return ctx;
}

// ---------------------------------------------------------------------------
// Scene helpers
// ---------------------------------------------------------------------------
const matsOf = (m) => [m.material].flat();

function meshesOf(name) {
  const out = [];
  ctx.root.traverse((o) => {
    if (o.isMesh && (o.name === name || o.parent?.name === name)) out.push(o);
  });
  if (out.length === 0) throw new Error(`no mesh "${name}"`);
  return out;
}

function show(names) {
  ctx.root.traverse((o) => {
    if (o.isMesh) o.visible = names.some((n) => o.name === n || o.parent?.name === n);
  });
}

function materialNamed(meshes, name) {
  const mat = findMaterial(meshes, name);
  if (!mat) throw new Error(`no material "${name}" on ${meshes.map((m) => m.name).join(',')}`);
  return mat;
}

function findMaterial(meshes, name) {
  for (const m of meshes) for (const mat of matsOf(m)) if (mat.name === name) return mat;
  return null;
}

function outlineMaterials() {
  const out = new Set();
  ctx.root.traverse((o) => {
    if (o.isMesh) for (const mat of matsOf(o)) if (mat.isOutline) out.add(mat);
  });
  return [...out];
}

function worldVertices(mesh) {
  const position = mesh.geometry.attributes.position;
  const v = new THREE.Vector3();
  const out = [];
  if (mesh.isSkinnedMesh) mesh.skeleton.update();
  for (let i = 0; i < position.count; i++) {
    v.fromBufferAttribute(position, i);
    if (mesh.isSkinnedMesh) mesh.applyBoneTransform(i, v);
    out.push(v.clone().applyMatrix4(mesh.matrixWorld));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Camera, render, pixel mapping (glTF x, y on screen; z is depth)
// ---------------------------------------------------------------------------
function camera(name) {
  const c = ctx.targets[name];
  const s = c.side === 'front' ? 1 : -1;
  const cam = new THREE.OrthographicCamera(-c.half, c.half, c.half, -c.half, 0.1, 20);
  cam.position.set(c.center[0], c.center[1], c.center[2] + (s > 0 ? 5 : -5));
  cam.lookAt(c.center[0], c.center[1], c.center[2]);
  cam.updateMatrixWorld(true);
  return { cam, c, s, S: c.size };
}

function render(name) {
  const R = camera(name);
  const { renderer, scene } = ctx;
  renderer.setSize(R.S, R.S, false);
  renderer.render(scene, R.cam);
  const gl = renderer.getContext();
  const raw = new Uint8Array(R.S * R.S * 4);
  gl.readPixels(0, 0, R.S, R.S, gl.RGBA, gl.UNSIGNED_BYTE, raw);
  const data = new Uint8Array(R.S * R.S * 4);
  const row = R.S * 4;
  for (let j = 0; j < R.S; j++) data.set(raw.subarray((R.S - 1 - j) * row, (R.S - j) * row), j * row);
  return { ...R, data };
}

const pixelSize = (R) => (2 * R.c.half) / R.S;
const worldOfPixel = (R, i, j) => [
  R.c.center[0] + R.s * ((i + 0.5) / R.S * 2 - 1) * R.c.half,
  R.c.center[1] + (1 - (j + 0.5) / R.S * 2) * R.c.half,
];
const pixelOfWorld = (R, x, y) => [
  Math.floor(((R.s * (x - R.c.center[0])) / R.c.half + 1) / 2 * R.S),
  Math.floor(((1 - (y - R.c.center[1]) / R.c.half) / 2) * R.S),
];
const rgbaAt = (R, i, j) => {
  const o = (j * R.S + i) * 4;
  return [R.data[o], R.data[o + 1], R.data[o + 2], R.data[o + 3]];
};
const isBackground = (R, i, j) => {
  const p = rgbaAt(R, i, j);
  return p[0] === BACKGROUND[0] && p[1] === BACKGROUND[1] && p[2] === BACKGROUND[2];
};
const differs = (A, B, i, j) => {
  const o = (j * A.S + i) * 4;
  return A.data[o] !== B.data[o] || A.data[o + 1] !== B.data[o + 1] || A.data[o + 2] !== B.data[o + 2] || A.data[o + 3] !== B.data[o + 3];
};
const isDark = (R, i, j) => {
  const p = rgbaAt(R, i, j);
  return Math.max(p[0], p[1], p[2]) <= DARK;
};
const srgbToLinear = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);

// Raycast one pixel centre. 'front' = each material's own side; 'any' = both sides.
function pick(R, meshes, i, j, sides) {
  ctx.ray.setFromCamera(new THREE.Vector2((i + 0.5) / R.S * 2 - 1, (1 - (j + 0.5) / R.S * 2)), R.cam);
  const saved = [];
  if (sides === 'any') {
    for (const m of meshes) for (const mat of matsOf(m)) {
      saved.push([mat, mat.side]);
      mat.side = THREE.DoubleSide;
    }
  }
  const hit = ctx.ray.intersectObjects(meshes, false)[0];
  for (const [mat, side] of saved) mat.side = side;
  if (!hit) return null;
  return { uv: hit.uv, point: hit.point, mat: matsOf(hit.object)[hit.face.materialIndex] };
}

function boundsPx(R, meshes) {
  const box = new THREE.Box3();
  for (const m of meshes) box.expandByObject(m);
  let i0 = Infinity, j0 = Infinity, i1 = -Infinity, j1 = -Infinity;
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) {
    const [i, j] = pixelOfWorld(R, x, y);
    i0 = Math.min(i0, i); i1 = Math.max(i1, i);
    j0 = Math.min(j0, j); j1 = Math.max(j1, j);
  }
  return [Math.max(0, i0), Math.max(0, j0), Math.min(R.S - 1, i1), Math.min(R.S - 1, j1)];
}

// Decoded texture pixels (raw bytes, flipY=false so row 0 is glTF v=0).
function decode(tex) {
  if (!tex) return null;
  if (!texCache.has(tex.uuid)) {
    const img = tex.image;
    const canvas = new OffscreenCanvas(img.width, img.height);
    const g = canvas.getContext('2d');
    g.drawImage(img, 0, 0);
    texCache.set(tex.uuid, { w: img.width, h: img.height, d: g.getImageData(0, 0, img.width, img.height).data });
  }
  return texCache.get(tex.uuid);
}

// Byte offset of the nearest texel (REPEAT wrap, glTF v=0 = data row 0).
const nearestOffset = (T, u, v) => {
  const x = Math.floor((((u % 1) + 1) % 1) * T.w);
  const y = Math.floor((((v % 1) + 1) % 1) * T.h);
  return (y * T.w + x) * 4;
};

// REPEAT-wrapped 3x3 neighbourhood; returns the key if all agree, else null (used for mask edges only).
function uniformAt(T, u, v, key) {
  const x0 = Math.floor((((u % 1) + 1) % 1) * T.w);
  const y0 = Math.floor((((v % 1) + 1) % 1) * T.h);
  let out;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const x = (x0 + dx + T.w) % T.w;
    const y = (y0 + dy + T.h) % T.h;
    const k = key(T.d, (y * T.w + x) * 4);
    if (out === undefined) out = k;
    else if (k !== out) return null;
  }
  return out;
}
const keyRGBA = (d, o) => `${d[o]},${d[o + 1]},${d[o + 2]},${d[o + 3]}`;
const keyMaskB = (d, o) => d[o + 2] / 255;
// Bilinear alpha (GPU linear magnification, REPEAT wrap, texel centres at (k+0.5)/size), in 0..1.
const bilinearAlpha = (T, u, v) => {
  const x = u * T.w - 0.5, y = v * T.h - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const a = (xx, yy) => T.d[(((yy % T.h) + T.h) % T.h * T.w + ((xx % T.w) + T.w) % T.w) * 4 + 3] / 255;
  return (1 - fy) * ((1 - fx) * a(x0, y0) + fx * a(x0 + 1, y0)) + fy * ((1 - fx) * a(x0, y0 + 1) + fx * a(x0 + 1, y0 + 1));
};
// GPU bilinear weights are 8-bit, so alpha is resolved only to 1/255. Within that band of the cutoff the
// class is ambiguous (0): reported as `ambiguous`, not asserted. Outside it: 1 = solid, -1 = hole.
const ALPHA_LSB = 1 / 255;
const alphaClass = (cut) => (T, u, v) => {
  const a = bilinearAlpha(T, u, v);
  return a >= cut + ALPHA_LSB ? 1 : a < cut - ALPHA_LSB ? -1 : 0;
};
// BLEND: alpha 0 is exact in the texture, so the class is exact: 1 if alpha > 0, else -1.
const alphaPositive = (T, u, v) => (bilinearAlpha(T, u, v) > 0 ? 1 : -1);

// ---------------------------------------------------------------------------
// Check 1: smooth head shading continuity vs flat-normal control.
// Region: head disk (projected) minus 2 px margin, minus eye disks (+2 px).
// Metric: sum of |luminance difference| over horizontal/vertical neighbour pairs in the region.
// ---------------------------------------------------------------------------
function totalVariation(R, inside) {
  const L = (i, j) => {
    const p = rgbaAt(R, i, j);
    return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
  };
  let sum = 0, pairs = 0, max = 0;
  for (let j = 0; j < R.S; j++) for (let i = 0; i < R.S; i++) {
    if (!inside(i, j)) continue;
    for (const [di, dj] of [[1, 0], [0, 1]]) {
      const ni = i + di, nj = j + dj;
      if (ni >= R.S || nj >= R.S || !inside(ni, nj)) continue;
      const diff = Math.abs(L(i, j) - L(ni, nj));
      sum += diff; pairs++; max = Math.max(max, diff);
    }
  }
  return { sum, pairs, mean: sum / pairs, max };
}

// Head disk and eye disks, projected along the view axis (orthographic: projection of a sphere is a disk).
function headInside(R, spec) {
  const pad = 2 * pixelSize(R);
  const hc = G(spec.HEAD_MESH.center);
  const r = spec.HEAD_MESH.radius;
  const eyes = Object.values(spec.EYES).map((e) => ({ c: G(e.center), r: e.radius }));
  return (i, j) => {
    const [x, y] = worldOfPixel(R, i, j);
    if (Math.hypot(x - hc[0], y - hc[1]) >= r - pad) return false;
    return eyes.every((e) => Math.hypot(x - e.c[0], y - e.c[1]) > e.r + pad);
  };
}

function checkSmoothShading(spec) {
  show([spec.MESHES.head]);
  const meshes = meshesOf(spec.MESHES.head);
  const smooth = render('head');
  const inside = headInside(smooth, spec);
  const smoothTV = totalVariation(smooth, inside);
  const originals = meshes.map((m) => m.geometry);
  let flatTV;
  try {
    for (const m of meshes) {
      const flat = m.geometry.index ? m.geometry.toNonIndexed() : m.geometry.clone();
      flat.computeVertexNormals();
      flat.groups = m.geometry.groups.map((g) => ({ ...g }));
      m.geometry = flat;
    }
    flatTV = totalVariation(render('head'), inside);
  } finally {
    meshes.forEach((m, k) => {
      m.geometry.dispose();
      m.geometry = originals[k];
    });
  }
  return { smoothTV, flatTV };
}

// ---------------------------------------------------------------------------
// Check 2: outline pixels. Body (cloth, outlineWidth, worldCoordinates) vs the same render with outline
// width 0. Silhouette xs = max |x| of posed body vertices; band is xs < |x| < xs + width (glTF X).
// ---------------------------------------------------------------------------
function checkOutline(spec) {
  const width = spec.MATERIALS.cloth.outlineWidth;
  const rowsPotterZ = [1.0, 1.35]; // straight body section (caps at potter z 0.85 / 1.5), glTF y == potter z
  show([spec.MESHES.body]);
  const body = meshesOf(spec.MESHES.body);
  const xs = Math.max(...body.flatMap((m) => worldVertices(m)).map((v) => Math.abs(v.x)));
  const on = render('body');
  const outlines = outlineMaterials();
  const saved = outlines.map((m) => m.outlineWidthFactor);
  let off;
  try {
    outlines.forEach((m) => (m.outlineWidthFactor = 0));
    off = render('body');
  } finally {
    outlines.forEach((m, k) => (m.outlineWidthFactor = saved[k]));
  }
  const px = pixelSize(on);
  const r = { xs, width, outlineMaterials: outlines.length, innerPixels: 0, innerOutline: 0, bandOutline: 0, beyondOutline: 0, zeroWidthDarkBand: 0 };
  for (let j = 0; j < on.S; j++) {
    const y = worldOfPixel(on, 0, j)[1];
    if (y < rowsPotterZ[0] || y > rowsPotterZ[1]) continue;
    for (let i = 0; i < on.S; i++) {
      const d = Math.abs(worldOfPixel(on, i, j)[0]);
      const diff = differs(on, off, i, j);
      if (d > xs && d < xs + 0.9 * width) {
        r.innerPixels++;
        if (diff && isDark(on, i, j)) r.innerOutline++;
      }
      if (d > xs && d <= xs + width) {
        if (diff && isDark(on, i, j)) r.bandOutline++;
        if (isDark(off, i, j)) r.zeroWidthDarkBand++;
      }
      if (d > xs + width + 2 * px && diff) r.beyondOutline++;
    }
  }
  // Head has no outline material: no dark pixels in a 4 mm ring just outside its silhouette.
  show([spec.MESHES.head]);
  const head = render('head');
  const hc = G(spec.HEAD_MESH.center);
  let headRingDark = 0, headRingPixels = 0;
  for (let j = 0; j < head.S; j++) for (let i = 0; i < head.S; i++) {
    const [x, y] = worldOfPixel(head, i, j);
    const d = Math.hypot(x - hc[0], y - hc[1]) - spec.HEAD_MESH.radius;
    if (d > 0 && d < 0.004) {
      headRingPixels++;
      if (isDark(head, i, j)) headRingDark++;
    }
  }
  return { ...r, headRingPixels, headRingDark };
}

// ---------------------------------------------------------------------------
// Checks 3, 4, 8: visibility of the ribbon planes against their texel class.
// Texel class is the GPU's: bilinear alpha of the surface hit by the 'any' raycast, against the material's
// alphaCutoff (MASK) or zero (BLEND). Nearest-texel classification is not used for alpha.
// 'front' raycast uses the material's real side, so it hits only where winding faces the camera.
// ---------------------------------------------------------------------------
function visibilityReport(R, meshes, solid) {
  const [i0, j0, i1, j1] = boundsPx(R, meshes);
  const out = { solid: 0, hole: 0, ambiguous: 0, frontSolid: 0, frontHole: 0, visibleSolid: 0, visibleHole: 0, visibleTotal: 0 };
  for (let j = 0; j < R.S; j++) for (let i = 0; i < R.S; i++) {
    if (!isBackground(R, i, j)) out.visibleTotal++;
  }
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
    const any = pick(R, meshes, i, j, 'any');
    if (!any) continue;
    const front = pick(R, meshes, i, j, 'front') !== null;
    const visible = !isBackground(R, i, j);
    const T = decode(any.mat.map);
    if (!T) throw new Error(`no texture on ${any.mat.name}`);
    const cls = solid(T, any.uv.x, any.uv.y);
    if (cls === 0) { out.ambiguous++; continue; }
    if (cls > 0) {
      out.solid++;
      if (front) out.frontSolid++;
      if (visible) out.visibleSolid++;
    } else {
      out.hole++;
      if (front) out.frontHole++;
      if (visible) out.visibleHole++;
    }
  }
  return out;
}

function checkCutout(spec, cam) {
  const cut = spec.MATERIALS.cutout.alphaCutoff;
  show([spec.MESHES.ribbonMirror]);
  const meshes = meshesOf(spec.MESHES.ribbonMirror);
  const R = render(cam);
  return { cutoff: cut, ...visibilityReport(R, meshes, alphaClass(cut)) };
}

function checkRibbonDoubleSided(spec) {
  show([spec.MESHES.ribbon]);
  const R = render('ribbonBack');
  return visibilityReport(R, meshesOf(spec.MESHES.ribbon), alphaPositive);
}

// ---------------------------------------------------------------------------
// Check 5: removing the skin normal map changes only pixels that hit skin faces.
// Every differing pixel is classified by the material of the front-most surface it hits.
// ---------------------------------------------------------------------------
function checkNormalMap(spec) {
  show([spec.MESHES.head]);
  const meshes = meshesOf(spec.MESHES.head);
  const skin = materialNamed(meshes, 'skin');
  const withMap = render('head');
  const normalMap = skin.normalMap;
  let without;
  try {
    skin.normalMap = null;
    skin.needsUpdate = true;
    without = render('head');
  } finally {
    skin.normalMap = normalMap;
    skin.needsUpdate = true;
  }
  const hc = G(spec.HEAD_MESH.center);
  const r = { skinNormalMap: !!normalMap, diffPixels: 0, outsideDiff: 0, diffByMaterial: {} };
  for (let j = 0; j < withMap.S; j++) for (let i = 0; i < withMap.S; i++) {
    if (!differs(withMap, without, i, j)) continue;
    r.diffPixels++;
    const [x, y] = worldOfPixel(withMap, i, j);
    if (Math.hypot(x - hc[0], y - hc[1]) >= spec.HEAD_MESH.radius + pixelSize(withMap)) { r.outsideDiff++; continue; }
    const name = pick(withMap, meshes, i, j, 'front')?.mat.name ?? 'none';
    r.diffByMaterial[name] = (r.diffByMaterial[name] ?? 0) + 1;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Check 6: UV animation mask gates scrolling. Two times, same camera.
// MToon samples the base texture at uv + offset * B(mask at uv), with rotation phase 0 (spec sets no rotation).
// Prediction per pixel: changed iff nearest texel at (u0 + off0*m) differs from nearest texel at (u0 + off1*m).
// ---------------------------------------------------------------------------
function checkUvAnimation(spec) {
  const cloth = spec.MATERIALS.cloth;
  show([spec.MESHES.body]);
  const body = meshesOf(spec.MESHES.body);
  const surface = materialNamed(body, 'cloth');
  if (!surface.map) throw new Error('cloth has no base texture: UV scroll is not observable');
  // Unique MToon materials (a material shared across primitives must update once per frame).
  const mtoon = new Set();
  ctx.root.traverse((o) => { if (o.isMesh) for (const mat of matsOf(o)) if (mat.isMToonMaterial) mtoon.add(mat); });
  const DT = 1;
  for (const m of mtoon) m.update(0);
  const off0 = surface.uvAnimationScrollXOffset;
  const A0 = render('body');
  for (const m of mtoon) m.update(DT);
  const off1 = surface.uvAnimationScrollXOffset;
  const A1 = render('body');
  const mask = decode(surface.uvAnimationMaskTexture);
  const map = decode(surface.map);
  const r = { speed: cloth.uvAnimationScrollX, dt: DT, offsetDelta: off1 - off0, expectedOffsetDelta: cloth.uvAnimationScrollX * DT,
    determined: 0, predictedChanged: 0, predictedUnchanged: 0, observedChanged: 0, mismatch: 0, undetermined: 0, maskZeroDetermined: 0 };
  const colour = (u, v) => keyRGBA(map.d, nearestOffset(map, u, v));
  const [i0, j0, i1, j1] = boundsPx(A0, body);
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
    const hit = pick(A0, body, i, j, 'front');
    if (!hit) continue;
    const [u0, v0] = [hit.uv.x, hit.uv.y];
    const m = uniformAt(mask, u0, v0, keyMaskB);
    if (m === null) { r.undetermined++; continue; }
    const observed = differs(A0, A1, i, j);
    if (m === 0) {
      r.maskZeroDetermined++;
      if (observed) r.mismatch++;
      continue;
    }
    const c0 = colour(u0 + off0 * m, v0);
    const c1 = colour(u0 + off1 * m, v0);
    const predicted = c0 !== c1;
    r.determined++;
    if (predicted) r.predictedChanged++; else r.predictedUnchanged++;
    if (observed) r.observedChanged++;
    if (predicted !== observed) r.mismatch++;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Check 7: per-face material colour, independent of MToon's lighting maths.
// Reference renders: the whole head with the eye material (eye sample), and the whole head with the
// skin material (skin sample). A correctly split head shows identical pixels at the eye and skin samples.
// Observed chroma ratios vs the spec base colour are reported but not asserted: the renderer's lit colour
// is not a scalar multiple of base colour (see report).
// ---------------------------------------------------------------------------
function checkEyeSkinColour(spec) {
  show([spec.MESHES.head]);
  const meshes = meshesOf(spec.MESHES.head);
  const eyeMat = findMaterial(meshes, 'eye');
  const skinMat = materialNamed(meshes, 'skin');
  const R = render('head');
  const ratios = (c) => [c[1] / c[0], c[2] / c[0]];
  const linear = (rgb) => rgb.map((v) => srgbToLinear(v / 255));
  const eyeSamples = Object.entries(spec.EYES).map(([side, e]) => {
    const ec = G(e.center);
    const [i, j] = pixelOfWorld(R, ec[0], ec[1]);
    const hit = pick(R, meshes, i, j, 'front');
    return { side, i, j, center: ec, radius: e.radius, hitMaterial: hit?.mat.name ?? null, hitPoint: hit?.point.toArray() ?? null };
  });
  const [ci, cj] = pixelOfWorld(R, G(spec.HEAD_MESH.center)[0], G(spec.HEAD_MESH.center)[1] - 0.08);
  const skinHit = pick(R, meshes, ci, cj, 'front');
  const skinSample = { i: ci, j: cj, hitMaterial: skinHit?.mat.name ?? null };
  const rgbAt = (Rr, i, j) => rgbaAt(Rr, i, j).slice(0, 3);
  // Set every head primitive to one material, render, read the sample pixels, restore.
  const withAll = (mat, read) => {
    const saved = meshes.map((m) => m.material);
    try {
      meshes.forEach((m) => (m.material = mat));
      return read(render('head'));
    } finally {
      meshes.forEach((m, k) => (m.material = saved[k]));
    }
  };
  const out = {
    eyeMaterialPresent: !!eyeMat,
    eyeLoadedColour: eyeMat ? eyeMat.color.toArray() : null,
    skinLoadedColour: skinMat.color.toArray(),
    eyeSpecColour: spec.MATERIALS.eye.baseColor.slice(0, 3),
    skinSpecColour: spec.MATERIALS.skin.baseColor.slice(0, 3),
  };
  if (eyeMat) {
    const refEye = withAll(eyeMat, (Rr) => eyeSamples.map((s) => rgbAt(Rr, s.i, s.j)));
    for (const [k, s] of eyeSamples.entries()) {
      const actual = rgbAt(R, s.i, s.j);
      out[`eye_${s.side}`] = {
        hitMaterial: s.hitMaterial,
        hitInsideEye: !!s.hitPoint && Math.hypot(s.hitPoint[0] - s.center[0], s.hitPoint[1] - s.center[1], s.hitPoint[2] - s.center[2]) < s.radius * 1.5,
        actual,
        reference: refEye[k],
        observedRatios: ratios(linear(actual)),
        specRatios: ratios(linear(spec.MATERIALS.eye.baseColor.slice(0, 3))),
      };
    }
  } else {
    for (const s of eyeSamples) out[`eye_${s.side}`] = { hitMaterial: s.hitMaterial, hitInsideEye: false, actual: rgbAt(R, s.i, s.j), reference: null };
  }
  const refSkin = withAll(skinMat, (Rr) => rgbAt(Rr, skinSample.i, skinSample.j));
  const actualSkin = rgbAt(R, skinSample.i, skinSample.j);
  out.skin = {
    hitMaterial: skinSample.hitMaterial,
    actual: actualSkin,
    reference: refSkin,
    observedRatios: ratios(linear(actualSkin)),
    specRatios: ratios(linear(spec.MATERIALS.skin.baseColor.slice(0, 3))),
  };
  return out;
}

// ---------------------------------------------------------------------------
// Entry point (called from Node via page.evaluate).
// ---------------------------------------------------------------------------
export async function check(name, spec) {
  await init(spec);
  switch (name) {
    case 'smoothShading': return checkSmoothShading(spec);
    case 'outline': return checkOutline(spec);
    case 'cutoutFront': return checkCutout(spec, 'mirrorFront');
    case 'cutoutBack': return checkCutout(spec, 'mirrorBack');
    case 'ribbonBackDoubleSided': return checkRibbonDoubleSided(spec);
    case 'normalMap': return checkNormalMap(spec);
    case 'uvAnimation': return checkUvAnimation(spec);
    case 'eyeSkinColour': return checkEyeSkinColour(spec);
    default: throw new Error(`unknown check ${name}`);
  }
}
