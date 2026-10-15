#!/usr/bin/env node
// Builds the VRM consumer fixture through the pot CLI only.
// Outputs (tests/vrm-consumer/.cache/fixture/): v1-full.vrm, v1-expression-lookat.vrm,
// v0.vrm, reference.json. Scenes are built in an OS temp dir; nothing is written
// into the repo except the export outputs. Any pot error aborts with the envelope.
import { spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as S from '../fixture/spec.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = join(ROOT, '.cache', 'fixture');
const POT = process.env.POT ?? resolve(ROOT, '..', '..', 'target', 'debug', 'pot');
const SKIN_SIGMA = 0.1; // metres; weight falloff width
const MAX_BONES = 3;
const MIN_WEIGHT = 1e-3; // bones below this (relative to the nearest) are dropped
const BLEND_MIN = 0.05; // a vertex is "blended" when its second bone weight is at least this
const ALL_VERTICES = { domain: 'vertex', selector: { type: 'position_box', min: [-10, -10, -10], max: [10, 10, 10] } };
const ALL_FACES = { domain: 'face', selector: { type: 'normal_cone', axis: [0, 0, 1], angle: Math.PI } };
const VARIANTS = [
  { key: 'v1Full', format: 'vrm', file: S.FILES.v1Full },
  { key: 'v1LookAt', format: 'vrm', file: S.FILES.v1LookAt },
  { key: 'v0', format: 'vrm0', file: S.FILES.v0 },
];

const revisions = new Map();

function pot(args, stdin) {
  const started = Date.now();
  const r = spawnSync(POT, args, { input: stdin, encoding: 'utf8', maxBuffer: 1 << 30 });
  console.error(`[pot ${args.slice(0, 2).join(' ')}] ${Date.now() - started} ms`);
  let env;
  try {
    env = JSON.parse(r.stdout);
  } catch {
    throw new Error(`pot ${args.join(' ')}: no JSON envelope (exit ${r.status})\n${r.stderr}`);
  }
  if (!env.ok) {
    throw new Error(`pot ${args.join(' ')} failed (exit ${r.status}):\n${JSON.stringify(env, null, 2)}`);
  }
  return env;
}

function apply(dir, operations) {
  const body = JSON.stringify({ schema_version: 1, base_revision: revisions.get(dir) ?? 0, operations });
  const env = pot(['apply', dir, '--file', '-', '--json'], body);
  revisions.set(dir, env.scene.revision);
  return env;
}

function item(dir, id) {
  const [it] = pot(['inspect', dir, '--id', id, '--json']).result.items;
  return it;
}

function copyProject(src, dst) {
  cpSync(src, dst, { recursive: true });
  revisions.set(dst, revisions.get(src));
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
const dist2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
const translationOf = (m) => [m[12], m[13], m[14]]; // column-major 4x4
const isIdentity = (m) => m.every((v, i) => Math.abs(v - (i % 5 === 0 ? 1 : 0)) < 1e-9);

function assertClose(actual, expected, tol, label) {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`self-check failed: ${label}: got ${actual}, expected ${expected} (tol ${tol})`);
  }
}

function bbox(positions) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of positions) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], p[k]);
      max[k] = Math.max(max[k], p[k]);
    }
  }
  return { min, max };
}

// Quaternion [x,y,z,w] for Rz * Ry * Rx (degrees).
function quatFromEulerDeg([rx, ry, rz]) {
  const axis = (ax, deg) => {
    const h = (deg * Math.PI) / 360;
    const s = Math.sin(h);
    return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(h)];
  };
  const mul = (a, b) => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
  return mul(mul(axis([0, 0, 1], rz), axis([0, 1, 0], ry)), axis([1, 0, 0], rx));
}

// ---------------------------------------------------------------------------
// Pixel data (deterministic)
// ---------------------------------------------------------------------------
function pixelsFor(id, w, h) {
  const out = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = x / w;
      switch (id) {
        case 'tex_skin':
          out.push([0.96, 0.78 + (0.12 * x) / (w - 1), 0.7, 1]);
          break;
        case 'tex_skin_normal':
          out.push([0.5, 0.5, 1, 1]);
          break;
        case 'tex_cloth':
          out.push([x / (w - 1), 0.3, 1 - x / (w - 1), 1]);
          break;
        case 'tex_uv_mask':
          out.push([0, 0, u >= 0.5 ? 1 : 0, 1]);
          break;
        case 'tex_cutout': {
          const inside = Math.hypot(x + 0.5 - w / 2, y + 0.5 - h / 2) < (w * 3) / 8;
          out.push([1, 1, 1, inside ? 1 : 0]);
          break;
        }
        case 'tex_ribbon':
          out.push([1, 1, 1, 1]);
          break;
        case 'thumb':
          out.push([x / (w - 1), y / (h - 1), 0.5, 1]);
          break;
        default:
          throw new Error(`no pixel pattern for image ${id}`);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Batch 1: rig extras, images, materials, meshes
// ---------------------------------------------------------------------------
const alphaMode = { OPAQUE: 'opaque', BLEND: 'blend', MASK: 'clip' };

function imageOps() {
  return Object.entries(S.IMAGES).flatMap(([id, img]) => [
    { op: 'image.create', id, name: id, width: img.width, height: img.height, colorspace: img.colorspace, fill_color: [0, 0, 0, 1] },
    { op: 'image.set_pixels', id, width: img.width, height: img.height, pixels: pixelsFor(id, img.width, img.height) },
  ]);
}

function materialOps() {
  return Object.entries(S.MATERIALS).map(([id, m]) => {
    const op = {
      op: 'material.create',
      id,
      name: id,
      base_color: m.baseColor,
      alpha_mode: alphaMode[m.alphaMode],
      double_sided: m.doubleSided,
    };
    if (m.alphaCutoff !== undefined) op.alpha_threshold = m.alphaCutoff;
    if (m.baseTexture) op.base_color_texture = { image: m.baseTexture, interpolation: 'linear' };
    if (m.normalMap) op.normal_texture = { image: m.normalMap };
    return op;
  });
}

function graphMaterialOps() {
  return Object.entries(S.GRAPH_MATERIALS).map(([id, g]) => ({
    op: 'material.create',
    id,
    name: id,
    graph: {
      kind: 'shader',
      nodes: {
        out: { type: 'OutputMaterial' },
        bsdf: { type: 'ShaderNodeBsdfPrincipled', inputs: { 'Emission Color': g.emissionColor, 'Emission Strength': g.emissionStrength } },
        base: { type: 'ShaderNodeTexImage', properties: { image: g.baseTexture } },
        clip: { type: 'ShaderNodeMath', properties: { operation: 'GREATER_THAN' }, inputs: { Value_001: g.alphaCutoff } },
        normal_img: { type: 'ShaderNodeTexImage', properties: { image: g.normalMap } },
        normal: { type: 'ShaderNodeNormalMap', properties: { space: 'TANGENT' } },
      },
      links: [
        { from_node: 'base', from_socket: 'Color', to_node: 'bsdf', to_socket: 'Base Color' },
        { from_node: 'base', from_socket: 'Alpha', to_node: 'clip', to_socket: 'Value' },
        { from_node: 'clip', from_socket: 'Value', to_node: 'bsdf', to_socket: 'Alpha' },
        { from_node: 'normal_img', from_socket: 'Color', to_node: 'normal', to_socket: 'Color' },
        { from_node: 'normal', from_socket: 'Normal', to_node: 'bsdf', to_socket: 'Normal' },
        { from_node: 'bsdf', from_socket: 'BSDF', to_node: 'out', to_socket: 'Surface' },
      ],
    },
  }));
}

function tubeOps(id, t, material) {
  return [
    { op: 'node.create', id, kind: 'cylinder', params: { radius: t.radius, depth: t.depth, vertices: t.vertices, end_fill_type: 'NOTHING' }, materials: [material] },
    { op: 'mesh.transform_elements', target: { id, elements: ALL_VERTICES }, rotation: t.rotationDeg.map((d) => (d * Math.PI) / 180) },
    { op: 'mesh.transform_elements', target: { id, elements: ALL_VERTICES }, translation: t.center },
    { op: 'mesh.subdivide', target: { id }, elements: ALL_FACES, cuts: t.cuts },
    { op: 'uv.unwrap', target: { id }, method: 'smart' },
  ];
}

function attachToBone(id, bone) {
  return { op: 'node.parent', target: { id }, parent: S.RIG, parent_type: 'bone', parent_bone: bone, keep_world: true };
}

function buildBatchOne(dir) {
  const { MESHES: M } = S;
  const ops = [
    { op: 'node.create', id: S.RIG, kind: 'armature' },
    { op: 'rig.generate_basic_human', target: { id: S.RIG } },
    ...Object.entries(S.BONE_ADDED).map(([id, b]) => ({
      op: 'bone.create', target: { id: S.RIG }, id, name: id, parent: b.parent, head: b.head, tail: b.tail,
    })),
    ...imageOps(),
    ...materialOps(),
    ...graphMaterialOps(),
    { op: 'node.create', id: M.head, kind: 'uv_sphere', params: { radius: S.HEAD_MESH.radius, segments: S.HEAD_MESH.segments, ring_count: S.HEAD_MESH.ringCount }, materials: ['skin', 'eye'] },
    { op: 'mesh.transform_elements', target: { id: M.head, elements: ALL_VERTICES }, translation: S.HEAD_MESH.center },
    { op: 'uv.unwrap', target: { id: M.head }, method: 'smart' },
    ...Object.values(S.EYES).map((e) => ({
      op: 'mesh.set_attribute', target: { id: M.head }, attribute: 'material_index', value: 1,
      elements: { domain: 'face', selector: { type: 'position_sphere', center: e.center, radius: e.radius } },
    })),
    ...tubeOps(M.body, S.TUBES[M.body], 'cloth'),
    ...tubeOps(M.armL, S.TUBES[M.armL], 'skin'),
    ...tubeOps(M.armR, S.TUBES[M.armR], 'skin'),
    ...tubeOps(M.legL, S.TUBES[M.legL], 'skin'),
    ...tubeOps(M.legR, S.TUBES[M.legR], 'skin'),
    { op: 'node.create', id: M.hat, kind: 'cylinder', params: { radius: S.HAT.radius, depth: S.HAT.depth, vertices: 16 }, materials: ['cloth'], transform: { translation: S.HAT.center } },
    { op: 'uv.unwrap', target: { id: M.hat }, method: 'smart' },
    attachToBone(M.hat, S.HAT.parentBone),
    { op: 'node.create', id: M.ribbon, kind: 'plane', params: { size: S.RIBBON.size }, materials: ['ribbon'], transform: { translation: S.RIBBON.center, rotation_deg: S.RIBBON.rotationDeg } },
    { op: 'uv.unwrap', target: { id: M.ribbon }, method: 'smart' },
    attachToBone(M.ribbon, S.RIBBON.parentBone),
    { op: 'node.create', id: M.ribbonMirror, kind: 'plane', params: { size: S.RIBBON.size }, materials: ['cutout'], transform: { translation: S.RIBBON.mirrorCenter, rotation_deg: S.RIBBON.rotationDeg, scale: S.RIBBON.mirrorScale } },
    { op: 'uv.unwrap', target: { id: M.ribbonMirror }, method: 'smart' },
    attachToBone(M.ribbonMirror, S.RIBBON.parentBone),
  ];
  apply(dir, ops);
}

// ---------------------------------------------------------------------------
// Batch 2: skin weights, head shape keys, smooth shading (computed from inspect)
// ---------------------------------------------------------------------------
function skinWeights(positions, chain, heads) {
  return positions.map((p) => {
    const d2 = chain.map((bone) => dist2(p, heads[bone]));
    const dMin = Math.min(...d2);
    const scored = chain
      .map((bone, i) => ({ bone, w: Math.exp(-(d2[i] - dMin) / (2 * SKIN_SIGMA * SKIN_SIGMA)) }))
      .sort((a, b) => b.w - a.w)
      .filter((e) => e.w >= MIN_WEIGHT)
      .slice(0, MAX_BONES);
    const sum = scored.reduce((s, e) => s + e.w, 0);
    return scored.map((e) => [e.bone, e.w / sum]);
  });
}

function morphPositions(restPositions, regions) {
  const out = {};
  restPositions.forEach((p, index) => {
    let moved = null;
    for (const r of regions) {
      const d = Math.sqrt(dist2(p, r.center));
      if (d >= r.radius) continue;
      const f = (1 - d / r.radius) ** 2;
      moved ??= [...p];
      for (let k = 0; k < 3; k++) moved[k] += r.delta[k] * f;
    }
    if (moved) out[index] = moved;
  });
  return out;
}

const SKIN_MESHES = [S.MESHES.head, S.MESHES.body, S.MESHES.armL, S.MESHES.armR, S.MESHES.legL, S.MESHES.legR];

function buildBatchTwo(dir) {
  const rig = item(dir, S.RIG);
  const heads = Object.fromEntries(Object.entries(rig.evaluated_bones).map(([name, m]) => [name, translationOf(m)]));
  // glTF skin export matches vertex groups to deform bones by bone *name* and silently drops
  // unmatched influences. The generated rig names bones "upper_arm.L" etc., so name each skin
  // bone after its id and mark it deform.
  const skinBones = [...new Set(Object.values(S.SKIN_CHAINS).flat())];
  const ops = skinBones.map((bone) => ({ op: 'bone.update', target: { id: S.RIG }, bone_id: bone, set: { name: bone, deform: true } }));
  const weightsByMesh = {};
  const restByMesh = {};
  for (const mesh of SKIN_MESHES) {
    const it = item(dir, mesh);
    if (!isIdentity(it.transform.world.matrix)) throw new Error(`${mesh}: object transform must be identity`);
    const positions = it.evaluated_geometry.positions;
    restByMesh[mesh] = positions;
    const chain = S.SKIN_CHAINS[mesh];
    const weights = skinWeights(positions, chain, heads);
    weightsByMesh[mesh] = weights;
    ops.push({ op: 'modifier.create', target: { id: mesh }, id: 'skin', type: 'armature', params: { object: S.RIG, use_vertex_groups: true } });
    for (const bone of chain) {
      ops.push({ op: 'vertex_group.create', target: { id: mesh }, id: bone, name: bone });
      const assigned = [];
      weights.forEach((w, vertex) => {
        const entry = w.find(([b]) => b === bone);
        if (entry) assigned.push({ vertex_id: vertex, weight: entry[1] });
      });
      ops.push({ op: 'vertex_group.assign', target: { id: mesh }, group_id: bone, weights: assigned });
    }
  }
  const headRest = restByMesh[S.MESHES.head];
  for (const [id, regions] of Object.entries(S.MORPH_REGIONS)) {
    ops.push({ op: 'shape_key.create', target: { id: S.MESHES.head }, id, name: id, positions: morphPositions(headRest, regions) });
  }
  ops.push({ op: 'mesh.shade_smooth', target: { id: S.MESHES.head, elements: ALL_FACES } });
  apply(dir, ops);
  return { weightsByMesh, restByMesh };
}

// ---------------------------------------------------------------------------
// Self-checks on the geometry the batches produced
// ---------------------------------------------------------------------------
function selfCheckGeometry(dir) {
  const head = item(dir, S.MESHES.head).evaluated_geometry.positions;
  for (const p of head) {
    const d = Math.sqrt(dist2(p, S.HEAD_MESH.center));
    assertClose(d, S.HEAD_MESH.radius, 1e-5, 'head sphere radius');
  }
  const arm = bbox(item(dir, S.MESHES.armL).evaluated_geometry.positions);
  assertClose(arm.min[0], 0, 1e-5, 'arm_l min x');
  assertClose(arm.max[0], 1, 1e-5, 'arm_l max x');
  const armR = bbox(item(dir, S.MESHES.armR).evaluated_geometry.positions);
  assertClose(armR.min[0], -1, 1e-5, 'arm_r min x');
  assertClose(armR.max[0], 0, 1e-5, 'arm_r max x');
  const legL = bbox(item(dir, S.MESHES.legL).evaluated_geometry.positions);
  assertClose(legL.min[2], 0, 1e-5, 'leg_l min z');
  assertClose(legL.max[2], 0.9, 1e-5, 'leg_l max z');
  const body = bbox(item(dir, S.MESHES.body).evaluated_geometry.positions);
  assertClose(body.min[2], 0.85, 1e-5, 'body min z');
  assertClose(body.max[2], 1.5, 1e-5, 'body max z');
}

// ---------------------------------------------------------------------------
// Variant avatar settings (pot avatar.update, snake_case fields)
// ---------------------------------------------------------------------------
const rangeMap = (r) => ({ input_max_value: r.inputMaxValue, output_scale: r.outputScale });
const shapeSnake = (s) =>
  s.type === 'sphere'
    ? { type: 'sphere', offset: s.offset, radius: s.radius }
    : { type: 'capsule', offset: s.offset, tail: s.tail, radius: s.radius };

function expressionSet(variant) {
  const v0 = variant === 'v0';
  return Object.fromEntries(
    Object.entries(S.EXPRESSIONS).map(([name, e]) => {
      const out = {};
      if (e.isBinary) out.is_binary = true;
      if (!v0 && e.overrideBlink) out.override_blink = e.overrideBlink;
      if (!v0 && e.overrideMouth) out.override_mouth = e.overrideMouth;
      out.morph_target_binds = e.morphs.map(([shape_key, weight]) => ({ node: S.MESHES.head, shape_key, weight }));
      if (e.materialColorBinds) {
        out.material_color_binds = e.materialColorBinds.map((b) => ({ material: b.material, color_type: b.colorType, target_value: b.targetValue }));
      }
      if (e.textureTransformBinds) {
        out.texture_transform_binds = e.textureTransformBinds.map((b) => ({ material: b.material, scale: b.scale, offset: b.offset }));
      }
      return [name, out];
    }),
  );
}

function avatarSet(variant) {
  const look = S.LOOKAT[variant];
  const colliders = S.COLLIDERS[variant === 'v1LookAt' ? 'v1Full' : variant];
  const groups = S.COLLIDER_GROUPS[variant === 'v1LookAt' ? 'v1Full' : variant];
  const springGroups = S.SPRING_COLLIDER_GROUPS[variant];
  const springJoints = variant === 'v0' ? S.SPRING.joints.map((j) => ({ ...j, ...S.SPRING.v0Uniform })) : S.SPRING.joints;
  const c = S.MATERIALS.cloth;
  return {
    armature: S.RIG,
    humanoid: S.HUMANOID[variant],
    meta: {
      name: S.META.name,
      version: S.META.version,
      authors: S.META.authors,
      license_url: S.META.licenseUrl,
      thumbnail: S.META.thumbnail,
      modification: S.META.rights.modification,
      allow_redistribution: S.META.rights.allowRedistribution,
      commercial_usage: S.META.rights.commercialUsage,
      avatar_permission: S.META.rights.avatarPermission,
      credit_notation: S.META.rights.creditNotation,
      legacy: { license_name: S.META.legacy.licenseName, other_permission_url: S.META.legacy.otherPermissionUrl },
    },
    expressions: expressionSet(variant),
    look_at: {
      look_at_type: look.lookAtType,
      offset_from_head_bone: look.offsetFromHeadBone,
      range_map_horizontal_inner: rangeMap(look.rangeMapHorizontalInner),
      range_map_horizontal_outer: rangeMap(look.rangeMapHorizontalOuter),
      range_map_vertical_down: rangeMap(look.rangeMapVerticalDown),
      range_map_vertical_up: rangeMap(look.rangeMapVerticalUp),
    },
    spring_bone: {
      colliders: colliders.map((col) => ({ node: { node: S.RIG, bone: col.bone }, shape: shapeSnake(col.shape) })),
      collider_groups: groups.map((g) => ({ name: g.name, colliders: g.colliders })),
      springs: [
        {
          name: S.SPRING.name,
          center: null,
          collider_groups: springGroups,
          joints: springJoints.map((j) => ({
            node: { node: S.RIG, bone: j.bone },
            stiffness: j.stiffness,
            drag_force: j.dragForce,
            gravity_power: j.gravityPower,
            gravity_dir: j.gravityDir,
            hit_radius: j.hitRadius,
          })),
        },
      ],
    },
    mtoon: {
      cloth: {
        outline_width_mode: 'world_coordinates',
        outline_width_factor: c.outlineWidth,
        outline_color_factor: c.outlineColor,
        uv_animation_mask_texture: { image: c.uvAnimationMask, interpolation: 'linear' },
        uv_animation_scroll_x_speed_factor: c.uvAnimationScrollX,
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Reference data (pose applied on a copy of the v1-full project)
// ---------------------------------------------------------------------------
const toGltfAll = (points) => points.map((p) => S.toGltf(p));

function buildReference(dir, restByMesh, weightsByMesh) {
  const refDir = join(dir, '..', 'reference-copy');
  copyProject(dir, refDir);
  // The rig's IK constraints re-solve arm and leg chains from targets. VRM has no IK and the
  // consumer poses by forward kinematics, so the reference copy drops them (base project unchanged).
  const ikConstraints = item(refDir, S.RIG).constraints.map((c) => c.id);
  apply(refDir, ikConstraints.map((id) => ({ op: 'constraint.delete', target: { id: S.RIG }, id })));
  const restBones = item(refDir, S.RIG).evaluated_bones;
  const restFromRefDir = Object.fromEntries(SKIN_MESHES.map((m) => [m, item(refDir, m).evaluated_geometry.positions]));
  for (const mesh of SKIN_MESHES) {
    restFromRefDir[mesh].forEach((p, i) =>
      p.forEach((v, k) => assertClose(v, restByMesh[mesh][i][k], S.TOL.position, `${mesh} rest vertex ${i} without IK`)),
    );
  }
  apply(
    refDir,
    S.POSE.map((p) => ({
      op: 'pose.set',
      target: { id: S.RIG },
      bone_id: p.bone,
      set: { rotation: quatFromEulerDeg(p.eulerDeg) },
    })),
  );
  const posedBones = item(refDir, S.RIG).evaluated_bones;
  const meshes = {};
  const blended = {};
  for (const mesh of SKIN_MESHES) {
    const rest = restFromRefDir[mesh];
    const posed = item(refDir, mesh).evaluated_geometry.positions;
    meshes[mesh] = { vertexCount: rest.length, restPotter: rest, restGltf: toGltfAll(rest) };
    if (mesh === S.MESHES.armL || mesh === S.MESHES.armR || mesh === S.MESHES.legL || mesh === S.MESHES.legR) {
      blended[mesh] = weightsByMesh[mesh]
        .map((w, index) => ({ index, weights: w }))
        .filter((e) => e.weights.length >= 2 && e.weights[1][1] >= BLEND_MIN)
        .map((e) => ({
          index: e.index,
          weights: e.weights,
          restGltf: S.toGltf(rest[e.index]),
          posedGltf: S.toGltf(posed[e.index]),
        }));
    }
  }
  const headRest = restByMesh[S.MESHES.head];
  const morphs = {};
  for (const id of S.MORPHS) {
    const displaced = morphPositions(headRest, S.MORPH_REGIONS[id]);
    morphs[id] = headRest.map((p, i) => {
      const abs = displaced[i] ?? p;
      return S.toGltf(abs.map((v, k) => v - p[k]));
    });
  }
  return {
    format: 1,
    units: 'metres',
    basis: { potter: 'Z-up, front -Y, left +X', gltf: 'Y-up, front +Z, left +X', gltfFromPotter: '[x,z,-y]' },
    pose: {
      bones: S.POSE.map((p) => ({
        bone: p.bone,
        eulerDeg: p.eulerDeg,
        quatXyzw: quatFromEulerDeg(p.eulerDeg),
        // Column-major 4x4 bone matrices in Potter world space, before and after the pose.
        restWorldPotter: restBones[p.bone],
        posedWorldPotter: posedBones[p.bone],
      })),
    },
    meshes,
    blended,
    morphs,
    correspondence:
      'glTF primitive vertices are matched to Potter indices by restGltf position (TOL.position); the exporter may split vertices at seams/normals',
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
function exportVariant(dir, variant) {
  const out = join(OUT, variant.file);
  pot(['export', dir, '--format', variant.format, '--out', out, '--overwrite', '--json']);
  pot(['validate', dir, '--format', variant.format, '--json']);
  return out;
}

function main() {
  mkdirSync(OUT, { recursive: true });
  const work = mkdtempSync(join(tmpdir(), 'potter-vrm-fixture-'));
  const base = join(work, 'base');
  pot(['init', base, '--json']);
  revisions.set(base, 0);

  buildBatchOne(base);
  const { weightsByMesh, restByMesh } = buildBatchTwo(base);
  selfCheckGeometry(base);

  const variantDirs = {};
  for (const variant of VARIANTS) {
    const dir = join(work, variant.key);
    copyProject(base, dir);
    apply(dir, [{ op: 'avatar.update', set: avatarSet(variant.key) }]);
    variantDirs[variant.key] = dir;
  }

  const written = [];
  for (const variant of VARIANTS) {
    written.push(exportVariant(variantDirs[variant.key], variant));
  }
  const reference = buildReference(variantDirs.v1Full, restByMesh, weightsByMesh);
  const refPath = join(OUT, S.FILES.reference);
  writeFileSync(refPath, `${JSON.stringify(reference, null, 1)}\n`);
  written.push(refPath);

  rmSync(work, { recursive: true, force: true });
  for (const p of written) console.log(p);
}

try {
  main();
} catch (err) {
  console.error(err.message);
  process.exitCode = 1;
}
