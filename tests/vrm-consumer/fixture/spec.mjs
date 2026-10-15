// Single source of truth for the VRM consumer fixture.
// Consumers and checks import this file; values MUST NOT be duplicated elsewhere.
//
// Units: Potter world space is Z-up, front = -Y, left = +X, metres.
// glTF/VRM space is Y-up, front = +Z, left = +X. Basis: gltf = [x, z, -y].
// Bone-local values (lookAt offset, capsule tail, spring gravity) are in the
// bone's Potter local frame, as the pot `avatar.update` schema defines them.
// All skinned meshes have identity object transforms: their vertices are baked
// into world space, so Potter positions are world positions.

export const FILES = {
  v1Full: 'v1-full.vrm',
  v1LookAt: 'v1-expression-lookat.vrm',
  v0: 'v0.vrm',
  reference: 'reference.json',
};

// ---------------------------------------------------------------------------
// Scene objects (Potter ids). Pattern: ^[a-z][a-z0-9_-]{0,63}$
// ---------------------------------------------------------------------------
export const RIG = 'rig'; // armature object (rig.generate_basic_human)
export const MESHES = {
  head: 'head_mesh', // uv_sphere, skin slot 0 + eye slot 1, skinned to head
  body: 'body', // cylinder, skinned to hips/spine chain, material cloth
  armL: 'arm_l', // one continuous cylinder shoulder->elbow->hand
  armR: 'arm_r',
  legL: 'leg_l', // one continuous cylinder hip->knee->foot
  legR: 'leg_r',
  hat: 'hat', // bone-parented to head, material cloth
  ribbon: 'ribbon', // single-sided plane, bone-parented to head, material ribbon
  ribbonMirror: 'ribbon_mirror', // same plane with scale [-1,1,1], material cutout
};

// Head sphere placement (world, metres) and radius.
export const HEAD_MESH = { center: [0, 0, 1.8], radius: 0.12, segments: 32, ringCount: 16 };

// Eye faces on head_mesh (material_index 1) are the faces whose centroid lies
// within `radius` of these points (on the head surface). Eye bones sit there too.
export const EYES = {
  l: { bone: 'eye_l', center: [0.045, -0.111, 1.822], tail: [0.045, -0.131, 1.822], radius: 0.03 },
  r: { bone: 'eye_r', center: [-0.045, -0.111, 1.822], tail: [-0.045, -0.131, 1.822], radius: 0.03 },
};

// Shape keys (morph targets) on head_mesh. Potter ids are lowercase; the VRM
// expression names are camelCase presets (see EXPRESSIONS).
export const MORPHS = ['blink', 'aa', 'ih', 'ou', 'ee', 'oh', 'happy', 'look_left', 'look_right', 'look_up', 'look_down'];

const MOUTH = [0, -0.1, 1.74];
// Each morph moves head vertices within `radius` of `center` by
// delta * (1 - d / radius)^2 (world space). Shape key = rest + displacement.
export const MORPH_REGIONS = {
  blink: [
    { center: EYES.l.center, radius: 0.06, delta: [0, 0, -0.012] },
    { center: EYES.r.center, radius: 0.06, delta: [0, 0, -0.012] },
  ],
  aa: [{ center: MOUTH, radius: 0.06, delta: [0, 0, -0.02] }],
  ih: [{ center: MOUTH, radius: 0.06, delta: [0.012, 0, 0] }],
  ou: [{ center: MOUTH, radius: 0.05, delta: [0, 0, -0.01] }],
  ee: [{ center: MOUTH, radius: 0.06, delta: [0.02, 0, 0] }],
  oh: [{ center: MOUTH, radius: 0.05, delta: [0, 0.01, -0.015] }],
  happy: [
    { center: [0.035, -0.1, 1.76], radius: 0.035, delta: [0.01, 0, 0.01] },
    { center: [-0.035, -0.1, 1.76], radius: 0.035, delta: [-0.01, 0, 0.01] },
  ],
  look_left: [
    { center: EYES.l.center, radius: 0.03, delta: [0.01, 0, 0] },
    { center: EYES.r.center, radius: 0.03, delta: [0.01, 0, 0] },
  ],
  look_right: [
    { center: EYES.l.center, radius: 0.03, delta: [-0.01, 0, 0] },
    { center: EYES.r.center, radius: 0.03, delta: [-0.01, 0, 0] },
  ],
  look_up: [
    { center: EYES.l.center, radius: 0.03, delta: [0, 0, 0.01] },
    { center: EYES.r.center, radius: 0.03, delta: [0, 0, 0.01] },
  ],
  look_down: [
    { center: EYES.l.center, radius: 0.03, delta: [0, 0, -0.01] },
    { center: EYES.r.center, radius: 0.03, delta: [0, 0, -0.01] },
  ],
};

// ---------------------------------------------------------------------------
// Skeleton
// rig.generate_basic_human creates: root, hips, spine, spine_001, spine_002,
// spine_003, neck, head, shoulder_{l,r}, upper_arm_{l,r}, forearm_{l,r},
// hand_{l,r}, thigh_{l,r}, shin_{l,r}, foot_{l,r}, toe_{l,r}.
// Added by the fixture: eye_l, eye_r (children of head) and the hair chain
// hair_1 -> hair_2 -> hair_3 -> hair_tip -> hair_end. hair_end is the tail: pot's VRM 1.0 export
// drops a listed joint that has no child (tail-only leaf), and three-vrm pairs consecutive
// joints, so the chain lists hair_tip as its last joint and hair_end is the child that gives it a tail.
// ---------------------------------------------------------------------------
export const BONE_ADDED = {
  eye_l: { parent: 'head', head: EYES.l.center, tail: EYES.l.tail },
  eye_r: { parent: 'head', head: EYES.r.center, tail: EYES.r.tail },
  hair_1: { parent: 'head', head: [0, 0.13, 1.84], tail: [0, 0.15, 1.7] },
  hair_2: { parent: 'hair_1', head: [0, 0.15, 1.7], tail: [0, 0.15, 1.55] },
  hair_3: { parent: 'hair_2', head: [0, 0.15, 1.55], tail: [0, 0.15, 1.42] },
  hair_tip: { parent: 'hair_3', head: [0, 0.15, 1.42], tail: [0, 0.15, 1.3] },
  hair_end: { parent: 'hair_tip', head: [0, 0.15, 1.3], tail: [0, 0.15, 1.2] },
};

// World-space bone head positions of the generated rig (used for skin weights).
export const BONE_HEADS = {
  hips: [0, 0, 0.85],
  spine: [0, 0, 0.95],
  spine_001: [0, 0, 1.15],
  spine_002: [0, 0, 1.35],
  spine_003: [0, 0, 1.5],
  neck: [0, 0, 1.6],
  head: [0, 0, 1.7],
  shoulder_l: [0, 0, 1.5],
  upper_arm_l: [0.25, 0, 1.5],
  forearm_l: [0.65, 0.025, 1.5],
  hand_l: [1.0, 0, 1.5],
  thigh_l: [0.15, 0, 0.9],
  shin_l: [0.15, -0.025, 0.5],
  foot_l: [0.15, 0, 0.1],
};

// Potter world rotation of the head bone, columns = bone-local X, Y, Z in Potter world. Evidence:
// `pot inspect <rig> --id rig` matrix for "head" (col-major X=(1,0,0), Y=(0,0,1), Z=(0,-1,0)).
// Bone-local collider offsets are rotated by this before the Potter->glTF basis change.
export const HEAD_BONE_POTTER_COLUMNS = [
  [1, 0, 0],
  [0, 0, 1],
  [0, -1, 0],
];

// Head-local look-at offset -> glTF world displacement at rest. Both VRM versions place the gaze origin here
// (the offset is the same physical point), so tests use this instead of a version-specific head frame.
export const headLocalToGltf = (v) =>
  toGltf(HEAD_BONE_POTTER_COLUMNS.reduce((sum, axis, i) => sum.map((s, k) => s + v[i] * axis[k]), [0, 0, 0]));

// Bone chains that skin each mesh. Weights are computed from distance to the
// chain's bone heads with a smooth falloff, keeping at most 3 bones.
export const SKIN_CHAINS = {
  [MESHES.head]: ['head'],
  [MESHES.body]: ['hips', 'spine', 'spine_002', 'spine_003'],
  [MESHES.armL]: ['upper_arm_l', 'forearm_l', 'hand_l'],
  [MESHES.armR]: ['upper_arm_r', 'forearm_r', 'hand_r'],
  [MESHES.legL]: ['thigh_l', 'shin_l', 'foot_l'],
  [MESHES.legR]: ['thigh_r', 'shin_r', 'foot_r'],
};

// Continuous tube geometry: one open-ended cylinder per limb (no end caps),
// vertices per ring as listed, then 2 subdivision cuts (each cut doubles the
// segments along the tube) so elbow and knee vertices exist. Rotation is a
// single-axis angle in degrees about Y, applied before translation.
// Arms run along X (T-pose, shoulder x=0 to hand x=1); legs run along Z from the
// floor (z=0) to the hip (z=0.9). Body spans hips (0.85) to spine_003 (1.5).
export const TUBES = {
  [MESHES.body]: { center: [0, 0, 1.175], rotationDeg: [0, 0, 0], radius: 0.16, depth: 0.65, vertices: 16, cuts: 2 },
  [MESHES.armL]: { center: [0.5, 0, 1.5], rotationDeg: [0, 90, 0], radius: 0.05, depth: 1.0, vertices: 16, cuts: 2 },
  [MESHES.armR]: { center: [-0.5, 0, 1.5], rotationDeg: [0, -90, 0], radius: 0.05, depth: 1.0, vertices: 16, cuts: 2 },
  [MESHES.legL]: { center: [0.15, 0, 0.45], rotationDeg: [0, 0, 0], radius: 0.08, depth: 0.9, vertices: 16, cuts: 2 },
  [MESHES.legR]: { center: [-0.15, 0, 0.45], rotationDeg: [0, 0, 0], radius: 0.08, depth: 0.9, vertices: 16, cuts: 2 },
};

export const HAT = { parentBone: 'head', center: [0, 0, 1.95], radius: 0.13, depth: 0.06 };
// Ribbon planes are vertical, facing front (Potter -Y; glTF +Z). rotationDeg [90,0,0] turns the
// plane's +Z normal to -Y. Centres sit just in front of the head sphere (front face at y=-0.12).
export const RIBBON = {
  parentBone: 'head',
  size: 0.3,
  rotationDeg: [90, 0, 0],
  center: [0.15, -0.16, 1.9],
  mirrorCenter: [-0.15, -0.16, 1.9],
  mirrorScale: [-1, 1, 1],
};

// ---------------------------------------------------------------------------
// Humanoid mapping (VRM humanoid name -> Potter bone). Per variant.
// ---------------------------------------------------------------------------
const HUMANOID_BASE = {
  hips: 'hips',
  spine: 'spine',
  chest: 'spine_002',
  upperChest: 'spine_003',
  neck: 'neck',
  head: 'head',
  leftShoulder: 'shoulder_l',
  leftUpperArm: 'upper_arm_l',
  leftLowerArm: 'forearm_l',
  leftHand: 'hand_l',
  rightShoulder: 'shoulder_r',
  rightUpperArm: 'upper_arm_r',
  rightLowerArm: 'forearm_r',
  rightHand: 'hand_r',
  leftUpperLeg: 'thigh_l',
  leftLowerLeg: 'shin_l',
  leftFoot: 'foot_l',
  leftToes: 'toe_l',
  rightUpperLeg: 'thigh_r',
  rightLowerLeg: 'shin_r',
  rightFoot: 'foot_r',
  rightToes: 'toe_r',
};
export const HUMANOID = {
  v1Full: { ...HUMANOID_BASE, leftEye: 'eye_l', rightEye: 'eye_r' },
  v1LookAt: { ...HUMANOID_BASE }, // leftEye/rightEye deliberately unmapped
  v0: { ...HUMANOID_BASE, leftEye: 'eye_l', rightEye: 'eye_r' },
};

// ---------------------------------------------------------------------------
// Materials (glTF material names). Colors are linear RGBA.
// ---------------------------------------------------------------------------
export const MATERIALS = {
  skin: { baseColor: [0.96, 0.8, 0.7, 1], alphaMode: 'OPAQUE', doubleSided: false, baseTexture: 'tex_skin', normalMap: 'tex_skin_normal' },
  eye: { baseColor: [0.2, 0.4, 0.8, 1], alphaMode: 'OPAQUE', doubleSided: false },
  cloth: {
    baseColor: [0.25, 0.45, 0.7, 1],
    alphaMode: 'OPAQUE',
    doubleSided: false,
    baseTexture: 'tex_cloth', // smooth horizontal gradient along U, every texel distinct (see IMAGES)
    outlineWidth: 0.002,
    outlineWidthMode: 'worldCoordinates',
    outlineColor: [0, 0, 0],
    uvAnimationMask: 'tex_uv_mask', // packed mask, value in channel B: B=0 for u<0.5, B=1 for u>=0.5
    uvAnimationScrollX: 0.1,
  },
  ribbon: { baseColor: [0.9, 0.9, 0.2, 0.6], alphaMode: 'BLEND', doubleSided: true, baseTexture: 'tex_ribbon' },
  cutout: { baseColor: [0.2, 0.8, 0.3, 1], alphaMode: 'MASK', alphaCutoff: 0.5, doubleSided: false, baseTexture: 'tex_cutout' },
};

// Node-graph materials (material.create with `graph`). Kept apart from MATERIALS because their
// surface comes from the graph, not from the material fields. Not assigned to any mesh: the
// glTF exporter writes every material, so this one is exported as a material only. Images are
// existing IMAGES entries, so the image count is unchanged.
export const GRAPH_MATERIALS = {
  graph_avatar: {
    baseTexture: 'tex_cutout', // RGB white; alpha disc, clipped at 0.5 by Math GREATER_THAN
    alphaCutoff: 0.5,
    normalMap: 'tex_skin_normal', // NormalMap (TANGENT) input
    emissionColor: [1.0, 0.5, 0.2, 1],
    emissionStrength: 1.5,
  },
};

// Images (created with pot image ops, pixels generated deterministically in JS).
// Channel semantics: tex_uv_mask carries the animation mask in B; tex_cutout in A.
export const IMAGES = {
  tex_skin: { width: 64, height: 64, colorspace: 'srgb' },
  tex_skin_normal: { width: 64, height: 64, colorspace: 'linear' }, // flat normal (0.5,0.5,1)
  tex_cloth: { width: 64, height: 64, colorspace: 'srgb' }, // gradient along U: red 0->1, blue 1->0
  tex_uv_mask: { width: 64, height: 64, colorspace: 'linear' }, // B: 0 for u<0.5, 1 for u>=0.5
  tex_cutout: { width: 64, height: 64, colorspace: 'linear' }, // A: 1 inside a disc, 0 outside
  tex_ribbon: { width: 64, height: 64, colorspace: 'srgb' },
  thumb: { width: 128, height: 128, colorspace: 'srgb' },
};

// ---------------------------------------------------------------------------
// Expressions. morphs: [shape key, weight]. Overrides exist in v1 variants only.
// ---------------------------------------------------------------------------
export const EXPRESSIONS = {
  blink: { preset: true, morphs: [['blink', 1]] },
  aa: { preset: true, morphs: [['aa', 1]] },
  ih: { preset: true, morphs: [['ih', 1]] },
  ou: { preset: true, morphs: [['ou', 1]] },
  ee: { preset: true, morphs: [['ee', 1]] },
  oh: { preset: true, morphs: [['oh', 1]] },
  lookLeft: { preset: true, morphs: [['look_left', 1]] },
  lookRight: { preset: true, morphs: [['look_right', 1]] },
  lookUp: { preset: true, morphs: [['look_up', 1]] },
  lookDown: { preset: true, morphs: [['look_down', 1]] },
  happy: { preset: true, morphs: [['happy', 1]], overrideBlink: 'block', overrideMouth: 'blend' },
  surprised: { preset: true, isBinary: true, morphs: [['oh', 0.8]], overrideMouth: 'block' },
  sad: { preset: true, morphs: [['ih', 0.5]], overrideBlink: 'blend' },
  cloth_shift: {
    preset: false,
    morphs: [],
    materialColorBinds: [{ material: 'cloth', colorType: 'color', targetValue: [0.9, 0.3, 0.3, 1] }],
    textureTransformBinds: [{ material: 'cloth', scale: [2, 2], offset: [0.25, 0] }],
  },
};

// VRM 0.x blend shape group preset names (BlendShapePresetName). Every expression not listed is "unknown".
export const V0_PRESET = {
  aa: 'a',
  ih: 'i',
  ou: 'u',
  ee: 'e',
  oh: 'o',
  blink: 'blink',
  happy: 'joy',
  sad: 'sorrow',
  lookUp: 'lookup',
  lookDown: 'lookdown',
  lookLeft: 'lookleft',
  lookRight: 'lookright',
};

// ---------------------------------------------------------------------------
// Spring bones (hair). Joints ordered root -> tip. VRM 1.0 pairs consecutive joints, so the
// last joint (hair_tip) is the tail-only node: it is emitted with its parameters but is not
// simulated. Simulated bones: hair_1, hair_2, hair_3 (3 bones from 4 joints).
// ---------------------------------------------------------------------------
export const SPRING = {
  name: 'hair',
  joints: [
    { bone: 'hair_1', stiffness: 1.0, dragForce: 0.4, gravityPower: 0, gravityDir: [0, 0, -1], hitRadius: 0.02 },
    { bone: 'hair_2', stiffness: 1.2, dragForce: 0.5, gravityPower: 0.1, gravityDir: [0, 0, -1], hitRadius: 0.02 },
    { bone: 'hair_3', stiffness: 1.4, dragForce: 0.6, gravityPower: 0.2, gravityDir: [0, 0, -1], hitRadius: 0.02 },
    { bone: 'hair_tip', stiffness: 1.6, dragForce: 0.7, gravityPower: 0.3, gravityDir: [0, 0, -1], hitRadius: 0.02 },
  ],
  // VRM 0.x has one parameter set per spring chain; the v0 variant uses these for every joint.
  v0Uniform: { stiffness: 1.0, dragForce: 0.5, gravityPower: 0.1, gravityDir: [1, 0, 0], hitRadius: 0.02 }, // sideways: Potter +X is the character's left
};

// Colliders. Index order is the collider index used by collider groups.
export const COLLIDERS = {
  v1Full: [
    { bone: 'head', shape: { type: 'sphere', offset: [0, 0, 0.05], radius: 0.12 } },
    { bone: 'spine', shape: { type: 'capsule', offset: [0, 0, 0], tail: [0, 0, 0.2], radius: 0.15 } },
  ],
  v0: [{ bone: 'head', shape: { type: 'sphere', offset: [0, 0, 0.05], radius: 0.12 } }],
};
export const COLLIDER_GROUPS = {
  v1Full: [
    { name: 'head', colliders: [0] },
    { name: 'body', colliders: [1] },
  ],
  v0: [{ name: 'head', colliders: [0] }],
};
// Hair spring uses all groups (v1) or the head group (v0).
export const SPRING_COLLIDER_GROUPS = { v1Full: [0, 1], v1LookAt: [0, 1], v0: [0] };

// ---------------------------------------------------------------------------
// lookAt. Bone type uses degrees for output; expression type uses weights.
// ---------------------------------------------------------------------------
const LOOKAT_BONE = {
  lookAtType: 'bone',
  offsetFromHeadBone: [0, -0.08, 0.06],
  rangeMapHorizontalInner: { inputMaxValue: 30, outputScale: 10 },
  rangeMapHorizontalOuter: { inputMaxValue: 30, outputScale: 8 },
  rangeMapVerticalDown: { inputMaxValue: 30, outputScale: 6 },
  rangeMapVerticalUp: { inputMaxValue: 30, outputScale: 4 },
};
export const LOOKAT = {
  v1Full: LOOKAT_BONE,
  v0: LOOKAT_BONE,
  v1LookAt: {
    lookAtType: 'expression',
    offsetFromHeadBone: [0, -0.08, 0.06],
    rangeMapHorizontalInner: { inputMaxValue: 30, outputScale: 1 },
    rangeMapHorizontalOuter: { inputMaxValue: 30, outputScale: 0.8 },
    rangeMapVerticalDown: { inputMaxValue: 30, outputScale: 0.6 },
    rangeMapVerticalUp: { inputMaxValue: 30, outputScale: 0.4 },
  },
};

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------
export const META = {
  name: 'Potter VRM Fixture',
  version: '1.0.0',
  authors: ['Potter'],
  licenseUrl: 'https://vrm.dev/licenses/1.0/',
  thumbnail: 'thumb',
  // Canonical rights: no modification, no redistribution, non-profit personal use.
  // Consistent with the legacy licence below (redistribution_prohibited).
  rights: {
    modification: 'prohibited',
    allowRedistribution: false,
    commercialUsage: 'personal_non_profit',
    avatarPermission: 'only_author',
    creditNotation: 'required',
  },
  // VRM 0.x cannot encode political/religious or antisocial usage permissions;
  // the legacy block carries the permission URL that covers them.
  legacy: {
    licenseName: 'redistribution_prohibited',
    otherPermissionUrl: 'https://vrm.dev/licenses/1.0/',
  },
};

// ---------------------------------------------------------------------------
// Reference pose. Each entry gives Potter bone-local Euler degrees (Rz * Ry * Rx) and the
// quaternion [x,y,z,w] Q passed to pot pose.set. Pot applies Q in the bone's parent-relative
// frame: L_posed = L_rest * Q, where L = parent_world^-1 * world (verified to 1e-16 against pot
// evaluated_bones for the arm and leg bones). Parents are posed too, so world frames change
// along the chain; consumers must use parent-relative frames. The reference copy has the rig's
// IK constraints removed (IK re-solves arms and legs; VRM has none), so the pose is pure forward
// kinematics. restWorldPotter/posedWorldPotter are evaluated bone matrices (column-major, Potter
// world) before and after the pose; the parent of each bone is given by the rig hierarchy.
// ---------------------------------------------------------------------------
export const POSE = [
  { bone: 'shoulder_l', eulerDeg: [0, 0, 10] },
  { bone: 'shoulder_r', eulerDeg: [0, 0, -10] },
  { bone: 'upper_arm_l', eulerDeg: [0, 0, 30] },
  { bone: 'upper_arm_r', eulerDeg: [0, 0, -30] },
  { bone: 'forearm_l', eulerDeg: [0, 0, 60] }, // elbow bend
  { bone: 'forearm_r', eulerDeg: [0, 0, -60] },
  { bone: 'thigh_l', eulerDeg: [30, 0, 0] },
  { bone: 'thigh_r', eulerDeg: [30, 0, 0] },
  { bone: 'shin_l', eulerDeg: [-60, 0, 0] }, // knee bend
  { bone: 'shin_r', eulerDeg: [-60, 0, 0] },
  { bone: 'hips', eulerDeg: [0, 0, 5] },
];

// Basis from Potter to glTF/VRM for positions and directions: gltf = [x, z, -y].
export const toGltf = ([x, y, z]) => [x, z, -y];

// VRM 0.x vectors are Unity-space and the model faces -Z: VRM 1.0 glTF direction g is stored as [-g.x, g.y, g.z].
// three-vrm 3.5.5 reads v0 gravityDir as-is; VRMUtils.rotateVRM0 turns the scene but not world gravity.
export const v0GravityDir = ([x, y, z]) => [-x, y, z];

// Comparison tolerances. Units: positions in metres, angles in degrees,
// colors/uv/weights/morph weights unitless. Consumers import these; never inline.
export const TOL = {
  position: 1e-5, // rest positions, glTF float vs Potter float
  posedPosition: 1e-4, // posed/skinned positions after humanoid pose
  angleDeg: 0.01,
  color: 1e-4,
  uv: 1e-5,
  weight: 1e-6,
  morph: 1e-6,
};

// reference.json format (written by scripts/build-fixture.mjs):
// {
//   "format": 1,
//   "units": "metres",
//   "basis": { "potter": "Z-up, front -Y, left +X", "gltf": "Y-up, front +Z, left +X", "gltfFromPotter": "[x,z,-y]" },
//   "pose": { "bones": [ { "bone": "<potter bone>", "eulerDeg": [x,y,z], "quatXyzw": [x,y,z,w], "restWorldPotter": [16], "posedWorldPotter": [16] } ] },
//   "meshes": {
//     "<potter mesh id>": {
//       "vertexCount": N,
//       "restPotter": [[x,y,z], ...],   // all vertices, Potter index = array index
//       "restGltf": [[x,y,z], ...]      // same vertices in glTF space
//     }
//   },
//   "blended": {                         // arm/leg meshes: vertices whose 2nd bone weight >= 0.05 (joint blends)
//     "<potter mesh id>": [ { "index": i, "weights": [[bone, w], ...], "restGltf": [..], "posedGltf": [..] } ]
//   },
//   "morphs": {                          // head_mesh only; deltas (shape - rest) at weight 1, glTF space
//     "<shape key>": [[dx,dy,dz], ...]   // Potter index order
//   },
//   "correspondence": "glTF primitive vertices are matched to Potter indices by restGltf position (TOL.position); the exporter may split vertices at seams/normals"
// }
