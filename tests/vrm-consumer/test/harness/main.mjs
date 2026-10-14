// Browser-side API for the consumer checks. Everything returned to Node is plain JSON (flat number arrays).
// Loaded by test/harness/index.html through the import map; Node drives it via page.evaluate.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from '@pixiv/three-vrm';

// name -> { gltf, vrm }
export const models = new Map();
const springColliderGroups = new WeakMap();
const tmpV = new THREE.Vector3();
const tmpQ = new THREE.Quaternion();
const tmpM = new THREE.Matrix4();

const model = (name) => {
  const m = models.get(name);
  if (!m) throw new Error(`no model loaded as "${name}"`);
  return m;
};

// Every Mesh (primitive) under the glTF node named nodeName (the node itself or its descendants).
function primitivesOf(root, nodeName) {
  const top = root.getObjectByName(nodeName);
  if (!top) throw new Error(`no node "${nodeName}"`);
  const out = [];
  top.traverse((o) => {
    if (o.isMesh) out.push(o);
  });
  if (out.length === 0) throw new Error(`node "${nodeName}" has no mesh`);
  return out;
}

const worldOf = (o) => {
  o.getWorldPosition(tmpV);
  o.getWorldQuaternion(tmpQ);
  return { position: tmpV.toArray(), quaternion: tmpQ.toArray() };
};

// Recompute matrices like the renderer, plus spring-bone matrices (matrixAutoUpdate=false).
function sync(gltf) {
  gltf.scene.traverse((o) => {
    if (!o.matrixAutoUpdate) o.updateMatrix();
  });
  gltf.scene.updateMatrixWorld(true);
}

export async function load(name, url) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(`${url} has no VRM extension`);
  // No-op for VRM 1.0; rotates VRM 0.x models by 180 degrees around Y, as consumers do.
  VRMUtils.rotateVRM0(vrm);
  const rest = new Map();
  for (const bone of Object.values(VRMHumanBoneName)) {
    const node = vrm.humanoid.getRawBoneNode(bone);
    if (node) rest.set(bone, { position: node.position.clone(), quaternion: node.quaternion.clone() });
  }
  models.set(name, { gltf, vrm, rest });
  sync(gltf);
  return { metaVersion: vrm.meta.metaVersion };
}

// Restores every raw humanoid bone to its rest local position and rotation (as loaded).
export function resetPose(name) {
  const { gltf, vrm, rest } = model(name);
  for (const [bone, r] of rest) {
    const node = vrm.humanoid.getRawBoneNode(bone);
    node.position.copy(r.position);
    node.quaternion.copy(r.quaternion);
  }
  sync(gltf);
}

// Sets raw humanoid bones to the given world matrices (column-major glTF; humanoid bone name keys).
// Applied parent-first: local = parentWorld^-1 * world, decomposed into position and rotation.
// Unlisted bones keep their local transform, so they follow their (possibly posed) parents.
export function setWorldPose(name, entries) {
  const { gltf, vrm } = model(name);
  const items = entries.map(({ bone, world }) => {
    const node = vrm.humanoid.getRawBoneNode(bone);
    if (!node) throw new Error(`no raw bone "${bone}"`);
    return { node, world: new THREE.Matrix4().fromArray(world) };
  });
  const depth = (o) => {
    let d = 0;
    for (let p = o.parent; p; p = p.parent) d++;
    return d;
  };
  items.sort((a, b) => depth(a.node) - depth(b.node));
  for (const { node, world } of items) {
    node.parent.updateWorldMatrix(true, false);
    const local = new THREE.Matrix4().copy(node.parent.matrixWorld).invert().multiply(world);
    local.decompose(node.position, node.quaternion, node.scale);
    node.updateMatrix();
  }
  sync(gltf);
}

// World matrix of a humanoid bone (column-major, glTF space), for comparison with the reference frames.
export function humanoidBoneMatrixWorld(name, bone) {
  const { gltf, vrm } = model(name);
  const node = vrm.humanoid.getRawBoneNode(bone);
  if (!node) throw new Error(`no bone "${bone}"`);
  sync(gltf);
  return node.matrixWorld.toArray();
}

// Raw glTF node name per humanoid bone (three.js sanitizes names, e.g. "spine.002" -> "spine002", so the
// name is read from the glTF JSON node via the parser association).
export function humanoidBones(name) {
  const { gltf, vrm } = model(name);
  if (!vrm.humanoid) return null;
  const nodes = gltf.parser.json.nodes;
  const rawName = (node) => {
    const idx = node ? gltf.parser.associations.get(node)?.nodes : undefined;
    return idx === undefined ? null : nodes[idx].name;
  };
  return Object.fromEntries(
    Object.values(VRMHumanBoneName).map((bone) => [bone, rawName(vrm.humanoid.getRawBoneNode(bone))]),
  );
}

// Sets normalized bone rotations (quaternion [x,y,z,w]) and applies them to the raw bones.
export function setPose(name, entries) {
  const { gltf, vrm } = model(name);
  for (const { bone, quaternion } of entries) {
    const node = vrm.humanoid.getNormalizedBoneNode(bone);
    if (!node) throw new Error(`no normalized bone "${bone}"`);
    node.quaternion.fromArray(quaternion);
  }
  vrm.humanoid.update();
  sync(gltf);
}

export function setExpression(name, expression, weight) {
  const { vrm } = model(name);
  if (!vrm.expressionManager.getExpression(expression)) throw new Error(`no expression "${expression}"`);
  vrm.expressionManager.setValue(expression, weight);
}

export function expressionState(name, expression) {
  const { vrm } = model(name);
  const e = vrm.expressionManager.getExpression(expression);
  if (!e) throw new Error(`no expression "${expression}"`);
  return {
    weight: e.weight,
    outputWeight: e.outputWeight,
    isBinary: e.isBinary,
    overrideBlink: e.overrideBlink,
    overrideLookAt: e.overrideLookAt,
    overrideMouth: e.overrideMouth,
  };
}

// Full VRM update (humanoid, look-at, expressions, spring bones, node constraints, materials).
export function update(name, dt) {
  const { gltf, vrm } = model(name);
  vrm.update(dt);
  sync(gltf);
}

export function lookAt(name, target) {
  const { gltf, vrm } = model(name);
  if (!vrm.lookAt) throw new Error('model has no lookAt');
  vrm.lookAt.lookAt(new THREE.Vector3().fromArray(target));
  sync(gltf);
}

export function lookAtAngles(name) {
  const { vrm } = model(name);
  return { yaw: vrm.lookAt.yaw, pitch: vrm.lookAt.pitch };
}

export function nodeWorld(name, nodeName) {
  const { gltf } = model(name);
  const o = gltf.scene.getObjectByName(nodeName);
  if (!o) throw new Error(`no node "${nodeName}"`);
  return worldOf(o);
}

export function humanBoneWorld(name, bone) {
  const { vrm } = model(name);
  const node = vrm.humanoid.getRawBoneNode(bone);
  if (!node) throw new Error(`no bone "${bone}"`);
  return worldOf(node);
}

// All vertices of the node's primitives, concatenated, in world space. Morph targets are summed on
// the CPU from morphTargetInfluences, then skinned with SkinnedMesh.applyBoneTransform.
export function vertexPositions(name, nodeName) {
  const { gltf } = model(name);
  const v = new THREE.Vector3();
  const d = new THREE.Vector3();
  const out = [];
  for (const mesh of primitivesOf(gltf.scene, nodeName)) {
    const position = mesh.geometry.attributes.position;
    const influences = mesh.morphTargetInfluences ?? [];
    const morphs = mesh.geometry.morphAttributes.position ?? [];
    if (mesh.isSkinnedMesh) mesh.skeleton.update();
    for (let i = 0; i < position.count; i++) {
      v.fromBufferAttribute(position, i);
      influences.forEach((w, k) => {
        if (w !== 0) v.addScaledVector(d.fromBufferAttribute(morphs[k], i), w);
      });
      if (mesh.isSkinnedMesh) mesh.applyBoneTransform(i, v);
      v.applyMatrix4(mesh.matrixWorld);
      out.push(v.x, v.y, v.z);
    }
  }
  return out;
}

// Rest (un-morphed, un-skinned) normals in glTF space, concatenated per primitive.
export function restNormals(name, nodeName) {
  const { gltf } = model(name);
  return primitivesOf(gltf.scene, nodeName).flatMap((mesh) => Array.from(mesh.geometry.attributes.normal.array));
}

// Per primitive: material name, rest triangle centroids (glTF space) and effective world face normals.
// Effective normal = winding cross product, sign-flipped for negative-determinant matrixWorld (viewer rule).
export function primitiveTriangles(name, nodeName) {
  const { gltf } = model(name);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  return primitivesOf(gltf.scene, nodeName).map((mesh) => {
    const pos = mesh.geometry.attributes.position;
    const index = mesh.geometry.index;
    const n = index ? index.count : pos.count;
    const sign = Math.sign(mesh.matrixWorld.determinant());
    const centroids = [];
    const normals = [];
    for (let t = 0; t < n; t += 3) {
      const [i0, i1, i2] = index ? [index.getX(t), index.getX(t + 1), index.getX(t + 2)] : [t, t + 1, t + 2];
      a.fromBufferAttribute(pos, i0);
      b.fromBufferAttribute(pos, i1);
      c.fromBufferAttribute(pos, i2);
      centroids.push((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3);
      a.applyMatrix4(mesh.matrixWorld);
      b.applyMatrix4(mesh.matrixWorld);
      c.applyMatrix4(mesh.matrixWorld);
      const e = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a));
      e.multiplyScalar(sign).normalize();
      normals.push(e.x, e.y, e.z);
    }
    const material = [mesh.material].flat()[0];
    return { material: material?.name ?? null, count: n / 3, centroids, normals };
  });
}

export function morphWeights(name, nodeName) {
  const { gltf } = model(name);
  const mesh = primitivesOf(gltf.scene, nodeName)[0];
  const dict = mesh.morphTargetDictionary ?? {};
  return Object.fromEntries(Object.entries(dict).map(([key, index]) => [key, mesh.morphTargetInfluences[index]]));
}

// Material by glTF name: base color, opacity and every texture's UV offset/repeat (any texture slot).
export function materialState(name, materialName) {
  const { gltf } = model(name);
  let found = null;
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    for (const m of [o.material].flat()) if (!found && m?.name === materialName) found = m;
  });
  if (!found) throw new Error(`no material "${materialName}"`);
  const textures = {};
  for (const key of TEXTURE_SLOTS) {
    const value = found[key];
    if (value?.isTexture) textures[key] = { offset: value.offset.toArray(), repeat: value.repeat.toArray() };
  }
  return { color: found.color.toArray(), opacity: found.opacity, textures };
}

// Texture slots of three.js and three-vrm materials that take a UV transform (three-vrm TextureTransformBind list).
// Read by name: MToon textures are accessors and do not appear in Object.entries.
const TEXTURE_SLOTS = [
  'map', 'normalMap', 'emissiveMap', 'bumpMap', 'displacementMap', 'roughnessMap', 'metalnessMap', 'alphaMap',
  'specularMap', 'shadeMultiplyTexture', 'rimMultiplyTexture', 'outlineWidthMultiplyTexture', 'uvAnimationMaskTexture',
];

// Spring joints in VRM order. child is null for a tail-only joint (VRM 0.x end joint has no child node).
export function springJoints(name) {
  const { vrm } = model(name);
  return [...vrm.springBoneManager.joints].map((j) => ({
    bone: j.bone.name,
    child: j.child?.name ?? null,
    hitRadius: j.settings.hitRadius,
    stiffness: j.settings.stiffness,
    gravityPower: j.settings.gravityPower,
    gravityDir: j.settings.gravityDir.toArray(),
    dragForce: j.settings.dragForce,
    bonePos: worldOf(j.bone).position,
    tailPos: j.child ? worldOf(j.child).position : null,
  }));
}

// Turns collision on/off for every spring joint (off = the no-collider control on the same model).
export function setSpringColliders(name, enabled) {
  const { vrm } = model(name);
  for (const j of vrm.springBoneManager.joints) {
    if (!springColliderGroups.has(j)) springColliderGroups.set(j, j.colliderGroups);
    j.colliderGroups = enabled ? springColliderGroups.get(j) : [];
  }
}

// Collider shapes in world space, read from the scene (independent of the joints' colliderGroups, so the
// no-collider control still sees them). The collider matrix is refreshed first; three-vrm only refreshes it
// during a spring update. head = collider matrix position (includes offset); capsule tail = head + R*(tail - offset).
export function colliders(name) {
  const { gltf } = model(name);
  const out = [];
  gltf.scene.traverse((c) => {
    if (!c.shape || !c.colliderMatrix) return;
    c.updateWorldMatrix(true, false);
    const s = c.shape;
    const m = c.colliderMatrix;
    const head = new THREE.Vector3().setFromMatrixPosition(m);
    const rot = tmpM.copy(m).setPosition(0, 0, 0);
    const tail =
      s.type === 'capsule'
        ? head.clone().add(new THREE.Vector3().subVectors(s.tail, s.offset).applyMatrix4(rot))
        : head.clone();
    out.push({ node: c.parent?.name ?? null, type: s.type, radius: s.radius, inside: s.inside, head: head.toArray(), tail: tail.toArray() });
  });
  return out;
}

export function resetSprings(name) {
  const { gltf, vrm } = model(name);
  vrm.springBoneManager.reset();
  sync(gltf);
}

// Decodes every glTF image from its bufferView through createImageBitmap (not the loader's image cache,
// which recurses on these files). Returns the decoded size per image index.
export async function imageInfo(name) {
  const { gltf } = model(name);
  const images = gltf.parser.json.images ?? [];
  const out = [];
  for (let i = 0; i < images.length; i++) {
    const bv = images[i].bufferView;
    if (bv === undefined) {
      out.push({ index: i, mimeType: images[i].mimeType ?? null, bufferView: null, width: null, height: null });
      continue;
    }
    const bytes = await gltf.parser.getDependency('bufferView', bv);
    const bitmap = await createImageBitmap(new Blob([bytes], { type: images[i].mimeType }));
    out.push({ index: i, mimeType: images[i].mimeType ?? null, bufferView: bv, width: bitmap.width, height: bitmap.height });
    bitmap.close();
  }
  return out;
}

window.potter = {
  load,
  humanoidBones,
  setPose,
  resetPose,
  setWorldPose,
  humanoidBoneMatrixWorld,
  setExpression,
  expressionState,
  update,
  lookAt,
  lookAtAngles,
  nodeWorld,
  humanBoneWorld,
  vertexPositions,
  restNormals,
  primitiveTriangles,
  morphWeights,
  materialState,
  springJoints,
  setSpringColliders,
  colliders,
  resetSprings,
  imageInfo,
};
window.potterReady = true;
