// Checks 1 (humanoid load, upright grounded rest) and 2 (skinning after posing), VRM 1.0 full and VRM 0.x.
// Pose checks use the frame-invariant delta transfer (fixture.mjs vrmPoseWorlds): they do not assume VRM bone
// axes equal Potter's. The absolute rest-frame check is kept and reports the exporter's rest axes.
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
  HUMANOID,
  TOL,
  toGltf,
  correspondence,
  triples,
  humanoidNameOf,
  toGltfMatrix,
  translationOf,
  vrmPoseWorlds,
  captureRestWorlds,
  assertVecClose,
  approx,
} from './support/fixture.mjs';
import { BONE_HEADS, BONE_ADDED } from '../fixture/spec.mjs';

const reference = await readReference();
const legs = [MESHES.legL, MESHES.legR];

// The exporter names glTF nodes Blender-style: spine_002 -> "spine.002", upper_arm_l -> "upper_arm.L".
// This is a naming convention only (no values); it maps a glTF node name back to the Potter bone id.
const potterIdOf = (gltfName) => gltfName.replace(/\.([LR])$/, (_, s) => `_${s.toLowerCase()}`).replace(/\./g, '_');

let h;
let potter;
before(async () => {
  h = await openHarness();
  potter = api(h.page);
});
after(async () => {
  await h?.close();
});

const variants = [
  { name: 'v1Full', variant: 'v1Full', file: FILES.v1Full, version: '1' },
  { name: 'v0', variant: 'v0', file: FILES.v0, version: '0' },
];

for (const { name, variant, file, version } of variants) {
  describe(`${name} (${file})`, () => {
    it('loads as humanoid with the spec bone mapping', async () => {
      const loaded = await potter.load(name, fixtureUrl(file));
      assert.equal(loaded.metaVersion, version);
      const bones = await potter.humanoidBones(name);
      assert.ok(bones, 'humanoid missing');
      for (const [human, node] of Object.entries(bones)) {
        const expected = HUMANOID[variant][human] ?? null;
        const actual = node === null ? null : potterIdOf(node);
        assert.equal(actual, expected, `humanoid ${human} (glTF node ${node})`);
      }
    });

    it('stands upright with head above hips and faces +Z (after rotateVRM0 for VRM 0.x)', async () => {
      await potter.load(name, fixtureUrl(file));
      const head = await potter.nodeWorld(name, 'head');
      const hips = await potter.nodeWorld(name, 'hips');
      assert.ok(head.position[1] > hips.position[1], 'head is not above hips');
      assertVecClose(head.position, toGltf(BONE_HEADS.head), TOL.position, 'head bone rest');
      assertVecClose(hips.position, toGltf(BONE_HEADS.hips), TOL.position, 'hips bone rest');
      const eye = await potter.nodeWorld(name, 'eye_l');
      assertVecClose(eye.position, toGltf(BONE_ADDED.eye_l.head), TOL.position, 'eye_l rest');
      assert.ok(eye.position[2] > head.position[2], 'eyes are not in front (+Z)');
    });

    it('feet touch the floor: lowest leg vertex is at Y = 0', async () => {
      await potter.load(name, fixtureUrl(file));
      const minY = Math.min(
        ...(await Promise.all(legs.map((m) => potter.vertexPositions(name, m)))).flatMap((flat) => triples(flat).map((p) => p[1])),
      );
      approx(minY, 0, TOL.position, 'lowest leg vertex Y');
    });

    it('every reference vertex of every mesh has a glTF vertex at the same rest position', async () => {
      await potter.load(name, fixtureUrl(file));
      for (const [id, mesh] of Object.entries(reference.meshes)) {
        const flat = await potter.vertexPositions(name, id);
        // The exporter may split vertices at seams/normals, so the glTF count can exceed vertexCount.
        assert.ok(flat.length / 3 >= mesh.vertexCount, `${id}: fewer glTF vertices than Potter vertices`);
        correspondence(flat, mesh.restGltf, id);
      }
    });

    if (variant === 'v1Full') {
      it('pose bone rest frames equal reference.json restWorldPotter (absolute rotation and position)', async () => {
        await potter.load(name, fixtureUrl(file));
        const names = humanoidNameOf(variant);
        for (const { bone, restWorldPotter } of reference.pose.bones) {
          const got = await potter.humanoidBoneMatrixWorld(name, names[bone]);
          assertVecClose(got, toGltfMatrix(restWorldPotter), TOL.posedPosition, `${bone} rest world matrix`);
        }
      });
    } else {
      // VRM 0.x requires every humanoid bone node to have identity local rotation and unit scale (0.0 contract).
      // Asserted on the glTF JSON, not the runtime scene (rotateVRM0 only changes the root).
      it('humanoid bone nodes have identity rotation and unit scale in the glTF JSON (VRM 0.x requirement)', async () => {
        const json = await readGlbJson(file);
        const bones = json.extensions.VRM.humanoid.humanBones;
        const bad = [];
        for (const { bone, node } of bones) {
          const n = json.nodes[node];
          const q = n.rotation ?? [0, 0, 0, 1];
          const s = n.scale ?? [1, 1, 1];
          const qBad = q.some((v, i) => Math.abs(v - [0, 0, 0, 1][i]) > TOL.weight);
          const sBad = s.some((v) => Math.abs(v - 1) > TOL.weight);
          if (qBad || sBad) bad.push({ bone, node: n.name, rotation: q, scale: s });
        }
        assert.ok(bones.length > 0, 'no humanoid bones listed');
        assert.equal(bad.length, 0, `${bad.length} of ${bones.length} humanoid bone nodes are not identity: ${JSON.stringify(bad)}`);
      });
    }

    it('pose bone rest positions equal restWorldPotter translations (frame-invariant)', async () => {
      await potter.load(name, fixtureUrl(file));
      const names = humanoidNameOf(variant);
      for (const { bone, restWorldPotter } of reference.pose.bones) {
        const got = await potter.humanoidBoneMatrixWorld(name, names[bone]);
        assertVecClose(translationOf(got), translationOf(toGltfMatrix(restWorldPotter)), TOL.position, `${bone} rest position`);
      }
    });

    it('delta-transferred pose bones reach their Potter posed world frames (frame-invariant)', async () => {
      await potter.load(name, fixtureUrl(file));
      const restWorldVrm = await captureRestWorlds(potter, name, variant, reference);
      const entries = vrmPoseWorlds(variant, reference, restWorldVrm);
      await potter.setWorldPose(name, entries);
      const names = humanoidNameOf(variant);
      for (const { bone, posedWorldPotter } of reference.pose.bones) {
        const got = await potter.humanoidBoneMatrixWorld(name, names[bone]);
        assertVecClose(translationOf(got), translationOf(toGltfMatrix(posedWorldPotter)), TOL.posedPosition, `${bone} posed position`);
        assertVecClose(got, entries.find((e) => e.bone === names[bone]).world, TOL.posedPosition, `${bone} posed world matrix`);
      }
    });

    it('skinned elbow and knee vertices match Potter after the delta-transferred reference pose', async () => {
      await potter.load(name, fixtureUrl(file));
      const blendedBones = new Set(Object.values(reference.blended).flat().flatMap((e) => e.weights.map(([b]) => b)));
      for (const bone of ['forearm_l', 'forearm_r', 'shin_l', 'shin_r']) {
        assert.ok(blendedBones.has(bone), `reference has no blended vertices for ${bone}`);
      }
      const rest = {};
      for (const id of Object.keys(reference.blended)) rest[id] = await potter.vertexPositions(name, id);
      const restWorldVrm = await captureRestWorlds(potter, name, variant, reference);
      await potter.setWorldPose(name, vrmPoseWorlds(variant, reference, restWorldVrm));
      let checked = 0;
      for (const [id, entries] of Object.entries(reference.blended)) {
        const posed = triples(await potter.vertexPositions(name, id));
        for (const e of entries) {
          const k = correspondence(rest[id], [e.restGltf], id)[0];
          assertVecClose(posed[k], e.posedGltf, TOL.posedPosition, `${id} potter vertex ${e.index}`);
          checked++;
        }
      }
      assert.ok(checked > 0, 'no blended vertices checked');
    });
  });
}
