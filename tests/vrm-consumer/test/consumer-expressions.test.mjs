// Checks 3 (morph vertices), 4 (shared cloth material binds), 5 (procedural overrides + isBinary),
// 6 (bone look-at vs expression look-at). Expectations come from spec.mjs and the VRM 1.0 expressions spec.
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { openHarness } from './support/browser.mjs';
import {
  api,
  readReference,
  fixtureUrl,
  FILES,
  MESHES,
  TOL,
  toGltf,
  correspondence,
  triples,
  assertVecClose,
  approx,
  saturate,
} from './support/fixture.mjs';
import { EXPRESSIONS, MORPHS, LOOKAT, MATERIALS } from '../fixture/spec.mjs';

const reference = await readReference();
const head = MESHES.head;

const BLINK = ['blink', 'blinkLeft', 'blinkRight'];
const MOUTH = ['aa', 'ih', 'ou', 'ee', 'oh'];
const LOOK = ['lookUp', 'lookDown', 'lookLeft', 'lookRight'];
const OVERRIDE_GROUP = { overrideBlink: BLINK, overrideMouth: MOUTH, overrideLookAt: LOOK };

let h;
let potter;
before(async () => {
  h = await openHarness();
  potter = api(h.page);
});
after(async () => {
  await h?.close();
});

// Look-at target 2 m away from the look-at origin (head bone world position + head rotation * spec offset).
// Direction: yaw > 0 towards the character's left (+X), pitch > 0 looks down (-Y).
async function lookTargetFor(name, yawDeg, pitchDeg, offset) {
  const headNode = await potter.nodeWorld(name, 'head');
  const origin = new THREE.Vector3()
    .fromArray(headNode.position)
    .add(new THREE.Vector3().fromArray(toGltf(offset)).applyQuaternion(new THREE.Quaternion().fromArray(headNode.quaternion)));
  const yaw = THREE.MathUtils.degToRad(yawDeg);
  const pitch = THREE.MathUtils.degToRad(pitchDeg);
  const dir = new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
  return origin.add(dir.multiplyScalar(2)).toArray();
}

// Morph -> the single preset expression that drives it at weight 1 (blink -> blink, look_left -> lookLeft, ...).
const morphExpression = Object.fromEntries(
  Object.entries(EXPRESSIONS)
    .filter(([, s]) => s.morphs.length === 1 && s.morphs[0][1] === 1 && !s.isBinary)
    .map(([expr, s]) => [s.morphs[0][0], expr]),
);

describe('check 3: morph vertices vs Potter reference', () => {
  it('every shape key has exactly one preset expression at weight 1', () => {
    assert.deepEqual(Object.keys(morphExpression).sort(), [...MORPHS].sort());
  });

  for (const morph of MORPHS) {
    it(`${morph}: vertices follow rest + w * delta; weight 0 is exactly the base`, async () => {
      const deltas = reference.morphs?.[morph];
      assert.ok(deltas, `reference.json has no morph deltas for ${morph}`);
      await potter.load('m', fixtureUrl(FILES.v1Full));
      const base = await potter.vertexPositions('m', head);
      const k = correspondence(base, reference.meshes[head].restGltf, head);
      for (const w of [0, 0.25, 0.5, 1, 0]) {
        await potter.setExpression('m', morphExpression[morph], w);
        await potter.update('m', 0);
        const weights = await potter.morphWeights('m', head);
        approx(weights[morph], w, TOL.morph, `${morph} influence at ${w}`);
        const posed = triples(await potter.vertexPositions('m', head));
        if (w === 0) {
          assert.deepEqual(posed.flat(), triples(base).flat(), `${morph}: weight 0 is not exactly the base`);
          continue;
        }
        reference.meshes[head].restGltf.forEach((rest, i) => {
          const expected = rest.map((c, a) => c + w * deltas[i][a]);
          assertVecClose(posed[k[i]], expected, TOL.morph, `${morph} w=${w} potter vertex ${i}`);
        });
      }
    });
  }
});

describe('check 4: shared cloth material binds', () => {
  const bind = EXPRESSIONS.cloth_shift;
  const colorBind = bind.materialColorBinds[0];
  const uvBind = bind.textureTransformBinds[0];
  const base = MATERIALS.cloth.baseColor;

  it('color = base + w*(target - base); UV transform applied once; base at 0', async () => {
    await potter.load('c', fixtureUrl(FILES.v1Full));
    for (const w of [0, 0.25, 0.5, 1, 0]) {
      await potter.setExpression('c', 'cloth_shift', w);
      await potter.update('c', 0);
      const m = await potter.materialState('c', 'cloth');
      const expectedColor = [0, 1, 2].map((a) => base[a] + w * (colorBind.targetValue[a] - base[a]));
      assertVecClose(m.color, expectedColor, TOL.color, `cloth color at w=${w}`);
      approx(m.opacity, base[3], TOL.color, `cloth opacity at w=${w}`);
      // All UV-accessed textures of the material share one transform.
      assert.deepEqual(Object.keys(m.textures).sort(), ['map', 'uvAnimationMaskTexture'], 'cloth textures');
      for (const t of Object.values(m.textures)) {
        assertVecClose(t.offset, uvBind.offset.map((o) => w * o), TOL.uv, `uv offset at w=${w}`);
        // Default scale is [1,1]; applied scale is default + w * (target - default).
        assertVecClose(t.repeat, [0, 1].map((a) => 1 + w * (uvBind.scale[a] - 1)), TOL.uv, `uv scale at w=${w}`);
      }
    }
  });
});

describe('check 5: procedural overrides and isBinary against the VRM 1.0 expressions spec', () => {
  // Output: isBinary gives 1 when weight > 0.5 (strict), else 0; otherwise the saturated weight.
  // Override amount: block -> 1 while output > 0; blend -> output. Group multiplier = max(0, 1 - sum of amounts).
  // A binary expression that receives any effect (multiplier < 1) is suppressed completely.
  function expectedMorphs(weights) {
    const o = {};
    for (const [e, s] of Object.entries(EXPRESSIONS)) {
      const w = weights[e] ?? 0;
      o[e] = s.isBinary ? (w > 0.5 ? 1 : 0) : saturate(w);
    }
    const multiplier = {};
    for (const [key, group] of Object.entries(OVERRIDE_GROUP)) {
      let sum = 0;
      for (const [e, s] of Object.entries(EXPRESSIONS)) {
        const mode = s[key] ?? 'none';
        if (mode === 'block') sum += o[e] > 0 ? 1 : 0;
        else if (mode === 'blend') sum += o[e];
      }
      multiplier[key] = { group, value: Math.max(0, 1 - sum) };
    }
    const out = {};
    for (const [e, s] of Object.entries(EXPRESSIONS)) {
      let m = 1;
      for (const { group, value } of Object.values(multiplier)) if (group.includes(e)) m *= value;
      let actual = o[e] * m;
      if (s.isBinary && m < 1) actual = 0;
      for (const [morph, bindWeight] of s.morphs) out[morph] = (out[morph] ?? 0) + actual * bindWeight;
    }
    return out;
  }

  it('override and isBinary fields read back from three-vrm match the spec', async () => {
    await potter.load('o', fixtureUrl(FILES.v1LookAt));
    for (const [e, s] of Object.entries(EXPRESSIONS)) {
      const st = await potter.expressionState('o', e);
      assert.equal(st.isBinary, !!s.isBinary, `${e} isBinary`);
      assert.equal(st.overrideBlink, s.overrideBlink ?? 'none', `${e} overrideBlink`);
      assert.equal(st.overrideMouth, s.overrideMouth ?? 'none', `${e} overrideMouth`);
      assert.equal(st.overrideLookAt, s.overrideLookAt ?? 'none', `${e} overrideLookAt`);
    }
  });

  for (const surprised of [0.4, 0.5, 0.9]) {
    it(`happy+blink+aa+sad+surprised=${surprised}+lookAt: outputs and morphs follow the spec`, async () => {
      await potter.load('o', fixtureUrl(FILES.v1LookAt));
      const weights = { happy: 0.6, blink: 0.8, aa: 0.7, sad: 0.3, surprised };
      for (const e of Object.keys(EXPRESSIONS)) {
        if (!(e in weights) && !LOOK.includes(e)) await potter.setExpression('o', e, 0);
      }
      for (const [e, w] of Object.entries(weights)) await potter.setExpression('o', e, w);
      await potter.lookAt('o', await lookTargetFor('o', 20, 0, LOOKAT.v1LookAt.offsetFromHeadBone));
      await potter.update('o', 0);
      const full = { ...weights };
      for (const e of LOOK) full[e] = (await potter.expressionState('o', e)).weight;
      for (const e of Object.keys(EXPRESSIONS)) {
        const st = await potter.expressionState('o', e);
        const w = full[e] ?? 0;
        const expectedOutput = EXPRESSIONS[e].isBinary ? (w > 0.5 ? 1 : 0) : saturate(w);
        approx(st.outputWeight, expectedOutput, TOL.weight, `${e} outputWeight`);
      }
      const morphs = await potter.morphWeights('o', head);
      const expected = expectedMorphs(full);
      for (const morph of MORPHS) approx(morphs[morph] ?? 0, expected[morph] ?? 0, TOL.morph, `morph ${morph}`);
    });
  }
});

describe('check 6: bone look-at rotates eyes, expression look-at drives look weights', () => {
  const angleDeg = (a, b) => {
    const qa = new THREE.Quaternion().fromArray(a);
    const qb = new THREE.Quaternion().fromArray(b);
    return THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(qa.dot(qb)))));
  };
  const rangeValue = (range, deg) => range.outputScale * saturate(Math.abs(deg) / range.inputMaxValue);

  for (const [yawDeg, pitchDeg] of [
    [15, 0],
    [-15, 0],
    [45, 0],
    [0, 20],
    [0, -20],
  ]) {
    it(`bone look-at yaw=${yawDeg} pitch=${pitchDeg}: eyes rotate by the spec range maps`, async () => {
      const spec = LOOKAT.v1Full;
      await potter.load('b', fixtureUrl(FILES.v1Full));
      const restLeft = (await potter.humanBoneWorld('b', 'leftEye')).quaternion;
      const restRight = (await potter.humanBoneWorld('b', 'rightEye')).quaternion;
      await potter.lookAt('b', await lookTargetFor('b', yawDeg, pitchDeg, spec.offsetFromHeadBone));
      await potter.update('b', 0);
      const angles = await potter.lookAtAngles('b');
      approx(angles.yaw, yawDeg, TOL.angleDeg, 'look-at yaw');
      approx(angles.pitch, pitchDeg, TOL.angleDeg, 'look-at pitch');
      // Single-axis rotations, so the eye's rotation angle is the range-map output for that axis.
      // Left eye uses the outer map for yaw > 0 and the inner map for yaw < 0; the right eye is mirrored.
      const horiz = (leftEye) => {
        const outer = leftEye ? yawDeg > 0 : yawDeg < 0;
        return rangeValue(outer ? spec.rangeMapHorizontalOuter : spec.rangeMapHorizontalInner, yawDeg);
      };
      const vert = rangeValue(pitchDeg > 0 ? spec.rangeMapVerticalDown : spec.rangeMapVerticalUp, pitchDeg);
      const eyeLeft = (await potter.humanBoneWorld('b', 'leftEye')).quaternion;
      const eyeRight = (await potter.humanBoneWorld('b', 'rightEye')).quaternion;
      approx(angleDeg(restLeft, eyeLeft), Math.hypot(horiz(true), vert), TOL.angleDeg, 'left eye rotation');
      approx(angleDeg(restRight, eyeRight), Math.hypot(horiz(false), vert), TOL.angleDeg, 'right eye rotation');
    });
  }

  for (const [yawDeg, pitchDeg] of [
    [15, 0],
    [-45, 0],
    [0, -20],
  ]) {
    it(`expression look-at yaw=${yawDeg} pitch=${pitchDeg}: look weights follow the spec range maps`, async () => {
      const spec = LOOKAT.v1LookAt;
      await potter.load('e', fixtureUrl(FILES.v1LookAt));
      await potter.lookAt('e', await lookTargetFor('e', yawDeg, pitchDeg, spec.offsetFromHeadBone));
      await potter.update('e', 0);
      const angles = await potter.lookAtAngles('e');
      approx(angles.yaw, yawDeg, TOL.angleDeg, 'look-at yaw');
      approx(angles.pitch, pitchDeg, TOL.angleDeg, 'look-at pitch');
      const weight = async (e) => (await potter.expressionState('e', e)).weight;
      approx(await weight('lookLeft'), yawDeg > 0 ? rangeValue(spec.rangeMapHorizontalOuter, yawDeg) : 0, TOL.weight, 'lookLeft');
      approx(await weight('lookRight'), yawDeg < 0 ? rangeValue(spec.rangeMapHorizontalOuter, yawDeg) : 0, TOL.weight, 'lookRight');
      approx(await weight('lookDown'), pitchDeg > 0 ? rangeValue(spec.rangeMapVerticalDown, pitchDeg) : 0, TOL.weight, 'lookDown');
      approx(await weight('lookUp'), pitchDeg < 0 ? rangeValue(spec.rangeMapVerticalDown, pitchDeg) : 0, TOL.weight, 'lookUp');
    });
  }

  it('expression look-at model has no eye bones, so the eye node does not move', async () => {
    await potter.load('e', fixtureUrl(FILES.v1LookAt));
    const bones = await potter.humanoidBones('e');
    assert.equal(bones.leftEye, null);
    assert.equal(bones.rightEye, null);
    const before = (await potter.nodeWorld('e', 'eye_l')).position;
    await potter.lookAt('e', await lookTargetFor('e', 30, 0, LOOKAT.v1LookAt.offsetFromHeadBone));
    await potter.update('e', 0);
    assertVecClose((await potter.nodeWorld('e', 'eye_l')).position, before, TOL.position, 'eye_l position');
  });
});
