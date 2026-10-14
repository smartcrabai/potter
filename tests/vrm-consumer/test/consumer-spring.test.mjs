// Check 7: hair spring bones against the spec.mjs joint parameters and colliders.
// Tails must stay outside the head sphere and the body capsule (end caps included) by radius + hitRadius
// (minus TOL.position); the no-collider control on the same motion must penetrate; bone length is preserved;
// reset returns to rest; varying dt stays finite.
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { openHarness } from './support/browser.mjs';
import { api, fixtureUrl, readGlbJson, FILES, TOL, toGltf, assertVecClose, approx } from './support/fixture.mjs';
import { SPRING, COLLIDERS, HEAD_BONE_POTTER_COLUMNS } from '../fixture/spec.mjs';

const M = 'spring';
const FRAMES = 360;
const DT = 1 / 60;

let h;
let potter;
before(async () => {
  h = await openHarness();
  potter = api(h.page);
});
after(async () => {
  await h?.close();
});

// Bone-local offset -> glTF world displacement: the offset is in the bone's Potter local frame (spec.mjs header),
// so rotate it by the bone's Potter world rotation, then change basis Potter -> glTF.
function boneOffsetToGltfWorld(columns, offset) {
  const p = [0, 1, 2].map((r) => columns[0][r] * offset[0] + columns[1][r] * offset[1] + columns[2][r] * offset[2]);
  return vec(toGltf(p));
}

const vec = (a) => new THREE.Vector3().fromArray(a);
// Distance from p to the collider's centre (sphere) or axis segment (capsule), without the radius.
function shapeDistance(p, c) {
  const P = vec(p);
  if (c.type !== 'capsule') return P.distanceTo(vec(c.head));
  const a = vec(c.head);
  const ab = vec(c.tail).sub(a);
  const t = Math.min(1, Math.max(0, P.clone().sub(a).dot(ab) / ab.lengthSq()));
  return P.distanceTo(a.clone().addScaledVector(ab, t));
}
const finite = (xs) => xs.every((x) => Number.isFinite(x));

// Deterministic motion: a large head nod (2.5 rad, 0.8 s period) plus a spine sway. Probe data for the control
// (no colliders): 1.5 rad at 0.8 s -> worst clearance +0.054 m; 2.5 rad at 0.8 s -> -0.130 m (penetrates).
// Gravity and drag are the spec.mjs joint values; the stimulus is chosen to exercise the collider, not fitted.
function motion(t) {
  const q = (x, y, z) => new THREE.Quaternion().setFromEuler(new THREE.Euler(x, y, z)).toArray();
  return [
    { bone: 'head', quaternion: q(2.5 * Math.sin((2 * Math.PI * t) / 0.8), 0, 0) },
    { bone: 'spine', quaternion: q(0, 0.2 * Math.sin((2 * Math.PI * t) / 1.5), 0.2 * Math.sin((2 * Math.PI * t) / 0.9)) },
  ];
}

// Signed distance from a tail point to a collider surface, minus the joint hitRadius; >= 0 means outside.
// Sphere: centre = collider.head. Capsule: segment head..tail including both end caps.
const margin = (tail, collider, hitRadius) => shapeDistance(tail, collider) - (collider.radius + hitRadius);

// Runs the motion on the loaded model; returns the worst collider margin over all tails and frames,
// and the worst deviation of the bone-to-tail length from its rest value.
async function runMotion(name) {
  const lengths = (await potter.springJoints(name)).map((j) => vec(j.bonePos).distanceTo(vec(j.tailPos)));
  let worst = Infinity;
  let lengthError = 0;
  for (let f = 0; f < FRAMES; f++) {
    await potter.setPose(name, motion(f * DT));
    await potter.update(name, DT);
    const joints = await potter.springJoints(name);
    const colliders = await potter.colliders(name);
    joints.forEach((j, i) => {
      for (const c of colliders) worst = Math.min(worst, margin(j.tailPos, c, j.hitRadius));
      lengthError = Math.max(lengthError, Math.abs(vec(j.bonePos).distanceTo(vec(j.tailPos)) - lengths[i]));
    });
  }
  return { worst, lengthError };
}

describe('spring bones (VRM 1.0 full)', () => {
  // VRM 1.0 pairs consecutive joints: the last spec joint is the tail-only end and is not simulated.
  const SIMULATED = SPRING.joints.slice(0, -1);

  it('simulated spring bones are the spec joints minus the tail-only end', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const joints = await potter.springJoints(M);
    assert.deepEqual(joints.map((j) => j.bone), SIMULATED.map((s) => s.bone));
  });

  it('file joint settings match spec.mjs SPRING for all joints (glTF JSON, incl. the tail-only end)', async () => {
    const json = await readGlbJson(FILES.v1Full);
    const joints = json.extensions.VRMC_springBone.springs[0].joints;
    assert.equal(joints.length, SPRING.joints.length, 'joints listed in the file');
    joints.forEach((j, i) => {
      const s = SPRING.joints[i];
      assert.equal(json.nodes[j.node].name, s.bone, `joint ${i} node`);
      approx(j.stiffness, s.stiffness, TOL.weight, `${s.bone} stiffness`);
      approx(j.dragForce, s.dragForce, TOL.weight, `${s.bone} dragForce`);
      approx(j.gravityPower, s.gravityPower, TOL.weight, `${s.bone} gravityPower`);
      approx(j.hitRadius, s.hitRadius, TOL.position, `${s.bone} hitRadius`);
      assertVecClose(j.gravityDir, toGltf(s.gravityDir), TOL.weight, `${s.bone} gravityDir (glTF)`);
    });
  });

  it('simulated joint parameters (three-vrm) match spec.mjs SPRING', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const joints = await potter.springJoints(M);
    for (const s of SIMULATED) {
      const j = joints.find((x) => x.bone === s.bone);
      assert.ok(j, `no simulated spring bone for ${s.bone}`);
      approx(j.stiffness, s.stiffness, TOL.weight, `${s.bone} stiffness`);
      approx(j.dragForce, s.dragForce, TOL.weight, `${s.bone} dragForce`);
      approx(j.gravityPower, s.gravityPower, TOL.weight, `${s.bone} gravityPower`);
      approx(j.hitRadius, s.hitRadius, TOL.position, `${s.bone} hitRadius`);
      assertVecClose(vec(j.gravityDir).toArray(), toGltf(s.gravityDir), TOL.weight, `${s.bone} gravityDir (glTF)`);
    }
  });

  it('collider shapes match spec.mjs COLLIDERS in world space', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const [sphere, capsule] = COLLIDERS.v1Full;
    const headNode = await potter.nodeWorld(M, sphere.bone);
    const spineNode = await potter.nodeWorld(M, capsule.bone);
    const cols = await potter.colliders(M);
    assert.equal(cols.length, 2, 'collider count');

    const centre = vec(headNode.position).add(boneOffsetToGltfWorld(HEAD_BONE_POTTER_COLUMNS, sphere.shape.offset));
    assert.equal(cols[0].type, 'sphere');
    approx(cols[0].radius, sphere.shape.radius, TOL.position, 'head sphere radius');
    assertVecClose(cols[0].head, centre.toArray(), TOL.position, 'head sphere centre');

    // Capsule: head = node + R*offset; tail = head + R*(tail - offset).
    const spineQ = new THREE.Quaternion().fromArray(spineNode.quaternion);
    const off = vec(toGltf(capsule.shape.offset));
    const head = vec(spineNode.position).add(off.clone().applyQuaternion(spineQ));
    const tail = head.clone().add(vec(toGltf(capsule.shape.tail)).sub(off).applyQuaternion(spineQ));
    assert.equal(cols[1].type, 'capsule');
    approx(cols[1].radius, capsule.shape.radius, TOL.position, 'body capsule radius');
    assertVecClose(cols[1].head, head.toArray(), TOL.position, 'body capsule head');
    assertVecClose(cols[1].tail, tail.toArray(), TOL.position, 'body capsule tail');
  });
  it('simulated joint roots (bone bases) lie outside every collider by radius + hitRadius at rest', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const joints = await potter.springJoints(M);
    const cols = await potter.colliders(M);
    for (const j of joints) {
      for (const [ci, c] of cols.entries()) {
        const m = margin(j.bonePos, c, j.hitRadius);
        assert.ok(m >= 0, `joint root ${j.bone} inside collider ${ci} (${c.type}): margin ${m} m < 0`);
      }
    }
  });

  it('no-collider control penetrates a collider on the same motion', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    await potter.setSpringColliders(M, false);
    const control = await runMotion(M);
    assert.ok(control.worst < -TOL.position, `control never penetrates (worst margin ${control.worst}); collider checks would be vacuous`);
  });

  // Per sample (frame, joint, collider): distance from the joint root and from the tail to the collider shape
  // (sphere centre, or capsule segment), the joint's rest bone length L and the collider radius R and hitRadius h.
  async function collisionRecords(name) {
    const rest = await potter.springJoints(name);
    const L = rest.map((j) => vec(j.bonePos).distanceTo(vec(j.tailPos)));
    const recs = [];
    for (let f = 0; f < FRAMES; f++) {
      await potter.setPose(name, motion(f * DT));
      await potter.update(name, DT);
      const joints = await potter.springJoints(name);
      const cols = await potter.colliders(name);
      joints.forEach((j, ji) =>
        cols.forEach((c, ci) =>
          recs.push({
            f, joint: j.bone, ci, type: c.type, L: L[ji], R: c.radius, h: j.hitRadius,
            dRoot: shapeDistance(j.bonePos, c),
            dTail: shapeDistance(j.tailPos, c),
          }),
        ),
      );
    }
    return recs;
  }

  // VRMC_springBone collision: a tail inside the collider (distance < R + h) is pushed out and then re-constrained
  // to the bone length. The re-constraint can pull it back in, but only when the root is within
  // L + R + h of the collider, so every inside tail must satisfy that (triangle inequality bound).
  it('spring tail inside a collider implies its root is within boneLength + R + hitRadius (spec length constraint)', async (t) => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const recs = await collisionRecords(M);
    const inside = recs.filter((r) => r.dTail < r.R + r.h);
    for (const r of inside) {
      assert.ok(
        r.dRoot < r.L + r.R + r.h,
        `frame ${r.f} ${r.joint} vs ${r.type}: root distance ${r.dRoot} >= L+R+h = ${r.L + r.R + r.h}`,
      );
    }
    const slack = Math.min(...inside.map((r) => r.L + r.R + r.h - r.dRoot));
    const depth = Math.max(...inside.map((r) => r.R + r.h - r.dTail));
    t.diagnostic(`samples ${recs.length}; tails inside (dTail < R+h): ${inside.length}; min slack L+R+h-dRoot ${slack}; max depth R+h-dTail ${depth}`);
    assert.ok(inside.length > 0, 'no tail entered a collider; assertion would be vacuous');
  });

  // Strict guarantee: if the root is at least L + R from the collider shape, every tail lies at least R from it
  // (the tail is at distance L from the root, and distance to the shape is 1-Lipschitz).
  it('spring tail with root at >= boneLength + R from a collider is outside it by R - TOL', async (t) => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const recs = await collisionRecords(M);
    const guaranteed = recs.filter((r) => r.dRoot >= r.L + r.R);
    for (const r of guaranteed) {
      assert.ok(
        r.dTail >= r.R - TOL.position,
        `frame ${r.f} ${r.joint} vs ${r.type}: tail distance ${r.dTail} < R - TOL = ${r.R - TOL.position}`,
      );
    }
    const minMargin = Math.min(...guaranteed.map((r) => r.dTail - r.R));
    const byType = (ty) => guaranteed.filter((r) => r.type === ty).length;
    t.diagnostic(`guaranteed samples ${guaranteed.length} (sphere ${byType('sphere')}, capsule ${byType('capsule')}); min dTail - R ${minMargin}`);
    assert.ok(guaranteed.length > 0, 'no root was far enough from any collider; assertion would be vacuous');
  });

  it('bone length is preserved while the hair moves', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const run = await runMotion(M);
    assert.ok(run.lengthError <= TOL.position, `bone length error ${run.lengthError} > ${TOL.position}`);
  });

  it('reset returns the chain to rest', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    const rest = (await potter.springJoints(M)).map((j) => j.tailPos);
    await runMotion(M);
    await potter.setPose(M, [
      { bone: 'head', quaternion: [0, 0, 0, 1] },
      { bone: 'spine', quaternion: [0, 0, 0, 1] },
    ]);
    await potter.resetSprings(M);
    const after = (await potter.springJoints(M)).map((j) => j.tailPos);
    after.forEach((p, i) => assertVecClose(p, rest[i], TOL.position, `hair joint ${i} after reset`));
  });

  it('varying dt gives finite joints and colliders', async () => {
    await potter.load(M, fixtureUrl(FILES.v1Full));
    for (const dt of [1 / 240, 1 / 30, 0.1, 0.5, 1e-4]) {
      for (let f = 0; f < 30; f++) {
        await potter.setPose(M, motion(f * dt));
        await potter.update(M, dt);
        const joints = await potter.springJoints(M);
        const cols = await potter.colliders(M);
        assert.ok(joints.every((j) => finite(j.tailPos) && finite(j.bonePos)), `non-finite joint at dt=${dt}`);
        assert.ok(cols.every((c) => finite(c.head) && finite(c.tail)), `non-finite collider at dt=${dt}`);
      }
    }
  });
});

describe('spring bones (VRM 0.x)', () => {
  it('every joint uses the single VRM 0.x parameter set (spec.mjs SPRING.v0Uniform)', async () => {
    await potter.load('v0s', fixtureUrl(FILES.v0));
    const joints = await potter.springJoints('v0s');
    assert.ok(joints.length > 0, 'v0 has no spring joints');
    const u = SPRING.v0Uniform;
    for (const j of joints) {
      approx(j.stiffness, u.stiffness, TOL.weight, `${j.bone} stiffness`);
      approx(j.dragForce, u.dragForce, TOL.weight, `${j.bone} dragForce`);
      approx(j.gravityPower, u.gravityPower, TOL.weight, `${j.bone} gravityPower`);
      approx(j.hitRadius, u.hitRadius, TOL.position, `${j.bone} hitRadius`);
    }
  });

  it('uses the head collider group only (spec.mjs SPRING_COLLIDER_GROUPS.v0)', async () => {
    await potter.load('v0s', fixtureUrl(FILES.v0));
    const [head] = COLLIDERS.v0;
    const headNode = await potter.nodeWorld('v0s', head.bone);
    const cols = await potter.colliders('v0s');
    assert.equal(cols.length, 1, 'collider count');
    assert.equal(cols[0].type, 'sphere');
    approx(cols[0].radius, head.shape.radius, TOL.position, 'head sphere radius');
    const centre = vec(headNode.position).add(boneOffsetToGltfWorld(HEAD_BONE_POTTER_COLUMNS, head.shape.offset));
    assertVecClose(cols[0].head, centre.toArray(), TOL.position, 'head sphere centre');
  });
});
