// Second, independent consumer: the VRM Add-on for Blender (pinned in blender/vrm-addon.lock.json) imports each fixture
// with bpy.ops.import_scene.vrm. blender/probe.py reports facts; they are compared with fixture/spec.mjs, reference.json
// and the exported glTF JSON. Blender world equals Potter world (both Z-up, front -Y) and the armature object is the
// identity (checked below), so a Blender bone head is the Potter rest head.
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, rm } from 'node:fs/promises';
import { setupBlender, probeVrm } from './support/blender.mjs';
import { FILES, MESHES, fixturePath, readGlbJson, assertVecClose } from './support/fixture.mjs';
import {
  BONE_HEADS,
  COLLIDERS,
  COLLIDER_GROUPS,
  EXPRESSIONS,
  HUMANOID,
  LOOKAT,
  MATERIALS,
  META,
  MORPHS,
  SPRING,
  SPRING_COLLIDER_GROUPS,
  TOL,
  toGltf,
  V0_PRESET,
  v0GravityDir,
} from '../fixture/spec.mjs';

const SUITE = 'VRM Add-on for Blender';
const addonLock = JSON.parse(await readFile(new URL('../blender/vrm-addon.lock.json', import.meta.url), 'utf8'));
const setup = await setupBlender();

// Blender names these glTF nodes with its .L/.R suffix on import (observed in both VRM versions).
const BLENDER_BONE = { shoulder_l: 'shoulder.L', shoulder_r: 'shoulder.R', toe_l: 'toe.L', toe_r: 'toe.R' };
const blenderBone = (potterBone) => BLENDER_BONE[potterBone] ?? potterBone;

// Potter rest head of a rig bone. Right-side bones are mirror images of the left ones (left is +X).
const mirrorX = ([x, y, z]) => [-x, y, z];
function restHead(bone) {
  if (BONE_HEADS[bone]) return BONE_HEADS[bone];
  if (bone.endsWith('_r') && BONE_HEADS[`${bone.slice(0, -2)}_l`]) return mirrorX(BONE_HEADS[`${bone.slice(0, -2)}_l`]);
  return undefined;
}

// VRM 1.0 expression preset names; Blender exposes them as snake_case properties.
const VRM1_PRESETS = [
  'happy', 'angry', 'sad', 'relaxed', 'surprised', 'aa', 'ih', 'ou', 'ee', 'oh',
  'blink', 'blinkLeft', 'blinkRight', 'lookUp', 'lookDown', 'lookLeft', 'lookRight', 'neutral',
];
const snake = (name) => name.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
// Spec names (only_author, personal_non_profit) in VRM 1.0 camelCase and VRM 0.x forms (OnlyAuthor, Redistribution_Prohibited).
const camel = (s) => s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
const pascal = (s) => camel(s).replace(/^[a-z]/, (c) => c.toUpperCase());
const v0Case = (s) => s.replace(/(^|_)[a-z]/g, (c) => c.toUpperCase());
// The add-on stores a VRM 0.x vector in Blender axes by swapping the file's y and z (importer/vrm0_importer.py).
const v0AddonAxes = ([x, y, z]) => [x, z, y];
// Add-on log lines are "[VRM Add-on:LEVEL] message". Every level from WARNING up is checked against the allowlist
// (blender/warning-allowlist.json): each entry has a level, a reason, applies to one VRM version, and must still be
// produced (a stale entry fails).
const LOG_LINE = /^\[VRM Add-on:(WARNING|ERROR|CRITICAL)\] (.*)$/;
const WARNING_ALLOWLIST = JSON.parse(await readFile(new URL('../blender/warning-allowlist.json', import.meta.url), 'utf8')).map(
  (entry) => {
    assert.ok(entry.reason?.trim(), `warning allowlist entry needs a reason: ${entry.message}`);
    assert.ok(['1.0', '0.0'].includes(entry.spec), `warning allowlist entry has a bad spec: ${entry.message}`);
    assert.ok(['WARNING', 'ERROR', 'CRITICAL'].includes(entry.level), `warning allowlist entry has a bad level: ${entry.message}`);
    return { ...entry, regex: new RegExp(entry.message) };
  },
);

const MTOON_OUTLINE = Object.entries(MATERIALS)
  .filter(([, m]) => m.outlineWidth > 0)
  .map(([name]) => `MToon Outline (${name})`);

// Expected Blender properties of one VRM 1.0 expression.
function expressionFacts(s) {
  return {
    is_binary: !!s.isBinary,
    override_blink: s.overrideBlink ?? 'none',
    override_look_at: s.overrideLookAt ?? 'none',
    override_mouth: s.overrideMouth ?? 'none',
    morph_target_binds: s.morphs.map(([index, weight]) => ({ mesh: MESHES.head, index, weight })),
    material_color_binds: s.materialColorBinds?.length ?? 0,
    texture_transform_binds: s.textureTransformBinds?.length ?? 0,
  };
}

const EMPTY_PRESET = expressionFacts({ morphs: [] });

// Compares facts with expectations: same keys at every level, numbers within tol (Blender stores float32).
function assertFacts(actual, expected, tol, label) {
  if (typeof expected === 'number') {
    assert.ok(typeof actual === 'number' && Math.abs(actual - expected) <= tol, `${label}: got ${actual}, expected ${expected}`);
  } else if (Array.isArray(expected)) {
    const ok = Array.isArray(actual) && actual.length === expected.length;
    assert.ok(ok, `${label}: expected ${expected.length} items, got ${JSON.stringify(actual)}`);
    expected.forEach((e, i) => assertFacts(actual[i], e, tol, `${label}[${i}]`));
  } else if (expected && typeof expected === 'object') {
    assert.ok(actual && typeof actual === 'object', `${label}: expected an object, got ${JSON.stringify(actual)}`);
    assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${label}: keys`);
    for (const key of Object.keys(expected)) assertFacts(actual[key], expected[key], tol, `${label}.${key}`);
  } else {
    assert.equal(actual, expected, label);
  }
}

// glTF node name -> material names of its primitives (Blender object name = glTF node name).
function gltfMeshSlots(json) {
  const out = {};
  for (const node of json.nodes) {
    if (node.mesh === undefined) continue;
    const names = json.meshes[node.mesh].primitives.map((p) => json.materials[p.material].name);
    out[node.name] = [...new Set(names)];
  }
  return out;
}

const mtoonNames = (json, variant) =>
  variant === 'v0'
    ? json.extensions.VRM.materialProperties.filter((p) => p.shader === 'VRM/MToon').map((p) => p.name)
    : json.materials.filter((m) => m.extensions?.VRMC_materials_mtoon).map((m) => m.name);

const VARIANTS = [
  { variant: 'v1Full', file: FILES.v1Full, spec: '1.0' },
  { variant: 'v1LookAt', file: FILES.v1LookAt, spec: '1.0' },
  { variant: 'v0', file: FILES.v0, spec: '0.0' },
];

function runVariants() {
  for (const { variant, file, spec } of VARIANTS) {
    describe(`${file} imported by the add-on`, () => {
      let facts;
      let stderr;
      let json;
      before(async () => {
        ({ facts, stderr } = probeVrm(setup, fixturePath(file), { acceptLicensePrompt: spec === '0.0' }));
        json = await readGlbJson(file);
      });

      it(`import is FINISHED; spec_version ${spec}; add-on ${addonLock.version} in Blender 5.2.x`, () => {
        assert.equal(facts.spec_version, spec);
        assert.equal(facts.addon, addonLock.version);
        assert.match(facts.blender, /^5\.2\./);
      });

      it('add-on logs (WARNING, ERROR, CRITICAL) all match blender/warning-allowlist.json; every applicable entry is still produced', () => {
        const logged = stderr
          .split('\n')
          .map((line) => LOG_LINE.exec(line))
          .filter(Boolean)
          .map(([, level, message]) => ({ level, message }));
        const applicable = WARNING_ALLOWLIST.filter((entry) => entry.spec === spec);
        const matches = (entry, log) => entry.level === log.level && entry.regex.test(log.message);
        const unknown = logged.filter((log) => !applicable.some((entry) => matches(entry, log)));
        assert.deepEqual(unknown, [], 'unknown add-on log lines');
        for (const entry of applicable) {
          assert.ok(logged.some((log) => matches(entry, log)), `allowlist entry is stale: ${entry.message}`);
        }
        if (spec === '0.0') {
          // The license prompt names the legacy otherPermissionUrl; the allowlist only matches the message line.
          assert.ok(
            stderr.includes(`url=${META.legacy.otherPermissionUrl}\njson_key=otherPermissionUrl\n`),
            'license prompt does not name the legacy otherPermissionUrl',
          );
        }
      });

      it('humanoid: every fixture bone is mapped to its Blender bone, and nothing else', () => {
        const expected = Object.fromEntries(Object.entries(HUMANOID[variant]).map(([human, potter]) => [human, blenderBone(potter)]));
        assert.deepEqual(facts.humanoid, expected);
      });

      it('rest: armature is the identity and mapped bone heads equal the Potter rig', () => {
        assert.deepEqual(facts.armature_matrix, [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ]);
        let checked = 0;
        for (const [human, potter] of Object.entries(HUMANOID[variant])) {
          const head = restHead(potter);
          if (!head) continue;
          assertVecClose(facts.bone_heads[blenderBone(potter)], head, TOL.position, `${human} (${potter}) head`);
          checked++;
        }
        assert.ok(checked >= 14, `only ${checked} mapped bones have a Potter rest head`);
      });

      it('meshes: one Blender object per glTF mesh node, with the glTF material slots', () => {
        const slots = gltfMeshSlots(json);
        assert.deepEqual(Object.keys(facts.meshes).sort(), Object.keys(slots).sort());
        assert.deepEqual(Object.keys(facts.meshes).sort(), Object.values(MESHES).sort());
        for (const [name, names] of Object.entries(slots)) assert.deepEqual(facts.meshes[name].material_slots, names, `${name} slots`);
      });

      it('materials: glTF materials plus the add-on outline material; MToon flag only on MToon materials', () => {
        // Blender imports only materials a mesh uses; the fixture's graph material is unassigned.
        const usedIndices = new Set(json.meshes.flatMap((mesh) => mesh.primitives.map((prim) => prim.material)));
        const gltfNames = json.materials.filter((_, index) => usedIndices.has(index)).map((m) => m.name);
        assert.deepEqual(Object.keys(facts.materials).sort(), [...gltfNames, ...MTOON_OUTLINE].sort());
        const mtoon = new Set(mtoonNames(json, spec === '1.0' ? 'v1' : 'v0'));
        assert.ok(mtoon.size > 0, 'the file has no MToon material');
        const expected = Object.fromEntries([...gltfNames, ...MTOON_OUTLINE].map((name) => [name, { mtoon: mtoon.has(name) }]));
        assert.deepEqual(facts.materials, expected);
      });

      it('shape keys: head_mesh has Basis and the spec morphs; no other mesh has shape keys', () => {
        for (const [name, mesh] of Object.entries(facts.meshes)) {
          const expected = name === MESHES.head ? ['Basis', ...MORPHS].sort() : [];
          assert.deepEqual([...mesh.shape_keys].sort(), expected, `${name} shape keys`);
        }
      });

      if (spec === '1.0') {
        it('expressions: presets and custom expression bind the spec shape keys with isBinary and overrides', () => {
          const presets = {};
          for (const name of VRM1_PRESETS) {
            const s = EXPRESSIONS[name];
            presets[snake(name)] = s?.preset ? expressionFacts(s) : EMPTY_PRESET;
          }
          for (const [name, s] of Object.entries(EXPRESSIONS)) {
            if (s.preset) assert.ok(VRM1_PRESETS.includes(name), `${name} is not a VRM 1.0 preset`);
          }
          assertFacts(facts.expressions.preset, presets, TOL.morph, 'presets');
          const custom = Object.fromEntries(Object.entries(EXPRESSIONS).filter(([, s]) => !s.preset).map(([n, s]) => [n, expressionFacts(s)]));
          assertFacts(facts.expressions.custom, custom, TOL.morph, 'custom');
        });

        it('look-at: bone or expression type, offset and range maps', () => {
          const spec1 = LOOKAT[variant];
          const range = (r) => ({ input_max_value: r.inputMaxValue, output_scale: r.outputScale });
          assertFacts(
            facts.look_at,
            {
              type: spec1.lookAtType,
              offset: spec1.offsetFromHeadBone,
              range_maps: {
                horizontal_inner: range(spec1.rangeMapHorizontalInner),
                horizontal_outer: range(spec1.rangeMapHorizontalOuter),
                vertical_down: range(spec1.rangeMapVerticalDown),
                vertical_up: range(spec1.rangeMapVerticalUp),
              },
            },
            TOL.morph,
            'look_at',
          );
        });

        it('first person: no mesh annotations (the file has no firstPerson annotations)', () => {
          const annotations = json.extensions.VRMC_vrm.firstPerson?.meshAnnotations ?? [];
          assert.equal(facts.first_person.mesh_annotations, annotations.length);
          assert.equal(annotations.length, 0);
        });

        it('springs: hair joints, colliders and collider groups', () => {
          const shapeOf = (c) => (c.shape.type === 'sphere' ? 'Sphere' : 'Capsule');
          const spring = {
            name: SPRING.name,
            joints: SPRING.joints.map((j) => ({
              bone: j.bone,
              stiffness: j.stiffness,
              gravity_power: j.gravityPower,
              gravity_dir: j.gravityDir,
              drag_force: j.dragForce,
              hit_radius: j.hitRadius,
            })),
            collider_groups: SPRING_COLLIDER_GROUPS[variant].map((i) => COLLIDER_GROUPS.v1Full[i].name),
          };
          const expected = {
            springs: [spring],
            colliders: COLLIDERS.v1Full.map((c) => ({ bone: c.bone, shape: shapeOf(c), radius: c.shape.radius })),
            collider_groups: COLLIDER_GROUPS.v1Full.map((g) => ({ name: g.name, colliders: g.colliders })),
          };
          assertFacts(facts.spring, expected, TOL.morph, 'spring');
        });

        it('meta: name, version, authors and permissions match the spec', () => {
          assertFacts(
            facts.meta,
            {
              vrm_name: META.name,
              version: META.version,
              authors: META.authors,
              avatar_permission: camel(META.rights.avatarPermission),
              commercial_usage: camel(META.rights.commercialUsage),
              credit_notation: META.rights.creditNotation,
              allow_redistribution: META.rights.allowRedistribution,
              modification: META.rights.modification,
              // META grants none of these; VRM 1.0 false is the value for each.
              allow_excessively_violent_usage: false,
              allow_excessively_sexual_usage: false,
              allow_political_or_religious_usage: false,
              allow_antisocial_or_hate_usage: false,
            },
            TOL.morph,
            'meta',
          );
        });
      } else {
        it('blend shape groups: one per expression, with the VRM 0.x preset name and the spec shape keys', () => {
          const groups = Object.fromEntries(
            Object.entries(EXPRESSIONS).map(([name, s]) => [
              name,
              {
                preset: V0_PRESET[name] ?? 'unknown',
                is_binary: !!s.isBinary,
                binds: s.morphs.map(([index, weight]) => ({ mesh: MESHES.head, index, weight })),
                material_values: (s.materialColorBinds?.length ?? 0) + (s.textureTransformBinds?.length ?? 0),
              },
            ]),
          );
          const actual = Object.fromEntries(
            Object.entries(facts.expressions.blend_shape_groups).map(([name, g]) => [
              name,
              { preset: g.preset, is_binary: g.is_binary, binds: g.binds, material_values: g.material_values },
            ]),
          );
          assertFacts(actual, groups, TOL.morph, 'blend_shape_groups');
        });

        it('look-at: bone type with VRM 0.x range maps (x_range = input max, y_range = output scale)', () => {
          const spec0 = LOOKAT.v0;
          const range = (r) => ({ x_range: r.inputMaxValue, y_range: r.outputScale });
          assertFacts(
            facts.look_at,
            {
              type_name: 'Bone',
              range_maps: {
                horizontal_inner: range(spec0.rangeMapHorizontalInner),
                horizontal_outer: range(spec0.rangeMapHorizontalOuter),
                vertical_down: range(spec0.rangeMapVerticalDown),
                vertical_up: range(spec0.rangeMapVerticalUp),
              },
            },
            TOL.morph,
            'look_at',
          );
        });

        it('first person: head bone, spec look-at offset in Blender axes, no mesh annotations', () => {
          assert.equal(facts.first_person.bone, blenderBone(HUMANOID.v0.head));
          assertVecClose(facts.first_person.bone_offset, v0AddonAxes(LOOKAT.v0.offsetFromHeadBone), TOL.morph, 'first person offset');
          assert.equal(facts.first_person.mesh_annotations, 0);
        });

        it('secondary animation: one hair bone group rooted at the first joint, one collider group', () => {
          const expected = {
            bone_groups: [
              {
                comment: SPRING.name,
                bones: [SPRING.joints[0].bone],
                stiffness: SPRING.v0Uniform.stiffness,
                drag_force: SPRING.v0Uniform.dragForce,
                gravity_power: SPRING.v0Uniform.gravityPower,
                gravity_dir: v0AddonAxes(v0GravityDir(toGltf(SPRING.v0Uniform.gravityDir))),
                hit_radius: SPRING.v0Uniform.hitRadius,
                collider_groups: SPRING_COLLIDER_GROUPS.v0.length,
              },
            ],
          };
          assertFacts({ bone_groups: facts.spring.bone_groups }, expected, TOL.morph, 'secondary animation');
          assert.equal(facts.spring.collider_groups.length, COLLIDER_GROUPS.v0.length);
          const group = COLLIDER_GROUPS.v0[0];
          assert.equal(facts.spring.collider_groups[0].bone, COLLIDERS.v0[group.colliders[0]].bone);
          assert.equal(facts.spring.collider_groups[0].colliders.length, group.colliders.length);
        });

        it('meta: title, version, author and license fields match the spec', () => {
          assertFacts(
            facts.meta,
            {
              title: META.name,
              version: META.version,
              author: META.authors[0],
              allowed_user_name: pascal(META.rights.avatarPermission),
              // META grants no violent, sexual or commercial usage.
              violent_ussage_name: 'Disallow',
              sexual_ussage_name: 'Disallow',
              commercial_ussage_name: 'Disallow',
              license_name: v0Case(META.legacy.licenseName),
              other_permission_url: META.legacy.otherPermissionUrl,
            },
            TOL.morph,
            'meta',
          );
        });
      }
    });
  }
}


if (setup.reason) {
  if (process.env.POTTER_REQUIRE_BLENDER) {
    describe(SUITE, () => {
      it('Blender 5.2.x and the VRM Add-on are available (POTTER_REQUIRE_BLENDER is set)', () => assert.fail(setup.reason));
    });
  } else {
    describe(SUITE, { skip: setup.reason }, () => {});
  }
} else {
  describe(SUITE, () => {
    after(() => rm(setup.userResources, { recursive: true, force: true }));
    runVariants();
  });
}
