# Imports one VRM with the VRM Add-on for Blender and prints the facts that test/blender-addon.test.mjs checks.
# Run: blender -b --factory-startup --python-exit-code 1 --python blender/probe.py -- <file.vrm>
# Needs the add-on installed in BLENDER_USER_RESOURCES and, for VRM 0.x files, BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION=true
# (the add-on's switch for accepting its license prompt). Prints one line "POTTER_VRM_FACTS {json}"; any failure exits non-zero.
# The add-on's own warnings go to stderr and are asserted by the test.
import json
import sys

import bpy

MODULE = "bl_ext.user_default.vrm"
PREFIX = "POTTER_VRM_FACTS "


def fail(message):
    print(f"probe: {message}", file=sys.stderr)
    sys.exit(1)


def vec(values):
    return [float(v) for v in values]


def bone_name(item):
    return item.node.bone_name if hasattr(item, "node") else item.bone_name


def binds(expression):
    return [{"mesh": b.node.mesh_object_name, "index": b.index, "weight": float(b.weight)} for b in expression.morph_target_binds]


def expression_facts(e):
    return {
        "is_binary": bool(e.is_binary),
        "override_blink": str(e.override_blink),
        "override_look_at": str(e.override_look_at),
        "override_mouth": str(e.override_mouth),
        "morph_target_binds": binds(e),
        "material_color_binds": len(e.material_color_binds),
        "texture_transform_binds": len(e.texture_transform_binds),
    }


def collider_shape(collider):
    return collider.shape.sphere if collider.shape_type == "Sphere" else collider.shape.capsule


def range_map(r):
    return {"input_max_value": float(r.input_max_value), "output_scale": float(r.output_scale)}


path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
if bpy.ops.preferences.addon_enable(module=MODULE) != {"FINISHED"}:
    fail(f"add-on {MODULE} did not enable")
result = bpy.ops.import_scene.vrm(filepath=path)
if result != {"FINISHED"}:
    fail(f"import_scene.vrm returned {sorted(result)} for {path}")

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if len(armatures) != 1:
    fail(f"expected one armature, found {len(armatures)}")
armature = armatures[0]
ext = armature.data.vrm_addon_extension
spec = ext.spec_version
facts = {
    "blender": ".".join(map(str, bpy.app.version)),
    "addon": ".".join(map(str, ext.addon_version)),
    "spec_version": spec,
}

# Humanoid: VRM bone name -> Blender bone name (the glTF node name, unless the add-on renamed it).
if spec == "1.0":
    v1 = ext.vrm1
    humanoid = {name.value: hb.node.bone_name for name, hb in v1.humanoid.human_bones.human_bone_name_to_human_bone().items()}
    humanoid = {k: v for k, v in humanoid.items() if v}
    facts["meta"] = {
        "vrm_name": v1.meta.vrm_name,
        "version": v1.meta.version,
        "authors": [a.value for a in v1.meta.authors],
        "avatar_permission": str(v1.meta.avatar_permission),
        "commercial_usage": str(v1.meta.commercial_usage),
        "credit_notation": str(v1.meta.credit_notation),
        "allow_redistribution": bool(v1.meta.allow_redistribution),
        "modification": str(v1.meta.modification),
        "allow_excessively_violent_usage": bool(v1.meta.allow_excessively_violent_usage),
        "allow_excessively_sexual_usage": bool(v1.meta.allow_excessively_sexual_usage),
        "allow_political_or_religious_usage": bool(v1.meta.allow_political_or_religious_usage),
        "allow_antisocial_or_hate_usage": bool(v1.meta.allow_antisocial_or_hate_usage),
    }
    look = v1.look_at
    facts["look_at"] = {
        "type": str(look.type),
        "offset": vec(look.offset_from_head_bone),
        "range_maps": {
            "horizontal_inner": range_map(look.range_map_horizontal_inner),
            "horizontal_outer": range_map(look.range_map_horizontal_outer),
            "vertical_down": range_map(look.range_map_vertical_down),
            "vertical_up": range_map(look.range_map_vertical_up),
        },
    }
    facts["first_person"] = {"mesh_annotations": len(v1.first_person.mesh_annotations)}
    presets = v1.expressions.preset
    preset_names = [p.identifier for p in presets.bl_rna.properties if p.type == "POINTER" and p.identifier != "rna_type"]
    facts["expressions"] = {
        "preset": {name: expression_facts(getattr(presets, name)) for name in preset_names},
        "custom": {c.custom_name: expression_facts(c) for c in v1.expressions.custom},
    }
    s = ext.spring_bone1
    collider_uuid = [c.uuid for c in s.colliders]
    group_uuid = {g.uuid: g.vrm_name for g in s.collider_groups}
    facts["spring"] = {
        "springs": [
            {
                "name": sp.vrm_name,
                "joints": [
                    {
                        "bone": j.node.bone_name,
                        "stiffness": float(j.stiffness),
                        "gravity_power": float(j.gravity_power),
                        "gravity_dir": vec(j.gravity_dir),
                        "drag_force": float(j.drag_force),
                        "hit_radius": float(j.hit_radius),
                    }
                    for j in sp.joints
                ],
                "collider_groups": [group_uuid[g.collider_group_uuid] for g in sp.collider_groups],
            }
            for sp in s.springs
        ],
        "colliders": [
            {"bone": c.node.bone_name, "shape": str(c.shape_type), "radius": float(collider_shape(c).radius)} for c in s.colliders
        ],
        "collider_groups": [
            {"name": g.vrm_name, "colliders": [collider_uuid.index(c.collider_uuid) for c in g.colliders]} for g in s.collider_groups
        ],
    }
else:
    v0 = ext.vrm0
    humanoid = {hb.bone: hb.node.bone_name for hb in v0.humanoid.human_bones if hb.node.bone_name}
    m = v0.meta
    facts["meta"] = {
        "title": m.title,
        "version": m.version,
        "author": m.author,
        "allowed_user_name": str(m.allowed_user_name),
        "violent_ussage_name": str(m.violent_ussage_name),
        "sexual_ussage_name": str(m.sexual_ussage_name),
        "commercial_ussage_name": str(m.commercial_ussage_name),
        "license_name": str(m.license_name),
        "other_permission_url": m.other_permission_url,
    }
    fp = v0.first_person
    facts["look_at"] = {
        "type_name": str(fp.look_at_type_name),
        "range_maps": {
            "horizontal_inner": {"x_range": float(fp.look_at_horizontal_inner.x_range), "y_range": float(fp.look_at_horizontal_inner.y_range)},
            "horizontal_outer": {"x_range": float(fp.look_at_horizontal_outer.x_range), "y_range": float(fp.look_at_horizontal_outer.y_range)},
            "vertical_down": {"x_range": float(fp.look_at_vertical_down.x_range), "y_range": float(fp.look_at_vertical_down.y_range)},
            "vertical_up": {"x_range": float(fp.look_at_vertical_up.x_range), "y_range": float(fp.look_at_vertical_up.y_range)},
        },
    }
    facts["first_person"] = {
        "bone": fp.first_person_bone.bone_name,
        "bone_offset": vec(fp.first_person_bone_offset),
        "mesh_annotations": len(fp.mesh_annotations),
    }
    facts["expressions"] = {
        "blend_shape_groups": {
            g.name: {
                "preset": str(g.preset_name),
                "is_binary": bool(g.is_binary),
                "binds": [{"mesh": b.mesh.mesh_object_name, "index": b.index, "weight": float(b.weight)} for b in g.binds],
                "material_values": len(g.material_values),
            }
            for g in v0.blend_shape_master.blend_shape_groups
        },
    }
    sec = v0.secondary_animation
    facts["spring"] = {
        "bone_groups": [
            {
                "comment": g.comment,
                "bones": [bone_name(b) for b in g.bones],
                "stiffness": float(g.stiffiness),
                "drag_force": float(g.drag_force),
                "gravity_power": float(g.gravity_power),
                "gravity_dir": vec(g.gravity_dir),
                "hit_radius": float(g.hit_radius),
                "collider_groups": len(g.collider_groups),
            }
            for g in sec.bone_groups
        ],
        "collider_groups": [
            {"bone": g.node.bone_name, "colliders": [c.bpy_object.name if c.bpy_object else None for c in g.colliders]}
            for g in sec.collider_groups
        ],
    }

facts["humanoid"] = dict(sorted(humanoid.items()))
world = armature.matrix_world
facts["bone_heads"] = {
    bone: vec(world @ armature.data.bones[bone].head_local) for bone in sorted(set(humanoid.values())) if bone in armature.data.bones
}
facts["armature_matrix"] = [vec(row) for row in world]
facts["meshes"] = {
    o.name: {
        "material_slots": [slot.material.name if slot.material else None for slot in o.material_slots],
        "shape_keys": [kb.name for kb in o.data.shape_keys.key_blocks] if o.data.shape_keys else [],
    }
    for o in sorted(bpy.data.objects, key=lambda o: o.name)
    if o.type == "MESH"
}
facts["materials"] = {m.name: {"mtoon": bool(m.vrm_addon_extension.mtoon1.enabled)} for m in sorted(bpy.data.materials, key=lambda m: m.name)}
print(PREFIX + json.dumps(facts, sort_keys=True))
