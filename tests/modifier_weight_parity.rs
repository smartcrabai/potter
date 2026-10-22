use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
    process::Command,
};

use glam::{DMat4, DVec3};
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{
        Mesh,
        modifiers::attributes::{
            AttributeLayerSelection, SceneOperand, SceneOperandMatricesF32, evaluate_with_context,
        },
    },
    model::{Id, Modifier, ModifierRuntime, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script.rs"]
mod blender_script;

const WEIGHT_ORACLE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
result = {}

EDIT_POSITIONS = [(0.1, 0.2, 0.3), (0.8, -0.1, 0.5), (1.7, 0.4, -0.2),
                  (2.6, -0.3, 0.8), (3.4, 0.6, 0.1), (4.2, 0.0, -0.4)]
WEIGHTS_A = [0.15, 0.4, 0.7, 0.85, 0.55, None]
WEIGHTS_B = [0.65, None, 0.2, 0.45, None, 0.9]
WEIGHTS_MASK = [0.0, 0.3, 0.55, None, 0.9, 0.35]
FALLOFFS = ["LINEAR", "CURVE", "SHARP", "SMOOTH", "ROOT",
            "ICON_SPHERECURVE", "RANDOM", "STEP"]
MIX_MODES = ["SET", "ADD", "SUB", "MUL", "DIV", "DIF", "AVG", "MIN", "MAX"]
MIX_SETS = ["ALL", "A", "B", "OR", "AND"]
CURVE_POINTS = [(0.0, 0.0), (0.27, 0.72), (0.61, 0.38), (1.0, 1.0)]


def make_object(name, vertices, faces=(), location=(0, 0, 0), scale=(1, 1, 1)):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], list(faces))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.scale = scale
    bpy.context.collection.objects.link(obj)
    return obj


def add_group(obj, name, weights):
    group = obj.vertex_groups.new(name=name)
    for index, weight in enumerate(weights):
        if weight is not None:
            group.add([index], float(weight), "REPLACE")
    return group


def read_group(obj, name):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    group = evaluated.vertex_groups.get(name)
    weights = []
    members = []
    for vertex in mesh.vertices:
        assignment = None
        if group is not None:
            assignment = next((item for item in vertex.groups if item.group == group.index), None)
        members.append(assignment is not None)
        weights.append(float(assignment.weight) if assignment is not None else 0.0)
    evaluated.to_mesh_clear()
    return {"weights": weights, "members": members}


def configure_curve(modifier):
    curve = modifier.map_curve.curves[0]
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    curve.points[0].location = CURVE_POINTS[0]
    curve.points[1].location = CURVE_POINTS[-1]
    for point in CURVE_POINTS[1:-1]:
        curve.points.new(*point)
    modifier.map_curve.update()


def weight_object(name):
    obj = make_object(name, EDIT_POSITIONS)
    add_group(obj, "A", WEIGHTS_A)
    add_group(obj, "B", WEIGHTS_B)
    add_group(obj, "Mask", WEIGHTS_MASK)
    return obj


edit = weight_object("Weight Edit Oracle")
edit_modifier = edit.modifiers.new("weights", "VERTEX_WEIGHT_EDIT")
edit_modifier.vertex_group = "A"
edit_modifier.default_weight = 0.23
edit_modifier.normalize = False
result["edit_falloffs"] = {}
for falloff in FALLOFFS:
    edit_modifier.falloff_type = falloff
    if falloff == "CURVE":
        configure_curve(edit_modifier)
    edit_modifier.invert_falloff = False
    result["edit_falloffs"][falloff] = read_group(edit, "A")
edit_modifier.falloff_type = "CURVE"
configure_curve(edit_modifier)
edit_modifier.invert_falloff = True
result["edit_invert_curve"] = read_group(edit, "A")
edit_modifier.falloff_type = "SMOOTH"
edit_modifier.invert_falloff = False
edit_modifier.mask_vertex_group = "Mask"
edit_modifier.mask_constant = 0.75
result["edit_mask"] = read_group(edit, "A")
edit_modifier.invert_mask_vertex_group = True
result["edit_mask_invert"] = read_group(edit, "A")
edit_modifier.mask_vertex_group = "Missing Mask"
edit_modifier.mask_constant = 0.5
result["edit_missing_mask"] = read_group(edit, "A")
edit_modifier.invert_mask_vertex_group = False
edit_modifier.mask_vertex_group = "Mask"
edit_modifier.mask_constant = -0.5
result["edit_mask_negative_constant"] = read_group(edit, "A")
edit_modifier.mask_vertex_group = "Mask"
edit_modifier.mask_constant = 1.5
result["edit_mask_unclamped_constant"] = read_group(edit, "A")

missing_edit = make_object("Missing Edit Group", EDIT_POSITIONS)
add_group(missing_edit, "A", WEIGHTS_A)
missing_edit_modifier = missing_edit.modifiers.new("weights", "VERTEX_WEIGHT_EDIT")
missing_edit_modifier.vertex_group = "Missing"
result["edit_missing_group"] = read_group(missing_edit, "A")

missing_mix = make_object("Missing Mix Group", EDIT_POSITIONS)
add_group(missing_mix, "A", WEIGHTS_A)
missing_mix_modifier = missing_mix.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
missing_mix_modifier.vertex_group_a = "A"
missing_mix_modifier.vertex_group_b = "Missing"
result["mix_missing_group_b"] = read_group(missing_mix, "A")

missing_mix_modifier.vertex_group_a = "Missing"
missing_mix_modifier.vertex_group_b = "A"
result["mix_missing_group_a"] = read_group(missing_mix, "A")


control = weight_object("Weight Edit Controls")
control_modifier = control.modifiers.new("weights", "VERTEX_WEIGHT_EDIT")
control_modifier.vertex_group = "A"
control_modifier.falloff_type = "LINEAR"
control_modifier.default_weight = 0.5
control_modifier.use_add = True
control_modifier.add_threshold = 0.4
control_modifier.use_remove = True
control_modifier.remove_threshold = 0.2
result["edit_add_remove"] = read_group(control, "A")
control_modifier.use_add = False
control_modifier.use_remove = False
control_modifier.default_weight = 0.23
control_modifier.normalize = True
result["edit_normalize"] = read_group(control, "A")

thresholds = weight_object("Weight Edit Inclusive Thresholds")
threshold_modifier = thresholds.modifiers.new("weights", "VERTEX_WEIGHT_EDIT")
threshold_modifier.vertex_group = "A"
threshold_modifier.falloff_type = "LINEAR"
threshold_modifier.default_weight = 0.23
threshold_modifier.use_add = True
threshold_modifier.add_threshold = 0.23
threshold_modifier.use_remove = True
threshold_modifier.remove_threshold = 0.15
result["edit_inclusive_thresholds"] = read_group(thresholds, "A")

result["mix_modes"] = {}
for mix_mode in MIX_MODES:
    obj = weight_object("Mix Mode " + mix_mode)
    modifier = obj.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
    modifier.vertex_group_a = "A"
    modifier.vertex_group_b = "B"
    modifier.mix_set = "ALL"
    modifier.mix_mode = mix_mode
    modifier.default_weight_a = 0.23
    modifier.default_weight_b = 0.31
    modifier.invert_vertex_group_a = True
    result["mix_modes"][mix_mode] = read_group(obj, "A")

result["mix_sets"] = {}
for mix_set in MIX_SETS:
    obj = weight_object("Mix Set " + mix_set)
    modifier = obj.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
    modifier.vertex_group_a = "A"
    modifier.vertex_group_b = "B"
    modifier.mix_set = mix_set
    modifier.mix_mode = "ADD"
    modifier.default_weight_a = 0.23
    modifier.default_weight_b = 0.31
    modifier.invert_vertex_group_b = True
    result["mix_sets"][mix_set] = read_group(obj, "A")

mix_normalize = weight_object("Mix Normalize")
mix_normalize_modifier = mix_normalize.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
mix_normalize_modifier.vertex_group_a = "A"
mix_normalize_modifier.vertex_group_b = "B"
mix_normalize_modifier.mix_set = "OR"
mix_normalize_modifier.mix_mode = "ADD"
mix_normalize_modifier.default_weight_a = 0.23
mix_normalize_modifier.default_weight_b = 0.31
mix_normalize_modifier.normalize = True
result["mix_normalize"] = read_group(mix_normalize, "A")

mix_empty_b = weight_object("Mix Empty B")
mix_empty_b.vertex_groups.remove(mix_empty_b.vertex_groups.get("B"))
mix_empty_b_modifier = mix_empty_b.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
mix_empty_b_modifier.vertex_group_a = "A"
mix_empty_b_modifier.vertex_group_b = ""
mix_empty_b_modifier.mix_set = "ALL"
mix_empty_b_modifier.mix_mode = "SET"
mix_empty_b_modifier.default_weight_b = 0.25
result["mix_empty_group_b"] = read_group(mix_empty_b, "A")

zero_div = weight_object("Mix Divide Zero")
zero_div_modifier = zero_div.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
zero_div_modifier.vertex_group_a = "A"
zero_div_modifier.vertex_group_b = "B"
zero_div_modifier.mix_set = "ALL"
zero_div_modifier.mix_mode = "DIV"
zero_div_modifier.default_weight_b = 0.0
result["mix_divide_zero"] = read_group(zero_div, "A")

PROX_SUBJECT = [(0.1, 0.2, 0.3), (0.8, -0.1, 0.5), (1.7, 0.4, -0.2),
                (2.6, -0.3, 0.8), (3.4, 0.6, 0.1), (4.2, 0.0, -0.4)]
PROX_TARGET = [(0, 0, 0), (2, 0, 0), (2, 1.5, 0), (0, 1.5, 0)]
PROX_FACES = [(0, 1, 2, 3)]


mix_zero = weight_object("Mix Set Zero")
mix_zero_modifier = mix_zero.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
mix_zero_modifier.vertex_group_a = "A"
mix_zero_modifier.vertex_group_b = "B"
mix_zero_modifier.mix_set = "ALL"
mix_zero_modifier.mix_mode = "SET"
mix_zero_modifier.default_weight_b = 0.0
result["mix_set_zero"] = read_group(mix_zero, "A")

mix_mask = weight_object("Mix Mask")
mix_mask_modifier = mix_mask.modifiers.new("weights", "VERTEX_WEIGHT_MIX")
mix_mask_modifier.vertex_group_a = "A"
mix_mask_modifier.vertex_group_b = "B"
mix_mask_modifier.mix_set = "ALL"
mix_mask_modifier.mix_mode = "ADD"
mix_mask_modifier.mask_vertex_group = "Mask"
mix_mask_modifier.mask_constant = 0.75
result["mix_mask"] = read_group(mix_mask, "A")
mix_mask_modifier.invert_mask_vertex_group = True
result["mix_mask_invert"] = read_group(mix_mask, "A")

missing_proximity = make_object("Missing Proximity Group", PROX_SUBJECT)
add_group(missing_proximity, "A", WEIGHTS_A)
missing_target = make_object("Missing Proximity Target", PROX_TARGET, PROX_FACES)
missing_proximity_modifier = missing_proximity.modifiers.new("weights", "VERTEX_WEIGHT_PROXIMITY")
missing_proximity_modifier.vertex_group = "Missing"
missing_proximity_modifier.target = missing_target
result["proximity_missing_group"] = read_group(missing_proximity, "A")

null_target_proximity = make_object("Proximity Null Target", PROX_SUBJECT)
add_group(null_target_proximity, "A", WEIGHTS_A)
null_target_modifier = null_target_proximity.modifiers.new("weights", "VERTEX_WEIGHT_PROXIMITY")
null_target_modifier.vertex_group = "A"
null_target_modifier.target = None
result["proximity_null_target"] = read_group(null_target_proximity, "A")
def matrix_columns(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]

def proximity_case(case_name, mode, geometry, falloff="LINEAR", invert=False, normalize=False,
                   min_dist=0.2, max_dist=5.0):
    subject = make_object(case_name, PROX_SUBJECT, location=(2, -1, 0.5), scale=(1.4, 0.75, 1.1))
    add_group(subject, "A", WEIGHTS_A)
    target = make_object(case_name + " Target", PROX_TARGET, PROX_FACES,
                         location=(-1, 1.5, 0), scale=(0.6, 1.7, 1.1))
    modifier = subject.modifiers.new("weights", "VERTEX_WEIGHT_PROXIMITY")
    modifier.vertex_group = "A"
    modifier.target = target
    modifier.proximity_mode = mode
    modifier.proximity_geometry = geometry
    modifier.min_dist = min_dist
    modifier.max_dist = max_dist
    modifier.falloff_type = falloff
    modifier.invert_falloff = invert
    modifier.normalize = normalize
    if falloff == "CURVE":
        configure_curve(modifier)
    if case_name == "Proximity Object":
        bpy.context.view_layer.update()
        subject_world = subject.matrix_world.copy()
        target_world = target.matrix_world.copy()
        target_to_subject = subject_world.inverted() @ target_world
        result["proximity_matrices"] = {
            "subject_world": matrix_columns(subject_world),
            "target_world": matrix_columns(target_world),
            "local_to_subject": matrix_columns(target_to_subject),
            "subject_to_local": matrix_columns(target_to_subject.inverted()),
        }
    return read_group(subject, "A")

def proximity_mask_case(case_name, invert):
    subject = make_object(case_name, PROX_SUBJECT, location=(2, -1, 0.5), scale=(1.4, 0.75, 1.1))
    add_group(subject, "A", WEIGHTS_A)
    add_group(subject, "Mask", WEIGHTS_MASK)
    target = make_object(case_name + " Target", PROX_TARGET, PROX_FACES,
                         location=(-1, 1.5, 0), scale=(0.6, 1.7, 1.1))
    modifier = subject.modifiers.new("weights", "VERTEX_WEIGHT_PROXIMITY")
    modifier.vertex_group = "A"
    modifier.target = target
    modifier.proximity_mode = "GEOMETRY"
    modifier.proximity_geometry = {"VERTEX", "EDGE", "FACE"}
    modifier.min_dist = 0.2
    modifier.max_dist = 5.0
    modifier.mask_vertex_group = "Mask"
    modifier.mask_constant = 0.75
    modifier.invert_mask_vertex_group = invert
    return read_group(subject, "A")


result["proximity_mask"] = proximity_mask_case("Proximity Mask", False)
result["proximity_mask_invert"] = proximity_mask_case("Proximity Mask Invert", True)


result["proximity_object"] = proximity_case("Proximity Object", "OBJECT", {"VERTEX", "EDGE", "FACE"})
result["proximity_geometry_empty"] = proximity_case(
    "Proximity Geometry Empty", "GEOMETRY", set())
result["proximity_geometry"] = {}
for geometry in ["VERTEX", "EDGE", "FACE"]:
    result["proximity_geometry"][geometry] = proximity_case(
        "Proximity " + geometry, "GEOMETRY", {geometry})
result["proximity_falloffs"] = {}
for falloff in FALLOFFS:
    result["proximity_falloffs"][falloff] = proximity_case(
        "Proximity Falloff " + falloff, "GEOMETRY", {"VERTEX", "EDGE", "FACE"}, falloff)
result["proximity_invert"] = proximity_case(
    "Proximity Invert", "GEOMETRY", {"VERTEX", "EDGE", "FACE"}, "SMOOTH", True)

result["proximity_normalize"] = proximity_case(
    "Proximity Normalize", "GEOMETRY", {"VERTEX", "EDGE", "FACE"}, normalize=True)
result["proximity_reversed_range"] = proximity_case(
    "Proximity Reversed Range", "GEOMETRY", {"VERTEX"}, min_dist=4.0, max_dist=2.0)

with open(os.path.join(root, "weight-oracle.json"), "w", encoding="utf-8") as handle:
    json.dump(result, handle, sort_keys=True)
print("WEIGHT_ORACLE_WRITTEN")
"#;
const WEIGHT_TEXTURE_ORACLE: &str = r#"
import bpy
import json
import math
import os
import platform
import sys

def build_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
# Potter's plane is built in grid order (geom/primitive_order.rs), so ids 2 and 3 are swapped.
vertices = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0)]
faces = [(0, 1, 3, 2)]
weights = [0.2, 0.4, 0.6, 0.8]
uvs = [(0.15, 0.2), (0.8, 0.1), (0.25, 0.75), (0.7, 0.85)]
pixels = [(0.05, 0.15, 0.8, 0.2), (0.35, 0.6, 0.1, 0.4),
          (0.9, 0.25, 0.45, 0.7), (0.65, 0.85, 0.3, 1.0)]
channels = ["INT", "RED", "GREEN", "BLUE", "HUE", "SAT", "VAL", "ALPHA"]
image = bpy.data.images.new("MaskImage", width=2, height=2, alpha=True, float_buffer=True)
image.colorspace_settings.name = "Linear Rec.709"
image.pixels.foreach_set([component for color in pixels for component in color])
image.update()
texture = bpy.data.textures.new("MaskTexture", type="IMAGE")
texture.image = image

frame = bpy.data.objects.new("TextureFrame", None)
bpy.context.collection.objects.link(frame)
frame.location = (-0.15, 0.3, 0.0)
frame.scale = (0.7, 1.25, 1.0)

armature_data = bpy.data.armatures.new("TextureRig")
armature = bpy.data.objects.new("TextureRig", armature_data)
bpy.context.collection.objects.link(armature)
bpy.context.view_layer.objects.active = armature
armature.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bone = armature_data.edit_bones.new("MaskBone")
bone.head = (0, 0, 0)
bone.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode="OBJECT")
armature.location = (0.1, -0.1, 0.0)
armature.scale = (1.2, 0.8, 1.0)
pose_bone = armature.pose.bones["MaskBone"]
pose_bone.rotation_mode = "QUATERNION"
pose_bone.rotation_quaternion = (math.cos(0.23), 0.0, 0.0, math.sin(0.23))
pose_bone.location = (0.2, 0.3, 0.0)

results = {}
results["blender_build"] = {
    "version": bpy.app.version_string,
    "build_hash": build_string(bpy.app.build_hash),
    "build_date": build_string(bpy.app.build_date),
    "build_time": build_string(bpy.app.build_time),
    "platform": build_string(bpy.app.build_platform),
    "machine": platform.machine(),
}
for mapping in ["LOCAL", "GLOBAL", "OBJECT", "OBJECT_NULL", "UV", "UV_FALLBACK", "BONE", "BONE_MISSING"]:
    for channel in channels:
        name = "Owner_" + mapping + "_" + channel
        mesh = bpy.data.meshes.new(name + "Mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        owner = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(owner)
        group = owner.vertex_groups.new(name="A")
        for index, weight in enumerate(weights):
            group.add([index], weight, "REPLACE")
        if mapping in ("UV", "UV_FALLBACK"):
            layer = mesh.uv_layers.new(name="UVMask")
            for loop in mesh.loops:
                layer.data[loop.index].uv = uvs[loop.vertex_index]
        if mapping in ("GLOBAL", "OBJECT", "OBJECT_NULL", "BONE", "BONE_MISSING"):
            owner.location = (0.35, -0.2, 0.15)
            owner.scale = (1.4, 0.65, 1.0)
        modifier = owner.modifiers.new("WeightEdit", "VERTEX_WEIGHT_EDIT")
        modifier.vertex_group = "A"
        modifier.falloff_type = "SHARP"
        modifier.normalize = False
        modifier.mask_texture = texture
        modifier.mask_tex_use_channel = channel
        modifier.mask_constant = 0.75
        if mapping == "GLOBAL":
            modifier.mask_tex_mapping = "GLOBAL"
        elif mapping in ("OBJECT", "OBJECT_NULL"):
            modifier.mask_tex_mapping = "OBJECT"
            if mapping == "OBJECT":
                modifier.mask_tex_map_object = frame
        elif mapping in ("UV", "UV_FALLBACK"):
            modifier.mask_tex_mapping = "UV"
            modifier.mask_tex_uv_layer = "UVMask" if mapping == "UV" else "MissingUV"
        elif mapping in ("BONE", "BONE_MISSING"):
            modifier.mask_tex_mapping = "OBJECT"
            modifier.mask_tex_map_object = armature
            modifier.mask_tex_map_bone = "MaskBone" if mapping == "BONE" else "MissingBone"
        else:
            modifier.mask_tex_mapping = "LOCAL"
        evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
        evaluated_mesh = evaluated.to_mesh()
        evaluated_group = evaluated.vertex_groups["A"]
        result = []
        for vertex in evaluated_mesh.vertices:
            result.append(next(
                (assignment.weight for assignment in vertex.groups
                 if assignment.group == evaluated_group.index), 0.0))
        evaluated.to_mesh_clear()
        results[mapping + "_" + channel] = result

# Procedural (legacy) texture masks: a CLOUDS texture sampled with LOCAL coordinates.
cloud_texture = bpy.data.textures.new("CloudsMask", type="CLOUDS")
results["procedural_CLOUDS_input"] = {
    "texture_type": cloud_texture.type,
    "cloud_type": cloud_texture.cloud_type,
    "group_weights": weights,
}
for channel in ["INT", "RED", "ALPHA"]:
    name = "Owner_CLOUDS_" + channel
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    owner = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(owner)
    group = owner.vertex_groups.new(name="A")
    for index, weight in enumerate(weights):
        group.add([index], weight, "REPLACE")
    modifier = owner.modifiers.new("WeightEdit", "VERTEX_WEIGHT_EDIT")
    modifier.vertex_group = "A"
    modifier.falloff_type = "SHARP"
    modifier.normalize = False
    modifier.mask_texture = cloud_texture
    modifier.mask_tex_use_channel = channel
    modifier.mask_constant = 0.75
    modifier.mask_tex_mapping = "LOCAL"
    evaluated = owner.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    evaluated_group = evaluated.vertex_groups["A"]
    result = []
    membership = []
    for vertex in evaluated_mesh.vertices:
        assignment = next(
            (assignment for assignment in vertex.groups
             if assignment.group == evaluated_group.index), None)
        membership.append(assignment is not None)
        result.append(assignment.weight if assignment is not None else 0.0)
    evaluated.to_mesh_clear()
    results["procedural_CLOUDS_" + channel] = result
    results["procedural_CLOUDS_" + channel + "_members"] = membership
    results["procedural_CLOUDS_" + channel + "_settings"] = {
        "vertex_group": modifier.vertex_group,
        "texture_type": modifier.mask_texture.type,
        "cloud_type": cloud_texture.cloud_type,
        "falloff_type": modifier.falloff_type,
        "invert_falloff": modifier.invert_falloff,
        "normalize": modifier.normalize,
        "default_weight": modifier.default_weight,
        "use_add": modifier.use_add,
        "add_threshold": modifier.add_threshold,
        "use_remove": modifier.use_remove,
        "remove_threshold": modifier.remove_threshold,
        "mask_constant": modifier.mask_constant,
        "channel": modifier.mask_tex_use_channel,
        "mapping": modifier.mask_tex_mapping,
    }

with open(os.path.join(root, "weight-texture-oracle.json"), "w", encoding="utf-8") as handle:
    json.dump(results, handle, sort_keys=True)
print("WEIGHT_TEXTURE_ORACLE_WRITTEN")
"#;

fn modifier(modifier_type: &str, params: &Value) -> Result<Modifier, Box<dyn Error>> {
    Ok(Modifier {
        baked_ocean_cache: None,
        id: Id::new("weight_test_modifier")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: params
            .as_object()
            .ok_or("modifier parameters must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn source_mesh() -> Result<Mesh, Box<dyn Error>> {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![
            DVec3::new(0.1, 0.2, 0.3),
            DVec3::new(0.8, -0.1, 0.5),
            DVec3::new(1.7, 0.4, -0.2),
            DVec3::new(2.6, -0.3, 0.8),
            DVec3::new(3.4, 0.6, 0.1),
            DVec3::new(4.2, 0.0, -0.4),
        ],
        vec![],
    )?;
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({
            "A":{"v0":0.15,"v1":0.4,"v2":0.7,"v3":0.85,"v4":0.55},
            "B":{"v0":0.65,"v2":0.2,"v3":0.45,"v5":0.9},
            "Mask":{"v0":0.0,"v1":0.3,"v2":0.55,"v4":0.9,"v5":0.35}
        }),
    );
    Ok(mesh)
}

fn target_mesh() -> Result<Mesh, Box<dyn Error>> {
    Ok(Mesh::from_positions_and_faces(
        vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(2.0, 1.5, 0.0),
            DVec3::new(0.0, 1.5, 0.0),
        ],
        vec![vec![0, 1, 2, 3]],
    )?)
}

fn weights(mesh: &Mesh, group: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let values = mesh.attributes["vertex_groups"][group]
        .as_object()
        .ok_or("vertex group values are missing")?;
    mesh.vertices
        .iter()
        .map(|vertex| {
            Ok(values
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0))
        })
        .collect()
}

fn members(mesh: &Mesh, group: &str) -> Result<Vec<bool>, Box<dyn Error>> {
    let values = mesh.attributes["vertex_groups"][group]
        .as_object()
        .ok_or("vertex group values are missing")?;
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| values.contains_key(&format!("v{}", vertex.id)))
        .collect())
}

fn assert_weights(
    label: &str,
    actual: &[f64],
    expected: &Value,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let expected = expected
        .as_array()
        .ok_or("oracle weight vector must be an array")?
        .iter()
        .map(|value| value.as_f64().ok_or("oracle weight must be numeric"))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual.len(), expected.len(), "{label} vertex count");
    let error = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("Blender weight parity {label}: max absolute error {error:.8e}");
    assert!(
        error <= tolerance,
        "{label} max error {error} exceeds {tolerance}; actual={actual:?} expected={expected:?}"
    );
    Ok(())
}

fn assert_members(label: &str, actual: &[bool], expected: &Value) -> Result<(), Box<dyn Error>> {
    let expected = expected
        .as_array()
        .ok_or("oracle membership vector must be an array")?
        .iter()
        .map(|value| value.as_bool().ok_or("oracle membership must be boolean"))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(actual, expected, "{label} membership differs from Blender");
    Ok(())
}

fn oracle(root: &Path) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        return Ok(None);
    };
    blender_script::run_blender_script(
        &blender,
        "weight_oracle.py",
        WEIGHT_ORACLE,
        root,
        "Blender weight modifier oracle failed",
    )?;
    Ok(Some(serde_json::from_slice(&fs::read(
        root.join("weight-oracle.json"),
    )?)?))
}

fn texture_oracle(root: &Path) -> Result<Option<Value>, Box<dyn Error>> {
    let Some(blender) = blender_file::blender_executable() else {
        return Ok(None);
    };
    blender_script::run_blender_script(
        &blender,
        "weight_texture_oracle.py",
        WEIGHT_TEXTURE_ORACLE,
        root,
        "Blender weight texture oracle failed",
    )?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("weight-texture-oracle.json"))?)?;
    // Set only with the target macOS Blender executable to regenerate the pinned CLOUDS oracle.
    if let Some(path) = std::env::var_os("POTTER_RECORD_CLOUDS_ORACLE") {
        let cloud_oracle = json!({
            "blender": expected["blender_build"].clone(),
            "input": expected["procedural_CLOUDS_input"].clone(),
            "channels": {
                "INT": {
                    "weights": expected["procedural_CLOUDS_INT"].clone(),
                    "members": expected["procedural_CLOUDS_INT_members"].clone(),
                    "settings": expected["procedural_CLOUDS_INT_settings"].clone(),
                },
                "RED": {
                    "weights": expected["procedural_CLOUDS_RED"].clone(),
                    "members": expected["procedural_CLOUDS_RED_members"].clone(),
                    "settings": expected["procedural_CLOUDS_RED_settings"].clone(),
                },
                "ALPHA": {
                    "weights": expected["procedural_CLOUDS_ALPHA"].clone(),
                    "members": expected["procedural_CLOUDS_ALPHA_members"].clone(),
                    "settings": expected["procedural_CLOUDS_ALPHA_settings"].clone(),
                },
            },
        });
        fs::write(path, serde_json::to_vec_pretty(&cloud_oracle)?)?;
    }
    Ok(Some(expected))
}

fn operand(
    mesh: Mesh,
    world_matrix_f32: Option<[f32; 16]>,
    local_to_subject_f32: Option<[f32; 16]>,
    subject_to_local_f32: Option<[f32; 16]>,
) -> SceneOperand {
    let local_to_subject =
        local_to_subject_f32.map(|matrix| DMat4::from_cols_array(&matrix.map(f64::from)));
    SceneOperand {
        mesh,
        local_to_subject,
        blender_matrices_f32: Some(SceneOperandMatricesF32 {
            world: world_matrix_f32,
            local_to_subject: local_to_subject_f32,
            subject_to_local: subject_to_local_f32,
        }),
        camera: None,
        armature: false,
        object_scale: DVec3::ONE,
        selected_bones: BTreeSet::new(),
        deforming_bones: BTreeSet::new(),
        layer_selection: AttributeLayerSelection::default(),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender matrix elements are serialized to JSON from f32"
)]
fn blender_matrix_f32(value: &Value) -> Result<[f32; 16], Box<dyn Error>> {
    let values = value.as_array().ok_or("oracle matrix must be an array")?;
    if values.len() != 16 {
        return Err("oracle matrix must have 16 values".into());
    }
    let mut matrix = [0.0; 16];
    for (slot, value) in matrix.iter_mut().zip(values) {
        *slot = value
            .as_f64()
            .ok_or("oracle matrix component must be numeric")? as f32;
    }
    Ok(matrix)
}

fn assert_case(
    label: &str,
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    subject_world: DMat4,
    subject_name: &str,
    expected: &Value,
) -> Result<Mesh, Box<dyn Error>> {
    let result = evaluate_with_context(
        mesh,
        modifier,
        operands,
        None,
        None,
        subject_world,
        subject_name,
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(label, &weights(&result, "A")?, &expected["weights"], 2.0e-5)?;
    assert_members(label, &members(&result, "A")?, &expected["members"])?;
    Ok(result)
}

#[test]
fn weight_modifiers_match_blender_falloff_mix_masks_and_proximity() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let Some(expected) = oracle(directory.path())? else {
        eprintln!("skipping weight modifier parity test: Blender is unavailable");
        return Ok(());
    };
    let mesh = source_mesh()?;
    let falloffs = [
        "LINEAR",
        "CURVE",
        "SHARP",
        "SMOOTH",
        "ROOT",
        "ICON_SPHERECURVE",
        "RANDOM",
        "STEP",
    ];
    for falloff in falloffs {
        let mut params = json!({
            "vertex_group":"A",
            "falloff_type":falloff,
            "default_weight":0.23,
            "normalize":false
        });
        if falloff == "CURVE" {
            params["map_curve"] = json!([[0.0, 0.0], [0.27, 0.72], [0.61, 0.38], [1.0, 1.0]]);
        }
        let modifier_1 = modifier("vertex_weight_edit", &params)?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_1,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Edit Oracle",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            &format!("weight edit {falloff}"),
            &weights(&result, "A")?,
            &expected["edit_falloffs"][falloff]["weights"],
            2.0e-5,
        )?;
        assert_members(
            &format!("weight edit {falloff}"),
            &members(&result, "A")?,
            &expected["edit_falloffs"][falloff]["members"],
        )?;
    }

    let curve_invert = modifier(
        "vertex_weight_edit",
        &json!({
            "vertex_group":"A","falloff_type":"CURVE","invert_falloff":true,
            "map_curve":[[0.0,0.0],[0.27,0.72],[0.61,0.38],[1.0,1.0]],
            "default_weight":0.23,"normalize":false
        }),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &curve_invert,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Edit Oracle",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight edit inverted curve",
        &weights(&result, "A")?,
        &expected["edit_invert_curve"]["weights"],
        2.0e-5,
    )?;

    for (label, mask_name, inverted, constant, oracle_key) in [
        ("mask", "Mask", false, 0.75, "edit_mask"),
        ("inverted mask", "Mask", true, 0.75, "edit_mask_invert"),
        (
            "missing mask",
            "Missing Mask",
            false,
            0.5,
            "edit_missing_mask",
        ),
    ] {
        let modifier_2 = modifier(
            "vertex_weight_edit",
            &json!({
                "vertex_group":"A","falloff_type":"SMOOTH","mask_vertex_group":mask_name,
                "invert_mask_vertex_group":inverted,"mask_constant":constant,"default_weight":0.23
            }),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_2,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Edit Oracle",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            label,
            &weights(&result, "A")?,
            &expected[oracle_key]["weights"],
            2.0e-5,
        )?;
    }

    for (label, constant, oracle_key) in [
        (
            "weight edit negative mask constant",
            -0.5,
            "edit_mask_negative_constant",
        ),
        (
            "weight edit mask constant above one",
            1.5,
            "edit_mask_unclamped_constant",
        ),
    ] {
        let modifier_3 = modifier(
            "vertex_weight_edit",
            &json!({
                "vertex_group":"A","falloff_type":"SMOOTH","mask_vertex_group":"Mask",
                "mask_constant":constant,"default_weight":0.23
            }),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_3,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Edit Oracle",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            label,
            &weights(&result, "A")?,
            &expected[oracle_key]["weights"],
            2.0e-5,
        )?;
    }
    let missing_edit_group = modifier(
        "vertex_weight_edit",
        &json!({"vertex_group":"Missing","falloff_type":"SHARP"}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &missing_edit_group,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Missing Edit Group",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "missing edit group pass-through",
        &weights(&result, "A")?,
        &expected["edit_missing_group"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "missing edit group pass-through",
        &members(&result, "A")?,
        &expected["edit_missing_group"]["members"],
    )?;

    let add_remove = modifier(
        "vertex_weight_edit",
        &json!({
            "vertex_group":"A","falloff_type":"LINEAR","default_weight":0.5,
            "use_add":true,"add_threshold":0.4,"use_remove":true,"remove_threshold":0.2
        }),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &add_remove,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Edit Controls",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight edit add and remove",
        &weights(&result, "A")?,
        &expected["edit_add_remove"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "weight edit add and remove",
        &members(&result, "A")?,
        &expected["edit_add_remove"]["members"],
    )?;
    let inclusive_thresholds = modifier(
        "vertex_weight_edit",
        &json!({
            "vertex_group":"A","falloff_type":"LINEAR","default_weight":0.23,
            "use_add":true,"add_threshold":0.23,"use_remove":true,"remove_threshold":0.15
        }),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &inclusive_thresholds,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Edit Inclusive Thresholds",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight edit inclusive thresholds",
        &weights(&result, "A")?,
        &expected["edit_inclusive_thresholds"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "weight edit inclusive thresholds",
        &members(&result, "A")?,
        &expected["edit_inclusive_thresholds"]["members"],
    )?;

    let normalize = modifier(
        "vertex_weight_edit",
        &json!({"vertex_group":"A","falloff_type":"LINEAR","default_weight":0.23,"normalize":true}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &normalize,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Edit Controls",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight edit normalize",
        &weights(&result, "A")?,
        &expected["edit_normalize"]["weights"],
        2.0e-5,
    )?;

    for mix_mode in [
        "SET", "ADD", "SUB", "MUL", "DIV", "DIF", "AVG", "MIN", "MAX",
    ] {
        let modifier_4 = modifier(
            "vertex_weight_mix",
            &json!({
                "vertex_group_a":"A","vertex_group_b":"B","mix_set":"ALL","mix_mode":mix_mode,
                "default_weight_a":0.23,"default_weight_b":0.31,"invert_vertex_group_a":true
            }),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_4,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Mix",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            &format!("weight mix {mix_mode}"),
            &weights(&result, "A")?,
            &expected["mix_modes"][mix_mode]["weights"],
            2.0e-5,
        )?;
        assert_members(
            &format!("weight mix {mix_mode}"),
            &members(&result, "A")?,
            &expected["mix_modes"][mix_mode]["members"],
        )?;
    }
    for mix_set in ["ALL", "A", "B", "OR", "AND"] {
        let modifier_5 = modifier(
            "vertex_weight_mix",
            &json!({
                "vertex_group_a":"A","vertex_group_b":"B","mix_set":mix_set,"mix_mode":"ADD",
                "default_weight_a":0.23,"default_weight_b":0.31,"invert_vertex_group_b":true
            }),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_5,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Mix",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            &format!("weight mix set {mix_set}"),
            &weights(&result, "A")?,
            &expected["mix_sets"][mix_set]["weights"],
            2.0e-5,
        )?;
        assert_members(
            &format!("weight mix set {mix_set}"),
            &members(&result, "A")?,
            &expected["mix_sets"][mix_set]["members"],
        )?;
    }
    let divide_zero = modifier(
        "vertex_weight_mix",
        &json!({"vertex_group_a":"A","vertex_group_b":"B","mix_set":"ALL","mix_mode":"DIV","default_weight_b":0.0}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &divide_zero,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Mix",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight mix division by zero",
        &weights(&result, "A")?,
        &expected["mix_divide_zero"]["weights"],
        2.0e-5,
    )?;
    let mix_zero = modifier(
        "vertex_weight_mix",
        &json!({"vertex_group_a":"A","vertex_group_b":"B","mix_set":"ALL","mix_mode":"SET","default_weight_b":0.0}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &mix_zero,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Weight Mix",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight mix selected zero",
        &weights(&result, "A")?,
        &expected["mix_set_zero"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "weight mix selected zero",
        &members(&result, "A")?,
        &expected["mix_set_zero"]["members"],
    )?;
    for (label, inverted, oracle_key) in [
        ("weight mix mask", false, "mix_mask"),
        ("weight mix inverted mask", true, "mix_mask_invert"),
    ] {
        let modifier_6 = modifier(
            "vertex_weight_mix",
            &json!({
                "vertex_group_a":"A","vertex_group_b":"B","mix_set":"ALL","mix_mode":"ADD",
                "mask_vertex_group":"Mask","mask_constant":0.75,"invert_mask_vertex_group":inverted
            }),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_6,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Weight Mix",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            label,
            &weights(&result, "A")?,
            &expected[oracle_key]["weights"],
            2.0e-5,
        )?;
    }
    for (group_a, group_b, oracle_key) in [
        ("A", "Missing", "mix_missing_group_b"),
        ("Missing", "A", "mix_missing_group_a"),
    ] {
        let modifier_7 = modifier(
            "vertex_weight_mix",
            &json!({"vertex_group_a":group_a,"vertex_group_b":group_b,"mix_set":"ALL","mix_mode":"ADD"}),
        )?;
        let result = evaluate_with_context(
            &mesh,
            &modifier_7,
            &BTreeMap::new(),
            None,
            None,
            DMat4::IDENTITY,
            "Missing Mix Group",
            &AttributeLayerSelection::default(),
        )?;
        assert_weights(
            "missing mix group pass-through",
            &weights(&result, "A")?,
            &expected[oracle_key]["weights"],
            2.0e-5,
        )?;
        assert_members(
            "missing mix group pass-through",
            &members(&result, "A")?,
            &expected[oracle_key]["members"],
        )?;
    }
    let mix_normalize = modifier(
        "vertex_weight_mix",
        &json!({
            "vertex_group_a":"A","vertex_group_b":"B","mix_set":"OR","mix_mode":"ADD",
            "default_weight_a":0.23,"default_weight_b":0.31,"normalize":true
        }),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &mix_normalize,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Mix Normalize",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight mix normalize",
        &weights(&result, "A")?,
        &expected["mix_normalize"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "weight mix normalize",
        &members(&result, "A")?,
        &expected["mix_normalize"]["members"],
    )?;
    let mix_empty_b = modifier(
        "vertex_weight_mix",
        &json!({
            "vertex_group_a":"A","vertex_group_b":"","mix_set":"ALL","mix_mode":"SET",
            "default_weight_b":0.25
        }),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &mix_empty_b,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Mix Empty B",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "weight mix empty B",
        &weights(&result, "A")?,
        &expected["mix_empty_group_b"]["weights"],
        2.0e-5,
    )?;

    let matrices = &expected["proximity_matrices"];
    let subject_world_f32 = blender_matrix_f32(&matrices["subject_world"])?;
    let target_world_f32 = blender_matrix_f32(&matrices["target_world"])?;
    let local_to_subject_f32 = blender_matrix_f32(&matrices["local_to_subject"])?;
    let subject_to_local_f32 = blender_matrix_f32(&matrices["subject_to_local"])?;
    let subject_transform = DMat4::from_cols_array(&subject_world_f32.map(f64::from));
    let target_mesh = target_mesh()?;
    let object_target = operand(target_mesh.clone(), Some(target_world_f32), None, None);
    let geometry_target = operand(
        target_mesh,
        None,
        Some(local_to_subject_f32),
        Some(subject_to_local_f32),
    );
    let mut object_operands = BTreeMap::new();
    object_operands.insert("target".to_owned(), vec![object_target]);
    let mut geometry_operands = BTreeMap::new();
    geometry_operands.insert("target".to_owned(), vec![geometry_target]);
    let object_proximity = modifier(
        "vertex_weight_proximity",
        &json!({"vertex_group":"A","target":"target","proximity_mode":"OBJECT","proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0}),
    )?;
    assert_case(
        "weight proximity OBJECT",
        &mesh,
        &object_proximity,
        &object_operands,
        subject_transform,
        "Proximity Object",
        &expected["proximity_object"],
    )?;
    let geometry_empty = modifier(
        "vertex_weight_proximity",
        &json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":[],"min_dist":0.2,"max_dist":5.0
        }),
    )?;
    assert_case(
        "weight proximity GEOMETRY empty set",
        &mesh,
        &geometry_empty,
        &object_operands,
        subject_transform,
        "Proximity Geometry Empty",
        &expected["proximity_geometry_empty"],
    )?;
    for geometry in ["VERTEX", "EDGE", "FACE"] {
        let modifier_8 = modifier(
            "vertex_weight_proximity",
            &json!({"vertex_group":"A","target":"target","proximity_mode":"GEOMETRY","proximity_geometry":[geometry],"min_dist":0.2,"max_dist":5.0}),
        )?;
        assert_case(
            &format!("weight proximity GEOMETRY/{geometry}"),
            &mesh,
            &modifier_8,
            &geometry_operands,
            subject_transform,
            &format!("Proximity {geometry}"),
            &expected["proximity_geometry"][geometry],
        )?;
    }
    let proximity_normalize = modifier(
        "vertex_weight_proximity",
        &json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0,
            "normalize":true
        }),
    )?;
    assert_case(
        "weight proximity normalize",
        &mesh,
        &proximity_normalize,
        &geometry_operands,
        subject_transform,
        "Proximity Normalize",
        &expected["proximity_normalize"],
    )?;
    let reversed_range = modifier(
        "vertex_weight_proximity",
        &json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":["VERTEX"],"min_dist":4.0,"max_dist":2.0
        }),
    )?;
    assert_case(
        "weight proximity reversed range",
        &mesh,
        &reversed_range,
        &geometry_operands,
        subject_transform,
        "Proximity Reversed Range",
        &expected["proximity_reversed_range"],
    )?;
    let proximity_mask = modifier(
        "vertex_weight_proximity",
        &json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0,
            "mask_vertex_group":"Mask","mask_constant":0.75
        }),
    )?;
    assert_case(
        "weight proximity mask",
        &mesh,
        &proximity_mask,
        &geometry_operands,
        subject_transform,
        "Proximity Mask",
        &expected["proximity_mask"],
    )?;
    let proximity_mask_invert = modifier(
        "vertex_weight_proximity",
        &json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0,
            "mask_vertex_group":"Mask","mask_constant":0.75,"invert_mask_vertex_group":true
        }),
    )?;
    assert_case(
        "weight proximity inverted mask",
        &mesh,
        &proximity_mask_invert,
        &geometry_operands,
        subject_transform,
        "Proximity Mask Invert",
        &expected["proximity_mask_invert"],
    )?;
    let missing_proximity_group = modifier(
        "vertex_weight_proximity",
        &json!({"vertex_group":"Missing","target":"target","proximity_mode":"OBJECT","min_dist":0.0,"max_dist":5.0}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &missing_proximity_group,
        &object_operands,
        None,
        None,
        subject_transform,
        "Proximity Missing Group",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "missing proximity group pass-through",
        &weights(&result, "A")?,
        &expected["proximity_missing_group"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "missing proximity group pass-through",
        &members(&result, "A")?,
        &expected["proximity_missing_group"]["members"],
    )?;
    let null_target_proximity = modifier(
        "vertex_weight_proximity",
        &json!({"vertex_group":"A","target":null,"proximity_mode":"OBJECT"}),
    )?;
    let result = evaluate_with_context(
        &mesh,
        &null_target_proximity,
        &BTreeMap::new(),
        None,
        None,
        DMat4::IDENTITY,
        "Proximity Null Target",
        &AttributeLayerSelection::default(),
    )?;
    assert_weights(
        "null proximity target pass-through",
        &weights(&result, "A")?,
        &expected["proximity_null_target"]["weights"],
        2.0e-5,
    )?;
    assert_members(
        "null proximity target pass-through",
        &members(&result, "A")?,
        &expected["proximity_null_target"]["members"],
    )?;
    for falloff in falloffs {
        let mut params = json!({
            "vertex_group":"A","target":"target","proximity_mode":"GEOMETRY",
            "proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0,
            "falloff_type":falloff
        });
        if falloff == "CURVE" {
            params["map_curve"] = json!([[0.0, 0.0], [0.27, 0.72], [0.61, 0.38], [1.0, 1.0]]);
        }
        let modifier_9 = modifier("vertex_weight_proximity", &params)?;
        let name = format!("Proximity Falloff {falloff}");
        assert_case(
            &format!("weight proximity falloff {falloff}"),
            &mesh,
            &modifier_9,
            &geometry_operands,
            subject_transform,
            &name,
            &expected["proximity_falloffs"][falloff],
        )?;
    }
    let inverted_proximity = modifier(
        "vertex_weight_proximity",
        &json!({"vertex_group":"A","target":"target","proximity_mode":"GEOMETRY","proximity_geometry":["VERTEX","EDGE","FACE"],"min_dist":0.2,"max_dist":5.0,"falloff_type":"SMOOTH","invert_falloff":true}),
    )?;
    assert_case(
        "weight proximity inverted falloff",
        &mesh,
        &inverted_proximity,
        &geometry_operands,
        subject_transform,
        "Proximity Invert",
        &expected["proximity_invert"],
    )?;
    Ok(())
}

#[test]
fn cli_applies_and_inspects_all_vertex_weight_modifiers() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let operations = root.join("weight-operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"plane","params":{}},
                {"op":"node.create","id":"target","kind":"empty","transform":{"translation":[0.0,0.0,2.0]}},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"group_a","name":"A"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_a","weights":[
                    {"vertex_id":0,"weight":0.2},{"vertex_id":1,"weight":0.4},
                    {"vertex_id":2,"weight":0.6},{"vertex_id":3,"weight":0.8}
                ]},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"group_b","name":"B"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_b","weights":[
                    {"vertex_id":0,"weight":0.1},{"vertex_id":1,"weight":0.3},
                    {"vertex_id":2,"weight":0.5},{"vertex_id":3,"weight":0.7}
                ]},
                {"op":"modifier.create","target":{"id":"body"},"id":"edit","type":"vertex_weight_edit","params":{"vertex_group":"A","falloff_type":"SHARP","normalize":false}},
                {"op":"modifier.create","target":{"id":"body"},"id":"mix","type":"vertex_weight_mix","params":{"vertex_group_a":"A","vertex_group_b":"B","mix_set":"ALL","mix_mode":"ADD","normalize":false}},
                {"op":"modifier.create","target":{"id":"body"},"id":"proximity","type":"vertex_weight_proximity","params":{"vertex_group":"A","target":"target","proximity_mode":"OBJECT","min_dist":0.0,"max_dist":4.0,"normalize":false}}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let inspection: Value = serde_json::from_slice(&inspected.stdout)?;
    assert!(inspection["result"]["items"][0]["evaluated_geometry"]["positions"].is_array());

    let document: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate_with_cache(
        &document,
        &EvaluationContext::default(),
        None,
        Some(&scene),
    )?;
    let body = snapshot
        .meshes
        .get(&Id::new("body")?)
        .ok_or("evaluated body mesh missing")?;
    let weights = weights(body, "A")?;
    assert_eq!(weights.len(), 4);
    assert!(weights.iter().all(|weight| (0.0..=1.0).contains(weight)));
    let body_node = document
        .nodes
        .get(&Id::new("body")?)
        .ok_or("body node missing")?;
    for id in ["edit", "mix", "proximity"] {
        assert!(
            body_node
                .modifiers
                .iter()
                .any(|modifier| modifier.id.as_str() == id)
        );
    }
    Ok(())
}

#[test]
fn cli_null_target_proximity_preserves_input_weights() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    let operations_path = root.join("null-target-operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":0,
            "operations":[
                {"op":"node.create","id":"body","kind":"plane","params":{"size":1.0}},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"group_a","name":"A"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"group_a","weights":[
                    {"vertex_id":0,"weight":0.2},{"vertex_id":1,"weight":0.4},
                    {"vertex_id":2,"weight":0.6},{"vertex_id":3,"weight":0.8}
                ]},
                {"op":"modifier.create","target":{"id":"body"},"id":"proximity","type":"vertex_weight_proximity","params":{"vertex_group":"A","target":null}}
            ]
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["inspect"])
        .arg(&scene)
        .args(["--id", "body", "--json"])
        .output()?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let inspection: Value = serde_json::from_slice(&inspected.stdout)?;
    assert!(inspection["result"]["items"][0]["evaluated_geometry"]["positions"].is_array());
    let document: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate_with_cache(
        &document,
        &EvaluationContext::default(),
        None,
        Some(&scene),
    )?;
    let body = snapshot
        .meshes
        .get(&Id::new("body")?)
        .ok_or("evaluated body mesh missing")?;
    assert_eq!(weights(body, "A")?, [0.2, 0.4, 0.6, 0.8]);
    Ok(())
}

#[test]
fn texture_weight_masks_match_blender_channels_mappings_and_bone_space()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let Some(expected) = texture_oracle(root)? else {
        eprintln!("skipping weight texture parity test: Blender is unavailable");
        return Ok(());
    };
    let scene = root.join("scene");
    let operations_path = root.join("texture-mask-operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );

    // Potter stores image rows top-down, while the Blender oracle lists them bottom-up, so the
    // same 2x2 image is passed to `image.set_pixels` with its rows swapped.
    let pixels = json!([
        [0.9, 0.25, 0.45, 0.7],
        [0.65, 0.85, 0.3, 1.0],
        [0.05, 0.15, 0.8, 0.2],
        [0.35, 0.6, 0.1, 0.4]
    ]);
    let mut operations = vec![
        json!({"op":"image.create","id":"mask","name":"MaskImage","width":2,"height":2,"colorspace":"linear","fill_color":[0.0,0.0,0.0,1.0]}),
        json!({"op":"image.set_pixels","id":"mask","width":2,"height":2,"pixels":pixels}),
        json!({"op":"node.create","id":"texture_frame","kind":"empty","transform":{"translation":[-0.15,0.3,0.0],"scale":[0.7,1.25,1.0]}}),
        json!({"op":"node.create","id":"texture_rig","kind":"armature","transform":{"translation":[0.1,-0.1,0.0],"scale":[1.2,0.8,1.0]}}),
        json!({"op":"bone.create","target":{"id":"texture_rig"},"id":"mask_bone","name":"MaskBone","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]}),
        json!({"op":"pose.set","target":{"id":"texture_rig"},"bone_id":"mask_bone","set":{"translation":[0.2,0.3,0.0],"rotation":[0.0,0.0,0.227_977_523_535_188_4,0.973_666_395_005_374_9]}}),
    ];
    let mappings = [
        "LOCAL",
        "GLOBAL",
        "OBJECT",
        "OBJECT_NULL",
        "UV",
        "UV_FALLBACK",
        "BONE",
        "BONE_MISSING",
    ];
    let channels = ["INT", "RED", "GREEN", "BLUE", "HUE", "SAT", "VAL", "ALPHA"];
    // The canonical store is V-down; the Blender oracle fixture uses its native V-up UVs.
    // Per corner of the grid plane, whose face is [v0, v1, v3, v2] (geom/primitive_order.rs grid).
    let uv_values = json!([[0.15, 0.8], [0.8, 0.9], [0.7, 0.15], [0.25, 0.25]]);
    for mapping in mappings {
        for channel in channels {
            let object_id = format!("mask_{}_{}", mapping.to_lowercase(), channel.to_lowercase());
            let group_id = format!("group_{object_id}");
            let modifier_id = format!("modifier_{object_id}");
            let mut node =
                json!({"op":"node.create","id":object_id,"kind":"plane","params":{"size":1.0}});
            if matches!(
                mapping,
                "GLOBAL" | "OBJECT" | "OBJECT_NULL" | "BONE" | "BONE_MISSING"
            ) {
                node["transform"] = json!({
                    "translation":[0.35,-0.2,0.15],
                    "scale":[1.4,0.65,1.0]
                });
            }
            operations.push(node);
            operations.push(json!({"op":"vertex_group.create","target":{"id":object_id},"id":group_id,"name":"A"}));
            operations.push(json!({"op":"vertex_group.assign","target":{"id":object_id},"group_id":group_id,"weights":[
                {"vertex_id":0,"weight":0.2},{"vertex_id":1,"weight":0.4},
                {"vertex_id":2,"weight":0.6},{"vertex_id":3,"weight":0.8}
            ]}));
            let texture_mapping = match mapping {
                "BONE" | "BONE_MISSING" | "OBJECT_NULL" => "OBJECT",
                "UV_FALLBACK" => "UV",
                other => other,
            };
            let mut params = json!({
                "vertex_group":"A","falloff_type":"SHARP","normalize":false,
                "mask_texture":"mask","mask_tex_use_channel":channel,
                "mask_tex_mapping":texture_mapping,"mask_constant":0.75
            });
            if mapping == "OBJECT" {
                params["mask_tex_map_object"] = json!("texture_frame");
            } else if matches!(mapping, "BONE" | "BONE_MISSING") {
                params["mask_tex_map_object"] = json!("texture_rig");
                params["mask_tex_map_bone"] = json!(if mapping == "BONE" {
                    "MaskBone"
                } else {
                    "MissingBone"
                });
            } else if matches!(mapping, "UV" | "UV_FALLBACK") {
                params["mask_tex_uv_layer"] = json!(if mapping == "UV" {
                    "UVMask"
                } else {
                    "MissingUV"
                });
            }
            operations.push(json!({
                "op":"modifier.create","target":{"id":object_id},"id":modifier_id,
                "type":"vertex_weight_edit","params":params
            }));
        }
    }
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":0,"operations":operations
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );

    let mut document: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    for mapping in ["UV", "UV_FALLBACK"] {
        for channel in channels {
            let node_id = Id::new(format!(
                "mask_{}_{}",
                mapping.to_lowercase(),
                channel.to_lowercase()
            ))?;
            let data_id = document
                .nodes
                .get(&node_id)
                .and_then(|node| node.data.as_ref())
                .cloned()
                .ok_or("UV texture test node has no mesh datablock")?;
            let mesh = document
                .data_blocks
                .get_mut(&data_id)
                .and_then(|data| data.mesh.as_mut())
                .ok_or("UV texture test node mesh is missing")?;
            let face = mesh
                .faces
                .first()
                .ok_or("UV texture test mesh has no face")?;
            mesh.attributes.insert(
                "uv_layers".to_owned(),
                json!({
                    "layers":[{"name":"UVMask","faces":[{"face_id":face.id,"uv":uv_values.clone()}]}],
                    "active":"UVMask",
                    "render":"UVMask"
                }),
            );
        }
    }
    let snapshot = Snapshot::evaluate_with_cache(
        &document,
        &EvaluationContext::default(),
        Some(&scene),
        Some(&scene),
    )?;
    for mapping in mappings {
        for channel in channels {
            let object_id = Id::new(format!(
                "mask_{}_{}",
                mapping.to_lowercase(),
                channel.to_lowercase()
            ))?;
            let mesh = snapshot
                .meshes
                .get(&object_id)
                .ok_or("evaluated texture mask mesh is missing")?;
            let label = format!("texture mask {mapping}/{channel}");
            assert_weights(
                &label,
                &weights(mesh, "A")?,
                &expected[&format!("{mapping}_{channel}")],
                3.0e-5,
            )?;
        }
    }
    let local_red = snapshot
        .meshes
        .get(&Id::new("mask_local_red")?)
        .ok_or("LOCAL red texture mesh is missing")?;
    let uv_red = snapshot
        .meshes
        .get(&Id::new("mask_uv_red")?)
        .ok_or("UV red texture mesh is missing")?;
    let uv_fallback_red = snapshot
        .meshes
        .get(&Id::new("mask_uv_fallback_red")?)
        .ok_or("UV fallback red texture mesh is missing")?;
    let local_weights = weights(local_red, "A")?;
    let uv_weights = weights(uv_red, "A")?;
    assert!(
        local_weights
            .iter()
            .zip(&uv_weights)
            .any(|(local, uv)| (local - uv).abs() > 1.0e-4),
        "UV texture mapping fell back to LOCAL coordinates",
    );
    assert_eq!(
        uv_weights,
        weights(uv_fallback_red, "A")?,
        "missing UV layer must use the active layer"
    );
    Ok(())
}

#[test]
fn procedural_texture_weight_mask_matches_blender_clouds_local() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    // Blender leaves scalar CLOUDS alpha unset; pin the measured macOS arm64 5.2.2 build.
    let expected: Value = serde_json::from_str(include_str!(
        "fixtures/procedural_clouds_blender_5_2_2_macos_arm64.json"
    ))?;
    assert_eq!(expected["blender"]["version"].as_str(), Some("5.2.2 LTS"));
    assert_eq!(
        expected["blender"]["build_hash"].as_str(),
        Some("d13f752e3b9c")
    );
    assert_eq!(expected["blender"]["platform"].as_str(), Some("Darwin"));
    assert_eq!(expected["blender"]["machine"].as_str(), Some("arm64"));
    assert_eq!(expected["input"]["texture_type"].as_str(), Some("CLOUDS"));
    assert_eq!(expected["input"]["cloud_type"].as_str(), Some("GRAYSCALE"));
    assert_eq!(
        expected["input"]["group_weights"],
        json!([0.2, 0.4, 0.6, 0.8])
    );
    let scene = root.join("scene");
    let operations_path = root.join("procedural-mask-operations.json");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["init"])
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(
        initialized.status.success(),
        "{}",
        String::from_utf8_lossy(&initialized.stdout)
    );
    let channels = ["INT", "RED", "ALPHA"];
    let input_weights = expected["input"]["group_weights"]
        .as_array()
        .ok_or("CLOUDS fixture group weights are missing")?;
    assert_eq!(input_weights.len(), 4, "CLOUDS fixture vertex count");
    let group_weights = input_weights
        .iter()
        .enumerate()
        .map(|(vertex_id, weight)| json!({"vertex_id":vertex_id,"weight":weight}))
        .collect::<Vec<_>>();
    let mut operations = vec![json!({
        "op":"texture.create","id":"clouds_mask",
        "texture_type":expected["input"]["texture_type"]
    })];
    for channel in channels {
        let object_id = format!("clouds_{}", channel.to_lowercase());
        let group_id = format!("group_{object_id}");
        let settings = &expected["channels"][channel]["settings"];
        operations
            .push(json!({"op":"node.create","id":object_id,"kind":"plane","params":{"size":1.0}}));
        operations.push(
            json!({"op":"vertex_group.create","target":{"id":object_id},"id":group_id,"name":"A"}),
        );
        operations.push(json!({
            "op":"vertex_group.assign","target":{"id":object_id},
            "group_id":group_id,"weights":group_weights.clone()
        }));
        operations.push(json!({
            "op":"modifier.create","target":{"id":object_id},"id":format!("modifier_{object_id}"),
            "type":"vertex_weight_edit","params":{
                "vertex_group":settings["vertex_group"],
                "falloff_type":settings["falloff_type"],
                "invert_falloff":settings["invert_falloff"],
                "normalize":settings["normalize"],
                "default_weight":settings["default_weight"],
                "use_add":settings["use_add"],
                "add_threshold":settings["add_threshold"],
                "use_remove":settings["use_remove"],
                "remove_threshold":settings["remove_threshold"],
                "mask_texture":"clouds_mask",
                "mask_tex_use_channel":settings["channel"],
                "mask_tex_mapping":settings["mapping"],
                "mask_constant":settings["mask_constant"]
            }
        }));
    }
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,"base_revision":0,"operations":operations
        }))?,
    )?;
    let applied = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(["apply"])
        .arg(&scene)
        .args(["--file"])
        .arg(&operations_path)
        .arg("--json")
        .output()?;
    assert!(
        applied.status.success(),
        "{}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let document: SceneDoc = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let snapshot = Snapshot::evaluate_with_cache(
        &document,
        &EvaluationContext::default(),
        Some(&scene),
        Some(&scene),
    )?;
    for channel in channels {
        let object_id = Id::new(format!("clouds_{}", channel.to_lowercase()))?;
        let mesh = snapshot
            .meshes
            .get(&object_id)
            .ok_or("evaluated procedural mask mesh is missing")?;
        let label = format!("procedural CLOUDS LOCAL/{channel}");
        let channel_oracle = &expected["channels"][channel];
        assert_eq!(
            json!(members(mesh, "A")?),
            channel_oracle["members"],
            "{label} vertex-group membership"
        );
        assert_weights(
            &label,
            &weights(mesh, "A")?,
            &channel_oracle["weights"],
            3.0e-5,
        )?;
    }
    Ok(())
}
