import json
from pathlib import Path

import bpy
from mathutils import Euler, Quaternion


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"oracle requires Blender 5.2.2, found {bpy.app.version_string}")

OUTPUT = Path(__file__).resolve().with_name("oracle.json")
SPACES = [
    "WORLD",
    "POSE",
    "LOCAL_WITH_PARENT",
    "LOCAL",
    "LOCAL_OWNER_ORIENT",
    "CUSTOM",
]
CHANNELS = [
    ("LOCATION_X", -0.5, 1.5),
    ("LOCATION_Y", -1.0, 1.0),
    ("LOCATION_Z", -0.5, 1.5),
    ("ROTATION_X", -45.0, 45.0),
    ("ROTATION_Y", -60.0, 60.0),
    ("ROTATION_Z", -90.0, 90.0),
    ("SCALE_X", 0.5, 2.5),
    ("SCALE_Y", 0.2, 2.0),
    ("SCALE_Z", 0.5, 2.5),
]
MIX_MODES = [
    "REPLACE",
    "BEFORE_FULL",
    "BEFORE",
    "BEFORE_SPLIT",
    "AFTER_FULL",
    "AFTER",
    "AFTER_SPLIT",
]

BLENDER_TO_POTTER_QUATERNION = (3, 0, 1, 2)
QUATERNION_KEYS = (
    Euler((-0.25, 0.4, -0.1), "XYZ").to_quaternion(),
    Euler((0.8, -0.3, 1.0), "XYZ").to_quaternion(),
)
AXIS_ANGLE_KEYS = (
    (0.35, 0.0, 1.0, 0.0),
    (1.1, 0.6, 0.0, 0.8),
)


def transform(translation, rotation, scale, rotation_mode="quaternion"):
    quat = Euler(rotation, "XYZ").to_quaternion()
    return {
        "translation": list(translation),
        "rotation": [quat.x, quat.y, quat.z, quat.w],
        "scale": list(scale),
        "rotation_mode": rotation_mode,
    }


def new_case(index, space, channel, minimum, maximum):
    owner_kind = "object" if index % 2 == 0 else "pose"
    if space in ("POSE", "LOCAL_WITH_PARENT", "LOCAL_OWNER_ORIENT"):
        target_kind = "bone"
    else:
        target_kind = "object" if index % 4 < 2 else "bone"
    rotation_mode = ("quaternion", "XYZ", "AXIS_ANGLE")[index % 3]
    case = {
        "id": f"{space.lower()}_{channel.lower()}",
        "owner_kind": owner_kind,
        "rotation_mode": rotation_mode,
        "owner": f"owner_{index}" if owner_kind == "object" else f"owner_arm_{index}",
        "owner_bone": None if owner_kind == "object" else "owner_bone",
        "target_kind": target_kind,
        "target": "driver" if target_kind == "object" else "target_arm",
        "subtarget": None if target_kind == "object" else "target_bone",
        "channel": channel,
        "target_space": space,
        "min": minimum,
        "max": maximum,
        "mix_mode": MIX_MODES[index % len(MIX_MODES)],
        "frame_start": 1,
        "frame_end": 11,
        "use_eval_time": index == 0,
        "eval_time": 0.72,
        "use_bone_object_action": index == 1,
    }
    if space == "CUSTOM":
        if channel == "LOCATION_Y":
            case["space_object"] = "space_arm"
            case["space_subtarget"] = "SpaceBone"
        else:
            case["space_object"] = "space_ref"
            case["space_subtarget"] = ""
    return case


CASES = [
    new_case(index, space, channel, minimum, maximum)
    for index, (space, (channel, minimum, maximum)) in enumerate(
        (space, channel) for space in SPACES for channel in CHANNELS
    )
]
CASES.append({
    "id": "parented_object_default_owner_space",
    "owner_kind": "object",
    "rotation_mode": "quaternion",
    "owner": "owner_default_space",
    "owner_bone": None,
    "owner_parent": "owner_default_space_parent",
    "owner_parent_transform": transform((2.1, -1.4, 0.75), (0.43, -0.31, 0.62), (0.7, 1.35, 0.9)),
    "owner_transform": transform((-0.55, 0.85, 1.25), (-0.24, 0.51, -0.18), (1.2, 0.8, 1.45)),
    "owner_space_omitted": True,
    "blender_mix_mode_omitted": True,
    "target_kind": "object",
    "target": "driver",
    "subtarget": None,
    "channel": "LOCATION_X",
    "target_space": "WORLD",
    "min": -0.5,
    "max": 1.5,
    "mix_mode": "AFTER",
    "frame_start": 1,
    "frame_end": 11,
    "use_eval_time": False,
    "eval_time": 0.72,
    "use_bone_object_action": False,
})


def operations():
    ops = [
        {"op": "node.create", "id": "driver_parent", "kind": "empty", "transform": transform((1.2, -0.4, 0.7), (0.2, 0.4, -0.2), (0.8, 1.2, 1.1))},
        {"op": "node.create", "id": "driver", "kind": "empty", "parent": "driver_parent", "transform": transform((0.35, -0.4, 0.65), (0.38, -0.52, 0.71), (1.4, 0.7, 1.8))},
        {"op": "node.create", "id": "target_arm", "kind": "armature", "transform": transform((0.4, 0.3, -0.2), (0.14, 0.32, -0.11), (1.1, 0.9, 1.2))},
        {"op": "bone.create", "target": {"id": "target_arm"}, "id": "target_parent_bone", "name": "TargetParent", "head": [1.1, -0.2, 0.3], "tail": [1.7, 0.5, 0.8]},
        {"op": "bone.create", "target": {"id": "target_arm"}, "id": "target_bone", "name": "TargetBone", "head": [2.0, 0.4, 0.1], "tail": [2.2, 1.3, 0.4], "parent": "target_parent_bone"},
        {"op": "pose.set", "target": {"id": "target_arm"}, "bone_id": "target_parent_bone", "set": {"translation": [-0.2, 0.1, 0.25], "rotation": transform((0, 0, 0), (-0.18, 0.26, -0.33), (1, 1, 1))["rotation"], "scale": [0.9, 1.1, 0.95]}},
        {"op": "pose.set", "target": {"id": "target_arm"}, "bone_id": "target_bone", "set": {"translation": [0.25, -0.3, 0.45], "rotation": transform((0, 0, 0), (0.28, -0.19, 0.42), (1, 1, 1))["rotation"], "scale": [1.1, 0.85, 1.25]}},
        {"op": "node.create", "id": "space_ref", "kind": "empty", "transform": transform((-0.7, 1.3, 0.9), (-0.31, 0.22, -0.18), (1.0, 1.0, 1.0))},
        {"op": "node.create", "id": "space_arm", "kind": "armature", "transform": transform((0.5, -1.1, 0.6), (0.23, 0.39, -0.16), (1.0, 1.0, 1.0))},
        {"op": "bone.create", "target": {"id": "space_arm"}, "id": "space_bone", "name": "SpaceBone", "head": [0.3, -0.2, 0.5], "tail": [0.9, 0.8, 0.7]},
        {"op": "action.create", "id": "drive", "name": "Action Constraint Oracle"},
    ]
    curves = []
    components = [
        ("transform.translation", "pose.owner_bone.location", 0, -1.0, 4.0),
        ("transform.translation", "pose.owner_bone.location", 1, 2.0, -3.0),
        ("transform.translation", "pose.owner_bone.location", 2, 0.5, 7.0),
        ("transform.rotation_euler", "pose.owner_bone.rotation_euler", 0, -0.35, 1.1),
        ("transform.rotation_euler", "pose.owner_bone.rotation_euler", 1, 0.2, -0.9),
        ("transform.rotation_euler", "pose.owner_bone.rotation_euler", 2, -0.5, 1.7),
        ("transform.scale", "pose.owner_bone.scale", 0, 0.5, 2.6),
        ("transform.scale", "pose.owner_bone.scale", 1, 1.8, 0.4),
        ("transform.scale", "pose.owner_bone.scale", 2, 0.7, 1.9),
    ]
    def add_curve(path, index, start, end):
        curves.append({
            "path": path,
            "index": index,
            "keyframes": [
                {"frame": 1.0, "value": start, "interpolation": "linear"},
                {"frame": 11.0, "value": end, "interpolation": "linear"},
            ],
        })

    for object_path, pose_path, index, start, end in components:
        for path in (object_path, pose_path):
            add_curve(path, index, start, end)
    for potter_index, component in enumerate(("x", "y", "z", "w")):
        for path in ("transform.rotation_quaternion", "pose.owner_bone.rotation"):
            add_curve(
                path,
                potter_index,
                getattr(QUATERNION_KEYS[0], component),
                getattr(QUATERNION_KEYS[1], component),
            )
    for index in range(4):
        for path in ("transform.rotation_axis_angle", "pose.owner_bone.rotation_axis_angle"):
            add_curve(path, index, AXIS_ANGLE_KEYS[0][index], AXIS_ANGLE_KEYS[1][index])
    ops.append({"op": "action.update", "target": {"id": "drive"}, "set": {"fcurves": curves}})

    for case in CASES:
        if case.get("owner_parent"):
            ops.append({
                "op": "node.create",
                "id": case["owner_parent"],
                "kind": "empty",
                "transform": case["owner_parent_transform"],
            })
        if case["owner_kind"] == "object":
            owner_node = {
                "op": "node.create",
                "id": case["owner"],
                "kind": "empty",
                "transform": case.get(
                    "owner_transform",
                    transform(
                        (1.2, -0.7, 2.0),
                        (0.17, -0.22, 0.31),
                        (1.1, 0.9, 1.2),
                        case["rotation_mode"],
                    ),
                ),
            }
            if case.get("owner_parent"):
                owner_node["parent"] = case["owner_parent"]
            ops.append(owner_node)
        else:
            ops.extend([
                {"op": "node.create", "id": case["owner"], "kind": "armature", "transform": transform((1.2, -0.7, 2.0), (0.17, -0.22, 0.31), (1.1, 0.9, 1.2))},
                {"op": "bone.create", "target": {"id": case["owner"]}, "id": case["owner_bone"], "name": "OwnerBone", "head": [0.2, -0.3, 0.5], "tail": [1.2, 0.8, 0.7]},
                {"op": "pose.set", "target": {"id": case["owner"]}, "bone_id": case["owner_bone"], "set": {"translation": [0.15, -0.25, 0.4], "rotation": transform((0, 0, 0), (-0.13, 0.34, 0.21), (1, 1, 1))["rotation"], "rotation_mode": case["rotation_mode"], "scale": [1.05, 0.95, 1.1]}},
            ])
        params = {
            "target": case["target"],
            "action": "drive",
            "transform_channel": case["channel"],
            "target_space": case["target_space"],
            "min": case["min"],
            "max": case["max"],
            "frame_start": case["frame_start"],
            "frame_end": case["frame_end"],
            "use_eval_time": case["use_eval_time"],
            "eval_time": case["eval_time"],
            "use_bone_object_action": case["use_bone_object_action"],
        }
        params["mix_mode"] = case["mix_mode"]
        if not case.get("owner_space_omitted", False):
            params["owner_space"] = "WORLD"
        if "space_object" in case:
            params["space_object"] = case["space_object"]
            params["space_subtarget"] = case["space_subtarget"]
        constraint = {
            "op": "constraint.create",
            "target": {"id": case["owner"]},
            "id": "action_" + case["id"],
            "type": "action",
            "constraint_target": case["target"],
            "params": params,
        }
        if case["owner_bone"] is not None:
            constraint["owner_bone"] = case["owner_bone"]
        if case["subtarget"] is not None:
            constraint["subtarget"] = case["subtarget"]
        ops.append(constraint)
    return ops


def set_transform(obj, value):
    obj.location = value["translation"]
    rotation = value["rotation"]
    quat = Quaternion((rotation[3], rotation[0], rotation[1], rotation[2]))
    mode = value.get("rotation_mode", "quaternion")
    obj.rotation_mode = mode.upper()
    if mode == "AXIS_ANGLE":
        axis, angle = quat.to_axis_angle()
        obj.rotation_axis_angle = (angle, *axis)
    elif mode == "quaternion":
        obj.rotation_quaternion = quat
    else:
        obj.rotation_euler = quat.to_euler(mode)
    obj.scale = value["scale"]


def make_armature(scene, name, bone_name, transform_value, head, tail, parent_spec=None):
    bpy.ops.object.select_all(action="DESELECT")
    data = bpy.data.armatures.new(name + "Data")
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    parent = None
    if parent_spec is not None:
        parent = data.edit_bones.new(parent_spec[0])
        parent.head = parent_spec[1]
        parent.tail = parent_spec[2]
    bone = data.edit_bones.new(bone_name)
    bone.head = head
    bone.tail = tail
    if parent is not None:
        bone.parent = parent
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    set_transform(obj, transform_value)
    return obj


def set_pose(obj, bone_name, translation, rotation, scale, rotation_mode="quaternion"):
    pose_bone = obj.pose.bones[bone_name]
    pose_bone.location = translation
    quat = Quaternion((rotation[3], rotation[0], rotation[1], rotation[2]))
    pose_bone.rotation_mode = rotation_mode.upper()
    if rotation_mode == "AXIS_ANGLE":
        axis, angle = quat.to_axis_angle()
        pose_bone.rotation_axis_angle = (angle, *axis)
    elif rotation_mode == "quaternion":
        pose_bone.rotation_quaternion = quat
    else:
        pose_bone.rotation_euler = quat.to_euler(rotation_mode)
    pose_bone.scale = scale


def matrix_values(matrix):
    return [float(matrix[row][column]) for column in range(4) for row in range(4)]


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene


def add_empty(name, transform_value, parent=None):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    set_transform(obj, transform_value)
    return obj


driver_parent = add_empty("DriverParent", transform((1.2, -0.4, 0.7), (0.2, 0.4, -0.2), (0.8, 1.2, 1.1)))
driver = add_empty("Driver", transform((0.35, -0.4, 0.65), (0.38, -0.52, 0.71), (1.4, 0.7, 1.8)), driver_parent)
target_arm = make_armature(
    scene,
    "TargetArmature",
    "TargetBone",
    transform((0.4, 0.3, -0.2), (0.14, 0.32, -0.11), (1.1, 0.9, 1.2)),
    (2.0, 0.4, 0.1),
    (2.2, 1.3, 0.4),
    ("TargetParent", (1.1, -0.2, 0.3), (1.7, 0.5, 0.8)),
)
set_pose(target_arm, "TargetParent", (-0.2, 0.1, 0.25), transform((0, 0, 0), (-0.18, 0.26, -0.33), (1, 1, 1))["rotation"], (0.9, 1.1, 0.95))
set_pose(target_arm, "TargetBone", (0.25, -0.3, 0.45), transform((0, 0, 0), (0.28, -0.19, 0.42), (1, 1, 1))["rotation"], (1.1, 0.85, 1.25))
space_ref = add_empty("SpaceReference", transform((-0.7, 1.3, 0.9), (-0.31, 0.22, -0.18), (1, 1, 1)))
space_arm = make_armature(scene, "SpaceArmature", "SpaceBone", transform((0.5, -1.1, 0.6), (0.23, 0.39, -0.16), (1, 1, 1)), (0.3, -0.2, 0.5), (0.9, 0.8, 0.7))

owners = {}
for case in CASES:
    if case["owner_kind"] == "object":
        parent = None
        if case.get("owner_parent"):
            parent = add_empty(case["owner_parent"], case["owner_parent_transform"])
        owner = add_empty(
            case["owner"],
            case.get(
                "owner_transform",
                transform((1.2, -0.7, 2.0), (0.17, -0.22, 0.31), (1.1, 0.9, 1.2), case["rotation_mode"]),
            ),
            parent,
        )
    else:
        owner = make_armature(scene, case["owner"], "OwnerBone", transform((1.2, -0.7, 2.0), (0.17, -0.22, 0.31), (1.1, 0.9, 1.2)), (0.2, -0.3, 0.5), (1.2, 0.8, 0.7))
        set_pose(owner, "OwnerBone", (0.15, -0.25, 0.4), transform((0, 0, 0), (-0.13, 0.34, 0.21), (1, 1, 1))["rotation"], (1.05, 0.95, 1.1), case["rotation_mode"])
    owners[case["id"]] = owner

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
baselines = {}
for case in CASES:
    owner = owners[case["id"]].evaluated_get(depsgraph)
    if case["owner_kind"] == "pose":
        matrix = owner.matrix_world @ owner.pose.bones["OwnerBone"].matrix
    else:
        matrix = owner.matrix_world
    baselines[case["id"]] = matrix_values(matrix)


action = bpy.data.actions.new("Action Constraint Oracle")
slot = action.slots.new(id_type="OBJECT", name="Action Constraint Oracle")
dummy = bpy.data.objects.new("ActionSlotOwner", None)
scene.collection.objects.link(dummy)
dummy.animation_data_create()
dummy.animation_data.action = action
dummy.animation_data.action_slot = slot
curves = [
    ("location", 'pose.bones["OwnerBone"].location', 0, -1.0, 4.0),
    ("location", 'pose.bones["OwnerBone"].location', 1, 2.0, -3.0),
    ("location", 'pose.bones["OwnerBone"].location', 2, 0.5, 7.0),
    ("rotation_euler", 'pose.bones["OwnerBone"].rotation_euler', 0, -0.35, 1.1),
    ("rotation_euler", 'pose.bones["OwnerBone"].rotation_euler', 1, 0.2, -0.9),
    ("rotation_euler", 'pose.bones["OwnerBone"].rotation_euler', 2, -0.5, 1.7),
    ("scale", 'pose.bones["OwnerBone"].scale', 0, 0.5, 2.6),
    ("scale", 'pose.bones["OwnerBone"].scale', 1, 1.8, 0.4),
    ("scale", 'pose.bones["OwnerBone"].scale', 2, 0.7, 1.9),
]
def add_blender_curve(path, index, start, end):
    curve = action.fcurve_ensure_for_datablock(dummy, path, index=index)
    for frame, value in ((1.0, start), (11.0, end)):
        point = curve.keyframe_points.insert(frame, value)
        point.interpolation = "LINEAR"
    curve.update()


for object_path, pose_path, index, start, end in curves:
    for path in (object_path, pose_path):
        add_blender_curve(path, index, start, end)

# Blender quaternion channels are WXYZ; Potter action fcurves store XYZW.
for blender_index, potter_index in enumerate(BLENDER_TO_POTTER_QUATERNION):
    component = ("x", "y", "z", "w")[potter_index]
    for path in ("rotation_quaternion", 'pose.bones["OwnerBone"].rotation_quaternion'):
        add_blender_curve(
            path,
            blender_index,
            getattr(QUATERNION_KEYS[0], component),
            getattr(QUATERNION_KEYS[1], component),
        )
for index in range(4):
    for path in ("rotation_axis_angle", 'pose.bones["OwnerBone"].rotation_axis_angle'):
        add_blender_curve(path, index, AXIS_ANGLE_KEYS[0][index], AXIS_ANGLE_KEYS[1][index])
dummy.animation_data.action = None

for case in CASES:
    owner = owners[case["id"]]
    constraint_owner = owner if case["owner_kind"] == "object" else owner.pose.bones["OwnerBone"]
    constraint = constraint_owner.constraints.new("ACTION")
    constraint.target = driver if case["target_kind"] == "object" else target_arm
    if case["subtarget"] is not None:
        constraint.subtarget = "TargetBone"
    constraint.action = action
    constraint.action_slot = slot
    constraint.transform_channel = case["channel"]
    constraint.target_space = case["target_space"]
    constraint.min = case["min"]
    constraint.max = case["max"]
    constraint.frame_start = case["frame_start"]
    constraint.frame_end = case["frame_end"]
    if not case.get("blender_mix_mode_omitted", False):
        constraint.mix_mode = case["mix_mode"]
    if not case.get("owner_space_omitted", False):
        constraint.owner_space = "WORLD"
    constraint.use_eval_time = case["use_eval_time"]
    constraint.eval_time = case["eval_time"]
    constraint.use_bone_object_action = case["use_bone_object_action"]
    if case.get("space_object"):
        constraint.space_object = space_arm if case["space_object"] == "space_arm" else space_ref
        constraint.space_subtarget = case["space_subtarget"]

scene.frame_set(1)
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
expected = {}
for case in CASES:
    owner = owners[case["id"]].evaluated_get(depsgraph)
    if case["owner_kind"] == "pose":
        matrix = owner.matrix_world @ owner.pose.bones["OwnerBone"].matrix
    else:
        matrix = owner.matrix_world
    expected[case["id"]] = {"matrix": matrix_values(matrix), "baseline": baselines[case["id"]]}


fixture = {
    "blender_version": bpy.app.version_string,
    "operations": operations(),
    "cases": CASES,
    "expected": expected,
}
OUTPUT.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {len(CASES)} Action constraint cases to {OUTPUT}")
