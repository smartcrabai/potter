import json
from pathlib import Path

import bpy
from mathutils import Euler, Matrix


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
JSON_PATH = DIRECTORY / "initial_constraint_captures.json"
FRAME = 2


def model_transform(value):
    location, rotation, scale = value
    quaternion = Euler(rotation, "XYZ").to_quaternion()
    return {
        "translation": [float(component) for component in location],
        "rotation": [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)],
        "scale": [float(component) for component in scale],
        "rotation_mode": "XYZ",
    }


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0.0 else round(value, 7) for value in values]


def action_curves(action):
    curves = []
    for slot in getattr(action, "slots", ()):
        for layer in action.layers:
            for strip in layer.strips:
                if strip.type == "KEYFRAME":
                    channelbag = strip.channelbag(slot)
                    if channelbag is not None:
                        curves.extend(channelbag.fcurves)
    return curves or list(getattr(action, "fcurves", ()))


def make_animated_transform(obj, frame_one, frame_two):
    obj.rotation_mode = "XYZ"
    for frame, (location, rotation, scale) in ((1, frame_one), (FRAME, frame_two)):
        obj.location = location
        obj.rotation_euler = Euler(rotation, "XYZ")
        obj.scale = scale
        for channel in ("location", "rotation_euler", "scale"):
            obj.keyframe_insert(data_path=channel, frame=frame)
    for curve in action_curves(obj.animation_data.action):
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def make_animated_empty(name, frame_one, frame_two):
    obj = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(obj)
    make_animated_transform(obj, frame_one, frame_two)
    return obj


def fcurves_for_node(obj):
    path_map = {
        "location": "transform.translation",
        "rotation_euler": "transform.rotation_euler",
        "scale": "transform.scale",
    }
    curves = []
    for curve in action_curves(obj.animation_data.action):
        path = path_map.get(curve.data_path)
        if path is None:
            continue
        curves.append({
            "path": path,
            "index": int(curve.array_index),
            "keyframes": [
                {
                    "frame": float(point.co.x),
                    "value": float(point.co.y),
                    "interpolation": point.interpolation.lower(),
                }
                for point in curve.keyframe_points
            ],
            "extrapolation": curve.extrapolation.lower(),
        })
    return curves


def make_constraint(owner, specification, target):
    owner_bone = specification.get("owner_bone")
    collection = owner.pose.bones[owner_bone].constraints if owner_bone else owner.constraints
    constraint = collection.new(specification["kind"].upper())
    constraint.name = specification["key"]
    constraint.target = target
    constraint.owner_space = "WORLD"
    constraint.target_space = "WORLD"
    if owner_bone:
        constraint.influence = specification["influence"]
    return constraint


def set_constraint_params(constraint, kind, params, *, initialize=False):
    params = params or {}
    if kind == "limit_distance":
        constraint.distance = params.get("distance", 0.0)
        if initialize:
            constraint.limit_mode = "LIMITDIST_INSIDE"
    elif kind == "stretch_to":
        constraint.rest_length = params.get("rest_length", 0.0)
    else:
        constraint.set_inverse_pending = params.get("set_inverse_pending", True)
        if "inverse_matrix" in params:
            constraint.inverse_matrix = Matrix([
                [params["inverse_matrix"][column * 4 + row] for column in range(4)]
                for row in range(4)
            ])


def constraint_cache(constraint, kind):
    if kind == "limit_distance":
        return {"distance": round(float(constraint.distance), 7)}
    if kind == "stretch_to":
        return {"rest_length": round(float(constraint.rest_length), 7)}
    return {
        "set_inverse_pending": bool(constraint.set_inverse_pending),
        "inverse_matrix": column_major(constraint.inverse_matrix),
    }


def owner_world_matrix(specification, depsgraph):
    if specification.get("owner_bone"):
        evaluated_armature = armature_object.evaluated_get(depsgraph)
        pose_bone = evaluated_armature.pose.bones[specification["owner_bone"]]
        return column_major(evaluated_armature.matrix_world @ pose_bone.matrix)
    owner = objects[specification["key"]][0]
    return column_major(owner.evaluated_get(depsgraph).matrix_world)


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = FRAME
scene.frame_set(1)

specifications = [
    {
        "key": "create_limit_distance",
        "kind": "limit_distance",
        "owner_id": "create_limit_owner",
        "target_id": "create_limit_target",
        "frame_one_owner": ((0.1, -0.2, 0.3), (0.1, -0.15, 0.2), (1.0, 1.0, 1.0)),
        "frame_two_owner": ((-0.35, 0.25, 0.15), (0.2, -0.25, 0.3), (1.0, 1.0, 1.0)),
        "frame_one_target": ((2.0, 0.5, -0.2), (-0.1, 0.2, 0.15), (1.0, 1.0, 1.0)),
        "frame_two_target": ((2.6, 1.1, -0.45), (-0.2, 0.25, -0.1), (1.0, 1.0, 1.0)),
        "create_params": None,
        "update_params": None,
    },
    {
        "key": "update_limit_distance",
        "kind": "limit_distance",
        "owner_id": "update_limit_owner",
        "target_id": "update_limit_target",
        "frame_one_owner": ((-0.2, 0.1, 0.4), (0.05, 0.1, -0.2), (1.0, 1.0, 1.0)),
        "frame_two_owner": ((0.35, -0.4, 0.2), (-0.15, 0.3, 0.25), (1.0, 1.0, 1.0)),
        "frame_one_target": ((1.8, 0.6, 0.1), (-0.1, -0.2, 0.25), (1.0, 1.0, 1.0)),
        "frame_two_target": ((-1.2, 1.7, 0.85), (0.15, 0.25, -0.3), (1.0, 1.0, 1.0)),
        "create_params": {"distance": 2.0},
        "update_params": {"distance": 0.0},
    },
    {
        "key": "create_stretch_to",
        "kind": "stretch_to",
        "owner_id": "create_stretch_owner",
        "target_id": "create_stretch_target",
        "frame_one_owner": ((0.1, 0.2, -0.15), (0.15, -0.2, 0.25), (1.0, 1.4, 1.0)),
        "frame_two_owner": ((-0.25, 0.1, 0.35), (-0.2, 0.3, 0.1), (1.0, 1.8, 1.0)),
        "frame_one_target": ((0.4, 5.0, 0.2), (0.0, 0.1, -0.2), (1.0, 1.0, 1.0)),
        "frame_two_target": ((0.2, 8.3, -0.35), (0.1, -0.2, 0.2), (1.0, 1.0, 1.0)),
        "create_params": None,
        "update_params": None,
    },
    {
        "key": "update_stretch_to",
        "kind": "stretch_to",
        "owner_id": "update_stretch_owner",
        "target_id": "update_stretch_target",
        "frame_one_owner": ((-0.2, -0.1, 0.25), (0.1, 0.2, -0.1), (1.0, 1.25, 1.0)),
        "frame_two_owner": ((0.3, 0.25, -0.2), (-0.15, -0.25, 0.3), (1.0, 2.2, 1.0)),
        "frame_one_target": ((0.6, 4.2, -0.25), (-0.15, 0.1, 0.25), (1.0, 1.0, 1.0)),
        "frame_two_target": ((-0.4, 9.5, 0.75), (0.2, -0.1, -0.25), (1.0, 1.0, 1.0)),
        "create_params": {"rest_length": 3.0},
        "update_params": {"rest_length": 0.0},
    },
    {
        "key": "create_child_of",
        "kind": "child_of",
        "owner_id": "create_child_owner",
        "target_id": "create_child_target",
        "frame_one_owner": ((-0.25, 0.15, 0.4), (0.1, -0.25, 0.15), (1.0, 1.0, 1.0)),
        "frame_two_owner": ((0.35, -0.2, 0.45), (0.2, 0.35, -0.15), (1.0, 1.0, 1.0)),
        "frame_one_target": ((1.2, 0.4, -0.2), (0.15, -0.3, 0.2), (1.1, 0.9, 1.2)),
        "frame_two_target": ((-0.8, 1.35, 0.55), (-0.25, 0.4, 0.3), (0.85, 1.2, 1.1)),
        "create_params": None,
        "update_params": None,
    },
    {
        "key": "update_child_of",
        "kind": "child_of",
        "owner_id": "update_child_owner",
        "target_id": "update_child_target",
        "frame_one_owner": ((0.15, -0.3, 0.25), (-0.2, 0.1, 0.35), (1.0, 1.0, 1.0)),
        "frame_two_owner": ((-0.45, 0.35, -0.15), (0.25, -0.3, 0.2), (1.0, 1.0, 1.0)),
        "frame_one_target": ((-1.0, 0.75, 0.2), (-0.1, 0.2, -0.35), (1.2, 0.8, 1.1)),
        "frame_two_target": ((1.6, -0.55, 0.8), (0.3, 0.15, -0.25), (0.9, 1.1, 1.25)),
        "create_params": {"set_inverse_pending": False, "inverse_matrix": column_major(Matrix.Identity(4))},
        "update_params": {"set_inverse_pending": True},
    },
]

ARMATURE_ID = "capture_armature"
FRAME_ONE_ARMATURE = ((1.25, -0.75, 0.5), (0.2, -0.3, 0.15), (1.1, 0.9, 1.2))
FRAME_TWO_ARMATURE = ((-0.65, 1.1, 0.35), (-0.25, 0.4, -0.2), (0.85, 1.2, 0.95))
BONE_SETUP = [
    {"id": "bone_limit_distance", "name": "bone_limit_distance",
     "head": (0.5, -0.4, 0.2), "tail": (0.5, 0.6, 0.2)},
    {"id": "bone_stretch_to", "name": "bone_stretch_to",
     "head": (-0.8, 0.25, 0.4), "tail": (-0.8, 1.25, 0.4)},
    {"id": "bone_child_of", "name": "bone_child_of",
     "head": (0.2, 0.55, -0.3), "tail": (0.2, 1.55, -0.3)},
]
BONE_SPECIFICATIONS = [
    {"key": "bone_limit_distance", "kind": "limit_distance",
     "owner_id": ARMATURE_ID, "owner_bone": "bone_limit_distance",
     "target_id": "create_limit_target", "influence": 0.35,
     "create_params": None, "update_params": None},
    {"key": "bone_stretch_to", "kind": "stretch_to",
     "owner_id": ARMATURE_ID, "owner_bone": "bone_stretch_to",
     "target_id": "create_stretch_target", "influence": 0.55,
     "create_params": None, "update_params": None},
    {"key": "bone_child_of", "kind": "child_of",
     "owner_id": ARMATURE_ID, "owner_bone": "bone_child_of",
     "target_id": "create_child_target", "influence": 0.7,
     "create_params": None, "update_params": None},
]

armature_data = bpy.data.armatures.new(ARMATURE_ID + "_data")
armature_object = bpy.data.objects.new(ARMATURE_ID, armature_data)
scene.collection.objects.link(armature_object)
make_animated_transform(armature_object, FRAME_ONE_ARMATURE, FRAME_TWO_ARMATURE)
for selected in bpy.context.selected_objects:
    selected.select_set(False)
armature_object.select_set(True)
bpy.context.view_layer.objects.active = armature_object
bpy.ops.object.mode_set(mode="EDIT")
for setup in BONE_SETUP:
    bone = armature_data.edit_bones.new(setup["name"])
    bone.head = setup["head"]
    bone.tail = setup["tail"]
bpy.ops.object.mode_set(mode="OBJECT")

bone_records = [
    {
        "armature_id": ARMATURE_ID,
        "id": bone.name,
        "name": bone.name,
        "head": [float(component) for component in bone.head_local],
        "tail": [float(component) for component in bone.tail_local],
    }
    for bone in armature_data.bones
]

objects = {}
objects_by_id = {}
for specification in specifications:
    owner = make_animated_empty(
        specification["owner_id"],
        specification["frame_one_owner"],
        specification["frame_two_owner"],
    )
    target = make_animated_empty(
        specification["target_id"],
        specification["frame_one_target"],
        specification["frame_two_target"],
    )
    objects[specification["key"]] = (owner, target)
    objects_by_id[specification["owner_id"]] = owner
    objects_by_id[specification["target_id"]] = target
objects_by_id[ARMATURE_ID] = armature_object

update_constraints = {}
for specification in specifications:
    if specification["key"].startswith("create_"):
        continue
    owner, target = objects[specification["key"]]
    constraint = make_constraint(owner, specification, target)
    set_constraint_params(
        constraint, specification["kind"], specification["create_params"], initialize=True
    )
    update_constraints[specification["key"]] = constraint

scene.frame_set(1)
bpy.context.view_layer.update()
for specification in specifications:
    if specification["key"].startswith("create_"):
        continue
    set_constraint_params(
        update_constraints[specification["key"]],
        specification["kind"],
        specification["update_params"],
    )

frame_one_constraints = {}
for specification in specifications:
    if not specification["key"].startswith("create_"):
        continue
    owner, target = objects[specification["key"]]
    constraint = make_constraint(owner, specification, target)
    set_constraint_params(
        constraint, specification["kind"], specification["create_params"], initialize=True
    )
    frame_one_constraints[specification["key"]] = constraint
for specification in BONE_SPECIFICATIONS:
    constraint = make_constraint(
        armature_object, specification, objects_by_id[specification["target_id"]]
    )
    set_constraint_params(
        constraint, specification["kind"], specification["create_params"], initialize=True
    )
    frame_one_constraints[specification["key"]] = constraint

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
frame_one_caches = {}
frame_one_world_matrices = {}
for specification in specifications + BONE_SPECIFICATIONS:
    constraint = update_constraints.get(specification["key"])
    if constraint is None:
        constraint = frame_one_constraints[specification["key"]]
    frame_one_caches[specification["key"]] = constraint_cache(
        constraint, specification["kind"]
    )
    frame_one_world_matrices[specification["key"]] = owner_world_matrix(
        specification, depsgraph
    )

for specification in specifications:
    if specification["key"].startswith("create_"):
        owner, _ = objects[specification["key"]]
        owner.constraints.remove(frame_one_constraints[specification["key"]])
for specification in BONE_SPECIFICATIONS:
    armature_object.pose.bones[specification["owner_bone"]].constraints.remove(
        frame_one_constraints[specification["key"]]
    )

scene.frame_set(FRAME)
for key, constraint in update_constraints.items():
    specification = next(item for item in specifications if item["key"] == key)
    set_constraint_params(
        constraint, specification["kind"], specification["update_params"]
    )

all_constraints = dict(update_constraints)
for specification in specifications:
    if not specification["key"].startswith("create_"):
        continue
    owner, target = objects[specification["key"]]
    constraint = make_constraint(owner, specification, target)
    set_constraint_params(
        constraint, specification["kind"], specification["create_params"], initialize=True
    )
    all_constraints[specification["key"]] = constraint
for specification in BONE_SPECIFICATIONS:
    constraint = make_constraint(
        armature_object, specification, objects_by_id[specification["target_id"]]
    )
    set_constraint_params(
        constraint, specification["kind"], specification["create_params"], initialize=True
    )
    all_constraints[specification["key"]] = constraint

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
node_records = []
constraint_records = []
for specification in specifications:
    owner, target = objects[specification["key"]]
    for node_id, node, frame_one, frame_two in (
        (specification["owner_id"], owner,
         specification["frame_one_owner"], specification["frame_two_owner"]),
        (specification["target_id"], target,
         specification["frame_one_target"], specification["frame_two_target"]),
    ):
        node_records.append({
            "id": node_id,
            "kind": "empty",
            "transform": model_transform(frame_two),
            "frame_one_transform": model_transform(frame_one),
            "frame_two_transform": model_transform(frame_two),
            "fcurves": fcurves_for_node(node),
        })

node_records.append({
    "id": ARMATURE_ID,
    "kind": "armature",
    "transform": model_transform(FRAME_TWO_ARMATURE),
    "frame_one_transform": model_transform(FRAME_ONE_ARMATURE),
    "frame_two_transform": model_transform(FRAME_TWO_ARMATURE),
    "fcurves": fcurves_for_node(armature_object),
    "bones": bone_records,
})

for specification in specifications + BONE_SPECIFICATIONS:
    constraint = all_constraints[specification["key"]]
    cache = constraint_cache(constraint, specification["kind"])
    if specification["kind"] in ("limit_distance", "stretch_to"):
        value_name = "distance" if specification["kind"] == "limit_distance" else "rest_length"
        assert cache[value_name] > 0.0
    else:
        assert not constraint.set_inverse_pending
    record = {
        "key": specification["key"],
        "owner_id": specification["owner_id"],
        "target_id": specification["target_id"],
        "type": specification["kind"],
        "create_params": specification["create_params"],
        "update_params": specification["update_params"],
        "frame_one_cache": frame_one_caches[specification["key"]],
        "cache": cache,
        "frame_one_world_matrix": frame_one_world_matrices[specification["key"]],
        "world_matrix": owner_world_matrix(specification, depsgraph),
    }
    if specification.get("owner_bone"):
        record["owner_bone"] = specification["owner_bone"]
        record["influence"] = specification["influence"]
    assert record["frame_one_cache"] != record["cache"], (
        specification["key"], record["frame_one_cache"], record["cache"]
    )
    constraint_records.append(record)

assert scene.frame_current == FRAME
assert len(node_records) == 13 and len(constraint_records) == 9
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"frame": FRAME, "nodes": node_records, "constraints": constraint_records}, output,
              separators=(",", ":"), allow_nan=False)
    output.write("\n")
