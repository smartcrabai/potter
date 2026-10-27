import json
from pathlib import Path

import bpy


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

DIRECTORY = Path(__file__).resolve().parent
SOURCES = ("copy_scale", "copy_location", "floor_volume", "rotation_parenting")
BLEND_PATH = DIRECTORY / "combined_constraints.blend"
JSON_PATH = DIRECTORY / "combined_constraints.json"


def column_major(matrix):
    values = [float(matrix[row][column]) for column in range(4) for row in range(4)]
    return [0.0 if round(value, 7) == 0 else round(value, 7) for value in values]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3

owners = []
source_records = []
for stem in SOURCES:
    source_path = DIRECTORY / f"{stem}.blend"
    oracle_path = DIRECTORY / f"{stem}.json"
    with oracle_path.open(encoding="utf-8") as source_file:
        oracle = json.load(source_file)

    with bpy.data.libraries.load(str(source_path), link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
    source_objects = list(data_to.objects)
    assert source_objects, source_path

    for obj in source_objects:
        obj.name = f"{stem}__{obj.name}"
        if obj.data is not None:
            obj.data.name = f"{stem}__{obj.data.name}"
        scene.collection.objects.link(obj)

    for source_owner in oracle["owners"]:
        owner = dict(source_owner)
        if owner["kind"] == "object":
            source_key = source_owner["name"]
            owner["name"] = f"{stem}__{source_key}"
            capture_key = owner["name"]
        else:
            assert owner["kind"] == "bone", owner
            source_armature = owner["armature"]
            owner["armature"] = f"{stem}__{source_armature}"
            source_key = f"Bone:{source_armature}:{owner['name']}"
            capture_key = f"Bone:{owner['armature']}:{owner['name']}"
        owners.append(owner)
        source_records.append((oracle, source_key, capture_key, owner))

assert len(scene.objects) == len({obj.name for obj in scene.objects})
scene_objects = {obj.name: obj for obj in scene.objects}
for obj in scene.objects:
    assert obj.parent is None or obj.parent.name in scene_objects, obj.name
    for constraint in obj.constraints:
        target = getattr(constraint, "target", None)
        if target is not None:
            assert target.name in scene_objects and scene_objects[target.name] == target, (obj.name, constraint.name)
            subtarget = getattr(constraint, "subtarget", "")
            if subtarget:
                assert target.type == "ARMATURE" and subtarget in target.data.bones, (obj.name, constraint.name)
    if obj.type == "ARMATURE":
        for bone in obj.pose.bones:
            for constraint in bone.constraints:
                target = getattr(constraint, "target", None)
                if target is not None:
                    assert target.name in scene_objects and scene_objects[target.name] == target, (obj.name, bone.name, constraint.name)
                    subtarget = getattr(constraint, "subtarget", "")
                    if subtarget:
                        assert target.type == "ARMATURE" and subtarget in target.data.bones, (obj.name, bone.name, constraint.name)

for _, _, _, owner in source_records:
    if owner["kind"] == "object":
        assert owner["name"] in scene_objects and scene_objects[owner["name"]].type != "ARMATURE", owner
    else:
        armature = scene_objects.get(owner["armature"])
        assert armature is not None and armature.type == "ARMATURE", owner
        assert owner["name"] in armature.data.bones, owner

captured = {}
for frame in (1, 2, 3):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frame_values = {}
    for _, _, capture_key, owner in source_records:
        if owner["kind"] == "object":
            evaluated = scene_objects[owner["name"]].evaluated_get(depsgraph)
            frame_values[capture_key] = column_major(evaluated.matrix_world)
        else:
            armature = scene_objects[owner["armature"]].evaluated_get(depsgraph)
            bone = armature.pose.bones[owner["name"]]
            frame_values[capture_key] = column_major(armature.matrix_world @ bone.matrix)
    assert len(frame_values) == len(owners)
    captured[str(frame)] = frame_values

for oracle, source_key, capture_key, owner in source_records:
    expected_values = [oracle["frames"][str(frame)][source_key] for frame in (1, 2, 3)]
    actual_values = [captured[str(frame)][capture_key] for frame in (1, 2, 3)]
    assert actual_values == expected_values, owner
    if len({tuple(values) for values in expected_values}) > 1:
        assert len({tuple(values) for values in actual_values}) > 1, owner

scene.frame_set(1)
bpy.context.view_layer.update()
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
with JSON_PATH.open("w", encoding="utf-8", newline="\n") as output:
    json.dump({"owners": owners, "frames": captured}, output, separators=(",", ":"), allow_nan=False)
    output.write("\n")
