import bpy
import json
import os
import sys


ROOT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
BLEND_PATH = os.path.join(ROOT, "source.blend")
MEMORY_ONLY_BLEND_PATH = os.path.join(ROOT, "memory_only.blend")
ORACLE_PATH = os.path.join(ROOT, "oracle.json")
FRAME_START = 1
FRAME_END = 9
FRAME_STEP = 1  # Cloth and soft-body cache IDs clamp frame_step to one in Blender 5.2.2.
SAMPLE_FRAMES = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 9.0, 10.0)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = FRAME_START
scene.frame_end = FRAME_END
scene.gravity = (0.0, 0.0, -9.81)
for frame, gravity_z in ((1, -9.81), (3, 2.5), (5, -6.0), (7, 4.0), (9, -12.0)):
    scene.gravity[2] = gravity_z
    scene.keyframe_insert(data_path="gravity", index=2, frame=frame)


def grid_object(name, x_offset):
    width = 5
    vertices = [
        (x_offset + column * 0.35, 0.0, 2.0 - row * 0.5)
        for row in range(width)
        for column in range(width)
    ]
    faces = []
    for row in range(width - 1):
        for column in range(width - 1):
            first = row * width + column
            faces.append((first, first + 1, first + width + 1, first + width))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


cloth_obj = grid_object("ClothBaked", 0.0)
cloth_pins = cloth_obj.vertex_groups.new(name="ClothPins")
cloth_pins.add([0, 4], 1.0, "REPLACE")
cloth = cloth_obj.modifiers.new("BakedCloth", "CLOTH")
cloth.settings.quality = 2
cloth.settings.vertex_group_mass = cloth_pins.name
cloth.point_cache.frame_start = FRAME_START
cloth.point_cache.frame_end = FRAME_END
cloth.point_cache.frame_step = FRAME_STEP
memory_cloth_obj = grid_object("MemoryClothBaked", 4.0)
memory_cloth_pins = memory_cloth_obj.vertex_groups.new(name="MemoryClothPins")
memory_cloth_pins.add([0, 4], 1.0, "REPLACE")
memory_cloth = memory_cloth_obj.modifiers.new("BakedMemoryCloth", "CLOTH")
memory_cloth.settings.quality = 2
memory_cloth.settings.vertex_group_mass = memory_cloth_pins.name
memory_cloth.point_cache.frame_start = FRAME_START
memory_cloth.point_cache.frame_end = FRAME_END
memory_cloth.point_cache.frame_step = FRAME_STEP


soft_obj = grid_object("SoftBodyBaked", 2.0)
soft_goal = soft_obj.vertex_groups.new(name="SoftBodyGoal")
soft_goal.add(list(range(5)), 1.0, "REPLACE")
soft_body = soft_obj.modifiers.new("BakedSoftBody", "SOFT_BODY")
soft_body.settings.use_goal = True
soft_body.settings.vertex_group_goal = soft_goal.name
soft_body.point_cache.frame_start = FRAME_START
soft_body.point_cache.frame_end = FRAME_END
soft_body.point_cache.frame_step = FRAME_STEP

soft_obj.location.x = 0.0
soft_obj.keyframe_insert(data_path="location", frame=FRAME_START)
soft_obj.location.x = 4.0
soft_obj.keyframe_insert(data_path="location", frame=FRAME_START + 1)
scene.frame_set(FRAME_START)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
cloth.point_cache.use_disk_cache = True
soft_body.point_cache.use_disk_cache = True
memory_cloth.point_cache.use_disk_cache = True
soft_body.point_cache.name = "SoftBodyExternalCache"
soft_body.point_cache.use_external = True
soft_body.point_cache.filepath = os.path.join(ROOT, "softbody_external_cache")

for obj, modifier in (
    (cloth_obj, cloth),
    (memory_cloth_obj, memory_cloth),
    (soft_obj, soft_body),
):
    scene.frame_set(FRAME_START)
    cache = modifier.point_cache
    with bpy.context.temp_override(
        scene=scene, object=obj, active_object=obj, point_cache=cache
    ):
        result = bpy.ops.ptcache.bake(bake=True)
    if "FINISHED" not in result or not cache.is_baked:
        raise RuntimeError("failed to bake point cache for " + obj.name)
    if obj is memory_cloth_obj:
        if not cache.use_disk_cache or cache.use_external:
            raise RuntimeError("memory cache must first bake to disk for " + obj.name)
        cache.use_disk_cache = False
        if cache.use_disk_cache or cache.use_external or not cache.is_baked:
            raise RuntimeError("memory point cache settings were not retained for " + obj.name)
    elif not (cache.use_disk_cache or cache.use_external):
        raise RuntimeError("disk point cache settings were not retained for " + obj.name)
    if cache.frame_step != FRAME_STEP:
        raise RuntimeError("point cache settings were not retained for " + obj.name)

bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
bpy.ops.wm.open_mainfile(filepath=BLEND_PATH)
scene = bpy.context.scene
cloth_obj = bpy.data.objects["ClothBaked"]
cloth = cloth_obj.modifiers["BakedCloth"]
memory_cloth_obj = bpy.data.objects["MemoryClothBaked"]
memory_cloth = memory_cloth_obj.modifiers["BakedMemoryCloth"]
soft_obj = bpy.data.objects["SoftBodyBaked"]
soft_body = soft_obj.modifiers["BakedSoftBody"]
if not memory_cloth.point_cache.is_baked or memory_cloth.point_cache.use_disk_cache:
    raise RuntimeError("saved .blend did not retain the memory-baked cloth cache")

objects = (cloth_obj, memory_cloth_obj, soft_obj)
rest = {
    obj.name: [[float(component) for component in vertex.co] for vertex in obj.data.vertices]
    for obj in objects
}
samples = {obj.name: [] for obj in objects}
for frame in SAMPLE_FRAMES:
    whole = int(frame // 1)
    scene.frame_set(whole, subframe=frame - whole)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        if obj is soft_obj and frame == 1.5:
            location_x = float(evaluated.matrix_world.translation.x)
            if not 0.0 < location_x < 4.0:
                raise RuntimeError("soft-body location animation did not evaluate at frame 1.5")
        mesh = evaluated.to_mesh()
        try:
            positions = [
                [float(vertex.co.x), float(vertex.co.y), float(vertex.co.z)]
                for vertex in mesh.vertices
            ]
        finally:
            evaluated.to_mesh_clear()
        samples[obj.name].append({"frame": frame, "positions": positions})
def differs(first, second):
    return any(
        abs(first[index][axis] - second[index][axis]) > 1.0e-5
        for index in range(min(len(first), len(second)))
        for axis in range(3)
    )


for obj in objects:
    by_frame = {sample["frame"]: sample["positions"] for sample in samples[obj.name]}

    if obj is soft_obj and (
        differs(by_frame[1.0], rest[obj.name])
        or not differs(by_frame[2.0], rest[obj.name])
        or differs(by_frame[1.5], rest[obj.name])
    ):
        raise RuntimeError(
            "soft-body frame 1.5 must pass through its input mesh before the cache starts"
        )
    for frame, before, after in ((2.0, 1.0, 3.0), (2.5, 2.0, 3.0), (4.0, 3.0, 5.0)):
        if not differs(by_frame[frame], by_frame[before]) or not differs(
            by_frame[frame], by_frame[after]
        ):
            raise RuntimeError(
                "interpolated point-cache geometry is not discriminating for "
                + obj.name
                + " at frame "
                + str(frame)
            )


for obj in objects:
    if not any(
        abs(position[axis] - rest[obj.name][index][axis]) > 1.0e-5
        for sample in samples[obj.name]
        if sample["frame"] == FRAME_END
        for index, position in enumerate(sample["positions"])
        for axis in range(3)
    ):
        raise RuntimeError("baked motion did not move " + obj.name + " from its rest mesh")

with open(ORACLE_PATH, "w", encoding="utf-8") as output:
    json.dump({"objects": {name: {"rest_positions": rest[name], "samples": samples[name]}
                           for name in samples}}, output, separators=(",", ":"))

for obj in (cloth_obj, soft_obj):
    bpy.data.objects.remove(obj, do_unlink=True)
scene.animation_data_clear()
bpy.data.orphans_purge(do_recursive=True)
bpy.ops.wm.save_as_mainfile(filepath=MEMORY_ONLY_BLEND_PATH)
bpy.ops.wm.open_mainfile(filepath=MEMORY_ONLY_BLEND_PATH)
scene = bpy.context.scene
if len(scene.objects) != 1 or scene.objects[0].name != "MemoryClothBaked":
    raise RuntimeError("memory-only .blend must contain only the baked cloth")
memory_cloth = bpy.data.objects["MemoryClothBaked"].modifiers["BakedMemoryCloth"]
if not memory_cloth.point_cache.is_baked or memory_cloth.point_cache.use_disk_cache:
    raise RuntimeError("memory-only .blend did not retain the memory-baked cloth cache")
