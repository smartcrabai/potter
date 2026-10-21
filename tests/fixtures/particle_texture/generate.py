"""Blender 5.2.2 oracle for particle emission schedule and texture influences.

Usage (from the repo root):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tests/fixtures/particle_texture/generate.py -- <out_dir>

Writes into <out_dir>:
    particle_texture.blend   the scene Potter imports
    blender_expected.json    Blender's per-particle values at each case's evaluation frame

Each case is one emitter object with one vertex, so every particle is born at the same point and
Potter's emitter sample is deterministic. Density cases record how many particles exist: Blender's
evaluated particle list leaves out the particles that the density texture removes.
"""

import json
import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("particle texture oracle requires Blender 5.2.2")

OUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
POINT = (0.3, 0.2, 0.1)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
texture = bpy.data.textures.new("PotterBlend", type="BLEND")
texture.progression = "LINEAR"


def point_object(name, location, vertex=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([vertex], [], [])
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    scene.collection.objects.link(obj)
    return obj


def add_emitter(obj, count, frame_start, frame_end, lifetime, particle_size):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.particle_system_add()
    system = obj.particle_systems[-1]
    system.seed = 3
    settings = system.settings
    settings.count = count
    settings.frame_start = frame_start
    settings.frame_end = frame_end
    settings.lifetime = lifetime
    settings.lifetime_random = 0.0
    settings.emit_from = "VERT"
    settings.physics_type = "NO"
    settings.normal_factor = 0.0
    settings.particle_size = particle_size
    settings.size_random = 0.0
    return settings


def add_slot(settings, channel, factor, default, coords="GLOBAL", anchor=None,
             scale=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0), blend="MIX"):
    slot = settings.texture_slots.add()
    slot.texture = texture
    slot.texture_coords = coords
    if anchor is not None:
        slot.object = anchor
    slot.mapping_x = "X"
    slot.mapping_y = "Y"
    slot.mapping_z = "Z"
    slot.scale = scale
    slot.offset = offset
    slot.default_value = default
    slot.blend_type = blend
    setattr(slot, "use_map_" + channel, True)
    setattr(slot, channel + "_factor", factor)
    return slot


def particle_values(obj):
    particles = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0].particles
    return [
        {"birth_frame": float(pa.birth_time), "size": float(pa.size),
         "lifetime": float(pa.lifetime)}
        for pa in particles
    ]


# Each case names its object and the frame it is read at. Blender's particle reset on a backward
# frame jump samples textures before the birth point is set, so the scene is stepped forward once.
cases = {}
reads = []


def case(name, obj, frame, visible=False):
    reads.append((frame, name, obj, visible))


# Schedule: particle i is born at frame_start + (frame_end - frame_start) * i / count.
obj = point_object("schedule_count", (0.0, 0.0, 0.0), POINT)
add_emitter(obj, 5, 10, 20, 500.0, 1.0)
case("schedule_count", obj, 21)

# Size: GLOBAL coordinates, MIX with default value 0 (size = 1 - value * factor).
obj = point_object("size_global", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 500.0, 1.0)
add_slot(settings, "size", 1.0, 0.0)
case("size_global", obj, 2)

# Size: default value 1 makes MIX a no-op (`texture_value_blend` uses def_var as `tex`).
obj = point_object("size_default_one", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 500.0, 1.0)
add_slot(settings, "size", 1.0, 1.0)
case("size_default_one", obj, 2)

# Size: OBJECT coordinates through an anchor's world_to_object, with scale and offset placement.
anchor = bpy.data.objects.new("size_anchor", None)
anchor.location = (1.0, 0.0, 0.0)
scene.collection.objects.link(anchor)
obj = point_object("size_object", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 500.0, 1.0)
add_slot(settings, "size", 1.0, 0.0, coords="OBJECT", anchor=anchor,
         scale=(2.0, 2.0, 2.0), offset=(0.1, 0.0, 0.0))
case("size_object", obj, 2)

# Life: the lifetime is scaled by the texture value.
obj = point_object("life_global", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 20.0, 1.0)
add_slot(settings, "life", 1.0, 0.0)
case("life_global", obj, 2)

# Time: births move with the texture value (factor 0.5 mixes with index/count).
obj = point_object("time_global", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 4, 1, 9, 500.0, 1.0)
add_slot(settings, "time", 0.5, 0.0)
case("time_global", obj, 10)

# Density: exist = 1 - value * factor (MIX, default 0); particles with exist < frand are removed.
obj = point_object("density_global", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 8, 1, 1, 500.0, 1.0)
add_slot(settings, "density", 1.0, 0.0)
case("density_global", obj, 2, visible=True)

frames = sorted({frame for frame, _, _, _ in reads})
for frame in range(1, max(frames) + 1):
    scene.frame_set(frame)
    for read_frame, name, obj, visible in reads:
        if read_frame != frame:
            continue
        entry = {"frame": frame, "particles": particle_values(obj)}
        if visible:
            # Blender's evaluated particle list holds the existing particles ( removed).
            entry["existing"] = len(entry["particles"])
        cases[name] = entry

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "particle_texture.blend"), compress=False)
with open(os.path.join(OUT, "blender_expected.json"), "w", encoding="utf-8") as handle:
    json.dump(cases, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("PARTICLE_TEXTURE_ORACLE_WRITTEN", OUT)
