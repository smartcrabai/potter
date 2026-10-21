"""Blender 5.2.2 oracle for particle emission, texture influences, explode, and emitter placement.

Usage (from the repo root):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tests/fixtures/particle_texture/generate.py -- <out_dir>

Writes into <out_dir>:
    particle_texture.blend   the scene Potter imports (the image is packed into it)
    blender_expected.json    Blender's per-particle values and evaluated Explode counts

Particle cases use one emitter vertex, so every particle is born at the same point. Explode cases
use a cube whose particles are emitted from its vertices through Blender's vertex distribution.
Face and volume cases emit from mesh faces with the jittered, random, and Hammersley distributions,
and record each particle's birth `location`, which is its birth point because physics is off.
Density removes non-existing particles from Blender's particle list (`free_unexisting_particles`),
so the existing count is the length of that list.
"""

import json
import os
import sys

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("particle texture oracle requires Blender 5.2.2")

OUT = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
POINT = (0.3, 0.2, 0.1)
CUBE_VERTICES = [
    (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
texture = bpy.data.textures.new("PotterBlend", type="BLEND")
texture.progression = "LINEAR"

# A 4 by 4 gradient, packed into the .blend so Potter imports the same pixels.
gradient = bpy.data.images.new("PotterGradient", width=4, height=4, alpha=False)
gradient.pixels = [
    value
    for y in range(4)
    for x in range(4)
    for value in (x / 3.0, y / 3.0, 0.5, 1.0)
]
gradient.pack()
image_texture = bpy.data.textures.new("PotterImage", type="IMAGE")
image_texture.image = gradient


def point_object(name, location, vertex=(0.0, 0.0, 0.0)):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([vertex], [], [])
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    scene.collection.objects.link(obj)
    return obj


def add_emitter(obj, count, frame_start, frame_end, lifetime, particle_size, emit_from="VERT",
                distribution="JIT", even=True, emit_random=True, userjit=0):
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
    settings.emit_from = emit_from
    settings.distribution = distribution
    settings.use_even_distribution = even
    settings.use_emit_random = emit_random
    settings.userjit = userjit
    settings.physics_type = "NO"
    settings.normal_factor = 0.0
    settings.particle_size = particle_size
    settings.size_random = 0.0
    return settings


def add_slot(settings, channel, factor, default, coords="GLOBAL", anchor=None,
             scale=(1.0, 1.0, 1.0), offset=(0.0, 0.0, 0.0), blend="MIX",
             tex=None, mapping="FLAT"):
    slot = settings.texture_slots.add()
    slot.texture = tex if tex is not None else texture
    slot.texture_coords = coords
    if anchor is not None:
        slot.object = anchor
    slot.mapping_x = "X"
    slot.mapping_y = "Y"
    slot.mapping_z = "Z"
    slot.mapping = mapping
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
         "lifetime": float(pa.lifetime), "location": [float(c) for c in pa.location]}
        for pa in particles
    ]


def explode_counts(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    counts = {"vertex_count": len(mesh.vertices), "face_count": len(mesh.polygons)}
    evaluated.to_mesh_clear()
    return counts


# Each case names its object and the frame it is read at. Blender's particle reset on a backward
# frame jump samples textures before the birth point is set, so the scene is stepped forward once.
reads = []


def case(name, obj, frame, kind="particles", visible=False):
    reads.append((frame, name, obj, kind, visible))


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

# Image: a packed gradient sampled with flat mapping, and with tube mapping.
obj = point_object("image_flat", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 500.0, 1.0)
add_slot(settings, "size", 1.0, 0.0, tex=image_texture, mapping="FLAT")
case("image_flat", obj, 2)

obj = point_object("image_tube", (0.0, 0.0, 0.0), POINT)
settings = add_emitter(obj, 1, 1, 1, 500.0, 1.0)
add_slot(settings, "size", 1.0, 0.0, tex=image_texture, mapping="TUBE")
case("image_tube", obj, 2)

# Explode with density: the cube's vertices emit eight particles, and density removes some of them.
for name, with_density in (("explode_plain", False), ("explode_density", True)):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(CUBE_VERTICES, [], CUBE_FACES)
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    settings = add_emitter(obj, 8, 1, 1, 500.0, 1.0)
    if with_density:
        add_slot(settings, "density", 1.0, 0.0)
    obj.modifiers.new("Explode", "EXPLODE")
    # Read at frame 1, where the particles are born (birth 1), so Potter's inspect of the scene matches.
    case(name, obj, 1, kind="explode")

# Face and volume emitters sit at the origin with identity transforms. Each case is read at frame 2,
# after the one-frame emission at frame 1, so every particle is alive and its location is its birth.
def emitter_object(name, vertices, faces, smooth=False):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    return obj


SQUARE_VERTICES = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
# A triangle of area 1 and a quad of area 4, so area weights decide which element each particle takes.
MIXED_VERTICES = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (4.0, 0.0, 0.0),
                  (4.0, 2.0, 0.0), (2.0, 2.0, 0.0)]
MIXED_FACES = [(0, 1, 2), (1, 3, 4, 5)]
# A tetrahedron, closed, so a volume grid keeps only the cells inside it.
TETRA_VERTICES = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)]
TETRA_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
# A concave pentagon and a concave hexagon: their triangulation decides the face each particle takes.
PENTAGON_VERTICES = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 3.0, 0.0), (1.5, 1.0, 0.0), (0.0, 3.0, 0.0)]
HEXAGON_VERTICES = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (2.0, 1.0, 0.0),
                    (0.0, 4.0, 0.0), (-1.0, 2.0, 0.0)]

# (name, vertices, faces, count, emit_from, options): options set the distribution fields.
EMITTER_CASES = [
    ("face_quad_jit", SQUARE_VERTICES, [(0, 1, 2, 3)], 8, "FACE", {}),
    ("face_quad_rand", SQUARE_VERTICES, [(0, 1, 2, 3)], 8, "FACE", {"distribution": "RAND"}),
    ("face_quad_hammersley", SQUARE_VERTICES, [(0, 1, 2, 3)], 30, "FACE", {"userjit": 30}),
    ("face_quad_userjit_norandom", SQUARE_VERTICES, [(0, 1, 2, 3)], 12, "FACE",
     {"userjit": 5, "emit_random": False}),
    ("face_mixed_jit", MIXED_VERTICES, MIXED_FACES, 20, "FACE", {}),
    ("face_mixed_rand_uneven", MIXED_VERTICES, MIXED_FACES, 13, "FACE",
     {"distribution": "RAND", "even": False}),
    ("face_mixed_norandom", MIXED_VERTICES, MIXED_FACES, 13, "FACE", {"emit_random": False}),
    ("face_cube_jit", CUBE_VERTICES, CUBE_FACES, 60, "FACE", {}),
    ("face_cube_smooth_rand", CUBE_VERTICES, CUBE_FACES, 25, "FACE",
     {"distribution": "RAND", "smooth": True}),
    ("volume_cube_jit", CUBE_VERTICES, CUBE_FACES, 40, "VOLUME", {}),
    ("volume_cube_rand", CUBE_VERTICES, CUBE_FACES, 40, "VOLUME", {"distribution": "RAND"}),
    ("volume_cube_smooth", CUBE_VERTICES, CUBE_FACES, 40, "VOLUME", {"smooth": True}),
    # Grid cases: a grid emitter holds `grid` cubed cells, so its count is unused.
    ("grid_face_cube", CUBE_VERTICES, CUBE_FACES, 1, "FACE", {"grid": 4}),
    ("grid_face_cube_options", CUBE_VERTICES, CUBE_FACES, 1, "FACE",
     {"grid": 6, "hexagonal": True, "invert": True, "grid_random": 0.3}),
    ("grid_face_mixed", MIXED_VERTICES, MIXED_FACES, 1, "FACE", {"grid": 3}),
    ("grid_volume_tetra", TETRA_VERTICES, TETRA_FACES, 1, "VOLUME", {"grid": 4}),
    ("grid_volume_tetra_options", TETRA_VERTICES, TETRA_FACES, 1, "VOLUME",
     {"grid": 6, "hexagonal": True, "grid_random": 0.6}),
    # N-gon cases: the faces are triangulated the way Blender's tessellation does.
    ("ngon_face_pentagon_jit", PENTAGON_VERTICES, [(0, 1, 2, 3, 4)], 20, "FACE", {}),
    ("ngon_face_hexagon_rand_uneven", HEXAGON_VERTICES, [(0, 1, 2, 3, 4, 5)], 25, "FACE",
     {"distribution": "RAND", "even": False}),
]
for name, vertices, faces, count, emit_from, options in EMITTER_CASES:
    obj = emitter_object(name, vertices, faces, smooth=options.get("smooth", False))
    settings = add_emitter(obj, count, 1, 1, 500.0, 1.0, emit_from=emit_from,
                           distribution=options.get("distribution", "JIT"),
                           even=options.get("even", True),
                           emit_random=options.get("emit_random", True),
                           userjit=options.get("userjit", 0))
    if "grid" in options:
        settings.distribution = "GRID"
        settings.grid_resolution = options["grid"]
        settings.hexagonal_grid = options.get("hexagonal", False)
        settings.invert_grid = options.get("invert", False)
        settings.grid_random = options.get("grid_random", 0.0)
    case(name, obj, 2)

frames = sorted({frame for frame, _, _, _, _ in reads})
cases = {}
for frame in range(1, max(frames) + 1):
    scene.frame_set(frame)
    for read_frame, name, obj, kind, visible in reads:
        if read_frame != frame:
            continue
        if kind == "explode":
            cases[name] = {"frame": frame, "explode": explode_counts(obj)}
            continue
        entry = {"frame": frame, "particles": particle_values(obj)}
        if visible:
            # Blender's evaluated particle list holds the existing particles only.
            entry["existing"] = len(entry["particles"])
        cases[name] = entry

os.makedirs(OUT, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "particle_texture.blend"), compress=False)
with open(os.path.join(OUT, "blender_expected.json"), "w", encoding="utf-8") as handle:
    json.dump(cases, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("PARTICLE_TEXTURE_ORACLE_WRITTEN", OUT)
