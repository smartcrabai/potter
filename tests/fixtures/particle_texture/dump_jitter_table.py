"""Blender 5.2.2 jitter tables for face particle emission (`particle_distribute.cc`).

Usage (from the repo root):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tests/fixtures/particle_texture/dump_jitter_table.py

Each table is read back from particle birth locations. A unit-square quad, with `userjit` equal to
the particle count and no random placement, puts particle `p` on jitter pair `p`, and its location
is that pair. Prints the pairs that `sim/particle_jitter.rs` tests against.
"""

import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError("jitter table dump requires Blender 5.2.2")

SQUARE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]


def dump(count, jitter_factor):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new("square")
    mesh.from_pydata(SQUARE, [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("square", mesh)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.particle_system_add()
    system = obj.particle_systems[-1]
    system.seed = 3
    settings = system.settings
    settings.count = count
    settings.frame_start = 1
    settings.frame_end = 1
    settings.lifetime = 500.0
    settings.lifetime_random = 0.0
    settings.emit_from = "FACE"
    settings.distribution = "JIT"
    settings.userjit = count
    settings.jitter_factor = jitter_factor
    settings.use_emit_random = False
    settings.use_even_distribution = True
    settings.physics_type = "NO"
    settings.normal_factor = 0.0
    settings.particle_size = 1.0
    settings.size_random = 0.0
    scene.frame_set(2)
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [
        (pa.location[0], pa.location[1])
        for pa in evaluated.particle_systems[0].particles
    ]


for count, jitter_factor in ((3, 1.0), (7, 0.5), (30, 1.0)):
    pairs = dump(count, jitter_factor)
    text = ", ".join(f"[{x!r}, {y!r}]" for x, y in pairs)
    print(f"JITTER_TABLE {count} {jitter_factor} {text}")
