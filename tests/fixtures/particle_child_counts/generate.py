"""Blender 5.2.2 oracle for the particle child-count recorded gap.

Rebuilds the scene of `indexed_particles_lifecycle_children_and_bake_are_evaluated` in
tests/modifiers_simulation.rs and writes each instancer's evaluated face count at frames 1 and 3,
plus the child and particle counts of the emitter. The test reads the committed output
(blender_expected.json), so the recorded gap values are reproducible from the repository.

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
      --disable-autoexec --python tests/fixtures/particle_child_counts/generate.py -- \
      tests/fixtures/particle_child_counts/blender_expected.json
"""

import json
import sys

import bpy

if tuple(bpy.app.version[:3]) != (5, 2, 2):
    raise RuntimeError("the child-count oracle requires Blender 5.2.2")

out_path = sys.argv[sys.argv.index("--") + 1]
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 3


def plane(name):
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    return obj


emitter = plane("a_emitter")
settings = bpy.data.particles.new("particles")
settings.emit_from = "VERT"
settings.count = 2
settings.frame_start = 1.0
settings.frame_end = 2.0
settings.lifetime = 1.0
settings.physics_type = "NO"
settings.normal_factor = 0.0
settings.particle_size = 0.75
settings.child_radius = 0.6
settings.child_type = "SIMPLE"
settings.child_percent = 2
settings.rendered_child_count = 2
first = emitter.modifiers.new("particles", "PARTICLE_SYSTEM")
first.particle_system.settings = settings
first.particle_system.child_seed = 31
first.particle_system.seed = 11

second_settings = bpy.data.particles.new("particles_second")
second = emitter.modifiers.new("particles_second", "PARTICLE_SYSTEM")
second.particle_system.settings = second_settings


def instancer(name, system_index, use_children, use_size, show_alive, show_dead, show_unborn):
    obj = plane(name)
    modifier = obj.modifiers.new("instances", "PARTICLE_INSTANCE")
    modifier.object = emitter
    modifier.particle_system_index = system_index
    modifier.use_normal = True
    modifier.use_children = use_children
    modifier.use_size = use_size
    modifier.show_alive = show_alive
    modifier.show_dead = show_dead
    modifier.show_unborn = show_unborn
    modifier.position = 0.0
    modifier.random_position = 0.0
    modifier.axis = "X"
    modifier.space = "WORLD"
    modifier.use_path = False
    return obj


# The same objects and flags as the Potter scene; the object name is the Potter node id.
objects = {
    "b_alive": instancer("b_alive", 1, True, False, True, False, False),
    "c_secondary": instancer("c_secondary", 2, False, False, True, False, False),
    "d_no_unborn": instancer("d_no_unborn", 1, False, False, True, False, False),
    "g_with_unborn": instancer("g_with_unborn", 1, False, False, True, False, True),
    "e_dead": instancer("e_dead", 1, True, True, False, True, False),
}

result = {"blender_version": bpy.app.version_string, "faces": {}, "emitter": {}}
for frame in (1, 3):
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result["faces"][str(frame)] = {
        name: len(obj.evaluated_get(depsgraph).data.polygons) for name, obj in objects.items()
    }
    evaluated = emitter.evaluated_get(depsgraph)
    result["emitter"][str(frame)] = [
        {
            "totpart": len(system.particles),
            "totchild": len(system.child_particles),
            "alive": sorted({particle.alive_state for particle in system.particles}),
        }
        for system in evaluated.particle_systems
    ]

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("CHILD_COUNT_ORACLE_WRITTEN", out_path)
