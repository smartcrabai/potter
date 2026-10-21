"""Blender 5.2.2 oracle: a free soft-body UV sphere falling onto a COLLISION plane.

The scene matches `tests/physics.rs::soft_body_sphere_stops_on_collider_without_penetrating`:
an 8x6 UV sphere of radius 1 centred at z = 1.5 and a 20 x 20 collider plane at z = -0.5. Both
use Blender's default soft-body and collision settings (the sphere uses no goal). Positions are
world space (the object's matrix applied to the evaluated vertices), so the plane is at z = -0.5.

Run with: blender -b --factory-startup --python generate_softbody_oracle.py -- [fps] [output.json]

The default is 24 fps, the scene the Potter test replays. A second run at 48 fps measures how
much Blender's own integration changes the trajectory: that difference is the tolerance the test
derives (see `tests/physics.rs`).
"""

import json
import os
import sys

import bpy

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
FPS = int(ARGS[0]) if ARGS else 24
OUTPUT = ARGS[1] if len(ARGS) > 1 else os.path.join(FIXTURE_DIR, "blender_sphere_collision.json")
FRAMES = 60 * FPS // 24
PLANE_Z = -0.5
SPHERE_Z = 1.5


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = FRAMES
    scene.gravity = (0.0, 0.0, -9.81)

    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0.0, 0.0, PLANE_Z))
    plane = bpy.context.object
    plane.name = "collider"
    plane.modifiers.new("collision", "COLLISION")

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=8, ring_count=6, radius=1.0, location=(0.0, 0.0, SPHERE_Z)
    )
    sphere = bpy.context.object
    sphere.name = "ball"
    soft = sphere.modifiers.new("softbody", "SOFT_BODY")
    soft.settings.use_goal = False
    return scene, plane, sphere, soft


def main():
    scene, plane, sphere, soft = build_scene()
    rows = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = sphere.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        zs = [(evaluated.matrix_world @ vertex.co).z for vertex in mesh.vertices]
        rows.append(
            {
                "frame": frame,
                "center_z": sum(zs) / len(zs),
                "min_z": min(zs),
                "max_z": max(zs),
            }
        )
        evaluated.to_mesh_clear()

    settings = soft.settings
    document = {
        "blender_version": bpy.app.version_string,
        "plane_z": PLANE_Z,
        "sphere_center_z": SPHERE_Z,
        "fps": scene.render.fps / scene.render.fps_base,
        "soft_body": {
            "friction": settings.friction,
            "damping": settings.damping,
            "mass": settings.mass,
            "use_goal": settings.use_goal,
            "use_edges": settings.use_edges,
            "pull": settings.pull,
            "push": settings.push,
            "bend": settings.bend,
            "shear": settings.shear,
            "speed": settings.speed,
        },
        "collision": {
            "thickness_outer": plane.collision.thickness_outer,
            "thickness_inner": plane.collision.thickness_inner,
            "damping": plane.collision.damping,
            "friction": plane.collision.cloth_friction,
            "use_culling": plane.collision.use_culling,
            "use_normal": plane.collision.use_normal,
        },
        "rows": rows,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    print("WROTE", OUTPUT)


main()
