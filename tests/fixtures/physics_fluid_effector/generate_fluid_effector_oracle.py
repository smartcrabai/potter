"""Blender 5.2.2 oracle: liquid spawned inside a fluid effector.

A liquid domain (4 units, resolution 16) contains an INFLOW liquid flow (1 unit cube at z = 1.5, so
its bottom face at z = 1.0 overlaps the effector) and an optional effector (0.3 unit cube at z =
0.95, spanning z 0.8 to 1.1). Each frame records the liquid mesh vertices in the effector's column
(|x|, |y| < 0.16): their count, their lowest and highest z, and how many lie below the effector's
bottom face. The run is repeated without the effector as the control.

Run with: blender -b --factory-startup --python generate_fluid_effector_oracle.py
"""

import json
import os

import bpy

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT = os.path.join(FIXTURE_DIR, "blender_fluid_effector.json")
FRAMES = 16
COLUMN_HALF_WIDTH = 0.16
EFFECTOR_BOTTOM = 0.8


def cube(name, size, location):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj


def build_scene(with_effector):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAMES

    domain = cube("domain", 4.0, (0.0, 0.0, 0.0))
    domain_mod = domain.modifiers.new("Fluid", "FLUID")
    domain_mod.fluid_type = "DOMAIN"
    domain_mod.domain_settings.domain_type = "LIQUID"
    domain_mod.domain_settings.resolution_max = 16

    flow = cube("flow", 1.0, (0.0, 0.0, 1.5))
    flow_mod = flow.modifiers.new("Fluid", "FLUID")
    flow_mod.fluid_type = "FLOW"
    flow_mod.flow_settings.flow_type = "LIQUID"
    flow_mod.flow_settings.flow_behavior = "INFLOW"
    flow_mod.flow_settings.flow_source = "MESH"

    if with_effector:
        effector = cube("effector", 0.3, (0.0, 0.0, 0.95))
        effector.modifiers.new("Fluid", "FLUID").fluid_type = "EFFECTOR"
    return scene, domain


def run(with_effector):
    scene, domain = build_scene(with_effector)
    rows = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated = domain.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        points = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        column = [p for p in points if abs(p.x) < COLUMN_HALF_WIDTH and abs(p.y) < COLUMN_HALF_WIDTH]
        zs = sorted(p.z for p in column)
        rows.append(
            {
                "frame": frame,
                "vertices": len(points),
                "column_vertices": len(column),
                "column_z_min": zs[0] if zs else None,
                "column_z_max": zs[-1] if zs else None,
                "column_below_effector_bottom": sum(1 for z in zs if z < EFFECTOR_BOTTOM),
            }
        )
        evaluated.to_mesh_clear()
    return {"blender_version": bpy.app.version_string, "with_effector": with_effector, "rows": rows}


def main():
    document = {"runs": [run(True), run(False)]}
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    print("WROTE", OUTPUT)


main()
