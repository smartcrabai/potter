"""Blender 5.2.2 oracle for the voxel remesh and the sharp-angle operators that tests/mesh_edit_more.rs pins.
Run headless and write the JSON next to this file:

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --disable-autoexec \
        --python-exit-code 3 --python tests/fixtures/remesh-reproject/generate.py -- \
        tests/fixtures/remesh-reproject/expected.json

Every value comes from a Blender operator or RNA call on the default cube (2 m, 8 vertices, 6 quads):
`object.voxel_remesh` with the default remesh flags, `mesh.shade_smooth`/`use_smooth`, `Mesh.set_sharp_from_angle`
and `mesh.set_sharpness_by_angle`. Nothing is computed by Potter.
"""

import json
import math
import sys

import bpy

REMESH_VOXEL_SIZE = 0.5


def fresh_cube():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    return bpy.context.object


def sharp_face_state(mesh):
    """Blender's `sharp_face` attribute: whether it exists, and its per-face values when it does."""
    if "sharp_face" not in mesh.attributes:
        return {"sharp_face_present": False}
    return {
        "sharp_face_present": True,
        "sharp_face": [bool(attribute.value) for attribute in mesh.attributes["sharp_face"].data],
    }


def pattern(flags):
    return "".join("1" if flag else "0" for flag in flags)


def voxel_remesh_case(make_shading):
    obj = fresh_cube()
    make_shading(obj.data)
    obj.data.remesh_voxel_size = REMESH_VOXEL_SIZE
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.voxel_remesh()
    mesh = obj.data
    flat = [not polygon.use_smooth for polygon in mesh.polygons]
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.polygons),
        "flat_pattern": pattern(flat),
        "all_smooth": not any(flat),
        # Storage order, which `mesh_apply_spatial_organization` sets: the material of each face in order, and
        # the first vertex positions, rounded to 6 digits.
        "material_indices": [polygon.material_index for polygon in mesh.polygons],
        "first_vertices": [[round(value, 6) for value in vertex.co] for vertex in mesh.vertices[:8]],
        **sharp_face_state(mesh),
    }


def two_materials(mesh):
    # Two material slots, faces 3..5 in the second: the spatial order splits leaves by material.
    mesh.materials.append(bpy.data.materials.new("MaterialA"))
    mesh.materials.append(bpy.data.materials.new("MaterialB"))
    for polygon in mesh.polygons:
        polygon.material_index = 1 if polygon.index >= 3 else 0
        polygon.use_smooth = True


def leave_default(mesh):
    # The default cube is flat: Blender's primitive has no `shade_smooth`, so every face stays flat.
    return None


def mixed(mesh):
    # Face 4 flat, the rest smooth: the probe the Rust test pins.
    for polygon in mesh.polygons:
        polygon.use_smooth = polygon.index != 4


def all_smooth(mesh):
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    # `mesh.shade_smooth()` removes Blender's `sharp_face` attribute (`mesh_smooth_set(true)`).
    bpy.context.view_layer.objects.active = bpy.context.object
    bpy.ops.object.shade_smooth()


def sharp_from_angle_case(angle_degrees):
    """`Mesh.set_sharp_from_angle` on the cube with face 4 flat, from a mixed start."""
    obj = fresh_cube()
    for polygon in obj.data.polygons:
        polygon.use_smooth = polygon.index != 4
    obj.data.set_sharp_from_angle(angle=math.radians(angle_degrees))
    mesh = obj.data
    return {
        "smooth": sum(1 for polygon in mesh.polygons if polygon.use_smooth),
        "faces": len(mesh.polygons),
        "sharp_edges": sum(1 for edge in mesh.edges if edge.use_edge_sharp),
        "edges": len(mesh.edges),
        **sharp_face_state(mesh),
    }


def sharpness_by_angle_case(angle_degrees):
    """`mesh.set_sharpness_by_angle` in edit mode with every edge selected (`extend` off)."""

    obj = fresh_cube()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.set_sharpness_by_angle(angle=math.radians(angle_degrees))
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    return {
        "sharp_edges": sum(1 for edge in mesh.edges if edge.use_edge_sharp),
        "edges": len(mesh.edges),
        "smooth": sum(1 for polygon in mesh.polygons if polygon.use_smooth),
        "faces": len(mesh.polygons),
    }


def main():
    out_path = sys.argv[sys.argv.index("--") + 1]
    result = {
        "blender_version": bpy.app.version_string,
        "remesh_voxel_size": REMESH_VOXEL_SIZE,
        "remesh": {
            "flat_cube": voxel_remesh_case(leave_default),
            "mixed_cube": voxel_remesh_case(mixed),
            "all_smooth_cube": voxel_remesh_case(all_smooth),
            "two_material_cube": voxel_remesh_case(two_materials),
        },
        "sharp_from_angle": {
            str(angle): sharp_from_angle_case(angle) for angle in (30, 0, 180, 60)
        },
        "sharpness_by_angle": {
            str(angle): sharpness_by_angle_case(angle) for angle in (30, 120)
        },
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("REMESH_REPROJECT_ORACLE=" + json.dumps({"version": bpy.app.version_string}))


main()
