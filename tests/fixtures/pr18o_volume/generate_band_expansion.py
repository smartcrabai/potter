"""Blender 5.2.2 oracle for Mesh to Volume with an interior band wider than two voxels.

OpenVDB's meshToVolume expands the narrow band only when a band is wider than two voxels
(`minBandWidth = 2 * voxelSize`, MeshToVolume.h `meshToVolume`). Blender's fog conversion passes
an interior band of max(1, band / voxel) voxels, so a 0.8 band at voxel 0.25 (3.2 voxels) takes
the expansion path, while the 0.2 band (1 voxel) does not.

The operand is the cube of `volume_displace_image.json` (size 2 at x = 1), with the volume at
(2, 1, 0), so the operand sits at (-1, -1, 0) in volume-local space. Each case records the vertex
and polygon counts of Blender's Volume to Mesh at the listed thresholds.

Run with: blender -b --factory-startup --python generate_band_expansion.py
"""

import json
import os

import bpy

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT = os.path.join(FIXTURE_DIR, "band_expansion.json")
VOXEL_SIZE = 0.25
THRESHOLDS = (0.1, 0.35)


def build_case(interior_band_width):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 0.0, 0.0))
    source = bpy.context.object
    volume_data = bpy.data.volumes.new("volume_data")
    volume = bpy.data.objects.new("volume", volume_data)
    bpy.context.collection.objects.link(volume)
    volume.location = (2.0, 1.0, 0.0)
    mesh_to_volume = volume.modifiers.new("mesh_to_volume", "MESH_TO_VOLUME")
    mesh_to_volume.object = source
    mesh_to_volume.resolution_mode = "VOXEL_SIZE"
    mesh_to_volume.voxel_size = VOXEL_SIZE
    mesh_to_volume.interior_band_width = interior_band_width
    mesh_to_volume.density = 1.0
    bpy.context.view_layer.update()

    counts = {}
    for threshold in THRESHOLDS:
        carrier_data = bpy.data.meshes.new("carrier_data")
        carrier = bpy.data.objects.new("carrier", carrier_data)
        bpy.context.collection.objects.link(carrier)
        volume_to_mesh = carrier.modifiers.new("volume_to_mesh", "VOLUME_TO_MESH")
        volume_to_mesh.object = volume
        volume_to_mesh.grid_name = "density"
        volume_to_mesh.threshold = threshold
        volume_to_mesh.adaptivity = 0.0
        bpy.context.view_layer.update()
        evaluated = carrier.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        counts[str(threshold)] = {
            "vertex_count": len(mesh.vertices),
            "polygon_count": len(mesh.polygons),
        }
        evaluated.to_mesh_clear()
        bpy.data.objects.remove(carrier, do_unlink=True)
    return counts


def main():
    document = {
        "blender_version": bpy.app.version_string,
        "voxel_size": VOXEL_SIZE,
        "operand_local_center": [-1.0, -1.0, 0.0],
        "operand_size": 2.0,
        "thresholds": list(THRESHOLDS),
        "narrow_band": {
            "interior_band_width": 0.2,
            "expanded": False,
            "counts": build_case(0.2),
        },
        "expanded_band": {
            "interior_band_width": 0.8,
            "expanded": True,
            "counts": build_case(0.8),
        },
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    print("WROTE", OUTPUT)


main()
