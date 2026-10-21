"""Blender 5.2.2 oracle for Volume to Mesh resolution modes (VOXEL_SIZE and VOXEL_AMOUNT).

The operand is a cube of size 2 at x = 1, converted by Mesh to Volume at voxel size 0.25 with an
interior band of 0.25 and density 1. Each case re-extracts that fog with one resolution mode and
records the canonical mesh of Blender's Volume to Mesh at each threshold, with smooth shading on.

Blender resamples the grid with OpenVDB's GridTransformer (BoxSampler) before meshing; the cases
cover factors above one (finer), at one half (no mipmap), and below one half (mipmap halvings with a
remainder), and VOXEL_AMOUNT, whose extent comes from the active voxel bounds of the grid.

Run with: blender -b --factory-startup --python generate_volume_to_mesh_resolution.py
"""

import json
import os

import bpy

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT = os.path.join(FIXTURE_DIR, "volume_to_mesh_resolution.json")
VOXEL_SIZE = 0.25
BAND = 0.25
THRESHOLDS = (0.1, 0.35)
RESOLUTIONS = (
    ("VOXEL_SIZE", 0.2, 32),
    ("VOXEL_SIZE", 0.125, 32),
    ("VOXEL_SIZE", 0.5, 32),
    ("VOXEL_SIZE", 0.7, 32),
    ("VOXEL_SIZE", 1.0, 32),
    ("VOXEL_AMOUNT", 0.25, 8),
    ("VOXEL_AMOUNT", 0.25, 16),
    ("VOXEL_AMOUNT", 0.25, 32),
)


def canonical_mesh(mesh):
    vertices = [tuple(float(component) for component in vertex.co) for vertex in mesh.vertices]
    ordered = sorted(enumerate(vertices), key=lambda item: item[1])
    remap = {old: new for new, (old, _) in enumerate(ordered)}
    sorted_vertices = [list(co) for _, co in ordered]
    polygons = []
    for polygon in mesh.polygons:
        face = [remap[index] for index in polygon.vertices]
        rotations = [face[offset:] + face[:offset] for offset in range(len(face))]
        polygons.append((min(rotations), bool(polygon.use_smooth)))
    polygons.sort()
    return {
        "vertex_count": len(sorted_vertices),
        "edge_count": len(mesh.edges),
        "polygon_count": len(polygons),
        "corner_count": sum(len(face) for face, _ in polygons),
        "vertices": sorted_vertices,
        "polygons": [face for face, _ in polygons],
        "smooth_flags": [smooth for _, smooth in polygons],
    }


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(1.0, 0.0, 0.0))
    source = bpy.context.object
    source_mesh = {
        "vertices": [[float(c) for c in vertex.co] for vertex in source.data.vertices],
        "faces": [list(polygon.vertices) for polygon in source.data.polygons],
    }
    volume_data = bpy.data.volumes.new("volume_data")
    volume = bpy.data.objects.new("volume", volume_data)
    bpy.context.collection.objects.link(volume)
    mesh_to_volume = volume.modifiers.new("mesh_to_volume", "MESH_TO_VOLUME")
    mesh_to_volume.object = source
    mesh_to_volume.resolution_mode = "VOXEL_SIZE"
    mesh_to_volume.voxel_size = VOXEL_SIZE
    mesh_to_volume.interior_band_width = BAND
    mesh_to_volume.density = 1.0
    bpy.context.view_layer.update()

    cases = []
    for resolution_mode, voxel_size, voxel_amount in RESOLUTIONS:
        results = []
        for threshold in THRESHOLDS:
            carrier_data = bpy.data.meshes.new("carrier_data")
            carrier = bpy.data.objects.new("carrier", carrier_data)
            bpy.context.collection.objects.link(carrier)
            volume_to_mesh = carrier.modifiers.new("volume_to_mesh", "VOLUME_TO_MESH")
            volume_to_mesh.object = volume
            volume_to_mesh.grid_name = "density"
            volume_to_mesh.threshold = threshold
            volume_to_mesh.adaptivity = 0.0
            volume_to_mesh.use_smooth_shade = True
            volume_to_mesh.resolution_mode = resolution_mode
            volume_to_mesh.voxel_size = voxel_size
            volume_to_mesh.voxel_amount = voxel_amount
            bpy.context.view_layer.update()
            evaluated = carrier.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            results.append({"threshold": threshold, **canonical_mesh(mesh)})
            evaluated.to_mesh_clear()
            bpy.data.objects.remove(carrier, do_unlink=True)
            bpy.data.meshes.remove(carrier_data)
        cases.append({
            "resolution_mode": resolution_mode,
            "voxel_size": voxel_size,
            "voxel_amount": voxel_amount,
            "volume_to_mesh": results,
        })
    document = {
        "blender_version": bpy.app.version_string,
        "source_voxel_size": VOXEL_SIZE,
        "interior_band_width": BAND,
        "density": 1.0,
        "source": source_mesh,
        "thresholds": list(THRESHOLDS),
        "cases": cases,
    }
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")
    print("WROTE", OUTPUT)


main()
