import bpy
import json
import math
import os
import sys
from mathutils import Euler, Vector


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
    if sorted_vertices:
        bounds_min = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        bounds_max = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    else:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]
    return {
        "vertex_count": len(sorted_vertices),
        "edge_count": len(mesh.edges),
        "polygon_count": len(polygons),
        "corner_count": sum(len(face) for face, _ in polygons),
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "vertices": sorted_vertices,
        "polygons": [face for face, _ in polygons],
        "smooth_flags": [smooth for _, smooth in polygons],
    }


def create_source(name, size, scale, rotation):
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    source = bpy.context.object
    source.name = name
    for vertex in source.data.vertices:
        vertex.co.x *= size[0]
        vertex.co.y *= size[1]
        vertex.co.z *= size[2]
    source.scale = scale
    source.rotation_euler = Euler(rotation, "XYZ")
    bpy.context.view_layer.update()
    return source


def run(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_dirs = {output_dir, "/tmp/pr18o/volume_oracle"}
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    shapes = {
        "asymmetric_box": {
            "size": (1.7, 1.1, 0.8),
            "scale": (1.0, 1.0, 1.0),
            "rotation": (0.0, 0.0, 0.0),
            "voxel_size": 0.25,
            "voxel_amount": 9,
        },
        "rotated_nonuniform": {
            "size": (0.8, 1.1, 0.6),
            "scale": (1.6, 0.7, 1.3),
            "rotation": (0.31, 0.21, 0.56),
            "voxel_size": 0.22,
            "voxel_amount": 10,
        },
        "tiny_voxels": {
            "size": (0.45, 0.35, 0.5),
            "scale": (1.0, 1.0, 1.0),
            "rotation": (0.0, 0.0, 0.0),
            "voxel_size": 0.09,
            "voxel_amount": 8,
        },
    }
    thresholds = (0.1, 0.35)
    for shape_name, shape in shapes.items():
        cases = []
        for resolution_mode in ("VOXEL_SIZE", "VOXEL_AMOUNT"):
            for band_width in (0.2, 0.05):
                for density in (1.0, 0.5):
                    bpy.ops.wm.read_factory_settings(use_empty=True)
                    source = create_source(
                        "operand", shape["size"], shape["scale"], shape["rotation"]
                    )
                    if "operand_vertices" not in shape:
                        shape["operand_vertices"] = [
                            list(source.matrix_world @ vertex.co)
                            for vertex in source.data.vertices
                        ]
                        shape["operand_faces"] = [
                            list(polygon.vertices) for polygon in source.data.polygons
                        ]
                        local_min = [
                            min(vertex.co[axis] for vertex in source.data.vertices)
                            for axis in range(3)
                        ]
                        local_max = [
                            max(vertex.co[axis] for vertex in source.data.vertices)
                            for axis in range(3)
                        ]
                        transformed_min = source.matrix_world @ Vector(local_min)
                        transformed_max = source.matrix_world @ Vector(local_max)
                        shape["local_bounds_min"] = local_min
                        shape["local_bounds_max"] = local_max
                        shape["amount_diagonal"] = (transformed_max - transformed_min).length
                    volume_data = bpy.data.volumes.new("volume_data")
                    volume_object = bpy.data.objects.new("volume", volume_data)
                    bpy.context.collection.objects.link(volume_object)
                    mesh_to_volume = volume_object.modifiers.new("mesh_to_volume", "MESH_TO_VOLUME")
                    mesh_to_volume.object = source
                    mesh_to_volume.resolution_mode = resolution_mode
                    mesh_to_volume.voxel_size = shape["voxel_size"]
                    mesh_to_volume.voxel_amount = shape["voxel_amount"]
                    mesh_to_volume.interior_band_width = band_width
                    mesh_to_volume.density = density

                    carrier_data = bpy.data.meshes.new("carrier_data")
                    carrier = bpy.data.objects.new("carrier", carrier_data)
                    bpy.context.collection.objects.link(carrier)
                    for adaptivity in (0.0, 0.1, 0.5, 1.0):
                        results = []
                        for threshold in thresholds:
                            volume_to_mesh = carrier.modifiers.new("volume_to_mesh", "VOLUME_TO_MESH")
                            volume_to_mesh.object = volume_object
                            volume_to_mesh.grid_name = "density"
                            volume_to_mesh.threshold = threshold
                            volume_to_mesh.adaptivity = adaptivity
                            volume_to_mesh.use_smooth_shade = True
                            bpy.context.view_layer.update()
                            depsgraph = bpy.context.evaluated_depsgraph_get()
                            evaluated = carrier.evaluated_get(depsgraph)
                            result_mesh = evaluated.to_mesh()
                            results.append({"threshold": threshold, **canonical_mesh(result_mesh)})
                            evaluated.to_mesh_clear()
                            carrier.modifiers.remove(volume_to_mesh)
                        evaluated_volume = volume_object.evaluated_get(bpy.context.evaluated_depsgraph_get())
                        grid_names = [grid.name for grid in evaluated_volume.data.grids]
                        cases.append({
                            "resolution_mode": resolution_mode,
                            "voxel_size": shape["voxel_size"],
                            "voxel_amount": shape["voxel_amount"],
                            "interior_band_width": band_width,
                            "density": density,
                            "adaptivity": adaptivity,
                            "grid_names": grid_names,
                            "volume_to_mesh": results,
                        })
                        evaluated_volume.to_mesh_clear() if hasattr(evaluated_volume, "to_mesh_clear") else None
                    bpy.data.objects.remove(carrier, do_unlink=True)
                    bpy.data.objects.remove(volume_object, do_unlink=True)
                    bpy.data.objects.remove(source, do_unlink=True)
                    bpy.data.meshes.remove(carrier_data)
                    bpy.data.volumes.remove(volume_data)
        document = {
            "blender_version": bpy.app.version_string,
            "shape": shape_name,
            "source": shape,
            "thresholds": list(thresholds),
            "cases": cases,
        }
        for directory in output_dirs:
            with open(os.path.join(directory, f"{shape_name}.json"), "w", encoding="utf-8") as handle:
                json.dump(document, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")


if __name__ == "__main__":
    output = "/tmp/pr18o/volume_oracle"
    if "--" in sys.argv:
        output = sys.argv[sys.argv.index("--") + 1]
    run(output)
