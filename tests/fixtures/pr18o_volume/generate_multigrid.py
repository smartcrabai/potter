import bpy
import hashlib
import json
import os
import sys
from mathutils import Vector


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
    bounds_min = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    bounds_max = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
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


def make_multigrid_vdb(path):
    import openvdb

    density = openvdb.FloatGrid(0.0)
    density.name = "density"
    density.gridClass = openvdb.GridClass.FOG_VOLUME
    density.transform = openvdb.createLinearTransform(0.25)
    density_accessor = density.getAccessor()

    velocity_background = (0.125, 0.25, 0.375)
    velocity = openvdb.Vec3SGrid(velocity_background)
    velocity.name = "velocity"
    velocity.transform = openvdb.createLinearTransform(0.25)
    velocity_accessor = velocity.getAccessor()

    for z in range(8):
        for y in range(8):
            for x in range(8):
                if 1 <= x <= 6 and 1 <= y <= 6 and 1 <= z <= 6:
                    density_accessor.setValueOn((x, y, z), 1.0)
                    velocity_accessor.setValueOn((x, y, z), (x / 7.0, y / 7.0, z / 7.0))
    # OpenVDB orders grids by name; evaluators resolve the selected grid explicitly.
    openvdb.write(path, [velocity, density])
    with open(path, "rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read()).hexdigest()

def make_precision_vdb(path):
    import openvdb

    precision = openvdb.DoubleGrid(0.0)
    precision.name = "precision"
    precision.transform = openvdb.createLinearTransform(0.25)
    accessor = precision.getAccessor()
    for z in range(8):
        for y in range(8):
            for x in range(8):
                if 1 <= x <= 6 and 1 <= y <= 6 and 1 <= z <= 6:
                    accessor.setValueOn((x, y, z), 0.5 + (x - 3.5) * 1.0e-8)
    openvdb.write(path, precision)
    with open(path, "rb") as handle:
        return "sha256:" + hashlib.sha256(handle.read()).hexdigest()


def run(output_dirs):
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    fixture_dir = os.path.dirname(os.path.realpath(__file__))
    vdb_path = os.path.join(fixture_dir, "multigrid.vdb")
    output_dirs.add(fixture_dir)
    vdb_hash = make_multigrid_vdb(vdb_path)
    precision_vdb_path = os.path.join(fixture_dir, "precision.vdb")
    precision_vdb_hash = make_precision_vdb(precision_vdb_path)

    velocity_background = (0.125, 0.25, 0.375)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.volume_import(filepath=vdb_path)
    volume = bpy.context.object
    volume.name = "volume"
    volume.location = (0.0, 0.0, 0.0)
    bpy.ops.object.volume_import(filepath=precision_vdb_path)
    precision_volume = bpy.context.active_object
    precision_volume.name = "precision_volume"
    precision_carrier = bpy.data.objects.new(
        "precision_carrier", bpy.data.meshes.new("precision_carrier_data")
    )
    bpy.context.collection.objects.link(precision_carrier)
    precision_to_mesh = precision_carrier.modifiers.new("precision_to_mesh", "VOLUME_TO_MESH")
    precision_to_mesh.object = precision_volume
    precision_to_mesh.grid_name = "precision"
    precision_to_mesh.threshold = 0.5
    precision_to_mesh.adaptivity = 0.0
    texture_data = {
        "width": 16,
        "height": 16,
        "pixels": [
            [x / 15.0, y / 15.0, 0.5, 1.0]
            for y in range(16)
            for x in range(16)
        ],
    }
    image = bpy.data.images.new(
        "texture", width=texture_data["width"], height=texture_data["height"], alpha=True, float_buffer=True
    )
    image.pixels = [component for pixel in texture_data["pixels"] for component in pixel]
    image.update()
    texture = bpy.data.textures.new("texture_image", "IMAGE")
    texture.image = image
    texture.extension = "REPEAT"
    texture.use_interpolation = True
    probe_index = (3, 3, 3)
    texture_probe = list(texture.evaluate(Vector((0.75, 0.75, 0.75)))[:3])
    index_displacement = [
        (component - 0.5) * (0.5 / 0.25) for component in texture_probe
    ]
    velocity_sample_index = [
        probe_index[axis] - index_displacement[axis] for axis in range(3)
    ]
    expected_velocity_sample = [coordinate / 7.0 for coordinate in velocity_sample_index]

    displace = volume.modifiers.new("volume_displace", "VOLUME_DISPLACE")
    displace.texture = texture
    displace.texture_map_mode = "LOCAL"
    displace.strength = 0.5
    displace.texture_sample_radius = 1.0
    displace.texture_mid_level = (0.5, 0.5, 0.5)

    carrier_data = bpy.data.meshes.new("carrier_data")
    carrier = bpy.data.objects.new("carrier", carrier_data)
    bpy.context.collection.objects.link(carrier)
    volume_to_mesh = carrier.modifiers.new("volume_to_mesh", "VOLUME_TO_MESH")
    volume_to_mesh.object = volume
    volume_to_mesh.grid_name = "density"
    volume_to_mesh.threshold = 0.5
    volume_to_mesh.adaptivity = 0.0
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_volume = volume.evaluated_get(depsgraph)
    output_grid_metadata = [
        {"name": grid.name, "data_type": grid.data_type, "channels": grid.channels}
        for grid in evaluated_volume.data.grids
    ]
    evaluated_carrier = carrier.evaluated_get(depsgraph)
    mesh = evaluated_carrier.to_mesh()
    oracle_mesh = canonical_mesh(mesh)
    evaluated_precision_carrier = precision_carrier.evaluated_get(depsgraph)
    precision_mesh = evaluated_precision_carrier.to_mesh()
    oracle_precision_mesh = canonical_mesh(precision_mesh)
    evaluated_precision_carrier.to_mesh_clear()

    document = {
        "blender_version": bpy.app.version_string,
        "vdb_sha256": vdb_hash,
        "precision_vdb_sha256": precision_vdb_hash,
        "voxel_size": 0.25,
        "source_grids": [
            {"name": "density", "value_type": "Tree_float_5_4_3", "grid_class": "FOG_VOLUME", "stored_class": "fog volume"},
            {"name": "velocity", "value_type": "Tree_vec3s_5_4_3", "background": list(velocity_background)},
        ],
        "precision_source_grid": {"name": "precision", "value_type": "Tree_double_5_4_3"},
        "evaluated_grids": output_grid_metadata,
        "texture": texture_data,
        "volume_displace": {
            "texture_map_mode": "LOCAL",
            "strength": 0.5,
            "texture_sample_radius": 1.0,
            "texture_mid_level": [0.5, 0.5, 0.5],
        },
        "velocity_probe": {
            "index": list(probe_index),
            "texture_value": texture_probe,
            "displacement_index": index_displacement,
            "expected_sample": expected_velocity_sample,
        },
        "volume_to_mesh": {
            "grid_name": "density",
            "threshold": 0.5,
            "adaptivity": 0.0,
            **oracle_mesh,
        },
        "precision_volume_to_mesh": {
            "grid_name": "precision",
            "threshold": 0.5,
            "adaptivity": 0.0,
            **oracle_precision_mesh,
        },
    }
    for directory in output_dirs:
        with open(os.path.join(directory, "volume_multigrid_vdb.json"), "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        if directory != fixture_dir:
            with open(vdb_path, "rb") as source, open(os.path.join(directory, "multigrid.vdb"), "wb") as destination:
                destination.write(source.read())
            with open(precision_vdb_path, "rb") as source, open(os.path.join(directory, "precision.vdb"), "wb") as destination:
                destination.write(source.read())


if __name__ == "__main__":
    output = "/tmp/pr18o/volume_oracle"
    if "--" in sys.argv:
        output = sys.argv[sys.argv.index("--") + 1]
    fixture_dir = os.path.dirname(os.path.realpath(__file__))
    run({os.path.realpath(output), fixture_dir})
