"""Blender 5.2.2 / OpenVDB 13 oracle for typed grids and nonlinear transforms."""

import json
import os
import sys

import bpy
import openvdb


EXPECTED_BLENDER = (5, 2, 2)
EXPECTED_OPENVDB = (13, 0, 0)
EXPECTED_FILE_FORMAT = 225
# Deterministic archive UUIDs keep captures byte-stable without colliding with each other.
FILE_UUIDS = {
    "value_types.vdb": b"00000000-0000-0000-0000-000000000001",
    "frustum.vdb": b"00000000-0000-0000-0000-000000000002",
    "density.vdb": b"00000000-0000-0000-0000-000000000003",
}
FILE_UUID_OFFSET = 21  # The file-format 225 fixed header precedes the UUID text.


def write_vdb(path, grids):
    fixture_uuid = FILE_UUIDS[os.path.basename(path)]
    openvdb.write(path, grids=grids)
    with open(path, "r+b") as handle:
        header = handle.read(FILE_UUID_OFFSET + len(fixture_uuid))
        if int.from_bytes(header[8:12], "little") != EXPECTED_FILE_FORMAT:
            raise RuntimeError("OpenVDB writer did not emit file format 225")
        if header[FILE_UUID_OFFSET : FILE_UUID_OFFSET + 36].count(b"-") != 4:
            raise RuntimeError("OpenVDB header UUID is not at the expected offset")
        handle.seek(FILE_UUID_OFFSET)
        handle.write(fixture_uuid)


AFFINE_MATRIX = (
    (0.0, 2.0, 0.0, 0.0),
    (-3.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0),
    (1.25, -2.0, 0.75, 1.0),
)
AFFINE_POINTS = ((-2, 0, 1), (0, 1, 2), (2, -1, 3))
FRUSTUM_POINTS = ((2, 2, 2), (3, 3, 3), (4, 4, 4))


def check_version():
    blender_version = tuple(bpy.app.version[:3])
    openvdb_version = tuple(openvdb.LIBRARY_VERSION)
    if blender_version != EXPECTED_BLENDER:
        raise RuntimeError(
            "this oracle is pinned to Blender 5.2.2, got " + bpy.app.version_string
        )
    if openvdb_version != EXPECTED_OPENVDB:
        raise RuntimeError("expected OpenVDB 13.0.0, got " + repr(openvdb_version))
    if openvdb.FILE_FORMAT_VERSION != EXPECTED_FILE_FORMAT:
        raise RuntimeError(
            "expected OpenVDB file format 225, got " + str(openvdb.FILE_FORMAT_VERSION)
        )


def json_value(value):
    if isinstance(value, tuple):
        return [json_value(component) for component in value]
    if isinstance(value, list):
        return [json_value(component) for component in value]
    if isinstance(value, (bool, int, float)):
        return value
    raise TypeError("unexpected OpenVDB value: " + repr(value))


def bbox(grid):
    minimum, maximum = grid.evalActiveVoxelBoundingBox()
    if minimum[0] > maximum[0]:
        return None
    return [list(minimum), list(maximum)]


def world_point(grid, index):
    return list(grid.transform.indexToWorld(index))


def world_bounds(grid):
    bounds = bbox(grid)
    if bounds is None:
        return None
    minimum, maximum = bounds
    corners = [
        world_point(
            grid,
            tuple(maximum[axis] if corner & (1 << axis) else minimum[axis] for axis in range(3)),
        )
        for corner in range(8)
    ]
    return [
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
    ]


def grid_oracle(grid, points):
    accessor = grid.getConstAccessor()
    return {
        "name": grid.name,
        "grid_type": type(grid).__name__,
        "map_type": grid.transform.typeName,
        "background": json_value(grid.background),
        "active_bbox": bbox(grid),
        "active_bbox_world": world_bounds(grid),
        "samples": [
            {
                "index": list(index),
                "value": json_value(accessor.getValue(index)),
                "active": accessor.isValueOn(index),
                "world": world_point(grid, index),
            }
            for index in points
        ],
        "background_sample": {
            "index": [17, -19, 23],
            "value": json_value(accessor.getValue((17, -19, 23))),
            "active": accessor.isValueOn((17, -19, 23)),
        },
    }


def write_value_types(path):
    transform = openvdb.createLinearTransform(AFFINE_MATRIX)
    values = (
        ("bool", openvdb.BoolGrid, False, (True, False, True)),
        ("int32", openvdb.Int32Grid, -17, (-1_234_567, 2_000_003, 1_234_567)),
        ("int64", openvdb.Int64Grid, -23, (4_294_967_301, -4_294_967_299, 8_589_934_601)),
        (
            "vec3i",
            openvdb.Vec3IGrid,
            (-2, 3, -5),
            ((7, -11, 13), (-17, 19, -23), (29, -31, 37)),
        ),
        (
            "vec3d",
            openvdb.Vec3DGrid,
            (-0.25, 0.5, 1.75),
            (
                (0.12345678901234568, -2.5000000000000004, 9.750000000000002),
                (-13.125000000000002, 0.06250000000000001, 1.5000000000000002),
                (3.2500000000000004, 8.5, -0.7500000000000001),
            ),
        ),
        ("float", openvdb.FloatGrid, -0.75, (0.125, -2.5, 9.75)),
        (
            "double",
            openvdb.DoubleGrid,
            -0.25,
            (0.12500000000000003, -2.5000000000000004, 9.750000000000002),
        ),
        (
            "vec3s",
            openvdb.Vec3SGrid,
            (-0.25, 0.5, 1.75),
            ((0.125, -2.5, 9.75), (-13.125, 0.0625, 1.5), (3.25, 8.5, -0.75)),
        ),
    )
    grids = []
    for name, grid_type, background, samples in values:
        grid = grid_type(background)
        grid.name = name
        grid.transform = transform.deepCopy()
        accessor = grid.getAccessor()
        for index, value in zip(AFFINE_POINTS, samples):
            accessor.setValueOn(index, value)
        grids.append(grid)

    half_double = openvdb.DoubleGrid(-0.75)
    half_double.name = "half_double"
    half_double.transform = transform.deepCopy()
    half_double.saveFloatAsHalf = True
    half_double_accessor = half_double.getAccessor()
    for index, value in zip(
        AFFINE_POINTS, (1.234567890123, 2.34567890123, -3.4567890123)
    ):
        half_double_accessor.setValueOn(index, value)
    grids.append(half_double)

    half_vec3d = openvdb.Vec3DGrid((-0.25, 0.5, 1.75))
    half_vec3d.name = "half_vec3d"
    half_vec3d.transform = transform.deepCopy()
    half_vec3d.saveFloatAsHalf = True
    half_vec3d_accessor = half_vec3d.getAccessor()
    for index, value in zip(
        AFFINE_POINTS,
        (
            (1.234567890123, 2.34567890123, -3.4567890123),
            (-4.567890123, 5.678901234, 6.789012345),
            (7.890123456, -8.901234567, 9.012345678),
        ),
    ):
        half_vec3d_accessor.setValueOn(index, value)
    grids.append(half_vec3d)

    write_vdb(path, grids)
    return [
        grid_oracle(openvdb.read(path, grid.name), AFFINE_POINTS)
        for grid in grids
    ]


def write_frustum(path):
    grid = openvdb.FloatGrid(0.0)
    grid.name = "density"
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    grid.transform = openvdb.createFrustumTransform(
        (0, 0, 0), (8, 8, 8), 0.5, 3.0, 0.25
    )
    accessor = grid.getAccessor()
    for z in range(1, 6):
        for y in range(1, 6):
            for x in range(1, 6):
                accessor.setValueOn((x, y, z), 1.0)
    for index, value in zip(FRUSTUM_POINTS, (1.0, 1.5, 2.0)):
        accessor.setValueOn(index, value)
    write_vdb(path, [grid])
    return grid_oracle(openvdb.read(path, grid.name), FRUSTUM_POINTS)


def write_density(path):
    grid = openvdb.FloatGrid(0.0)
    grid.name = "density"
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    grid.transform = openvdb.createLinearTransform(0.25)
    grid.transform.postTranslate((-0.5, -0.5, -0.5))
    accessor = grid.getAccessor()
    for z in range(5):
        for y in range(5):
            for x in range(5):
                # The active 3³ block crosses the 0.5 iso-level against a zero background.
                accessor.setValueOn(
                    (x, y, z),
                    1.0 if 1 <= x <= 3 and 1 <= y <= 3 and 1 <= z <= 3 else 0.0,
                )
    write_vdb(path, [grid])
    return grid_oracle(grid, ((1, 1, 1), (2, 2, 2), (3, 3, 3)))


def import_volume_grids(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.volume_import(filepath=path)
    volume = bpy.context.object
    if volume is None or volume.type != "VOLUME":
        raise RuntimeError("Blender did not import the VDB as a Volume object")
    volume.data.grids.load()
    return volume, list(volume.data.grids)


def grid_metadata(grids):
    return [
        {
            "name": grid.name,
            "channels": grid.channels,
            "is_loaded": grid.is_loaded,
            "data_type": grid.data_type,
            "matrix_object": [list(row) for row in grid.matrix_object],
        }
        for grid in grids
    ]


def volume_grid_metadata(path):
    volume, grids = import_volume_grids(path)
    return {
        "grid_names": [grid.name for grid in grids],
        "grids": grid_metadata(grids),
    }


def volume_metadata_and_mesh(path):
    volume, grids = import_volume_grids(path)
    grids_json = grid_metadata(grids)
    tree = bpy.data.node_groups.new("OpenVDB Oracle Volume To Mesh", "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    object_info = tree.nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = volume
    object_info.transform_space = "ORIGINAL"
    group_output = tree.nodes.new("NodeGroupOutput")
    to_mesh = tree.nodes.new("GeometryNodeVolumeToMesh")
    to_mesh.inputs["Threshold"].default_value = 0.5
    to_mesh.inputs["Adaptivity"].default_value = 0.0
    tree.links.new(object_info.outputs["Geometry"], to_mesh.inputs["Volume"])
    tree.links.new(to_mesh.outputs["Mesh"], group_output.inputs["Geometry"])
    carrier_mesh = bpy.data.meshes.new("OpenVDB Oracle Carrier")
    carrier = bpy.data.objects.new("OpenVDB Oracle Carrier", carrier_mesh)
    bpy.context.collection.objects.link(carrier)
    modifier = carrier.modifiers.new("OpenVDB Oracle Volume To Mesh", "NODES")
    modifier.node_group = tree
    evaluated = carrier.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    if mesh is None:
        raise RuntimeError("Blender Volume to Mesh node produced no mesh")
    positions = [list(vertex.co) for vertex in mesh.vertices]
    faces = [
        sorted(positions[index] for index in polygon.vertices)
        for polygon in mesh.polygons
    ]
    positions.sort()
    faces.sort()
    bounds = None
    if positions:
        bounds = [
            [min(position[axis] for position in positions) for axis in range(3)],
            [max(position[axis] for position in positions) for axis in range(3)],
        ]
    result = {
        "grid_names": [grid.name for grid in grids],
        "grids": grids_json,
        "volume_to_mesh": {
            "threshold": 0.5,
            "adaptivity": 0.0,
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
            "vertices": positions,
            "faces": faces,
            "bounds": bounds,
        },
    }
    evaluated.to_mesh_clear()
    return result


def volume_mesh_resampling_cases(path):
    cases = []
    for mode, voxel_size, voxel_amount in (
        ("VOXEL_SIZE", 0.1, 8),
        ("VOXEL_AMOUNT", 0.1, 8),
    ):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.object.volume_import(filepath=path)
        volume = bpy.context.object
        carrier_data = bpy.data.meshes.new("Frustum Resample Carrier Data")
        carrier = bpy.data.objects.new("Frustum Resample Carrier", carrier_data)
        bpy.context.collection.objects.link(carrier)
        modifier = carrier.modifiers.new("Frustum Resample Volume To Mesh", "VOLUME_TO_MESH")
        modifier.object = volume
        modifier.grid_name = "density"
        modifier.threshold = 0.5
        modifier.adaptivity = 0.0
        modifier.resolution_mode = mode
        modifier.voxel_size = voxel_size
        modifier.voxel_amount = voxel_amount
        bpy.context.view_layer.update()
        evaluated = carrier.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh()
        if mesh is None:
            raise RuntimeError("Blender frustum resampling produced no mesh")
        positions = [list(vertex.co) for vertex in mesh.vertices]
        bounds = None
        if positions:
            bounds = [
                [min(position[axis] for position in positions) for axis in range(3)],
                [max(position[axis] for position in positions) for axis in range(3)],
            ]
        cases.append(
            {
                "resolution_mode": mode,
                "voxel_size": voxel_size,
                "voxel_amount": voxel_amount,
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.polygons),
                "bounds": bounds,
            }
        )
        evaluated.to_mesh_clear()
    return cases


def main():
    check_version()
    argv = sys.argv[sys.argv.index("--") + 1 :]
    if len(argv) != 1:
        raise RuntimeError(
            "usage: blender --background --python generate.py -- <fixture-directory>"
        )
    output = os.path.abspath(argv[0])
    os.makedirs(output, exist_ok=True)

    value_types_path = os.path.join(output, "value_types.vdb")
    value_types = write_value_types(value_types_path)
    bpy_value_types = volume_grid_metadata(value_types_path)

    frustum_path = os.path.join(output, "frustum.vdb")
    frustum = write_frustum(frustum_path)
    frustum["bpy_volume"] = volume_metadata_and_mesh(frustum_path)
    frustum["bpy_resampled"] = volume_mesh_resampling_cases(frustum_path)

    density_path = os.path.join(output, "density.vdb")
    density = write_density(density_path)
    bpy_metadata = volume_metadata_and_mesh(density_path)
    expected = {
        "blender_version": bpy.app.version_string,
        "openvdb_library_version": list(openvdb.LIBRARY_VERSION),
        "openvdb_file_format_version": openvdb.FILE_FORMAT_VERSION,
        "affine_map_type": openvdb.createLinearTransform(AFFINE_MATRIX).typeName,
        "affine_matrix": [list(row) for row in AFFINE_MATRIX],
        "value_types": value_types,
        "bpy_value_types": bpy_value_types,
        "frustum": frustum,
        "density": density,
        "bpy_volume": bpy_metadata,
    }
    with open(os.path.join(output, "expected.json"), "w", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print("OPENVDB_TRANSFORM_TYPES_OK", output)


if __name__ == "__main__":
    main()
