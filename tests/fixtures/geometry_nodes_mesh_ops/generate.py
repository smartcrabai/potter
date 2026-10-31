"""Generate geometry-node mesh topology expectations with Blender 5.2.2."""
import json
import sys
from pathlib import Path

from mathutils import Matrix, Quaternion, Vector
import bpy

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"geometry mesh-op oracle requires Blender 5.2.2, found {bpy.app.version_string}")

OUTPUT = Path(__file__).resolve().with_name("expected.json")
if "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if args:
        OUTPUT = Path(args[0]).resolve()

ATTRIBUTE_NAMES = (
    "weight", "vector_tag", "face_tag", "edge_flag", "face_selected", "point_selected", "edge_selected",
)
BLENDER_EDGE_ATTRIBUTES = ("edge_int16_2d", "edge_float4x4")
CLOSED_DUAL_ATTRIBUTES = (
    "closed_point_value",
    "closed_edge_mark",
    "closed_face_tag",
    "closed_corner_vector",
)
FIELD_SPECS = [
    ("edge_unsigned_angle", "GeometryNodeInputMeshEdgeAngle", "Unsigned Angle", "EDGE", "FLOAT"),
    ("edge_signed_angle", "GeometryNodeInputMeshEdgeAngle", "Signed Angle", "EDGE", "FLOAT"),
    ("edge_unsigned_angle_corner", "GeometryNodeInputMeshEdgeAngle", "Unsigned Angle", "CORNER", "FLOAT"),
    ("edge_signed_angle_corner", "GeometryNodeInputMeshEdgeAngle", "Signed Angle", "CORNER", "FLOAT"),
    ("edge_face_count", "GeometryNodeInputMeshEdgeNeighbors", "Face Count", "EDGE", "INT"),
    ("edge_vertex_1", "GeometryNodeInputMeshEdgeVertices", "Vertex Index 1", "EDGE", "INT"),
    ("edge_vertex_2", "GeometryNodeInputMeshEdgeVertices", "Vertex Index 2", "EDGE", "INT"),
    ("edge_position_1", "GeometryNodeInputMeshEdgeVertices", "Position 1", "EDGE", "FLOAT_VECTOR"),
    ("edge_position_2", "GeometryNodeInputMeshEdgeVertices", "Position 2", "EDGE", "FLOAT_VECTOR"),
    ("face_area", "GeometryNodeInputMeshFaceArea", "Area", "FACE", "FLOAT"),
    ("face_vertex_count", "GeometryNodeInputMeshFaceNeighbors", "Vertex Count", "FACE", "INT"),
    ("face_neighbor_count", "GeometryNodeInputMeshFaceNeighbors", "Face Count", "FACE", "INT"),
    ("vertex_neighbor_vertex_count", "GeometryNodeInputMeshVertexNeighbors", "Vertex Count", "POINT", "INT"),
    ("vertex_neighbor_face_count", "GeometryNodeInputMeshVertexNeighbors", "Face Count", "POINT", "INT"),
    ("corners_of_edge_index", "GeometryNodeCornersOfEdge", "Corner Index", "EDGE", "INT"),
    ("corners_of_edge_total", "GeometryNodeCornersOfEdge", "Total", "EDGE", "INT"),
    ("corners_of_face_index", "GeometryNodeCornersOfFace", "Corner Index", "FACE", "INT"),
    ("corners_of_face_total", "GeometryNodeCornersOfFace", "Total", "FACE", "INT"),
    ("corners_of_face_default_index", "GeometryNodeCornersOfFace", "Corner Index", "FACE", "INT"),
    ("corners_of_vertex_index", "GeometryNodeCornersOfVertex", "Corner Index", "POINT", "INT"),
    ("corners_of_vertex_total", "GeometryNodeCornersOfVertex", "Total", "POINT", "INT"),
    ("edges_of_vertex_index", "GeometryNodeEdgesOfVertex", "Edge Index", "POINT", "INT"),
    ("edges_of_vertex_total", "GeometryNodeEdgesOfVertex", "Total", "POINT", "INT"),
    ("edges_of_corner_next", "GeometryNodeEdgesOfCorner", "Next Edge Index", "CORNER", "INT"),
    ("edges_of_corner_previous", "GeometryNodeEdgesOfCorner", "Previous Edge Index", "CORNER", "INT"),
    ("face_of_corner_index", "GeometryNodeFaceOfCorner", "Face Index", "CORNER", "INT"),
    ("face_of_corner_offset", "GeometryNodeFaceOfCorner", "Index in Face", "CORNER", "INT"),
    ("offset_corner_next", "GeometryNodeOffsetCornerInFace", "Corner Index", "CORNER", "INT"),
    ("vertex_of_corner_index", "GeometryNodeVertexOfCorner", "Vertex Index", "CORNER", "INT"),
]

CASES = {
    "split_edges": {
        "type": "GeometryNodeSplitEdges", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Selection": True},
    },
    "delete_faces": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "FACE", "mode": "ALL"}, "inputs": {}, "selection_attribute": "face_selected",
    },
    "delete_all_points": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "POINT", "mode": "ALL"}, "inputs": {"Selection": True},
    },
    "delete_points_edge_face": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "POINT", "mode": "EDGE_FACE"}, "inputs": {}, "selection_attribute": "point_selected",
    },
    "delete_points_only_face": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "POINT", "mode": "ONLY_FACE"}, "inputs": {}, "selection_attribute": "point_selected",
    },
    "delete_edges_edge_face": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "EDGE", "mode": "EDGE_FACE"}, "inputs": {}, "selection_attribute": "edge_selected",
    },
    "delete_edges_only_face": {
        "type": "GeometryNodeDeleteGeometry", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "EDGE", "mode": "ONLY_FACE"}, "inputs": {}, "selection_attribute": "edge_selected",
    },
    "separate_faces": {
        "type": "GeometryNodeSeparateGeometry", "geometry": "Geometry", "result": "Selection",
        "properties": {"domain": "FACE"}, "inputs": {}, "selection_attribute": "face_selected",
    },
    "separate_faces_inverted": {
        "type": "GeometryNodeSeparateGeometry", "geometry": "Geometry", "result": "Inverted",
        "properties": {"domain": "FACE"}, "inputs": {}, "selection_attribute": "face_selected",
    },
    "triangulate": {
        "type": "GeometryNodeTriangulate", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Selection": True, "Quad Method": "Beauty", "N-gon Method": "Beauty"},
    },
    "scale_elements": {
        "type": "GeometryNodeScaleElements", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "FACE"},
        "inputs": {"Selection": True, "Scale": 0.65, "Center": [0.25, 0.25, 0.0], "Scale Mode": "Uniform", "Axis": [0.0, 0.0, 1.0]},
    },
    "scale_elements_single_axis": {
        "type": "GeometryNodeScaleElements", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "FACE"},
        "inputs": {"Selection": True, "Scale": 0.65, "Center": [0.25, 0.25, 0.0], "Scale Mode": "Single Axis", "Axis": [0.0, 0.0, 1.0]},
    },
    "scale_elements_single_axis_zero_axis": {
        "type": "GeometryNodeScaleElements", "geometry": "Geometry", "result": "Geometry",
        "properties": {"domain": "EDGE"},
        "inputs": {"Selection": True, "Scale": 2.0, "Center": [0.0, 0.0, 0.0], "Scale Mode": "Single Axis", "Axis": [0.0, 0.0, 0.0]},
    },
    "subdivide": {
        "type": "GeometryNodeSubdivideMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Level": 1},
    },
    "subdivide_attributes": {
        "type": "GeometryNodeSubdivideMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Level": 1},
    },
    "subdivide_level_2": {
        "type": "GeometryNodeSubdivideMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Level": 2},
    },
    "extrude": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "FACES"}, "inputs": {"Selection": True, "Offset Scale": 1.0, "Individual": False},
    },
    "extrude_face_fields": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "FACES"},
        "inputs": {"Selection": True, "Offset Scale": 1.0, "Individual": False},
        "store_outputs": [("extrude_face_top", "Top", "FACE"), ("extrude_face_side", "Side", "FACE")],
    },
    "extrude_face_individual_fields": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "FACES"},
        "inputs": {"Selection": True, "Offset Scale": 1.0, "Individual": True},
        "store_outputs": [("extrude_face_individual_top", "Top", "FACE"), ("extrude_face_individual_side", "Side", "FACE")],
    },
    "extrude_face_top_on_points": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "FACES"},
        "inputs": {"Selection": True, "Offset Scale": 1.0, "Individual": False},
        "store_outputs": [("extrude_face_top_on_points", "Top", "POINT"), ("extrude_face_side_on_points", "Side", "FACE")],
    },
    "extrude_edge_fields": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "EDGES"},
        "inputs": {"Selection": True, "Offset Scale": 1.0},
        "store_outputs": [("extrude_edge_top", "Top", "EDGE"), ("extrude_edge_side", "Side", "FACE")],
    },
    "extrude_vertex_fields": {
        "type": "GeometryNodeExtrudeMesh", "geometry": "Mesh", "result": "Mesh",
        "properties": {"mode": "VERTICES"},
        "inputs": {"Selection": True, "Offset Scale": 1.0},
        "store_outputs": [("extrude_vertex_top", "Top", "POINT"), ("extrude_vertex_side", "Side", "EDGE")],
    },
    "merge_by_distance": {
        "type": "GeometryNodeMergeByDistance", "geometry": "Geometry", "result": "Geometry",
        "properties": {}, "inputs": {"Selection": True, "Mode": "All", "Distance": 0.001},
    },
    "merge_by_distance_connected": {
        "type": "GeometryNodeMergeByDistance", "geometry": "Geometry", "result": "Geometry",
        "properties": {}, "inputs": {"Selection": True, "Mode": "Connected", "Distance": 0.001},
    },
    "flip_faces": {
        "type": "GeometryNodeFlipFaces", "geometry": "Mesh", "result": "Mesh",
        "properties": {}, "inputs": {"Selection": True},
    },
    "dual_mesh": {
        "type": "GeometryNodeDualMesh", "geometry": "Mesh", "result": "Dual Mesh",
        "properties": {}, "inputs": {"Keep Boundaries": True},
    },
    "dual_mesh_drop_boundaries": {
        "type": "GeometryNodeDualMesh", "geometry": "Mesh", "result": "Dual Mesh",
        "properties": {}, "inputs": {"Keep Boundaries": False},
    },
}


def make_source():
    points = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 1.0, 1.0),
        (0.0, 2.0, 0.0), (1.0, 2.0, 0.5), (0.75, 0.125, 0.5),
        (3.0, 0.0, 0.0), (0.0, 0.0, 0.0),
    ]
    faces = [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (1, 4, 8)]
    mesh = bpy.data.meshes.new("gn_mesh_ops_input")
    mesh.from_pydata(points, [(8, 9)], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("gn_mesh_ops_input", mesh)
    bpy.context.scene.collection.objects.link(obj)

    attr = mesh.attributes.new("weight", "FLOAT", "POINT")
    for index, item in enumerate(attr.data):
        item.value = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, -0.5, 0.875][index]
    attr = mesh.attributes.new("vector_tag", "FLOAT_VECTOR", "POINT")
    for index, item in enumerate(attr.data):
        item.vector = (float(index) / 4.0, float(index % 3) / 2.0, -float(index) / 8.0)
    attr = mesh.attributes.new("face_tag", "INT", "FACE")
    for item, value in zip(attr.data, (4, 9, 15, -2)):
        item.value = value
    attr = mesh.attributes.new("face_selected", "BOOLEAN", "FACE")
    for item, value in zip(attr.data, (False, True, True, True)):
        item.value = value
    attr = mesh.attributes.new("point_selected", "BOOLEAN", "POINT")
    for index, item in enumerate(attr.data):
        item.value = index in (4, 10)
    attr = mesh.attributes.new("edge_selected", "BOOLEAN", "EDGE")
    for edge, item in zip(mesh.edges, attr.data):
        item.value = tuple(sorted(edge.vertices)) == (1, 4)
    attr = mesh.attributes.new("edge_flag", "BOOLEAN", "EDGE")
    for index, item in enumerate(attr.data):
        item.value = index % 3 == 1
    short2 = mesh.attributes.new("edge_int16_2d", "INT16_2D", "EDGE")
    for index, item in enumerate(short2.data):
        item.value = (3 * index - 11, 9 - 2 * index)
    matrix = mesh.attributes.new("edge_float4x4", "FLOAT4X4", "EDGE")
    for index, item in enumerate(matrix.data):
        axis = Vector((0.25 + 0.05 * index, 0.7, 1.0)).normalized()
        rotation = Quaternion(axis, 0.1 + 0.035 * index).to_matrix().to_4x4()
        scale = Matrix.Diagonal(Vector((
            1.0 + 0.05 * index,
            0.8 + 0.03 * index,
            1.2 + 0.02 * index,
            1.0,
        )))
        translation = Matrix.Translation(Vector((
            0.2 * index,
            -0.125 * index,
            0.3 + 0.05 * index,
        )))
        item.value = translation @ rotation @ scale
    return install_primary_mesh_attributes(obj)


def install_primary_mesh_attributes(obj):
    mesh = obj.data
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for face_index, polygon in enumerate(mesh.polygons):
        for corner, loop_index in enumerate(polygon.loop_indices):
            uv_layer.data[loop_index].uv = (
                0.25 * face_index + 0.125 * corner,
                0.0625 * face_index + 0.125 * corner,
            )
    mesh.uv_layers.active_index = 0
    mesh.uv_layers.active_render_index = 0

    color = mesh.color_attributes.new(name="Paint", type="FLOAT_COLOR", domain="CORNER")
    for polygon in mesh.polygons:
        for corner, loop_index in enumerate(polygon.loop_indices):
            color.data[loop_index].color = (
                0.1 + 0.05 * polygon.index,
                0.15 + 0.05 * corner,
                0.25 + 0.025 * polygon.index,
                0.5 + 0.1 * corner,
            )
    mesh.color_attributes.active_color_index = 0
    mesh.color_attributes.render_color_index = 0

    edge_crease = mesh.attributes.new("crease_edge", "FLOAT", "EDGE")
    for edge, item in zip(mesh.edges, edge_crease.data):
        endpoints = tuple(sorted(edge.vertices))
        item.value = 0.75 if endpoints == (1, 4) else (0.25 if edge.index % 3 == 0 else 0.0)
    vertex_crease = mesh.attributes.new("crease_vert", "FLOAT", "POINT")
    for vertex, item in zip(mesh.vertices, vertex_crease.data):
        item.value = 0.6 if vertex.index == 4 else (0.35 if vertex.index == 8 else 0.0)

    for polygon in mesh.polygons:
        polygon.use_smooth = polygon.index in (0, 2)
    for edge in mesh.edges:
        edge.use_edge_sharp = edge.index in (3, 7)
        edge.use_seam = edge.index in (1, 10)
    return obj
def make_closed_dual_source():
    vertices = [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    mesh = bpy.data.meshes.new("gn_mesh_ops_closed_dual_input")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)

    point_value = mesh.attributes.new("closed_point_value", "FLOAT", "POINT")
    for index, item in enumerate(point_value.data):
        item.value = [0.125, 0.375, 0.625, 0.875][index]
    edge_mark = mesh.attributes.new("closed_edge_mark", "BOOLEAN", "EDGE")
    for index, item in enumerate(edge_mark.data):
        item.value = index % 2 == 0
    face_tag = mesh.attributes.new("closed_face_tag", "INT", "FACE")
    for item, value in zip(face_tag.data, (2, 3, 5, 7)):
        item.value = value
    face_paint = mesh.attributes.new("ClosedFacePaint", "FLOAT_COLOR", "FACE")
    for face_index, item in enumerate(face_paint.data):
        item.color = (
            0.1 + 0.15 * face_index,
            0.2 + 0.1 * face_index,
            0.3 + 0.05 * face_index,
            0.55 + 0.1 * face_index,
        )
    corner_vector = mesh.attributes.new("closed_corner_vector", "FLOAT_VECTOR", "CORNER")
    for index, item in enumerate(corner_vector.data):
        item.vector = (index / 8.0, -index / 4.0, 0.5 + index / 16.0)

    obj = bpy.data.objects.new("gn_mesh_ops_closed_dual_input", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj




def new_tree(name):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return tree


def node(tree, node_type, name):
    result = tree.nodes.new(node_type)
    result.name = name
    result.label = name
    return result


def assign_input(node_value, name, value):
    socket = node_value.inputs.get(name)
    if socket is None:
        socket = next((socket for socket in node_value.inputs if socket.name == name), None)
    if socket is None:
        raise RuntimeError(
            f"{node_value.bl_idname} has no input {name!r}; "
            f"found {[socket.name for socket in node_value.inputs]}"
        )
    try:
        socket.default_value = value
    except (AttributeError, TypeError, ValueError):
        menu_defaults = {"Quad Method": "Beauty", "N-gon Method": "Beauty"}
        if name not in menu_defaults or value != menu_defaults[name]:
            raise

def attach_modifier(obj, tree, name):
    modifier = obj.modifiers.new(name, "NODES")
    modifier.node_group = tree
    return modifier


def read_attribute(mesh, name):
    attr = mesh.attributes.get(name)
    if attr is None:
        raise RuntimeError(f"evaluated mesh is missing named attribute {name!r}")
    if attr.data_type == "FLOAT_VECTOR":
        data = [list(map(float, item.vector)) for item in attr.data]
    elif attr.data_type == "BOOLEAN":
        data = [bool(item.value) for item in attr.data]
    elif attr.data_type == "INT":
        data = [int(item.value) for item in attr.data]
    else:
        data = [float(item.value) for item in attr.data]
    return {"domain": attr.domain, "data_type": attr.data_type, "data": data}


def mesh_attributes(mesh, names):
    return {name: read_attribute(mesh, name) for name in names}
def blender_attribute_records(mesh):
    records = []
    for name in BLENDER_EDGE_ATTRIBUTES:
        attribute = mesh.attributes.get(name)
        if attribute is None:
            continue
        if attribute.data_type == "INT16_2D":
            values = [list(map(int, item.value)) for item in attribute.data]
        elif attribute.data_type == "FLOAT4X4":
            values = [
                [list(map(float, row)) for row in item.value]
                for item in attribute.data
            ]
        else:
            raise RuntimeError(f"unexpected data type for {name}: {attribute.data_type}")
        records.append({
            "name": name,
            "domain": attribute.domain,
            "data_type": attribute.data_type,
            "values": values,
        })
    return records
def mesh_attributes_present(mesh, names):
    return {name: read_attribute(mesh, name) for name in names if mesh.attributes.get(name) is not None}


def subdivision_attributes(mesh, by_ids, required=False):
    result = {}
    uv_layers = list(mesh.uv_layers)
    if not uv_layers and required:
        raise RuntimeError("source mesh lacks its required UV layer")
    if uv_layers:
        uv_records = []
        for layer in uv_layers:
            faces = []
            for polygon in mesh.polygons:
                uv = [
                    [float(layer.data[loop].uv.x), 1.0 - float(layer.data[loop].uv.y)]
                    for loop in polygon.loop_indices
                ]
                faces.append({"face_id": polygon.index, "uv": uv} if by_ids else uv)
            uv_records.append({"name": layer.name, "faces": faces})
        active_uv = uv_layers[mesh.uv_layers.active_index].name
        render_uv = uv_layers[mesh.uv_layers.active_render_index].name
        result["uv_layers"] = {
            "layers": uv_records,
            "active": active_uv,
            "render": render_uv,
        }

    color_attributes = list(mesh.color_attributes)
    color_names = {color.name for color in color_attributes}
    colors = color_attributes + [
        attribute for attribute in mesh.attributes
        if attribute.name not in color_names
        and attribute.data_type in ("FLOAT_COLOR", "BYTE_COLOR")
    ]
    if not colors and required:
        raise RuntimeError("source mesh lacks its required color layer")
    if colors:
        active_color = (
            color_attributes[mesh.color_attributes.active_color_index].name
            if color_attributes
            else None
        )
        render_color = (
            color_attributes[mesh.color_attributes.render_color_index].name
            if color_attributes
            else None
        )
        result["color_layers"] = {
            "layers": [color.name for color in colors],
            "active": active_color,
            "render": render_color,
        }
        metadata = {"color_data_types": {color.name: color.data_type for color in colors}}
        if active_color is not None:
            metadata["color_active_blender"] = active_color
        result["blender_metadata"] = metadata
        for color in colors:
            if color.domain == "CORNER":
                rows = [
                    [
                        list(map(float, color.data[loop].color))
                        for loop in polygon.loop_indices
                    ]
                    for polygon in mesh.polygons
                ]
                values = (
                    {f"f{polygon.index}": row for polygon, row in zip(mesh.polygons, rows)}
                    if by_ids
                    else rows
                )
            elif color.domain == "POINT":
                rows = [list(map(float, item.color)) for item in color.data]
                values = {f"v{index}": row for index, row in enumerate(rows)} if by_ids else rows
            elif color.domain == "FACE":
                rows = [list(map(float, color.data[face.index].color)) for face in mesh.polygons]
                values = {f"f{index}": row for index, row in enumerate(rows)} if by_ids else rows
            else:
                raise RuntimeError(f"unsupported subdivision color domain {color.domain}")
            result[color.name] = {
                "domain": color.domain.lower(),
                "type": "color",
                "values": values,
            }

    sharp_attribute = mesh.attributes.get("sharp_edge")
    seam_attribute = mesh.attributes.get("uv_seam")
    if required or sharp_attribute is not None or seam_attribute is not None:
        sharp = [bool(edge.use_edge_sharp) for edge in mesh.edges]
        seam = [bool(edge.use_seam) for edge in mesh.edges]
        result["blender_edge_flags"] = [
            {"sharp": is_sharp, "seam": is_seam}
            for is_sharp, is_seam in zip(sharp, seam)
        ]
        for name, domain, prefix, exists, flags in (
            ("sharp", "edge", "e", sharp_attribute is not None or required, sharp),
            ("seam", "edge", "e", seam_attribute is not None or required, seam),
        ):
            if exists:
                result[name] = {
                    "domain": domain,
                    "type": "bool",
                    "values": (
                        {f"{prefix}{index}": True for index, value in enumerate(flags) if value}
                        if by_ids
                        else flags
                    ),
                }

    sharp_faces = mesh.attributes.get("sharp_face")
    if required or sharp_faces is not None:
        smooth = [bool(polygon.use_smooth) for polygon in mesh.polygons]
        shade = {
            "domain": "face",
            "type": "bool",
            "values": (
                {f"f{index}": value for index, value in enumerate(smooth)}
                if by_ids
                else smooth
            ),
        }
        if sharp_faces is not None:
            shade["sharp_face_present"] = True
        result["shade_smooth"] = shade

    for name, domain, prefix in (
        ("crease_edge", "edge", "e"),
        ("crease_vert", "point", "v"),
    ):
        attribute = mesh.attributes.get(name)
        if attribute is None:
            if required:
                raise RuntimeError(f"source mesh lacks its required {name} attribute")
            continue
        values = [float(item.value) for item in attribute.data]
        result[name] = {
            "domain": domain,
            "type": "float",
            "values": (
                {f"{prefix}{index}": value for index, value in enumerate(values) if value != 0.0}
                if by_ids
                else values
            ),
        }
    return result


def mesh_record(mesh, names, include_special_attributes=False, require_special_attributes=False):
    attributes = mesh_attributes(mesh, names)
    if include_special_attributes:
        attributes.update(
            subdivision_attributes(
                mesh,
                by_ids=True,
                required=require_special_attributes,
            )
        )
    blender_attributes = blender_attribute_records(mesh)
    if blender_attributes:
        attributes["blender_attributes"] = blender_attributes
    return {
        "vertices": [{"id": v.index, "co": list(map(float, v.co))} for v in mesh.vertices],
        "edges": [{"id": e.index, "v": list(map(int, e.vertices))} for e in mesh.edges],
        "faces": [
            {"id": face.index, "v": list(map(int, face.vertices)), "material_index": face.material_index}
            for face in mesh.polygons
        ],
        "attributes": attributes,
        "next_id": {"vertex": len(mesh.vertices), "edge": len(mesh.edges), "face": len(mesh.polygons)},
    }


def mesh_snapshot(mesh, names, include_special_attributes=False):
    attributes = mesh_attributes_present(mesh, names)
    if include_special_attributes:
        attributes.update(subdivision_attributes(mesh, by_ids=False))
    blender_attributes = blender_attribute_records(mesh)
    if blender_attributes:
        attributes["blender_attributes"] = blender_attributes
    return {
        "vertices": [list(map(float, vertex.co)) for vertex in mesh.vertices],
        "edges": [sorted(map(int, edge.vertices)) for edge in mesh.edges],
        "faces": [list(map(int, face.vertices)) for face in mesh.polygons],
        "attributes": attributes,
    }


def evaluate_tree(obj, tree, names, include_special_attributes=False):
    attach_modifier(obj, tree, tree.name)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        result = mesh_snapshot(mesh, names, include_special_attributes)
    finally:
        evaluated.to_mesh_clear()
        obj.modifiers.remove(obj.modifiers[-1])
        bpy.data.node_groups.remove(tree)
    return result




def topology_tree():
    tree = new_tree("gn_mesh_ops_topology_fields")
    nodes = tree.nodes
    group_input = nodes.get("Group Input")
    if group_input is None:
        group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = nodes.get("Group Output")
    if group_output is None:
        group_output = node(tree, "NodeGroupOutput", "Group Output")
    index = node(tree, "GeometryNodeInputIndex", "Index")
    field_nodes = {}
    for name, node_type, output, domain, data_type in FIELD_SPECS:
        node_key = (
            f"{node_type}:{name}"
            if name == "corners_of_face_default_index"
            else node_type
        )
        if node_key not in field_nodes:
            field_nodes[node_key] = node(tree, node_type, node_key)
    stores = []
    for ordinal, (name, node_type, output, domain, data_type) in enumerate(FIELD_SPECS):
        store = node(tree, "GeometryNodeStoreNamedAttribute", f"store_{name}")
        store.domain = domain
        store.data_type = data_type
        assign_input(store, "Name", name)
        stores.append(store)
        node_key = (
            f"{node_type}:{name}"
            if name == "corners_of_face_default_index"
            else node_type
        )
        source = field_nodes[node_key]
        tree.links.new(source.outputs[output], store.inputs["Value"])
        if node_type in {
            "GeometryNodeCornersOfEdge", "GeometryNodeCornersOfFace",
            "GeometryNodeCornersOfVertex", "GeometryNodeEdgesOfVertex",
        }:
            key = {
                "GeometryNodeCornersOfEdge": "Edge Index",
                "GeometryNodeCornersOfFace": "Face Index",
                "GeometryNodeCornersOfVertex": "Vertex Index",
                "GeometryNodeEdgesOfVertex": "Vertex Index",
            }[node_type]
            tree.links.new(index.outputs["Index"], source.inputs[key])
            if name != "corners_of_face_default_index":
                assign_input(source, "Sort Index", 0)
        elif node_type in {
            "GeometryNodeEdgesOfCorner", "GeometryNodeFaceOfCorner",
            "GeometryNodeOffsetCornerInFace", "GeometryNodeVertexOfCorner",
        }:
            tree.links.new(index.outputs["Index"], source.inputs["Corner Index"])
            if node_type == "GeometryNodeOffsetCornerInFace":
                assign_input(source, "Offset", 1)
        if ordinal == 0:
            tree.links.new(group_input.outputs["Geometry"], store.inputs["Geometry"])
        else:
            tree.links.new(stores[ordinal - 1].outputs["Geometry"], store.inputs["Geometry"])
    tree.links.new(stores[-1].outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def operation_tree(name, case):
    tree = new_tree("gn_mesh_ops_" + name)
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    operation = node(tree, case["type"], "operation")
    for prop, value in case["properties"].items():
        setattr(operation, prop, value)
    for input_name, value in case["inputs"].items():
        assign_input(operation, input_name, value)
    tree.links.new(group_input.outputs["Geometry"], operation.inputs[case["geometry"]])
    if case["type"] == "GeometryNodeScaleElements":
        center = node(tree, "ShaderNodeCombineXYZ", "Center")
        for axis, value in zip("XYZ", case["inputs"]["Center"]):
            assign_input(center, axis, value)
        tree.links.new(center.outputs["Vector"], operation.inputs["Center"])
    selection_attribute = case.get("selection_attribute")
    if selection_attribute is not None:
        selection = node(tree, "GeometryNodeInputNamedAttribute", "Selection")
        selection.data_type = "BOOLEAN"
        assign_input(selection, "Name", selection_attribute)
        tree.links.new(selection.outputs["Attribute"], operation.inputs["Selection"])
    if case.get("store_outputs"):
        previous = None
        for attribute_name, output_name, domain in case["store_outputs"]:
            store = node(tree, "GeometryNodeStoreNamedAttribute", "store_" + attribute_name)
            store.domain = domain
            store.data_type = "BOOLEAN"
            assign_input(store, "Name", attribute_name)
            tree.links.new(operation.outputs[output_name], store.inputs["Value"])
            geometry = operation.outputs[case["result"]] if previous is None else previous.outputs["Geometry"]
            tree.links.new(geometry, store.inputs["Geometry"])
            previous = store
        output = previous.outputs["Geometry"]
    else:
        output = operation.outputs[case["result"]]
    tree.links.new(output, group_output.inputs["Geometry"])
    return tree


def closed_dual_tree():
    tree = new_tree("gn_mesh_ops_closed_dual")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    dual = node(tree, "GeometryNodeDualMesh", "dual")
    assign_input(dual, "Keep Boundaries", False)
    tree.links.new(group_input.outputs["Geometry"], dual.inputs["Mesh"])
    tree.links.new(dual.outputs["Dual Mesh"], group_output.inputs["Geometry"])
    return tree
def top_side_subdivide_tree():
    tree = new_tree("gn_mesh_ops_top_side_subdivide")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    selection = node(tree, "GeometryNodeInputNamedAttribute", "Face Selection")
    selection.data_type = "BOOLEAN"
    assign_input(selection, "Name", "face_selected")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    subdivide = node(tree, "GeometryNodeSubdivideMesh", "Subdivide")
    assign_input(subdivide, "Level", 1)
    top = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top")
    top.domain = "FACE"
    top.data_type = "BOOLEAN"
    assign_input(top, "Name", "top_after_subdivide")
    side = node(tree, "GeometryNodeStoreNamedAttribute", "Store Side")
    side.domain = "FACE"
    side.data_type = "BOOLEAN"
    assign_input(side, "Name", "side_after_subdivide")
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(selection.outputs["Attribute"], extrude.inputs["Selection"])
    tree.links.new(extrude.outputs["Mesh"], subdivide.inputs["Mesh"])
    tree.links.new(subdivide.outputs["Mesh"], top.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], top.inputs["Value"])
    tree.links.new(top.outputs["Geometry"], side.inputs["Geometry"])
    tree.links.new(extrude.outputs["Side"], side.inputs["Value"])
    tree.links.new(side.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def extrude_all_faces_tree():
    tree = new_tree("gn_mesh_ops_extrude_all_faces")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(extrude.outputs["Mesh"], group_output.inputs["Geometry"])
    return tree

def top_after_separate_tree():
    tree = new_tree("gn_mesh_ops_top_after_separate")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    selection = node(tree, "GeometryNodeInputNamedAttribute", "Face Selection")
    selection.data_type = "BOOLEAN"
    assign_input(selection, "Name", "face_selected")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    separate = node(tree, "GeometryNodeSeparateGeometry", "Separate")
    separate.domain = "FACE"
    store = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top")
    store.domain = "FACE"
    store.data_type = "BOOLEAN"
    assign_input(store, "Name", "top_after_separate")
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(selection.outputs["Attribute"], extrude.inputs["Selection"])
    tree.links.new(extrude.outputs["Mesh"], separate.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], separate.inputs["Selection"])
    tree.links.new(separate.outputs["Selection"], store.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], store.inputs["Value"])
    tree.links.new(store.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def sibling_topology_fields_tree():
    tree = new_tree("gn_mesh_ops_sibling_topology_fields")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    area = node(tree, "GeometryNodeInputMeshFaceArea", "Face Area")
    angle = node(tree, "GeometryNodeInputMeshEdgeAngle", "Edge Angle")
    scale_x = node(tree, "GeometryNodeTransform", "Scale X")
    assign_input(scale_x, "Scale", (2.0, 1.0, 1.0))
    scale_y = node(tree, "GeometryNodeTransform", "Scale Y")
    assign_input(scale_y, "Scale", (1.0, 3.0, 1.0))
    area_x = node(tree, "GeometryNodeStoreNamedAttribute", "Store Area X")
    area_x.domain = "FACE"
    area_x.data_type = "FLOAT"
    assign_input(area_x, "Name", "area_shared")
    angle_x = node(tree, "GeometryNodeStoreNamedAttribute", "Store Angle X")
    angle_x.domain = "EDGE"
    angle_x.data_type = "FLOAT"
    assign_input(angle_x, "Name", "angle_shared")
    area_y = node(tree, "GeometryNodeStoreNamedAttribute", "Store Area Y")
    area_y.domain = "FACE"
    area_y.data_type = "FLOAT"
    assign_input(area_y, "Name", "area_shared")
    angle_y = node(tree, "GeometryNodeStoreNamedAttribute", "Store Angle Y")
    angle_y.domain = "EDGE"
    angle_y.data_type = "FLOAT"
    assign_input(angle_y, "Name", "angle_shared")
    join = node(tree, "GeometryNodeJoinGeometry", "Join")
    tree.links.new(group_input.outputs["Geometry"], scale_x.inputs["Geometry"])
    tree.links.new(group_input.outputs["Geometry"], scale_y.inputs["Geometry"])
    tree.links.new(scale_x.outputs["Geometry"], area_x.inputs["Geometry"])
    tree.links.new(area.outputs["Area"], area_x.inputs["Value"])
    tree.links.new(area_x.outputs["Geometry"], angle_x.inputs["Geometry"])
    tree.links.new(angle.outputs["Unsigned Angle"], angle_x.inputs["Value"])
    tree.links.new(scale_y.outputs["Geometry"], area_y.inputs["Geometry"])
    tree.links.new(area.outputs["Area"], area_y.inputs["Value"])
    tree.links.new(area_y.outputs["Geometry"], angle_y.inputs["Geometry"])
    tree.links.new(angle.outputs["Unsigned Angle"], angle_y.inputs["Value"])
    tree.links.new(angle_x.outputs["Geometry"], join.inputs["Geometry"])
    tree.links.new(angle_y.outputs["Geometry"], join.inputs["Geometry"])
    tree.links.new(join.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def make_single_vertex_source():
    mesh = bpy.data.meshes.new("gn_mesh_ops_single_vertex")
    mesh.from_pydata([(1.0, 0.0, 0.0)], [], [])
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("gn_mesh_ops_single_vertex", mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def vertex_top_edge_tree():
    tree = new_tree("gn_mesh_ops_vertex_top_edge")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    position = node(tree, "GeometryNodeInputPosition", "Offset")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "VERTICES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Selection", True)
    store = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top")
    store.domain = "EDGE"
    store.data_type = "BOOLEAN"
    assign_input(store, "Name", "top_on_edge")
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(position.outputs["Position"], extrude.inputs["Offset"])
    tree.links.new(extrude.outputs["Mesh"], store.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], store.inputs["Value"])
    tree.links.new(store.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree



def top_on_original_tree():
    tree = new_tree("gn_mesh_ops_top_on_original")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    store = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top")
    store.domain = "FACE"
    store.data_type = "BOOLEAN"
    assign_input(store, "Name", "top_on_original")
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(group_input.outputs["Geometry"], store.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], store.inputs["Value"])
    tree.links.new(store.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree
def unconsumed_top_distribute_dead_consumer_tree():
    tree = new_tree("gn_mesh_ops_unconsumed_top_dead_consumer")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    dead_store = node(tree, "GeometryNodeStoreNamedAttribute", "Unused Top Store")
    dead_store.domain = "FACE"
    dead_store.data_type = "BOOLEAN"
    assign_input(dead_store, "Name", "unused_top")
    distribute = node(tree, "GeometryNodeDistributePointsOnFaces", "Distribute")
    assign_input(distribute, "Selection", True)
    assign_input(distribute, "Density", 0.0)
    assign_input(distribute, "Density Factor", 1.0)
    assign_input(distribute, "Density Max", 1.0)
    assign_input(distribute, "Seed", 17)
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(extrude.outputs["Mesh"], distribute.inputs["Mesh"])
    tree.links.new(extrude.outputs["Mesh"], dead_store.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], dead_store.inputs["Value"])
    tree.links.new(distribute.outputs["Points"], group_output.inputs["Geometry"])
    return tree

def unconsumed_top_distribute_tree():
    tree = new_tree("gn_mesh_ops_unconsumed_top_distribute")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    extrude = node(tree, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    dead_store = node(tree, "GeometryNodeStoreNamedAttribute", "Unused Top Store")
    dead_store.domain = "FACE"
    dead_store.data_type = "BOOLEAN"
    assign_input(dead_store, "Name", "unused_top")
    distribute = node(tree, "GeometryNodeDistributePointsOnFaces", "Distribute")
    points_to_vertices = node(tree, "GeometryNodePointsToVertices", "Points to Vertices")
    radius_input = node(tree, "GeometryNodeInputRadius", "Input Radius")
    store_radius = node(tree, "GeometryNodeStoreNamedAttribute", "Store Radius")
    store_radius.domain = "POINT"
    store_radius.data_type = "FLOAT"
    assign_input(store_radius, "Name", "radius_probe")
    assign_input(distribute, "Selection", True)
    assign_input(distribute, "Density", 0.1)
    assign_input(distribute, "Density Factor", 1.0)
    assign_input(distribute, "Density Max", 1.0)
    assign_input(distribute, "Seed", 17)
    tree.links.new(group_input.outputs["Geometry"], extrude.inputs["Mesh"])
    tree.links.new(extrude.outputs["Mesh"], distribute.inputs["Mesh"])
    tree.links.new(extrude.outputs["Mesh"], dead_store.inputs["Geometry"])
    tree.links.new(extrude.outputs["Top"], dead_store.inputs["Value"])
    tree.links.new(distribute.outputs["Points"], store_radius.inputs["Geometry"])
    tree.links.new(radius_input.outputs["Radius"], store_radius.inputs["Value"])
    tree.links.new(store_radius.outputs["Geometry"], points_to_vertices.inputs["Points"])
    tree.links.new(points_to_vertices.outputs["Mesh"], group_output.inputs["Geometry"])
    return tree

def nested_top_group_tree():
    child = new_tree("gn_mesh_ops_nested_top_child")
    child.interface.new_socket(
        name="Selection",
        in_out="INPUT",
        socket_type="NodeSocketBool",
    )
    child.interface.new_socket(
        name="Top",
        in_out="OUTPUT",
        socket_type="NodeSocketBool",
    )
    child_input = node(child, "NodeGroupInput", "Group Input")
    child_output = node(child, "NodeGroupOutput", "Group Output")
    extrude = node(child, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    subdivide = node(child, "GeometryNodeSubdivideMesh", "Subdivide")
    assign_input(subdivide, "Level", 1)
    child.links.new(child_input.outputs["Geometry"], extrude.inputs["Mesh"])
    child.links.new(child_input.outputs["Selection"], extrude.inputs["Selection"])
    child.links.new(extrude.outputs["Mesh"], subdivide.inputs["Mesh"])
    child.links.new(subdivide.outputs["Mesh"], child_output.inputs["Geometry"])
    child.links.new(extrude.outputs["Top"], child_output.inputs["Top"])

    tree = new_tree("gn_mesh_ops_nested_top_calls")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    calls = []
    for name, selected in (("Call A", True), ("Call B", False), ("Call Dead", True)):
        call = node(tree, "GeometryNodeGroup", name)
        call.node_tree = child
        assign_input(call, "Selection", selected)
        tree.links.new(group_input.outputs["Geometry"], call.inputs["Geometry"])
        calls.append(call)

    store_a = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top A")
    store_a.domain = "FACE"
    store_a.data_type = "BOOLEAN"
    assign_input(store_a, "Name", "top_a")
    store_b = node(tree, "GeometryNodeStoreNamedAttribute", "Store Cross-Call Top")
    store_b.domain = "FACE"
    store_b.data_type = "BOOLEAN"
    assign_input(store_b, "Name", "cross_call_top_b")
    dead_store = node(tree, "GeometryNodeStoreNamedAttribute", "Unused Top Store")
    dead_store.domain = "FACE"
    dead_store.data_type = "BOOLEAN"
    assign_input(dead_store, "Name", "unused_top")
    distribute = node(tree, "GeometryNodeDistributePointsOnFaces", "Distribute")
    assign_input(distribute, "Selection", True)
    assign_input(distribute, "Density", 0.1)
    assign_input(distribute, "Density Factor", 1.0)
    assign_input(distribute, "Density Max", 1.0)
    assign_input(distribute, "Seed", 17)
    radius_input = node(tree, "GeometryNodeInputRadius", "Input Radius")
    store_radius = node(tree, "GeometryNodeStoreNamedAttribute", "Store Radius")
    store_radius.domain = "POINT"
    store_radius.data_type = "FLOAT"
    assign_input(store_radius, "Name", "radius_probe")
    points_to_vertices = node(tree, "GeometryNodePointsToVertices", "Points to Vertices")
    join = node(tree, "GeometryNodeJoinGeometry", "Join")

    tree.links.new(calls[0].outputs["Geometry"], store_a.inputs["Geometry"])
    tree.links.new(calls[0].outputs["Top"], store_a.inputs["Value"])
    tree.links.new(store_a.outputs["Geometry"], store_b.inputs["Geometry"])
    tree.links.new(calls[1].outputs["Top"], store_b.inputs["Value"])
    tree.links.new(calls[2].outputs["Geometry"], dead_store.inputs["Geometry"])
    tree.links.new(calls[2].outputs["Top"], dead_store.inputs["Value"])
    tree.links.new(calls[2].outputs["Geometry"], distribute.inputs["Mesh"])
    tree.links.new(distribute.outputs["Points"], store_radius.inputs["Geometry"])
    tree.links.new(radius_input.outputs["Radius"], store_radius.inputs["Value"])
    tree.links.new(store_radius.outputs["Geometry"], points_to_vertices.inputs["Points"])
    tree.links.new(store_b.outputs["Geometry"], join.inputs["Geometry"])
    tree.links.new(points_to_vertices.outputs["Mesh"], join.inputs["Geometry"])
    tree.links.new(join.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree





def nested_top_group_callsite_tree():
    child = new_tree("gn_mesh_ops_nested_top_callsite_child")
    child.interface.new_socket(
        name="Selection",
        in_out="INPUT",
        socket_type="NodeSocketBool",
    )
    child.interface.new_socket(
        name="Top",
        in_out="OUTPUT",
        socket_type="NodeSocketBool",
    )
    child_input = node(child, "NodeGroupInput", "Group Input")
    child_output = node(child, "NodeGroupOutput", "Group Output")
    extrude = node(child, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    subdivide = node(child, "GeometryNodeSubdivideMesh", "Subdivide")
    assign_input(subdivide, "Level", 1)
    child.links.new(child_input.outputs["Geometry"], extrude.inputs["Mesh"])
    child.links.new(child_input.outputs["Selection"], extrude.inputs["Selection"])
    child.links.new(extrude.outputs["Mesh"], subdivide.inputs["Mesh"])
    child.links.new(subdivide.outputs["Mesh"], child_output.inputs["Geometry"])
    child.links.new(extrude.outputs["Top"], child_output.inputs["Top"])

    tree = new_tree("gn_mesh_ops_nested_top_callsite")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    calls = []
    for name, selected in (("Call A", True), ("Call B", False)):
        call = node(tree, "GeometryNodeGroup", name)
        call.node_tree = child
        assign_input(call, "Selection", selected)
        tree.links.new(group_input.outputs["Geometry"], call.inputs["Geometry"])
        calls.append(call)

    store_a = node(tree, "GeometryNodeStoreNamedAttribute", "Store Top A")
    store_a.domain = "FACE"
    store_a.data_type = "BOOLEAN"
    assign_input(store_a, "Name", "top_a")
    store_b = node(tree, "GeometryNodeStoreNamedAttribute", "Store Cross-Call Top")
    store_b.domain = "FACE"
    store_b.data_type = "BOOLEAN"
    assign_input(store_b, "Name", "cross_call_top_b")

    tree.links.new(calls[0].outputs["Geometry"], store_a.inputs["Geometry"])
    tree.links.new(calls[0].outputs["Top"], store_a.inputs["Value"])
    tree.links.new(store_a.outputs["Geometry"], store_b.inputs["Geometry"])
    tree.links.new(calls[1].outputs["Top"], store_b.inputs["Value"])
    tree.links.new(store_b.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def nested_extrude_all_faces_tree():
    child = new_tree("gn_mesh_ops_nested_extrude_child")
    child.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    child.interface.new_socket(
        name="Selection",
        in_out="INPUT",
        socket_type="NodeSocketBool",
    )
    child.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )
    child_input = node(child, "NodeGroupInput", "Group Input")
    child_output = node(child, "NodeGroupOutput", "Group Output")
    extrude = node(child, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    child.links.new(child_input.outputs["Geometry"], extrude.inputs["Mesh"])
    child.links.new(child_input.outputs["Selection"], extrude.inputs["Selection"])
    child.links.new(extrude.outputs["Mesh"], child_output.inputs["Geometry"])

    tree = new_tree("gn_mesh_ops_nested_extrude")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    call = node(tree, "GeometryNodeGroup", "Call")
    call.node_tree = child
    assign_input(call, "Selection", True)
    tree.links.new(group_input.outputs["Geometry"], call.inputs["Geometry"])
    tree.links.new(call.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree
def unrequested_group_top_tree():
    child = new_tree("gn_mesh_ops_unrequested_top_child")
    child.interface.new_socket(
        name="Selection",
        in_out="INPUT",
        socket_type="NodeSocketBool",
    )
    child.interface.new_socket(
        name="Top",
        in_out="OUTPUT",
        socket_type="NodeSocketBool",
    )
    child_input = node(child, "NodeGroupInput", "Group Input")
    child_output = node(child, "NodeGroupOutput", "Group Output")
    extrude = node(child, "GeometryNodeExtrudeMesh", "Extrude")
    extrude.mode = "FACES"
    assign_input(extrude, "Offset Scale", 1.0)
    assign_input(extrude, "Individual", False)
    subdivide = node(child, "GeometryNodeSubdivideMesh", "Subdivide")
    assign_input(subdivide, "Level", 1)
    child.links.new(child_input.outputs["Geometry"], extrude.inputs["Mesh"])
    child.links.new(child_input.outputs["Selection"], extrude.inputs["Selection"])
    child.links.new(extrude.outputs["Mesh"], subdivide.inputs["Mesh"])
    child.links.new(subdivide.outputs["Mesh"], child_output.inputs["Geometry"])
    child.links.new(extrude.outputs["Top"], child_output.inputs["Top"])

    tree = new_tree("gn_mesh_ops_unrequested_top_live_geometry")
    group_input = node(tree, "NodeGroupInput", "Group Input")
    group_output = node(tree, "NodeGroupOutput", "Group Output")
    call = node(tree, "GeometryNodeGroup", "Call")
    call.node_tree = child
    assign_input(call, "Selection", True)
    dead_store = node(tree, "GeometryNodeStoreNamedAttribute", "Unused Top Store")
    dead_store.domain = "FACE"
    dead_store.data_type = "BOOLEAN"
    assign_input(dead_store, "Name", "unused_top")
    tree.links.new(group_input.outputs["Geometry"], call.inputs["Geometry"])
    tree.links.new(call.outputs["Geometry"], dead_store.inputs["Geometry"])
    tree.links.new(call.outputs["Top"], dead_store.inputs["Value"])
    tree.links.new(call.outputs["Geometry"], group_output.inputs["Geometry"])
    return tree


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = make_source()
    input_names = list(ATTRIBUTE_NAMES)
    input_record = mesh_record(
        obj.data,
        input_names,
        include_special_attributes=True,
        require_special_attributes=True,
    )
    topology_names = input_names + [spec[0] for spec in FIELD_SPECS]
    fields = evaluate_tree(
        obj,
        topology_tree(),
        topology_names,
        include_special_attributes=True,
    )
    results = {}
    for name, case in CASES.items():
        attribute_names = [store[0] for store in case.get("store_outputs", [])]
        results[name] = evaluate_tree(
            obj,
            operation_tree(name, case),
            input_names + attribute_names,
            include_special_attributes=True,
        )

    empty_mesh = bpy.data.meshes.new("gn_mesh_ops_empty")
    empty_mesh.from_pydata([], [], [])
    empty_mesh.update(calc_edges=True)
    empty_obj = bpy.data.objects.new("gn_mesh_ops_empty", empty_mesh)
    bpy.context.scene.collection.objects.link(empty_obj)
    empty_input = mesh_record(empty_mesh, [])
    empty_fields = evaluate_tree(
        empty_obj,
        topology_tree(),
        [spec[0] for spec in FIELD_SPECS],
    )
    empty_operations = {}
    for name, case in CASES.items():
        attribute_names = [store[0] for store in case.get("store_outputs", [])]
        empty_operations[name] = evaluate_tree(
            empty_obj,
            operation_tree(name, case),
            attribute_names,
        )
    closed_dual_obj = make_closed_dual_source()
    closed_dual_names = list(CLOSED_DUAL_ATTRIBUTES)
    closed_dual_input = mesh_record(
        closed_dual_obj.data,
        closed_dual_names,
        include_special_attributes=True,
    )
    closed_dual = evaluate_tree(
        closed_dual_obj,
        closed_dual_tree(),
        closed_dual_names,
        include_special_attributes=True,
    )
    top_side_subdivide = evaluate_tree(
        obj,
        top_side_subdivide_tree(),
        input_names + ["top_after_subdivide", "side_after_subdivide"],
        include_special_attributes=True,
    )
    extrude_all_faces = evaluate_tree(
        obj,
        extrude_all_faces_tree(),
        input_names,
        include_special_attributes=True,
    )
    nested_extrude_all_faces = evaluate_tree(
        obj,
        nested_extrude_all_faces_tree(),
        input_names,
        include_special_attributes=True,
    )
    top_after_separate = evaluate_tree(
        obj,
        top_after_separate_tree(),
        input_names + ["top_after_separate"],
        include_special_attributes=True,
    )
    sibling_topology_fields = evaluate_tree(
        obj,
        sibling_topology_fields_tree(),
        input_names + ["area_shared", "angle_shared"],
        include_special_attributes=True,
    )
    single_vertex_obj = make_single_vertex_source()
    single_vertex_input = mesh_record(single_vertex_obj.data, [])
    vertex_top_edge = evaluate_tree(
        single_vertex_obj,
        vertex_top_edge_tree(),
        ["top_on_edge"],
    )
    top_on_original = evaluate_tree(
        obj,
        top_on_original_tree(),
        input_names + ["top_on_original"],
        include_special_attributes=True,
    )
    unconsumed_top_distribute = evaluate_tree(
        obj,
        unconsumed_top_distribute_tree(),
        input_names + ["id", "radius_probe"],
        include_special_attributes=True,
    )
    radius = unconsumed_top_distribute["attributes"].pop("radius_probe", None)
    if radius is None:
        raise RuntimeError("distributed point radius did not survive the mesh oracle conversion")
    unconsumed_top_distribute["attributes"]["radius"] = radius
    unconsumed_top_distribute_dead_consumer = evaluate_tree(
        obj,
        unconsumed_top_distribute_dead_consumer_tree(),
        input_names,
        include_special_attributes=True,
    )
    nested_top_group = evaluate_tree(
        obj,
        nested_top_group_tree(),
        input_names + ["id", "radius_probe", "top_a", "cross_call_top_b"],
        include_special_attributes=True,
    )
    radius = nested_top_group["attributes"].pop("radius_probe", None)
    if radius is None:
        raise RuntimeError("nested group point radius did not survive the mesh oracle conversion")
    nested_top_group["attributes"]["radius"] = radius
    nested_top_group_callsite = evaluate_tree(
        obj,
        nested_top_group_callsite_tree(),
        input_names + ["top_a", "cross_call_top_b"],
        include_special_attributes=True,
    )
    unrequested_group_top = evaluate_tree(
        obj,
        unrequested_group_top_tree(),
        input_names,
        include_special_attributes=True,
    )
    payload = {
        "blender_version": bpy.app.version_string,
        "input": input_record,
        "topology_fields": fields,
        "empty_input": empty_input,
        "empty_topology_fields": empty_fields,
        "empty_operations": empty_operations,
        "closed_dual_input": closed_dual_input,
        "closed_dual": closed_dual,
        "operations": results,
        "top_side_subdivide": top_side_subdivide,
        "extrude_all_faces": extrude_all_faces,
        "nested_extrude_all_faces": nested_extrude_all_faces,
        "single_vertex_input": single_vertex_input,
        "vertex_extrude_top_on_edge": vertex_top_edge,
        "top_on_original": top_on_original,
        "unconsumed_top_distribute": unconsumed_top_distribute,
        "unconsumed_top_distribute_dead_consumer": unconsumed_top_distribute_dead_consumer,
        "nested_top_group": nested_top_group,
        "nested_top_group_callsite": nested_top_group_callsite,
        "top_after_separate": top_after_separate,
        "unrequested_group_top": unrequested_group_top,
        "sibling_topology_fields": sibling_topology_fields,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(results)} operations, {len(FIELD_SPECS)} topology fields)")


main()
