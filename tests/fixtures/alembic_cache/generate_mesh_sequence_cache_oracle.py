"""Blender 5.2 oracle for the native Alembic MeshSequenceCache (PR18 section Q).

Regenerate from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/alembic_cache/generate_mesh_sequence_cache_oracle.py

The script writes, next to itself:

- `animated.abc`: asymmetric animated PolyMesh `/Animated` (mixed quad/tri faces, point and corner
  colors, corner custom normals, a `velocity` point attribute from Geometry Nodes, and a non-trivial
  object transform). Frame k holds exactly the positions `P[k]` through shape key `K<k>`.
- `topo.abc`: `/TopoSource`, a cube whose faces are triangulated from frame 3 on (topology change).
- `uv_only.abc`: static `/UvOnly` PolyMesh with one asymmetric face-corner UV layer for the
  independent UV-only MeshSequenceCache case.
- `seq.000<k>.abc`: one single-frame file per frame of `/Animated` for `is_sequence` cases.
- `reordered.000<k>.abc`: same-size file-sequence samples with reversed face-corner connectivity and
  asymmetric UVs, exercising the cache topology-change reread.
- `reordered_animated.abc`: a non-sequence archive whose `/Reordered` face-corner connectivity
  reverses the first face at frame 2, with asymmetric UVs for the animated-topology regression.
- `oracle.json`: per case, the CacheFile settings, the modifier settings, and the evaluated mesh of
  each frame (local positions, faces in Blender loop order, corner normals, custom-normal flag and
  every non-internal attribute with full values).

Every case builds a fresh receiver object whose own mesh is the input to the modifier, so the topology
branch (same connectivity, different counts, or different connectivity) is set by the receiver.
"""

import json
import math
import os
import sys

import bpy

OUT = os.path.dirname(os.path.abspath(__file__))
FPS = 24
# Blender writes each Alembic object as an Xform with the PolyMesh as its child of the same name.
ANIMATED_PATH = "/Animated/Animated"
TOPO_PATH = "/TopoSource/TopoSource"
UV_ONLY_PATH = "/UvOnly/UvOnly"
FRAMES = [1, 2, 3, 4]

REST = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (2.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (-0.5, 1.5, 0.5),
    (1.5, 2.5, -0.5),
]
FACES = [(0, 1, 2, 3), (0, 3, 4), (3, 2, 5, 4)]
REORDERED_PATH = "/Reordered/Reordered"
REORDERED_FACES = [tuple(reversed(FACES[0])), *FACES[1:]]
DELTA = [
    (0.10, 0.00, 0.20),
    (0.00, 0.30, 0.00),
    (-0.20, 0.10, 0.05),
    (0.05, -0.10, 0.30),
    (0.40, 0.00, -0.10),
    (0.00, 0.20, 0.25),
]
CUBE_VERTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]
CUBE_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def positions_at(frame):
    """Exact per-frame positions `P[frame]` of `/Animated` in its own (object) space."""
    return [
        (
            x + frame * dx + 0.01 * frame * frame * (index + 1),
            y + frame * dy - 0.005 * frame * (index + 2),
            z + frame * dz,
        )
        for index, ((x, y, z), (dx, dy, dz)) in enumerate(zip(REST, DELTA))
    ]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.frame_start = FRAMES[0]
    scene.frame_end = FRAMES[-1]
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "CONSTANT"
    return scene


def new_object(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_velocity_and_topology_nodes(obj, name, triangulate_from=None):
    """Geometry Nodes that store `velocity = Position * 0.5 + (0, 0.1 * Frame, 0)` and optionally
    triangulate once `Frame > triangulate_from`."""
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    time = nodes.new("GeometryNodeInputSceneTime")
    position = nodes.new("GeometryNodeInputPosition")
    half = nodes.new("ShaderNodeVectorMath")
    half.operation = "SCALE"
    half.inputs["Scale"].default_value = 0.5
    links.new(position.outputs["Position"], half.inputs[0])
    frame_scaled = nodes.new("ShaderNodeMath")
    frame_scaled.operation = "MULTIPLY"
    frame_scaled.inputs[1].default_value = 0.1
    links.new(time.outputs["Frame"], frame_scaled.inputs[0])
    offset = nodes.new("ShaderNodeCombineXYZ")
    links.new(frame_scaled.outputs["Value"], offset.inputs["Y"])
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    links.new(half.outputs["Vector"], add.inputs[0])
    links.new(offset.outputs["Vector"], add.inputs[1])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT_VECTOR"
    store.domain = "POINT"
    store.inputs["Name"].default_value = "velocity"
    links.new(add.outputs["Vector"], store.inputs["Value"])
    links.new(group_in.outputs["Geometry"], store.inputs["Geometry"])
    if triangulate_from is not None:
        compare = nodes.new("FunctionNodeCompare")
        compare.data_type = "FLOAT"
        compare.operation = "GREATER_THAN"
        compare.inputs["B"].default_value = triangulate_from
        links.new(time.outputs["Frame"], compare.inputs["A"])
        triangulate = nodes.new("GeometryNodeTriangulate")
        switch = nodes.new("GeometryNodeSwitch")
        switch.input_type = "GEOMETRY"
        links.new(compare.outputs["Result"], switch.inputs["Switch"])
        links.new(store.outputs["Geometry"], switch.inputs["False"])
        links.new(store.outputs["Geometry"], triangulate.inputs["Mesh"])
        links.new(triangulate.outputs["Mesh"], switch.inputs["True"])
        last = switch
        last_output = switch.outputs["Output"]
    else:
        last_output = store.outputs["Geometry"]
    links.new(last_output, group_out.inputs["Geometry"])
    modifier = obj.modifiers.new("Velocity", "NODES")
    modifier.node_group = tree
    return modifier


def build_animated():
    obj = new_object("Animated", [REST[i] for i in range(len(REST))], FACES)
    mesh = obj.data
    obj.shape_key_add(name="Basis", from_mix=False)
    for frame in FRAMES:
        key = obj.shape_key_add(name=f"K{frame}", from_mix=False)
        for index, point in enumerate(positions_at(frame)):
            key.data[index].co = point
        key.value = 0.0
        for value_frame in FRAMES:
            key.value = 1.0 if value_frame == frame else 0.0
            key.keyframe_insert("value", frame=value_frame)
    # Corner custom normals (asymmetric, not normalized on input) and colors.
    loops = len(mesh.loops)
    normals = []
    for loop in range(loops):
        vector = (0.3 + 0.1 * loop, -0.2 * loop + 0.5, 0.7 - 0.05 * loop)
        length = math.sqrt(sum(c * c for c in vector))
        normals.append(tuple(c / length for c in vector))
    mesh.normals_split_custom_set(normals)
    # Blender's Alembic writer exports only BYTE_COLOR corner layers (`vcolors`); the point and
    # float layers below are present as negative probes and must not appear in the evaluated output.
    point_color = mesh.color_attributes.new("PointCol", "FLOAT_COLOR", "POINT")
    for index, item in enumerate(point_color.data):
        item.color = (0.1 * index, 1.0 - 0.15 * index, 0.5, 1.0)
    float_corner = mesh.color_attributes.new("FloatCornerProbe", "FLOAT_COLOR", "CORNER")
    for loop, item in enumerate(float_corner.data):
        item.color = (0.5, 0.25, 0.75, 1.0)
    corner_color = mesh.color_attributes.new("CornerCol", "BYTE_COLOR", "CORNER")
    for loop, item in enumerate(corner_color.data):
        item.color = (0.05 * loop, 0.3, 0.9 - 0.05 * loop, 0.5 + 0.05 * loop)
    byte_point = mesh.color_attributes.new("BytePointProbe", "BYTE_COLOR", "POINT")
    for index, item in enumerate(byte_point.data):
        item.color = (0.2, 0.4 * index / 5.0, 0.6, 1.0)
    weights = mesh.attributes.new("weight_probe", "FLOAT", "POINT")
    for index, item in enumerate(weights.data):
        item.value = 0.25 * index
    corner_probe = mesh.attributes.new("corner_probe", "FLOAT", "CORNER")
    for index, item in enumerate(corner_probe.data):
        item.value = 0.5 + 0.125 * index
    obj.location = (1.0, -2.0, 0.5)
    obj.rotation_euler = (0.3, -0.4, 1.1)
    obj.scale = (1.5, 0.75, 1.0)
    add_velocity_and_topology_nodes(obj, "AnimatedVelocity")
    return obj


def build_uv_only():
    obj = new_object("UvOnly", REST, FACES)
    uv_values = [
        ("Base", [
            (0.137, 0.223), (0.619, 0.357), (0.283, 0.881), (0.947, 0.569),
            (0.731, 0.163), (0.194, 0.907), (0.853, 0.421),
            (0.367, 0.689), (0.521, 0.241), (0.063, 0.773), (0.917, 0.487),
        ]),
        ("Light", [
            (0.731, 0.163), (0.194, 0.907), (0.853, 0.421), (0.367, 0.689),
            (0.137, 0.223), (0.619, 0.357), (0.283, 0.881), (0.947, 0.569),
            (0.861, 0.147), (0.293, 0.953), (0.742, 0.312),
        ]),
    ]
    for name, corners in uv_values:
        layer = obj.data.uv_layers.new(name=name)
        for item, uv in zip(layer.data, corners):
            item.uv = uv
    obj.data.uv_layers.active_index = obj.data.uv_layers.find("Light")
    obj.data.uv_layers.active_render_index = 0
    return obj


def build_reordered_source(faces):
    obj = new_object("Reordered", REST, faces)
    layer = obj.data.uv_layers.new(name="UVMap")
    rows = [
        [(0.137, 0.223), (0.619, 0.357), (0.283, 0.881), (0.947, 0.569)],
        [(0.731, 0.163), (0.194, 0.907), (0.853, 0.421)],
        [(0.367, 0.689), (0.521, 0.241), (0.063, 0.773), (0.917, 0.487)],
    ]
    loop = 0
    for face_index, face in enumerate(faces):
        uv_by_vertex = dict(zip(FACES[face_index], rows[face_index]))
        for vertex in face:
            layer.data[loop].uv = uv_by_vertex[vertex]
            loop += 1
    return obj


def add_reordered_topology_nodes(obj):
    """Reverse only the first face after frame 1, keeping mesh dimensions unchanged."""
    tree = bpy.data.node_groups.new("ReorderedTopology", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    index = nodes.new("GeometryNodeInputIndex")
    first_face = nodes.new("FunctionNodeCompare")
    first_face.data_type = "INT"
    first_face.operation = "EQUAL"
    first_face.inputs["B"].default_value = 0
    links.new(index.outputs["Index"], first_face.inputs["A"])
    flip = nodes.new("GeometryNodeFlipFaces")
    links.new(group_in.outputs["Geometry"], flip.inputs["Mesh"])
    links.new(first_face.outputs["Result"], flip.inputs["Selection"])
    time = nodes.new("GeometryNodeInputSceneTime")
    after_first_frame = nodes.new("FunctionNodeCompare")
    after_first_frame.data_type = "FLOAT"
    after_first_frame.operation = "GREATER_THAN"
    after_first_frame.inputs["B"].default_value = 1.5
    links.new(time.outputs["Frame"], after_first_frame.inputs["A"])
    switch = nodes.new("GeometryNodeSwitch")
    switch.input_type = "GEOMETRY"
    links.new(after_first_frame.outputs["Result"], switch.inputs["Switch"])
    links.new(group_in.outputs["Geometry"], switch.inputs["False"])
    links.new(flip.outputs["Mesh"], switch.inputs["True"])
    links.new(switch.outputs["Output"], group_out.inputs["Geometry"])
    modifier = obj.modifiers.new("AnimatedTopology", "NODES")
    modifier.node_group = tree


def export_selected(obj, path, start, end, *, uvs=False):
    # shortcut: existing payloads stay unchanged; remove one deliberately when its Blender input changes.
    if os.path.isfile(path):
        return
    # The exporter samples the current frame for single-frame ranges, so start there explicitly.
    bpy.context.scene.frame_set(start)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.alembic_export(
        filepath=path,
        selected=True,
        start=start,
        end=end,
        uvs=uvs,
        normals=True,
        vcolors=True,
        orcos=False,
        face_sets=False,
        global_scale=1.0,
        triangulate=False,
        flatten=False,
        use_instancing=False,
        export_custom_properties=False,
        export_hair=False,
        export_particles=False,
        init_scene_frame_range=False,
        as_background_job=False,
    )


def attribute_payload(attribute):
    values = []
    for item in attribute.data:
        if hasattr(item, "vector"):
            values.append([float(c) for c in item.vector])
        elif hasattr(item, "color"):
            values.append([float(c) for c in item.color])
        elif hasattr(item, "value"):
            value = item.value
            values.append([float(c) for c in value] if hasattr(value, "__len__") else value)
        else:
            raise RuntimeError(f"unsupported attribute {attribute.name} item {item}")
    return {"domain": attribute.domain, "type": attribute.data_type, "values": values}


def evaluate_frame(obj, frame):
    scene = bpy.context.scene
    scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        corner_normals = [[float(c) for c in normal.vector] for normal in mesh.corner_normals]
        attributes = {}
        for attribute in mesh.attributes:
            if attribute.name.startswith(".") or attribute.name == "position":
                continue
            attributes[attribute.name] = attribute_payload(attribute)
        return {
            "frame": frame,
            "positions": [[float(c) for c in v.co] for v in mesh.vertices],
            "faces": [list(p.vertices) for p in mesh.polygons],
            "smooth_faces": [bool(p.use_smooth) for p in mesh.polygons],
            "corner_normals": corner_normals,
            "has_custom_normals": bool(mesh.has_custom_normals),
            "attributes": attributes,
        }
    finally:
        evaluated.to_mesh_clear()


def cache_settings(cache):
    return {
        "filepath": cache.filepath,
        "is_sequence": cache.is_sequence,
        "override_frame": cache.override_frame,
        "frame": cache.frame,
        "frame_offset": cache.frame_offset,
        "forward_axis": cache.forward_axis,
        "up_axis": cache.up_axis,
        "scale": cache.scale,
        "velocity_name": cache.velocity_name,
        "velocity_unit": cache.velocity_unit,
    }


def open_cache_file(path):
    """`CacheFile` datablocks are created by the Alembic open operator in background mode."""
    name = os.path.basename(path)
    if name in bpy.data.cache_files:
        bpy.data.batch_remove((bpy.data.cache_files[name],))
    result = bpy.ops.cachefile.open(filepath=path)
    if result != {"FINISHED"}:
        raise RuntimeError(f"cachefile.open failed for {path}: {result}")
    return bpy.data.cache_files[name]


# `cachefile.open` leaves scale 0, empty velocity_name and FRAME units in background mode, so every
# case states the Blender `cache_file_init_data` values it relies on explicitly.
BASE_CACHE = {"velocity_name": ".velocities", "velocity_unit": "SECOND", "scale": 1.0}


def run_case(case_id, receiver_obj, cache_path, object_path, cache_kwargs, modifier_kwargs, frames):
    cache = open_cache_file(cache_path)
    for key, value in {**BASE_CACHE, **cache_kwargs}.items():
        setattr(cache, key, value)
    modifier = receiver_obj.modifiers.new("MeshSequenceCache", "MESH_SEQUENCE_CACHE")
    modifier.cache_file = cache
    modifier.object_path = object_path
    for key, value in modifier_kwargs.items():
        if key == "read_data":
            modifier.read_data = set(value)
        else:
            setattr(modifier, key, value)
    frames_out = [evaluate_frame(receiver_obj, frame) for frame in frames]
    result = {
        "id": case_id,
        "cache_file": cache_settings(cache),
        "modifier": {
            "object_path": modifier.object_path,
            "read_data": sorted(modifier.read_data),
            "use_vertex_interpolation": modifier.use_vertex_interpolation,
            "velocity_scale": modifier.velocity_scale,
        },
        "modifier_error": getattr(modifier, "error", None),
        "frames": frames_out,
    }
    receiver_obj.modifiers.remove(modifier)
    bpy.data.batch_remove((cache,))
    return result


def main():
    oracle_path = os.path.join(OUT, "oracle.json")
    existing = None
    if os.path.isfile(oracle_path):
        with open(oracle_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("blender_version") != bpy.app.version_string:
            raise RuntimeError("existing Alembic oracle uses a different Blender version")
    scene = reset_scene()
    animated = build_animated()
    topo = new_object("TopoSource", CUBE_VERTS, CUBE_FACES)
    add_velocity_and_topology_nodes(topo, "TopoGraph", triangulate_from=2.5)
    uv_only = build_uv_only()

    export_selected(animated, os.path.join(OUT, "animated.abc"), FRAMES[0], FRAMES[-1])
    export_selected(topo, os.path.join(OUT, "topo.abc"), FRAMES[0], FRAMES[-1])
    for frame in FRAMES:
        export_selected(animated, os.path.join(OUT, f"seq.{frame:04d}.abc"), frame, frame)
    export_selected(
        uv_only,
        os.path.join(OUT, "uv_only.abc"),
        FRAMES[0],
        FRAMES[-1],
        uvs=True,
    )
    for frame, faces in [(1, FACES), (2, REORDERED_FACES)]:
        reordered = build_reordered_source(faces)
        mesh = reordered.data
        export_selected(
            reordered,
            os.path.join(OUT, f"reordered.{frame:04d}.abc"),
            frame,
            frame,
            uvs=True,
        )
        bpy.data.objects.remove(reordered, do_unlink=True)
        bpy.data.meshes.remove(mesh)
    reordered_animated = build_reordered_source(FACES)
    add_reordered_topology_nodes(reordered_animated)
    mesh = reordered_animated.data
    export_selected(
        reordered_animated,
        os.path.join(OUT, "reordered_animated.abc"),
        FRAMES[0],
        FRAMES[1],
        uvs=True,
    )
    bpy.data.objects.remove(reordered_animated, do_unlink=True)
    bpy.data.meshes.remove(mesh)

    # Remove the source objects; receivers are created below from fresh meshes.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    animated_path = os.path.join(OUT, "animated.abc")
    topo_path = os.path.join(OUT, "topo.abc")
    seq_path = os.path.join(OUT, "seq.0001.abc")
    uv_only_path = os.path.join(OUT, "uv_only.abc")
    reordered_path = os.path.join(OUT, "reordered.0001.abc")
    reordered_animated_path = os.path.join(OUT, "reordered_animated.abc")

    def receiver(name, verts, faces):
        return new_object(name, verts, faces)

    same_topology = lambda: receiver("RecvSame", REST, FACES)
    cube_topology = lambda: receiver("RecvCube", CUBE_VERTS, CUBE_FACES)

    def same_topology_with_uv():
        obj = same_topology()
        layer = obj.data.uv_layers.new(name="UVMap")
        for index, item in enumerate(layer.data):
            item.uv = (20.0 + index, -10.0 - index)
        return obj


    cases = []
    cases.append(run_case(
        "default_same_topology", same_topology(), animated_path, ANIMATED_PATH,
        {}, {}, [0, 1, 2, 3, 4, 5],
    ))
    cases.append(run_case(
        "frame_offset_positive", same_topology(), animated_path, ANIMATED_PATH,
        {"frame_offset": 1.0}, {}, [1, 2, 3, 4, 5],
    ))
    cases.append(run_case(
        "frame_offset_fractional", same_topology(), animated_path, ANIMATED_PATH,
        {"frame_offset": -0.5}, {"use_vertex_interpolation": True}, [1, 2, 3, 4],
    ))
    cases.append(run_case(
        "override_frame_fractional_interpolated", same_topology(), animated_path, ANIMATED_PATH,
        {"override_frame": True, "frame": 2.5}, {"use_vertex_interpolation": True}, [1, 2, 3, 4],
    ))
    cases.append(run_case(
        "override_frame_fractional_no_interpolation", same_topology(), animated_path, ANIMATED_PATH,
        {"override_frame": True, "frame": 2.5}, {"use_vertex_interpolation": False}, [1, 4],
    ))
    cases.append(run_case(
        "velocity_unit_frame", same_topology(), animated_path, ANIMATED_PATH,
        {"velocity_unit": "FRAME"}, {"velocity_scale": 0.5}, [1, 2],
    ))
    cases.append(run_case(
        "velocity_unit_second", same_topology(), animated_path, ANIMATED_PATH,
        {"velocity_unit": "SECOND"}, {"velocity_scale": 0.5}, [1, 2],
    ))
    cases.append(run_case(
        "velocity_name_missing", same_topology(), animated_path, ANIMATED_PATH,
        {"velocity_name": "velocities"}, {}, [1],
    ))
    cases.append(run_case(
        "velocity_name_empty", same_topology(), animated_path, ANIMATED_PATH,
        {"velocity_name": ""}, {}, [1],
    ))
    cases.append(run_case(
        "velocity_scale_zero", same_topology(), animated_path, ANIMATED_PATH,
        {}, {"velocity_scale": 0.0}, [1],
    ))
    cases.append(run_case(
        "scale_two_ignored", same_topology(), animated_path, ANIMATED_PATH,
        {"scale": 2.0}, {}, [1, 3],
    ))
    cases.append(run_case(
        "read_vert_only", same_topology(), animated_path, ANIMATED_PATH,
        {}, {"read_data": ["VERT"]}, [1, 3],
    ))
    cases.append(run_case(
        "read_poly_only", same_topology(), animated_path, ANIMATED_PATH,
        {}, {"read_data": ["POLY"]}, [1, 3],
    ))
    cases.append(run_case(
        "read_no_color", same_topology(), animated_path, ANIMATED_PATH,
        {}, {"read_data": ["VERT", "POLY"]}, [1, 3],
    ))
    cases.append(run_case(
        "read_uv_only", same_topology(), uv_only_path, UV_ONLY_PATH,
        {}, {"read_data": ["UV"]}, [1, 3],
    ))
    cases.append(run_case(
        "read_attributes_only", same_topology(), animated_path, ANIMATED_PATH,
        {}, {"read_data": ["ATTRIBUTES"]}, [1, 3],
    ))
    cases.append(run_case(
        "topology_changed_cube_input", cube_topology(), animated_path, ANIMATED_PATH,
        {}, {}, [1, 2],
    ))
    cases.append(run_case(
        "topology_changed_cube_input_vert_only", cube_topology(), animated_path, ANIMATED_PATH,
        {}, {"read_data": ["VERT"]}, [1, 2],
    ))
    cases.append(run_case(
        "topology_face_change_full_read", cube_topology(), topo_path, TOPO_PATH,
        {}, {}, [1, 2, 3, 4],
    ))
    cases.append(run_case(
        "topology_face_change_vert_only", cube_topology(), topo_path, TOPO_PATH,
        {}, {"read_data": ["VERT"]}, [1, 3],
    ))
    cases.append(run_case(
        "sequence_same_size_reordered_connectivity", same_topology_with_uv(),
        reordered_path, REORDERED_PATH, {"is_sequence": True},
        {"read_data": ["VERT"]}, [1, 2],
    ))
    cases.append(run_case(
        "is_sequence_per_frame_files", same_topology(), seq_path, ANIMATED_PATH,
        {"is_sequence": True}, {}, [1, 2, 3, 4],
    ))
    cases.append(run_case(
        "sequence_offset_ignored", same_topology(), seq_path, ANIMATED_PATH,
        {"is_sequence": True, "frame_offset": 10.0}, {}, [1, 2],
    ))
    cases.append(run_case(
        "sequence_override_frame", same_topology(), seq_path, ANIMATED_PATH,
        {"is_sequence": True, "override_frame": True, "frame": 3.0}, {}, [1, 2],
    ))
    cases.append(run_case(
        "is_sequence_missing_file", same_topology(), seq_path, ANIMATED_PATH,
        {"is_sequence": True}, {}, [5],
    ))
    cases.append(run_case(
        "axis_settings_mesh_data_unchanged", same_topology(), animated_path, ANIMATED_PATH,
        {"forward_axis": "POS_Y", "up_axis": "POS_Z"}, {}, [1, 3],
    ))
    cases.append(run_case(
        "nonsequence_animated_same_size_reordered_connectivity",
        same_topology_with_uv(), reordered_animated_path, REORDERED_PATH,
        {"is_sequence": False}, {"read_data": ["VERT"]}, [1, 2],
    ))

    bpy.ops.wm.alembic_import(filepath=animated_path, as_background_job=False)
    imported_cache_modifier = next(
        modifier
        for obj in bpy.data.objects
        for modifier in obj.modifiers
        if modifier.type == "MESH_SEQUENCE_CACHE"
    )
    defaults = {
        "cache_file": cache_settings(imported_cache_modifier.cache_file),
        "modifier": {
            "read_data": sorted(imported_cache_modifier.read_data),
            "use_vertex_interpolation": imported_cache_modifier.use_vertex_interpolation,
            "velocity_scale": imported_cache_modifier.velocity_scale,
            "frame_offset_exists": hasattr(imported_cache_modifier, "frame_offset"),
            "scale_exists": hasattr(imported_cache_modifier, "scale"),
            "override_frame_exists": hasattr(imported_cache_modifier, "override_frame"),
        },
    }

    payload = {
        "blender_version": bpy.app.version_string,
        "fps": FPS,
        "fps_base": 1.0,
        "frames": FRAMES,
        "defaults": defaults,
        "cases": cases,
    }
    if existing is not None:
        existing_cases = existing.get("cases", [])
        existing_ids = {case["id"] for case in existing_cases}
        for case in cases:
            if case["id"] not in existing_ids:
                existing_cases.append(case)
                existing_ids.add(case["id"])
        existing["cases"] = existing_cases
        payload = existing
    with open(oracle_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True, default=lambda value: list(value))
        handle.write("\n")
    print("ORACLE_WRITTEN", oracle_path, len(payload["cases"]))


if __name__ == "__main__":
    sys.exit(main())
