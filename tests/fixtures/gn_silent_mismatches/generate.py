"""Generate Blender 5.2.2 geometry-node cases for silent-mismatch parity tests.

Run with Blender in background mode and pass an output directory after ``--``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy


POISSON_OBJECT = "GNPoissonInstances"
SUBDIVISION_OBJECT = "GNCreasedSubdivision"
SIMULATION_OBJECT = "GNSimulationDeltaTime"
POISSON_CANDIDATE_OBJECT = "GNPoissonCandidates"
RANDOM_FLOAT_OBJECT = "GNRandomValueFloat"
RANDOM_INT_OBJECT = "GNRandomValueInt"
RANDOM_BOOLEAN_OBJECT = "GNRandomValueBoolean"
RANDOM_VECTOR_OBJECT = "GNRandomValueVector"
RANDOM_POISSON_ID_OBJECT = "GNRandomValuePoissonID"
RANDOM_SELECTION_OBJECT = "GNRandomSelectionFalse"
SMOOTH_RANDOM_OBJECT = "GNSmoothRandomDistribution"
ZERO_CREASE_SUBDIVISION_OBJECT = "GNZeroCreaseThenSubdivision"
NEGATIVE_CREASE_SUBDIVISION_OBJECT = "GNNegativeCreaseThenSubdivision"
STORE_READ_INT_OBJECT = "GNStoreReadInt"
STORE_READ_BOOLEAN_OBJECT = "GNStoreReadBoolean"
STORE_READ_QUATERNION_OBJECT = "GNStoreReadQuaternion"
OBJECT_NAMES = (
    POISSON_OBJECT,
    SUBDIVISION_OBJECT,
    SIMULATION_OBJECT,
    POISSON_CANDIDATE_OBJECT,
    RANDOM_FLOAT_OBJECT,
    RANDOM_INT_OBJECT,
    RANDOM_BOOLEAN_OBJECT,
    RANDOM_VECTOR_OBJECT,
    RANDOM_POISSON_ID_OBJECT,
    RANDOM_SELECTION_OBJECT,
    SMOOTH_RANDOM_OBJECT,
    ZERO_CREASE_SUBDIVISION_OBJECT,
    NEGATIVE_CREASE_SUBDIVISION_OBJECT,
    STORE_READ_INT_OBJECT,
    STORE_READ_BOOLEAN_OBJECT,
    STORE_READ_QUATERNION_OBJECT,
)


def mesh_object(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]):
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def geometry_group(obj, name: str):
    group = bpy.data.node_groups.new(name, "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes = group.nodes
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    modifier = obj.modifiers.new(name, "NODES")
    modifier.node_group = group
    return group, group_input, group_output


def evaluated_vertices(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        world = evaluated.matrix_world
        return [
            [round(float(value), 9) for value in world @ vertex.co]
            for vertex in mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()


def evaluated_int_attribute(obj, depsgraph, name):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.attributes.get(name)
        if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT":
            raise RuntimeError(f"evaluated {name} INT point attribute is missing")
        return [int(item.value) for item in attribute.data]
    finally:
        evaluated.to_mesh_clear()

def evaluated_quaternion_attribute(obj, depsgraph, name):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        attribute = mesh.attributes.get(name)
        if (
            attribute is None
            or attribute.domain != "POINT"
            or attribute.data_type != "QUATERNION"
        ):
            raise RuntimeError(f"evaluated {name} QUATERNION point attribute is missing")
        return [
            [float(item.value[1]), float(item.value[2]), float(item.value[3]), float(item.value[0])]
            for item in attribute.data
        ]
    finally:
        evaluated.to_mesh_clear()


def build_quaternion_attribute_case():
    obj = mesh_object(
        STORE_READ_QUATERNION_OBJECT,
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        [],
    )
    source = obj.data.attributes.new(name="source_quaternion", type="QUATERNION", domain="POINT")
    source.data[0].value = (2.0, 0.0, 0.0, 0.0)
    source.data[1].value = (0.0, 0.0, 0.0, 0.0)
    return {
        "data_type": "QUATERNION",
        "attribute_name": "source_quaternion",
        "domain": "POINT",
        "point_count": 2,
    }


def build_poisson_case():
    surface = mesh_object(
        POISSON_OBJECT,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [(0, 1, 2, 3)],
    )
    group, group_input, group_output = geometry_group(surface, "GN Poisson Nondefaults")
    nodes, links = group.nodes, group.links
    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.inputs["Density"].default_value = 23.0
    distribute.distribute_method = "POISSON"
    distribute.inputs["Density Factor"].default_value = 0.35
    distribute.inputs["Density Max"].default_value = 6.0
    distribute.inputs["Distance Min"].default_value = 0.45
    distribute.inputs["Seed"].default_value = 17
    cube = nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.2, 0.2, 0.2)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.inputs["Depth"].default_value = 5
    links.new(group_input.outputs["Geometry"], distribute.inputs["Mesh"])
    links.new(distribute.outputs["Points"], instance.inputs["Points"])
    links.new(cube.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])

def build_poisson_candidate_case():
    surface = mesh_object(
        POISSON_CANDIDATE_OBJECT,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [(0, 1, 2, 3)],
    )
    group, group_input, group_output = geometry_group(surface, "GN Poisson Candidate Samples")
    nodes, links = group.nodes, group.links
    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.inputs["Density"].default_value = 23.0
    distribute.distribute_method = "POISSON"
    distribute.inputs["Density Factor"].default_value = 1.0
    distribute.inputs["Density Max"].default_value = 6.0
    distribute.inputs["Distance Min"].default_value = 0.0
    distribute.inputs["Seed"].default_value = 17
    cube = nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.2, 0.2, 0.2)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.inputs["Depth"].default_value = 5
    links.new(group_input.outputs["Geometry"], distribute.inputs["Mesh"])
    links.new(distribute.outputs["Points"], instance.inputs["Points"])
    links.new(cube.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])

def build_random_value_case(name, data_type, minimum, maximum, seed, probability=None):
    obj = mesh_object(name, [], [])
    group, _, group_output = geometry_group(obj, f"GN {name}")
    nodes, links = group.nodes, group.links
    mesh_line = nodes.new("GeometryNodeMeshLine")
    mesh_line.inputs["Count"].default_value = 8
    mesh_line.inputs["Offset"].default_value = (0.0, 0.0, 0.0)
    random = nodes.new("FunctionNodeRandomValue")
    random.data_type = data_type
    random.inputs["Seed"].default_value = seed
    if data_type == "BOOLEAN":
        random.inputs["Probability"].default_value = probability
    else:
        random.inputs["Min"].default_value = minimum
        random.inputs["Max"].default_value = maximum
    if data_type == "INT":
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type = "INT"
        store.domain = "POINT"
        store.inputs["Name"].default_value = "oracle_random_int"
        links.new(mesh_line.outputs["Mesh"], store.inputs["Geometry"])
        links.new(random.outputs["Value"], store.inputs["Value"])
        links.new(store.outputs["Geometry"], group_output.inputs["Geometry"])
    else:
        set_position = nodes.new("GeometryNodeSetPosition")
        links.new(mesh_line.outputs["Mesh"], set_position.inputs["Geometry"])
        if data_type == "BOOLEAN":
            set_position.inputs["Offset"].default_value = (0.0, 0.0, 1.0)
            links.new(random.outputs["Value"], set_position.inputs["Selection"])
        elif data_type == "FLOAT":
            combine = nodes.new("ShaderNodeCombineXYZ")
            links.new(random.outputs["Value"], combine.inputs["X"])
            links.new(combine.outputs["Vector"], set_position.inputs["Offset"])
        else:
            links.new(random.outputs["Value"], set_position.inputs["Offset"])
        links.new(set_position.outputs["Geometry"], group_output.inputs["Geometry"])


def build_random_poisson_id_case():
    obj = mesh_object(
        RANDOM_POISSON_ID_OBJECT,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [(0, 1, 2, 3)],
    )
    group, group_input, group_output = geometry_group(obj, "GN Random Value Unlinked ID")
    nodes, links = group.nodes, group.links
    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.inputs["Density"].default_value = 23.0
    distribute.distribute_method = "POISSON"
    distribute.inputs["Density Factor"].default_value = 1.0
    distribute.inputs["Density Max"].default_value = 6.0
    distribute.inputs["Distance Min"].default_value = 0.0
    distribute.inputs["Seed"].default_value = 17
    random = nodes.new("FunctionNodeRandomValue")
    random.data_type = "FLOAT_VECTOR"
    random.inputs["Min"].default_value = (-0.3, -0.4, -0.5)
    random.inputs["Max"].default_value = (0.3, 0.4, 0.5)
    random.inputs["Seed"].default_value = 29
    set_position = nodes.new("GeometryNodeSetPosition")
    cube = nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.2, 0.2, 0.2)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(group_input.outputs["Geometry"], distribute.inputs["Mesh"])
    links.new(distribute.outputs["Points"], set_position.inputs["Geometry"])
    links.new(random.outputs["Value"], set_position.inputs["Offset"])
    links.new(set_position.outputs["Geometry"], instance.inputs["Points"])
    links.new(cube.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])
    return random.inputs["ID"].is_linked


def build_random_selection_case():
    obj = mesh_object(
        RANDOM_SELECTION_OBJECT,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [(0, 1, 2, 3)],
    )
    group, group_input, group_output = geometry_group(obj, "GN Random Distribution Selection False")
    nodes, links = group.nodes, group.links
    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.distribute_method = "RANDOM"
    distribute.inputs["Density"].default_value = 8.0
    distribute.inputs["Selection"].default_value = False
    distribute.inputs["Seed"].default_value = 17
    cube = nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.2, 0.2, 0.2)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    links.new(group_input.outputs["Geometry"], distribute.inputs["Mesh"])
    links.new(distribute.outputs["Points"], instance.inputs["Points"])
    links.new(cube.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])


def build_smooth_random_case():
    surface = mesh_object(
        SMOOTH_RANDOM_OBJECT,
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
        [(0, 1, 2, 3)],
    )
    for polygon in surface.data.polygons:
        polygon.use_smooth = True
    surface.data.normals_split_custom_set(
        [(0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.0, 0.2, 1.0), (-0.2, 0.0, 1.0)]
    )
    group, group_input, group_output = geometry_group(surface, "GN Smooth Random Distribution")
    nodes, links = group.nodes, group.links
    distribute = nodes.new("GeometryNodeDistributePointsOnFaces")
    distribute.distribute_method = "RANDOM"
    distribute.inputs["Density"].default_value = 2.0
    distribute.inputs["Seed"].default_value = 29
    cube = nodes.new("GeometryNodeMeshCube")
    cube.inputs["Size"].default_value = (0.2, 0.2, 0.2)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.inputs["Depth"].default_value = 5
    links.new(group_input.outputs["Geometry"], distribute.inputs["Mesh"])
    links.new(distribute.outputs["Points"], instance.inputs["Points"])
    links.new(cube.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], group_output.inputs["Geometry"])
    return {
        "method": "RANDOM",
        "density": float(distribute.inputs["Density"].default_value),
        "seed": int(distribute.inputs["Seed"].default_value),
        "selection": bool(distribute.inputs["Selection"].default_value),
        "smooth_shading": all(polygon.use_smooth for polygon in surface.data.polygons),
        "has_custom_normals": bool(surface.data.has_custom_normals),
    }


def build_crease_subdivision_case(name, edge_crease):
    obj = mesh_object(
        name,
        [(-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
         (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)],
        [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)],
    )
    crease = obj.data.attributes.new(name="crease_edge", type="FLOAT", domain="EDGE")
    for edge in obj.data.edges:
        crease.data[edge.index].value = 0.75 if edge.index == 0 else 0.0
    group, group_input, group_output = geometry_group(obj, f"GN Crease {name}")
    subdiv = group.nodes.new("GeometryNodeSubdivisionSurface")
    subdiv.inputs["Level"].default_value = 1
    subdiv.inputs["Edge Crease"].default_value = edge_crease
    group.links.new(group_input.outputs["Geometry"], subdiv.inputs["Mesh"])
    group.links.new(subdiv.outputs["Mesh"], group_output.inputs["Geometry"])
    modifier = obj.modifiers.new("Regular Subdivision Surface", "SUBSURF")
    modifier.levels = 1
    modifier.render_levels = 1
    modifier.use_creases = True
    return {
        "node_level": int(subdiv.inputs["Level"].default_value),
        "edge_crease_default": float(subdiv.inputs["Edge Crease"].default_value),
        "vertex_crease_default": float(subdiv.inputs["Vertex Crease"].default_value),
        "input_edge_crease_max": max(item.value for item in crease.data),
        "modifier_level": modifier.levels,
        "modifier_uses_creases": modifier.use_creases,
    }


def build_store_read_case(name, data_type):
    obj = mesh_object(name, [], [])
    group, _, group_output = geometry_group(obj, f"GN Store Read {data_type}")
    nodes, links = group.nodes, group.links
    mesh_line = nodes.new("GeometryNodeMeshLine")
    mesh_line.inputs["Count"].default_value = 4
    mesh_line.inputs["Offset"].default_value = (0.0, 0.0, 0.0)
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = data_type
    store.domain = "POINT"
    attribute_name = "roundtrip_value"
    store.inputs["Name"].default_value = attribute_name
    named_attribute = nodes.new("GeometryNodeInputNamedAttribute")
    named_attribute.data_type = data_type
    named_attribute.inputs["Name"].default_value = attribute_name
    if data_type == "INT":
        index = nodes.new("GeometryNodeInputIndex")
        copy_store = nodes.new("GeometryNodeStoreNamedAttribute")
        copy_store.data_type = "INT"
        copy_store.domain = "POINT"
        copy_store.inputs["Name"].default_value = "roundtrip_copy"
        links.new(mesh_line.outputs["Mesh"], store.inputs["Geometry"])
        links.new(index.outputs["Index"], store.inputs["Value"])
        links.new(store.outputs["Geometry"], copy_store.inputs["Geometry"])
        links.new(named_attribute.outputs["Attribute"], copy_store.inputs["Value"])
        links.new(copy_store.outputs["Geometry"], group_output.inputs["Geometry"])
    else:
        set_position = nodes.new("GeometryNodeSetPosition")
        links.new(mesh_line.outputs["Mesh"], store.inputs["Geometry"])
        links.new(store.outputs["Geometry"], set_position.inputs["Geometry"])
        store.inputs["Value"].default_value = True
        set_position.inputs["Offset"].default_value = (0.0, 0.0, 1.0)
        links.new(named_attribute.outputs["Attribute"], set_position.inputs["Selection"])
        links.new(set_position.outputs["Geometry"], group_output.inputs["Geometry"])
    return {
        "data_type": data_type,
        "attribute_name": attribute_name,
        "domain": "POINT",
        "mesh_line_count": int(mesh_line.inputs["Count"].default_value),
        "copy_attribute_name": "roundtrip_copy" if data_type == "INT" else None,
    }


def build_subdivision_case():
    obj = mesh_object(
        SUBDIVISION_OBJECT,
        [(-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
         (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)],
        [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7)],
    )
    group, group_input, group_output = geometry_group(obj, "GN Subdivision Crease Inputs")
    subdiv = group.nodes.new("GeometryNodeSubdivisionSurface")
    subdiv.inputs["Level"].default_value = 2
    subdiv.inputs["Edge Crease"].default_value = 0.65
    subdiv.inputs["Vertex Crease"].default_value = 0.8
    group.links.new(group_input.outputs["Geometry"], subdiv.inputs["Mesh"])
    group.links.new(subdiv.outputs["Mesh"], group_output.inputs["Geometry"])


def build_simulation_case(scene):
    obj = mesh_object(
        SIMULATION_OBJECT,
        [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.0, 0.5, 0.0)],
        [(0, 1, 2)],
    )
    group, group_input, group_output = geometry_group(obj, "GN Delta Time Simulation")
    nodes, links = group.nodes, group.links
    simulation_input = nodes.new("GeometryNodeSimulationInput")
    simulation_output = nodes.new("GeometryNodeSimulationOutput")
    simulation_input.pair_with_output(simulation_output)
    delta_translation = nodes.new("ShaderNodeCombineXYZ")
    transform = nodes.new("GeometryNodeTransform")
    links.new(group_input.outputs["Geometry"], simulation_input.inputs["Geometry"])
    links.new(simulation_input.outputs["Delta Time"], delta_translation.inputs["Z"])
    links.new(simulation_input.outputs["Geometry"], transform.inputs["Geometry"])
    links.new(delta_translation.outputs["Vector"], transform.inputs["Translation"])
    links.new(transform.outputs["Geometry"], simulation_output.inputs["Geometry"])
    links.new(simulation_output.outputs["Geometry"], group_output.inputs["Geometry"])
    scene.frame_start = 1
    scene.frame_end = 2


def main():
    if bpy.app.version[:3] != (5, 2, 2):
        raise RuntimeError(f"expected Blender 5.2.2, got {bpy.app.version_string}")
    if "--" not in sys.argv or len(sys.argv) != sys.argv.index("--") + 2:
        raise SystemExit("expected one output directory after --")
    output_dir = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.render.fps_base = 1.001

    build_poisson_case()
    build_poisson_candidate_case()
    build_random_value_case(RANDOM_FLOAT_OBJECT, "FLOAT", -1.25, 0.75, 29)
    build_random_value_case(RANDOM_INT_OBJECT, "INT", -7, 12, 29)
    build_random_value_case(RANDOM_BOOLEAN_OBJECT, "BOOLEAN", None, None, 29, 0.35)
    build_random_value_case(
        RANDOM_VECTOR_OBJECT,
        "FLOAT_VECTOR",
        (-0.75, -1.25, -0.5),
        (0.75, 1.25, 0.5),
        29,
    )
    random_poisson_id_linked = build_random_poisson_id_case()
    build_random_selection_case()
    build_subdivision_case()
    smooth_random = build_smooth_random_case()
    zero_crease_subdivision = build_crease_subdivision_case(ZERO_CREASE_SUBDIVISION_OBJECT, 0.0)
    negative_crease_subdivision = build_crease_subdivision_case(
        NEGATIVE_CREASE_SUBDIVISION_OBJECT, -1.0
    )
    named_attributes = {
        "int": build_store_read_case(STORE_READ_INT_OBJECT, "INT"),
        "boolean": build_store_read_case(STORE_READ_BOOLEAN_OBJECT, "BOOLEAN"),
        "quaternion": build_quaternion_attribute_case(),
    }
    build_simulation_case(scene)

    scene.frame_set(1)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    evaluated_vertices(bpy.data.objects[SIMULATION_OBJECT], depsgraph)
    scene.frame_set(2)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()
    objects = {name: evaluated_vertices(bpy.data.objects[name], depsgraph) for name in OBJECT_NAMES}
    random_int_values = evaluated_int_attribute(
        bpy.data.objects[RANDOM_INT_OBJECT], depsgraph, "oracle_random_int"
    )
    named_attributes["int"]["values"] = evaluated_int_attribute(
        bpy.data.objects[STORE_READ_INT_OBJECT],
        depsgraph,
        named_attributes["int"]["copy_attribute_name"],
    )
    named_attributes["quaternion"]["values"] = evaluated_quaternion_attribute(
        bpy.data.objects[STORE_READ_QUATERNION_OBJECT],
        depsgraph,
        named_attributes["quaternion"]["attribute_name"],
    )
    oracle = {
        "blender_version": bpy.app.version_string,
        "frame": 2,
        "fps": scene.render.fps,
        "fps_base": scene.render.fps_base,
        "poisson": {
            "method": "POISSON",
            "density": 23.0,
            "density_factor": 0.35,
            "density_max": 6.0,
            "distance_min": 0.45,
            "seed": 17,
            "instance_vertex_count": 8,
            "point_count": len(objects[POISSON_OBJECT]) // 8,
        },
        "poisson_candidates": {
            "method": "POISSON",
            "density": 23.0,
            "density_factor": 1.0,
            "density_max": 6.0,
            "distance_min": 0.0,
            "seed": 17,
            "point_count": len(objects[POISSON_CANDIDATE_OBJECT]) // 8,
        },
        "random_values": {
            "float": {
                "data_type": "FLOAT",
                "minimum": -1.25,
                "maximum": 0.75,
                "seed": 29,
                "mesh_line_count": 8,
            },
            "int": {
                "data_type": "INT",
                "minimum": -7,
                "maximum": 12,
                "seed": 29,
                "mesh_line_count": 8,
                "attribute_name": "oracle_random_int",
                "values": random_int_values,
            },
            "boolean": {
                "data_type": "BOOLEAN",
                "probability": 0.35,
                "seed": 29,
                "mesh_line_count": 8,
            },
            "vector": {
                "data_type": "FLOAT_VECTOR",
                "minimum": [-0.75, -1.25, -0.5],
                "maximum": [0.75, 1.25, 0.5],
                "seed": 29,
                "mesh_line_count": 8,
            },
        },
        "random_poisson_id": {
            "method": "POISSON",
            "density": 23.0,
            "density_factor": 1.0,
            "density_max": 6.0,
            "distance_min": 0.0,
            "distribute_seed": 17,
            "data_type": "FLOAT_VECTOR",
            "minimum": [-0.3, -0.4, -0.5],
            "maximum": [0.3, 0.4, 0.5],
            "random_seed": 29,
            "id_linked": random_poisson_id_linked,
            "point_count": len(objects[RANDOM_POISSON_ID_OBJECT]) // 8,
        },
        "random_selection": {
            "method": "RANDOM",
            "density": 8.0,
            "selection": False,
            "point_count": len(objects[RANDOM_SELECTION_OBJECT]) // 8,
        },
        "smooth_random": {
            **smooth_random,
            "point_count": len(objects[SMOOTH_RANDOM_OBJECT]) // 8,
        },
        "zero_crease_subdivision": zero_crease_subdivision,
        "negative_crease_subdivision": negative_crease_subdivision,
        "named_attributes": named_attributes,
        "realize": {"depth": 5, "realize_all": True, "selection": True},
        "subdivision": {"level": 2, "edge_crease": 0.65, "vertex_crease": 0.8},
        "objects": objects,
    }
    with (output_dir / "oracle.json").open("w", encoding="utf-8") as handle:
        json.dump(oracle, handle, separators=(",", ":"))
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "gn_silent_mismatches.blend"))


if __name__ == "__main__":
    main()
