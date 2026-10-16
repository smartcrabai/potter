# Blender fixtures for tests/mesh_cache_uv_layers.rs. Writes `two_layers.abc` and a second
# animated archive whose `UVMap` has asymmetric loop values while a same-size face winding flips.
# Run: Blender -b --factory-startup -P generate.py -- <output directory>
import bpy
import sys

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_plane_add(size=2)
ob = bpy.context.object
ob.name = "Plane"
me = ob.data
me.uv_layers[0].name = "Base"
me.uv_layers.new(name="Light")
# Each corner is unique and nonsymmetric to make the cache's position/value association visible.
BASE = [(0.137, 0.223), (0.619, 0.357), (0.283, 0.881), (0.947, 0.569)]
LIGHT = [(0.731, 0.163), (0.194, 0.907), (0.853, 0.421), (0.367, 0.689)]
for layer, values in zip(me.uv_layers, (BASE, LIGHT)):
    for datum, uv in zip(layer.data, values):
        datum.uv = uv
me.uv_layers.active_index = 1
me.uv_layers.active_render_index = 0
bpy.ops.wm.alembic_export(filepath=out + "/two_layers.abc", uvs=True, selected=False)

# A single animated archive changes only face-corner connectivity at frame 2. The
# UV loop values are asymmetric so a topology reread can be matched to positions.
scene = bpy.context.scene
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 2
reordered_vertices = [
    (-1.0, -1.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
]
reordered_mesh = bpy.data.meshes.new("Reordered")
reordered_mesh.from_pydata(reordered_vertices, [], [(0, 1, 2, 3)])
reordered_mesh.update()
reordered = bpy.data.objects.new("Reordered", reordered_mesh)
scene.collection.objects.link(reordered)
reordered_uv = reordered_mesh.uv_layers.new(name="UVMap")
for datum, uv in zip(
    reordered_uv.data,
    [(0.137, 0.223), (0.619, 0.357), (0.283, 0.881), (0.947, 0.569)],
):
    datum.uv = uv

tree = bpy.data.node_groups.new("ReorderedFaceAtFrameTwo", "GeometryNodeTree")
tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
group_in = tree.nodes.new("NodeGroupInput")
group_out = tree.nodes.new("NodeGroupOutput")
scene_time = tree.nodes.new("GeometryNodeInputSceneTime")
after_first_frame = tree.nodes.new("FunctionNodeCompare")
after_first_frame.data_type = "FLOAT"
after_first_frame.operation = "GREATER_THAN"
after_first_frame.inputs["B"].default_value = 1.5
flip = tree.nodes.new("GeometryNodeFlipFaces")
tree.links.new(scene_time.outputs["Frame"], after_first_frame.inputs["A"])
tree.links.new(group_in.outputs["Geometry"], flip.inputs["Mesh"])
tree.links.new(after_first_frame.outputs["Result"], flip.inputs["Selection"])
tree.links.new(flip.outputs["Mesh"], group_out.inputs["Geometry"])
modifier = reordered.modifiers.new("ReorderedFaceAtFrameTwo", "NODES")
modifier.node_group = tree

reordered_archive = out + "/reordered_topology.abc"
bpy.ops.object.select_all(action="DESELECT")
reordered.select_set(True)
bpy.context.view_layer.objects.active = reordered
scene.frame_set(1)
bpy.ops.wm.alembic_export(
    filepath=reordered_archive,
    selected=True,
    start=1,
    end=2,
    uvs=True,
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
oracle_frames = []
for frame in (1, 2):
    scene.frame_set(frame)
    evaluated = reordered.evaluated_get(bpy.context.evaluated_depsgraph_get())
    evaluated_mesh = evaluated.to_mesh()
    try:
        oracle_frames.append({
            "frame": frame,
            "positions": [list(vertex.co) for vertex in evaluated_mesh.vertices],
            "faces": [list(poly.vertices) for poly in evaluated_mesh.polygons],
            "uvs": [list(loop.uv) for loop in evaluated_mesh.uv_layers["UVMap"].data],
        })
    finally:
        evaluated.to_mesh_clear()
with open(out + "/reordered_topology.json", "w", encoding="utf-8") as handle:
    json.dump({"frames": oracle_frames}, handle, indent=2)
    handle.write("\n")
