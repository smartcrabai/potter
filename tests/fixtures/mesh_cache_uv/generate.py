# Blender oracle for tests/mesh_cache_uv_layers.rs. Writes two_layers.abc: a plane with two UV
# layers, Base (render) and Light (active), exported with Blender's Alembic exporter.
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
