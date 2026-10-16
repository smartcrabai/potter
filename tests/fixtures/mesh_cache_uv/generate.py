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
for li, layer in enumerate(me.uv_layers):
    for i, d in enumerate(layer.data):
        d.uv = (0.1 * i + 0.5 * li, 0.05 * i)
me.uv_layers.active_index = 1
me.uv_layers.active_render_index = 0
bpy.ops.wm.alembic_export(filepath=out + "/two_layers.abc", uvs=True, selected=False)
