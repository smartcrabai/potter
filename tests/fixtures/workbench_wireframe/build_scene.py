"""Builds the wireframe oracle scene: a 2 m cube and a radius-0.8 UV sphere, both unselected."""
import sys
import bpy

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(-1.2, 0.0, 0.0))
cube = bpy.context.active_object
cube.name = "Cube"
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.8, location=(1.2, 0.0, 0.0))
sphere = bpy.context.active_object
sphere.name = "Sphere"
bpy.ops.object.shade_smooth()
bpy.ops.object.select_all(action="DESELECT")
bpy.context.view_layer.objects.active = None
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SCENE_SAVED", out)
