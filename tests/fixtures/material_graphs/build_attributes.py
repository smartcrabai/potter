"""Blender 5.2 mesh-attribute fixture for the glTF export test.

Writes one .blend (path in POTTER_ATTRIBUTES_BLEND) with:
- a cube with two material slots, a sharp edge, a UV seam, a hidden face and a smooth face
  (exercises `material_index`, `sharp_edge`, `uv_seam`, `.hide_poly`, `sharp_face`), and
- a plane with an identity UV Warp modifier, whose evaluated UVs are the oracle for the V flip
  (the modifier writes its output as Blender-layer data with V up).
"""
import os

import bpy

OUT = os.environ["POTTER_ATTRIBUTES_BLEND"]
bpy.ops.wm.read_factory_settings(use_empty=True)

mat_a = bpy.data.materials.new("AttrA")
mat_b = bpy.data.materials.new("AttrB")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
cube = bpy.context.active_object
cube.name = "attr_cube"
cube.data.materials.append(mat_a)
cube.data.materials.append(mat_b)
cube.data.polygons[1].material_index = 1
cube.data.polygons[4].material_index = 1
cube.data.edges[0].use_edge_sharp = True
cube.data.edges[5].use_seam = True
cube.data.polygons[2].hide = True
cube.data.polygons[3].use_smooth = True

mat_warp = bpy.data.materials.new("AttrWarp")
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(3.0, 0.0, 0.0))
plane = bpy.context.active_object
plane.name = "attr_warp"
plane.data.materials.append(mat_warp)
warp = plane.modifiers.new("Warp", "UV_WARP")
warp.uv_layer = "UVMap"

bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("ATTRIBUTES_BUILT", OUT)
