"""Blender 5.2.2 oracle for tests/pr18f_displacement.rs `displace_texture_uses_named_native_uv_layer`.

The Rust scene: a size-2 plane with no UV layer, a UVMap layer whose corners come from the product's
box projection of a +Z face (raw `(x, y)` per corner), then `uv' = scale*uv + translation` with
scale [-0.5, 0.5] and translation [0.5, 0.5], a BLEND texture (DIAGONAL), and a DISPLACE modifier
(direction X, strength 1, mid level 0.5, texture coords UV on UVMap).

Potter stores UV V-down; Blender's loop UV is V-up. The oracle writes Blender loop UVs as
`(u, 1 - v)` for each Potter-stored `(u, v)`, the conversion `displacement.rs` applies.

Blender's `uv.cube_project` normalises the box projection into [0, 1]; the product does not. The oracle
therefore sets the explicit corner UVs the product produces, so the displacement is evaluated on the
identical UV input, and the box-normalisation difference is reported separately.

`primitive_plane_add` creates its own UV layer named UVMap; the scene must start with no UV layer, so
every default layer is removed first. Otherwise `uv_layers.new("UVMap")` becomes "UVMap.001" and the
DISPLACE `uv_layer` name resolves to the primitive's default layer, silently ignoring the UVs.

Writes the loop UVs and evaluated vertex positions, in storage order, to OUT.json.

Usage:
  Blender -b --factory-startup --python generate_named_uv_exact_oracle.py -- OUT.json
"""

import bpy
import json
import sys


def out_path():
    return sys.argv[sys.argv.index("--") + 1]


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_plane_add(size=2.0)
obj = bpy.context.active_object
mesh = obj.data

for default_layer in list(mesh.uv_layers):
    mesh.uv_layers.remove(default_layer)

tex = bpy.data.textures.new("blend_tex", "BLEND")
tex.progression = "DIAGONAL"

layer = mesh.uv_layers.new(name="UVMap")
mesh.uv_layers.active = layer
assert layer.name == "UVMap", "UVMap layer name did not resolve: %r" % layer.name

for index, loop in enumerate(layer.data):
    co = mesh.vertices[mesh.loops[index].vertex_index].co
    stored_u, stored_v = co.x * -0.5 + 0.5, co.y * 0.5 + 0.5
    loop.uv = (stored_u, 1.0 - stored_v)
uvs = [[round(c, 9) for c in loop.uv] for loop in layer.data]

mod = obj.modifiers.new("deform", "DISPLACE")
mod.texture = tex
mod.texture_coords = "UV"
mod.uv_layer = "UVMap"
mod.direction = "X"
mod.strength = 1.0
mod.mid_level = 0.5

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
positions = [[round(c, 9) for c in v.co] for v in evaluated_mesh.vertices]
evaluated.to_mesh_clear()

with open(out_path(), "w") as handle:
    json.dump({"uvs": uvs, "positions": positions}, handle, indent=1)
print("WROTE", out_path(), len(positions))
