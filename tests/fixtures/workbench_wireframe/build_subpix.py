"""Lines at controlled sub-pixel screen positions: horizontal rows and vertical columns.
Screen mapping of the fixed capture camera: ndc_x = 0.11574 * x, ndc_y = 0.13622 * z (ortho, view along +y)."""
import json, sys
import bpy
argv = sys.argv[sys.argv.index("--") + 1:]
out_blend, out_json = argv[0], argv[1]
W, H = 818, 695
KX, KY = 0.11574074880910873, 0.13622435927391052
def world_x(sx): return (2 * sx / W - 1) / KX
def world_z(sy): return (1 - 2 * sy / H) / KY
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("Sub")
verts, edges, meta = [], [], []
FRACS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
for k, f in enumerate(FRACS):
    sy = 60.0 + 40.0 * k + f           # horizontal row at screen y = sy
    x0, x1 = 300.0, 520.0
    a = len(verts); verts += [(world_x(x0), 0.0, world_z(sy)), (world_x(x1), 0.0, world_z(sy))]; edges.append((a, a + 1))
    meta.append({"kind": "h", "frac": f, "y": sy, "x0": x0, "x1": x1})
for k, f in enumerate(FRACS):
    sx = 560.0 + 22.0 * k + f          # vertical column at screen x = sx
    y0, y1 = 100.0, 600.0
    # screen y = (1 - ndc_y) * H / 2 with ndc_y = 0.13622 * z
    a = len(verts); verts += [(0.0, 0.0, world_z(y0)), (0.0, 0.0, world_z(y1))]; edges.append((a, a + 1))
    verts[-2] = (world_x(sx), 0.0, verts[-2][2]); verts[-1] = (world_x(sx), 0.0, verts[-1][2])
    meta.append({"kind": "v", "frac": f, "x": sx, "y0": y0, "y1": y1})
mesh.from_pydata(verts, edges, []); mesh.update()
obj = bpy.data.objects.new("Sub", mesh); bpy.context.scene.collection.objects.link(obj)
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
json.dump(meta, open(out_json, "w"), indent=1)
print("SUBPIX_SAVED", len(edges))
