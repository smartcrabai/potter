"""Oracle scene: identical-depth duplicate edges, a point cloud with radii, and a poly curve."""
import sys
import bpy

out = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)


def add_mesh(name, verts, edges, loc=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = loc
    bpy.context.scene.collection.objects.link(obj)
    return obj


# Front/back pair at z=-1.5 (depths differ).
add_mesh("Single", [(-1.5, 0.0, -4.5), (1.5, 0.0, -4.5)], [(0, 1)])
add_mesh("PairFB", [(-1.5, -1.0, -1.5), (1.5, -1.0, -1.5), (-1.5, 1.0, -1.5), (1.5, 1.0, -1.5)], [(0, 1), (2, 3)])
# Identical-depth duplicates at z=1.5 (two objects, same geometry, same depth).
add_mesh("DupA", [(-1.5, 0.0, 1.5), (1.5, 0.0, 1.5)], [(0, 1)])
add_mesh("DupB", [(-1.5, 0.0, 1.5), (1.5, 0.0, 1.5)], [(0, 1)])

# Point cloud: four points with radii, at z=3.5.
pc = bpy.data.pointclouds.new("PC")
pc.resize(4)
coords = [(-1.5, 0.0, 3.5), (-0.5, 0.0, 3.5), (0.5, 0.0, 3.5), (1.5, 0.0, 3.5)]
pc.points.foreach_set("co", [c for p in coords for c in p])
if "radius" not in pc.attributes:
    pc.attributes.new("radius", "FLOAT", "POINT")
pc.attributes["radius"].data.foreach_set("value", [0.05, 0.1, 0.15, 0.2])
pco = bpy.data.objects.new("PC", pc)
bpy.context.scene.collection.objects.link(pco)

# Curve: a poly spline from (-1.5,0,5.5) to (1.5,0,5.5) to (1.5,0,6.5), bevel off.
cu = bpy.data.curves.new("Curve", "CURVE")
cu.dimensions = "3D"
spline = cu.splines.new("POLY")
spline.points.add(2)
for i, co in enumerate([(-1.5, 0.0, 5.5), (1.5, 0.0, 5.5), (1.5, 0.0, 6.5)]):
    spline.points[i].co = (co[0], co[1], co[2], 1.0)
cuo = bpy.data.objects.new("Curve", cu)
bpy.context.scene.collection.objects.link(cuo)

bpy.ops.wm.save_as_mainfile(filepath=out)
print("PC_SAVED", out)
