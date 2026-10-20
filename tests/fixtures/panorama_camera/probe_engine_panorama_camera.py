# Headless Blender 5.2.2 probe: does a PANO camera render like a perspective camera with the same `lens` in each engine?
# Run: Blender -b --factory-startup --threads 1 --python probe_engine_panorama_camera.py -- BLENDER_WORKBENCH BLENDER_EEVEE CYCLES
# Recorded result: Workbench and EEVEE give max abs diff 0.0 for both PANO types; Cycles differs (0.59 and 0.46).
import bpy, math, sys, os
out = "/tmp/probe"
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.resolution_x = 160
sc.cycles.samples = 8 if sc.render.engine == "CYCLES" else 1
sc.render.resolution_y = 160
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
# a cube at the origin, and a light for EEVEE
bpy.ops.mesh.primitive_cube_add(size=1.6, location=(0, 0, 0.8))
bpy.ops.object.light_add(type='SUN', location=(3, -3, 5))
def cam(name, ctype, pano=None):
    c = bpy.data.cameras.new(name)
    c.type = ctype
    if pano:
        c.panorama_type = pano
    c.lens = 10.5
    c.sensor_width = 36
    o = bpy.data.objects.new(name, c)
    sc.collection.objects.link(o)
    o.location = (0, -4, 4)
    o.rotation_euler = (math.radians(45), 0, 0)
    return o
cams = {
    "persp": cam("persp", 'PERSP'),
    "pano_eq": cam("pano_eq", 'PANO', 'EQUIRECTANGULAR'),
    "pano_eqd": cam("pano_eqd", 'PANO', 'FISHEYE_EQUIDISTANT'),
}
engines = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["BLENDER_WORKBENCH"]
def pixels(path):
    img = bpy.data.images.load(path)
    px = list(img.pixels)
    bpy.data.images.remove(img)
    return px
for eng in engines:
    sc.render.engine = eng
    paths = {}
    for k, o in cams.items():
        sc.camera = o
        p = f"{out}/{eng}_{k}.png"
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths[k] = p
    base = pixels(paths["persp"])
    for k in ("pano_eq", "pano_eqd"):
        other = pixels(paths[k])
        diff = max(abs(a - b) for a, b in zip(base, other))
        print(f"RESULT {eng} persp vs {k}: max abs diff = {diff:.6f}")
