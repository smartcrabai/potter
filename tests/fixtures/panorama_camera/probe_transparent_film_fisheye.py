# Headless Blender 5.2.2 probe: a pixel with no fisheye ray under a transparent film. Cycles writes it opaque black.
# Run: Blender -b --factory-startup --threads 1 --python probe_transparent_film_fisheye.py
# Recorded result: the corner (no ray, fisheye_fov 90 degrees) is (0, 0, 0, 1); the centre, which has a ray to the empty
# background, is (0, 0, 0, 0). So a no-ray pixel is opaque black under a transparent film too.
import bpy, math, os
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.samples = 4
sc.render.resolution_x = 64
sc.render.resolution_y = 64
sc.render.resolution_percentage = 100
sc.render.film_transparent = True
sc.render.image_settings.file_format = 'OPEN_EXR'
sc.render.image_settings.color_mode = 'RGBA'
c = bpy.data.cameras.new("fisheye")
c.type = 'PANO'
c.panorama_type = 'FISHEYE_EQUIDISTANT'
c.fisheye_fov = math.pi / 2
o = bpy.data.objects.new("fisheye", c)
sc.collection.objects.link(o)
o.location = (0, 0, 0)
o.rotation_euler = (math.radians(90), 0, 0)
sc.camera = o
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transparent_fisheye.exr")
sc.render.filepath = path
bpy.ops.render.render(write_still=True)
img = bpy.data.images.load(path)
w, h = img.size
px = list(img.pixels)
corner = px[0:4]
centre = px[((h // 2) * w + w // 2) * 4:((h // 2) * w + w // 2) * 4 + 4]
print("RESULT corner", [round(v, 6) for v in corner], "centre", [round(v, 6) for v in centre])
