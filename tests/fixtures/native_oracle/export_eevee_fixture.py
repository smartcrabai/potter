"""Writes an EEVEE render's RGB pixels as little-endian f32, for the realtime oracle tests.

Run by Blender after one of the oracle generators has rendered its EXR:

    Blender -b --factory-startup -P export_eevee_fixture.py -- RENDER.exr OUT.f32le

The output holds the render's rows bottom-up (Blender's image order), each row's pixels left to right, and three f32
values (red, green, blue) per pixel. The alpha channel is dropped. The size is the image's, checked by the tests.
"""

import sys
from array import array

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :][:2]
source, target = argv
image = bpy.data.images.load(source, check_existing=False)
width, height = image.size
pixels = array("f", [0.0]) * (width * height * 4)
image.pixels.foreach_get(pixels)
rgb = array("f")
for index in range(width * height):
    rgb.extend(pixels[index * 4 : index * 4 + 3])
if sys.byteorder == "big":
    rgb.byteswap()
with open(target, "wb") as handle:
    handle.write(rgb.tobytes())
print(f"export_eevee_fixture: {width}x{height} -> {target}")
