"""Writes the object ray visibility round-trip scene (Blender 5.2.2).

Arguments after `--`: blend_out. The objects are:
  HiddenFromCamera         mesh; visible_camera and visible_shadow off
  GlossyTransmissionOff    mesh; visible_glossy and visible_transmission off
  Lamp                     point light with Blender's default flags (camera hidden)
  Plain                    mesh with every flag at its default
`tests/blend_ray_visibility_roundtrip.rs` checks these flags against Potter's import and export.
"""

import sys

import bpy

blend_out = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-2.0, 0.0, 0.0))
hidden = bpy.context.object
hidden.name = "HiddenFromCamera"
hidden.visible_camera = False
hidden.visible_shadow = False

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
glossy_off = bpy.context.object
glossy_off.name = "GlossyTransmissionOff"
glossy_off.visible_glossy = False
glossy_off.visible_transmission = False

light_data = bpy.data.lights.new("Lamp", "POINT")
lamp = bpy.data.objects.new("Lamp", light_data)
bpy.context.scene.collection.objects.link(lamp)
lamp.location = (0.0, 2.0, 3.0)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2.0, 0.0, 0.0))
bpy.context.object.name = "Plain"

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
