"""Blender 5.2.2 oracle for compositor exposure followed by the AgX display transform.

Opens the native oracle scene, keeps its constant-linear emitter, and writes a `.blend` plus the
final Cycles PNG. The scene view exposure and compositor Exposure node are both +0.5; AgX, look
None, and gamma 1 are explicit so Potter can compare the final 8-bit centre pixel.

Arguments after `--`: blend_out, png_out.
"""

import sys

import bpy

blend_out, png_out = sys.argv[sys.argv.index("--") + 1:][:2]
scene = bpy.context.scene
emitter = bpy.data.objects["Emitter"]
for item in emitter.data.color_attributes["Ink"].data:
    item.color = (0.18, 0.18, 0.18, 1.0)
for obj in scene.objects:
    if obj is not emitter:
        obj.hide_render = True
scene.world = None

scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.5
scene.view_settings.gamma = 1.0
scene.display_settings.display_device = "sRGB"

tree = bpy.data.node_groups.new("Exposure Oracle", "CompositorNodeTree")
scene.compositing_node_group = tree
scene.render.use_compositing = True
tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
render_layers = tree.nodes.new("CompositorNodeRLayers")
exposure = tree.nodes.new("CompositorNodeExposure")
exposure.inputs["Exposure"].default_value = 0.5
group_output = tree.nodes.new("NodeGroupOutput")
tree.links.new(render_layers.outputs["Image"], exposure.inputs["Image"])
tree.links.new(exposure.outputs["Image"], group_output.inputs["Image"])

camera_data = bpy.data.cameras.new("CompositorCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 4.0
camera = bpy.data.objects.new("CompositorCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 6.0)
scene.camera = camera

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 8
scene.cycles.use_denoising = False
scene.cycles.max_bounces = 0
scene.cycles.pixel_filter_type = "BOX"
scene.cycles.filter_width = 0.01
scene.render.resolution_x = 16
scene.render.resolution_y = 16
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.image_settings.color_depth = "8"

bpy.ops.wm.save_as_mainfile(filepath=blend_out)
scene.render.filepath = png_out
bpy.ops.render.render(write_still=True)
