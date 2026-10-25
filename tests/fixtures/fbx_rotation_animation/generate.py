import bpy
import json
import os
import sys

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"FBX rotation fixture requires Blender 5.2.2, found {bpy.app.version_string}")

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 9
scene.render.fps = 24
scene.render.fps_base = 1.0

probe_mesh = bpy.data.meshes.new("RotationProbeMesh")
probe_mesh.from_pydata([(-1, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
probe_mesh.update()
rotation_probe = bpy.data.objects.new("RotationProbe", probe_mesh)
scene.collection.objects.link(rotation_probe)
rotation_probe.rotation_mode = "XYZ"
for frame, rotation in (
    (1, (0.0, 0.0, 0.0)),
    (9, (0.7, -0.4, 0.9)),
):
    rotation_probe.rotation_euler = rotation
    rotation_probe.keyframe_insert(data_path="rotation_euler", frame=frame)

for layer in rotation_probe.animation_data.action.layers:
    for strip in layer.strips:
        for slot in rotation_probe.animation_data.action.slots:
            channelbag = strip.channelbag(slot)
            if channelbag is None:
                continue
            for curve in channelbag.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
                curve.update()

frame = 5
scene.frame_set(frame)
depsgraph = bpy.context.evaluated_depsgraph_get()
depsgraph.update()
world_rotation = rotation_probe.evaluated_get(depsgraph).matrix_world.to_quaternion()
with open(os.path.join(root, "blender_expected.json"), "w", encoding="utf-8") as output:
    json.dump(
        {
            "frame": frame,
            "rotation_mode": rotation_probe.rotation_mode,
            "rotation_euler": [float(value) for value in rotation_probe.rotation_euler],
            "rotation_xyzw": [
                float(world_rotation.x),
                float(world_rotation.y),
                float(world_rotation.z),
                float(world_rotation.w),
            ],
        },
        output,
    )

fbx_path = os.path.join(root, "rotation_animation.fbx")
bpy.ops.object.select_all(action="DESELECT")
rotation_probe.select_set(True)
bpy.context.view_layer.objects.active = rotation_probe
bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=True,
    axis_forward="Y",
    axis_up="Z",
    bake_anim=True,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0,
)
