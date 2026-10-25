import bpy
import json
import math
import os
import struct
import sys

from io_scene_fbx import encode_bin

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"FBX rotation-order oracle requires Blender 5.2.2, found {bpy.app.version_string}")

root_dir = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
orders = ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")
order_values = {order: index for index, order in enumerate(orders)}
frame = 5
frames = (frame,)
original_write = encode_bin.write


def write_rotation_order(filepath, root, version, name, order, rotation_degrees):
    objects = next(elem for elem in root.elems if elem.id == b"Objects")
    models = []
    for model in objects.elems:
        if model.id != b"Model":
            continue
        encoded_name = model.props[1][4:].decode("utf-8", "replace")
        object_name = encoded_name.split("\x00\x01", 1)[0].rsplit("::", 1)[-1]
        if object_name == name:
            models.append(model)
    if len(models) != 1:
        available = [
            model.props[1].decode("utf-8", "replace")
            for model in objects.elems
            if model.id == b"Model"
        ]
        raise RuntimeError(
            f"expected one FBX Model for {name}, found {len(models)}; available={available}"
        )
    properties = next(elem for elem in models[0].elems if elem.id == b"Properties70")

    def set_property(property_name, property_type, value):
        prop = next(
            (
                prop
                for prop in properties.elems
                if prop.id == b"P" and prop.props[0][4:] == property_name
            ),
            None,
        )
        if prop is not None:
            properties.elems.remove(prop)
        prop = encode_bin.FBXElem(b"P")
        prop.add_string(property_name)
        prop.add_string(property_type)
        prop.add_string(b"")
        prop.add_string(b"")
        prop.add_int32(int(value))
        properties.elems.append(prop)

    set_property(b"RotationOrder", b"enum", order_values[order])
    set_property(b"RotationActive", b"bool", True)

    rotation_property = next(
        prop
        for prop in properties.elems
        if prop.id == b"P" and prop.props[0][4:] == b"Lcl Rotation"
    )
    rotation_property.props[4:7] = [
        struct.pack("<d", angle) for angle in rotation_degrees
    ]
    return original_write(filepath, root, version)


world_rotations = {}
for order in orders:
    for kind in ("Static", "Animated"):
        name = f"{kind}_{order}"
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = 9
        scene.render.fps = 24
        scene.render.fps_base = 1.0
        mesh = bpy.data.meshes.new(f"{name}Mesh")
        mesh.from_pydata([(-1, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.rotation_euler = (0.31, -0.47, 0.73)
        if kind == "Animated":
            for key_frame, x_angle, y_angle in (
                (1, 0.2, -0.35),
                (4, 0.4, 0.15),
                (5, 0.4, 0.15),
                (9, 0.8, -0.25),
            ):
                obj.rotation_euler[0] = x_angle
                obj.keyframe_insert(data_path="rotation_euler", index=0, frame=key_frame)
                obj.rotation_euler[1] = y_angle
                obj.keyframe_insert(data_path="rotation_euler", index=1, frame=key_frame)
            for layer in obj.animation_data.action.layers:
                for strip in layer.strips:
                    for slot in obj.animation_data.action.slots:
                        channelbag = strip.channelbag(slot)
                        if channelbag is None:
                            continue
                        for curve in channelbag.fcurves:
                            for point in curve.keyframe_points:
                                point.interpolation = "LINEAR"
                            curve.update()
        source_path = os.path.join(root_dir, f"{name}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=source_path)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        fbx_path = os.path.join(root_dir, f"{name}.fbx")

        def write_ordered_fbx(filepath, root, version):
            rotation_degrees = [math.degrees(angle) for angle in (0.31, -0.47, 0.73)]
            return write_rotation_order(
                filepath,
                root,
                version,
                name,
                order,
                rotation_degrees,
            )

        encode_bin.write = write_ordered_fbx
        try:
            bpy.ops.export_scene.fbx(
                filepath=fbx_path,
                use_selection=True,
                use_space_transform=False,
                axis_forward="Y",
                axis_up="Z",
                bake_anim=kind == "Animated",
                bake_anim_use_nla_strips=False,
                bake_anim_use_all_actions=False,
                bake_anim_step=1.0,
                bake_anim_simplify_factor=0.0,
            )
        finally:
            encode_bin.write = original_write

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(
            filepath=fbx_path,
            use_manual_orientation=True,
            axis_forward="Y",
            axis_up="Z",
        )
        imported = bpy.data.objects.get(name)
        if imported is None:
            raise RuntimeError(f"Blender FBX importer omitted {name}")
        scene = bpy.context.scene
        world_rotations[name] = {}
        for sample_frame in frames:
            scene.frame_set(sample_frame)
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            depsgraph.update()
            rotation = imported.evaluated_get(depsgraph).matrix_world.to_quaternion()
            world_rotations[name][str(sample_frame)] = [
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ]

with open(os.path.join(root_dir, "rotation_orders_expected.json"), "w", encoding="utf-8") as output:
    json.dump(
        {
            "blender_version": bpy.app.version_string,
            "frame": frame,
            "world_rotations": world_rotations,
        },
        output,
        indent=2,
        sort_keys=True,
    )
