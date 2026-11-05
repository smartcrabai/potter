"""Capture the same Holdout/Diffuse graph under EEVEE DITHERED and BLENDED modes.

Regenerate with Blender 5.2.2:

    Blender -b --factory-startup --threads 1 \
        -P tests/fixtures/native_oracle/holdout_surface_render_method/generate.py -- \
        tests/fixtures/native_oracle/holdout_surface_render_method
"""



import json
import os
import sys

import bpy
from mathutils import Vector

WIDTH = 128
HEIGHT = 128
ORTHO_SCALE = 8.0
PATCH_SIZE = 4.0
TARGET_PIXEL = (64, 64)
BACKPLATE_PIXEL = (16, 64)
WORLD_PIXEL = (112, 64)
PROBE_CORNER = (40, 40)
BACKPLATE_CORNER = (8, 56)
WORLD_CORNER = (104, 56)

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

args = sys.argv[sys.argv.index("--") + 1 :]
if len(args) != 1:
    raise RuntimeError("usage: generate.py -- <output-directory>")
out_dir = os.path.abspath(args[0])
os.makedirs(out_dir, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
for existing_material in list(bpy.data.materials):
    bpy.data.materials.remove(existing_material)
scene.name = "Holdout Surface Render Method"
engine_ids = [item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
scene.render.engine = next(engine for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if engine in engine_ids)
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.threads_mode = "FIXED"
scene.render.threads = 1
scene.eevee.taa_render_samples = 1
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "32"
scene.render.image_settings.exr_codec = "NONE"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

world = bpy.data.worlds.new("Black World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
scene.world = world

camera_data = bpy.data.cameras.new("Holdout Surface Method Camera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = ORTHO_SCALE
camera = bpy.data.objects.new("Holdout Surface Method Camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 10.0)
camera.rotation_euler = (0.0, 0.0, 0.0)
scene.camera = camera

backplate_color = (0.8, 0.8, 0.8, 1.0)
backplate_material = bpy.data.materials.new("Opaque Diffuse Backplate")
backplate_material.use_nodes = True
backplate_material.surface_render_method = "DITHERED"
backplate_tree = backplate_material.node_tree
backplate_tree.nodes.clear()
backplate_output = backplate_tree.nodes.new("ShaderNodeOutputMaterial")
backplate_diffuse = backplate_tree.nodes.new("ShaderNodeBsdfDiffuse")
backplate_diffuse.inputs["Color"].default_value = backplate_color
backplate_diffuse.inputs["Roughness"].default_value = 0.0
backplate_tree.links.new(backplate_diffuse.outputs["BSDF"], backplate_output.inputs["Surface"])
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(-3.0, 0.0, -0.05))
backplate_object = bpy.context.object
backplate_object.name = "Opaque Diffuse Backplate Control"
backplate_object.data.materials.append(backplate_material)

probe_color = (0.8, 0.32, 0.12, 1.0)
bpy.ops.mesh.primitive_plane_add(size=PATCH_SIZE, location=(0.0, 0.0, 0.0))
probe_object = bpy.context.object
probe_object.name = "Holdout Diffuse Probe Patch"

sun_data = bpy.data.lights.new("Fixed White Point Sun", "SUN")
sun_data.energy = 1.0
sun_data.color = (1.0, 1.0, 1.0)
sun_data.angle = 0.0
sun_data.use_shadow = True
sun = bpy.data.objects.new("Fixed White Point Sun", sun_data)
scene.collection.objects.link(sun)
sun.location = (0.0, 0.0, 10.0)
sun.rotation_euler = (0.0, 0.0, 0.0)

captures = {}
previous_probe_material = None

for mode in ("DITHERED", "BLENDED"):
    if previous_probe_material is not None:
        probe_object.data.materials.clear()
        bpy.data.materials.remove(previous_probe_material)
    probe_material = bpy.data.materials.new("Holdout 0.5 Plus Opaque Diffuse 0.5")
    probe_material.surface_render_method = mode
    probe_material.use_nodes = True
    probe_tree = probe_material.node_tree
    probe_tree.nodes.clear()
    probe_output = probe_tree.nodes.new("ShaderNodeOutputMaterial")
    probe_holdout = probe_tree.nodes.new("ShaderNodeHoldout")
    probe_diffuse = probe_tree.nodes.new("ShaderNodeBsdfDiffuse")
    probe_diffuse.inputs["Color"].default_value = probe_color
    probe_diffuse.inputs["Roughness"].default_value = 0.0
    probe_mix = probe_tree.nodes.new("ShaderNodeMixShader")
    probe_mix.inputs["Fac"].default_value = 0.5
    probe_tree.links.new(probe_holdout.outputs["Holdout"], probe_mix.inputs[1])
    probe_tree.links.new(probe_diffuse.outputs["BSDF"], probe_mix.inputs[2])
    probe_tree.links.new(probe_mix.outputs["Shader"], probe_output.inputs["Surface"])
    probe_object.data.materials.append(probe_material)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def ray_provenance(pixel, expected_object, expected_material):
        x_ndc = (pixel[0] + 0.5) / WIDTH - 0.5
        y_ndc = (pixel[1] + 0.5) / HEIGHT - 0.5
        ray_origin = camera.matrix_world @ Vector(
            (x_ndc * ORTHO_SCALE, y_ndc * ORTHO_SCALE, 0.0)
        )
        ray_direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        hit, location, normal, face_index, hit_object, matrix = scene.ray_cast(
            depsgraph, ray_origin, ray_direction
        )
        if expected_object is None:
            if hit:
                raise RuntimeError(f"transparent-world ray hit {hit_object.name} at {pixel}")
            return {
                "pixel_xy": list(pixel),
                "hit": False,
                "object_id": None,
                "face_index": None,
                "material_index": None,
                "evaluated_material_id": None,
            }
        if not hit or hit_object.name_full != expected_object.name_full:
            raise RuntimeError(
                f"ray at {pixel} expected {expected_object.name_full}, "
                f"hit {getattr(hit_object, 'name_full', None)}"
            )
        evaluated_object = hit_object.evaluated_get(depsgraph)
        face = evaluated_object.data.polygons[face_index]
        slots = evaluated_object.material_slots
        evaluated_material = (
            slots[face.material_index].material
            if face.material_index < len(slots)
            else None
        )
        if evaluated_material is None or evaluated_material.name_full != expected_material.name_full:
            raise RuntimeError(
                f"ray at {pixel} resolved to material "
                f"{getattr(evaluated_material, 'name_full', None)}, "
                f"expected {expected_material.name_full}"
            )
        return {
            "pixel_xy": list(pixel),
            "hit": True,
            "object_id": hit_object.name_full,
            "face_index": face_index,
            "material_index": face.material_index,
            "evaluated_material_id": evaluated_material.name_full,
        }

    raycast_provenance = {
        "probe_target": {
            "expected_object_id": probe_object.name_full,
            "expected_evaluated_material_id": probe_material.name_full,
            "center": ray_provenance(TARGET_PIXEL, probe_object, probe_material),
            "interior_corner": ray_provenance(PROBE_CORNER, probe_object, probe_material),
        },
        "opaque_backplate_control": {
            "expected_object_id": backplate_object.name_full,
            "expected_evaluated_material_id": backplate_material.name_full,
            "center": ray_provenance(BACKPLATE_PIXEL, backplate_object, backplate_material),
            "interior_corner": ray_provenance(
                BACKPLATE_CORNER, backplate_object, backplate_material
            ),
        },
        "transparent_world": {
            "expected_hit": False,
            "center": ray_provenance(WORLD_PIXEL, None, None),
            "interior_corner": ray_provenance(WORLD_CORNER, None, None),
        },
    }


    base = "holdout_" + mode.lower()
    blend_path = os.path.join(out_dir, base + ".blend")
    exr_path = os.path.join(out_dir, base + ".exr")
    scene.render.filepath = "//" + base + ".exr"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.render.render(write_still=False)
    bpy.data.images["Render Result"].save_render(exr_path, scene=scene)

    image = bpy.data.images.load(exr_path, check_existing=False)
    if tuple(image.size) != (WIDTH, HEIGHT) or image.channels != 4:
        raise RuntimeError(f"unexpected {mode} EXR shape/channels: {image.size[:]} / {image.channels}")

    def rgba_at(x, y):
        offset = (y * WIDTH + x) * image.channels
        return [float(channel) for channel in image.pixels[offset : offset + 4]]

    captures[mode] = {
        "surface_render_method": probe_material.surface_render_method,
        "raycast_provenance": raycast_provenance,
        "source_alpha_mode": image.alpha_mode,
        "target": {
            "pixel_xy": list(TARGET_PIXEL),
            "raw_associated_rgba": rgba_at(*TARGET_PIXEL),
        },
        "transparent_world_neighbor": {
            "pixel_xy": list(WORLD_PIXEL),
            "raw_associated_rgba": rgba_at(*WORLD_PIXEL),
        },
        "opaque_backplate_control": {
            "pixel_xy": list(BACKPLATE_PIXEL),
            "raw_associated_rgba": rgba_at(*BACKPLATE_PIXEL),
        },
        "artifacts": {"blend": os.path.basename(blend_path), "exr": os.path.basename(exr_path)},
    }
    bpy.data.images.remove(image)
    previous_probe_material = probe_material

scene_inputs = {
    "engine": scene.render.engine,
    "scene_objects": sorted(obj.name for obj in scene.objects),
    "resolution": [WIDTH, HEIGHT],
    "resolution_percentage": scene.render.resolution_percentage,
    "render_samples": scene.eevee.taa_render_samples,
    "threads": scene.render.threads,
    "film_transparent": scene.render.film_transparent,
    "view_transform": scene.view_settings.view_transform,
    "view_look": scene.view_settings.look,
    "exposure": scene.view_settings.exposure,
    "gamma": scene.view_settings.gamma,
    "camera": {
        "type": camera_data.type,
        "location": list(camera.location),
        "rotation_euler": list(camera.rotation_euler),
        "ortho_scale": camera_data.ortho_scale,
    },
    "sun": {
        "type": sun_data.type,
        "location": list(sun.location),
        "rotation_euler": list(sun.rotation_euler),
        "color": list(sun_data.color),
        "energy": sun_data.energy,
        "angle": getattr(sun_data, "angle", None),
        "use_shadow": sun_data.use_shadow,
    },
    "world_background_color": [0.0, 0.0, 0.0, 1.0],
    "world_background_strength": 0.0,
    "backplate": {
        "size_world_units": 2.0,
        "location": list(backplate_object.location),
        "surface_render_method": backplate_material.surface_render_method,
        "shader": "Diffuse BSDF -> Material Output",
        "color_rgba": list(backplate_color),
        "roughness": 0.0,
    },
    "probe": {
        "size_world_units": PATCH_SIZE,
        "location": list(probe_object.location),
        "holdout_mix_factor": probe_mix.inputs["Fac"].default_value,
        "graph": "Holdout -> Mix Shader input 1; opaque Diffuse BSDF -> input 2; Mix Shader -> Material Output",
        "diffuse_color_rgba": list(probe_color),
        "diffuse_roughness": probe_diffuse.inputs["Roughness"].default_value,
        "target_pixel_xy": list(TARGET_PIXEL),
        "backplate_control_pixel_xy": list(BACKPLATE_PIXEL),
        "transparent_world_neighbor_pixel_xy": list(WORLD_PIXEL),
        "pixel_note": "Probe target is 32 pixels from its edges; the adjacent backplate control is 16 pixels from its edges.",
    },
    "exr": {
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "codec": scene.render.image_settings.exr_codec,
        "captured_rgba": "raw source EXR RGBA; no unassociation or color conversion",
    },
}
capture = {
    "blender_version": bpy.app.version_string,
    "blender_build_hash": bpy.app.build_hash.decode("utf-8"),
    "scene_inputs": scene_inputs,
    "captures_by_surface_render_method": captures,
}
json_path = os.path.join(out_dir, "holdout_surface_render_method.json")
with open(json_path, "w", encoding="utf-8") as capture_file:
    json.dump(capture, capture_file, indent=2)
    capture_file.write("\n")
print("HOLDOUT_SURFACE_RENDER_METHOD_CAPTURE", json.dumps(captures, sort_keys=True), flush=True)
