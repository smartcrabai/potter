"""Regenerate the Blender 5.2 Ocean modifier parity corpus.

Run from any directory with:
    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        --python tests/fixtures/ocean_parity/generate.py

The script creates one deterministic scene per case, records source and evaluated
mesh topology/UV/color data, and bakes the requested frame sequences. Bake EXRs
are loaded through bpy and their pixel buffers plus on-disk channel precision are
stored in corpus.json; no binary bake files are retained in the fixture tree.
Render-evaluated meshes and corner colors are captured through the Blender USD
exporter's RENDER evaluation mode and imported back to expose topology/attributes;
this avoids the Y-up coordinate conversion used by the OBJ exporter. Background
evaluated_get() alone exposes only the viewport depsgraph.
Bake files live temporarily under /tmp/potter_ocean_parity_bake_v1.
"""

import json
import math
import os
import shutil
import struct
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.json")
SCRATCH_ROOT = "/tmp/potter_ocean_parity_bake_v1"

# Explicit values make every fixture independent of the user's saved preferences
# and make params a complete record of the modifier RNA values used by each case.
BASE_PARAMS = {
    "geometry_mode": "GENERATE",
    "resolution": 4,
    "viewport_resolution": 4,
    "spatial_size": 50,
    "size": 1.0,
    "repeat_x": 1,
    "repeat_y": 1,
    "use_normals": False,
    "use_foam": False,
    "use_spray": False,
    "invert_spray": False,
    "foam_layer_name": "Foam",
    "spray_layer_name": "Spray",
    "foam_coverage": 0.0,
    "random_seed": 1,
    "time": 0.5,
    "wave_scale": 1.0,
    "wave_scale_min": 0.01,
    "choppiness": 1.0,
    "wind_velocity": 30.0,
    "damping": 0.5,
    "wave_alignment": 0.0,
    "wave_direction": 0.0,
    "depth": 200.0,
    "spectrum": "PHILLIPS",
    "fetch_jonswap": 120.0,
    "sharpen_peak_jonswap": 0.0,
    "frame_start": 1,
    "frame_end": 4,
    "bake_foam_fade": 0.98,
    "filepath": "",
}

LIVE_CASES = [
    ("L1_PHILLIPS", "viewport", {"resolution": 4, "viewport_resolution": 4, "spatial_size": 50, "size": 1.0, "time": 0.5, "wave_scale": 1.0, "choppiness": 1.0, "random_seed": 1, "use_foam": True, "use_spray": True, "foam_coverage": 0.0, "foam_layer_name": "Foam", "spray_layer_name": "Spray"}),
    ("L2_PHILLIPS_wind", "viewport", {"random_seed": 17, "wind_velocity": 18.0, "wave_scale": 0.8, "wave_scale_min": 0.15, "wave_alignment": 0.35, "wave_direction": 0.6, "spatial_size": 16, "time": 2.25, "choppiness": 1.25}),
    ("L3_PIERSON_MOSKOWITZ", "viewport", {"spectrum": "PIERSON_MOSKOWITZ", "resolution": 3, "viewport_resolution": 3, "random_seed": 5, "wind_velocity": 20.0, "time": 1.0, "choppiness": 0.0, "use_spray": True}),
    ("L4_JONSWAP", "viewport", {"spectrum": "JONSWAP", "resolution": 4, "viewport_resolution": 4, "random_seed": 3, "fetch_jonswap": 120.0, "sharpen_peak_jonswap": 0.6, "wind_velocity": 25.0, "time": 1.5}),
    ("L5_TEXEL_MARSEN_ARSLOE", "viewport", {"spectrum": "TEXEL_MARSEN_ARSLOE", "resolution": 4, "viewport_resolution": 4, "random_seed": 3, "fetch_jonswap": 60.0, "sharpen_peak_jonswap": 0.3, "depth": 20.0, "time": 1.5}),
    ("L6_PHILLIPS_odd_M", "viewport", {"resolution": 5, "viewport_resolution": 5, "random_seed": 9, "damping": 0.0, "wave_alignment": 0.8, "wave_direction": 1.2, "time": 3.0}),
    ("L7_PHILLIPS_damping_one", "viewport", {"resolution": 4, "viewport_resolution": 4, "damping": 1.0, "wind_velocity": 5.0, "time": 0.0}),
    ("L8_invert_spray", "viewport", {"resolution": 4, "viewport_resolution": 4, "invert_spray": True, "use_foam": True, "use_spray": True, "foam_coverage": 0.3}),
    ("L9_empty_foam_name", "viewport", {"resolution": 4, "viewport_resolution": 4, "foam_coverage": 0.8, "use_foam": True, "foam_layer_name": ""}),
    ("L9_empty_foam_and_spray_names", "viewport", {"resolution": 4, "viewport_resolution": 4, "foam_coverage": 0.8, "use_foam": True, "use_spray": True, "foam_layer_name": "", "spray_layer_name": ""}),
    ("L10_repeat_layout", "viewport", {"resolution": 3, "viewport_resolution": 3, "repeat_x": 2, "repeat_y": 3, "size": 2.0, "spatial_size": 20, "time": 0.7, "use_foam": True, "use_spray": True}),
    ("L11_wave_scale_zero", "viewport", {"resolution": 4, "viewport_resolution": 4, "wave_scale": 0.0}),
    ("L11_wave_scale_2_5", "viewport", {"resolution": 4, "viewport_resolution": 4, "wave_scale": 2.5}),
    ("L12_normals_with_foam", "viewport", {"resolution": 4, "viewport_resolution": 4, "use_normals": True, "use_foam": True}),
    ("L13_render", "render", {"resolution": 5, "viewport_resolution": 3, "use_foam": True, "use_spray": True}),
    ("L13_viewport", "viewport", {"resolution": 5, "viewport_resolution": 3, "use_foam": True, "use_spray": True}),
]

BAKE_CASES = [
    ("B1_cache_behaviour", {"use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4, "bake_foam_fade": 0.98}),
    ("B2_fade_0_5", {"use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4, "bake_foam_fade": 0.5}),
    ("B2_fade_0", {"use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4, "bake_foam_fade": 0.0}),
    ("B3_normals", {"use_normals": True, "use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4}),
    ("B4_free_and_repeat", {"use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4}),
    ("B5_animated_time", {"use_foam": True, "use_spray": True, "frame_start": 1, "frame_end": 4}),

]


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_source(case_name):
    if case_name == "D1_displace_name_collision":
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=2, size=2.0)
        obj = bpy.context.object
        obj.location = (1.25, -0.5, 3.0)
        obj.scale = (1.5, 0.75, 1.25)
        for vertex in obj.data.vertices:
            vertex.co.z += 0.75
        foam = obj.data.color_attributes.new(name="Foam", type="BYTE_COLOR", domain="CORNER")
        for item in foam.data:
            item.color = (0.25, 0.5, 0.75, 1.0)
        return obj
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=1.0)
    return bpy.context.object


def attribute_descriptors(mesh):
    return [
        {"name": attr.name, "domain": attr.domain, "data_type": attr.data_type}
        for attr in mesh.attributes
        if attr.name != "position" and not attr.name.startswith(".")
    ]


def mesh_dump(mesh):
    colors = {}
    for attr in mesh.attributes:
        if attr.name == "position" or attr.name.startswith("."):
            continue
        if attr.data_type not in ("BYTE_COLOR", "FLOAT_COLOR"):
            continue
        colors[attr.name] = {
            "domain": attr.domain,
            "data_type": attr.data_type,
            "values": [list(item.color) for item in attr.data],
        }
    uv = None
    if mesh.uv_layers:
        uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        uv = [list(item.uv) for item in uv_layer.data]
    return {
        "vertices": [list(vertex.co) for vertex in mesh.vertices],
        "polygons": [list(poly.vertices) for poly in mesh.polygons],
        "loops": [loop.vertex_index for loop in mesh.loops],
        "uv": uv,
        "color_layers": colors,
        "attribute_names": [item["name"] for item in attribute_descriptors(mesh)],
        "attributes": attribute_descriptors(mesh),
    }


def scene_source_dump(obj):
    source = mesh_dump(obj.data)
    source["transform"] = {
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
    }
    return source


def set_ocean_params(modifier, overrides, filepath=None):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    if filepath is not None:
        params["filepath"] = filepath
    for name, value in params.items():
        setattr(modifier, name, value)
    # Record Blender's actual RNA values (for example, float properties are
    # rounded to Blender's stored float precision).
    return {name: getattr(modifier, name) for name in params}


def evaluated_dump(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return mesh_dump(mesh)
    finally:
        evaluated.to_mesh_clear()


def signature(dump):
    return {
        "vertices": dump["vertices"],
        "color_layers": dump["color_layers"],
        "polygons": dump["polygons"],
        "loops": dump["loops"],
    }


def render_mesh_and_colors(obj, case_dir):
    os.makedirs(case_dir, exist_ok=True)
    usd_path = os.path.join(case_dir, "render.usda")
    bpy.ops.wm.usd_export(
        filepath=usd_path,
        evaluation_mode="RENDER",
        export_mesh_colors=True,
    )
    existing = {item.as_pointer() for item in bpy.data.objects}
    bpy.ops.wm.usd_import(filepath=usd_path)
    imported = [
        item for item in bpy.data.objects
        if item.type == "MESH" and item.as_pointer() not in existing
    ]
    if len(imported) != 1:
        raise RuntimeError("Expected one imported render-evaluated mesh, got %d" % len(imported))
    dump = mesh_dump(imported[0].data)
    return dump, {
        "positions_export": "bpy.ops.wm.usd_export(evaluation_mode='RENDER'), then bpy.ops.wm.usd_import",
        "corner_colors_export": "bpy.ops.wm.usd_export(evaluation_mode='RENDER', export_mesh_colors=True), then bpy.ops.wm.usd_import",
        "usd_import_color_data_type": sorted({v["data_type"] for v in dump["color_layers"].values()}),
        "render_vertex_count": len(dump["vertices"]),
    }



def exr_channel_types(path):
    with open(path, "rb") as stream:
        blob = stream.read()
    offset = 8
    result = {}
    while offset < len(blob):
        end = blob.index(b"\0", offset)
        name = blob[offset:end].decode("ascii")
        offset = end + 1
        if not name:
            break
        end = blob.index(b"\0", offset)
        offset = end + 1  # attribute type
        size = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        payload = blob[offset:offset + size]
        offset += size
        if name != "channels":
            continue
        pos = 0
        while pos < len(payload):
            end = payload.index(0, pos)
            channel = payload[pos:end].decode("ascii")
            pos = end + 1
            if not channel:
                break
            pixel_type = struct.unpack_from("<i", payload, pos)[0]
            pos += 16  # pixel type, pLinear/reserved bytes, xSampling, ySampling
            result[channel] = {0: "uint", 1: "half", 2: "float"}[pixel_type]
    return result


def load_exr(path):
    image = bpy.data.images.load(path, check_existing=False)
    try:
        channel_types = exr_channel_types(path)
        types = set(channel_types.values())
        return {
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "channels": int(image.channels),
            "stored_channel_types": channel_types,
            "values_are_as_stored": len(types) == 1 and next(iter(types)) == "half",
            "pixel_values_rgba_row_major": [float(value) for value in image.pixels],
        }
    finally:
        bpy.data.images.remove(image)


def wait_for_bake(case_dir, frames, use_normals, timeout=120.0, required=True):
    types = ["disp", "foam", "spray", "spray_inverse"]
    if use_normals:
        types.append("normal")
    paths = [
        os.path.join(case_dir, "%s_%04d.exr" % (kind, frame))
        for frame in range(1, frames + 1)
        for kind in types
    ]
    deadline = time.monotonic() + timeout
    stable = 0
    previous = None
    while time.monotonic() < deadline:
        if all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths):
            current = tuple(os.path.getsize(path) for path in paths)
            if current == previous:
                stable += 1
                if stable >= 2:
                    return paths
            else:
                previous = current
                stable = 0
        time.sleep(0.05)
    if required:
        raise TimeoutError("Ocean bake did not produce all EXR files: %s" % case_dir)
    return []


def bake_images(case_dir, frames, use_normals):
    files = {}
    for frame in range(1, frames + 1):
        types = ["disp", "foam", "spray", "spray_inverse"]
        if use_normals:
            types.append("normal")
        for kind in types:
            filename = "%s_%04d.exr" % (kind, frame)
            files[filename] = load_exr(os.path.join(case_dir, filename))
    return files


def bake_case(name, overrides):
    reset_scene()
    obj = add_source(name)
    source = scene_source_dump(obj)
    modifier = obj.modifiers.new("Ocean", "OCEAN")
    case_dir = os.path.join(SCRATCH_ROOT, name)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)
    filepath = case_dir + os.sep
    params = set_ocean_params(modifier, overrides, filepath)
    scene = bpy.context.scene
    scene.frame_set(1)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    animated_time_by_frame = None
    animated_time_observation = None
    if name == "B5_animated_time":
        time_keys = ((1, 0.5), (2, 1.5), (4, 3.5))
        for frame, value in time_keys:
            scene.frame_set(frame)
            modifier.time = value
            modifier.keyframe_insert(data_path="time", frame=frame)
        action = obj.animation_data.action
        action_slot = obj.animation_data.action_slot
        if action_slot is None:
            raise RuntimeError("animated Ocean bake did not get an Action slot")
        time_path = modifier.path_from_id("time")
        channel_curves = []
        for layer in action.layers:
            for strip in layer.strips:
                channelbag = strip.channelbag(action_slot)
                if channelbag is not None:
                    channel_curves.extend(
                        {"data_path": curve.data_path, "array_index": curve.array_index}
                        for curve in channelbag.fcurves
                    )
        scene.frame_set(1)
        animated_time_by_frame = {}
        for frame in range(int(params["frame_start"]), int(params["frame_end"]) + 1):
            scene.frame_set(frame)
            animated_time_by_frame[str(frame)] = float(modifier.time)
        animated_time_observation = {
            "time_data_path": time_path,
            "action_slot_identifier": getattr(action_slot, "identifier", ""),
            "channel_curves": channel_curves,
            "keyframes": [{"frame": frame, "value": value} for frame, value in time_keys],
            "modifier_time_by_frame": animated_time_by_frame,
        }
        scene.frame_set(1)
    pre_bake_frame2 = None
    if name == "B4_free_and_repeat":
        scene.frame_set(2)
        pre_bake_frame2 = evaluated_dump(obj)
    bake_result = bpy.ops.object.ocean_bake(modifier=modifier.name, free=False)
    wait_for_bake(case_dir, int(params["frame_end"]), bool(params["use_normals"]))
    exr_files = bake_images(case_dir, int(params["frame_end"]), bool(params["use_normals"]))
    observations = {
        "bake_operator_result": sorted(bake_result),
        "is_cached_immediately_after_bake": bool(modifier.is_cached),
        "exr_files": sorted(exr_files),
    }
    if animated_time_observation is not None:
        observations["animated_time"] = animated_time_observation
    case = {
        "name": name,
        "purpose": "viewport",
        "params": params,
        "source_mesh": source,
        "exr_files": exr_files,
        "observations": observations,
    }
    if name not in ("B1_cache_behaviour", "B4_free_and_repeat"):
        scene.frame_set(2)
        case["output_frame"] = 2
        case["output"] = evaluated_dump(obj)
    if name == "B1_cache_behaviour":
        samples = {}
        baseline_frame2 = None
        for frame in (0, 1, 2, 3, 4, 7):
            scene.frame_set(frame)
            output = evaluated_dump(obj)
            samples[str(frame)] = {
                "is_cached": bool(modifier.is_cached),
                "output": output,
            }
            if frame == 2:
                baseline_frame2 = output
                case["output_frame"] = 2
                case["output"] = baseline_frame2
        old_time = float(modifier.time)
        modifier.time = old_time + 1.0
        scene.frame_set(2)
        after_time = evaluated_dump(obj)
        time_cached_equal = signature(after_time) == signature(baseline_frame2)
        time_is_cached = bool(modifier.is_cached)
        old_choppiness = float(modifier.choppiness)
        modifier.choppiness = 0.0
        scene.frame_set(2)
        after_choppiness = evaluated_dump(obj)
        chop_cached_equal = signature(after_choppiness) == signature(baseline_frame2)
        observations.update({
            "samples_by_scene_frame": samples,
            "post_bake_param_change_tests": {
                "cache_active_before_choppiness_change": time_is_cached,
                "time": {
                    "from": old_time,
                    "to": float(modifier.time),
                    "frame": 2,
                    "output_equal_to_pre_change_frame2": time_cached_equal,
                    "is_cached_after_change": time_is_cached,
                },
                "choppiness": {
                    "from": old_choppiness,
                    "to": float(modifier.choppiness),
                    "crosses_zero": True,
                    "frame": 2,
                    "output_equal_to_original_baked_frame2": chop_cached_equal,
                    "is_cached_after_change": bool(modifier.is_cached),
                },
            },
        })
    elif name == "B4_free_and_repeat":
        free_result = bpy.ops.object.ocean_bake(modifier=modifier.name, free=True)
        scene.frame_set(2)
        after_free = evaluated_dump(obj)
        observations["free_operator_result"] = sorted(free_result)
        observations["free_output_equal_to_pre_bake_live_frame2"] = signature(after_free) == signature(pre_bake_frame2)
        observations["is_cached_after_free"] = bool(modifier.is_cached)
        case["output_frame"] = 2
        case["output"] = after_free
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
        os.makedirs(case_dir, exist_ok=True)
        second_bake_result = bpy.ops.object.ocean_bake(modifier=modifier.name, free=False)
        second_bake_files = wait_for_bake(
            case_dir,
            int(params["frame_end"]),
            bool(params["use_normals"]),
            timeout=5.0,
            required=False,
        )
        observations["second_bake_operator_result"] = sorted(second_bake_result)
        observations["second_bake_file_wait_timeout_seconds"] = 5
        observations["second_bake_files_present"] = bool(second_bake_files)
        second_free_result = bpy.ops.object.ocean_bake(modifier=modifier.name, free=True)
        scene.frame_set(2)
        observations["second_free_operator_result"] = sorted(second_free_result)
        observations["second_free_output_equal_to_pre_bake_live_frame2"] = signature(evaluated_dump(obj)) == signature(pre_bake_frame2)
    shutil.rmtree(case_dir)
    return case


def live_case(name, purpose, overrides):
    reset_scene()
    obj = add_source(name)
    source = scene_source_dump(obj)
    modifier = obj.modifiers.new("Ocean", "OCEAN")
    params = set_ocean_params(modifier, overrides)
    if purpose == "render":
        output, capture = render_mesh_and_colors(obj, os.path.join(SCRATCH_ROOT, name))
    else:
        output = evaluated_dump(obj)
        capture = {"positions_export": "evaluated_get(depsgraph).to_mesh()", "corner_colors_export": "evaluated mesh BYTE_COLOR/FLOAT_COLOR attributes"}
    case = {
        "name": name,
        "purpose": purpose,
        "params": params,
        "source_mesh": source,
        "output": output,
        "capture": capture,
    }
    if name == "L10_repeat_layout":
        uv = output["uv"] or []
        case["observations"] = {
            "repeat_x": params["repeat_x"],
            "repeat_y": params["repeat_y"],
            "uv_min": [min(point[axis] for point in uv) for axis in (0, 1)] if uv else None,
            "uv_max": [max(point[axis] for point in uv) for axis in (0, 1)] if uv else None,
            "vertices": len(output["vertices"]),
            "polygons": len(output["polygons"]),
        }
    if name == "L12_normals_with_foam":
        case["observations"] = {
            "ocean_normal_style_attributes": [
                attr["name"] for attr in output["attributes"]
                if "ocean_normal" in attr["name"].lower() or attr["name"].lower() == "normal"
            ],
        }
    if name == "L3_PIERSON_MOSKOWITZ":
        spray_layer = output["color_layers"].get(params["spray_layer_name"])
        spray = spray_layer["values"] if spray_layer is not None else []
        case["observations"] = {
            "spray_layer_present": spray_layer is not None,
            "spray_values_all_finite": all(math.isfinite(value) for row in spray for value in row) if spray_layer is not None else None,
            "spray_values": spray,
        }
    if name == "L9_empty_foam_and_spray_names":
        case["observations"] = {
            "requested_foam_layer_name": params["foam_layer_name"],
            "requested_spray_layer_name": params["spray_layer_name"],
            "generated_color_layer_names": list(output["color_layers"]),
            "matches_expected_default_names": list(output["color_layers"]) == ["Attribute", "Attribute.001"],
        }
    return case


def displace_case():
    name = "D1_displace_name_collision"
    overrides = {
        "geometry_mode": "DISPLACE",
        "resolution": 4,
        "viewport_resolution": 4,
        "spectrum": "JONSWAP",
        "choppiness": 1.0,
        "use_foam": True,
        "use_spray": True,
        "invert_spray": False,
        "foam_layer_name": "Foam",
        "spray_layer_name": "Spray",
    }
    return live_case(name, "viewport", overrides)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    output_path = args[0] if args else CORPUS_PATH
    if os.path.isdir(SCRATCH_ROOT):
        shutil.rmtree(SCRATCH_ROOT)
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    corpus = {
        "format": "potter-ocean-parity-corpus",
        "format_version": 1,
        "blender_version": bpy.app.version_string,
        "cases": [],
    }
    for name, purpose, params in LIVE_CASES:
        corpus["cases"].append(live_case(name, purpose, params))
    corpus["cases"].append(displace_case())
    for name, params in BAKE_CASES:
        corpus["cases"].append(bake_case(name, params))
    # Explicit sensitivity comparisons keep the fade experiment interpretable.
    by_name = {case["name"]: case for case in corpus["cases"]}
    for name in ("B2_fade_0_5", "B2_fade_0"):
        foam_differences = {}
        case = by_name[name]
        reference_name = "B1_cache_behaviour"
        for frame in range(1, 5):
            filename = "foam_%04d.exr" % frame
            left = case["exr_files"][filename]["pixel_values_rgba_row_major"]
            right = by_name[reference_name]["exr_files"][filename]["pixel_values_rgba_row_major"]
            foam_differences[str(frame)] = max((abs(a - b) for a, b in zip(left, right)), default=0.0)
        case["observations"]["max_abs_foam_exr_diff_vs_bake_fade_0_98"] = foam_differences
    shutil.rmtree(SCRATCH_ROOT)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(corpus, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    size = os.path.getsize(output_path)
    print("WROTE %s (%d bytes; %d cases)" % (output_path, size, len(corpus["cases"])))
    print("CASES " + ", ".join(case["name"] for case in corpus["cases"]))


if __name__ == "__main__":
    main()
