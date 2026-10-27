"""Generate the Blender 5.2.2 frame-based export visibility oracle.

Run with:
    blender -b --factory-startup --python tests/fixtures/export_visibility_frames/generate.py

Writes `export_visibility_frames.blend` and `expected.json` beside this script.

Every object's triangle count is a distinct power of two, so the triangle total identifies the
objects written by unnamed formats (PLY, STL). The hide flags of the objects are keyed in NLA
strips, so the active slot of each object stays empty; FBX and USD refuse active-action hide
curves but accept NLA evaluation of the same frame state.

Each frame is exported by Blender's own writers, one frame per export (Alembic would otherwise
write the whole scene range), then every file is read back by `inspect_exports.inspect_file`.
The test reads Potter's exports with the same function.

Writer options follow Blender's defaults. GLB/glTF set `use_visible=True`; their actual object filter is captured in this Blender-generated oracle rather than inferred from the option.
"""

import json
import sys
import tempfile
from pathlib import Path

import bpy


ROOT = Path(__file__).resolve().parent
BLEND_PATH = ROOT / "export_visibility_frames.blend"
EXPECTED_PATH = ROOT / "expected.json"
FRAMES = (1, 5)

sys.path.insert(0, str(ROOT))
import inspect_exports  # noqa: E402

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(
        f"export visibility oracle requires Blender 5.2.2, found {bpy.app.version_string}"
    )

# name, triangles, stored hide_viewport, stored hide_render, keyed flag -> {frame: value}
OBJECTS = (
    ("vis_static_visible", 1, False, False, {}),
    ("vis_static_render_off", 2, False, True, {}),
    ("vis_static_viewport_off", 4, True, False, {}),
    ("vis_render_fades_out", 8, False, False, {"hide_render": {1: False, 5: True}}),
    ("vis_render_fades_in", 16, False, True, {"hide_render": {1: True, 5: False}}),
    ("vis_viewport_fades_out", 32, False, False, {"hide_viewport": {1: False, 5: True}}),
)

# format, file extension, Blender writer
WRITERS = (
    ("obj", "obj", lambda path: bpy.ops.wm.obj_export(filepath=path)),
    ("ply", "ply", lambda path: bpy.ops.wm.ply_export(filepath=path)),
    ("stl", "stl", lambda path: bpy.ops.wm.stl_export(filepath=path)),
    (
        "glb",
        "glb",
        lambda path: bpy.ops.export_scene.gltf(
            filepath=path, export_format="GLB", use_visible=True
        ),
    ),
    (
        "gltf",
        "gltf",
        lambda path: bpy.ops.export_scene.gltf(
            filepath=path, export_format="GLTF_SEPARATE", use_visible=True
        ),
    ),
    ("fbx", "fbx", lambda path: bpy.ops.export_scene.fbx(filepath=path)),
    ("fbx-binary", "fbx", lambda path: bpy.ops.export_scene.fbx(filepath=path)),
    ("usda", "usda", lambda path: bpy.ops.wm.usd_export(filepath=path)),
    ("usdc", "usdc", lambda path: bpy.ops.wm.usd_export(filepath=path)),
    ("usdz", "usdz", lambda path: bpy.ops.wm.usd_export(filepath=path)),
    ("alembic", "abc", lambda path: bpy.ops.wm.alembic_export(filepath=path)),
)


UNNAMED_FORMATS = ("ply", "stl")


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = FRAMES[0]
    scene.frame_end = FRAMES[-1]
    for object_index, (name, triangles, hide_viewport, hide_render, keyed) in enumerate(OBJECTS):
        mesh = bpy.data.meshes.new(name)
        vertices = []
        faces = []
        for index in range(triangles):
            base = len(vertices)
            # Disjoint triangles: an STL import merges coincident faces and would undercount.
            x = object_index * 100.0 + index * 2.0
            vertices += [(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 1.0, 0.0)]
            faces.append((base, base + 1, base + 2))
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.hide_viewport = hide_viewport
        obj.hide_render = hide_render
        for path, values in keyed.items():
            for frame, value in values.items():
                setattr(obj, path, value)
                obj.keyframe_insert(data_path=path, frame=frame)
            setattr(obj, path, values[FRAMES[0]])
            action = obj.animation_data.action
            obj.animation_data.action = None
            track = obj.animation_data.nla_tracks.new()
            track.name = f"{path} visibility"
            track.strips.new(f"{name} {path}", FRAMES[0], action)
    # RNA default 0 yields 0.8; stored hidden 1 as the base would leave it hidden.
    name = "vis_stored_hidden_mix"
    triangles = 64
    object_index = len(OBJECTS)
    mesh = bpy.data.meshes.new(name)
    vertices = []
    faces = []
    for index in range(triangles):
        base = len(vertices)
        x = object_index * 100.0 + index * 2.0
        vertices += [(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 1.0, 0.0)]
        faces.append((base, base + 1, base + 2))
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    for path in ("hide_viewport", "hide_render"):
        for frame in FRAMES:
            obj.keyframe_insert(data_path=path, frame=frame)
        action = obj.animation_data.action
        obj.animation_data.action = None
        for blend_type, influence in (("REPLACE", 0.6), ("ADD", 0.2)):
            track = obj.animation_data.nla_tracks.new()
            track.name = f"{path} {blend_type}"
            strip = track.strips.new(f"{name} {path} {blend_type}", FRAMES[0], action)
            strip.blend_type = blend_type
            strip.use_animated_influence = True
            strip.influence = influence
            for frame in FRAMES:
                strip.keyframe_insert(data_path="influence", frame=frame)
    scene.frame_set(FRAMES[0])
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), compress=False)


def open_scene(frame):
    """Reopen the saved oracle scene with the export range pinned to `frame`."""
    bpy.ops.wm.open_mainfile(filepath=str(BLEND_PATH))
    bpy.context.scene.frame_start = frame
    bpy.context.scene.frame_end = frame
    bpy.context.scene.frame_set(frame)


def render_capture():
    """Render-visible mesh objects at the open frame, with their edge total.

    SVG and PDF have no Blender writer. Potter draws each mesh edge once per object, and the
    fixture's triangles share no vertices, so the edge total is the drawn line count.
    """
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH" and not obj.hide_render]
    return {
        "names": sorted(obj.name for obj in meshes),
        "edges": sum(len(obj.data.edges) for obj in meshes),
    }


def capture(frame, directory):
    open_scene(frame)
    files = {}
    for format_name, extension, writer in WRITERS:
        path = directory / f"f{frame}-{format_name}.{extension}"
        writer(str(path))
        files[format_name] = path
    # Read the Render-visibility state before any export is read, since reading replaces the scene.
    inspected = {"svg": render_capture(), "pdf": render_capture()}
    for format_name, path in files.items():
        result = inspect_exports.inspect_file(format_name, str(path))
        if format_name in UNNAMED_FORMATS:
            # PLY and STL name nothing; their triangle totals identify the objects.
            result = {"triangles": result["triangles"]}
        inspected[format_name] = result
    return inspected


def main():
    build_scene()
    directory = Path(tempfile.mkdtemp(prefix="export_visibility_frames_"))
    formats = {}
    for frame in FRAMES:
        formats[str(frame)] = capture(frame, directory)
    document = {
        "blender_version": bpy.app.version_string,
        "frames": [str(frame) for frame in FRAMES],
        "formats": formats,
    }
    EXPECTED_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {EXPECTED_PATH}")


if __name__ == "__main__":
    main()
