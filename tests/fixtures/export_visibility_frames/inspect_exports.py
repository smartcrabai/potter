"""Read exported scene files back with Blender 5.2.2 and report which mesh objects each file holds.

Run with:
    blender -b --factory-startup --python tests/fixtures/export_visibility_frames/inspect_exports.py -- FORMAT=PATH ...

Each argument names an export format and the file written in it. The script prints one line,
`POTTER_EXPORT_INSPECTION <json>`, keyed by argument. A mesh object is named by its `potter:name`
custom property when the file carries one (Potter's USD writer does), and by its object name
otherwise. `triangles` totals the polygons of every mesh object, which identifies the set of
objects in unnamed formats (PLY, STL), because each oracle object has a distinct triangle count.
`hide_viewport` is the flag Blender's FBX importer derives from the file's Visibility property.

generate.py reuses `inspect_file` on Blender's own writer output, so the expected inclusion and the
Potter export are read by the same consumer.
"""

import json
import sys

import bpy


MARKER = "POTTER_EXPORT_INSPECTION"

IMPORTERS = {
    "obj": lambda path: bpy.ops.wm.obj_import(filepath=path),
    "ply": lambda path: bpy.ops.wm.ply_import(filepath=path),
    "stl": lambda path: bpy.ops.wm.stl_import(filepath=path),
    "glb": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    "gltf": lambda path: bpy.ops.import_scene.gltf(filepath=path),
    "fbx": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    "fbx-binary": lambda path: bpy.ops.import_scene.fbx(filepath=path),
    "usda": lambda path: bpy.ops.wm.usd_import(filepath=path),
    "usdc": lambda path: bpy.ops.wm.usd_import(filepath=path),
    "usdz": lambda path: bpy.ops.wm.usd_import(filepath=path),
    "alembic": lambda path: bpy.ops.wm.alembic_import(filepath=path),
}


def inspect_file(format_name, path):
    """Import `path` into an empty scene and describe its mesh objects."""
    importer = IMPORTERS.get(format_name)
    if importer is None:
        raise RuntimeError(f"no Blender importer for format {format_name}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    importer(path)
    names = {}
    triangles = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        name = obj.get("potter:name", obj.name)
        names[name] = {"hide_viewport": bool(obj.hide_viewport)}
        triangles += len(obj.data.polygons)
    return {
        "names": sorted(names),
        "visible": {name: not value["hide_viewport"] for name, value in sorted(names.items())},
        "triangles": triangles,
    }


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    result = {}
    for argument in arguments:
        format_name, _, path = argument.partition("=")
        result[argument] = inspect_file(format_name, path)
    print(MARKER, json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
