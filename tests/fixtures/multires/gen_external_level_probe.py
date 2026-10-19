"""Probe how Blender 5.2.2 reads and writes external multires (BCDF/CD_MDISPS) sidecars.

Run exactly:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 --python gen_external_level_probe.py

Outputs (next to this script):
  probe_uniform_level2.btx        written by Blender's own multires_external_save (total_levels=2, 24 corners)
  probe_uniform_level2.blend      .blend whose external layer points at the file above
  probe_uniform_level1_total2.btx Potter-layout file, every corner at level 1, modifier total_levels=2
  probe_mixed_levels_total2.btx   Potter-layout file, corners alternate levels 2 and 1, total_levels=2
  probe_report.json               sizes, header fields, per-corner offsets, and reload results

Reload results are printed to stdout; Blender's CLOG errors (for example "failed to read multires
displacement") go to stderr. Capture both with a redirect when running.
"""

import json
import os
import struct
import sys

import bpy

OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
TOTAL_LEVELS = 2
HEADER_BYTES = 112  # 20 file header + 4 mesh header + 88 layer header
CORNERS = 24  # cube: 6 quads


def grid_vectors(level):
    # Blender multires_grid_tot[level]: side = 2**(level-1)+1, samples = side*side; level 0 has none.
    if level == 0:
        return 0
    side = (1 << (level - 1)) + 1
    return side * side


def bcdf_bytes(levels):
    """Mirror the layout in multires_external.rs write_external_grids_for_grid_lengths."""
    lengths = [grid_vectors(level) for level in levels]
    data_size = 12 * sum(lengths)
    head = b"BCDF" + bytes([0, 0, 0, 0])
    head += struct.pack("<IIII", 20, 1, 1, 4)  # file hdr size, mesh type, layer count, mesh hdr size
    head += struct.pack("<II", 88, 0)  # layer hdr size, float datatype
    head += struct.pack("<Q", data_size)
    head += struct.pack("<I", 19)  # CD_MDISPS
    head += bytes(68)  # empty layer name
    assert len(head) == HEADER_BYTES, len(head)
    body = bytearray()
    for corner, length in enumerate(lengths):
        for _ in range(length):
            body += struct.pack("<fff", 0.0, 0.0, 0.0)
    return head + bytes(body), lengths


def parse_header(data):
    fields = {
        "magic": data[0:4].decode("ascii", "replace"),
        "endian": data[4],
        "version": data[5],
        "subversion": data[6],
        "file_header_size": struct.unpack_from("<I", data, 8)[0],
        "file_type": struct.unpack_from("<I", data, 12)[0],
        "layer_count": struct.unpack_from("<I", data, 16)[0],
        "mesh_header_size": struct.unpack_from("<I", data, 20)[0],
        "layer_header_size": struct.unpack_from("<I", data, 24)[0],
        "layer_datatype": struct.unpack_from("<I", data, 28)[0],
        "layer_data_size": struct.unpack_from("<Q", data, 32)[0],
        "layer_type": struct.unpack_from("<I", data, 40)[0],
    }
    return fields


def corner_offsets(lengths):
    """Blender cdf_read_layer seeks to the layer data start; layerRead_mdisps then reads
    sizeof(float[3]) * totdisp per corner in corner order, so corner i starts at the sum of prior sizes."""
    offsets = []
    cursor = HEADER_BYTES
    for length in lengths:
        offsets.append(cursor)
        cursor += 12 * length
    return offsets


def write_bytes(name, data):
    path = os.path.join(OUT, name)
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def with_cube_override(obj):
    return bpy.context.temp_override(
        object=obj, active_object=obj, selected_objects=[obj], selected_editable_objects=[obj])


report = {"blender_version": bpy.app.version_string, "total_levels": TOTAL_LEVELS}

# 1. Blender-native uniform write at the top level.
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Multires", type="MULTIRES")
with with_cube_override(obj):
    for _ in range(TOTAL_LEVELS):
        assert bpy.ops.object.multires_subdivide(modifier=mod.name, mode="CATMULL_CLARK") == {"FINISHED"}
assert mod.total_levels == TOTAL_LEVELS, mod.total_levels

uniform_btx = os.path.join(OUT, "probe_uniform_level2.btx")
with with_cube_override(obj):
    result = bpy.ops.object.multires_external_save(
        modifier=mod.name, filepath=uniform_btx, relative_path=False)
print("PROBE external_save result", result, "is_external", mod.is_external)
uniform_bytes = open(uniform_btx, "rb").read()
uniform_header = parse_header(uniform_bytes)
blender_lengths = [grid_vectors(TOTAL_LEVELS)] * CORNERS
report["blender_uniform_level2"] = {
    "file": os.path.basename(uniform_btx),
    "file_bytes": len(uniform_bytes),
    "expected_bytes_for_uniform_top_level": HEADER_BYTES + 12 * sum(blender_lengths),
    "header": uniform_header,
    "per_corner_offsets_first_4": corner_offsets(blender_lengths)[:4],
    "per_corner_stride": 12 * grid_vectors(TOTAL_LEVELS),
}
print("PROBE blender uniform bytes", len(uniform_bytes), uniform_header)

blend_path = os.path.join(OUT, "probe_uniform_level2.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path, check_existing=False, relative_remap=False)

# 2. Python-written variants in Potter's layout.
level1_bytes, level1_lengths = bcdf_bytes([1] * CORNERS)
level1_path = write_bytes("probe_uniform_level1_total2.btx", level1_bytes)
mixed_levels = [2 if corner % 2 == 0 else 1 for corner in range(CORNERS)]
mixed_bytes, mixed_lengths = bcdf_bytes(mixed_levels)
mixed_path = write_bytes("probe_mixed_levels_total2.btx", mixed_bytes)
report["python_uniform_level1_total2"] = {
    "file": os.path.basename(level1_path),
    "file_bytes": len(level1_bytes),
    "expected_bytes_for_uniform_top_level": HEADER_BYTES + 12 * sum(blender_lengths),
    "header": parse_header(level1_bytes),
    "per_corner_offsets_first_4": corner_offsets(level1_lengths)[:4],
}
report["python_mixed_levels_total2"] = {
    "file": os.path.basename(mixed_path),
    "levels": mixed_levels,
    "file_bytes": len(mixed_bytes),
    "expected_bytes_for_uniform_top_level": HEADER_BYTES + 12 * sum(blender_lengths),
    "header": parse_header(mixed_bytes),
    "per_corner_offsets_first_4": corner_offsets(mixed_lengths)[:4],
    "per_corner_offsets": corner_offsets(mixed_lengths),
}


# 3. Reload each file in a fresh .blend session with Blender's own external reader.
def reload_with(name, path):
    bpy.ops.wm.open_mainfile(filepath=blend_path, load_ui=False)
    obj = bpy.data.objects["Cube"]
    mod = obj.modifiers["Multires"]
    mod.levels = max(int(mod.levels), 1)
    mod.show_viewport = True
    mod.filepath = path
    print("PROBE reload start", name, "filepath", mod.filepath, flush=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph).data
    print("PROBE reload done", name, "evaluated_vertices", len(evaluated.vertices), flush=True)
    return {"evaluated_vertices": len(evaluated.vertices), "is_external": bool(mod.is_external)}


report["reload"] = {
    "blender_uniform_level2": reload_with("blender_uniform_level2", uniform_btx),
    "python_uniform_level1_total2": reload_with("python_uniform_level1_total2", level1_path),
    "python_mixed_levels_total2": reload_with("python_mixed_levels_total2", mixed_path),
}

with open(os.path.join(OUT, "probe_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("PROBE report written", os.path.join(OUT, "probe_report.json"), flush=True)
