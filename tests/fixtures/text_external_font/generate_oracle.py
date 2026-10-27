"""Generate the Blender 5.2.2 oracle for the external-font text fixture."""

import json
import struct
from pathlib import Path

import bpy


FIXTURE_DIR = Path(__file__).resolve().parent
FONT_PATHS = {
    "regular": FIXTURE_DIR / "Lato-Regular.ttf",
    "bold": FIXTURE_DIR / "Lato-Bold.ttf",
    "italic": FIXTURE_DIR / "Lato-Italic.ttf",
    "bold_italic": FIXTURE_DIR / "Lato-BoldItalic.ttf",
}
BLEND_PATH = FIXTURE_DIR / "text_external_font.blend"
ORACLE_PATH = FIXTURE_DIR / "text_external_font.oracle.json"
REVIEW_ORACLE_PATH = FIXTURE_DIR / "text_external_font_review.oracle.json"
IMPORT_LOSSES_BLEND_PATH = FIXTURE_DIR / "text_external_font_import_losses.blend"
VARIABLE_FONT_PATH = FIXTURE_DIR / "InterVariable.ttf"
VARIABLE_BLEND_PATH = FIXTURE_DIR / "text_external_font_variable.blend"
METRIC_FONT_PATH = FIXTURE_DIR / "Lato-MetricOracle.ttf"
METRIC_BLEND_PATH = FIXTURE_DIR / "text_external_font_metric_alignments.blend"
REVIEW_CASES = {
    "default_top_baseline": {"body": "AV", "align_y": None, "word_spacing": 1.0},
    "top": {"body": "AV", "align_y": "TOP", "word_spacing": 1.0},
    "bottom": {"body": "AV", "align_y": "BOTTOM", "word_spacing": 1.0},
    "combining_mark": {"body": "e\u0301", "align_y": None, "word_spacing": 1.0},
    "control_spacing": {
        "body": "A\u0001 V",
        "align_y": None,
        "word_spacing": 1.35,
    },
}
TEXT_BODY = "AV B8O Égj"


def vector3(vector):
    return [float(component) for component in vector]


def clear_scene_objects():
    for existing_object in list(bpy.context.scene.objects):
        bpy.data.objects.remove(existing_object, do_unlink=True)


def evaluated_mesh(text_object):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = text_object.evaluated_get(depsgraph)
    mesh = evaluated_object.to_mesh()
    try:
        vertices = [vector3(vertex.co) for vertex in mesh.vertices]
        if not vertices:
            raise RuntimeError(f"{text_object.name} produced no mesh vertices")
        return {
            "vertices": vertices,
            "edges": [list(edge.vertices) for edge in mesh.edges],
            "faces": [list(polygon.vertices) for polygon in mesh.polygons],
        }
    finally:
        evaluated_object.to_mesh_clear()


def make_text_object(name, body, font, align_y=None, word_spacing=1.0):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.font = font
    curve.align_x = "LEFT"
    if align_y is not None:
        curve.align_y = align_y
    curve.space_word = word_spacing
    curve.resolution_u = 8
    curve.fill_mode = "BOTH"
    curve.fill_solver = "CDT"
    curve.fill_rule = "EVEN_ODD"
    text_object = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(text_object)
    return text_object


def save_review_case(name, settings, regular_font):
    clear_scene_objects()
    text_object = make_text_object(
        f"ExternalFont{name.title()}Text",
        settings["body"],
        regular_font,
        settings["align_y"],
        settings["word_spacing"],
    )
    bpy.context.view_layer.objects.active = text_object
    text_object.select_set(True)
    mesh = evaluated_mesh(text_object)
    fixture_path = FIXTURE_DIR / f"text_external_font_{name}.blend"
    regular_font.filepath = bpy.path.relpath(
        str(FONT_PATHS["regular"]), start=str(fixture_path)
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(fixture_path))
    return {
        "blend_file": fixture_path.name,
        "object_name": text_object.name,
        "text": {
            "body": text_object.data.body,
            "align_x": text_object.data.align_x,
            "align_y": text_object.data.align_y,
            "word_spacing": float(text_object.data.space_word),
        },
        "mesh": mesh,
    }


def checksum(data):
    padded = data + bytes((-len(data)) % 4)
    return sum(struct.unpack(f">{len(padded) // 4}I", padded)) & 0xFFFFFFFF


def rename_metric_font_name_table(data):
    format_type, count, storage_offset = struct.unpack_from(">HHH", data)
    if format_type != 0:
        raise RuntimeError(f"Unsupported Lato name-table format {format_type}")
    records = []
    strings = bytearray()
    rename_ids = {1, 3, 4, 6, 16, 18, 21}
    for index in range(count):
        record_offset = 6 + index * 12
        platform, encoding, language, name_id, length, offset = struct.unpack_from(
            ">HHHHHH", data, record_offset
        )
        value = data[storage_offset + offset : storage_offset + offset + length]
        if name_id in rename_ids:
            codec = "utf-16-be" if platform in (0, 3) else "mac_roman"
            decoded = value.decode(codec)
            decoded = decoded.replace("Lato", "Potter Metric")
            if name_id == 6:
                decoded = decoded.replace(" ", "")
            value = decoded.encode(codec)
        records.append((platform, encoding, language, name_id, value))
        strings.extend(value)
    string_offset = 6 + count * 12
    table = bytearray(struct.pack(">HHH", format_type, count, string_offset))
    current_offset = 0
    for platform, encoding, language, name_id, value in records:
        table.extend(
            struct.pack(
                ">HHHHHH",
                platform,
                encoding,
                language,
                name_id,
                len(value),
                current_offset,
            )
        )
        current_offset += len(value)
    table.extend(strings)
    return table


def make_metric_oracle_font(source_path, output_path):
    original = source_path.read_bytes()
    table_count = struct.unpack_from(">H", original, 4)[0]
    tables = {}
    table_order = []
    for index in range(table_count):
        record_offset = 12 + index * 16
        tag, _, offset, length = struct.unpack_from(">4sIII", original, record_offset)
        table_order.append(tag)
        tables[tag] = bytearray(original[offset : offset + length])
    os2 = tables[b"OS/2"]
    hhea_ascender, hhea_descender, hhea_line_gap = struct.unpack_from(
        ">hhh", tables[b"hhea"], 4
    )
    typo_metrics = (hhea_ascender + 200, hhea_descender - 100, hhea_line_gap)
    struct.pack_into(">H", os2, 62, struct.unpack_from(">H", os2, 62)[0] | 0x0080)
    struct.pack_into(">hhh", os2, 68, *typo_metrics)
    tables[b"name"] = rename_metric_font_name_table(tables[b"name"])

    offsets = {}
    cursor = 12 + table_count * 16
    directory_records = []
    table_data = bytearray()
    for tag in table_order:
        content = tables[tag]
        original_length = len(content)
        if tag == b"head":
            struct.pack_into(">I", content, 8, 0)
        content.extend(bytes((-len(content)) % 4))
        offsets[tag] = cursor
        table_checksum = checksum(content)
        directory_records.append(
            struct.pack(">4sIII", tag, table_checksum, cursor, original_length)
        )
        table_data.extend(content)
        cursor += len(content)
    font = bytearray(original[:12] + b"".join(directory_records) + table_data)
    head_offset = offsets[b"head"]
    struct.pack_into(">I", font, head_offset + 8, 0)
    struct.pack_into(
        ">I", font, head_offset + 8, (0xB1B0AFBA - checksum(font)) & 0xFFFFFFFF
    )
    output_path.write_bytes(font)
    return {
        "hhea": [hhea_ascender, hhea_descender, hhea_line_gap],
        "os2_typo": list(typo_metrics),
        "use_typo_metrics": True,
    }


def save_import_losses_case(regular_font):
    clear_scene_objects()
    text_curve = bpy.data.curves.new("ExternalFontImportLossText", type="FONT")
    text_curve.body = "Loss settings"
    text_curve.font = regular_font
    text_curve.offset = 0.12
    if not text_curve.text_boxes:
        text_curve.text_boxes.new()
    text_box = text_curve.text_boxes[0]
    text_box.x = 0.25
    text_box.y = -0.5
    text_box.width = 3.0
    text_box.height = 1.5

    follow_curve = bpy.data.curves.new("ExternalFontFollowCurveData", type="CURVE")
    follow_curve.dimensions = "2D"
    follow_spline = follow_curve.splines.new("POLY")
    follow_spline.points.add(1)
    follow_spline.points[0].co = (0.0, 0.0, 0.0, 1.0)
    follow_spline.points[1].co = (3.0, 0.5, 0.0, 1.0)
    follow_object = bpy.data.objects.new("ExternalFontFollowCurve", follow_curve)
    bpy.context.scene.collection.objects.link(follow_object)
    text_curve.follow_curve = follow_object

    taper_curve = bpy.data.curves.new("ExternalFontTaperCurveData", type="CURVE")
    taper_curve.dimensions = "2D"
    taper_spline = taper_curve.splines.new("POLY")
    taper_spline.points.add(1)
    taper_spline.points[0].co = (0.0, 0.1, 0.0, 1.0)
    taper_spline.points[1].co = (1.0, 0.3, 0.0, 1.0)
    taper_object = bpy.data.objects.new("ExternalFontTaperCurve", taper_curve)
    bpy.context.scene.collection.objects.link(taper_object)
    text_curve.taper_object = taper_object

    text_object = bpy.data.objects.new("ExternalFontImportLossText", text_curve)
    bpy.context.scene.collection.objects.link(text_object)
    bpy.context.view_layer.objects.active = text_object
    text_object.select_set(True)
    regular_font.filepath = bpy.path.relpath(
        str(FONT_PATHS["regular"]), start=str(IMPORT_LOSSES_BLEND_PATH)
    )
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(IMPORT_LOSSES_BLEND_PATH))


def save_variable_font_case():
    clear_scene_objects()
    for font in list(bpy.data.fonts):
        bpy.data.fonts.remove(font)
    variable_font = bpy.data.fonts.load(str(VARIABLE_FONT_PATH))
    text_object = make_text_object(
        "ExternalFontVariableText", "Variable", variable_font
    )
    bpy.context.view_layer.objects.active = text_object
    text_object.select_set(True)
    variable_font.pack()
    variable_font.filepath = bpy.path.relpath(
        str(VARIABLE_FONT_PATH), start=str(VARIABLE_BLEND_PATH)
    )
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(VARIABLE_BLEND_PATH))
    return {
        "blend_file": VARIABLE_BLEND_PATH.name,
        "object_name": text_object.name,
        "font_file": VARIABLE_FONT_PATH.name,
        "text": {"body": text_object.data.body},
    }


def save_metric_alignment_case():
    metric_details = make_metric_oracle_font(
        FONT_PATHS["regular"], METRIC_FONT_PATH
    )
    clear_scene_objects()
    for font in list(bpy.data.fonts):
        bpy.data.fonts.remove(font)
    font = bpy.data.fonts.load(str(METRIC_FONT_PATH))
    cases = {}
    for name, align_y in (("top", "TOP"), ("center", "CENTER")):
        text_object = make_text_object(
            f"ExternalFontMetric{name.title()}Text", "AV", font, align_y
        )
        bpy.context.view_layer.objects.active = text_object
        text_object.select_set(True)
        cases[name] = {
            "object_name": text_object.name,
            "text": {
                "body": text_object.data.body,
                "align_y": text_object.data.align_y,
            },
            "mesh": evaluated_mesh(text_object),
        }
    font.pack()
    font.filepath = bpy.path.relpath(str(METRIC_FONT_PATH), start=str(METRIC_BLEND_PATH))
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(METRIC_BLEND_PATH))
    return {
        "blend_file": METRIC_BLEND_PATH.name,
        "font_file": METRIC_FONT_PATH.name,
        "metrics": metric_details,
        "cases": cases,
    }



def main():
    fonts = {
        style: bpy.data.fonts.load(str(path), check_existing=True)
        for style, path in FONT_PATHS.items()
    }
    clear_scene_objects()
    curve = bpy.data.curves.new("ExternalFontOracleText", type="FONT")
    curve.body = TEXT_BODY
    curve.font = fonts["regular"]
    curve.font_bold = fonts["bold"]
    curve.font_italic = fonts["italic"]
    curve.font_bold_italic = fonts["bold_italic"]
    curve.body_format[0].use_bold = True
    curve.body_format[1].use_italic = True
    curve.body_format[3].use_bold = True
    curve.body_format[3].use_italic = True
    curve.body_format[3].kerning = 6.0
    curve.size = 1.0
    curve.align_x = "LEFT"
    curve.align_y = "BOTTOM_BASELINE"
    curve.space_character = 1.15
    curve.space_word = 1.2
    curve.space_line = 1.0
    curve.resolution_u = 8
    curve.fill_mode = "BOTH"
    curve.fill_solver = "CDT"
    curve.fill_rule = "EVEN_ODD"
    curve.extrude = 0.08
    curve.bevel_depth = 0.0
    curve.bevel_resolution = 2

    text_object = bpy.data.objects.new("ExternalFontOracleText", curve)
    bpy.context.scene.collection.objects.link(text_object)
    bpy.context.view_layer.objects.active = text_object
    text_object.select_set(True)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_object = text_object.evaluated_get(depsgraph)
    mesh = evaluated_object.to_mesh()
    try:
        vertices = [vector3(vertex.co) for vertex in mesh.vertices]
        if not vertices:
            raise RuntimeError("The evaluated text produced no mesh vertices")

        mins = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        maxs = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
        document = {
            "blender_version": list(bpy.app.version),
            "font": {
                "file": FONT_PATHS["regular"].name,
                "filepath": FONT_PATHS["regular"].name,
                "name": fonts["regular"].name,
                "variants": {
                    style: {"file": path.name, "name": fonts[style].name}
                    for style, path in FONT_PATHS.items()
                    if style != "regular"
                },
            },
            "text": {
                "body": curve.body,
                "character_styles": {
                    "bold_only": {
                        "index": 0,
                        "character": curve.body[0],
                        "use_bold": bool(curve.body_format[0].use_bold),
                        "use_italic": bool(curve.body_format[0].use_italic),
                    },
                    "italic_only": {
                        "index": 1,
                        "character": curve.body[1],
                        "use_bold": bool(curve.body_format[1].use_bold),
                        "use_italic": bool(curve.body_format[1].use_italic),
                    },
                    "bold_italic": {
                        "index": 3,
                        "character": curve.body[3],
                        "use_bold": bool(curve.body_format[3].use_bold),
                        "use_italic": bool(curve.body_format[3].use_italic),
                    },
                },
                "align_x": curve.align_x,
                "align_y": curve.align_y,
                "size": float(curve.size),
                "space_character": float(curve.space_character),
                "space_word": float(curve.space_word),
                "space_line": float(curve.space_line),
                "resolution_u": int(curve.resolution_u),
                "fill_mode": curve.fill_mode,
                "fill_solver": curve.fill_solver,
                "fill_rule": curve.fill_rule,
                "manual_kerning": {
                    "index": 3,
                    "value": float(curve.body_format[3].kerning),
                },
                "extrude": float(curve.extrude),
                "bevel_depth": float(curve.bevel_depth),
                "bevel_resolution": int(curve.bevel_resolution),
                "object_location": vector3(text_object.location),
                "object_rotation_euler": vector3(text_object.rotation_euler),
                "object_scale": vector3(text_object.scale),
            },
            "coordinate_space": "evaluated_object_local",
            "bounds": {
                "min": mins,
                "max": maxs,
                "size": [maxs[axis] - mins[axis] for axis in range(3)],
            },
            "mesh": {
                "vertices": vertices,
                "edges": [list(edge.vertices) for edge in mesh.edges],
                "faces": [list(polygon.vertices) for polygon in mesh.polygons],
            },
        }
    finally:
        evaluated_object.to_mesh_clear()
    for style, font in fonts.items():
        font.pack()
        font.filepath = bpy.path.relpath(str(FONT_PATHS[style]), start=str(BLEND_PATH))
        if font.packed_file is None:
            raise RuntimeError(f"Failed to pack {style} font")

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))

    ORACLE_PATH.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    clear_scene_objects()
    for style in ("bold", "italic", "bold_italic"):
        bpy.data.fonts.remove(fonts[style])


    review = {
        "blender_version": list(bpy.app.version),
        "cases": {
            name: save_review_case(name, settings, fonts["regular"])
            for name, settings in REVIEW_CASES.items()
        },
    }
    save_import_losses_case(fonts["regular"])
    review["import_losses"] = {
        "blend_file": IMPORT_LOSSES_BLEND_PATH.name,
        "text": {
            "body": "Loss settings",
            "offset": 0.12,
            "text_box": {
                "x": 0.25,
                "y": -0.5,
                "width": 3.0,
                "height": 1.5,
            },
            "follow_curve": "ExternalFontFollowCurve",
            "taper_object": "ExternalFontTaperCurve",
        },
    }
    review["variable_font"] = save_variable_font_case()
    review["metric_alignments"] = save_metric_alignment_case()
    REVIEW_ORACLE_PATH.write_text(
        json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
