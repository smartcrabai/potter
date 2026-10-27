"""Blender 5.2.2 VSE text Unicode pixel oracle.

Regenerate from any directory with:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --threads 1 \
        -P /Users/admin/apps/potter-text-strip/tests/fixtures/native_oracle/vse_text_unicode/generate.py

The portable blend keeps its existing two captures and adds five Unicode/layout cases. A separate
blend captures a 50% resolution render; default-font captures remain Blender-only probes.
"""

import array
import hashlib
import json
import os
import struct
import sys

import bpy


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.environ.get("VSE_TEXT_UNICODE_OUTPUT_DIR", OUT_DIR))
os.makedirs(OUTPUT_DIR, exist_ok=True)
FONT_ASSETS = {
    "lato_regular": {
        "path": os.path.abspath(
            os.path.join(OUT_DIR, "..", "..", "text_external_font", "Lato-Regular.ttf")
        ),
        "repository_path": "tests/fixtures/text_external_font/Lato-Regular.ttf",
        "license": "tests/fixtures/text_external_font/OFL.txt",
    },
    "lato_italic": {
        "path": os.path.abspath(
            os.path.join(OUT_DIR, "..", "..", "text_external_font", "Lato-Italic.ttf")
        ),
        "repository_path": "tests/fixtures/text_external_font/Lato-Italic.ttf",
        "license": "tests/fixtures/text_external_font/OFL.txt",
    },
    "noto_cjk_unicode_subset": {
        "path": os.path.join(OUT_DIR, "fonts", "NotoSansCJKUnicodeSubset.otf"),
        "repository_path": (
            "tests/fixtures/native_oracle/vse_text_unicode/fonts/NotoSansCJKUnicodeSubset.otf"
        ),
        "license": "tests/fixtures/native_oracle/vse_text_unicode/fonts/OFL.txt",
        "format": (
            "Static OpenType CFF (OTF), decompressed from Blender's bundled WOFF2 with FontTools "
            "4.68.0."
        ),
    },
    "liberation_sans": {
        "path": os.path.join(OUT_DIR, "fonts", "LiberationSans-Regular.ttf"),
        "repository_path": (
            "tests/fixtures/native_oracle/vse_text_unicode/fonts/LiberationSans-Regular.ttf"
        ),
        "license": "tests/fixtures/native_oracle/vse_text_unicode/fonts/LiberationSans-OFL.txt",
        "license_source": (
            "https://github.com/liberationfonts/liberation-fonts/blob/2.1.5/LICENSE"
        ),
        "source": (
            "https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5"
        ),
    },
}
WIDTH = 1024
HEIGHT = 256
DEFAULT_CASES = (
    {"name": "default_cjk", "text": "日本語の文字", "probe_only": False},
    {
        "name": "default_accented_latin",
        "text": "Café déjà vu — Ångström naïve",
        "probe_only": False,
    },
    {"name": "default_emoji_probe", "text": "Emoji probe: 😀", "probe_only": True},
    {"name": "default_rtl_probe", "text": "مرحبا بالعالم", "probe_only": True},
    {"name": "default_multiline", "text": "first line\nsecond line", "probe_only": False},
)
PORTABLE_CASES = (
    {
        "name": "noto_cjk_multiline",
        "text": "日本語\n漢字中文",
        "probe_only": False,
        "font_asset": "noto_cjk_unicode_subset",
        "capture": "supported_frame_0001.png",
    },
    {
        "name": "lato_accented_multiline",
        "text": "AV Café\nÀ bientôt",
        "probe_only": False,
        "font_asset": "lato_regular",
        "capture": "supported_frame_0002.png",
    },
    {
        "name": "liberation_av_kern",
        "text": "AV",
        "probe_only": False,
        "font_asset": "liberation_sans",
        "capture": "supported_frame_0003.png",
        "oracle_expectation": "Static legacy kern-table pair AV is applied.",
    },
    {
        "name": "cjk_no_space_overflow",
        "text": "日本語漢字",
        "probe_only": False,
        "font_asset": "noto_cjk_unicode_subset",
        "capture": "supported_frame_0004.png",
        "wrap_width": 0.24,
        "oracle_expectation": "No-space CJK text remains one line beyond wrap_width.",
    },
    {
        "name": "cjk_ideographic_space_wrap",
        "text": "日本語　漢",
        "probe_only": False,
        "font_asset": "noto_cjk_unicode_subset",
        "capture": "supported_frame_0005.png",
        "wrap_width": 0.24,
        "oracle_expectation": (
            "Blender 5.2.2 wraps at U+0020 only; U+3000 remains on the overflowing line."
        ),
    },
    {
        "name": "cjk_ascii_space_wrap",
        "text": "日本語 漢",
        "probe_only": False,
        "font_asset": "noto_cjk_unicode_subset",
        "capture": "supported_frame_0006.png",
        "wrap_width": 0.24,
        "oracle_expectation": "U+0020 spaces wrap CJK words at the configured width.",
    },
    {
        "name": "lato_italic_final_negative_lsb",
        "text": "AF",
        "probe_only": False,
        "font_asset": "lato_italic",
        "capture": "supported_frame_0007.png",
        "oracle_expectation": (
            "The final italic F has a negative left side bearing; center the advance width."
        ),
    },
)
HALF_RESOLUTION_CASE = {
    "name": "scaled_size_50_percent",
    "text": "M",
    "probe_only": False,
    "font_asset": "lato_regular",
    "font_size": 60,
    "capture": "half_resolution_frame_0001.png",
    "blend_file": "vse_text_unicode_half_resolution.blend",
    "expected_font_size_pixels": 30,
}


def configure_scene(scene, frame_end, resolution_percentage=100):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = resolution_percentage
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.use_sequencer = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.frame_start = 1
    scene.frame_end = frame_end


def add_text_strips(scene, cases, fonts):
    editor = scene.sequence_editor_create()
    strips = editor.strips if hasattr(editor, "strips") else editor.sequences
    for frame, case in enumerate(cases, start=1):
        strip = strips.new_effect(
            name=f"{case['name']}_text",
            type="TEXT",
            channel=1,
            frame_start=frame,
            length=1,
        )
        strip.text = case["text"]
        if case.get("font_asset"):
            strip.font = fonts[case["font_asset"]]
        if "font_size" in case:
            strip.font_size = case["font_size"]
        if "wrap_width" in case:
            strip.wrap_width = case["wrap_width"]
        case["frame"] = frame
        case["strip"] = strip


def unicode_glyph_id(data, codepoint):
    table_count = struct.unpack_from(">H", data, 4)[0]
    cmap_offset = None
    for index in range(table_count):
        record = 12 + 16 * index
        if data[record : record + 4] == b"cmap":
            cmap_offset = struct.unpack_from(">I", data, record + 8)[0]
            break
    if cmap_offset is None:
        raise RuntimeError("static kern font has no cmap table")

    encoding_count = struct.unpack_from(">H", data, cmap_offset + 2)[0]
    for index in range(encoding_count):
        platform, encoding, relative_offset = struct.unpack_from(
            ">HHI", data, cmap_offset + 4 + 8 * index
        )
        offset = cmap_offset + relative_offset
        if struct.unpack_from(">H", data, offset)[0] != 4 or not (
            platform == 0 or (platform == 3 and encoding == 1)
        ):
            continue
        segment_count = struct.unpack_from(">H", data, offset + 6)[0] // 2
        end_codes_offset = offset + 14
        end_codes = struct.unpack_from(
            f">{segment_count}H", data, end_codes_offset
        )
        start_codes_offset = end_codes_offset + segment_count * 2 + 2
        start_codes = struct.unpack_from(
            f">{segment_count}H", data, start_codes_offset
        )
        deltas_offset = start_codes_offset + segment_count * 2
        deltas = struct.unpack_from(f">{segment_count}h", data, deltas_offset)
        ranges_offset = deltas_offset + segment_count * 2
        for index, (start, end) in enumerate(zip(start_codes, end_codes)):
            if start <= codepoint <= end:
                range_offset = struct.unpack_from(">H", data, ranges_offset + 2 * index)[0]
                if range_offset == 0:
                    return (codepoint + deltas[index]) & 0xFFFF
                glyph_offset = (
                    ranges_offset
                    + 2 * index
                    + range_offset
                    + 2 * (codepoint - start)
                )
                glyph = struct.unpack_from(">H", data, glyph_offset)[0]
                return (glyph + deltas[index]) & 0xFFFF if glyph else 0
    raise RuntimeError(f"static kern font has no Unicode glyph for U+{codepoint:04X}")


def static_kern_table_evidence(path):
    with open(path, "rb") as font_file:
        data = font_file.read()
    table_count = struct.unpack_from(">H", data, 4)[0]
    kern_offset = kern_length = None
    for index in range(table_count):
        record = 12 + 16 * index
        if data[record : record + 4] == b"kern":
            kern_offset, kern_length = struct.unpack_from(">II", data, record + 8)
            break
    if kern_offset is None:
        raise RuntimeError(f"expected a static legacy kern table in {path}")

    version, subtable_count = struct.unpack_from(">HH", data, kern_offset)
    if version != 0:
        raise RuntimeError(f"unsupported legacy kern version {version} in {path}")
    av_glyphs = (unicode_glyph_id(data, ord("A")), unicode_glyph_id(data, ord("V")))
    cursor = kern_offset + 4
    format_zero_pair_count = 0
    av_pair_adjustment = None
    subtables = []
    for _ in range(subtable_count):
        subtable_version, length, coverage = struct.unpack_from(">HHH", data, cursor)
        table_format = coverage >> 8
        if subtable_version != 0 or length < 6 or cursor + length > kern_offset + kern_length:
            raise RuntimeError(f"invalid legacy kern subtable in {path}")
        if table_format == 0:
            if length < 14:
                raise RuntimeError(f"truncated format 0 kern subtable in {path}")
            pair_count = struct.unpack_from(">H", data, cursor + 6)[0]
            if length < 14 + pair_count * 6:
                raise RuntimeError(f"truncated format 0 kern subtable in {path}")
            format_zero_pair_count += pair_count
            for pair_index in range(pair_count):
                left, right, adjustment = struct.unpack_from(
                    ">HHh", data, cursor + 14 + 6 * pair_index
                )
                if (left, right) == av_glyphs:
                    av_pair_adjustment = adjustment
        else:
            pair_count = None
        subtables.append({"format": table_format, "pair_count": pair_count})
        cursor += length
    if format_zero_pair_count == 0 or av_pair_adjustment is None:
        raise RuntimeError(f"legacy kern table has no AV format 0 pair in {path}")
    return {
        "table": "kern",
        "version": version,
        "subtables": subtables,
        "format_zero_pair_count": format_zero_pair_count,
        "AV_pair_adjustment_units": av_pair_adjustment,
    }


def capture_case(scene, case, comparison_scope):
    if bpy.context.window is not None:
        bpy.context.window.scene = scene
    scene.frame_set(case["frame"])
    capture_name = case.get("capture", f"{case['name']}.png")
    png_path = os.path.join(OUTPUT_DIR, capture_name)
    scene.render.filepath = os.path.splitext(png_path)[0]
    bpy.ops.render.render(write_still=True, scene=scene.name)
    if not os.path.isfile(png_path):
        raise RuntimeError(f"Blender did not write expected capture {png_path}")

    width = scene.render.resolution_x * scene.render.resolution_percentage // 100
    height = scene.render.resolution_y * scene.render.resolution_percentage // 100
    image = bpy.data.images.load(png_path, check_existing=False)
    if tuple(image.size) != (width, height) or image.channels != 4:
        raise RuntimeError(
            f"unexpected {case['name']} capture: size={image.size[:]}, channels={image.channels}"
        )
    pixels = image.pixels[:]
    float_pixels = array.array("f", pixels)
    if float_pixels.itemsize != 4:
        raise RuntimeError(f"unexpected Blender float pixel width {float_pixels.itemsize}")
    if sys.byteorder != "little":
        float_pixels.byteswap()
    rgba_f32le_sha256 = hashlib.sha256(float_pixels.tobytes()).hexdigest()
    alpha = pixels[3::4]
    visible = [(index % width, index // width) for index, value in enumerate(alpha) if value > 0.0]
    if not visible:
        raise RuntimeError(f"{case['name']} rendered no nontransparent pixels")
    bounds = None
    if visible:
        xs, ys = zip(*visible)
        bounds = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    with open(png_path, "rb") as capture:
        png_sha256 = hashlib.sha256(capture.read()).hexdigest()

    strip = case["strip"]
    font = getattr(strip, "font", None)
    asset = FONT_ASSETS.get(case.get("font_asset"))
    metadata = {
        "name": case["name"],
        "text": case["text"],
        **(
            {"oracle_expectation": case["oracle_expectation"]}
            if "oracle_expectation" in case
            else {}
        ),
        "probe_only": case["probe_only"],
        "comparison_scope": comparison_scope,
        "scene": scene.name,
        "frame": case["frame"],
        "blend_file": case.get("blend_file", "vse_text_unicode_supported.blend"),
        "capture": capture_name,
        "png_sha256": png_sha256,
        "render_settings": {
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "resolution_percentage": scene.render.resolution_percentage,
            "output_size": [width, height],
        },
        **(
            {"expected_font_size_pixels": case["expected_font_size_pixels"]}
            if "expected_font_size_pixels" in case
            else {}
        ),
        "pixels": {
            "size": list(image.size),
            "channels": image.channels,
            "alpha_mode": image.alpha_mode,
            "rgba_float32_le_sha256": rgba_f32le_sha256,
            "nontransparent_pixel_count": len(visible),
            "nontransparent_bounds_blender_xyxy": bounds,
        },
        "strip_defaults": {
            "font": font.name if font else None,
            "font_asset": asset["repository_path"] if asset else None,
            "font_license": asset["license"] if asset else None,
            "font_license_source": asset.get("license_source") if asset else None,
            "font_size": strip.font_size,
            "wrap_width": strip.wrap_width,
            "location": list(strip.location),
            "alignment_x": strip.alignment_x,
            "anchor_x": strip.anchor_x,
            "anchor_y": strip.anchor_y,
            "color": list(strip.color),
            "use_shadow": strip.use_shadow,
        },
    }
    print(
        "CAPTURE",
        case["name"],
        "scope=" + comparison_scope,
        "probe_only=" + str(case["probe_only"]),
        "pixels=" + str(len(visible)),
        "bounds=" + str(bounds),
        "rgba_f32le_sha256=" + rgba_f32le_sha256,
    )
    bpy.data.images.remove(image)
    return metadata


bpy.ops.wm.read_factory_settings(use_empty=True)
fonts = {}
font_asset_metadata = {}
for name, asset in FONT_ASSETS.items():
    if not os.path.isfile(asset["path"]):
        raise RuntimeError(f"missing licensed oracle font at {asset['path']}")
    fonts[name] = bpy.data.fonts.load(asset["path"], check_existing=False)
    font_asset_metadata[name] = {
        "repository_path": asset["repository_path"],
        "license": asset["license"],
    }
    if "format" in asset:
        font_asset_metadata[name]["format"] = asset["format"]
    if "license_source" in asset:
        font_asset_metadata[name]["license_source"] = asset["license_source"]
        font_asset_metadata[name]["source"] = asset["source"]
    if name == "liberation_sans":
        font_asset_metadata[name]["static_kern_table"] = static_kern_table_evidence(
            asset["path"]
        )

probe_scene = bpy.context.scene
probe_scene.name = "VSETextUnicodeDefaultProbes"
configure_scene(probe_scene, len(DEFAULT_CASES))
add_text_strips(probe_scene, DEFAULT_CASES, fonts)

portable_scene = bpy.data.scenes.new("VSETextUnicodePortable")
configure_scene(portable_scene, len(PORTABLE_CASES))
add_text_strips(portable_scene, PORTABLE_CASES, fonts)

case_metadata = []
for case in DEFAULT_CASES:
    case_metadata.append(capture_case(probe_scene, case, "blender_only_default_font"))
for case in PORTABLE_CASES:
    case_metadata.append(capture_case(portable_scene, case, "portable_cli_parity"))

if bpy.context.window is not None:
    bpy.context.window.scene = portable_scene
portable_scene.frame_set(1)
bpy.data.scenes.remove(probe_scene, do_unlink=True)

blend_path = os.path.join(OUTPUT_DIR, "vse_text_unicode_supported.blend")
for name, font in fonts.items():
    font.pack()
    font.filepath = bpy.path.relpath(FONT_ASSETS[name]["path"], start=blend_path)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("SAVED", blend_path)
portable_render_settings = {
    "engine": portable_scene.render.engine,
    "resolution": [WIDTH, HEIGHT],
    "resolution_percentage": portable_scene.render.resolution_percentage,
    "format": "PNG",
    "color_mode": "RGBA",
    "color_depth": 8,
    "film_transparent": portable_scene.render.film_transparent,
    "view_transform": portable_scene.view_settings.view_transform,
    "look": portable_scene.view_settings.look,
}

bpy.ops.wm.read_factory_settings(use_empty=True)
half_font = bpy.data.fonts.load(FONT_ASSETS["lato_regular"]["path"], check_existing=False)
half_resolution_scene = bpy.context.scene
half_resolution_scene.name = "VSETextUnicodeHalfResolution"
configure_scene(half_resolution_scene, 1, resolution_percentage=50)
add_text_strips(half_resolution_scene, (HALF_RESOLUTION_CASE,), {"lato_regular": half_font})
case_metadata.append(
    capture_case(half_resolution_scene, HALF_RESOLUTION_CASE, "portable_cli_parity")
)

half_blend_path = os.path.join(OUTPUT_DIR, HALF_RESOLUTION_CASE["blend_file"])
half_font.pack()
half_font.filepath = bpy.path.relpath(
    FONT_ASSETS["lato_regular"]["path"], start=half_blend_path
)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=half_blend_path)
print("SAVED", half_blend_path)

metadata = {
    "blender_version": bpy.app.version_string,
    "blend_file": "vse_text_unicode_supported.blend",
    "additional_blend_files": ["vse_text_unicode_half_resolution.blend"],
    "render": portable_render_settings,
    "font_policy": (
        "Default captures use Blender's installed fallback font stack and remain Blender-only. "
        "Portable cases use static Lato-Regular, Lato-Italic, Noto Sans CJK subset, and Liberation "
        "Sans fonts. Their SIL OFL 1.1 license files are retained beside the fixtures."
    ),
    "font_assets": font_asset_metadata,
    "pixel_hash_representation": (
        "Blender Image.pixels RGBA float32 values, serialized little-endian after reading each PNG."
    ),
    "cases": case_metadata,
}
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as output:
    json.dump(metadata, output, ensure_ascii=False, indent=2)
    output.write("\n")
print("WROTE", metadata_path)
