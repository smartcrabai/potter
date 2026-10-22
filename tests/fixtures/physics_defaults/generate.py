"""Generate crates/potter-core/src/physics_defaults.json from a headless Blender 5.2.2.

Run from the repository root:

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
        -P tests/fixtures/physics_defaults/generate.py

An optional output path may follow `--`. Every value is read from a freshly added
modifier or object instance, not from the RNA `default` field, because the DNA
defaults that Blender actually applies differ from the RNA declarations for some
properties (for example cloth `damping`).

Only writable, non-runtime RNA properties are emitted. Collections serialise as an
empty array, pointers as null, and float32 values are rounded to 7 significant
digits so that the JSON reads like the values Blender shows in its UI.
"""

import json
import math
import pathlib
import sys

import bpy

EXPECTED_VERSION = (5, 2, 2)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "crates" / "potter-core" / "src" / "physics_defaults.json"
SKIPPED_IDENTIFIERS = {"rna_type"}
MAX_STRUCT_DEPTH = 2


def output_path():
    argv = sys.argv
    if "--" in argv:
        extra = argv[argv.index("--") + 1 :]
        if extra:
            return pathlib.Path(extra[0])
    return DEFAULT_OUTPUT


def fresh_object(name):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.name = name
    bpy.context.view_layer.objects.active = obj
    return obj


def add_modifier(obj, modifier_type):
    with bpy.context.temp_override(object=obj, active_object=obj):
        result = bpy.ops.object.modifier_add(type=modifier_type)
    if "FINISHED" not in result:
        raise RuntimeError("Blender did not add %s to %s" % (modifier_type, obj.name))
    return obj.modifiers[-1]


def toggle_dynamic_paint(obj, modifier, role):
    modifier.ui_type = role
    with bpy.context.temp_override(object=obj, active_object=obj):
        result = bpy.ops.dpaint.type_toggle(type=role)
    if "FINISHED" not in result:
        raise RuntimeError("Blender did not create the Dynamic Paint %s" % role)


def serialise_value(prop, value, depth):
    if prop.type == "POINTER":
        # ID references and unset pointers have no comparable default value.
        if value is None or isinstance(value, bpy.types.ID) or depth >= MAX_STRUCT_DEPTH:
            return None
        return struct_defaults(value, depth + 1)
    if prop.type == "COLLECTION":
        return []
    if prop.type == "ENUM":
        if prop.is_enum_flag:
            return sorted(value)
        return str(value)
    if prop.type == "STRING":
        return str(value)
    if prop.type == "BOOLEAN":
        return sequence_value(value, bool)
    if prop.type == "INT":
        return sequence_value(value, int)
    if prop.type == "FLOAT":
        return sequence_value(value, round_float32)
    raise RuntimeError("unsupported RNA property type %s" % prop.type)


def sequence_value(value, convert):
    if isinstance(value, (bool, int, float)):
        return convert(value)
    return [sequence_value(item, convert) for item in value]


def round_float32(value):
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("non-finite default value %r" % number)
    return float("%.7g" % number)


# Read-only properties that the .blend bridge dumps and that Potter compares against defaults.
EXTRA_READONLY = {
    "PointCache": ["is_baked"],
    "DynamicPaintCanvasSettings": ["canvas_surfaces"],
}
# Path properties whose default is a temporary or host-specific location. They are written as an
# empty string so the committed table does not depend on the generating machine. The runtime treats
# them as non-semantic (physics_support::is_runtime_state).
HOST_PATH_PROPERTIES = {"cache_directory", "image_output_path", "filepath"}


def struct_defaults(struct, depth=0):
    values = {}
    for prop in struct.bl_rna.properties:
        identifier = prop.identifier
        if identifier in SKIPPED_IDENTIFIERS:
            continue
        if prop.is_readonly and identifier not in EXTRA_READONLY.get(struct.bl_rna.identifier, []):
            continue
        if getattr(prop, "is_runtime", False):
            continue
        if identifier in HOST_PATH_PROPERTIES:
            values[identifier] = ""
            continue
        values[identifier] = serialise_value(prop, getattr(struct, identifier), depth)
    return dict(sorted(values.items()))


def collect_systems():
    systems = {}

    collision_object = fresh_object("collision")
    add_modifier(collision_object, "COLLISION")
    systems["collision"] = struct_defaults(collision_object.collision)

    cloth_object = fresh_object("cloth")
    cloth = add_modifier(cloth_object, "CLOTH")
    systems["cloth"] = struct_defaults(cloth.settings)
    systems["cloth_collision"] = struct_defaults(cloth.collision_settings)
    systems["point_cache"] = struct_defaults(cloth.point_cache)

    soft_object = fresh_object("soft_body")
    soft = add_modifier(soft_object, "SOFT_BODY")
    systems["soft_body"] = struct_defaults(soft.settings)

    fluid_object = fresh_object("fluid_domain")
    fluid_domain = add_modifier(fluid_object, "FLUID")
    fluid_domain.fluid_type = "DOMAIN"
    # Blender's domain defaults depend on domain_type (liquid domains close their borders and add
    # FLIP particles), so each domain type gets its own table.
    fluid_domain.domain_settings.domain_type = "GAS"
    systems["fluid_domain"] = struct_defaults(fluid_domain.domain_settings)
    fluid_domain.domain_settings.domain_type = "LIQUID"
    systems["fluid_domain_liquid"] = struct_defaults(fluid_domain.domain_settings)

    fluid_flow_object = fresh_object("fluid_flow")
    fluid_flow = add_modifier(fluid_flow_object, "FLUID")
    fluid_flow.fluid_type = "FLOW"
    systems["fluid_flow"] = struct_defaults(fluid_flow.flow_settings)

    fluid_effector_object = fresh_object("fluid_effector")
    fluid_effector = add_modifier(fluid_effector_object, "FLUID")
    fluid_effector.fluid_type = "EFFECTOR"
    systems["fluid_effector"] = struct_defaults(fluid_effector.effector_settings)

    canvas_object = fresh_object("dynamic_paint_canvas")
    canvas_modifier = add_modifier(canvas_object, "DYNAMIC_PAINT")
    toggle_dynamic_paint(canvas_object, canvas_modifier, "CANVAS")
    canvas = canvas_modifier.canvas_settings
    systems["dynamic_paint_canvas"] = struct_defaults(canvas)
    if len(canvas.canvas_surfaces) == 0:
        raise RuntimeError("Dynamic Paint canvas has no default surface")
    systems["dynamic_paint_surface"] = struct_defaults(canvas.canvas_surfaces[0])

    brush_object = fresh_object("dynamic_paint_brush")
    brush_modifier = add_modifier(brush_object, "DYNAMIC_PAINT")
    toggle_dynamic_paint(brush_object, brush_modifier, "BRUSH")
    systems["dynamic_paint_brush"] = struct_defaults(brush_modifier.brush_settings)

    return systems


def main():
    if bpy.app.version != EXPECTED_VERSION:
        raise RuntimeError(
            "physics defaults must be generated by Blender %s, found %s"
            % (".".join(map(str, EXPECTED_VERSION)), bpy.app.version_string)
        )
    systems = collect_systems()
    document = {
        "blender_version": ".".join(map(str, EXPECTED_VERSION)),
        "systems": dict(sorted(systems.items())),
    }
    output = output_path()
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print("wrote %s with %d systems" % (output, len(systems)))


main()
