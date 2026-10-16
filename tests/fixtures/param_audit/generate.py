"""Audit param_table.json defaults against Blender's runtime modifier defaults.

Usage (from the repo root):

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tests/fixtures/param_audit/generate.py -- [--check]

Purpose:
    For every modifier type in crates/potter-core/src/param_table.json, create the
    modifier in a factory-fresh Blender session and record the runtime value of each
    property listed in the table. Unless --check is given, the recorded values and
    the EXCEPTIONS map are written to blender_defaults.json next to this script.
    Every table property that has a "default" is then compared with the runtime
    value: bool/int/str/enum/None must be equal; floats (scalar and array) are
    compared as float32. Mismatches and modifier types Blender cannot create make
    the script exit 1. Listed EXCEPTIONS are printed as EXCEPTION lines with their
    reason and are not counted as mismatches. Table properties that Blender does not
    expose on the modifier are reported as SKIP and not compared.
"""

import json
import os
import struct
import sys

import bpy
import mathutils

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TABLE_PATH = os.path.join(REPO_ROOT, "crates", "potter-core", "src", "param_table.json")
FIXTURE_PATH = os.path.join(HERE, "blender_defaults.json")

VOLUME_TYPES = ("mesh_to_volume", "volume_displace")

# Table keys whose Blender modifier identifier is not simply the upper-cased key.
# Mirrors the mapping in crates/potter-core/src/exchange/blend.rs.
TYPE_IDENTIFIERS = {
    "subdivision": "SUBSURF",
    "laplacian_deform": "LAPLACIANDEFORM",
    "laplacian_smooth": "LAPLACIANSMOOTH",
}

# "type.prop" -> reason. Listed properties are explicitly reported but never fail the check.
EXCEPTIONS = {
    "mesh_sequence_cache.frame_offset": (
        "Blender's MeshSequenceCacheModifier does not expose frame_offset as an RNA "
        "property, so the table default cannot be sampled at runtime."
    ),
    "mesh_sequence_cache.scale": (
        "Blender's MeshSequenceCacheModifier does not expose scale as an RNA "
        "property, so the table default cannot be sampled at runtime."
    ),
    "mesh_sequence_cache.override_frame": (
        "Blender's MeshSequenceCacheModifier does not expose override_frame as an RNA "
        "property, so the table default cannot be sampled at runtime."
    ),
    "nodes.inputs": (
        "Blender's NodesModifier does not expose an inputs RNA attribute on the "
        "modifier instance, so the table default cannot be sampled there."
    ),
    "ocean.filepath": (
        "Blender assigns a per-session temp directory to the Ocean cache; "
        "the value is not reproducible across runs."
    ),
    "uv_project.projectors": (
        "Blender creates one null projector; Potter stores projector refs only, "
        "[] is the unassigned state."
    ),
}


def blender_identifier(key):
    return TYPE_IDENTIFIERS.get(key, key.upper())


def to_json_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bpy.types.ID):
        return value.name
    if isinstance(value, mathutils.Matrix):
        return [float(x) for row in value for x in row]
    if isinstance(value, (mathutils.Vector, mathutils.Euler, mathutils.Quaternion, mathutils.Color)):
        return [float(x) for x in value]
    if isinstance(value, (set, frozenset)):
        # Enum-flag properties: Blender's set order is arbitrary, so sort for a stable fixture.
        return sorted(str(item) for item in value)
    if isinstance(value, bpy.types.CurveMapping):
        return [[float(p.location[0]), float(p.location[1])] for p in value.curves[0].points]
    if isinstance(value, bpy.types.bpy_struct):
        # Non-ID structs (e.g. UVProjector) have addresses in str(); use the RNA type name.
        return value.bl_rna.identifier
    if hasattr(value, "__len__") and hasattr(value, "__iter__"):
        return [to_json_value(item) for item in value]
    return str(value)


def float32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def same(expected, actual):
    if isinstance(expected, list) or isinstance(actual, list):
        return (
            isinstance(expected, list)
            and isinstance(actual, list)
            and len(expected) == len(actual)
            and all(same(a, b) for a, b in zip(expected, actual))
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        return float32(expected) == float32(actual)
    return expected == actual


def new_modifier(owner, key, identifier):
    try:
        mod = owner.modifiers.new(name="probe", type=identifier)
    except Exception as exc:  # Blender raises TypeError/RuntimeError for unknown types
        print(f"CREATE FAIL {key} ({identifier}): {exc}")
        return None
    if mod is None:
        print(f"CREATE FAIL {key} ({identifier}): modifiers.new returned None")
        return None
    return mod


def record_properties(key, mod, properties):
    values = {}
    for prop in properties:
        try:
            value = getattr(mod, prop)
        except AttributeError:
            print(f"SKIP {key}.{prop}: not exposed on Blender modifier, not compared")
            continue
        values[prop] = to_json_value(value)
    return values


def probe_mesh_type(key, identifier, properties):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    mod = new_modifier(bpy.context.active_object, key, identifier)
    if mod is None:
        return None
    return record_properties(key, mod, properties)


def probe_volume_type(key, identifier, properties):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vol = bpy.data.volumes.new("v")
    vo = bpy.data.objects.new("vo", vol)
    bpy.context.collection.objects.link(vo)
    mod = new_modifier(vo, key, identifier)
    if mod is None:
        return None
    return record_properties(key, mod, properties)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    unknown = [arg for arg in argv if arg != "--check"]
    if unknown:
        print(f"unknown arguments: {unknown}; usage: generate.py -- [--check]")
        sys.exit(2)
    check_only = "--check" in argv

    with open(TABLE_PATH, encoding="utf-8") as f:
        table = json.load(f)

    creation_failures = 0
    recorded = {}
    for key, spec in table["modifiers"].items():
        properties = list(spec["properties"].keys())
        identifier = blender_identifier(key)
        probe = probe_volume_type if key in VOLUME_TYPES else probe_mesh_type
        values = probe(key, identifier, properties)
        if values is None:
            creation_failures += 1
        else:
            recorded[key] = values

    if not check_only:
        doc = {
            "blender_version": bpy.app.version_string,
            "exceptions": {name: {"reason": reason} for name, reason in EXCEPTIONS.items()},
            "modifiers": recorded,
        }
        with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {FIXTURE_PATH}")

    checked = 0
    mismatches = 0
    excepted = 0
    for key, spec in table["modifiers"].items():
        values = recorded.get(key, {})
        for prop, prop_spec in spec["properties"].items():
            if "default" not in prop_spec:
                continue
            name = f"{key}.{prop}"
            expected = prop_spec["default"]
            present = prop in values
            actual = values.get(prop)
            if name in EXCEPTIONS:
                excepted += 1
                if present:
                    status = "matches" if same(expected, actual) else "differs"
                    blender_value = actual
                else:
                    status = "unavailable"
                    blender_value = "unavailable"
                print(
                    f"EXCEPTION {name} table={expected} blender={blender_value} "
                    f"({status}) reason={EXCEPTIONS[name]}"
                )
                continue
            checked += 1
            if not present:
                mismatches += 1
                print(f"MISSING {name} table={expected} blender=unavailable")
            elif not same(expected, actual):
                mismatches += 1
                print(f"{name} table={expected} blender={actual}")

    print(
        f"blender {bpy.app.version_string}: checked {checked} defaults, "
        f"{mismatches} mismatches, {excepted} exceptions, "
        f"{creation_failures} modifier types not created"
    )
    sys.exit(1 if mismatches or creation_failures else 0)


if __name__ == "__main__":
    main()
