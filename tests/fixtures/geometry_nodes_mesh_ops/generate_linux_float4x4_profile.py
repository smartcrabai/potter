"""Regenerate the frozen Linux FLOAT4X4 profile with its exact Blender build.

Captured from Blender source commit d13f752e3b9c4f8c261cda552b1021f8bcc0382c
in linux/amd64 Ubuntu 24.04 on a local ARM64 Docker daemon; the emulation
backend was not independently identified, so this is not a native-Linux claim.
"""
import ast
import hashlib
import json
import platform
import shutil
import struct
import sys
from pathlib import Path

import bpy

PINNED_BUILD_HASH = "d13f752e3b9c"
PINNED_GENERATOR_SHA256 = "c5cbb71e0e482dbd6ba27e0fdeef19df7807dbe2a9e07594ebc05b767e921573"
PINNED_FIXTURE_SHA256 = "6a3c6c10ce6df79de55a63f21dd4d060dd4ad8f49e3e8c81713e66fae5fa9d39"
PINNED_INPUT_SHA256 = "95cd5d07e1c60709c537b8f76adb272c3ab9c4af18a175117bf7cc1b84513371"
PINNED_PROFILE_SHA256 = "667854af6ceded20bab9fc7b78cfc3c9aac9923884ddd219e8b82a96dd715ced"


def app_string(key):
    value = getattr(bpy.app, key, "")
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"requires Blender 5.2.2, found {bpy.app.version_string}")
if app_string("build_hash") != PINNED_BUILD_HASH:
    raise RuntimeError(f"requires Blender build {PINNED_BUILD_HASH}, found {app_string('build_hash')}")
if app_string("build_platform") != "Linux" or app_string("build_type") != "Release":
    raise RuntimeError("requires the pinned Linux Release Blender build")
if sys.platform != "linux" or platform.machine() != "x86_64" or sys.byteorder != "little":
    raise RuntimeError("requires little-endian Linux x86_64 for the FLOAT4X4 host profile")
if platform.libc_ver() != ("glibc", "2.39"):
    raise RuntimeError(f"requires glibc 2.39, found {platform.libc_ver()}")
if "--" not in sys.argv:
    raise RuntimeError("usage: blender -b --python capture.py -- REPO_ROOT PROFILE_PATH")
args = sys.argv[sys.argv.index("--") + 1:]
if len(args) != 2:
    raise RuntimeError("usage: blender -b --python capture.py -- REPO_ROOT PROFILE_PATH")
repo, profile_path = map(lambda value: Path(value).resolve(), args)
generator_path = repo / "tests/fixtures/geometry_nodes_mesh_ops/generate.py"
fixture_path = repo / "tests/fixtures/geometry_nodes_mesh_ops/expected.json"
if hashlib.sha256(generator_path.read_bytes()).hexdigest() != PINNED_GENERATOR_SHA256:
    raise RuntimeError("pinned Geometry Nodes generator changed")
if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != PINNED_FIXTURE_SHA256:
    raise RuntimeError("pinned canonical Geometry Nodes fixture changed")
generator_copy = Path("/tmp/gn_mesh_ops_generate_matrix_profile.py")
shutil.copy2(generator_path, generator_copy)

# Load the exact pinned generator definitions but suppress its unconditional main() call.
syntax = ast.parse(generator_copy.read_text(encoding="utf-8"), filename=str(generator_copy))
syntax.body = [
    node for node in syntax.body
    if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "main"
    )
]
generator = {"__name__": "gn_mesh_ops_matrix_profile", "__file__": str(generator_copy)}
exec(compile(ast.fix_missing_locations(syntax), str(generator_copy), "exec"), generator)
fixture = json.loads(fixture_path.read_text(encoding="utf-8"))


def matrix_records(attributes):
    return {
        (record["name"], record["domain"], record["data_type"]): record["values"]
        for record in attributes.get("blender_attributes", [])
        if record.get("data_type") == "FLOAT4X4"
    }


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


canonical_records = matrix_records(fixture["input"]["attributes"])
if not canonical_records:
    raise RuntimeError("fixture has no FLOAT4X4 inputs")
canonical_bytes = b"".join(
    struct.pack("<f", float(component))
    for key in sorted(canonical_records)
    for matrix in canonical_records[key]
    for row in matrix
    for component in row
)
canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()
if canonical_hash != PINNED_INPUT_SHA256:
    raise RuntimeError(f"canonical FLOAT4X4 inputs changed: {canonical_hash}")

original_make_source = generator["make_source"]


def make_source_with_fixture_matrix_bits():
    obj = original_make_source()
    attributes = {
        (attr.name, attr.domain, attr.data_type): attr
        for attr in obj.data.attributes
        if attr.data_type == "FLOAT4X4"
    }
    if set(attributes) != set(canonical_records):
        raise RuntimeError(f"FLOAT4X4 input keys differ: {set(attributes)} vs {set(canonical_records)}")
    for key, matrices in canonical_records.items():
        attr = attributes[key]
        if len(attr.data) != len(matrices):
            raise RuntimeError(f"FLOAT4X4 input count differs for {key}")
        for item, rows in zip(attr.data, matrices):
            item.value = generator["Matrix"](rows)
        for index, (item, expected) in enumerate(zip(attr.data, matrices)):
            actual = [list(map(float, row)) for row in item.value]
            expected_bits = [[f32bits(x) for x in row] for row in expected]
            actual_bits = [[f32bits(x) for x in row] for row in actual]
            if actual_bits != expected_bits:
                raise RuntimeError(f"fixture FLOAT4X4 input {key}[{index}] changed")
    return obj


generator["make_source"] = make_source_with_fixture_matrix_bits
full_path = Path("/tmp/gn_mesh_ops_full_linux_output.json")
generator["OUTPUT"] = full_path
generator["main"]()
full = json.loads(full_path.read_text(encoding="utf-8"))
if matrix_records(full["input"]["attributes"]) != canonical_records:
    raise RuntimeError("generator output input matrices differ from the fixture")

outputs = []


def collect(value, path):
    if isinstance(value, dict):
        attrs = value.get("attributes")
        if isinstance(attrs, dict):
            for record in attrs.get("blender_attributes", []) or []:
                if record.get("data_type") == "FLOAT4X4":
                    outputs.append({
                        "path": path + ".attributes.blender_attributes[name=" + record["name"] + "]",
                        "name": record["name"],
                        "domain": record["domain"],
                        "data_type": record["data_type"],
                        "values": record["values"],
                    })
        for key, child in value.items():
            if key != "attributes":
                collect(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect(child, f"{path}[{index}]")


for key, value in full.items():
    if key != "input":
        collect(value, f"$.{key}")
outputs.sort(key=lambda record: record["path"])
libc_name, libc_version = platform.libc_ver()
profile = {
    "profile_schema": "blender-geometry-nodes-float4x4-host-v1",
    "source": {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": app_string("build_hash"),
        "blender_build_branch": app_string("build_branch"),
        "blender_build_date": app_string("build_date"),
        "blender_build_commit_date": app_string("build_commit_date"),
        "blender_build_platform": app_string("build_platform"),
        "blender_build_type": app_string("build_type"),
        "blender_build_cflags": app_string("build_cflags"),
        "blender_build_cxxflags": app_string("build_cxxflags"),
        "runtime_platform": sys.platform,
        "runtime_machine": platform.machine(),
        "generator_path": "tests/fixtures/geometry_nodes_mesh_ops/generate.py",
        "generator_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        "canonical_fixture_path": "tests/fixtures/geometry_nodes_mesh_ops/expected.json",
        "canonical_fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "canonical_input_float4x4_f32le_sha256": canonical_hash,
        "canonical_input_float4x4_f32le_byte_count": len(canonical_bytes),
        "canonical_input_records": [
            {"name": name, "domain": domain, "data_type": data_type, "count": len(values), "values": values}
            for (name, domain, data_type), values in sorted(canonical_records.items())
        ],
    },
    "record_count": len(outputs),
    "domain_counts": {
        domain: sum(record["domain"] == domain for record in outputs)
        for domain in ("POINT", "EDGE", "FACE", "CORNER")
    },
    "outputs": outputs,
    "execution": {
        "base_image": "ubuntu:24.04",
        "base_image_digest": "sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852",
        "container_platform": "linux/amd64",
        "docker_daemon_platform": "linux/aarch64",
        "architecture_execution": "x86_64 guest under the local ARM64 Docker daemon; emulation backend not independently identified",
        "runtime_libc": f"{libc_name} {libc_version}".strip(),
        "official_blender_archive": "https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz",
        "official_blender_archive_sha256": "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168",
    },
}
serialized_profile = json.dumps(profile, separators=(",", ":"), allow_nan=False) + "\n"
if hashlib.sha256(serialized_profile.encode("utf-8")).hexdigest() != PINNED_PROFILE_SHA256:
    raise RuntimeError("captured profile differs from the frozen host-profile bytes")
profile_path.parent.mkdir(parents=True, exist_ok=True)
profile_path.write_text(serialized_profile, encoding="utf-8")
print(json.dumps({
    "profile_path": str(profile_path),
    "profile_sha256": PINNED_PROFILE_SHA256,
    "record_count": profile["record_count"],
    "domain_counts": profile["domain_counts"],
    "canonical_input_sha256": canonical_hash,
    "blender_build_hash": profile["source"]["blender_build_hash"],
    "runtime_machine": profile["source"]["runtime_machine"],
    "runtime_libc": profile["execution"]["runtime_libc"],
}, sort_keys=True))
