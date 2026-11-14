#!/usr/bin/env python3
"""Plan and run the workspace's Cargo test targets across four CI shards."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARD_COUNT = 4
# Unlisted targets averaged 3.3 s in the latest CI runs.
DEFAULT_SECONDS = 4
# Target weights use latest CI timings; estimates are marked inline.
# Library-kind batch compilation is charged to potter_core; other package entries are runtime.
TARGET_SECONDS = {
    "world_volume_path_oracle": 298,
    "world_volume_gradient_distance": 1140,
    "world_volume_gradient_equiangular": 1001,
    "world_volume_gradient_multiple_importance": 1076,
    "world_volume_noise_distance": 659,
    "world_volume_noise_equiangular": 524,
    "world_volume_noise_multiple_importance": 734,  # Estimate with 1.30x slow-runner factor.
    "world_volume_spatial_emission": 823,
    "world_volume_override_spatial": 1081,
    ("potter_core", "lib-unit"): 582,
    "native_shader_bsdf_wave12": 525,
    "volume_nee": 49,
    "volume_spectral_extinction": 20,
    ("potter_workflow", "lib-doc"): 1,
    ("potter_workflow", "lib-unit"): 1,
    ("potter_core", "lib-doc"): 278,
    "native_principled_anisotropic": 215,
    "native_graph_transparent": 30,
    "vdb_frustum_render": 232,
    "vdb_reader": 28,
    "native_principled_thin_film": 210,
    "blend_roundtrip_constraints": 164,
    "path_transmission_bounce_budget": 131,
    "blend_modifiers_constraints": 118,
    "blend_realistic_scene": 68,
    "native_principled_thin_wall_indirect": 63,
    "cov_nodes_math_extended": 92,
    "shader_p1_oracle": 100,
    "realtime_volume_factor": 50,
    "realtime_shadow_terminator": 56,
    "blend_roundtrip_bbone": 57,
    "backface_culling": 44,
    "realtime_light_factors": 35,
    "closure_list_oracle": 30,
    "realtime_light_shapes_extra": 40,
    "native_metal_sphere_lights": 31,
    "native_light_tree": 32,
    "volume_interpolation": 7,
    "cov_nodes_math": 30,
    "realtime_glass_direct": 27,
    "realtime_direct_light_cases": 27,
    "native_principled_thin_wall": 31,
    "blend_udim_tiles": 30,
    "blend_collection_instances": 31,
    "realtime_flat_emission": 9,
    "constraints_more": 26,
    "native_area_light_mis": 23,
    "realtime_transparent_shadow": 18,
    "realtime_film_filter": 20,
}
KNOWN_KINDS = {
    "lib",
    "proc-macro",
    "bin",
    "test",
    "example",
    "bench",
    "custom-build",
    "dylib",
    "cdylib",
    "staticlib",
    "rlib",
}


def cargo_metadata():
    try:
        result = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"error: cargo metadata exited with status {error.returncode}", file=sys.stderr)
        if error.stderr:
            print(error.stderr, end="" if error.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise SystemExit(1) from error
    except OSError as error:
        print(f"error: could not run cargo metadata: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        print(f"error: cargo metadata returned invalid JSON: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def discover_targets(metadata):
    members = set(metadata["workspace_members"])
    packages = sorted(
        (package for package in metadata["packages"] if package["id"] in members),
        key=lambda package: (package["name"], package["version"], package["id"]),
    )
    targets = []

    for package in packages:
        for target in sorted(package["targets"], key=lambda item: (item["name"], tuple(item["kind"]))):
            kinds = set(target["kind"])
            unknown_kinds = kinds - KNOWN_KINDS
            if unknown_kinds:
                raise ValueError(
                    f"unknown target kind for {package['name']}::{target['name']}: "
                    f"{', '.join(sorted(unknown_kinds))}"
                )
            test = target.get("test", False)
            doctest = target.get("doctest", False)
            if not test and not doctest:
                continue

            name = target["name"]
            package_name = package["name"]
            seconds = TARGET_SECONDS.get(name, DEFAULT_SECONDS)
            if kinds.intersection({"lib", "proc-macro"}):
                if test:
                    targets.append(
                        (
                            package_name,
                            name,
                            "lib-unit",
                            ("--lib",),
                            TARGET_SECONDS.get((name, "lib-unit"), seconds),
                        )
                    )
                if doctest:
                    targets.append(
                        (
                            package_name,
                            name,
                            "lib-doc",
                            ("--doc",),
                            TARGET_SECONDS.get((name, "lib-doc"), seconds),
                        )
                    )
            elif "bin" in kinds and test and not doctest:
                targets.append((package_name, name, "bin-unit", ("--bin", name), seconds))
            elif "test" in kinds and test and not doctest:
                targets.append((package_name, name, "integration", ("--test", name), seconds))
            elif kinds.intersection({"example", "bench"}):
                raise ValueError(f"unsupported test target {package_name}::{name} ({', '.join(sorted(kinds))})")
            else:
                raise ValueError(f"unknown test target kind for {package_name}::{name}: {', '.join(sorted(kinds))}")

    return targets


def assign_targets(targets):
    assignments = [[] for _ in range(SHARD_COUNT)]
    loads = [0] * SHARD_COUNT
    batches = []
    library_units = []
    library_docs = []

    for target in targets:
        if target[2] == "lib-unit":
            library_units.append(target)
        elif target[2] == "lib-doc":
            library_docs.append(target)
        else:
            batches.append([target])
    for group in (library_units, library_docs):
        if group:
            batches.append(group)

    batches.sort(
        key=lambda batch: (
            -sum(target[4] for target in batch),
            tuple(sorted((target[0], target[1], target[2]) for target in batch)),
        )
    )
    for batch in batches:
        shard = min(range(SHARD_COUNT), key=lambda index: (loads[index], index))
        assignments[shard].extend(batch)
        loads[shard] += sum(target[4] for target in batch)

    return assignments


def cargo_commands(targets):
    package_groups = {}
    library_units = []
    library_docs = []

    for target in targets:
        if target[2] == "lib-unit":
            library_units.append(target)
        elif target[2] == "lib-doc":
            library_docs.append(target)
        else:
            package_groups.setdefault(target[0], []).append(target)

    commands = []
    for package_name, group in sorted(package_groups.items()):
        command = ["cargo", "test", "--package", package_name, "--all-features", "--no-fail-fast"]
        for target in group:
            command.extend(target[3])
        commands.append(command)

    for group, selector in ((library_units, "--lib"), (library_docs, "--doc")):
        if group:
            command = ["cargo", "test"]
            for package_name in sorted({target[0] for target in group}):
                command.extend(("--package", package_name))
            command.extend(("--all-features", "--no-fail-fast", selector))
            commands.append(command)

    return commands


def print_plan(shard_index, targets):
    ordered = sorted(targets, key=lambda target: (target[2], target[0], target[1]))
    print(f"Shard {shard_index}: estimated {sum(target[4] for target in ordered)}s")
    print("  targets:")
    for package_name, name, kind, _, seconds in ordered:
        print(f"    {package_name}::{name} ({kind}, {seconds}s)")
    print("  commands:")
    for command in cargo_commands(ordered):
        print(f"    {shlex.join(command)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=range(SHARD_COUNT), required=True)
    parser.add_argument("--run", action="store_true", help="execute the selected shard's Cargo test commands")
    args = parser.parse_args()

    try:
        assignments = assign_targets(discover_targets(cargo_metadata()))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    targets = assignments[args.shard]
    print_plan(args.shard, targets)
    failed = False
    if args.run:
        for command in cargo_commands(targets):
            try:
                subprocess.run(command, cwd=ROOT, check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"error: {error}: {shlex.join(command)}", file=sys.stderr)
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
