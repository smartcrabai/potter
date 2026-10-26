#!/usr/bin/env python3
"""Plan and run the workspace's Cargo test targets across three CI shards."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARD_COUNT = 3
DEFAULT_SECONDS = 10
TARGET_SECONDS = {
    "blend_realistic_scene": 1591,
    "workbench_studio_light": 274,
    ("potter_core", "lib-unit"): 257,
    "native_principled_thin_film": 231,
    "native_principled_anisotropic": 231,
    "potter_core": 223,
    ("potter_workflow", "lib-unit"): 211,
    "blend_roundtrip_constraints": 124,
    "blend_modifiers_constraints": 108,
    "native_principled_thin_wall_indirect": 79,
    "blend_roundtrip_bbone": 52,
    "realtime_volume_factor": 49,
    "backface_culling": 30,
    "closure_list_oracle": 29,
    "blend_collection_instances": 29,
    "blend_udim_tiles": 28,
    "realtime_light_shapes_extra": 25,
    "native_graph_transparent": 25,
    "native_metal_sphere_lights": 21,
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
    integration_targets = []

    for target in targets:
        if target[2] == "integration":
            integration_targets.append(target)
        else:
            assignments[0].append(target)
            loads[0] += target[4]

    integration_targets.sort(key=lambda target: (-target[4], target[0], target[1]))
    for target in integration_targets:
        shard = min(range(SHARD_COUNT), key=lambda index: (loads[index], index))
        assignments[shard].append(target)
        loads[shard] += target[4]

    return assignments


def cargo_commands(targets):
    groups = {}
    for target in targets:
        groups.setdefault((target[0], target[2] == "lib-doc"), []).append(target)

    commands = []
    for (package_name, doctests), group in sorted(groups.items()):
        command = ["cargo", "test", "--package", package_name, "--all-features", "--no-fail-fast"]
        if doctests:
            command.append("--doc")
        else:
            for target in group:
                command.extend(target[3])
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
