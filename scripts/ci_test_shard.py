#!/usr/bin/env python3
"""Plan and run the workspace's Cargo test targets across three CI shards."""

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHARD_COUNT = 3
DEFAULT_SECONDS = 10
TARGET_SECONDS = {
    "blend_realistic_scene": 1590,
    "constraints_more": 957,
}


@dataclass(frozen=True)
class Target:
    package_id: str
    package_name: str
    name: str
    kind: str
    selector: tuple[str, ...]
    seconds: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.package_id, self.name, self.kind)


def cargo_metadata() -> dict[str, Any]:
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


def discover_targets(metadata: dict[str, Any]) -> list[Target]:
    workspace_members = set(metadata["workspace_members"])
    packages = sorted(
        (package for package in metadata["packages"] if package["id"] in workspace_members),
        key=lambda package: (package["name"], package["version"], package["id"]),
    )
    targets: list[Target] = []

    for package in packages:
        for target in sorted(package["targets"], key=lambda item: (item["name"], tuple(item["kind"]))):
            name = target["name"]
            kinds = set(target["kind"])
            seconds = TARGET_SECONDS.get(name, DEFAULT_SECONDS)
            package_id = package["id"]
            package_name = package["name"]

            if kinds.intersection({"lib", "proc-macro"}):
                if target.get("test", False):
                    targets.append(Target(package_id, package_name, name, "lib-unit", ("--lib",), seconds))
                if target.get("doctest", False):
                    targets.append(Target(package_id, package_name, name, "lib-doc", ("--doc",), seconds))
            elif "bin" in kinds and target.get("test", False):
                targets.append(Target(package_id, package_name, name, "bin-unit", ("--bin", name), seconds))
            elif "test" in kinds and target.get("test", False):
                targets.append(Target(package_id, package_name, name, "integration", ("--test", name), seconds))

    return targets


def assign_targets(targets: list[Target]) -> list[list[Target]]:
    assignments: list[list[Target]] = [[] for _ in range(SHARD_COUNT)]
    loads = [0] * SHARD_COUNT
    integration_targets = []

    for target in targets:
        if target.kind == "integration":
            integration_targets.append(target)
        else:
            assignments[0].append(target)
            loads[0] += target.seconds

    integration_targets.sort(key=lambda target: (-target.seconds, target.package_name, target.name, target.package_id))
    for target in integration_targets:
        shard = min(range(SHARD_COUNT), key=lambda index: (loads[index], index))
        assignments[shard].append(target)
        loads[shard] += target.seconds

    return assignments


def validate_assignments(targets: list[Target], assignments: list[list[Target]]) -> None:
    expected = Counter(target.key for target in targets)
    actual = Counter(target.key for shard in assignments for target in shard)
    if expected != actual:
        raise ValueError("target assignment does not include every Cargo test target exactly once")
    for shard_index, shard in enumerate(assignments):
        if shard_index != 0 and any(target.kind != "integration" for target in shard):
            raise ValueError("library, binary, and documentation tests must be assigned to shard 0")


def package_groups(targets: list[Target]) -> list[list[Target]]:
    grouped: dict[str, list[Target]] = {}
    for target in targets:
        grouped.setdefault(target.package_name, []).append(target)
    return [grouped[name] for name in sorted(grouped)]


def cargo_command(targets: list[Target]) -> list[str]:
    return [
        "cargo",
        "test",
        "--package",
        targets[0].package_name,
        "--all-features",
        "--no-fail-fast",
        *(selector for target in targets for selector in target.selector),
    ]


def print_plan(shard_index: int, targets: list[Target]) -> None:
    ordered = sorted(targets, key=lambda target: (target.kind, target.package_name, target.name, target.package_id))
    estimate = sum(target.seconds for target in ordered)
    print(f"Shard {shard_index}: estimated {estimate}s")
    print("  targets:")
    if ordered:
        for target in ordered:
            print(f"    {target.package_name}::{target.name} ({target.kind}, {target.seconds}s)")
    else:
        print("    (none)")
    print("  commands:")
    groups = package_groups(ordered)
    if groups:
        for group in groups:
            print(f"    {shlex.join(cargo_command(group))}")
    else:
        print("    (none)")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify every Cargo test target is assigned once")
    parser.add_argument("--shard", type=int, choices=range(SHARD_COUNT), help="show or run shard 0, 1, or 2")
    parser.add_argument("--run", action="store_true", help="execute the selected shard's Cargo test commands")
    args = parser.parse_args()

    if args.check and args.shard is not None:
        parser.error("--check cannot be combined with --shard")
    if args.check and args.run:
        parser.error("--check cannot be combined with --run")
    if not args.check and args.shard is None:
        parser.error("specify --check or --shard 0|1|2")
    if args.run and args.shard is None:
        parser.error("--run requires --shard")

    metadata = cargo_metadata()
    targets = discover_targets(metadata)
    assignments = assign_targets(targets)
    try:
        validate_assignments(targets, assignments)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        print(f"OK: all {len(targets)} Cargo test targets are assigned exactly once across {SHARD_COUNT} shards.")
        return 0

    shard_targets = assignments[args.shard]
    print_plan(args.shard, shard_targets)
    if not args.run:
        return 0

    failed = False
    for group in package_groups(shard_targets):
        command = cargo_command(group)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            print(f"error: command exited with status {error.returncode}: {shlex.join(command)}", file=sys.stderr)
            failed = True
        except OSError as error:
            print(f"error: could not run {shlex.join(command)}: {error}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
