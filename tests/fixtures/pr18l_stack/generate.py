#!/usr/bin/env python3
"""Regenerate tests/fixtures/pr18l_stack/expected.json from Blender 5.2 (PR18L oracle).

Usage:
    python3 tests/fixtures/pr18l_stack/generate.py [--artifacts-dir DIR] [--output FILE]

The Blender executable is taken from $BLENDER, then /Applications/Blender.app, then PATH.
Blender runs blender_oracle.py headless; that script writes raw observations which this
script converts into the committed expected.json. The output is deterministic: no timestamps,
no absolute paths, sorted keys.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORACLE = HERE / "blender_oracle.py"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_ARTIFACTS = "/tmp/pr18l-artifacts/blender"
DEFAULT_OUTPUT = HERE / "expected.json"
DISABLED_MESSAGE = "Modifier is disabled"


def find_blender():
    candidates = [os.environ.get("BLENDER"), DEFAULT_BLENDER, shutil.which("blender")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    sys.exit("Blender executable not found; set BLENDER to the Blender 5.2 binary")


def run_blender(blender, artifacts_dir):
    with tempfile.TemporaryDirectory(prefix="pr18l_generate_") as work:
        raw_path = Path(work) / "raw.json"
        process = subprocess.run(
            [
                blender,
                "--background",
                "--factory-startup",
                "--python",
                str(ORACLE),
                "--",
                str(raw_path),
                str(artifacts_dir),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0 or not raw_path.is_file():
            sys.stderr.write(process.stdout)
            sys.stderr.write(process.stderr)
            sys.exit(f"Blender oracle failed with exit code {process.returncode}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    return raw, process.stdout + process.stderr


def build_expected(raw, blender_log):
    apply_results = raw["apply_viewport_disabled"]
    disabled_message_printed = DISABLED_MESSAGE in blender_log
    for result in apply_results.values():
        result["disabled_message_printed_during_probes"] = disabled_message_printed
    return {
        "schema": "pr18l.modifier_stack_semantics.v1",
        "generator": "tests/fixtures/pr18l_stack/generate.py",
        "oracle": "tests/fixtures/pr18l_stack/blender_oracle.py",
        "blender": raw["blender"],
        "position_hash": raw["position_hash"],
        "render_method": raw["render_method"],
        "kinds": raw["kinds"],
        "cases": raw["cases"],
        "checks": raw["checks"],
        "apply_viewport_disabled": apply_results,
        "notes": raw["notes"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()

    blender = find_blender()
    artifacts_dir = Path(arguments.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    raw, blender_log = run_blender(blender, artifacts_dir)
    expected = build_expected(raw, blender_log)

    output = Path(arguments.output)
    output.write_text(
        json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    for path in raw["artifacts"]:
        print(f"artifact {path}")


if __name__ == "__main__":
    main()
