"""Regenerate the primitive UV oracle in several Blender processes and record what is run-dependent.

Usage: python3 check_uv_sphere_runs.py <out.json> [runs]

Each run is a fresh Blender process running generate_primitive_uv_oracle.py. The script asserts that
UV sphere vertex lists are identical in every run, and that the UV sphere edge set and face set
(each face kept as its exact loop list, so loop starts are compared) are identical too. Only the
order of edges and faces may differ between runs. The first run's output is written to <out.json>
with an `uv_sphere_run_check` summary added at the top level.
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATOR = os.path.join(HERE, "generate_primitive_uv_oracle.py")
BLENDER = os.environ.get("POTTER_BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")


def run_generator(path):
    subprocess.run(
        [BLENDER, "-b", "--factory-startup", "--python", GENERATOR, "--", path],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def undirected_edges(case):
    return sorted(sorted(edge) for edge in case["edges"])


def face_lists(case):
    return sorted(case["faces"])


def main():
    out_path = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    directory = tempfile.mkdtemp(prefix="uv_oracle_runs_")
    documents = [run_generator(os.path.join(directory, f"run{i}.json")) for i in range(runs)]
    first = documents[0]

    summary = {}
    non_uv_differences = []
    for name, case in first["cases"].items():
        others = [document["cases"][name] for document in documents]
        if case["kind"] != "uv_sphere":
            if any(other != case for other in others):
                non_uv_differences.append(name)
            continue
        for other in others:
            assert other["vertices"] == case["vertices"], f"{name}: vertex order differs between runs"
            assert undirected_edges(other) == undirected_edges(case), f"{name}: edge set differs"
            assert face_lists(other) == face_lists(case), f"{name}: face set or loop starts differ"
        summary[name] = {
            "vertices_identical": True,
            "undirected_edge_set_identical": True,
            "face_set_and_loop_starts_identical": True,
            "distinct_directed_edge_orders": len({json.dumps(other["edges"]) for other in others}),
            "distinct_face_orders": len({json.dumps(other["faces"]) for other in others}),
        }

    first["uv_sphere_run_check"] = {
        "processes": runs,
        "cases": summary,
        "non_uv_cases_differing_between_runs": non_uv_differences,
    }
    with open(out_path, "w", encoding="utf-8") as destination:
        json.dump(first, destination, indent=1)
        destination.write("\n")
    print(json.dumps(first["uv_sphere_run_check"], indent=1))


if __name__ == "__main__":
    main()
