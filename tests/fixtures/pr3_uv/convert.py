# Usage: python3 tests/fixtures/pr3_uv/convert.py <outdir>
"""Writes tests/fixtures/pr3_uv/<variant>.json from generate.py output in <outdir>.

Each fixture holds the Potter input and the Blender corners. Corner rows are
[centroid x, y, z, corner x, y, z, then 2 floats per layer in `layers` order].
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = sys.argv[1]
cases = json.load(open(os.path.join(ORACLE, "cases_uv.json")))


def layer_block(layers):
    return [{"name": layer["name"], "values": layer["values"]} for layer in layers]


def modifiers_for(case):
    mods = [{"type": case["potter"]["type"], "params": case["potter"]["params"]}]
    for extra in case["potter_stack"]:
        kind = {"SUBSURF": "subdivision"}.get(extra["type"], extra["type"].lower())
        mods.append({"type": kind, "params": extra["params"]})
    return mods


for case in cases:
    name = case["variant"]
    blender = json.load(open(os.path.join(ORACLE, "blender", name + ".json")))
    layers = blender["layers"]
    assert layers == [layer["name"] for layer in case["uv_layers"]], name
    rows = []
    for key in sorted(blender["corners"]):
        centroid, point = key.split("#")[0].split("|")
        values = []
        for layer in layers:
            values.extend(blender["corners"][key][layer])
        rows.append([float(x) for x in centroid.split(",")] + [float(x) for x in point.split(",")] + values)
    input_doc = {
        "vertices": case["vertices"],
        "faces": case["faces"],
        "uv_layers": layer_block(case["uv_layers"]),
        "modifiers": modifiers_for(case),
        "caps": {
            prop: {"vertices": cap["vertices"], "faces": cap["faces"],
                   "uv_layers": layer_block(cap["uv_layers"]), "loc": cap["loc"]}
            for prop, cap in case["caps"].items()
        },
    }
    if case["edges"]:
        input_doc["edges"] = case["edges"]
    text = (
        "{\n"
        f'"variant": {json.dumps(name)},\n'
        f'"blender_version": {json.dumps(blender["blender_version"])},\n'
        f'"layers": {json.dumps(layers)},\n'
        f'"input": {json.dumps(input_doc, separators=(",", ":"))},\n'
        '"corners": [\n'
        + ",\n".join(json.dumps(row, separators=(",", ":")) for row in rows)
        + "\n]\n}\n"
    )
    json.loads(text)  # validate before writing
    with open(os.path.join(HERE, name + ".json"), "w") as handle:
        handle.write(text)
    print(name, len(rows), len(text))
