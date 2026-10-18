//! Blender oracle for `uv.unwrap` with `method: "box"`.
//!
//! Potter unwraps a size-2 cube and an offset size-4 plane with the box method. Blender 5.2.2's
//! `uv.cube_project` runs on the same geometry, one object at a time, with `cube_size` unset and
//! the default options. Each face corner of each object must carry the UV Blender gives it. Blender
//! writes V up and Potter stores V down, so Blender's `v` is compared with Potter's `1 - v`.
//!
//! The geometry comes from Potter's document, so the vertices and face order match exactly. Each
//! corner is keyed by its face centroid and its vertex position, which is unique for a box or a
//! plane. Positions are rounded to 1e-4 in both languages so the keys agree.
//!
//! Skips without Blender, as the other Blender-gated tests do. `POTTER_REQUIRE_BLENDER` turns the
//! skip into a failure.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{collections::BTreeMap, error::Error, fs, io, path::Path, process::Command};

use blender_checked::blender_executable;
use potter_core::{
    geom::Mesh,
    model::{Id, SceneDoc},
    ops::apply_batch,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Tolerance for UVs. Blender computes in float32.
const TOLERANCE: f64 = 1.0e-5;
/// Scene ID for the source document.
const SCENE_ID: &str = "6f1c2a4e-3b5d-4c8e-9a7f-1d2e3f4a5b6c";
/// Objects unwrapped with the box method, in build order.
const OBJECTS: [&str; 2] = ["cube", "plane"];

/// Blender side: each object gets a fresh mesh from the spec, is unwrapped with `uv.cube_project`
/// in edit mode, and every corner's UV is written under its centroid-and-vertex key.
const BLENDER_SCRIPT: &str = r#"
import json
import sys

import bpy

spec_path, out_path = sys.argv[sys.argv.index("--") + 1 :][:2]
with open(spec_path, encoding="utf-8") as handle:
    spec = json.load(handle)


def key(point):
    return ",".join("%d" % round(value * 10000) for value in point)


bpy.ops.wm.read_factory_settings(use_empty=True)
result = {}
for entry in spec["objects"]:
    mesh = bpy.data.meshes.new(entry["name"])
    mesh.from_pydata(entry["vertices"], [], entry["faces"])
    mesh.update()
    obj = bpy.data.objects.new(entry["name"], mesh)
    bpy.context.collection.objects.link(obj)
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    layer = mesh.uv_layers.active
    corners = {}
    for polygon in mesh.polygons:
        points = [mesh.vertices[index].co for index in polygon.vertices]
        centroid = [sum(point[axis] for point in points) / len(points) for axis in range(3)]
        for loop_index in polygon.loop_indices:
            loop = mesh.loops[loop_index]
            co = mesh.vertices[loop.vertex_index].co
            uv = layer.data[loop_index].uv
            corners[key(centroid) + "|" + key(co)] = [float(uv[0]), float(uv[1])]
    result[entry["name"]] = corners
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#;

/// The source document: a size-2 cube at the origin and a size-4 plane whose mesh vertices are translated off
/// the origin, both box-unwrapped. The plane keeps an object transform as well. The mesh's own centre is off the
/// origin, so a box projection that used the wrong centre would fail the oracle.
fn scene() -> TestResult<SceneDoc> {
    let empty = SceneDoc::new(SCENE_ID.to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": empty.revision,
        "operations": [
            {"op": "node.create", "id": "cube", "kind": "box", "params": {"size": 2.0}},
            {"op": "uv.unwrap", "target": {"id": "cube"}, "method": "box"},
            {
                "op": "node.create",
                "id": "plane",
                "kind": "plane",
                "params": {"size": 4.0},
                "transform": {"translation": [3.0, 0.0, 0.0]}
            },
            {
                "op": "mesh.transform_elements",
                "target": {
                    "id": "plane",
                    "elements": {"domain": "vertex", "ids": ["v0", "v1", "v2", "v3"]}
                },
                "translation": [3.0, 1.0, 0.5]
            },
            {"op": "uv.unwrap", "target": {"id": "plane"}, "method": "box"}
        ]
    });
    Ok(apply_batch(&empty, &batch)?.doc)
}

fn node_mesh<'a>(doc: &'a SceneDoc, name: &str) -> TestResult<&'a Mesh> {
    let node = doc
        .nodes
        .get(&Id::new(name)?)
        .ok_or_else(|| format!("node `{name}` is missing"))?;
    let data = node
        .data
        .as_ref()
        .ok_or_else(|| format!("node `{name}` has no data"))?;
    doc.data_blocks
        .get(data)
        .and_then(|block| block.mesh.as_ref())
        .ok_or_else(|| format!("node `{name}` has no mesh").into())
}

/// The rounded text form of a position, the same in the Blender script.
fn position_key(point: [f64; 3]) -> String {
    point
        .iter()
        .map(|value| format!("{}", (value * 10_000.0).round() as i64))
        .collect::<Vec<_>>()
        .join(",")
}

/// The corner key shared with the Blender script: rounded face centroid and corner position.
fn corner_key(centroid: [f64; 3], point: [f64; 3]) -> String {
    format!("{}|{}", position_key(centroid), position_key(point))
}

/// The geometry Blender rebuilds, with vertices and faces in Potter's storage order.
fn geometry(doc: &SceneDoc, name: &str) -> TestResult<Value> {
    let mesh = node_mesh(doc, name)?;
    let index = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| [vertex.co.x, vertex.co.y, vertex.co.z])
        .collect::<Vec<_>>();
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    index
                        .get(id)
                        .copied()
                        .ok_or_else(|| format!("face references missing vertex {id}"))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"name": name, "vertices": vertices, "faces": faces}))
}

/// Potter's stored UV per corner of `name`'s `UVMap` layer, under the shared corner key.
fn potter_corners(doc: &SceneDoc, name: &str) -> TestResult<BTreeMap<String, [f64; 2]>> {
    let mesh = node_mesh(doc, name)?;
    let store = &mesh.attributes["uv_layers"];
    let layer = store["layers"]
        .as_array()
        .and_then(|layers| layers.iter().find(|layer| layer["name"] == "UVMap"))
        .ok_or_else(|| format!("node `{name}` has no UVMap layer"))?;
    let entries = layer["faces"]
        .as_array()
        .ok_or_else(|| format!("node `{name}` UVMap has no faces"))?;
    let mut corners = BTreeMap::new();
    for face in &mesh.faces {
        let entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
            .ok_or_else(|| format!("face {} has no UV entry", face.id))?;
        let points = face
            .vertices
            .iter()
            .map(|id| {
                mesh.vertex(*id)
                    .map(|vertex| [vertex.co.x, vertex.co.y, vertex.co.z])
                    .ok_or_else(|| format!("missing vertex {id}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let count = points.len() as f64;
        let centroid =
            [0, 1, 2].map(|axis| points.iter().map(|point| point[axis]).sum::<f64>() / count);
        for (corner, point) in points.iter().enumerate() {
            let uv = [
                entry["uv"][corner][0]
                    .as_f64()
                    .ok_or_else(|| format!("face {} corner {corner} u is missing", face.id))?,
                entry["uv"][corner][1]
                    .as_f64()
                    .ok_or_else(|| format!("face {} corner {corner} v is missing", face.id))?,
            ];
            corners.insert(corner_key(centroid, *point), uv);
        }
    }
    Ok(corners)
}

/// Runs the Blender script over the geometry spec and returns its per-object corner UVs.
fn blender_corners(blender: &Path, root: &Path, spec: &Value) -> TestResult<Value> {
    let spec_path = root.join("geometry.json");
    let out_path = root.join("blender.json");
    let script = root.join("box_projection.py");
    fs::write(&spec_path, serde_json::to_vec(spec)?)?;
    fs::write(&script, BLENDER_SCRIPT)?;
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(&script)
        .arg("--")
        .arg(&spec_path)
        .arg(&out_path)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "blender cube_project failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(serde_json::from_slice(&fs::read(&out_path)?)?)
}

#[test]
fn box_projection_matches_blender_cube_project() -> TestResult {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let doc = scene()?;
    let geometries = OBJECTS
        .iter()
        .map(|name| geometry(&doc, name))
        .collect::<TestResult<Vec<_>>>()?;
    let directory = tempdir()?;
    let expected = blender_corners(&blender, directory.path(), &json!({"objects": geometries}))?;

    for name in OBJECTS {
        let potter = potter_corners(&doc, name)?;
        let blender_object = expected[name]
            .as_object()
            .ok_or_else(|| format!("Blender returned no corners for `{name}`"))?;
        assert_eq!(
            potter.len(),
            blender_object.len(),
            "{name}: corner count differs from Blender"
        );
        for (key, uv) in &potter {
            let blender_uv = blender_object
                .get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name}: Blender has no corner {key}"))?;
            let blender_u = blender_uv
                .first()
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("{name} corner {key}: Blender u is missing"))?;
            let blender_v = blender_uv
                .get(1)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("{name} corner {key}: Blender v is missing"))?;
            // Blender writes V up; Potter stores V down.
            let stored_v = 1.0 - blender_v;
            assert!(
                (uv[0] - blender_u).abs() < TOLERANCE && (uv[1] - stored_v).abs() < TOLERANCE,
                "{name} corner {key}: Potter stores {uv:?}, Blender gives [{blender_u}, {blender_v}] (V flipped to {stored_v})"
            );
            assert!(
                (0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]),
                "{name} corner {key} is outside [0, 1]: {uv:?}"
            );
        }
    }
    Ok(())
}
