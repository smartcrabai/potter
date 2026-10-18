// SPDX-License-Identifier: GPL-3.0-or-later

//! Shrinkwrap differential oracle against Blender 5.2.2. `generate.py` runs Blender on
//! `tests/fixtures/shrinkwrap-differential/corpus.json`, writes one `.blend` scene per case and
//! Blender's evaluated result. Each imported scene is evaluated through Potter and compared with
//! Blender: topology exactly, positions, corner normals, UVs, colors and vertex weights within the
//! corpus tolerances. Skips without Blender 5.2.2; `POTTER_REQUIRE_BLENDER` turns the skip into a
//! failure.

use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use tempfile::tempdir;

use super::{Mesh, corner_normal_vectors};
use crate::{
    eval::{EvaluationContext, Snapshot, mesh_vertex_weights},
    exchange::blend::import_blend,
    geom::{
        color_layers::{self, ColorDomain},
        uv_layers,
    },
    model::{Id, SceneDoc, VertexGroup},
};

#[path = "../../../../tests/common/blender_discovery.rs"]
mod discovery;
#[path = "../../../../tests/common/process.rs"]
mod process;

const CORPUS: &str = include_str!("../../../../tests/fixtures/shrinkwrap-differential/corpus.json");
const BLENDER_VERSION: &str = "5.2.2 LTS";

#[test]
fn shrinkwrap_matches_blender_522_differential_corpus() -> Result<(), Box<dyn std::error::Error>> {
    let corpus: Value = serde_json::from_str(CORPUS)?;
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    assert_eq!(corpus["blender_version"], BLENDER_VERSION);

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture_path = root.join("tests/fixtures/shrinkwrap-differential/corpus.json");
    let directory = tempdir()?;
    let live_path = directory.path().join("blender-oracle.json");
    let blend_directory = directory.path().join("blend-scenes");
    let live = generate_blender_oracle(&blender, &fixture_path, &live_path, &blend_directory)?;
    let absolute = corpus["comparison"]["absolute_tolerance"]
        .as_f64()
        .ok_or("missing absolute tolerance")?;
    let relative = corpus["comparison"]["relative_tolerance"]
        .as_f64()
        .ok_or("missing relative tolerance")?;
    assert_blender_match(
        "source mesh oracle",
        &corpus["source_blender_input"],
        &live["source_blender_input"],
        absolute,
        relative,
    );

    let cases = corpus["cases"]
        .as_array()
        .ok_or("missing Shrinkwrap cases")?;
    let live_cases = live["cases"]
        .as_array()
        .ok_or("missing live Blender cases")?;
    assert_eq!(cases.len(), live_cases.len(), "Blender case count changed");
    for case in cases {
        let id = case["id"].as_str().ok_or("case id is not a string")?;
        let parameters = case["params"]
            .as_object()
            .ok_or("case params are missing")?;
        assert_eq!(
            parameters.len(),
            corpus["defaults"]
                .as_object()
                .ok_or("missing defaults")?
                .len(),
            "{id}: fixture parameters must be fully specified"
        );
        let live_case = live_cases
            .iter()
            .find(|candidate| candidate["id"].as_str() == Some(id))
            .ok_or_else(|| format!("live Blender result is missing case {id}"))?;
        assert_eq!(
            case["params"], live_case["params"],
            "{id}: live inputs differ"
        );
        assert_blender_match(
            id,
            &case["blender"],
            &live_case["blender"],
            absolute,
            relative,
        );

        let blend_path = blend_directory.join(format!("{id}.blend"));
        let (doc, snapshot, source_id) = import_blender_scene(&blend_path, &blender)?;
        let source_data_id = doc
            .nodes
            .get(&source_id)
            .and_then(|node| node.data.as_ref())
            .ok_or("imported source has no data")?;
        let source_data = doc
            .data_blocks
            .get(source_data_id)
            .ok_or("imported source data is missing")?;
        let source_mesh = source_data
            .mesh
            .as_ref()
            .ok_or("imported source data has no mesh")?;
        let output_mesh = snapshot
            .meshes
            .get(&source_id)
            .ok_or("evaluated source mesh is missing")?;
        let output_weights = mesh_vertex_weights(output_mesh, &source_data.vertex_groups);
        let source_ids = mesh_summary(
            source_mesh,
            &source_data.vertex_groups,
            &source_data.vertex_weights,
        )?;
        let actual = mesh_summary(output_mesh, &source_data.vertex_groups, &output_weights)?;
        for field in ["vertex_ids", "edge_ids", "face_ids"] {
            assert_eq!(
                source_ids[field], actual[field],
                "{id}: {field} IDs changed"
            );
        }
        assert_potter_match(id, &live_case["blender"], &actual, absolute, relative);
    }
    Ok(())
}

fn blender_522() -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let Some(blender) = discovery::discover(|path| {
        if !path.is_file() {
            return None;
        }
        let mut command = Command::new(&path);
        command.arg("--version");
        process::run_guarded(command)
            .ok()
            .filter(|output| output.status.success())
            .map(|_| path)
    }) else {
        eprintln!("skipping Shrinkwrap differential corpus: Blender is unavailable");
        return Ok(None);
    };
    let mut version_command = Command::new(&blender);
    version_command.arg("--version");
    let version = process::run_guarded(version_command)?;
    assert!(version.status.success(), "Blender --version failed");
    let version_text = String::from_utf8_lossy(&version.stdout);
    if !version_text.contains("Blender 5.2.2") {
        assert!(
            env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
            "POTTER_REQUIRE_BLENDER is set but Blender is not 5.2.2:\n{version_text}"
        );
        eprintln!("skipping Shrinkwrap differential corpus: this fixture requires Blender 5.2.2");
        return Ok(None);
    }
    Ok(Some(blender))
}

fn generate_blender_oracle(
    blender: &Path,
    fixture_path: &Path,
    live_path: &Path,
    blend_directory: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script_path = root.join("tests/fixtures/shrinkwrap-differential/generate.py");
    let mut command = Command::new(blender);
    command.args([
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "3",
        "--python",
    ]);
    command
        .arg(script_path)
        .arg("--")
        .arg(fixture_path)
        .arg(live_path)
        .arg(blend_directory);
    let output = process::run_guarded(command)?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "Blender Shrinkwrap oracle failed:\n{text}"
    );
    let live: Value = serde_json::from_slice(&fs::read(live_path)?)?;
    assert_eq!(live["blender_version"], BLENDER_VERSION);
    Ok(live)
}

fn import_blender_scene(
    blend_path: &Path,
    blender: &Path,
) -> Result<(SceneDoc, Snapshot, Id), Box<dyn std::error::Error>> {
    let imported = import_blend(blend_path, Some(blender))?;
    let source_text = imported
        .id_mappings
        .get("Object:Source")
        .and_then(Value::as_str)
        .ok_or("Blender object `Source` has no import ID")?;
    let source_id = Id::new(source_text.to_owned())?;
    let snapshot = Snapshot::evaluate(&imported.doc, &EvaluationContext::default())?;
    Ok((imported.doc, snapshot, source_id))
}

fn mesh_summary(
    mesh: &Mesh,
    groups: &[VertexGroup],
    weights: &BTreeMap<u32, BTreeMap<Id, f64>>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let vertex_ids: Vec<_> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    let edge_ids: Vec<_> = mesh
        .edges
        .iter()
        .map(|edge| json!({"id":edge.id,"vertices":edge.vertices}))
        .collect();
    let face_ids: Vec<_> = mesh
        .faces
        .iter()
        .map(|face| json!({"id":face.id,"vertices":&face.vertices}))
        .collect();
    let positions: Vec<_> = mesh
        .vertices
        .iter()
        .map(|vertex| json!(vertex.co.to_array()))
        .collect();
    let mut edges: Vec<Vec<usize>> = mesh
        .edges
        .iter()
        .map(|edge| {
            let mut indices = [
                vertex_index[&edge.vertices[0]],
                vertex_index[&edge.vertices[1]],
            ];
            indices.sort_unstable();
            indices.to_vec()
        })
        .collect();
    edges.sort_unstable();
    let faces: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| vertex_index[id]).collect())
        .collect();
    let corner_normals: Vec<_> = corner_normal_vectors(mesh)?
        .into_iter()
        .map(|normal| json!(normal.to_array()))
        .collect();
    let uv_layers: Vec<Value> = uv_layers::read(mesh)
        .layers
        .into_iter()
        .map(|layer| {
            let values = mesh
                .faces
                .iter()
                .enumerate()
                .flat_map(|(index, face)| {
                    let face_values = layer.faces.get(index).and_then(Option::as_ref);
                    (0..face.vertices.len()).map(move |corner| {
                        let [u, v] = face_values
                            .and_then(|values| values.get(corner))
                            .copied()
                            .unwrap_or([0.0, 0.0]);
                        json!([u, 1.0 - v])
                    })
                })
                .collect::<Vec<_>>();
            json!({"name":layer.name,"values":values})
        })
        .collect();
    let color_data_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object);
    let color_attributes: Vec<Value> = color_layers::read(mesh)
        .layers
        .into_iter()
        .map(|layer| {
            let data_type = color_data_types
                .and_then(|types| types.get(&layer.name))
                .and_then(Value::as_str)
                .ok_or("Blender color data type metadata is missing")?;
            let domain = match layer.domain {
                ColorDomain::Point => "POINT",
                ColorDomain::Corner => "CORNER",
            };
            Ok(json!({
                "name":layer.name,
                "domain":domain,
                "data_type":data_type,
                "values":layer.values
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let vertex_groups: Vec<Value> = groups
        .iter()
        .map(|group| {
            let values: Vec<f64> = mesh
                .vertices
                .iter()
                .map(|vertex| {
                    weights
                        .get(&vertex.id)
                        .and_then(|groups| groups.get(&group.id))
                        .copied()
                        .unwrap_or(0.0)
                })
                .collect();
            json!({"name":group.name,"values":values})
        })
        .collect();
    Ok(json!({
        "positions":positions,
        "edges":edges,
        "faces":faces,
        "vertex_ids":vertex_ids,
        "edge_ids":edge_ids,
        "face_ids":face_ids,
        "corner_normals":corner_normals,
        "uv_layers":uv_layers,
        "color_attributes":color_attributes,
        "vertex_groups":vertex_groups
    }))
}

fn assert_blender_match(
    context: &str,
    expected: &Value,
    actual: &Value,
    absolute: f64,
    relative: f64,
) {
    assert_eq!(
        expected["edges"], actual["edges"],
        "{context}: Blender edges differ"
    );
    assert_eq!(
        expected["faces"], actual["faces"],
        "{context}: Blender faces differ"
    );
    for field in [
        "positions",
        "vertex_normals",
        "corner_normals",
        "uv_layers",
        "color_attributes",
        "vertex_groups",
    ] {
        if expected.get(field).is_some() {
            assert_json_close(
                &format!("{context}.{field}"),
                &expected[field],
                &actual[field],
                absolute,
                relative,
            );
        }
    }
}

fn assert_potter_match(
    context: &str,
    expected: &Value,
    actual: &Value,
    absolute: f64,
    relative: f64,
) {
    assert_eq!(
        expected["edges"], actual["edges"],
        "{context}: Potter edges differ from Blender"
    );
    assert_eq!(
        expected["faces"], actual["faces"],
        "{context}: Potter faces differ from Blender"
    );
    for field in [
        "positions",
        "corner_normals",
        "uv_layers",
        "color_attributes",
        "vertex_groups",
    ] {
        assert_json_close(
            &format!("{context}.{field}"),
            &expected[field],
            &actual[field],
            absolute,
            relative,
        );
    }
}

fn assert_json_close(
    context: &str,
    expected: &Value,
    actual: &Value,
    absolute: f64,
    relative: f64,
) {
    match (expected, actual) {
        (Value::Number(expected), Value::Number(actual)) => {
            let Some(expected) = expected.as_f64() else {
                panic!("{context}: expected finite JSON number");
            };
            let Some(actual) = actual.as_f64() else {
                panic!("{context}: actual finite JSON number");
            };
            let tolerance = absolute + relative * expected.abs();
            assert!(
                (expected - actual).abs() <= tolerance,
                "{context}: {actual} differs from Blender {expected} by more than {tolerance}"
            );
        }
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(
                expected.len(),
                actual.len(),
                "{context}: array length differs"
            );
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_json_close(
                    &format!("{context}[{index}]"),
                    expected,
                    actual,
                    absolute,
                    relative,
                );
            }
        }
        (Value::Object(expected), Value::Object(actual)) => {
            assert_eq!(
                expected.keys().collect::<Vec<_>>(),
                actual.keys().collect::<Vec<_>>(),
                "{context}: object fields differ"
            );
            for (key, expected) in expected {
                assert_json_close(
                    &format!("{context}.{key}"),
                    expected,
                    &actual[key],
                    absolute,
                    relative,
                );
            }
        }
        _ => assert_eq!(expected, actual, "{context}: values differ"),
    }
}
