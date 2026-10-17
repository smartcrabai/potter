//! Mesh Sequence Cache reads every UV geom parameter into the layer its `sourceName` names.
//!
//! `fixtures/mesh_cache_uv/two_layers.abc` is a Blender 5.2.2 export with asymmetric UVs that
//! identify every loop. The expected UVs are matched to Blender's cache-evaluated corner positions,
//! not to a presumed file or face order. Blender stores V up; Potter stores V down.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Blender 5.2.2 cache corners paired with evaluated positions, converted to Potter's V-down UVs.
type ExpectedCorner = ([f64; 3], [f64; 2]);
const BASE_CORNERS: [ExpectedCorner; 4] = [
    ([-1.0, -1.0, 0.0], [0.137, 0.777]),
    ([1.0, -1.0, 0.0], [0.619, 0.643]),
    ([1.0, 1.0, 0.0], [0.283, 0.119]),
    ([-1.0, 1.0, 0.0], [0.947, 0.431]),
];
const LIGHT_CORNERS: [ExpectedCorner; 4] = [
    ([-1.0, -1.0, 0.0], [0.731, 0.837]),
    ([1.0, -1.0, 0.0], [0.194, 0.093]),
    ([1.0, 1.0, 0.0], [0.853, 0.579]),
    ([-1.0, 1.0, 0.0], [0.367, 0.311]),
];
/// Float32 storage tolerance for the Alembic values.
const TOLERANCE: f64 = 1.0e-6;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mesh_cache_uv/two_layers.abc")
}

fn run_ok(command: &mut Command) -> TestResult<std::process::Output> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn init(scene: &Path) -> TestResult {
    run_ok(pot().args(["init"]).arg(scene).arg("--json"))?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> TestResult {
    let document_path = scene.join("scene.json");
    let document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    let batch = scene.with_extension("uv-layer-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    run_ok(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}

/// Creates a plane `body`, packs the fixture as the `cache` resource, and applies `operations`.
/// The archive is then moved into the project's asset store, as `alembic_cache.rs` does.
fn scene_with_cache(scene: &Path, operations: Vec<Value>) -> TestResult {
    init(scene)?;
    let mut batch = vec![
        json!({"op":"node.create","id":"body","kind":"plane","params":{}}),
        json!({"op":"resource.pack","id":"cache","uri":fixture(),"kind":"alembic"}),
    ];
    batch.extend(operations);
    apply(scene, &Value::Array(batch))?;

    let archive = fs::read(fixture())?;
    let archive_hash = potter_core::hash::sha256(&archive);
    let hash_hex = archive_hash.strip_prefix("sha256:").unwrap_or_default();
    let asset_uri = format!("assets/sha256/{hash_hex}/two_layers.abc");
    let asset_path = scene.join(&asset_uri);
    fs::create_dir_all(asset_path.parent().ok_or("asset path has no parent")?)?;
    fs::write(&asset_path, &archive)?;
    let scene_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    document["resources"]["cache"]["uri"] = json!(asset_uri);
    document["resources"]["cache"]["packed"] = json!(false);
    if let Some(resource) = document["resources"]["cache"].as_object_mut() {
        resource.remove("bytes");
    }
    fs::write(scene_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn cache_modifier(read_data: &[&str]) -> Value {
    json!({
        "op":"modifier.create",
        "target":{"id":"body"},
        "id":"cache",
        "type":"mesh_sequence_cache",
        "params":{
            "resource":"cache",
            "object_path":"/Plane/Plane",
            "read_data":read_data,
        }
    })
}

fn layer_op(op: &str, name: &str) -> Value {
    json!({"op":op, "target":{"id":"body"}, "name":name})
}

/// The evaluated `uv_layers` store, per-face-corner positions, and evaluation warnings.
struct Evaluated {
    store: Value,
    corner_positions: Vec<[f64; 3]>,
    warnings: Vec<Value>,
}

fn evaluate_body(scene: &Path) -> TestResult<Evaluated> {
    let project = potter_core::store::Project::open(scene)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    )?;
    let mesh = snapshot
        .meshes
        .get(&potter_core::model::Id::new("body")?)
        .ok_or("evaluated body mesh is missing")?;
    let corner_positions = mesh
        .faces
        .iter()
        .flat_map(|face| &face.vertices)
        .map(|id| -> TestResult<_> {
            let vertex = mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == *id)
                .ok_or("evaluated face vertex is missing")?;
            Ok(vertex.co.to_array())
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Evaluated {
        store: mesh
            .attributes
            .get("uv_layers")
            .cloned()
            .unwrap_or(Value::Null),
        corner_positions,
        warnings: snapshot
            .warnings
            .iter()
            .map(|warning| warning.details.clone())
            .collect(),
    })
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    match value
        .as_array()
        .ok_or("UV corner is not an array")?
        .as_slice()
    {
        [u, v] => Ok([
            u.as_f64().ok_or("UV coordinate is not a number")?,
            v.as_f64().ok_or("UV coordinate is not a number")?,
        ]),
        _ => Err("UV corner is not a pair".into()),
    }
}

/// The layers of a store in order, as `(name, corners)`. Each layer must hold exactly one face.
fn layers(store: &Value) -> TestResult<Vec<(String, Vec<[f64; 2]>)>> {
    let mut output = Vec::new();
    for layer in store["layers"]
        .as_array()
        .ok_or("uv_layers has no layers")?
    {
        let name = layer["name"]
            .as_str()
            .ok_or("UV layer has no name")?
            .to_owned();
        let [face] = layer["faces"]
            .as_array()
            .ok_or("UV layer has no faces")?
            .as_slice()
        else {
            return Err(format!("UV layer `{name}` must hold exactly one face").into());
        };
        let corners = face["uv"]
            .as_array()
            .ok_or("UV face has no corners")?
            .iter()
            .map(pair)
            .collect::<TestResult<Vec<_>>>()?;
        output.push((name, corners));
    }
    Ok(output)
}

fn layer_names(layers: &[(String, Vec<[f64; 2]>)]) -> Vec<&str> {
    layers.iter().map(|(name, _)| name.as_str()).collect()
}

fn layer<'a>(layers: &'a [(String, Vec<[f64; 2]>)], name: &str) -> TestResult<&'a [[f64; 2]]> {
    layers
        .iter()
        .find(|(layer, _)| layer == name)
        .map(|(_, corners)| corners.as_slice())
        .ok_or_else(|| format!("layer `{name}` is missing").into())
}

fn assert_corners(actual: &[[f64; 2]], expected: &[[f64; 2]], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: corner count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for axis in 0..2 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= TOLERANCE,
                "{label}: corner {index} axis {axis} is {}, expected {}",
                actual[axis],
                expected[axis]
            );
        }
    }
}
fn assert_corner_associations(
    positions: &[[f64; 3]],
    uvs: &[[f64; 2]],
    expected: &[ExpectedCorner],
    label: &str,
) {
    assert_eq!(positions.len(), uvs.len(), "{label}: corner count");
    assert_eq!(
        positions.len(),
        expected.len(),
        "{label}: oracle corner count"
    );
    let mut unmatched = expected.to_vec();
    for (position, uv) in positions.iter().zip(uvs) {
        let Some(index) = unmatched
            .iter()
            .position(|(expected_position, expected_uv)| {
                expected_position
                    .iter()
                    .zip(position)
                    .all(|(expected, actual)| (expected - actual).abs() <= TOLERANCE)
                    && expected_uv
                        .iter()
                        .zip(uv)
                        .all(|(expected, actual)| (expected - actual).abs() <= TOLERANCE)
            })
        else {
            panic!(
                "{label}: no Blender cache UV {uv:?} at corner position {position:?}; actual corner sequence: {:?}",
                positions.iter().zip(uvs).collect::<Vec<_>>()
            );
        };
        unmatched.swap_remove(index);
    }
    assert!(
        unmatched.is_empty(),
        "{label}: unmatched Blender cache corners"
    );
}

#[test]
fn cache_restores_every_uv_parameter_into_the_layer_it_names() -> TestResult {
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    scene_with_cache(&scene, Vec::new())?;
    apply(&scene, &json!([cache_modifier(&["VERT", "POLY", "UV"])]))?;

    let evaluated = evaluate_body(&scene)?;
    let layers = layers(&evaluated.store)?;
    assert_eq!(layer_names(&layers), ["Light", "Base"]);
    assert_corner_associations(
        &evaluated.corner_positions,
        layer(&layers, "Light")?,
        &LIGHT_CORNERS,
        "Light",
    );
    assert_corner_associations(
        &evaluated.corner_positions,
        layer(&layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    // The first cached layer, in file order, is the active and render layer, as in Blender.
    assert_eq!(evaluated.store["active"], json!("Light"));
    assert_eq!(evaluated.store["render"], json!("Light"));
    assert!(evaluated.warnings.is_empty(), "{:?}", evaluated.warnings);
    Ok(())
}

#[test]
fn cache_keeps_mesh_layers_it_does_not_name_and_keeps_selectors() -> TestResult {
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    scene_with_cache(
        &scene,
        vec![
            layer_op("uv.layer_create", "Base"),
            layer_op("uv.layer_create", "Extra"),
            layer_op("uv.layer_set_active", "Extra"),
            layer_op("uv.layer_set_render", "Base"),
        ],
    )?;
    let before = evaluate_body(&scene)?;
    apply(&scene, &json!([cache_modifier(&["VERT", "POLY", "UV"])]))?;
    let after = evaluate_body(&scene)?;

    let before_layers = layers(&before.store)?;
    let after_layers = layers(&after.store)?;
    assert_eq!(layer_names(&before_layers), ["Base", "Extra"]);
    assert_eq!(layer_names(&after_layers), ["Base", "Extra", "Light"]);
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Light")?,
        &LIGHT_CORNERS,
        "Light",
    );
    assert_corners(
        layer(&after_layers, "Extra")?,
        layer(&before_layers, "Extra")?,
        "Extra",
    );
    assert_eq!(after.store["active"], json!("Extra"));
    assert_eq!(after.store["render"], json!("Base"));
    assert!(after.warnings.is_empty(), "{:?}", after.warnings);
    Ok(())
}

#[test]
fn cache_writes_named_layers_past_the_first_and_reports_the_one_without_room() -> TestResult {
    let names = ["L0", "Base", "L2", "L3", "L4", "L5", "L6", "L7"];
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    let operations = names
        .iter()
        .map(|name| layer_op("uv.layer_create", name))
        .collect();
    scene_with_cache(&scene, operations)?;
    let before = evaluate_body(&scene)?;
    // Without topology reads the mesh keeps its own layers, so the limit applies to the store.
    apply(&scene, &json!([cache_modifier(&["UV"])]))?;
    let after = evaluate_body(&scene)?;

    let before_layers = layers(&before.store)?;
    let after_layers = layers(&after.store)?;
    assert_eq!(layer_names(&after_layers), names);
    // `Light` is the canonical cached layer and has no room: it is reported, not written.
    // `Base` is the second layer of the mesh and is overwritten by name, not the first layer.
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    for name in names.iter().filter(|name| **name != "Base") {
        assert_corners(
            layer(&after_layers, name)?,
            layer(&before_layers, name)?,
            name,
        );
    }
    assert_eq!(after.store["active"], json!("L0"));
    assert_eq!(after.store["render"], json!("L0"));
    let [warning] = after.warnings.as_slice() else {
        panic!("expected one warning, found {:?}", after.warnings);
    };
    assert_eq!(warning["feature_id"], json!("uv.layer_limit"));
    assert_eq!(warning["status"], json!("dropped"));
    assert_eq!(warning["layer"], json!("Light"));
    assert_eq!(warning["limit"], json!(8));
    Ok(())
}
