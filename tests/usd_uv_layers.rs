//! USD UV layers: every `texCoord2f` primvar imports as a canonical layer with its name, indices,
//! interpolation, V convention, and render selection. Forms that cannot be read are reported as
//! `mesh.uv_layer` losses, one per primvar, and Potter's usda output keeps every layer.

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{
    fs,
    path::{Path, PathBuf},
    process::Output,
};

use exchange_normals::{TestResult, export_strict, init, pot};
use serde_json::{Value, json};
use tempfile::tempdir;

/// Blender's UV values are float32, so they differ from the decimal constants by about 1e-8.
const TOLERANCE: f64 = 1.0e-6;
/// A round trip flips V twice, which can move a value by one rounding step.
const ROUND_TRIP_TOLERANCE: f64 = 1.0e-12;

/// The render layer (`Base` in Blender, `st` in USD) of `cube.blend`, one `[u, v]` per loop with
/// Blender's V-up values, probed in Blender 5.2.2. Loop `4 * face + corner` is corner `corner` of
/// face `face`, the order of `cube.usda`'s `faceVertexIndices`.
const BLEND_BASE_LOOPS: [[f64; 2]; 24] = [
    [0.0, 0.0],
    [0.043, 0.071],
    [0.086, 0.142],
    [0.129, 0.213],
    [0.172, 0.284],
    [0.215, 0.355],
    [0.258, 0.426],
    [0.301, 0.497],
    [0.344, 0.568],
    [0.387, 0.639],
    [0.43, 0.71],
    [0.473, 0.781],
    [0.516, 0.852],
    [0.559, 0.923],
    [0.602, 0.994],
    [0.645, 0.065],
    [0.688, 0.136],
    [0.731, 0.207],
    [0.774, 0.278],
    [0.817, 0.349],
    [0.86, 0.42],
    [0.903, 0.491],
    [0.946, 0.562],
    [0.989, 0.633],
];

/// The `Light` layer of `cube.blend`, in the same loop order and Blender's V-up values.
const BLEND_LIGHT_LOOPS: [[f64; 2]; 24] = [
    [0.17, 0.29],
    [0.213, 0.361],
    [0.256, 0.432],
    [0.299, 0.503],
    [0.342, 0.574],
    [0.385, 0.645],
    [0.428, 0.716],
    [0.471, 0.787],
    [0.514, 0.858],
    [0.557, 0.929],
    [0.6, 0.0],
    [0.643, 0.071],
    [0.686, 0.142],
    [0.729, 0.213],
    [0.772, 0.284],
    [0.815, 0.355],
    [0.858, 0.426],
    [0.901, 0.497],
    [0.944, 0.568],
    [0.987, 0.639],
    [0.03, 0.71],
    [0.073, 0.781],
    [0.116, 0.852],
    [0.159, 0.923],
];

/// One stored layer: its name and its corner values per face, in the mesh's face order.
struct StoredLayer {
    name: String,
    corners: Vec<Vec<[f64; 2]>>,
}

/// A mesh's UV store as the project holds it, with Potter's V-down values.
struct Store {
    layers: Vec<StoredLayer>,
    render: Option<String>,
    active: Option<String>,
}

fn cube_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/usd_uv_layers/cube.usda")
}

fn import(project: &Path, file: &Path, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(file)
        .args([
            "--format",
            "usda",
            "--base-revision",
            "0",
            "--mode",
            "replace",
        ]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

/// Imports `file` into `project` without `--allow-lossy`, asserting success, and returns the report.
fn import_strict(project: &Path, file: &Path) -> TestResult<Value> {
    let output = import(project, file, false)?;
    assert!(
        output.status.success(),
        "strict usda import failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// The one mesh of `project` that holds a `uv_layers` store, read from its `scene.json`.
fn uv_mesh(project: &Path) -> TestResult<Value> {
    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let meshes = scene["data_blocks"]
        .as_object()
        .ok_or("scene data blocks are not an object")?
        .values()
        .filter_map(|block| block.get("mesh"))
        .filter(|mesh| mesh["attributes"].get("uv_layers").is_some())
        .cloned()
        .collect::<Vec<_>>();
    match meshes.as_slice() {
        [mesh] => Ok(mesh.clone()),
        _ => Err(format!("expected one UV mesh, found {}", meshes.len()).into()),
    }
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    let values = value.as_array().ok_or("UV value is not an array")?;
    let (Some(u), Some(v)) = (
        values.first().and_then(Value::as_f64),
        values.get(1).and_then(Value::as_f64),
    ) else {
        return Err("UV value is not a pair of numbers".into());
    };
    Ok([u, v])
}

/// Reads the UV store of `mesh`: each layer's corner values in face order, and its selections.
fn store_of(mesh: &Value) -> TestResult<Store> {
    let store = &mesh["attributes"]["uv_layers"];
    let faces = mesh["faces"].as_array().ok_or("mesh faces are missing")?;
    let mut layers = Vec::new();
    for layer in store["layers"]
        .as_array()
        .ok_or("store layers are missing")?
    {
        let name = layer["name"]
            .as_str()
            .ok_or("layer name is missing")?
            .to_owned();
        let entries = layer["faces"].as_array().ok_or("layer faces are missing")?;
        let mut corners = Vec::new();
        for face in faces {
            let id = face["id"].as_u64().ok_or("face id is missing")?;
            let entry = entries
                .iter()
                .find(|entry| entry["face_id"].as_u64() == Some(id))
                .ok_or_else(|| format!("layer `{name}` has no values for face {id}"))?;
            let row = entry["uv"]
                .as_array()
                .ok_or("uv row is missing")?
                .iter()
                .map(pair)
                .collect::<TestResult<Vec<_>>>()?;
            corners.push(row);
        }
        layers.push(StoredLayer { name, corners });
    }
    Ok(Store {
        layers,
        render: store["render"].as_str().map(str::to_owned),
        active: store["active"].as_str().map(str::to_owned),
    })
}

fn names(layers: &[StoredLayer]) -> Vec<&str> {
    layers.iter().map(|layer| layer.name.as_str()).collect()
}

/// The corner values of `layer` in face order, one entry per corner.
fn flat(layer: &StoredLayer) -> Vec<[f64; 2]> {
    layer.corners.iter().flatten().copied().collect()
}

/// Blender's V-up values as Potter stores them (V down).
fn to_potter(values: &[[f64; 2]]) -> Vec<[f64; 2]> {
    values.iter().map(|[u, v]| [*u, 1.0 - v]).collect()
}

fn assert_values(actual: &[[f64; 2]], expected: &[[f64; 2]], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: corner count");
    for (index, (value, wanted)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (value[0] - wanted[0]).abs() <= tolerance && (value[1] - wanted[1]).abs() <= tolerance,
            "{label} corner {index}: {value:?} differs from {wanted:?}"
        );
    }
}

/// Imports a usda document strictly into a new project and returns its UV mesh.
fn import_text(usda: &str) -> TestResult<Value> {
    let directory = tempdir()?;
    let file = directory.path().join("input.usda");
    fs::write(&file, usda)?;
    let project = directory.path().join("project");
    init(&project)?;
    let report = import_strict(&project, &file)?;
    assert_eq!(report["result"]["losses"], json!([]));
    uv_mesh(&project)
}

const QUAD_HEADER: &str = r#"#usda 1.0
(
    defaultPrim = "Quad"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    uniform token subdivisionScheme = "none"
"#;

#[test]
fn cube_usda_imports_both_uv_primvars_as_layers_with_blender_values() -> TestResult {
    let directory = tempdir()?;
    let project = directory.path().join("project");
    init(&project)?;
    let report = import_strict(&project, &cube_fixture())?;
    assert_eq!(report["result"]["losses"], json!([]));
    let store = store_of(&uv_mesh(&project)?)?;
    assert_eq!(names(&store.layers), ["Light", "st"]);
    assert_eq!(store.render.as_deref(), Some("st"));
    assert_eq!(store.active.as_deref(), Some("st"));
    assert_eq!(store.layers[0].corners.len(), 6);
    assert_values(
        &flat(&store.layers[0]),
        &to_potter(&BLEND_LIGHT_LOOPS),
        TOLERANCE,
        "Light",
    );
    assert_values(
        &flat(&store.layers[1]),
        &to_potter(&BLEND_BASE_LOOPS),
        TOLERANCE,
        "st",
    );
    Ok(())
}

#[test]
fn indexed_face_varying_primvar_expands_through_its_indices() -> TestResult {
    let usda = format!(
        "{QUAD_HEADER}    texCoord2f[] primvars:uv = [(0.1, 0.2), (0.3, 0.4)] (\n        interpolation = \"faceVarying\"\n    )\n    int[] primvars:uv:indices = [0, 1, 1, 0]\n    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n}}\n"
    );
    let store = store_of(&import_text(&usda)?)?;
    assert_eq!(names(&store.layers), ["uv"]);
    assert_eq!(
        store.render.as_deref(),
        Some("uv"),
        "no st primvar: first layer renders"
    );
    assert_eq!(store.active, store.render);
    // Corner values (0.1, 0.2), (0.3, 0.4), (0.3, 0.4), (0.1, 0.2) with V flipped.
    assert_values(
        &flat(&store.layers[0]),
        &[[0.1, 0.8], [0.3, 0.6], [0.3, 0.6], [0.1, 0.8]],
        TOLERANCE,
        "uv",
    );
    Ok(())
}

#[test]
fn vertex_primvars_expand_each_corner_to_its_point_direct_and_indexed() -> TestResult {
    let usda = r#"#usda 1.0
(
    defaultPrim = "Quad"
    metersPerUnit = 1
    upAxis = "Z"
)

def Mesh "Quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    uniform token subdivisionScheme = "none"
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
    texCoord2f[] primvars:Detail = [(0.5, 0.5), (0.25, 0.75)] (
        interpolation = "vertex"
    )
    int[] primvars:Detail:indices = [0, 0, 1, 1]
}
"#;
    let store = store_of(&import_text(usda)?)?;
    assert_eq!(names(&store.layers), ["st", "Detail"]);
    assert_eq!(store.render.as_deref(), Some("st"));
    // Face 0 is points 0, 1, 2 and face 1 is points 0, 2, 3, with V flipped.
    assert_values(
        &flat(&store.layers[0]),
        &[
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        TOLERANCE,
        "st",
    );
    assert_values(
        &flat(&store.layers[1]),
        &[
            [0.5, 0.5],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.25, 0.25],
        ],
        TOLERANCE,
        "Detail",
    );
    Ok(())
}

#[test]
fn unreadable_uv_primvars_are_one_loss_each_and_the_rest_import() -> TestResult {
    let usda = format!(
        "{QUAD_HEADER}    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n    texCoord2f[] primvars:good = [(0, 0), (1, 0), (1, 1), (0, 1)] (\n        interpolation = \"vertex\"\n    )\n    texCoord3f[] primvars:uvw = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] (\n        interpolation = \"vertex\"\n    )\n    texCoord2f[] primvars:uniform_uv = [(0, 0)] (\n        interpolation = \"uniform\"\n    )\n    texCoord2f[] primvars:short = [(0, 0), (1, 1)] (\n        interpolation = \"faceVarying\"\n    )\n    texCoord2f[] primvars:badidx = [(0, 0), (1, 1)] (\n        interpolation = \"vertex\"\n    )\n    int[] primvars:badidx:indices = [0, 1, 2, 0]\n    texCoord2f[] primvars:noint = [(0, 0), (1, 0), (1, 1), (0, 1)]\n}}\n"
    );
    let directory = tempdir()?;
    let file = directory.path().join("input.usda");
    fs::write(&file, &usda)?;
    let project = directory.path().join("project");
    init(&project)?;
    let strict = import(&project, &file, false)?;
    assert!(
        !strict.status.success(),
        "unreadable UV primvars must need --allow-lossy"
    );
    let lossy = import(&project, &file, true)?;
    assert!(lossy.status.success());
    let report: Value = serde_json::from_slice(&lossy.stdout)?;
    let losses = report["result"]["losses"]
        .as_array()
        .ok_or("import report has no losses array")?
        .iter()
        .filter(|loss| loss["feature_id"] == "mesh.uv_layer")
        .collect::<Vec<_>>();
    let expected = ["uvw", "uniform_uv", "short", "badidx", "noint"];
    assert_eq!(losses.len(), expected.len(), "{losses:?}");
    for (loss, primvar) in losses.iter().zip(expected) {
        let reason = loss["reason"].as_str().unwrap_or_default();
        assert!(
            reason.contains(primvar),
            "loss does not name {primvar}: {reason}"
        );
    }
    assert_eq!(names(&store_of(&uv_mesh(&project)?)?.layers), ["good"]);
    Ok(())
}

/// Potter's own layers, with a name that is not a primvar identifier, a non-render active layer,
/// and a new layer named by Blender's suffix rule, survive export to usda and import.
#[test]
fn potter_usda_round_trip_keeps_every_layer_name_order_and_selection() -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    init(&source)?;
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op": "node.create", "id": "body", "kind": "box", "params": {"size": 2.0}},
            {"op": "uv.unwrap", "target": {"id": "body"}, "method": "box"},
            {"op": "uv.layer_create", "target": {"id": "body"}, "name": "Detail"},
            {"op": "uv.layer_rename", "target": {"id": "body"}, "name": "Detail", "to": "Light Map"},
            {"op": "uv.layer_create", "target": {"id": "body"}},
            {"op": "uv.layer_set_active", "target": {"id": "body"}, "name": "Light Map"}
        ]
    });
    let batch_file = directory.path().join("ops.json");
    fs::write(&batch_file, batch.to_string())?;
    let applied = pot()
        .arg("apply")
        .arg(&source)
        .arg("--file")
        .arg(&batch_file)
        .output()?;
    assert!(
        applied.status.success(),
        "pot apply failed: {}",
        String::from_utf8_lossy(&applied.stdout)
    );
    let before = store_of(&uv_mesh(&source)?)?;
    assert_eq!(names(&before.layers), ["UVMap", "Light Map", "UVMap.001"]);
    assert_eq!(before.render.as_deref(), Some("UVMap"));
    assert_eq!(before.active.as_deref(), Some("Light Map"));

    let usda = directory.path().join("roundtrip.usda");
    let exported = export_strict(&source, "usda", &usda)?;
    assert!(
        exported.status.success(),
        "pot export usda failed: {}",
        String::from_utf8_lossy(&exported.stdout)
    );
    let text = fs::read_to_string(&usda)?;
    assert!(text.contains("primvars:st"), "render layer is primvars:st");
    assert!(
        text.contains("primvars:Light_Map"),
        "other layers are primvars:<identifier>"
    );

    let destination = directory.path().join("destination");
    init(&destination)?;
    let report = import_strict(&destination, &usda)?;
    assert_eq!(report["result"]["losses"], json!([]));
    let after = store_of(&uv_mesh(&destination)?)?;
    assert_same_store(&after, &before);
    Ok(())
}

#[test]
fn imported_cube_round_trips_through_potter_usda_without_change() -> TestResult {
    let directory = tempdir()?;
    let first = directory.path().join("first");
    init(&first)?;
    import_strict(&first, &cube_fixture())?;
    let before = store_of(&uv_mesh(&first)?)?;

    let usda = directory.path().join("cube-roundtrip.usda");
    let exported = export_strict(&first, "usda", &usda)?;
    assert!(
        exported.status.success(),
        "pot export usda failed: {}",
        String::from_utf8_lossy(&exported.stdout)
    );
    let second = directory.path().join("second");
    init(&second)?;
    let report = import_strict(&second, &usda)?;
    assert_eq!(report["result"]["losses"], json!([]));
    let after = store_of(&uv_mesh(&second)?)?;
    assert_eq!(names(&after.layers), ["Light", "st"]);
    assert_same_store(&after, &before);
    assert_values(
        &flat(&after.layers[1]),
        &to_potter(&BLEND_BASE_LOOPS),
        TOLERANCE,
        "st",
    );
    Ok(())
}

fn assert_same_store(actual: &Store, expected: &Store) {
    assert_eq!(
        names(&actual.layers),
        names(&expected.layers),
        "layer names and order"
    );
    assert_eq!(actual.render, expected.render, "render layer");
    assert_eq!(actual.active, expected.active, "active layer");
    for (actual_layer, expected_layer) in actual.layers.iter().zip(&expected.layers) {
        assert_values(
            &flat(actual_layer),
            &flat(expected_layer),
            ROUND_TRIP_TOLERANCE,
            &actual_layer.name,
        );
    }
}
