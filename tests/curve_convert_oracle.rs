//! Blender 5.2.2 oracle coverage for `curve.convert` to Curves.
//!
//! Blender generates both a legacy mixed-spline source and the expected native CURVES result.
//! Potter imports the legacy `.blend`, applies the conversion through its CLI, and exports the
//! converted project; Blender then reopens that export and the same payload is checked again.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use blender_checked::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const GENERATOR: &str = "tests/fixtures/curve_convert/generate_oracle.py";
const REQUIRED_ATTRIBUTES: &[&str] = &[
    "radius",
    "tilt",
    "handle_left",
    "handle_right",
    "handle_type_left",
    "handle_type_right",
    "nurbs_weight",
    "curve_type",
    "cyclic",
    "resolution",
    "nurbs_order",
    "knots_mode",
    "normal_mode",
];
const TOLERANCE: f64 = 1.0e-5;

const BLENDER_READBACK: &str = r#"
import bpy
import json
import sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "potter_curves.blend"))
obj = bpy.data.objects.get("Mixed Legacy Curve")
assert obj is not None and obj.type == "CURVES", "Potter export did not restore the converted Curves object"
assert bpy.data.curves.get("Mixed Legacy Curve Data") is None, "export leaked its temporary legacy Curve"
data = obj.data

def attr_value(item):
    if hasattr(item, "vector"):
        return [float(component) for component in item.vector]
    value = item.value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if isinstance(value, float) else int(value)
    return value

attributes = {}
for name in (
    "radius", "tilt", "handle_left", "handle_right", "handle_type_left",
    "handle_type_right", "nurbs_weight", "curve_type", "cyclic", "resolution",
    "nurbs_order", "knots_mode", "normal_mode",
):
    attribute = data.attributes.get(name)
    assert attribute is not None, f"Potter export omitted native Curves attribute {name!r}"
    attributes[name] = {
        "domain": attribute.domain,
        "data_type": attribute.data_type,
        "values": [attr_value(item) for item in attribute.data],
    }

with (root / "potter_export_readback.json").open("w", encoding="utf-8") as output:
    json.dump({
        "object_name": obj.name,
        "data_name": data.name,
        "materials": [material.name if material else None for material in data.materials],
        "curves": [
            {
                "points": [[float(component) for component in point.position] for point in curve.points],
                "radius": float(data.points[curve.first_point_index].radius) if curve.points_length else 0.0,
            }
            for curve in data.curves
        ],
        "attributes": attributes,
    }, output, indent=2, sort_keys=True)
"#;

fn checked(command: Command, label: &str) -> TestResult<Output> {
    let output = process::run_guarded(command)?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn blender_522() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping curve.convert CURVES oracle: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = checked(command, "Blender version probe")?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2.2") {
        return Ok(Some(blender));
    }
    assert!(
        env::var_os("POTTER_REQUIRE_BLENDER").is_none_or(|value| value.is_empty()),
        "POTTER_REQUIRE_BLENDER is set but curve.convert CURVES requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping curve.convert CURVES oracle: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn run_blender_script(blender: &Path, script: &Path, root: &Path, label: &str) -> TestResult<()> {
    let root = fs::canonicalize(root)?;
    let root = root
        .to_str()
        .ok_or_else(|| io::Error::other("temporary Blender oracle path is not UTF-8"))?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script)
        .args(["--", root]);
    checked(command, label)?;
    Ok(())
}

fn run_pot(args: &[&str], label: &str) -> TestResult<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(args);
    checked(command, label)
}

fn read_json(path: &Path) -> TestResult<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn named_node<'a>(scene: &'a Value, name: &str) -> TestResult<(&'a str, &'a Value)> {
    let nodes = scene["nodes"]
        .as_object()
        .ok_or_else(|| io::Error::other("project nodes are missing"))?;
    nodes
        .iter()
        .find(|(_, node)| node["name"] == name)
        .map(|(id, node)| (id.as_str(), node))
        .ok_or_else(|| io::Error::other(format!("project node `{name}` is missing")).into())
}

fn assert_json_close(actual: &Value, expected: &Value, label: &str) {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            if let (Some(actual), Some(expected)) = (actual.as_i64(), expected.as_i64()) {
                assert_eq!(actual, expected, "{label}");
            } else {
                let actual = actual.as_f64().unwrap_or(f64::NAN);
                let expected = expected.as_f64().unwrap_or(f64::NAN);
                assert!(
                    (actual - expected).abs() <= TOLERANCE,
                    "{label}: Potter={actual:?}, Blender={expected:?}"
                );
            }
        }
        (Value::Array(actual), Value::Array(expected)) => {
            assert_eq!(actual.len(), expected.len(), "{label}: array length");
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_json_close(actual, expected, &format!("{label}[{index}]"));
            }
        }
        (Value::Object(actual), Value::Object(expected)) => {
            let actual_keys: Vec<_> = actual.keys().collect();
            let expected_keys: Vec<_> = expected.keys().collect();
            assert_eq!(actual_keys, expected_keys, "{label}: object keys");
            for key in expected.keys() {
                assert_json_close(&actual[key], &expected[key], &format!("{label}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{label}"),
    }
}

fn assert_curves_payload(actual: &Value, expected: &Value, label: &str) -> TestResult<()> {
    assert_eq!(
        actual["object_name"], expected["object_name"],
        "{label}: object name"
    );
    assert_json_close(
        &actual["curves"],
        &expected["curves"],
        &format!("{label}.curves"),
    );
    assert_eq!(
        actual["materials"], expected["materials"],
        "{label}: material slots"
    );

    let actual_attributes = actual["attributes"]
        .as_object()
        .ok_or_else(|| io::Error::other(format!("{label}: native attributes are missing")))?;
    let expected_attributes = expected["attributes"]
        .as_object()
        .ok_or_else(|| io::Error::other("Blender oracle attributes are missing"))?;
    assert_eq!(
        actual_attributes.len(),
        expected_attributes.len(),
        "{label}: native attribute count"
    );
    for name in REQUIRED_ATTRIBUTES {
        let expected = expected_attributes
            .get(*name)
            .ok_or_else(|| io::Error::other(format!("Blender oracle omits `{name}`")))?;
        let actual = actual_attributes
            .get(*name)
            .ok_or_else(|| io::Error::other(format!("{label}: Potter omitted `{name}`")))?;
        assert_json_close(actual, expected, &format!("{label}.attributes.{name}"));
    }
    Ok(())
}

fn assert_attribute_names(actual: &Value, expected: &Value, label: &str) -> TestResult<()> {
    let attributes = actual["attributes"]
        .as_object()
        .ok_or_else(|| io::Error::other(format!("{label}: native attributes are missing")))?;
    let mut actual_names: Vec<_> = attributes.keys().map(String::as_str).collect();
    actual_names.sort_unstable();
    let expected_names = expected
        .as_array()
        .ok_or_else(|| io::Error::other(format!("{label}: Blender attribute names are missing")))?
        .iter()
        .map(|name| {
            name.as_str().ok_or_else(|| {
                io::Error::other(format!("{label}: Blender attribute name is invalid"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        actual_names, expected_names,
        "{label}: native attribute names"
    );
    Ok(())
}

fn imported_hair_payload(scene: &Value, object_name: &str, label: &str) -> TestResult<Value> {
    let (_, node) = named_node(scene, object_name)?;

    assert_eq!(node["kind"], "hair_curves", "{label}: converted node kind");
    let data_id = node["data"]
        .as_str()
        .ok_or_else(|| io::Error::other(format!("{label}: converted node has no data ID")))?;
    let data = &scene["data_blocks"][data_id];
    assert_eq!(data["type"], "hair_curves", "{label}: converted data type");
    let hair = data["hair_curves"].clone();
    if !hair.is_object() {
        return Err(io::Error::other(format!("{label}: Curves payload is missing")).into());
    }
    let mut materials = Vec::new();
    for material_id in data["materials"].as_array().into_iter().flatten() {
        let material_id = material_id
            .as_str()
            .ok_or_else(|| io::Error::other(format!("{label}: material slot is not an ID")))?;
        materials.push(scene["materials"][material_id]["name"].clone());
    }
    Ok(json!({
        "object_name": node["name"],
        "materials": materials,
        "curves": hair["curves"],
        "attributes": hair["attributes"],
    }))
}

#[test]
fn legacy_splines_convert_to_native_hair_curves_through_cli() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let generator = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(GENERATOR);
    run_blender_script(&blender, &generator, root, "Blender curve.convert oracle")?;
    let oracle = read_json(&root.join("blender_expected.json"))?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(
        oracle["attributes"]["curve_type"]["values"],
        json!([2, 1, 3])
    );
    assert_eq!(
        oracle["attributes"]["cyclic"]["values"],
        json!([false, true, false])
    );
    assert_eq!(
        oracle["conversion_probes"],
        json!({
            "bevel_depth": "CURVE",
            "extrude": "CURVE",
            "bevel_object": "CURVE",
            "2d_fill": "CURVE",
            "taper": "CURVES",
            "use_path_path_duration": "CURVES",
            "eval_time": "CURVES"
        })
    );
    let nurbs_weights = oracle["attributes"]["nurbs_weight"]["values"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender oracle NURBS weights are missing"))?;
    assert!(nurbs_weights.len() >= 6);
    assert_eq!(
        &nurbs_weights[..6],
        &[
            json!(1.0),
            json!(1.0),
            json!(1.0),
            json!(1.0),
            json!(1.0),
            json!(1.0),
        ]
    );
    let oracle_curves = oracle["curves"]
        .as_array()
        .ok_or_else(|| io::Error::other("Blender oracle curves are missing"))?;
    let point_counts = oracle_curves
        .iter()
        .map(|curve| {
            curve["points"]
                .as_array()
                .map(Vec::len)
                .ok_or_else(|| io::Error::other("Blender oracle curve points are missing"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(point_counts, [3, 3, 4]);

    let project = root.join("project");
    run_pot(
        &[
            "init",
            project.to_str().ok_or("project path is not UTF-8")?,
            "--json",
        ],
        "pot init",
    )?;
    let source = root.join("source.blend");
    let source = source.to_str().ok_or("source blend path is not UTF-8")?;
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .args([
            "import",
            project_arg,
            "--file",
            source,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let imported = checked(import, "pot blend import")?;
    let imported_report: Value = serde_json::from_slice(&imported.stdout)?;
    assert_eq!(imported_report["result"]["losses"], json!([]));
    let scene = read_json(&project.join("scene.json"))?;
    let (node_id, node) = named_node(&scene, "Mixed Legacy Curve")?;
    let (poly_node_id, poly_node) = named_node(&scene, "Poly-only Legacy Curve")?;
    let (bezier_node_id, bezier_node) = named_node(&scene, "Bezier-only Legacy Curve")?;
    assert_eq!(node["kind"], "curve", "fixture starts as a legacy Curve");
    assert_eq!(
        poly_node["kind"], "curve",
        "poly fixture starts as a legacy Curve"
    );
    assert_eq!(
        bezier_node["kind"], "curve",
        "Bezier fixture starts as a legacy Curve"
    );
    let operations = json!({
        "schema_version": 1,
        "base_revision": scene["revision"],
        "operations": [
            {"op": "curve.convert", "target": {"id": node_id}, "to": "curves"},
            {"op": "curve.convert", "target": {"id": poly_node_id}, "to": "curves"},
            {"op": "curve.convert", "target": {"id": bezier_node_id}, "to": "curves"}
        ]
    });
    let operation_file = root.join("convert.json");
    fs::write(&operation_file, serde_json::to_vec(&operations)?)?;
    let operation_file = operation_file
        .to_str()
        .ok_or("operation path is not UTF-8")?;
    run_pot(
        &["apply", project_arg, "--file", operation_file, "--json"],
        "pot curve.convert to curves",
    )?;

    let scene = read_json(&project.join("scene.json"))?;
    let converted = imported_hair_payload(&scene, "Mixed Legacy Curve", "Potter project")?;
    assert_curves_payload(&converted, &oracle, "Potter project")?;
    for name in ["Poly-only Legacy Curve", "Bezier-only Legacy Curve"] {
        let payload = imported_hair_payload(&scene, name, "Potter project")?;
        assert_attribute_names(&payload, &oracle["attribute_names_by_object"][name], name)?;
    }
    let (_, converted_node) = named_node(&scene, "Mixed Legacy Curve")?;
    let data_id = converted_node["data"]
        .as_str()
        .ok_or_else(|| io::Error::other("converted Curves data ID is missing"))?;
    let inspected = run_pot(
        &["inspect", project_arg, "--id", data_id, "--json"],
        "pot inspect converted hair Curves",
    )?;
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let inspected_items = inspected["result"]["items"]
        .as_array()
        .ok_or_else(|| io::Error::other("pot inspect omitted data items"))?;
    let inspected_data = inspected_items
        .iter()
        .find(|item| item["id"] == data_id)
        .ok_or_else(|| io::Error::other("pot inspect omitted converted Curves data"))?;
    assert_eq!(inspected_data["data_type"], "hair_curves");
    assert_eq!(inspected_data["hair_curves"]["curve_count"], 3);
    assert_json_close(
        &inspected_data["hair_curves"]["curves"],
        &converted["curves"],
        "pot inspect converted curve positions",
    );
    assert_json_close(
        &inspected_data["hair_curves"]["attributes"],
        &converted["attributes"],
        "pot inspect native Curves attributes",
    );

    let exported_path = root.join("potter_curves.blend");
    let exported_path = exported_path
        .to_str()
        .ok_or("export blend path is not UTF-8")?;
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args([
            "export",
            project_arg,
            "--format",
            "blend",
            "--out",
            exported_path,
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let exported = checked(export, "pot blend export")?;
    let export_report: Value = serde_json::from_slice(&exported.stdout)?;
    assert_eq!(export_report["result"]["losses"], json!([]));

    let readback_script = root.join("readback.py");
    fs::write(&readback_script, BLENDER_READBACK)?;
    run_blender_script(
        &blender,
        &readback_script,
        root,
        "Blender Potter export readback",
    )?;
    let readback = read_json(&root.join("potter_export_readback.json"))?;
    assert_curves_payload(&readback, &oracle, "Blender re-import of Potter export")?;
    let bevel_project = root.join("bevel_object_project");
    let bevel_project_arg = bevel_project
        .to_str()
        .ok_or("bevel-object project path is not UTF-8")?;
    run_pot(
        &["init", bevel_project_arg, "--json"],
        "bevel-object pot init",
    )?;
    let bevel_source = root.join("bevel_object_source.blend");
    let bevel_source_arg = bevel_source
        .to_str()
        .ok_or("bevel-object source path is not UTF-8")?;
    let mut bevel_import = Command::new(env!("CARGO_BIN_EXE_pot"));
    bevel_import
        .args([
            "import",
            bevel_project_arg,
            "--file",
            bevel_source_arg,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--allow-lossy",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let bevel_imported = checked(bevel_import, "pot import bevel-object curve")?;
    let bevel_imported: Value = serde_json::from_slice(&bevel_imported.stdout)?;
    let import_losses = bevel_imported["result"]["losses"]
        .as_array()
        .ok_or_else(|| io::Error::other("bevel-object import losses are missing"))?;
    assert!(
        import_losses
            .iter()
            .any(|loss| loss["feature_id"] == "blender.curve.bevel_object"),
        "bevel-object import loss was not reported"
    );
    let bevel_scene = read_json(&bevel_project.join("scene.json"))?;
    let (bevel_node_id, bevel_node) = named_node(&bevel_scene, "Bevel Object Source")?;
    assert_eq!(bevel_node["kind"], "curve");
    let bevel_export = root.join("bevel_object_export.blend");
    let bevel_export_arg = bevel_export
        .to_str()
        .ok_or("bevel-object export path is not UTF-8")?;
    let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
    export
        .args([
            "export",
            bevel_project_arg,
            "--format",
            "blend",
            "--out",
            bevel_export_arg,
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let rejected_export = process::run_guarded(export)?;
    assert!(!rejected_export.status.success());
    let rejected_export: Value = serde_json::from_slice(&rejected_export.stdout)?;
    assert_eq!(rejected_export["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        rejected_export["error"]["details"]["losses"]
            .as_array()
            .is_some_and(|losses| losses
                .iter()
                .any(|loss| loss["feature_id"] == "blender.curve.bevel_object")),
        "strict export did not report the bevel-object loss"
    );
    let bevel_operations = json!({
        "schema_version": 1,
        "base_revision": bevel_scene["revision"],
        "operations": [{
            "op": "curve.convert",
            "target": {"id": bevel_node_id},
            "to": "curves"
        }]
    });
    let bevel_operation_file = root.join("bevel_object_convert.json");
    fs::write(
        &bevel_operation_file,
        serde_json::to_vec(&bevel_operations)?,
    )?;
    let mut bevel_apply = Command::new(env!("CARGO_BIN_EXE_pot"));
    bevel_apply
        .args(["apply", bevel_project_arg, "--file"])
        .arg(&bevel_operation_file)
        .args(["--json"]);
    let rejected = process::run_guarded(bevel_apply)?;
    assert!(!rejected.status.success());
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "curve.convert.production_settings"
    );
    let extrude_project = root.join("extrude_project");
    let extrude_project_arg = extrude_project
        .to_str()
        .ok_or("extrude project path is not UTF-8")?;
    run_pot(
        &["init", extrude_project_arg, "--json"],
        "extrude-source pot init",
    )?;
    let extrude_source = root.join("extrude_source.blend");
    let extrude_source_arg = extrude_source
        .to_str()
        .ok_or("extrude source path is not UTF-8")?;
    let mut extrude_import = Command::new(env!("CARGO_BIN_EXE_pot"));
    extrude_import
        .args([
            "import",
            extrude_project_arg,
            "--file",
            extrude_source_arg,
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    let extrude_import = checked(extrude_import, "pot import extruded Curve")?;
    let extrude_import: Value = serde_json::from_slice(&extrude_import.stdout)?;
    assert_eq!(extrude_import["result"]["losses"], json!([]));
    let extrude_scene = read_json(&extrude_project.join("scene.json"))?;
    let (extrude_node_id, extrude_node) = named_node(&extrude_scene, "Extrude Source")?;
    assert_eq!(extrude_node["kind"], "curve");
    let extrude_operations = json!({
        "schema_version": 1,
        "base_revision": extrude_scene["revision"],
        "operations": [{
            "op": "curve.convert",
            "target": {"id": extrude_node_id},
            "to": "curves"
        }]
    });
    let extrude_operation_file = root.join("extrude_convert.json");
    fs::write(
        &extrude_operation_file,
        serde_json::to_vec(&extrude_operations)?,
    )?;
    let mut extrude_apply = Command::new(env!("CARGO_BIN_EXE_pot"));
    extrude_apply
        .args(["apply", extrude_project_arg, "--file"])
        .arg(&extrude_operation_file)
        .args(["--json"]);
    let rejected = process::run_guarded(extrude_apply)?;
    assert!(!rejected.status.success());
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "curve.convert.production_settings"
    );
    Ok(())
}

#[test]
fn conversion_reports_losses_and_refuses_non_exact_inputs() -> TestResult {
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("custom_knots_project");
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    run_pot(&["init", project_arg, "--json"], "custom-knot pot init")?;
    let custom_operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {
                "op":"curve.create",
                "id":"custom_nurbs",
                "dimensions":"three_d",
                "splines":[{
                    "type":"nurbs",
                    "points":[
                        {"co":[0.0,0.0,0.0],"handle_left":[0.0,0.0,0.0],"handle_right":[0.0,0.0,0.0],"handle_type":"auto","weight":1.0,"radius":1.0,"tilt":0.0},
                        {"co":[1.0,0.0,0.0],"handle_left":[1.0,0.0,0.0],"handle_right":[1.0,0.0,0.0],"handle_type":"auto","weight":1.0,"radius":1.0,"tilt":0.0},
                        {"co":[2.0,0.0,0.0],"handle_left":[2.0,0.0,0.0],"handle_right":[2.0,0.0,0.0],"handle_type":"auto","weight":1.0,"radius":1.0,"tilt":0.0}
                    ],
                    "order":3,
                    "cyclic":false,
                    "resolution":4,
                    "use_endpoint":false,
                    "custom_knots":true
                }]
            },
            {"op":"curve.convert","target":{"id":"custom_nurbs"},"to":"curves"}
        ]
    });
    let custom_file = root.join("custom_knots.json");
    fs::write(&custom_file, serde_json::to_vec(&custom_operations)?)?;
    let mut apply = Command::new(env!("CARGO_BIN_EXE_pot"));
    apply
        .args(["apply", project_arg, "--file"])
        .arg(&custom_file)
        .args(["--json"]);
    let rejected = process::run_guarded(apply)?;
    assert!(!rejected.status.success());
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "curve.nurbs.knot_mode"
    );

    let project = root.join("empty_curve_project");
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    run_pot(&["init", project_arg, "--json"], "empty-curve pot init")?;
    let empty_operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"curve.create","id":"empty_curve","dimensions":"three_d","splines":[]},
            {"op":"curve.convert","target":{"id":"empty_curve"},"to":"curves"}
        ]
    });
    let empty_file = root.join("empty_curve.json");
    fs::write(&empty_file, serde_json::to_vec(&empty_operations)?)?;
    let mut apply = Command::new(env!("CARGO_BIN_EXE_pot"));
    apply
        .args(["apply", project_arg, "--file"])
        .arg(&empty_file)
        .args(["--json"]);
    let rejected = process::run_guarded(apply)?;
    assert!(!rejected.status.success());
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "curve.convert.empty"
    );

    let project = root.join("modifier_project");
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    run_pot(&["init", project_arg, "--json"], "modifier pot init")?;
    let modifier_operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {
                "op":"curve.create",
                "id":"modified_curve",
                "dimensions":"three_d",
                "splines":[{
                    "type":"poly",
                    "points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],
                    "order":2,
                    "cyclic":false,
                    "resolution":4,
                    "use_endpoint":false
                }]
            },
            {"op":"modifier.create","target":{"id":"modified_curve"},"id":"bevel","type":"bevel","params":{"width":0.1}},
            {"op":"curve.convert","target":{"id":"modified_curve"},"to":"curves"}
        ]
    });
    let modifier_file = root.join("modifier.json");
    fs::write(&modifier_file, serde_json::to_vec(&modifier_operations)?)?;
    let mut apply = Command::new(env!("CARGO_BIN_EXE_pot"));
    apply
        .args(["apply", project_arg, "--file"])
        .arg(&modifier_file)
        .args(["--json"]);
    let rejected = process::run_guarded(apply)?;
    assert!(!rejected.status.success());
    let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(rejected["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(
        rejected["error"]["details"]["feature_id"],
        "curve.convert.evaluated_modifiers"
    );

    for (case_name, settings) in [
        ("bevel_depth", json!({"bevel_depth": 0.1})),
        (
            "filled_2d",
            json!({"dimensions": "two_d", "fill_mode": "front"}),
        ),
    ] {
        let project = root.join(format!("{case_name}_project"));
        let project_arg = project
            .to_str()
            .ok_or("refusal project path is not UTF-8")?;
        run_pot(
            &["init", project_arg, "--json"],
            "production-setting refusal pot init",
        )?;
        let mut curve_create = json!({
            "op": "curve.create",
            "id": "production_curve",
            "dimensions": "three_d",
            "splines": [{
                "type": "poly",
                "points": [
                    {"co": [0.0, 0.0, 0.0]},
                    {"co": [2.0, 0.0, 0.0]},
                    {"co": [2.0, 2.0, 0.0]},
                    {"co": [0.0, 2.0, 0.0]}
                ],
                "order": 2,
                "cyclic": true,
                "resolution": 4,
                "use_endpoint": false
            }]
        });
        curve_create
            .as_object_mut()
            .ok_or_else(|| io::Error::other("curve.create operation is not an object"))?
            .extend(
                settings
                    .as_object()
                    .ok_or_else(|| io::Error::other("production settings are not an object"))?
                    .clone(),
            );
        let operations = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                curve_create,
                {"op": "curve.convert", "target": {"id": "production_curve"}, "to": "curves"}
            ]
        });
        let operation_file = root.join(format!("{case_name}.json"));
        fs::write(&operation_file, serde_json::to_vec(&operations)?)?;
        let mut apply = Command::new(env!("CARGO_BIN_EXE_pot"));
        apply
            .args(["apply", project_arg, "--file"])
            .arg(&operation_file)
            .args(["--json"]);
        let rejected = process::run_guarded(apply)?;
        assert!(
            !rejected.status.success(),
            "{case_name} conversion should be refused"
        );
        let rejected: Value = serde_json::from_slice(&rejected.stdout)?;
        assert_eq!(
            rejected["error"]["code"], "UNSUPPORTED_FEATURE",
            "{case_name} conversion error"
        );
        assert_eq!(
            rejected["error"]["details"]["feature_id"], "curve.convert.production_settings",
            "{case_name} conversion feature"
        );
    }
    let project = root.join("production_settings_project");
    let project_arg = project.to_str().ok_or("project path is not UTF-8")?;
    run_pot(&["init", project_arg, "--json"], "settings pot init")?;
    let settings_operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {
                "op":"curve.create",
                "id":"taper_profile",
                "data_id":"taper_profile_data",
                "dimensions":"two_d",
                "splines":[{
                    "type":"poly",
                    "points":[
                        {"co":[0.0,0.25,0.0]},
                        {"co":[0.5,0.6,0.0]},
                        {"co":[1.0,1.0,0.0]}
                    ],
                    "order":2,
                    "cyclic":false,
                    "resolution":4,
                    "use_endpoint":true
                }]
            },
            {
                "op":"curve.create",
                "id":"editable_curve",
                "dimensions":"three_d",
                "taper":"taper_profile_data",
                "splines":[{
                    "type":"poly",
                    "points":[{"co":[0.0,0.0,0.0]},{"co":[1.0,0.0,0.0]}],
                    "order":2,
                    "cyclic":false,
                    "resolution":4,
                    "use_endpoint":false
                }]
            },
            {"op":"curve.convert","target":{"id":"editable_curve"},"to":"curves"}
        ]
    });
    let settings_file = root.join("production_settings.json");
    fs::write(&settings_file, serde_json::to_vec(&settings_operations)?)?;
    let settings_file = settings_file
        .to_str()
        .ok_or("settings operations path is not UTF-8")?;
    run_pot(
        &["apply", project_arg, "--file", settings_file, "--json"],
        "curve conversion with legacy production settings",
    )?;
    let scene = read_json(&project.join("scene.json"))?;
    assert_eq!(
        scene["compatibility"]["losses"][0]["feature_id"],
        "curve.convert.production_settings"
    );
    Ok(())
}
