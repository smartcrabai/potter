use std::{
    error::Error,
    fs,
    path::Path,
    process::{Command, Output},
};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use process::run_guarded;
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn checked_output(command: Command, label: &str) -> TestResult<Output> {
    let output = run_guarded(command)?;
    assert!(
        output.status.success(),
        "{label} failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(output)
}

fn generate_fbx(blender: &Path, root: &Path) -> TestResult {
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fbx_rotation_animation/generate.py");
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(generator)
        .arg("--")
        .arg(root);
    checked_output(command, "Blender FBX rotation fixture")?;
    Ok(())
}

fn quaternion(value: &Value, label: &str) -> TestResult<[f64; 4]> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)
        .ok_or_else(|| format!("{label} quaternion is missing or malformed"))?;
    Ok([
        components[0]
            .as_f64()
            .ok_or_else(|| format!("{label} quaternion component is not numeric"))?,
        components[1]
            .as_f64()
            .ok_or_else(|| format!("{label} quaternion component is not numeric"))?,
        components[2]
            .as_f64()
            .ok_or_else(|| format!("{label} quaternion component is not numeric"))?,
        components[3]
            .as_f64()
            .ok_or_else(|| format!("{label} quaternion component is not numeric"))?,
    ])
}

#[test]
fn blender_fbx_lcl_rotation_curves_evaluate_through_inspect() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender FBX rotation animation: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    generate_fbx(&blender, root)?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;

    let project = root.join("project");
    let mut init_command = Command::new(env!("CARGO_BIN_EXE_pot"));
    init_command.arg("init").arg(&project).arg("--json");
    checked_output(init_command, "pot init")?;
    let mut import_command = Command::new(env!("CARGO_BIN_EXE_pot"));
    import_command
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(root.join("rotation_animation.fbx"))
        .args([
            "--format",
            "fbx-binary",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ]);
    checked_output(import_command, "pot binary FBX import")?;

    let frame = expected["frame"]
        .as_u64()
        .ok_or("Blender expected frame is missing")?
        .to_string();
    let mut inspect_command = Command::new(env!("CARGO_BIN_EXE_pot"));
    inspect_command
        .args(["inspect"])
        .arg(&project)
        .args(["--frame", frame.as_str(), "--json"]);
    let inspected = checked_output(inspect_command, "pot inspect at Blender frame")?;
    let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
    let probe = inspected["result"]["items"]
        .as_array()
        .ok_or("inspection items are missing")?
        .iter()
        .find(|item| item["name"] == "RotationProbe")
        .ok_or("imported RotationProbe is missing")?;
    let actual = quaternion(
        &probe["transform"]["evaluated"]["rotation"],
        "Potter evaluated",
    )?;
    let expected = quaternion(&expected["rotation_xyzw"], "Blender expected")?;
    let mut direct_error = 0.0_f64;
    let mut negated_error = 0.0_f64;
    for (actual, expected) in actual.into_iter().zip(expected) {
        direct_error = direct_error.max((actual - expected).abs());
        negated_error = negated_error.max((actual + expected).abs());
    }
    let error = direct_error.min(negated_error);
    assert!(
        error <= 1.0e-5,
        "Potter FBX rotation differs from Blender at frame {frame}: max quaternion error {error}, Potter={actual:?}, Blender={expected:?}"
    );
    Ok(())
}

fn inspected_rotation(project: &Path, frame: u32) -> TestResult<[f64; 4]> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .args(["inspect"])
        .arg(project)
        .args(["--frame"])
        .arg(frame.to_string())
        .arg("--json");
    let output = checked_output(command, "pot inspect FBX Euler rotation")?;
    let inspected: Value = serde_json::from_slice(&output.stdout)?;
    let probe = inspected["result"]["items"]
        .as_array()
        .and_then(|items| items.iter().find(|item| item["name"] == "MultiAxisProbe"))
        .ok_or("MultiAxisProbe is missing from inspection")?;
    quaternion(
        &probe["transform"]["evaluated"]["rotation"],
        "Potter evaluated",
    )
}

#[test]
fn potter_fbx_rotation_curve_roundtrip_preserves_node_euler_mode() -> TestResult {
    let directory = tempdir()?;
    let root = directory.path();
    let project = root.join("project");
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.arg("init").arg(&project).arg("--json");
    checked_output(command, "pot init FBX Euler round trip")?;
    let operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"action.create","id":"turn","name":"Turn"},
            {"op":"action.update","target":{"id":"turn"},"set":{"fcurves":[
                {"path":"transform.rotation_euler","index":0,"keyframes":[
                    {"frame":1.0,"value":0.35,"interpolation":"linear"},
                    {"frame":3.0,"value":0.65,"interpolation":"linear"}
                ]},
                {"path":"transform.rotation_euler","index":1,"keyframes":[
                    {"frame":1.0,"value":-0.25,"interpolation":"linear"},
                    {"frame":3.0,"value":-0.8,"interpolation":"linear"}
                ]}
            ]}},
            {"op":"node.create","id":"probe","name":"MultiAxisProbe","kind":"empty",
                "transform":{"rotation_deg":[15.0,-20.0,35.0],"rotation_mode":"XZY"}},
            {"op":"node.update","target":{"id":"probe"},"set":{"action":"turn"}}
        ]
    });
    let operations_path = root.join("operations.json");
    fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&operations_path)
        .arg("--json");
    checked_output(command, "pot apply FBX Euler round trip")?;
    let expected = [
        inspected_rotation(&project, 1)?,
        inspected_rotation(&project, 3)?,
    ];

    let fbx = root.join("multiaxis.fbx");
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .arg("export")
        .arg(&project)
        .args(["--format", "fbx", "--out"])
        .arg(&fbx)
        .arg("--json");
    checked_output(command, "pot export FBX Euler round trip")?;
    let exported = fs::read_to_string(&fbx)?;
    assert!(exported.contains("P: \"RotationOrder\", \"enum\", \"\", \"\",1"));
    assert!(exported.contains("P: \"RotationActive\", \"bool\", \"\", \"\",1"));
    let imported_project = root.join("imported");
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.arg("init").arg(&imported_project).arg("--json");
    checked_output(command, "pot init FBX Euler import")?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command
        .arg("import")
        .arg(&imported_project)
        .arg("--file")
        .arg(&fbx)
        .args([
            "--format",
            "fbx",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ]);
    checked_output(command, "pot import FBX Euler round trip")?;
    for (frame, expected) in [1, 3].into_iter().zip(expected) {
        let actual = inspected_rotation(&imported_project, frame)?;
        let direct_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        let negated_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual + expected).abs())
            .fold(0.0_f64, f64::max);
        let error = direct_error.min(negated_error);
        assert!(
            error <= 1.0e-5,
            "FBX round trip changed the multi-axis rotation at frame {frame}: max quaternion error {error}, original={expected:?}, imported={actual:?}"
        );
    }
    Ok(())
}

fn generate_rotation_order_fbx(blender: &Path, root: &Path) -> TestResult {
    let generator = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fbx_rotation_animation/generate_rotation_order_oracle.py");
    let root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(generator)
        .arg("--")
        .arg(root);
    checked_output(command, "Blender FBX rotation-order oracle")?;
    Ok(())
}

#[test]
fn blender_fbx_rotation_orders_match_imported_static_and_animated_transforms() -> TestResult {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender FBX rotation-order parity: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    generate_rotation_order_fbx(&blender, root)?;
    let expected: Value =
        serde_json::from_slice(&fs::read(root.join("rotation_orders_expected.json"))?)?;

    let frame = expected["frame"]
        .as_u64()
        .ok_or("Blender FBX rotation-order frame is missing")?
        .to_string();
    for order in ["XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX"] {
        for kind in ["Static", "Animated"] {
            let name = format!("{kind}_{order}");
            let project = root.join(format!("project_{name}"));
            let mut init_command = Command::new(env!("CARGO_BIN_EXE_pot"));
            init_command.arg("init").arg(&project).arg("--json");
            checked_output(init_command, "pot init FBX rotation order")?;

            let mut import_command = Command::new(env!("CARGO_BIN_EXE_pot"));
            import_command
                .arg("import")
                .arg(&project)
                .arg("--file")
                .arg(root.join(format!("{name}.fbx")))
                .args([
                    "--format",
                    "fbx-binary",
                    "--base-revision",
                    "0",
                    "--mode",
                    "replace",
                    "--json",
                ]);
            let imported = checked_output(import_command, "pot binary FBX rotation-order import")?;
            let imported: Value = serde_json::from_slice(&imported.stdout)?;
            assert_eq!(imported["result"]["losses"], json!([]));

            let sample_frame = frame.as_str();
            let mut inspect_command = Command::new(env!("CARGO_BIN_EXE_pot"));
            inspect_command.args(["inspect"]).arg(&project).args([
                "--frame",
                sample_frame,
                "--json",
            ]);
            let inspected = checked_output(inspect_command, "pot inspect FBX rotation order")?;
            let inspected: Value = serde_json::from_slice(&inspected.stdout)?;
            let item = inspected["result"]["items"]
                .as_array()
                .and_then(|items| items.iter().find(|item| item["name"] == name))
                .ok_or_else(|| format!("imported {name} is missing"))?;
            let actual = quaternion(
                &item["transform"]["evaluated"]["rotation"],
                "Potter evaluated",
            )?;
            let expected_rotation = quaternion(
                &expected["world_rotations"][&name][sample_frame],
                "Blender evaluated",
            )?;
            let direct_error = actual
                .iter()
                .zip(expected_rotation)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            let negated_error = actual
                .iter()
                .zip(expected_rotation)
                .map(|(actual, expected)| (actual + expected).abs())
                .fold(0.0_f64, f64::max);
            let error = direct_error.min(negated_error);
            assert!(
                error <= 1.0e-5,
                "{name} differs from Blender FBX import at frame {sample_frame}: max quaternion error {error}, Potter={actual:?}, Blender={expected_rotation:?}"
            );
        }
    }

    Ok(())
}
