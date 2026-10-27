use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "common/blender_file.rs"]
mod blender_file;

use blender_file::blender_executable;
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const INSPECTION_MARKER: &str = "POTTER_EXPORT_INSPECTION ";

/// Objects whose hide flags are keyed in the fixture's NLA strips. Every writer except `.blend`
/// must report their animated visibility as lost.
const ANIMATED_NODES: [&str; 4] = [
    "node_vis_render_fades_out",
    "node_vis_render_fades_in",
    "node_vis_viewport_fades_out",
    "node_vis_stored_hidden_mix",
];

/// Every frame-based exporter, with the extension of its output file.
const EXPORTS: [(&str, &str); 13] = [
    ("obj", "obj"),
    ("ply", "ply"),
    ("stl", "stl"),
    ("glb", "glb"),
    ("gltf", "gltf"),
    ("fbx", "fbx"),
    ("fbx-binary", "fbx"),
    ("usda", "usda"),
    ("usdc", "usdc"),
    ("usdz", "usdz"),
    ("alembic", "abc"),
    ("svg", "svg"),
    ("pdf", "pdf"),
];

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked_output(mut command: Command, label: &str) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
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

fn blender_522() -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping frame-based export visibility: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = checked_output(command, "checking Blender version")?;
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
        "POTTER_REQUIRE_BLENDER is set but this test requires Blender 5.2.2, found `{version}`"
    );
    eprintln!("skipping frame-based export visibility: found `{version}`, need Blender 5.2.2");
    Ok(None)
}

fn import_blend(project: &Path, blend: &Path, blender: &Path) -> TestResult {
    let mut command = pot();
    command
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(blender)
        .arg("--allow-lossy")
        .arg("--json");
    let output = checked_output(command, "importing the export visibility fixture")?;
    let imported: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(imported["ok"], true, "{imported}");
    Ok(())
}

fn export(
    project: &Path,
    format: &str,
    frame: f64,
    output: &Path,
    allow_lossy: bool,
) -> Result<Output, Box<dyn Error>> {
    let mut command = pot();
    command
        .arg("export")
        .arg(project)
        .args(["--format", format, "--frame"])
        .arg(frame.to_string())
        .arg("--out")
        .arg(output);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    command.arg("--json");
    Ok(command.output()?)
}

fn assert_visibility_losses(format: &str, frame: &str, report: &Value) -> TestResult {
    let losses = report["result"]["losses"]
        .as_array()
        .or_else(|| report["error"]["details"]["result"]["losses"].as_array())
        .ok_or_else(|| format!("{format} at frame {frame} reported no losses: {report}"))?;
    let reported = losses
        .iter()
        .filter(|loss| loss["feature_id"] == "animation.visibility")
        .filter_map(|loss| loss["data_id"].as_str())
        .collect::<Vec<_>>();
    for node in ANIMATED_NODES {
        assert!(
            reported.contains(&node),
            "{format} at frame {frame} did not report animated visibility on {node}: {report}"
        );
    }
    Ok(())
}

/// Reads the mesh objects of files with Blender 5.2.2. Each `format=path` argument yields one
/// inspection, keyed by the same argument.
fn blender_inspections(blender: &Path, files: &[(&str, PathBuf)]) -> Result<Value, Box<dyn Error>> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/export_visibility_frames/inspect_exports.py");
    let mut command = Command::new(blender);
    command
        .args(["-b", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--");
    for (format, path) in files {
        command.arg(format!("{format}={}", path.display()));
    }
    let output = checked_output(command, "reading Potter exports with Blender")?;
    let inspection = String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix(INSPECTION_MARKER).map(str::to_owned))
        .ok_or_else(|| io::Error::other("Blender printed no export inspection"))?;
    Ok(serde_json::from_str(&inspection)?)
}

/// Object names and Visibility of Potter's ASCII FBX, which Blender 5.2.2 cannot import.
fn ascii_fbx_objects(text: &str) -> Value {
    let mut names = Vec::new();
    let mut visible = serde_json::Map::new();
    let mut current: Option<String> = None;
    for line in text.lines().map(str::trim_start) {
        if let Some(model) = line.strip_prefix("Model: ") {
            current = model
                .split_once("\"Model::")
                .and_then(|(_, rest)| rest.split_once('"'))
                .map(|(name, _)| name.to_owned());
            if let Some(name) = &current {
                names.push(name.clone());
            }
        } else if line.starts_with("P: \"Visibility\"")
            && let (Some(name), Some(value)) = (&current, line.rsplit(',').next())
        {
            let flag = value.trim().parse::<f64>().unwrap_or(0.0) != 0.0;
            visible.insert(name.clone(), json!(flag));
        }
    }
    names.sort();
    json!({ "names": names, "visible": visible })
}

/// Compares what a format writes with what Blender writes at the same frame. PLY, STL, SVG, and PDF
/// name no objects, so their triangle or edge totals identify the written set.
fn assert_inclusion(format: &str, frame: &str, actual: &Value, expected: &Value) {
    let fields: &[&str] = match format {
        "ply" | "stl" => &["triangles"],
        // SVG and PDF name nothing; their line totals identify the drawn objects.
        "svg" | "pdf" => &["edges"],
        "fbx" => &["names", "visible"],
        "fbx-binary" => &["names", "triangles", "visible"],
        _ => &["names", "triangles"],
    };
    for field in fields {
        assert_eq!(
            actual[*field], expected[*field],
            "{format} at frame {frame} disagrees with Blender on `{field}`"
        );
    }
}

#[test]
fn frame_based_exports_include_the_objects_blender_writes_at_each_frame() -> TestResult {
    let Some(blender) = blender_522()? else {
        return Ok(());
    };
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/export_visibility_frames");
    let expected: Value = serde_json::from_slice(&fs::read(fixture.join("expected.json"))?)?;
    assert!(
        expected["blender_version"]
            .as_str()
            .is_some_and(|version| version.starts_with("5.2.2")),
        "oracle must be generated with Blender 5.2.2, found {}",
        expected["blender_version"]
    );

    let directory = tempdir()?;
    let project = directory.path().join("project");
    let mut init = pot();
    init.arg("init").arg(&project).arg("--json");
    checked_output(init, "initializing export visibility project")?;
    import_blend(
        &project,
        &fixture.join("export_visibility_frames.blend"),
        &blender,
    )?;

    let frames = expected["frames"]
        .as_array()
        .ok_or("expected frames are missing")?;
    for frame_value in frames {
        let frame_key = frame_value
            .as_str()
            .ok_or("expected frame is not a string")?;
        let frame: f64 = frame_key.parse()?;

        let mut outputs: Vec<(&str, PathBuf)> = Vec::new();
        for (format, extension) in EXPORTS {
            let output = directory
                .path()
                .join(format!("{frame_key}-{format}.{extension}"));
            let exported = export(&project, format, frame, &output, true)?;
            assert!(
                exported.status.success(),
                "{format} export at frame {frame_key} failed: {}",
                String::from_utf8_lossy(&exported.stdout)
            );
            let report: Value = serde_json::from_slice(&exported.stdout)?;
            assert_visibility_losses(format, frame_key, &report)?;
            outputs.push((format, output));
        }

        // Blender 5.2.2 cannot read ASCII FBX, SVG, or PDF, so those outputs are read as text.
        let blender_files: Vec<(&str, PathBuf)> = outputs
            .iter()
            .filter(|(format, _)| !matches!(*format, "fbx" | "svg" | "pdf"))
            .cloned()
            .collect();
        let inspections = blender_inspections(&blender, &blender_files)?;
        for (format, path) in &outputs {
            let actual = match *format {
                "fbx" => ascii_fbx_objects(&fs::read_to_string(path)?),
                // Potter draws each mesh edge as one stroked SVG path.
                "svg" => {
                    let text = fs::read_to_string(path)?;
                    json!({ "edges": text.matches("stroke=\"#292929\"").count() })
                }
                // Each mesh edge is one `m ... l S` stroke in the uncompressed PDF content.
                "pdf" => {
                    let text = String::from_utf8_lossy(&fs::read(path)?).into_owned();
                    json!({ "edges": text.matches(" l S").count() })
                }
                _ => {
                    let key = format!("{format}={}", path.display());
                    inspections[key.as_str()].clone()
                }
            };
            assert_inclusion(
                format,
                frame_key,
                &actual,
                &expected["formats"][frame_key][*format],
            );
        }
    }

    // Without `--allow-lossy`, the frame's visibility cannot be dropped silently either.
    let strict_output = directory.path().join("strict.obj");
    let strict = export(&project, "obj", 5.0, &strict_output, false)?;
    assert!(
        !strict.status.success(),
        "strict OBJ export accepted animated visibility"
    );
    assert!(
        !strict_output.exists(),
        "strict OBJ export created an output file"
    );
    let strict_report: Value = serde_json::from_slice(&strict.stdout)?;
    assert_eq!(strict_report["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert_visibility_losses("obj", "5", &strict_report)?;
    Ok(())
}
