#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test uses committed fixture data"
)]

use std::{error::Error, fs, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn successful_json(output: &std::process::Output, operation: &str) -> Value {
    assert!(
        output.status.success(),
        "{operation} failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("CLI JSON output")
}

fn inspect_geometry(project: &Path, object_name: &str) -> Value {
    let output = pot()
        .arg("inspect")
        .arg(project)
        .arg("--json")
        .output()
        .unwrap();
    let result = successful_json(&output, "pot inspect");
    result["result"]["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["name"] == object_name)
        .unwrap()["evaluated_geometry"]
        .clone()
}
fn inspect_text_data(project: &Path, object_name: &str) -> Value {
    let output = pot()
        .arg("inspect")
        .arg(project)
        .arg("--json")
        .output()
        .unwrap();
    let result = successful_json(&output, "pot inspect text data");
    let items = result["result"]["items"].as_array().unwrap();
    let node = items
        .iter()
        .find(|item| item["type"] == "node" && item["name"] == object_name)
        .unwrap();
    let data_id = node["data"].as_str().unwrap();
    let data = pot()
        .arg("inspect")
        .arg(project)
        .arg("--id")
        .arg(data_id)
        .arg("--json")
        .output()
        .unwrap();
    let result = successful_json(&data, "pot inspect text data block");
    result["result"]["items"]
        .as_array()
        .unwrap()
        .first()
        .unwrap()["text"]
        .clone()
}

fn assert_external_text_state(project: &Path) {
    let text = inspect_text_data(project, "ExternalFontOracleText");
    assert_eq!(text["fill_solver"], "cdt");
    assert_eq!(text["fill_rule"], "even_odd");
    assert_eq!(text["resolution_u"], 8);
    assert_eq!(text["character_formats"][3]["kerning"], 6.0);
}

fn inspect_positions(project: &Path, object_name: &str) -> Vec<[f64; 3]> {
    serde_json::from_value(inspect_geometry(project, object_name)["positions"].clone()).unwrap()
}

fn compare_positions(actual: &mut Vec<[f64; 3]>, expected: &mut Vec<[f64; 3]>) {
    let order = |left: &[f64; 3], right: &[f64; 3]| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
            .then_with(|| left[2].total_cmp(&right[2]))
    };
    actual.sort_by(order);
    expected.sort_by(order);
    actual.dedup();
    expected.dedup();
    assert_eq!(
        actual.len(),
        expected.len(),
        "unique evaluated mesh vertex count"
    );

    let mut matched = vec![false; expected.len()];
    for (index, actual_vertex) in actual.iter().enumerate() {
        let minimum_x = actual_vertex[0] - 1.0e-6;
        let maximum_x = actual_vertex[0] + 1.0e-6;
        let first = expected.partition_point(|vertex| vertex[0] < minimum_x);
        let end = expected.partition_point(|vertex| vertex[0] <= maximum_x);
        let mut closest = None;
        for candidate in first..end {
            if matched[candidate] {
                continue;
            }
            let expected_vertex = expected[candidate];
            let within_tolerance =
                (0..3).all(|axis| (actual_vertex[axis] - expected_vertex[axis]).abs() <= 1.0e-6);
            if within_tolerance {
                let distance = (0..3)
                    .map(|axis| (actual_vertex[axis] - expected_vertex[axis]).powi(2))
                    .sum::<f64>();
                if closest.is_none_or(|(_, best_distance)| distance < best_distance) {
                    closest = Some((candidate, distance));
                }
            }
        }
        let Some((candidate, _)) = closest else {
            panic!("vertex {index} has no Blender coordinate within 1e-6: {actual_vertex:?}");
        };
        matched[candidate] = true;
    }
    assert!(
        matched.iter().all(|matched| *matched),
        "some Blender vertices have no Potter match within 1e-6"
    );
}

fn import_blend_fixture(
    project: &Path,
    source: &Path,
    blender: &Path,
) -> Result<Value, Box<dyn Error>> {
    let initialized = pot().arg("init").arg(project).arg("--json").output()?;
    successful_json(&initialized, "pot init for Blender text fixture");
    let imported = pot()
        .arg("import")
        .arg(project)
        .arg("--file")
        .arg(source)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
        ])
        .arg("--blender")
        .arg(blender)
        .arg("--json")
        .output()?;
    Ok(successful_json(
        &imported,
        "pot import Blender text fixture",
    ))
}

fn assert_matches_blender_mesh(project: &Path, object_name: &str, oracle: &Value) {
    let geometry = inspect_geometry(project, object_name);
    let expected_vertices: Vec<[f64; 3]> =
        serde_json::from_value(oracle["mesh"]["vertices"].clone()).unwrap();
    let expected_faces = oracle["mesh"]["faces"].as_array().unwrap().len();
    assert_eq!(
        geometry["face_count"].as_u64(),
        Some(expected_faces as u64),
        "{object_name}: evaluated mesh face count"
    );
    let mut actual: Vec<[f64; 3]> = serde_json::from_value(geometry["positions"].clone()).unwrap();
    let mut expected = expected_vertices;
    compare_positions(&mut actual, &mut expected);
}

fn assert_inspection_refuses(project: &Path, feature_id: &str) {
    let output = pot()
        .arg("inspect")
        .arg(project)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "inspection must refuse {feature_id}, not approximate its text"
    );
    let response = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        response.contains(feature_id),
        "inspection refusal must report {feature_id}: {response}"
    );
}

#[test]
fn external_font_cli_import_evaluation_and_packed_round_trip_match_blender()
-> Result<(), Box<dyn Error>> {
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let directory = tempdir()?;
    let project = directory.path().join("packed_font_project");
    let initialized = pot().arg("init").arg(&project).arg("--json").output()?;
    successful_json(&initialized, "pot init");

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let source_blend = fixtures.join("text_external_font.blend");
    let imported = pot()
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(source_blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
        ])
        .arg("--blender")
        .arg(&blender)
        .arg("--json")
        .output()?;
    successful_json(&imported, "pot import packed Blender font fixture");
    assert_external_text_state(&project);

    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/text_external_font/text_external_font.oracle.json"
    ))?;
    let mut expected: Vec<[f64; 3]> = serde_json::from_value(oracle["mesh"]["vertices"].clone())?;
    let expected_face_count = oracle["mesh"]["faces"]
        .as_array()
        .ok_or("oracle faces missing")?
        .len();
    let geometry = inspect_geometry(&project, "ExternalFontOracleText");
    assert_eq!(
        geometry["face_count"].as_u64(),
        Some(expected_face_count as u64),
        "evaluated mesh face count"
    );
    let mut actual: Vec<[f64; 3]> = serde_json::from_value(geometry["positions"].clone())?;
    let mut first_expected = expected.clone();
    compare_positions(&mut actual, &mut first_expected);

    let blend = directory.path().join("font_round_trip.blend");
    let exported = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "blend", "--out"])
        .arg(&blend)
        .args(["--blender"])
        .arg(&blender)
        .arg("--pack")
        .arg("--json")
        .output()?;
    successful_json(&exported, "pot export packed external font");

    let reopened = directory.path().join("reopened_font_project");
    let initialized = pot().arg("init").arg(&reopened).arg("--json").output()?;
    successful_json(&initialized, "pot init for imported blend");
    let imported = pot()
        .arg("import")
        .arg(&reopened)
        .arg("--file")
        .arg(&blend)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
        ])
        .arg("--blender")
        .arg(&blender)
        .arg("--json")
        .output()?;
    successful_json(&imported, "pot import packed external font round-trip");
    assert_external_text_state(&reopened);
    let geometry = inspect_geometry(&reopened, "ExternalFontOracleText");
    assert_eq!(
        geometry["face_count"].as_u64(),
        Some(expected_face_count as u64),
        "round-tripped mesh face count"
    );
    let mut actual: Vec<[f64; 3]> = serde_json::from_value(geometry["positions"].clone())?;
    compare_positions(&mut actual, &mut expected);

    let linked_project = directory.path().join("relative_font_project");
    let initialized = pot()
        .arg("init")
        .arg(&linked_project)
        .arg("--json")
        .output()?;
    successful_json(&initialized, "pot init for relative font");
    let font_path = linked_project.join("assets/fonts/Lato-Regular.ttf");
    fs::create_dir_all(font_path.parent().ok_or("font asset has no parent")?)?;
    fs::copy(fixtures.join("Lato-Regular.ttf"), &font_path)?;
    let operations = directory.path().join("relative_font_operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {
                    "op": "text_object.create",
                    "id": "relative_font",
                    "name": "RelativeFontText",
                    "body": "AV",
                    "font": "//assets/fonts/Lato-Regular.ttf",
                    "resolution_u": 8
                },
                {
                    "op": "text_object.create",
                    "id": "builtin_font",
                    "name": "BuiltinText",
                    "body": "AV"
                }
            ]
        }))?,
    )?;
    let applied = pot()
        .arg("apply")
        .arg(&linked_project)
        .arg("--file")
        .arg(&operations)
        .arg("--json")
        .output()?;
    successful_json(&applied, "pot apply relative external font");
    let relative_font = inspect_positions(&linked_project, "RelativeFontText");
    let builtin_font = inspect_positions(&linked_project, "BuiltinText");
    assert!(
        relative_font.len() > 100,
        "relative external font did not produce glyph geometry"
    );
    assert_ne!(
        relative_font, builtin_font,
        "relative font must not fall back to Hershey"
    );
    Ok(())
}
#[test]
fn external_font_missing_glyph_is_explicitly_unsupported() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let project = directory.path().join("missing_glyph_project");
    let initialized = pot().arg("init").arg(&project).arg("--json").output()?;
    successful_json(&initialized, "pot init for missing external glyph");

    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let font_path = project.join("assets/fonts/Lato-Regular.ttf");
    fs::create_dir_all(font_path.parent().ok_or("font asset has no parent")?)?;
    fs::copy(fixtures.join("Lato-Regular.ttf"), &font_path)?;
    let operations = directory.path().join("missing_glyph_operations.json");
    fs::write(
        &operations,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {
                    "op": "text_object.create",
                    "id": "missing_glyph",
                    "name": "MissingGlyphText",
                    "body": "\u{10ffff}",
                    "font": "//assets/fonts/Lato-Regular.ttf"
                }
            ]
        }))?,
    )?;
    let applied = pot()
        .arg("apply")
        .arg(&project)
        .arg("--file")
        .arg(&operations)
        .arg("--json")
        .output()?;
    successful_json(&applied, "pot apply missing external glyph");

    let evaluated = pot().arg("inspect").arg(&project).arg("--json").output()?;
    assert!(
        !evaluated.status.success(),
        "missing glyph evaluation must not silently render an approximation"
    );
    let error = format!(
        "{}{}",
        String::from_utf8_lossy(&evaluated.stdout),
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert!(
        error.contains("text.font.external.missing_glyph"),
        "missing glyph must report text.font.external.missing_glyph: {error}"
    );
    Ok(())
}

#[test]
fn blender_alignment_and_control_spacing_cases_match_oracles() -> Result<(), Box<dyn Error>> {
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let directory = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/text_external_font/text_external_font_review.oracle.json"
    ))?;

    for (case_name, expected_align_y) in [
        ("default_top_baseline", "baseline"),
        ("top", "top"),
        ("control_spacing", "baseline"),
    ] {
        let case = &oracle["cases"][case_name];
        let project = directory.path().join(case_name);
        import_blend_fixture(
            &project,
            &fixtures.join(case["blend_file"].as_str().ok_or("fixture name missing")?),
            &blender,
        )?;
        let object_name = case["object_name"]
            .as_str()
            .ok_or("captured object name missing")?;
        assert_matches_blender_mesh(&project, object_name, case);
        let text = inspect_text_data(&project, object_name);
        assert_eq!(text["body"], case["text"]["body"], "{case_name}");
        assert_eq!(text["align_y"], expected_align_y, "{case_name}");
        if case_name == "control_spacing" {
            let expected_spacing = case["text"]["word_spacing"].as_f64().unwrap();
            let actual_spacing = text["word_spacing"].as_f64().unwrap();
            assert_ne!(expected_spacing, 1.0);
            assert!(
                (actual_spacing - expected_spacing).abs() <= 1.0e-6,
                "control word spacing differs from the Blender capture"
            );
            assert!(text["body"].as_str().unwrap().contains('\u{0001}'));
        }
    }
    Ok(())
}

#[test]
fn blender_typographic_metric_alignment_matches_oracles() -> Result<(), Box<dyn Error>> {
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let directory = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/text_external_font/text_external_font_review.oracle.json"
    ))?;
    let metric_case = &oracle["metric_alignments"];
    assert!(
        metric_case["metrics"]["use_typo_metrics"]
            .as_bool()
            .unwrap()
    );
    assert_ne!(
        metric_case["metrics"]["hhea"], metric_case["metrics"]["os2_typo"],
        "the Lato-derived metric fixture must distinguish OS/2 from hhea"
    );
    let project = directory.path().join("metric_alignment");
    import_blend_fixture(
        &project,
        &fixtures.join(
            metric_case["blend_file"]
                .as_str()
                .ok_or("metric fixture name missing")?,
        ),
        &blender,
    )?;
    for alignment in ["top", "center"] {
        let case = &metric_case["cases"][alignment];
        let object_name = case["object_name"]
            .as_str()
            .ok_or("metric object name missing")?;
        assert_matches_blender_mesh(&project, object_name, case);
        assert_eq!(
            inspect_text_data(&project, object_name)["align_y"],
            alignment,
            "{alignment} metric alignment"
        );
    }
    Ok(())
}

#[test]
fn unsupported_external_font_cases_are_refused_without_approximation() -> Result<(), Box<dyn Error>>
{
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let directory = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/text_external_font/text_external_font_review.oracle.json"
    ))?;
    let cases = [
        (
            &oracle["cases"]["bottom"],
            "text.font.external.align_y.bottom",
        ),
        (
            &oracle["cases"]["combining_mark"],
            "text.font.external.combining_mark",
        ),
        (&oracle["variable_font"], "text.font.external.variable"),
    ];

    for (index, (case, feature_id)) in cases.into_iter().enumerate() {
        let project = directory.path().join(format!("unsupported_{index}"));
        import_blend_fixture(
            &project,
            &fixtures.join(case["blend_file"].as_str().ok_or("fixture name missing")?),
            &blender,
        )?;
        assert_inspection_refuses(&project, feature_id);
    }
    Ok(())
}

#[test]
fn blender_font_option_losses_are_reported_on_import() -> Result<(), Box<dyn Error>> {
    let blender =
        blender_file::blender_executable().ok_or("pinned Blender executable is required")?;
    let directory = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text_external_font");
    let oracle: Value = serde_json::from_str(include_str!(
        "fixtures/text_external_font/text_external_font_review.oracle.json"
    ))?;
    let case = &oracle["import_losses"];
    let project = directory.path().join("import_losses");
    let initialized = pot().arg("init").arg(&project).arg("--json").output()?;
    successful_json(&initialized, "pot init for Blender text-loss fixture");
    let imported = pot()
        .arg("import")
        .arg(&project)
        .arg("--file")
        .arg(fixtures.join(case["blend_file"].as_str().ok_or("fixture name missing")?))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
        ])
        .arg("--blender")
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        !imported.status.success(),
        "strict import must not silently drop non-default FONT settings"
    );
    let response: Value = serde_json::from_slice(&imported.stdout)?;
    assert_eq!(response["error"]["code"], "UNREPRESENTABLE_FEATURE");
    let losses = response["error"]["details"]["result"]["losses"]
        .as_array()
        .ok_or("import loss list missing")?;
    for feature_id in [
        "blender.text.text_boxes",
        "blender.text.follow_curve",
        "blender.text.offset",
        "blender.text.taper_object",
    ] {
        assert!(
            losses.iter().any(|loss| loss["feature_id"] == feature_id),
            "pot import must report {feature_id}: {losses:?}"
        );
    }
    Ok(())
}
