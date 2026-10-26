#![expect(
    clippy::unwrap_used,
    reason = "Blender-generated oracle fixtures use fixed valid test data"
)]

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use serde_json::{Value, json};
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/blender_script_guarded.rs"]
mod blender_script_guarded;
#[path = "common/pot_json_guarded.rs"]
mod pot_json_guarded;
#[path = "common/process.rs"]
mod process;

use blender_file::blender_executable;
use blender_script_guarded::run_blender_script;
use pot_json_guarded::pot_json;
use process::run_guarded;

const ORACLE: &str = include_str!("fixtures/dynamic_paint_wetness/oracle.json");
const OUTPUT_TOLERANCE: f64 = 1.0 / 255.0 + 1.0e-6;

fn path_text(path: &Path) -> Result<&str, Box<dyn Error>> {
    path.to_str().ok_or_else(|| "path is not UTF-8".into())
}

fn number(value: &Value, context: &str) -> Result<f64, Box<dyn Error>> {
    value
        .as_f64()
        .ok_or_else(|| format!("{context}: expected number, got {value}").into())
}
fn linear_to_srgb(value: f64) -> f64 {
    if value < 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

fn compare_colors(
    expected: &Value,
    actual: &Value,
    tolerance: f64,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let expected = expected
        .as_array()
        .ok_or_else(|| format!("{context}: expected color array is missing"))?;
    let actual = actual
        .as_array()
        .ok_or_else(|| format!("{context}: actual color array is missing"))?;
    assert_eq!(expected.len(), actual.len(), "{context}: vertex count");
    for (vertex, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let expected = expected
            .as_array()
            .ok_or_else(|| format!("{context}[{vertex}]: expected color is malformed"))?;
        let actual = actual
            .as_array()
            .ok_or_else(|| format!("{context}[{vertex}]: actual color is malformed"))?;
        assert_eq!(
            expected.len(),
            actual.len(),
            "{context}[{vertex}]: channel count"
        );
        for (channel, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let expected = number(expected, context)?;
            let actual = number(actual, context)?;
            assert!(
                (expected - actual).abs() <= tolerance,
                "{context}[{vertex}][{channel}]: expected {expected}, got {actual}"
            );
        }
    }
    Ok(())
}

fn assert_corner_layer(
    mesh: &Value,
    name: &str,
    expected: &Value,
    context: &str,
) -> Result<(), Box<dyn Error>> {
    let vertices = mesh["vertices"]
        .as_array()
        .ok_or_else(|| format!("{context}: mesh vertices are missing"))?;
    let vertex_indices: BTreeMap<u64, usize> = vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex["id"].as_u64().unwrap(), index))
        .collect();
    let attribute = &mesh["attributes"][name];
    assert_eq!(attribute["domain"], "corner", "{context}: domain");
    let values = attribute["values"]
        .as_object()
        .ok_or_else(|| format!("{context}: corner values are missing"))?;
    for face in mesh["faces"].as_array().ok_or("mesh faces are missing")? {
        let face_id = face["id"].as_u64().unwrap();
        let corners = values[&format!("f{face_id}")]
            .as_array()
            .ok_or_else(|| format!("{context}: face {face_id} values are missing"))?;
        for (corner, vertex_id) in face["v"].as_array().unwrap().iter().enumerate() {
            let vertex_id = vertex_id.as_u64().unwrap();
            let vertex_index = vertex_indices[&vertex_id];
            compare_colors(
                &Value::Array(vec![expected[vertex_index].clone()]),
                &Value::Array(vec![corners[corner].clone()]),
                OUTPUT_TOLERANCE,
                &format!("{context}, face {face_id}, corner {corner}"),
            )?;
        }
    }
    Ok(())
}

fn imported_node<'a>(scene: &'a Value, name: &str) -> Result<(&'a str, &'a Value), Box<dyn Error>> {
    scene["nodes"]
        .as_object()
        .and_then(|nodes| {
            nodes
                .iter()
                .find(|(_, node)| node["name"].as_str() == Some(name))
                .map(|(id, node)| (id.as_str(), node))
        })
        .ok_or_else(|| format!("imported object `{name}` is missing").into())
}

fn assert_surface_settings(surface: &Value, expected: &Value) {
    for key in ["type", "format", "output_name_a", "output_name_b"] {
        let model_key = match key {
            "type" => "surface_type",
            "format" => "surface_format",
            key => key,
        };
        assert_eq!(surface[model_key], expected[key], "surface.{model_key}");
    }
    for key in [
        "use_output_b",
        "use_drying",
        "use_dry_log",
        "use_dissolve",
        "use_dissolve_log",
    ] {
        assert_eq!(surface[key], expected[key], "surface.{key}");
    }
    for key in ["dry_speed", "color_dry_threshold", "dissolve_speed"] {
        assert_eq!(
            number(&surface[key], key).unwrap(),
            number(&expected[key], key).unwrap(),
            "surface.{key}"
        );
    }
}
fn assert_committed_capture_matches(generated: &Value, committed: &Value) {
    assert_eq!(generated["blender_version"], committed["blender_version"]);
    for case_key in ["paint_wetness", "dissolve_only", "volume_paint"] {
        assert_eq!(
            generated[case_key]["surface"], committed[case_key]["surface"],
            "{case_key} surface settings"
        );
        assert_eq!(
            generated[case_key]["brushes"], committed[case_key]["brushes"],
            "{case_key} brush settings"
        );
        let output_a = generated[case_key]["surface"]["output_name_a"]
            .as_str()
            .unwrap();
        let output_b = generated[case_key]["surface"]["output_name_b"]
            .as_str()
            .unwrap();
        for frame_number in 1..=5 {
            let frame_key = frame_number.to_string();
            for output in [output_a, output_b] {
                assert_eq!(
                    generated[case_key]["frames"][frame_key.as_str()]["colors"][output][4],
                    committed[case_key]["frames"][frame_key.as_str()]["colors"][output][4],
                    "{case_key} frame {frame_number} center {output}"
                );
            }
        }
    }
    assert_eq!(
        generated["flag_only"]["surface"],
        committed["flag_only"]["surface"]
    );
    for frame_number in 1..=5 {
        let frame_key = frame_number.to_string();
        for output in ["FlagOnlyColor", "FlagOnlyWetness"] {
            assert_eq!(
                generated["flag_only"]["frames"][frame_key.as_str()]["colors"][output],
                committed["flag_only"]["frames"][frame_key.as_str()]["colors"][output]
            );
        }
        assert_eq!(
            generated["no_output"]["frames"][frame_key.as_str()]["colors"],
            committed["no_output"]["frames"][frame_key.as_str()]["colors"]
        );
    }
    assert_eq!(
        generated["no_output"]["surface"],
        committed["no_output"]["surface"]
    );
    assert_eq!(
        generated["no_output"]["brushes"],
        committed["no_output"]["brushes"]
    );
    for case_key in ["rotated_cube", "torus", "open_cube"] {
        let generated_case = &generated["volume_shapes"][case_key];
        let committed_case = &committed["volume_shapes"][case_key];
        for key in ["surface", "brushes", "positions", "aabb_mask"] {
            assert_eq!(generated_case[key], committed_case[key], "{case_key} {key}");
        }
        for frame_number in 1..=5 {
            let frame_key = frame_number.to_string();
            assert_eq!(
                generated_case["frames"][frame_key.as_str()]["colors"],
                committed_case["frames"][frame_key.as_str()]["colors"],
                "{case_key} frame {frame_number} colors"
            );
        }
    }
}
fn assert_imported_brush_settings(scene: &Value, brush_spec: &Value) -> Result<(), Box<dyn Error>> {
    let brush_name = brush_spec["name"]
        .as_str()
        .ok_or("oracle brush name is missing")?;
    let (_, brush) = imported_node(scene, brush_name)?;
    let settings = &brush["properties"]["physics_dynamic_paint"];
    assert_eq!(settings["paint_source"], brush_spec["source"]);
    assert_eq!(settings["use_absolute_alpha"], brush_spec["absolute_alpha"]);
    assert_eq!(
        number(&settings["paint_alpha"], "brush alpha")?,
        number(&brush_spec["alpha"], "oracle brush alpha")?,
        "{brush_name} paint alpha"
    );
    assert_eq!(
        number(&settings["paint_wetness"], "brush wetness")?,
        number(&brush_spec["wetness"], "oracle brush wetness")?,
        "{brush_name} paint wetness"
    );
    compare_colors(
        &Value::Array(vec![brush_spec["color"].clone()]),
        &Value::Array(vec![settings["paint_color"].clone()]),
        1.0e-6,
        &format!("{brush_name} paint color"),
    )
}

#[test]
fn paint_surface_drying_dissolve_and_wetness_output_match_blender_and_round_trip()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Dynamic Paint wetness oracle; Blender is unavailable");
        return Ok(());
    };
    let temporary = tempdir()?;
    let root = temporary.path();
    run_blender_script(
        &blender,
        "generate_dynamic_paint_wetness.py",
        include_str!("fixtures/dynamic_paint_wetness/generate.py"),
        root,
        &[],
        "Blender Dynamic Paint wetness oracle failed",
        run_guarded,
    )?;
    let generated: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;
    let committed: Value = serde_json::from_str(ORACLE)?;
    assert_committed_capture_matches(&generated, &committed);
    let oracle = generated;
    let volume_brush = &oracle["volume_paint"]["brushes"][0];
    assert_eq!(volume_brush["source"], "VOLUME");
    assert_eq!(volume_brush["falloff"], "RAMP");
    assert_eq!(volume_brush["locations"]["1"], json!([-2.0, 0.0, 0.0]));
    assert_eq!(volume_brush["locations"]["2"], json!([-1.0, 0.0, 0.0]));
    let volume_frame1 = &oracle["volume_paint"]["frames"]["1"]["colors"]["VolumePaintColor"];
    let volume_alphas = volume_frame1
        .as_array()
        .ok_or("VOLUME PAINT frame 1 colors are missing")?
        .iter()
        .map(|color| number(&color[3], "VOLUME PAINT alpha"))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    assert!(
        volume_alphas.iter().any(|alpha| *alpha > 0.0) && volume_alphas.contains(&0.0),
        "VOLUME PAINT must cover only part of the canvas"
    );
    assert_ne!(
        oracle["volume_paint"]["frames"]["1"]["colors"]["VolumePaintColor"],
        oracle["volume_paint"]["frames"]["2"]["colors"]["VolumePaintColor"],
        "moving the VOLUME brush mesh must change its coverage"
    );
    for case_key in ["rotated_cube", "torus", "open_cube"] {
        let shape = &oracle["volume_shapes"][case_key];
        let output_a = shape["surface"]["output_name_a"]
            .as_str()
            .ok_or("VOLUME shape output A is missing")?;
        let colors = shape["frames"]["1"]["colors"][output_a]
            .as_array()
            .ok_or("VOLUME shape frame colors are missing")?;
        let aabb = shape["aabb_mask"]
            .as_array()
            .ok_or("VOLUME shape AABB mask is missing")?;
        assert_eq!(colors.len(), aabb.len(), "{case_key} vertex count");
        let mut aabb_false_positives = 0;
        let mut painted = 0;
        for (inside_aabb, color) in aabb.iter().zip(colors) {
            let alpha = number(&color[3], "VOLUME shape alpha")?;
            if alpha > 0.0 {
                painted += 1;
            } else if inside_aabb.as_bool() == Some(true) {
                aabb_false_positives += 1;
            }
        }
        assert!(
            aabb_false_positives > 0,
            "{case_key}: an AABB substitute would match Blender"
        );
        if case_key == "torus" {
            let positions = shape["positions"]
                .as_array()
                .ok_or("VOLUME shape positions are missing")?;
            let center = positions
                .iter()
                .position(|position| {
                    position.as_array().is_some_and(|point| {
                        point.len() == 3
                            && point
                                .iter()
                                .all(|coordinate| coordinate.as_f64() == Some(0.0))
                    })
                })
                .ok_or("torus-hole canvas vertex is missing")?;
            assert!(
                aabb[center].as_bool().unwrap_or(false),
                "the torus hole lies inside its AABB"
            );
        }
        if case_key == "open_cube" {
            assert_eq!(
                painted, 0,
                "the missing brush face makes the opposite ray miss"
            );
        } else {
            assert!(painted > 0, "{case_key} should contain painted vertices");
        }
    }

    let project = root.join("project");
    pot_json(&["init", path_text(&project)?])?;
    let source_blend = root.join("DynamicPaintWetness.blend");
    let blender_text = path_text(&blender)?;
    let source_blend_text = path_text(&source_blend)?;
    let project_text = path_text(&project)?;
    let imported = pot_json(&[
        "import",
        project_text,
        "--file",
        source_blend_text,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_text,
    ])?;
    assert_eq!(imported["result"]["losses"], json!([]));

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let case_names = [
        ("paint_wetness", "WetnessCanvas"),
        ("dissolve_only", "DissolveOnlyCanvas"),
        ("flag_only", "FlagOnlyCanvas"),
        ("volume_paint", "VolumePaintCanvas"),
        ("no_output", "NoOutputCanvas"),
    ];
    let mut canvas_ids = BTreeMap::new();
    for (case_key, canvas_name) in case_names {
        let (canvas_id, canvas) = imported_node(&scene, canvas_name)?;
        canvas_ids.insert(case_key.to_owned(), canvas_id.to_owned());
        let expected_case = &oracle[case_key];
        let surface = &canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0];
        assert_surface_settings(surface, &expected_case["surface"]);
        for brush_spec in expected_case["brushes"].as_array().into_iter().flatten() {
            assert_imported_brush_settings(&scene, brush_spec)?;
        }
    }
    let mut shape_canvas_ids = BTreeMap::new();
    for case_key in ["rotated_cube", "torus", "open_cube"] {
        let expected_case = &oracle["volume_shapes"][case_key];
        let canvas_name = expected_case["canvas"]
            .as_str()
            .ok_or("VOLUME shape canvas name is missing")?;
        let (canvas_id, canvas) = imported_node(&scene, canvas_name)?;
        shape_canvas_ids.insert(case_key.to_owned(), canvas_id.to_owned());
        let surface = &canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0];
        assert_surface_settings(surface, &expected_case["surface"]);
        for brush_spec in expected_case["brushes"].as_array().into_iter().flatten() {
            assert_imported_brush_settings(&scene, brush_spec)?;
        }
    }

    let no_output_id = canvas_ids["no_output"].as_str();
    let no_output_data = scene["nodes"][no_output_id]["data"]
        .as_str()
        .ok_or("no-output canvas mesh data is missing")?;
    let no_output_attributes = &scene["data_blocks"][no_output_data]["mesh"]["attributes"];
    assert!(no_output_attributes.get("NoOutputColor").is_none());
    assert!(
        no_output_attributes["blender_metadata"]["color_data_types"]
            .get("NoOutputColor")
            .is_none()
    );

    let bake = root.join("bake");
    let bake_result = pot_json(&[
        "bake",
        project_text,
        "--kind",
        "simulation",
        "--frames",
        "1:5",
        "--out",
        path_text(&bake)?,
    ])?;
    assert_eq!(bake_result["result"]["frame_count"], 5);
    for case_key in ["paint_wetness", "dissolve_only", "volume_paint"] {
        let expected_case = &oracle[case_key];
        let canvas_id = canvas_ids[case_key].as_str();
        let output_a = expected_case["surface"]["output_name_a"]
            .as_str()
            .ok_or("oracle primary output name is missing")?;
        let output_b = expected_case["surface"]["output_name_b"]
            .as_str()
            .ok_or("oracle wetness output name is missing")?;
        for frame_number in 1..=5 {
            let frame: Value = serde_json::from_slice(&fs::read(
                bake.join(format!("frame_{:08}.json", frame_number - 1)),
            )?)?;
            let frame_key = frame_number.to_string();
            let expected_frame = &expected_case["frames"][frame_key.as_str()];
            let primary = &expected_frame["colors"][output_a];
            let wet_layer = &expected_frame["colors"][output_b];
            compare_colors(
                primary,
                &frame["paint_colors"][canvas_id],
                OUTPUT_TOLERANCE,
                &format!("{case_key} frame {frame_number} primary paint output"),
            )?;
            let expected_wetness = wet_layer
                .as_array()
                .ok_or("oracle wetness layer is missing")?
                .iter()
                .map(|color| number(&color[0], "oracle wetness").map(linear_to_srgb))
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
            let actual_wetness = frame["paint_wetness"][canvas_id]
                .as_array()
                .ok_or("Potter wetness values are missing")?;
            assert_eq!(
                expected_wetness.len(),
                actual_wetness.len(),
                "{case_key} frame {frame_number} wetness count"
            );
            for (vertex, (expected, actual)) in
                expected_wetness.iter().zip(actual_wetness).enumerate()
            {
                let actual = number(actual, "Potter wetness")?;
                assert!(
                    (expected - actual).abs() <= OUTPUT_TOLERANCE,
                    "{case_key} frame {frame_number} wetness vertex {vertex}: expected {expected}, got {actual}"
                );
            }
            let mesh = &frame["deformed_meshes"][canvas_id];
            assert_corner_layer(
                mesh,
                output_a,
                primary,
                &format!("{case_key} frame {frame_number} {output_a}"),
            )?;
            assert_corner_layer(
                mesh,
                output_b,
                wet_layer,
                &format!("{case_key} frame {frame_number} {output_b}"),
            )?;
        }
    }
    for case_key in ["rotated_cube", "torus", "open_cube"] {
        let expected_case = &oracle["volume_shapes"][case_key];
        let canvas_id = shape_canvas_ids[case_key].as_str();
        let output_a = expected_case["surface"]["output_name_a"]
            .as_str()
            .ok_or("VOLUME shape output A is missing")?;
        for frame_number in 1..=5 {
            let frame: Value = serde_json::from_slice(&fs::read(
                bake.join(format!("frame_{:08}.json", frame_number - 1)),
            )?)?;
            let frame_key = frame_number.to_string();
            let colors = &expected_case["frames"][frame_key.as_str()]["colors"][output_a];
            compare_colors(
                colors,
                &frame["paint_colors"][canvas_id],
                OUTPUT_TOLERANCE,
                &format!("{case_key} frame {frame_number} paint output"),
            )?;
            assert_corner_layer(
                &frame["deformed_meshes"][canvas_id],
                output_a,
                colors,
                &format!("{case_key} frame {frame_number} {output_a}"),
            )?;
        }
    }
    for frame_number in 1..=5 {
        let frame: Value = serde_json::from_slice(&fs::read(
            bake.join(format!("frame_{:08}.json", frame_number - 1)),
        )?)?;
        assert!(
            frame["paint_colors"].get(no_output_id).is_none(),
            "Blender canvas without output_toggle(A) produced paint colors"
        );
        let attributes = &frame["deformed_meshes"][no_output_id]["attributes"];
        assert!(
            attributes.get("NoOutputColor").is_none(),
            "baking created output A on a Blender canvas without output_toggle(A)"
        );
    }
    let flag_only_id = canvas_ids["flag_only"].as_str();
    for frame_number in 1..=5 {
        let frame: Value = serde_json::from_slice(&fs::read(
            bake.join(format!("frame_{:08}.json", frame_number - 1)),
        )?)?;
        assert!(
            frame["paint_wetness"].get(flag_only_id).is_none(),
            "use_output_b alone produced wetness values on frame {frame_number}"
        );
        assert!(
            frame["deformed_meshes"][flag_only_id]["attributes"]
                .get("FlagOnlyWetness")
                .is_none(),
            "use_output_b alone created a wetness layer on frame {frame_number}"
        );
    }

    let main_canvas_id = canvas_ids["paint_wetness"].as_str();
    let inspected = pot_json(&[
        "inspect",
        project_text,
        "--id",
        main_canvas_id,
        "--frame",
        "2",
    ])?;
    let item = inspected["result"]["items"]
        .as_array()
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(item["evaluated_geometry"]["vertex_count"], 9);

    let exported = root.join("roundtrip.blend");
    pot_json(&[
        "export",
        project_text,
        "--format",
        "blend",
        "--out",
        path_text(&exported)?,
        "--blender",
        blender_text,
    ])?;
    let roundtrip_project = root.join("roundtrip_project");
    pot_json(&["init", path_text(&roundtrip_project)?])?;
    pot_json(&[
        "import",
        path_text(&roundtrip_project)?,
        "--file",
        path_text(&exported)?,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        blender_text,
    ])?;
    let roundtrip_scene: Value =
        serde_json::from_slice(&fs::read(roundtrip_project.join("scene.json"))?)?;
    for (case_key, canvas_name) in case_names {
        let (_, roundtrip_canvas) = imported_node(&roundtrip_scene, canvas_name)?;
        assert_surface_settings(
            &roundtrip_canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"][0],
            &oracle[case_key]["surface"],
        );
        let data_id = roundtrip_canvas["data"]
            .as_str()
            .ok_or("round-trip canvas mesh data is missing")?;
        let attributes = roundtrip_scene["data_blocks"][data_id]["mesh"]["attributes"]
            .as_object()
            .ok_or("round-trip color attributes are missing")?;
        let type_map = attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(Value::as_object);
        let output_a = oracle[case_key]["surface"]["output_name_a"]
            .as_str()
            .ok_or("round-trip paint output name is missing")?;
        let output_b = oracle[case_key]["surface"]["output_name_b"]
            .as_str()
            .ok_or("round-trip wetness output name is missing")?;
        if matches!(case_key, "flag_only" | "no_output") {
            assert!(
                !type_map.is_some_and(|types| types.contains_key(output_b)),
                "{case_key} unexpectedly has output B"
            );
        } else {
            let types = type_map.ok_or("round-trip color type metadata is missing")?;
            assert_eq!(
                types.get(output_b).and_then(Value::as_str),
                Some("BYTE_COLOR")
            );
            assert_eq!(attributes[output_b]["domain"], "corner");
        }
        if case_key == "no_output" {
            assert!(attributes.get(output_a).is_none());
            assert!(
                !type_map.is_some_and(|types| types.contains_key(output_a)),
                "round-trip Blender canvas without output_toggle(A) gained output A"
            );
        }
    }
    for case_key in ["paint_wetness", "dissolve_only", "volume_paint"] {
        for brush_spec in oracle[case_key]["brushes"].as_array().into_iter().flatten() {
            let brush_name = brush_spec["name"]
                .as_str()
                .ok_or("oracle brush name is missing")?;
            let (_, brush) = imported_node(&roundtrip_scene, brush_name)?;
            let settings = &brush["properties"]["physics_dynamic_paint"];
            assert_eq!(
                number(&settings["paint_wetness"], "round-trip brush wetness")?,
                number(&brush_spec["wetness"], "oracle brush wetness")?
            );
            assert_eq!(settings["use_absolute_alpha"], brush_spec["absolute_alpha"]);
        }
    }
    Ok(())
}
