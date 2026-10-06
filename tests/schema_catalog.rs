#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test fixtures use fixed valid inputs"
)]

use std::{error::Error, fs, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run(args: &[&str]) -> Output {
    pot().args(args).output().unwrap()
}

fn value(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn init(scene: &Path) -> Value {
    let output = pot().arg("init").arg(scene).arg("--json").output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    value(&output)
}

fn apply(scene: &Path, operations: &Value) -> Value {
    let directory = tempdir().unwrap();
    let file = directory.path().join("operations.json");
    fs::write(&file, serde_json::to_vec(operations).unwrap()).unwrap();
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    value(&output)
}

fn apply_first_example(scene: &Path) {
    init(scene);
    let directory = tempdir().unwrap();
    let file = directory.path().join("first.json");
    fs::write(&file, include_bytes!("fixtures/first.json")).unwrap();
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

#[test]
fn schema_kinds_are_meta_schemas_and_operation_batches_validate() -> Result<(), Box<dyn Error>> {
    for kind in [
        "scene",
        "operations",
        "preview",
        "response",
        "capabilities",
        "formats",
    ] {
        let output = run(&["schema", "--kind", kind, "--json"]);
        assert!(output.status.success());
        let result = value(&output)["result"].clone();
        assert_eq!(result["kind"], kind);
        assert!(result["op"].is_null());
        let schema = result["schema"].clone();
        let validator = jsonschema::validator_for(&schema)?;
        if matches!(kind, "capabilities" | "formats") {
            let catalog = result["catalog"].clone();
            assert!(catalog.is_object());
            assert!(
                validator.is_valid(&catalog),
                "{kind} catalog should validate against its schema"
            );
            let (catalog_field, definition) = if kind == "capabilities" {
                ("features", "feature")
            } else {
                ("formats", "format")
            };
            let entry_schema = json!({
                "$schema": schema["$schema"],
                "$ref": format!("#/$defs/{definition}"),
                "$defs": schema["$defs"]
            });
            let entry_validator = jsonschema::validator_for(&entry_schema)?;
            for entry in catalog[catalog_field].as_array().unwrap() {
                assert!(
                    entry_validator.is_valid(entry),
                    "{kind} catalog entry should validate: {entry}"
                );
            }
            if kind == "formats" {
                let formats = catalog["formats"].as_array().unwrap();
                for name in [
                    "usda",
                    "usdc",
                    "usd",
                    "usdz",
                    "alembic",
                    "fbx",
                    "fbx-binary",
                    "bvh",
                ] {
                    let format = formats
                        .iter()
                        .find(|format| format["format"] == name)
                        .unwrap();
                    assert_eq!(format["status"], "supported");
                    assert!(format["direction"]["import"].as_bool().unwrap());
                    assert!(format["direction"]["export"].as_bool().unwrap());
                }
            }
        } else {
            assert!(result.get("catalog").is_none());
        }
    }
    let schema = potter::schema::schema("operations", None)?;
    let validator = jsonschema::validator_for(&schema)?;
    for batch in [
        serde_json::from_str::<Value>(include_str!("fixtures/first.json"))?,
        serde_json::from_str::<Value>(include_str!("fixtures/second.json"))?,
    ] {
        assert!(
            validator.is_valid(&batch),
            "example operation batch should validate: {batch}"
        );
    }
    for name in potter::ops::OP_NAMES {
        let output = run(&["schema", "--kind", "operations", "--op", name, "--json"]);
        let schema = value(&output)["result"]["schema"].clone();
        let _validator = jsonschema::validator_for(&schema)?;
        assert_eq!(schema["properties"]["op"]["const"], *name);
    }
    let parent_output = run(&[
        "schema",
        "--kind",
        "operations",
        "--op",
        "collection.parent",
        "--json",
    ]);
    let parent_schema = value(&parent_output)["result"]["schema"].clone();
    let parent_validator = jsonschema::validator_for(&parent_schema)?;
    assert!(parent_validator.is_valid(&json!({
        "op":"collection.parent",
        "target":{"id":"child"},
        "parent":null
    })));
    assert!(!parent_validator.is_valid(&json!({
        "op":"collection.parent",
        "target":{"id":"child"}
    })));
    let update_output = run(&[
        "schema",
        "--kind",
        "operations",
        "--op",
        "collection.update",
        "--json",
    ]);
    let update_schema = value(&update_output)["result"]["schema"].clone();
    let update_validator = jsonschema::validator_for(&update_schema)?;
    assert!(update_validator.is_valid(&json!({
        "op":"collection.update",
        "target":{"id":"child"},
        "set":{"exclude":true},
        "scene_id":"scene_main",
        "view_layer":"view_main"
    })));
    let features = potter::catalog::feature_catalog();
    for feature in features["features"].as_array().unwrap() {
        for operation in feature["pot_ops"].as_array().unwrap() {
            let operation = operation.as_str().unwrap();
            assert!(
                potter::ops::OP_NAMES.contains(&operation),
                "catalog operation must be registered: {operation}"
            );
        }
    }
    let doc = potter::model::SceneDoc::new("dispatch-test".to_owned());
    for operation in potter::ops::OP_NAMES {
        let batch = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{"op": operation}]
        });
        let error = potter::ops::apply_batch(&doc, &batch).unwrap_err();
        assert_ne!(
            error.message,
            format!("unsupported operation `{operation}`"),
            "OP_NAMES entry must have a dispatcher route: {operation}"
        );
    }
    Ok(())
}

#[test]
fn collection_parent_cycles_and_view_layer_exclusions_are_cli_visible() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("collection_scene");
    init(&scene);
    apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"collection.create","id":"parent_a"},
                {"op":"collection.create","id":"child"},
                {"op":"collection.parent","target":{"id":"child"},"parent":"parent_a"},
                {"op":"scene.create","id":"scene_alt","root_collection":"collection_root","view_layer":"view_alt"},
                {"op":"collection.update","target":{"id":"child"},"set":{"exclude":true},"scene_id":"scene_main","view_layer":"view_main"},
                {"op":"collection.update","target":{"id":"child"},"set":{"exclude":false},"scene_id":"scene_alt","view_layer":"view_alt"}
            ]
        }),
    );
    let catalog = potter::catalog::feature_catalog();
    assert_eq!(
        catalog["features"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["feature_id"] == "collection.graph")
            .unwrap()["status"],
        "supported"
    );
    let saved: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        saved["collections"]["collection_root"]["children"],
        json!(["parent_a"])
    );
    assert_eq!(
        saved["collections"]["parent_a"]["children"],
        json!(["child"])
    );
    assert_eq!(
        saved["scenes"]["scene_main"]["view_layers"]["view_main"]["excluded_collections"],
        json!(["child"])
    );
    assert_eq!(
        saved["scenes"]["scene_alt"]["view_layers"]["view_alt"]["excluded_collections"],
        json!([])
    );

    let cycle_directory = tempdir().unwrap();
    let cycle_file = cycle_directory.path().join("cycle.json");
    fs::write(
        &cycle_file,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 1,
            "operations": [
                {"op":"collection.parent","target":{"id":"parent_a"},"parent":"child"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let cycle = pot()
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&cycle_file)
        .arg("--json")
        .output()
        .unwrap();
    assert!(!cycle.status.success());
    assert_eq!(value(&cycle)["error"]["code"], "INVALID_OPERATION");
    let unchanged: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(unchanged["revision"], 1);
}

#[test]
fn tracking_masks_and_sequencer_operations_run_through_cli() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("media_scene");
    init(&scene);
    let result = apply(
        &scene,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"tracking.clip_create","id":"clip","name":"Clip"},
                {"op":"tracking.track_add","id":"clip","track":"track_main","name":"Track","frame":1,"co":[0.25,0.5]},
                {"op":"mask.create","id":"mask_main","splines":[]},
                {"op":"sequencer.update","set":{"channels":40}},
                {"op":"sequencer.strip_create","id":"color_strip","type":"color"}
            ]
        }),
    );
    assert!(result["result"]["changed"].as_bool().unwrap());
    let saved: Value =
        serde_json::from_slice(&fs::read(scene.join("scene.json")).unwrap()).unwrap();
    assert_eq!(
        saved["movie_clips"]["clip"]["tracking"]["tracks"][0]["id"],
        "track_main"
    );
    assert_eq!(saved["masks"]["mask_main"]["splines"], json!([]));
    assert_eq!(saved["scenes"]["scene_main"]["sequencer"]["channels"], 40);
    assert_eq!(
        saved["scenes"]["scene_main"]["sequencer"]["strips"][0]["id"],
        "color_strip"
    );

    let catalog = potter::catalog::feature_catalog();
    let rows = catalog["features"].as_array().unwrap();
    for supported in ["tracking", "tracking.object_solve", "mask", "sequencer"] {
        assert_eq!(
            rows.iter()
                .find(|row| row["feature_id"] == supported)
                .unwrap()["status"],
            "supported"
        );
    }
    for unsupported in [
        "mask.bezier_spline",
        "sequencer.strip.movie",
        "sequencer.strip.text.unicode",
    ] {
        assert_eq!(
            rows.iter()
                .find(|row| row["feature_id"] == unsupported)
                .unwrap()["status"],
            "not_supported"
        );
    }
}

#[test]
fn every_export_catalog_format_handles_the_spec_box_or_reports_loss() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("spec_box_scene");
    apply_first_example(&scene);
    let schema = run(&["schema", "--kind", "formats", "--json"]);
    assert!(schema.status.success());
    let formats = value(&schema)["result"]["catalog"]["formats"]
        .as_array()
        .unwrap()
        .clone();
    for format_row in formats
        .iter()
        .filter(|row| row["direction"]["export"].as_bool() == Some(true))
    {
        let format = format_row["format"].as_str().unwrap();
        let extension = match format {
            "alembic" => "abc",
            "fbx" | "fbx-binary" => "fbx",
            other => other,
        };
        let output_path = directory
            .path()
            .join(format!("spec_box_{format}.{extension}"));
        let output = pot()
            .arg("export")
            .arg(&scene)
            .arg("--format")
            .arg(format)
            .arg("--out")
            .arg(&output_path)
            .arg("--json")
            .output()
            .unwrap();
        let response = value(&output);
        if output.status.success() {
            assert_eq!(response["result"]["format"], format);
            assert!(
                output_path.is_file(),
                "{format} export did not create its output"
            );
        } else {
            let details = &response["error"]["details"];
            let losses = details["result"]["losses"]
                .as_array()
                .or_else(|| details["losses"].as_array())
                .unwrap();
            assert!(!losses.is_empty(), "{format} reported no losses");
            assert!(
                format_row["loss_conditions"]
                    .as_array()
                    .is_some_and(|conditions| !conditions.is_empty()),
                "{format} rejected the spec box without cataloged loss conditions"
            );
            for loss in losses {
                assert!(loss["feature_id"].is_string());
                assert!(loss["reason"].is_string());
            }
        }
    }
    for format_row in &formats {
        let format = format_row["format"].as_str().unwrap();
        let extension = match format {
            "alembic" => "abc",
            "fbx" | "fbx-binary" => "fbx",
            other => other,
        };
        let input_path = directory
            .path()
            .join(format!("missing_{format}.{extension}"));
        let output = pot()
            .arg("import")
            .arg(&scene)
            .arg("--file")
            .arg(&input_path)
            .arg("--format")
            .arg(format)
            .arg("--base-revision")
            .arg("1")
            .arg("--json")
            .output()
            .unwrap();
        assert_eq!(format_row["status"], "supported");
        assert!(format_row["reason"].is_null());
        assert!(!output.status.success());
        let expected_error = if format_row["direction"]["import"].as_bool().unwrap() {
            "FILE_NOT_FOUND"
        } else {
            "UNSUPPORTED_FEATURE"
        };
        assert_eq!(
            value(&output)["error"]["code"],
            expected_error,
            "{format} import direction disagrees with the CLI adapter"
        );
        assert_eq!(
            format_row["requires_blender"].as_bool().unwrap(),
            format == "blend"
        );
        assert_eq!(
            !format_row["required_runtime"]
                .as_array()
                .unwrap()
                .is_empty(),
            format == "blend"
        );
    }
}
#[test]
fn command_help_documents_each_cli_surface_and_version() {
    let version = pot().arg("--version").output().unwrap();
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8_lossy(&version.stdout).trim(),
        format!("pot {}", env!("CARGO_PKG_VERSION"))
    );
    let root_short_help = pot().arg("-h").output().unwrap();
    assert!(root_short_help.status.success());
    let root_short_help = String::from_utf8_lossy(&root_short_help.stdout);
    assert!(root_short_help.contains("Headless, JSON-driven scene editor"));
    assert!(root_short_help.contains("Print one JSON response envelope."));
    let root_long_help = pot().arg("--help").output().unwrap();
    assert!(root_long_help.status.success());
    assert!(
        String::from_utf8_lossy(&root_long_help.stdout)
            .contains("Create, inspect, edit, validate, render, and exchange Blender-compatible scenes through typed JSON.")
    );
    let commands: &[(&str, &str, &[&str])] = &[
        (
            "init",
            "Create a new project scene.",
            &["Project directory."],
        ),
        (
            "history",
            "List committed project history.",
            &["Project directory."],
        ),
        (
            "undo",
            "Restore an earlier committed project state.",
            &[
                "Project directory.",
                "Expected current project revision.",
                "Number of history steps to traverse.",
            ],
        ),
        (
            "redo",
            "Reapply a later committed project state.",
            &[
                "Project directory.",
                "Expected current project revision.",
                "Number of history steps to traverse.",
            ],
        ),
        (
            "apply",
            "Apply a typed operation batch atomically.",
            &[
                "Project directory.",
                "Operation batch JSON file, or - for stdin.",
                "Render preview views after applying the batch.",
                "Square preview image size in pixels.",
                "Validate the batch without committing changes.",
            ],
        ),
        (
            "import",
            "Import a supported scene interchange file.",
            &[
                "Project directory.",
                "Interchange file to import.",
                "Input format.",
                "Expected current project revision.",
                "Append to or replace the project.",
                "Copy imported assets or keep external links.",
                "Accept reported data loss during import.",
                "Validate the import without committing changes.",
                "Blender executable used only for .blend exchange.",
            ],
        ),
        (
            "inspect",
            "Inspect scene structure, state, and features.",
            &[
                "Project directory.",
                "Inspect one entity by ID.",
                "Inspect entities matching a tag.",
                "Include the feature capability catalog.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "preview",
            "Render selected scene views to preview images.",
            &[
                "Project directory.",
                "Comma-separated preview views.",
                "Camera ID for the preview.",
                "Preview shading and output mode.",
                "Square preview image size in pixels.",
                "Output file or directory for preview images.",
                "Replace existing preview outputs.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "pick",
            "Pick an object or element from a preview.",
            &[
                "Project directory.",
                "Preview manifest produced by pot preview.",
                "Pixel coordinate as x,y.",
                "Entity or element domain to pick.",
            ],
        ),
        (
            "validate",
            "Validate a scene and optional export format.",
            &[
                "Project directory.",
                "Fail when validation reports any issue.",
                "Check loss conditions for an export format.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "export",
            "Export a scene to a supported interchange format.",
            &[
                "Project directory.",
                "Output interchange format.",
                "Output file path.",
                "Allow export when loss conditions are reported.",
                "Pack supported external assets into the output.",
                "Replace existing output files.",
                "Blender executable used only for .blend exchange.",
                "Orthographic view for SVG or PDF export.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "render",
            "Render frames to an image sequence or movie.",
            &[
                "Project directory.",
                "Camera ID used for rendering.",
                "Rendering engine.",
                "Rendering device.",
                "Frame range as start:end[:step].",
                "Output image or movie format.",
                "Output directory.",
                "Replace existing output files.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "bake",
            "Bake simulation, texture, geometry, or animation data.",
            &[
                "Project directory.",
                "Data kind to bake.",
                "Target entity ID, when required by the bake kind.",
                "Frame range as start:end[:step].",
                "Output directory.",
                "Replace existing output files.",
                "Scene ID to evaluate; defaults to the active scene.",
                "View Layer ID to evaluate.",
                "Finite evaluation frame, including subframes.",
            ],
        ),
        (
            "assets",
            "List and verify project assets.",
            &[
                "Project directory.",
                "Rehash and verify referenced project assets.",
            ],
        ),
        (
            "schema",
            "Print a JSON Schema and optional typed catalog.",
            &[
                "Schema or typed catalog kind.",
                "Operation name for an individual operation schema.",
            ],
        ),
    ];
    for (command, about, descriptions) in commands {
        let output = pot().arg(command).arg("--help").output().unwrap();
        assert!(output.status.success(), "{command} --help failed");
        let help = String::from_utf8_lossy(&output.stdout);
        assert!(help.contains(about), "{command} help omitted its summary");
        for description in *descriptions {
            assert!(
                help.contains(description),
                "{command} help omitted: {description}"
            );
        }
    }
}
#[test]
fn inspect_features_returns_the_catalog_only_when_requested() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("inspect_features_scene");
    init(&scene);
    let with_features = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--features")
        .arg("--json")
        .output()
        .unwrap();
    assert!(with_features.status.success());
    let features = value(&with_features)["result"]["features"]
        .as_array()
        .unwrap()
        .clone();
    assert!(
        features
            .iter()
            .any(|feature| feature["feature_id"] == "scene.graph")
    );
    let without_features = pot()
        .arg("inspect")
        .arg(&scene)
        .arg("--json")
        .output()
        .unwrap();
    assert!(without_features.status.success());
    assert_eq!(value(&without_features)["result"]["features"], json!([]));
}

#[test]
fn validate_format_adds_a_loss_check() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("format_validation_scene");
    init(&scene);
    let output = pot()
        .arg("validate")
        .arg(&scene)
        .arg("--format")
        .arg("obj")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let report = value(&output)["result"].clone();
    assert!(report["valid"].as_bool().unwrap());
    assert!(
        report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|check| { check["check"] == "format_loss" && check["status"] == "supported" })
    );
}

#[test]
fn validation_reports_zero_area_and_strict_mode_fails_with_the_report() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("zero_area_scene");
    apply_first_example(&scene);
    let scene_file = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    let mesh = &mut document["data_blocks"]["body_mesh"]["mesh"];
    let face_vertex_ids = mesh["faces"][0]["v"].as_array().unwrap().clone();
    assert!(face_vertex_ids.len() >= 3);
    for (index, vertex_id) in face_vertex_ids.iter().enumerate() {
        let vertex = mesh["vertices"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|vertex| vertex["id"] == *vertex_id)
            .unwrap();
        vertex["co"] = json!([index as f64, 0.0, 0.0]);
    }
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let normal = pot()
        .arg("validate")
        .arg(&scene)
        .arg("--json")
        .output()
        .unwrap();
    assert!(normal.status.success());
    let report = value(&normal)["result"].clone();
    assert!(report["valid"].as_bool().unwrap());
    assert!(
        report["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| { issue["code"] == "ZERO_AREA_FACE" && issue["severity"] == "warning" })
    );

    let strict = pot()
        .arg("validate")
        .arg(&scene)
        .arg("--strict")
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(strict.status.code(), Some(4));
    let envelope = value(&strict);
    assert_eq!(envelope["error"]["code"], "VALIDATION_FAILED");
    assert!(
        envelope["result"]["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "ZERO_AREA_FACE")
    );
}

#[test]
fn validation_detects_parent_cycle_in_scene_json() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("parent_cycle_scene");
    init(&scene);
    apply(
        &scene,
        &json!({"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"left","kind":"group"},
            {"op":"node.create","id":"right","kind":"group"}
        ]}),
    );
    let scene_file = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    document["nodes"]["left"]["parent"] = json!("right");
    document["nodes"]["right"]["parent"] = json!("left");
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let output = pot()
        .arg("validate")
        .arg(&scene)
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let result = value(&output)["result"].clone();
    assert!(!result["valid"].as_bool().unwrap());
    assert!(
        result["issues"]
            .as_array()
            .unwrap()
            .iter()
            .any(|issue| issue["code"] == "PARENT_CYCLE")
    );
}

#[test]
fn assets_check_rehashes_external_resources() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("asset_scene");
    init(&scene);
    let asset = directory.path().join("texture.bin");
    fs::write(&asset, b"changed bytes").unwrap();
    let expected_hash = potter::hash::sha256(b"original bytes");
    let asset_uri = asset.to_string_lossy().into_owned();
    let scene_file = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    document["resources"]["texture_main"] = json!({
        "uri":asset_uri,"hash":expected_hash,"kind":"image","owner":"body","packed":false
    });
    fs::write(&scene_file, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let output = pot()
        .arg("assets")
        .arg(&scene)
        .arg("--check")
        .arg("--json")
        .output()
        .unwrap();
    assert!(output.status.success());
    let result = value(&output)["result"].clone();
    let resource = result["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["uri"] == asset.to_string_lossy().as_ref())
        .unwrap();
    assert_eq!(resource["kind"], "image");
    assert_eq!(resource["owner"], "body");
    assert_eq!(resource["packed"], false);
    assert_eq!(resource["missing"], false);
    assert_eq!(resource["changed"], true);
}

#[test]
fn bake_animation_and_geometry_write_outputs_without_changing_scene() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("bake_scene");
    apply_first_example(&scene);
    let scene_file = scene.join("scene.json");
    let before: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    let animation = directory.path().join("animation");
    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("animation")
        .arg("--frames")
        .arg("1:2")
        .arg("--out")
        .arg(&animation)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let result = value(&output)["result"].clone();
    assert_eq!(result["frame_count"], 2);
    assert_eq!(result["scene_modified"], false);
    assert!(animation.join("frame_00000000.json").is_file());
    assert!(animation.join("frame_00000001.json").is_file());

    let geometry = directory.path().join("geometry");
    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("geometry")
        .arg("--target")
        .arg("body")
        .arg("--frames")
        .arg("1:2")
        .arg("--out")
        .arg(&geometry)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(geometry.join("frame_00000000_body.ply").is_file());
    assert!(geometry.join("frame_00000001_body.ply").is_file());
    assert!(
        fs::read_to_string(geometry.join("frame_00000000_body.ply"))
            .unwrap()
            .starts_with("ply\n")
    );
    let after: Value = serde_json::from_slice(&fs::read(&scene_file).unwrap()).unwrap();
    assert_eq!(before["revision"], after["revision"]);
    assert_eq!(after["revision"], 1);
}

#[test]
fn unsupported_bake_kinds_fail_honestly() {
    let directory = tempdir().unwrap();
    let scene = directory.path().join("unsupported_bake_scene");
    init(&scene);
    let out = directory.path().join("bake");
    let output = pot()
        .arg("bake")
        .arg(&scene)
        .arg("--kind")
        .arg("texture")
        .arg("--out")
        .arg(&out)
        .arg("--json")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4));
    let envelope = value(&output);
    assert_eq!(envelope["error"]["code"], "UNSUPPORTED_FEATURE");
    assert_eq!(envelope["error"]["details"]["status"], "not_supported");
}

#[test]
fn feature_catalog_operations_and_unsupported_rows_are_consistent() {
    let catalog = potter::catalog::feature_catalog();
    let features = catalog["features"]
        .as_array()
        .expect("catalog features are an array");
    let mut feature_ids = std::collections::HashSet::new();
    for feature in features {
        let feature_id = feature["feature_id"]
            .as_str()
            .expect("each feature has a string ID");
        assert!(
            feature_ids.insert(feature_id),
            "duplicate catalog feature ID: {feature_id}"
        );
        let pot_ops = feature["pot_ops"]
            .as_array()
            .expect("each feature has a pot_ops array");
        match feature["status"]
            .as_str()
            .expect("each feature has a status")
        {
            "supported" => {
                for operation in pot_ops {
                    let operation = operation
                        .as_str()
                        .expect("catalog operation names are strings");
                    assert!(
                        potter::ops::OP_NAMES.contains(&operation),
                        "{feature_id} lists unknown operation {operation}"
                    );
                }
            }
            "not_supported" => {
                let reason = feature["reason"]
                    .as_str()
                    .expect("unsupported features have a string reason");
                assert!(
                    !reason.trim().is_empty(),
                    "{feature_id} must explain why it is unsupported"
                );
            }
            status => panic!("{feature_id} has unknown status {status}"),
        }
    }
    for feature_id in [
        "rig.advanced",
        "rig.bone_collections_shapes_envelopes",
        "rig.rigify",
        "constraint.spline_ik.nurbs",
        "constraint.spline_ik.multiple_splines",
        "driver.location_rotation_difference",
        "shape_key.absolute",
    ] {
        assert!(
            features.iter().any(|feature| {
                feature["feature_id"] == feature_id && feature["status"] == "supported"
            }),
            "{feature_id} must be reported as supported"
        );
    }
    assert!(
        features.iter().any(|feature| {
            feature["feature_id"] == "constraint.ik.swing_ellipse_limit"
                && feature["status"] == "not_supported"
        }),
        "legacy IK X/Z Swing ellipse limits must be reported as unsupported"
    );
    assert!(
        features.iter().any(|feature| {
            feature["feature_id"] == "rig.rigify.full" && feature["status"] == "not_supported"
        }),
        "full Rigify generation and metadata must remain a separate unsupported feature"
    );
}
