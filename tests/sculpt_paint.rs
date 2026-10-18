#![expect(
    clippy::unwrap_used,
    reason = "integration-test setup uses fixed valid data"
)]

use std::{error::Error, fs, process::Command};

use glam::DVec3;
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;
#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;

use potter_core::{
    geom::{Mesh, sculpt},
    hash::{canonicalize, sha256},
};

fn create_scene() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(initialized.status.success());
    Ok((directory, scene))
}

fn apply(scene: &std::path::Path, operations: &Value) -> Result<Value, Box<dyn Error>> {
    apply_at_revision(scene, operations, 0)
}

fn apply_at_revision(
    scene: &std::path::Path,
    operations: &Value,
    base_revision: u64,
) -> Result<Value, Box<dyn Error>> {
    let file = scene.with_extension("operations.json");
    fs::write(
        &file,
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": base_revision, "operations": operations}),
        )?,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn stored_mesh(scene: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    let bytes = fs::read(scene.join("scene.json"))?;
    let document: Value = serde_json::from_slice(&bytes)?;
    Ok(document["data_blocks"]["body_mesh"]["mesh"].clone())
}

fn inspected_geometry(scene: &std::path::Path) -> Result<Value, Box<dyn Error>> {
    inspected_geometry_for(scene, "body")
}

fn inspected_geometry_for(scene: &std::path::Path, id: &str) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(scene)
        .arg("--id")
        .arg(id)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    Ok(envelope["result"]["items"][0]["evaluated_geometry"].clone())
}

fn has_nonzero_number(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::Array(values) => values.iter().any(has_nonzero_number),
        _ => false,
    }
}

#[test]
fn sculpt_multires_stroke_updates_mdisps_and_base_mesh_mode_preserves_grids()
-> Result<(), Box<dyn Error>> {
    let (_baseline_directory, baseline_scene) = create_scene()?;
    apply(
        &baseline_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1}
        ]),
    )?;
    let baseline_mesh = stored_mesh(&baseline_scene)?;
    let baseline_displacement = baseline_mesh["attributes"]["multires_displacement"].clone();
    let baseline_geometry = inspected_geometry(&baseline_scene)?;

    let stroke = json!({
        "op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
        "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0}],
        "falloff":"linear"
    });
    let (_multires_directory, multires_scene) = create_scene()?;
    apply(
        &multires_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1},
            stroke
        ]),
    )?;
    let multires_mesh = stored_mesh(&multires_scene)?;
    let multires_displacement = &multires_mesh["attributes"]["multires_displacement"];
    assert_ne!(
        multires_displacement["values"],
        baseline_displacement["values"]
    );
    assert!(
        multires_displacement["values"]
            .as_object()
            .is_some_and(|values| values.values().any(has_nonzero_number)),
        "native multires stroke wrote no nonzero MDisps values"
    );
    let multires_geometry = inspected_geometry(&multires_scene)?;
    assert_ne!(
        multires_geometry["positions"], baseline_geometry["positions"],
        "native multires stroke did not change evaluated geometry"
    );

    let (_base_directory, base_scene) = create_scene()?;
    apply(
        &base_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1},
            {"op":"modifier.update", "target":{"id":"body"}, "id":"multi_one",
             "set":{"params":{"levels":1,"sculpt_levels":1,"render_levels":1,"total_levels":1,"use_sculpt_base_mesh":true}}},
            stroke
        ]),
    )?;
    let base_mesh = stored_mesh(&base_scene)?;
    assert_ne!(base_mesh["vertices"], baseline_mesh["vertices"]);
    assert_eq!(
        base_mesh["attributes"]["multires_displacement"], baseline_displacement,
        "base-mesh sculpting unexpectedly edited MDisps"
    );
    Ok(())
}

/// Blender 5.2.2 oracle for one Draw dab on the driven, shrinkwrapped multires body. Generated by
/// tests/fixtures/multires/sculpt_oracle/gen_brush_oracle.py, whose header derives every setting from the Blender source.
const DRIVEN_ARMATURE_ORACLE: &str =
    include_str!("fixtures/multires/sculpt_oracle/driven_armature_sculpt.json");
/// Both sides evaluate the same pre-stroke geometry, so vertices are paired by pre-stroke position, and the
/// pairing must hold within the same float32 bound as the stroke. Bound, with u = 2^-24 = 5.96e-8 and M = 3.34
/// (as for ORACLE_POSITION_TOLERANCE): N = 41 rounding steps on the pre-stroke chain (armature skin, shrinkwrap
/// projection, multires base, subdivision stencils), counted from the stencils, not from Blender's instructions.
/// Euclidean bound sqrt(3) * 41 * 5.96e-8 * 3.34 = 1.41e-5, rounded up to 1.42e-5.
const ORACLE_PRESTROKE_TOLERANCE: f64 = 1.42e-5;
/// Potter's Draw (crates/potter-core/src/geom/sculpt.rs, Draw branch) computes Blender's Draw offset
/// (draw.cc:196-197) exactly, so the remaining gap is float32 rounding in Blender. Bound, with u = 2^-24 = 5.96e-8:
/// - M = 3.34 bounds |coordinate| in driven_armature_sculpt.json (object space; the largest is 3.3338).
/// - N = 35 rounding steps on the stroke path, per axis: the offset scalars, fade and position add (5); tangent
///   storage and reconstruction (6); bilinear grid interpolation (7); and a 9-weight Catmull-Clark stencil with
///   8 adds (17). Counted from the stencil shapes, not from Blender's compiled instruction stream.
/// - Euclidean bound sqrt(3) * N * u * M = 1.2e-5 for the stroke path.
/// - Pre-stroke pairing is asserted separately to ORACLE_PRESTROKE_TOLERANCE (1.42e-5); adding it gives 2.6e-5.
/// Rounded up to a power of ten: 1e-4.
const ORACLE_POSITION_TOLERANCE: f64 = 1.0e-4;

fn evaluated_positions(geometry: &Value) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    Ok(serde_json::from_value(geometry["positions"].clone())?)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn multires_sculpt_uses_driven_armature_and_shrinkwrap_prefix_for_brush_base()
-> Result<(), Box<dyn Error>> {
    let (_deformed_directory, deformed_scene) = create_scene()?;
    apply(
        &deformed_scene,
        &json!([
            {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0},"transform":{"translation":[0.0,0.0,1.0]}},
            {"op":"node.create","id":"surface","kind":"plane","params":{"size":8.0}},
            {"op":"node.create","id":"arm","kind":"armature"},
            {"op":"bone.create","target":{"id":"arm"},"id":"root","name":"Root","head":[0.0,0.0,0.0],"tail":[0.0,1.0,0.0]},
            {"op":"node.create","id":"controller","kind":"empty","transform":{"translation":[2.5,0.0,0.0]}},
            {"op":"vertex_group.create","target":{"id":"body"},"id":"root_group","name":"Root"},
            {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"root_group","weights":[
                {"vertex_id":0,"weight":1.0},{"vertex_id":1,"weight":1.0},
                {"vertex_id":2,"weight":1.0},{"vertex_id":3,"weight":1.0}
            ]},
            {"op":"modifier.create","target":{"id":"body"},"id":"armature_deform","type":"armature","params":{"object":"arm","use_vertex_groups":true}},
            {"op":"driver.create","target":{"id":"arm"},"id":"pose_x","path":"pose.bones[\"root\"].location","index":0,"type":"scripted_expression","variables":[{"name":"var","type":"transforms","target":"controller","path":"transform.translation","index":0}],"expression":"var"},
            {"op":"modifier.create","target":{"id":"body"},"id":"shrink","type":"shrinkwrap","params":{"target":"surface","wrap_method":"NEAREST_SURFACEPOINT","wrap_mode":"ON_SURFACE"}},
            {"op":"sculpt.multires","target":{"id":"body"},"id":"multi_one","levels":1,"render_levels":1},
            {"op":"modifier.update","target":{"id":"body"},"id":"multi_one","set":{"params":{"levels":1,"sculpt_levels":1,"render_levels":1,"total_levels":1,"use_sculpt_base_mesh":false}}},
            {"op":"modifier.create","target":{"id":"body"},"id":"after_multires","type":"subdivision","params":{"levels":1}},
        ]),
    )?;
    let before_geometry = inspected_geometry_for(&deformed_scene, "body")?;
    let positions = before_geometry["positions"]
        .as_array()
        .ok_or("driven prefix geometry has no positions")?;
    let mean_x = positions
        .iter()
        .map(|position| position[0].as_f64().unwrap())
        .sum::<f64>()
        / positions.len() as f64;
    let mean_z = positions
        .iter()
        .map(|position| position[2].as_f64().unwrap())
        .sum::<f64>()
        / positions.len() as f64;
    assert!(
        (mean_x - 2.5).abs() < 1.0e-8,
        "Armature driver was not applied"
    );
    assert!(
        (mean_z + 1.0).abs() < 1.0e-8,
        "Shrinkwrap operand was not applied"
    );

    let stroke = json!([{
        "op":"sculpt.stroke","target":{"id":"body"},"brush":"draw","scope":"shared",
        "samples":[{"position":[2.5,0.0,0.0],"pressure":1.0,"radius":1.5,"strength":0.5,"time":0.0}],
        "falloff":"linear"
    }]);
    // Stroke samples are world space. The deformed body's evaluated surface is at world z=0 (its local z=-1 plus its
    // translation z=+1), so the sample lies on it. The oracle sends the same point as object-space (2.5,0,-1).
    apply_at_revision(&deformed_scene, &stroke, 1)?;

    let deformed_displacement =
        stored_mesh(&deformed_scene)?["attributes"]["multires_displacement"]["values"].clone();
    assert!(
        deformed_displacement
            .as_object()
            .is_some_and(|values| values.values().any(has_nonzero_number)),
        "driven prefix stroke wrote no nonzero MDisps"
    );
    let after_geometry = inspected_geometry_for(&deformed_scene, "body")?;
    assert_ne!(
        after_geometry["positions"], before_geometry["positions"],
        "brush did not alter evaluated geometry at the driven prefix position"
    );

    let oracle: Value = serde_json::from_str(DRIVEN_ARMATURE_ORACLE)?;
    let contract_name = oracle["contract_attempt"]
        .as_str()
        .ok_or("oracle has no contract_attempt")?;
    let contract = oracle["attempts"]
        .as_array()
        .and_then(|attempts| {
            attempts
                .iter()
                .find(|attempt| attempt["name"] == contract_name)
        })
        .ok_or("oracle contract attempt is missing")?;
    let oracle_before: Vec<[f64; 3]> = serde_json::from_value(contract["before_local"].clone())?;
    let oracle_after: Vec<[f64; 3]> = serde_json::from_value(contract["after_local"].clone())?;
    let potter_before = evaluated_positions(&before_geometry)?;
    let potter_after = evaluated_positions(&after_geometry)?;
    assert_eq!(
        potter_before.len(),
        oracle_before.len(),
        "pre-stroke vertex count differs from Blender"
    );
    assert_eq!(oracle_after.len(), oracle_before.len());
    for (index, blender_before) in oracle_before.iter().enumerate() {
        let (matched, prestroke_gap) = potter_before
            .iter()
            .enumerate()
            .map(|(candidate, point)| (candidate, distance(*point, *blender_before)))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("driven prefix geometry has no positions")?;
        assert!(
            prestroke_gap <= ORACLE_PRESTROKE_TOLERANCE,
            "pre-stroke vertex {index} is {prestroke_gap} from Blender's position"
        );
        let gap = distance(potter_after[matched], oracle_after[index]);
        assert!(
            gap <= ORACLE_POSITION_TOLERANCE,
            "vertex {index}: Potter {:?} vs Blender {:?} differ by {gap} (tolerance {ORACLE_POSITION_TOLERANCE})",
            potter_after[matched],
            oracle_after[index]
        );
    }
    Ok(())
}

#[test]
fn sculpt_multires_stroke_respects_interpolated_control_masks() -> Result<(), Box<dyn Error>> {
    let source = Mesh::box_mesh(potter_core::geom::BoxParams::default())?;
    let vertex_ids: Vec<String> = source
        .vertices
        .iter()
        .map(|vertex| format!("v{}", vertex.id))
        .collect();

    let (_baseline_directory, baseline_scene) = create_scene()?;
    apply(
        &baseline_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1}
        ]),
    )?;
    let baseline_mesh = stored_mesh(&baseline_scene)?;
    let baseline_displacement =
        baseline_mesh["attributes"]["multires_displacement"]["values"].clone();
    let baseline_geometry = inspected_geometry(&baseline_scene)?;

    let (_masked_directory, masked_scene) = create_scene()?;
    apply(
        &masked_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.mask", "target":{"id":"body"}, "scope":"shared", "action":"set",
             "elements":{"domain":"vertex", "ids":vertex_ids}, "value":1.0},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1},
            {"op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let masked_mesh = stored_mesh(&masked_scene)?;
    let stored_masks = masked_mesh["attributes"]["sculpt.mask"]["values"]
        .as_object()
        .ok_or("stored sculpt mask is missing")?;
    assert!(
        vertex_ids
            .iter()
            .all(|id| { stored_masks.get(id.as_str()).and_then(Value::as_f64) == Some(1.0) })
    );
    assert_eq!(
        masked_mesh["attributes"]["multires_displacement"]["values"], baseline_displacement,
        "fully masked multires stroke changed MDisps"
    );
    let masked_geometry = inspected_geometry(&masked_scene)?;
    assert_eq!(
        masked_geometry["positions"], baseline_geometry["positions"],
        "fully masked multires stroke moved evaluated subdivision points"
    );
    Ok(())
}

#[test]
fn multires_sculpt_levels_select_native_grid_resolution() -> Result<(), Box<dyn Error>> {
    let mut grid_values = Vec::new();
    for sculpt_levels in [1, 2] {
        let (_directory, scene) = create_scene()?;
        apply(
            &scene,
            &json!([
                {"op":"node.create", "id":"body", "kind":"box", "params":{}},
                {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":2, "render_levels":2},
                {"op":"modifier.update", "target":{"id":"body"}, "id":"multi_one",
                 "set":{"params":{"levels":2,"sculpt_levels":sculpt_levels,"render_levels":2,"total_levels":2}}},
                {"op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
                 "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0}],
                 "falloff":"linear"}
            ]),
        )?;
        let mesh = stored_mesh(&scene)?;
        let values = mesh["attributes"]["multires_displacement"]["values"].clone();
        assert!(
            values
                .as_object()
                .is_some_and(|faces| faces.values().any(has_nonzero_number)),
            "sculpt level {sculpt_levels} wrote no nonzero MDisps values"
        );
        grid_values.push(values);
    }
    assert_ne!(
        grid_values[0], grid_values[1],
        "sculpt_levels did not select a different MDisps sample resolution"
    );
    Ok(())
}
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one Blender round trip covers packed/external MDisps, continued sculpt, and apply"
)]
fn multires_blend_roundtrips_continue_sculpt_and_apply() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        return Ok(());
    };
    let (source_directory, source_scene) = create_scene()?;
    apply(
        &source_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"camera.create", "id":"multires_camera", "transform":{"translation":[0.0,0.0,4.0]},
             "projection":"orthographic", "ortho_scale":3.0},
            {"op":"scene.update", "target":{"id":"scene_main"}, "set":{"camera":"multires_camera"}},
            {"op":"render.update", "target":{"id":"scene_main"},
             "set":{"resolution_x":16,"resolution_y":16,"samples":1,"seed":0,"dicing_rate":1.25,
                    "preview_dicing_rate":6.0,"max_subdivisions":5,"offscreen_dicing_scale":2.0,
                    "dicing_camera":"multires_camera"}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1},
            {"op":"modifier.update", "target":{"id":"body"}, "id":"multi_one",
             "set":{"params":{"show_only_control_edges":false,"use_sculpt_base_mesh":false}}},
            {"op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let source_document: Value =
        serde_json::from_slice(&fs::read(source_scene.join("scene.json"))?)?;
    let source_mesh = &source_document["data_blocks"]["body_mesh"]["mesh"];
    let source_record = &source_mesh["attributes"]["multires_displacement"];
    assert!(
        source_record["values"]
            .as_object()
            .is_some_and(|values| values.values().any(has_nonzero_number)),
        "source sculpt wrote no nonzero MDisps values"
    );

    for (stem, pack) in [("external", false), ("packed", true)] {
        let output = source_directory
            .path()
            .join(format!("continuation-{stem}.blend"));
        let mut export = Command::new(env!("CARGO_BIN_EXE_pot"));
        export
            .args(["export"])
            .arg(&source_scene)
            .args(["--format", "blend", "--out"])
            .arg(&output)
            .args(["--blender"])
            .arg(&blender)
            .arg("--json");
        if pack {
            export.arg("--pack");
        }
        let result = export.output()?;
        assert!(
            result.status.success(),
            "Blender export failed:\n{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );

        let sidecar = source_directory
            .path()
            .join(format!("continuation-{stem}_assets"));
        assert_eq!(sidecar.exists(), !pack);
        let moved_directory = source_directory.path().join(format!("moved-{stem}"));
        fs::create_dir(&moved_directory)?;
        let moved_output =
            moved_directory.join(output.file_name().ok_or("missing blend filename")?);
        fs::rename(&output, &moved_output)?;
        if !pack {
            fs::rename(
                &sidecar,
                moved_directory.join(sidecar.file_name().ok_or("missing sidecar name")?),
            )?;
        }

        let (_import_directory, import_scene) = create_scene()?;
        let imported = Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import"])
            .arg(&import_scene)
            .args(["--file"])
            .arg(&moved_output)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json")
            .output()?;
        assert!(
            imported.status.success(),
            "Blender import failed:\n{}\n{}",
            String::from_utf8_lossy(&imported.stdout),
            String::from_utf8_lossy(&imported.stderr)
        );
        let imported_document: Value =
            serde_json::from_slice(&fs::read(import_scene.join("scene.json"))?)?;
        let imported_scene_id = imported_document["active_scene"]
            .as_str()
            .ok_or("imported active scene is missing")?;
        let imported_render = &imported_document["scenes"][imported_scene_id]["render"];
        assert_eq!(imported_render["dicing_rate"], json!(1.25));
        assert_eq!(imported_render["preview_dicing_rate"], json!(6.0));
        assert_eq!(imported_render["max_subdivisions"], json!(5));
        assert_eq!(imported_render["offscreen_dicing_scale"], json!(2.0));
        assert_eq!(imported_render["dicing_camera"], json!("multires_camera"));
        let (node_id, data_id, modifier_id) = imported_document["nodes"]
            .as_object()
            .and_then(|nodes| {
                nodes.iter().find_map(|(node_id, node)| {
                    let modifier = node.get("modifiers")?.as_array()?.iter().find(|modifier| {
                        modifier.get("type").and_then(Value::as_str) == Some("multires")
                    })?;
                    Some((
                        node_id.clone(),
                        node.get("data")?.as_str()?.to_owned(),
                        modifier.get("id")?.as_str()?.to_owned(),
                    ))
                })
            })
            .ok_or("imported multires node is missing")?;
        let imported_record = imported_document["data_blocks"][data_id.as_str()]["mesh"]["attributes"]
            ["multires_displacement"]
            .clone();
        let imported_modifier = imported_document["nodes"][node_id.as_str()]["modifiers"]
            .as_array()
            .and_then(|modifiers| {
                modifiers
                    .iter()
                    .find(|modifier| modifier["type"] == "multires")
            })
            .ok_or("imported Multires modifier is missing")?;
        assert_eq!(imported_modifier["params"]["is_external"], json!(!pack));
        assert_eq!(
            imported_modifier["params"]["show_only_control_edges"],
            json!(false)
        );
        assert_eq!(
            imported_modifier["params"]["use_sculpt_base_mesh"],
            json!(false)
        );
        assert_eq!(imported_modifier["params"]["levels"], json!(1));
        assert_eq!(imported_modifier["params"]["sculpt_levels"], json!(1));
        assert_eq!(imported_modifier["params"]["render_levels"], json!(1));
        assert_eq!(imported_modifier["params"]["total_levels"], json!(1));
        if !pack {
            let filepath = imported_record["external"]["filepath"]
                .as_str()
                .ok_or("external MDisps filepath is missing")?;
            assert!(filepath.starts_with("//continuation-external_assets/"));
            assert!(!filepath.contains("/tmp"));
        }

        let before_geometry = inspected_geometry_for(&import_scene, &node_id)?;
        let base_revision = imported_document["revision"]
            .as_u64()
            .ok_or("imported project revision is missing")?;
        apply_at_revision(
            &import_scene,
            &json!([{
                "op":"sculpt.stroke", "target":{"id":node_id}, "brush":"draw", "scope":"shared",
                "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":1.0,"strength":0.5,"time":0.0}],
                "falloff":"linear"
            }]),
            base_revision,
        )?;
        let continued_document: Value =
            serde_json::from_slice(&fs::read(import_scene.join("scene.json"))?)?;
        let continued_record = continued_document["data_blocks"][data_id.as_str()]["mesh"]["attributes"]
            ["multires_displacement"]
            .clone();
        assert_ne!(continued_record["values"], imported_record["values"]);
        assert!(
            continued_record["values"]
                .as_object()
                .is_some_and(|values| values.values().any(has_nonzero_number)),
            "{stem} continuation stroke wrote no nonzero MDisps values"
        );
        let continued_geometry = inspected_geometry_for(&import_scene, &node_id)?;
        assert_ne!(
            continued_geometry["positions"], before_geometry["positions"],
            "{stem} continuation stroke did not affect evaluated geometry"
        );
        let render_output = import_scene.join(format!("render-{stem}"));
        let rendered = Command::new(env!("CARGO_BIN_EXE_pot"))
            .arg("render")
            .arg(&import_scene)
            .args([
                "--engine", "path", "--frames", "1:1", "--format", "png", "--out",
            ])
            .arg(&render_output)
            .arg("--json")
            .output()?;
        assert!(
            rendered.status.success(),
            "{stem} native render failed:\n{}\n{}",
            String::from_utf8_lossy(&rendered.stdout),
            String::from_utf8_lossy(&rendered.stderr)
        );
        let rendered_image = render_output.join("frame_0001.png");
        assert!(
            fs::metadata(&rendered_image)?.len() > 0,
            "{stem} native render produced no image"
        );

        let continued_revision = continued_document["revision"]
            .as_u64()
            .ok_or("continued project revision is missing")?;
        apply_at_revision(
            &import_scene,
            &json!([{
                "op":"modifier.apply",
                "target":{"id":node_id},
                "modifier_id":modifier_id
            }]),
            continued_revision,
        )?;
        let applied_geometry = inspected_geometry_for(&import_scene, &node_id)?;
        assert_eq!(
            applied_geometry["positions"], continued_geometry["positions"],
            "{stem} native modifier apply changed the sculpted result"
        );
    }
    Ok(())
}

#[test]
fn draw_respects_radius_and_mask_and_vertex_paint_updates_touched_corners()
-> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_scene()?;
    apply(
        &scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}, "transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
             "samples":[{"position":[3.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":0.5,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let after_draw = stored_mesh(&scene)?;
    let drawn_document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert!(drawn_document["data_blocks"]["body_mesh"]["descriptor"].is_null());
    assert!(drawn_document["nodes"]["body"]["primitive"].is_null());
    let original = Mesh::box_mesh(potter_core::geom::BoxParams::default())?;
    let original_v6 = original
        .vertices
        .iter()
        .find(|vertex| vertex.id == 6)
        .unwrap()
        .co;
    let drawn_v6: [f64; 3] = serde_json::from_value(after_draw["vertices"][6]["co"].clone())?;
    let displacement = DVec3::from_array(drawn_v6) - original_v6;
    assert!(displacement.length() > 0.0);
    assert!(displacement.dot(DVec3::ONE.normalize()) > 0.0);
    let drawn_v0 = after_draw["vertices"][0]["co"].clone();
    assert_eq!(drawn_v0, json!(original.vertices[0].co.to_array()));

    let (_masked_directory, masked_scene) = create_scene()?;
    apply(
        &masked_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.mask", "target":{"id":"body"}, "scope":"shared", "action":"set",
             "elements":{"domain":"vertex","ids":["v6"]}, "value":1.0},
            {"op":"sculpt.stroke", "target":{"id":"body"}, "brush":"draw", "scope":"shared",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":0.5,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let masked = stored_mesh(&masked_scene)?;
    assert_eq!(
        masked["vertices"][6]["co"],
        json!(original.vertices[6].co.to_array())
    );
    assert_eq!(
        masked["attributes"]["sculpt.mask"]["values"]["v6"],
        json!(1.0)
    );

    let (_paint_directory, paint_scene) = create_scene()?;
    apply(
        &paint_scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"paint.vertex", "target":{"id":"body"}, "scope":"shared",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":1.0,"time":0.0}],
             "falloff":"linear", "color":[0.1,0.2,0.3,1.0], "blend":"mix"},
            {"op":"paint.weight", "target":{"id":"body"}, "scope":"shared", "group":"skin", "weight":0.75,
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":1.0,"time":0.0}],
             "falloff":"linear"},
            {"op":"vertex_group.create", "target":{"id":"body"}, "id":"bone", "name":"Bone"},
            {"op":"paint.weight", "target":{"id":"body"}, "scope":"shared", "group":"bone", "weight":0.25,
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":1.0,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let painted = stored_mesh(&paint_scene)?;
    let color = &painted["attributes"]["color"];
    assert_eq!(color["domain"], json!("corner"));
    assert_eq!(color["type"], json!("color"));
    assert_eq!(
        painted["attributes"]["color_layers"],
        json!({"layers": ["color"], "active": "color", "render": "color"})
    );
    let painted_at_v6 = painted["faces"].as_array().unwrap().iter().any(|face| {
        let row = &color["values"][format!("f{}", face["id"]).as_str()];
        face["v"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .any(|(corner, vertex)| {
                *vertex == json!(6) && row[corner] == json!([0.1, 0.2, 0.3, 1.0])
            })
    });
    assert!(painted_at_v6);
    assert_eq!(
        painted["attributes"]["weight:skin"]["values"]["v6"],
        json!(0.75)
    );
    let painted_document: Value =
        serde_json::from_slice(&fs::read(paint_scene.join("scene.json"))?)?;
    assert_eq!(
        painted_document["data_blocks"]["body_mesh"]["vertex_weights"]["6"]["bone"],
        json!(0.25)
    );
    Ok(())
}

#[test]
fn sculpt_multires_runs_catmull_clark_evaluation() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_scene()?;
    apply(
        &scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.multires", "target":{"id":"body"}, "id":"multi_one", "levels":1, "render_levels":1}
        ]),
    )?;
    let inspected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("inspect")
        .arg(&scene)
        .arg("--id")
        .arg("body")
        .arg("--json")
        .output()?;
    assert!(
        inspected.status.success(),
        "{}",
        String::from_utf8_lossy(&inspected.stdout)
    );
    let envelope: Value = serde_json::from_slice(&inspected.stdout)?;
    let item = &envelope["result"]["items"][0];
    assert_eq!(item["evaluated_geometry"]["vertex_count"], json!(26));
    assert_eq!(item["evaluated_geometry"]["face_count"], json!(24));
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    assert_eq!(
        document["data_blocks"]["body_mesh"]["mesh"]["attributes"]["multires_displacement"]["total_levels"],
        json!(1)
    );
    assert_eq!(
        document["nodes"]["body"]["modifiers"][0]["params"]["total_levels"],
        json!(1)
    );
    Ok(())
}
#[test]
fn sculpt_voxel_remesh_replaces_source_geometry() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_scene()?;
    let result = apply(
        &scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"sculpt.remesh_voxel", "target":{"id":"body"}, "scope":"shared", "voxel_size":0.5}
        ]),
    )?;
    assert_eq!(result["ok"], json!(true));
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let mesh = &document["data_blocks"]["body_mesh"]["mesh"];
    assert_ne!(mesh["vertices"].as_array().unwrap().len(), 8);
    assert!(
        !mesh["faces"].as_array().unwrap().is_empty(),
        "voxel remesh produced no faces"
    );
    assert!(document["data_blocks"]["body_mesh"]["descriptor"].is_null());
    Ok(())
}

#[test]
fn texture_paint_requires_image_reference() -> Result<(), Box<dyn Error>> {
    let (_directory, scene) = create_scene()?;
    let operations = json!({
        "schema_version":1,
        "base_revision":0,
        "operations":[
            {"op":"node.create", "id":"body", "kind":"box", "params":{"size":1.0}},
            {"op":"paint.texture", "target":{"id":"body"}}
        ]
    });
    let batch_file = scene.with_extension("texture.json");
    fs::write(&batch_file, serde_json::to_vec(&operations)?)?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(batch_file)
        .arg("--json")
        .output()?;
    assert!(!output.status.success());
    let envelope: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(envelope["error"]["code"], json!("INVALID_OPERATION"));
    assert_eq!(
        envelope["error"]["details"]["pointer"],
        json!("/operations/1/image")
    );
    Ok(())
}

#[test]
fn sculpt_shared_data_requires_scope_and_single_user_isolates_stroke() -> Result<(), Box<dyn Error>>
{
    let (_directory, scene) = create_scene()?;
    let shared_operations = json!([
        {"op":"node.create", "id":"body", "kind":"box", "params":{}},
        {"op":"node.duplicate", "target":{"id":"body"}, "id":"body_copy", "mode":"linked"},
        {"op":"sculpt.stroke", "target":{"id":"body_copy"}, "brush":"draw",
         "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":0.5,"time":0.0}],
         "falloff":"linear"}
    ]);
    let rejected_batch =
        json!({"schema_version":1,"base_revision":0,"operations":shared_operations});
    let rejected_file = scene.with_extension("shared.json");
    fs::write(&rejected_file, serde_json::to_vec(&rejected_batch)?)?;
    let rejected = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(rejected_file)
        .arg("--json")
        .output()?;
    assert!(!rejected.status.success());
    let error: Value = serde_json::from_slice(&rejected.stdout)?;
    assert_eq!(error["error"]["code"], json!("SHARED_DATA_REQUIRES_SCOPE"));

    apply(
        &scene,
        &json!([
            {"op":"node.create", "id":"body", "kind":"box", "params":{}},
            {"op":"node.duplicate", "target":{"id":"body"}, "id":"body_copy", "mode":"linked"},
            {"op":"sculpt.stroke", "target":{"id":"body_copy"}, "brush":"draw", "scope":"single_user",
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":0.5,"time":0.0}],
             "falloff":"linear"}
        ]),
    )?;
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let original_data = document["nodes"]["body"]["data"].as_str().unwrap();
    let isolated_data = document["nodes"]["body_copy"]["data"].as_str().unwrap();
    assert_ne!(original_data, isolated_data);
    let original_vertex = &document["data_blocks"][original_data]["mesh"]["vertices"][6]["co"];
    let edited_vertex = &document["data_blocks"][isolated_data]["mesh"]["vertices"][6]["co"];
    assert_eq!(original_vertex, &json!([1.0, 1.0, 1.0]));
    assert_ne!(edited_vertex, original_vertex);
    Ok(())
}

#[test]
fn sculpt_symmetry_and_repeat_strokes_are_deterministic() -> Result<(), Box<dyn Error>> {
    let base = Mesh::box_mesh(potter_core::geom::BoxParams::default())?;
    let stroke = |symmetry: &[&str]| sculpt::Stroke {
        brush: sculpt::Brush::Draw,
        samples: vec![sculpt::StrokeSample {
            position: DVec3::new(0.0, -1.0, 1.0),
            pressure: 1.0,
            radius: 1.1,
            strength: 0.5,
            time: 0.0,
        }],
        falloff: sculpt::Falloff::Linear,
        symmetry: symmetry.iter().map(|axis| (*axis).to_owned()).collect(),
        delta: None,
    };
    let mut mirrored = base.clone();
    sculpt::apply_stroke(&mut mirrored, &stroke(&["x"]))?;
    let movement = |mesh: &Mesh, id| {
        let original = base
            .vertices
            .iter()
            .find(|vertex| vertex.id == id)
            .unwrap()
            .co;
        let current = mesh
            .vertices
            .iter()
            .find(|vertex| vertex.id == id)
            .unwrap()
            .co;
        current.z - original.z
    };
    assert!(movement(&mirrored, 4) > 0.0);
    assert!((movement(&mirrored, 4) - movement(&mirrored, 5)).abs() < 1.0e-12);

    let mut first = base.clone();
    let mut second = base;
    let deterministic = stroke(&[]);
    sculpt::apply_stroke(&mut first, &deterministic)?;
    sculpt::apply_stroke(&mut second, &deterministic)?;
    let first_hash = sha256(&canonicalize(&serde_json::to_value(&first)?)?);
    let second_hash = sha256(&canonicalize(&serde_json::to_value(&second)?)?);
    assert_eq!(first_hash, second_hash);
    Ok(())
}

proptest! {
    #[test]
    fn smoothing_reduces_laplacian_energy(offset in -5.0_f64..5.0, factor in 0.1_f64..1.0) {
        let positions = vec![
            DVec3::new(offset, offset * 0.25, 0.3),
            DVec3::new(-1.0, -1.0, 0.0),
            DVec3::new(1.0, -1.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(-1.0, 1.0, 0.0),
        ];
        let mut mesh = Mesh::from_positions_and_faces(
            positions,
            vec![vec![0, 1, 2], vec![0, 2, 3], vec![0, 3, 4], vec![0, 4, 1]],
        ).unwrap();
        let before = sculpt::laplacian_energy(&mesh);
        sculpt::smooth_vertex(&mut mesh, 0, factor).unwrap();
        prop_assert!(sculpt::laplacian_energy(&mesh) < before);
    }
}
