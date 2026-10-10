#![expect(
    clippy::unwrap_used,
    reason = "integration-test setup uses fixed valid data"
)]

use std::{error::Error, fs, process::Command};

use glam::DVec3;
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;

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
    let file = scene.with_extension("operations.json");
    fs::write(
        &file,
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
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
             "falloff":"linear", "seed":7}
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
             "falloff":"linear", "seed":7}
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
             "falloff":"linear", "color":[0.1,0.2,0.3,1.0], "blend":"mix", "seed":7},
            {"op":"paint.weight", "target":{"id":"body"}, "scope":"shared", "group":"skin", "weight":0.75,
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":1.0,"time":0.0}],
             "falloff":"linear", "seed":7},
            {"op":"vertex_group.create", "target":{"id":"body"}, "id":"bone", "name":"Bone"},
            {"op":"paint.weight", "target":{"id":"body"}, "scope":"shared", "group":"bone", "weight":0.25,
             "samples":[{"position":[1.0,1.0,1.0],"pressure":1.0,"radius":0.25,"strength":1.0,"time":0.0}],
             "falloff":"linear", "seed":7}
        ]),
    )?;
    let painted = stored_mesh(&paint_scene)?;
    let corner_colors = painted["attributes"]["color"]["values"]
        .as_object()
        .unwrap();
    assert!(corner_colors
        .iter()
        .any(|(corner, color)| corner.ends_with(":v6") && color == &json!([0.1, 0.2, 0.3, 1.0])));
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
         "falloff":"linear", "seed":4}
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
             "falloff":"linear", "seed":4}
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
fn sculpt_symmetry_and_seed_are_deterministic() -> Result<(), Box<dyn Error>> {
    let base = Mesh::box_mesh(potter_core::geom::BoxParams::default())?;
    let stroke = |symmetry: &[&str], seed| sculpt::Stroke {
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
        seed,
        delta: None,
    };
    let mut mirrored = base.clone();
    sculpt::apply_stroke(&mut mirrored, &stroke(&["x"], 33))?;
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
    let deterministic = stroke(&[], 9127);
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
