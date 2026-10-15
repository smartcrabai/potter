//! Modifier provenance: vertex groups, creases, and other element data follow the vertices they
//! came from through mesh-rebuilding modifiers, and a dropped attribute is reported as a warning.

use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use potter_core::{
    geom::{
        GridParams, Mesh,
        modifiers::{evaluate_modifiers, evaluate_modifiers_with_groups_at_frame},
    },
    model::{Id, Modifier, ModifierRuntime, Registry, SceneDoc},
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[expect(
    clippy::needless_pass_by_value,
    reason = "modifier test fixtures accept convenient owned json! values"
)]
fn modifier(modifier_type: &str, params: Value) -> TestResult<Modifier> {
    Ok(Modifier {
        id: Id::new("test_modifier")?,
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        params: params
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

/// Vertex-group weights from a mesh's `vertex_groups` attribute, keyed by position.
fn weights_by_position(mesh: &Mesh, group: &str) -> TestResult<Vec<(DVec3, f64)>> {
    let groups = mesh.attributes["vertex_groups"]
        .as_object()
        .ok_or("vertex groups missing")?;
    let values = groups
        .get(group)
        .and_then(Value::as_object)
        .ok_or("group missing")?;
    Ok(mesh
        .vertices
        .iter()
        .filter_map(|vertex| {
            values
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .map(|weight| (vertex.co, weight))
        })
        .collect())
}

fn weight_at(entries: &[(DVec3, f64)], point: DVec3) -> Option<f64> {
    entries
        .iter()
        .find(|(co, _)| co.distance(point) < 1.0e-9)
        .map(|(_, weight)| *weight)
}

#[test]
fn mirror_swaps_side_group_names_and_averages_merged_plane_vertices() -> TestResult {
    // A on the mirror plane, B on +X, C on the plane as well.
    let mut mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    let ids: Vec<u32> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({
            "Arm.L": {format!("v{}", ids[0]): 0.5, format!("v{}", ids[1]): 0.25},
            "Arm.R": {}
        }),
    );

    let output = evaluate_modifiers(&mesh, &[modifier("mirror", json!({}))?])?;
    let left = weights_by_position(&output, "Arm.L")?;
    let right = weights_by_position(&output, "Arm.R")?;

    // The plane vertex is shared with its copy: both sides receive the average (0.5 + 0) / 2.
    assert_eq!(weight_at(&left, DVec3::ZERO), Some(0.25));
    assert_eq!(weight_at(&right, DVec3::ZERO), Some(0.25));
    // The +X vertex keeps Arm.L; its mirrored copy at -X carries the weight under Arm.R only.
    assert_eq!(weight_at(&left, DVec3::X), Some(0.25));
    assert_eq!(weight_at(&right, DVec3::X), None);
    assert_eq!(weight_at(&left, DVec3::NEG_X), None);
    assert_eq!(weight_at(&right, DVec3::NEG_X), Some(0.25));
    Ok(())
}

#[test]
fn mirror_copies_edge_creases_from_their_source_edges() -> TestResult {
    let mut mesh =
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])?;
    // Crease the edge between +X and +Y; its mirror image must keep the crease.
    let crease_edge = mesh
        .edges
        .iter()
        .find(|edge| {
            let positions: Vec<DVec3> = edge
                .vertices
                .iter()
                .filter_map(|id| mesh.vertices.iter().find(|vertex| vertex.id == *id))
                .map(|vertex| vertex.co)
                .collect();
            positions.contains(&DVec3::X) && positions.contains(&DVec3::Y)
        })
        .map(|edge| edge.id)
        .ok_or("crease edge missing")?;
    mesh.attributes.insert(
        "crease_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":{format!("e{crease_edge}"): 0.5}}),
    );

    let output = evaluate_modifiers(&mesh, &[modifier("mirror", json!({}))?])?;
    let creased: Vec<(DVec3, DVec3)> = output
        .edges
        .iter()
        .filter(|edge| {
            output
                .attributes
                .get("crease_edge")
                .and_then(|attribute| attribute["values"].get(format!("e{}", edge.id)))
                .and_then(Value::as_f64)
                == Some(0.5)
        })
        .filter_map(|edge| {
            let position = |id: u32| {
                output
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == id)
                    .map(|vertex| vertex.co)
            };
            Some((position(edge.vertices[0])?, position(edge.vertices[1])?))
        })
        .collect();
    assert_eq!(
        creased.len(),
        2,
        "the edge and its mirror image are creased"
    );
    assert!(creased.iter().any(|(a, b)| {
        (a.distance(DVec3::NEG_X) < 1.0e-9 && b.distance(DVec3::Y) < 1.0e-9)
            || (b.distance(DVec3::NEG_X) < 1.0e-9 && a.distance(DVec3::Y) < 1.0e-9)
    }));
    Ok(())
}

#[test]
fn decimate_keeps_each_surviving_vertex_group_weight_with_its_vertex() -> TestResult {
    let mut mesh = Mesh::grid(GridParams {
        size_x: 2.0,
        size_y: 2.0,
        x_subdivisions: 6,
        y_subdivisions: 6,
    })?;
    // A linear ramp along X: a correctly interpolated weight equals (x + 1) / 2 at every vertex.
    let ramp: BTreeMap<String, Value> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            (
                format!("v{}", vertex.id),
                json!(f64::midpoint(vertex.co.x, 1.0)),
            )
        })
        .collect();
    let source_count = mesh.vertices.len();
    mesh.attributes
        .insert("vertex_groups".to_owned(), json!({"Ramp": ramp}));

    let output = evaluate_modifiers(
        &mesh,
        &[modifier(
            "decimate",
            json!({"decimate_type":"COLLAPSE","ratio":0.5}),
        )?],
    )?;
    assert!(
        output.vertices.len() < source_count,
        "decimate removed vertices"
    );
    let weights = weights_by_position(&output, "Ramp")?;
    assert_eq!(
        weights.len(),
        output.vertices.len(),
        "every vertex keeps a weight"
    );
    for (co, weight) in &weights {
        assert!(
            (weight - f64::midpoint(co.x, 1.0)).abs() < 0.05,
            "weight {weight} at {co:?} does not follow its vertex"
        );
    }
    Ok(())
}

#[test]
fn subdivision_interpolates_vertex_group_weights_to_new_vertices() -> TestResult {
    let mut mesh = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    mesh.attributes
        .insert("vertex_groups".to_owned(), json!({"Corner": {"v0": 1.0}}));

    let output = evaluate_modifiers(&mesh, &[modifier("subdivision", json!({"levels":1}))?])?;
    let mut weights: Vec<f64> = weights_by_position(&output, "Corner")?
        .into_iter()
        .map(|(_, weight)| weight)
        .collect();
    weights.sort_by(f64::total_cmp);
    // Original corner keeps 1; its two adjacent edge points get half; the face point gets a quarter.
    // The opposite edge points and corners carry no membership.
    assert_eq!(weights, vec![0.25, 0.5, 0.5, 1.0]);
    Ok(())
}

#[test]
fn remesh_reports_dropped_vertex_groups_as_a_typed_warning() -> TestResult {
    let mut mesh = Mesh::box_mesh(potter_core::geom::BoxParams::default())?;
    let ids: Vec<u32> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"Body": ids.iter().map(|id| (format!("v{id}"), json!(1.0))).collect::<serde_json::Map<_, _>>()}),
    );
    let remesh = modifier(
        "remesh",
        json!({
            "mode":"BLOCKS","octree_depth":3,"scale":0.9,"sharpness":1.0,
            "use_remove_disconnected":false,"threshold":1.0,"use_smooth_shade":true
        }),
    )?;
    let evaluation =
        evaluate_modifiers_with_groups_at_frame(&mesh, &[remesh], &Registry::new(), 1.0)?;
    assert!(
        evaluation.mesh.attributes.get("vertex_groups").is_none(),
        "remesh output must not carry stale vertex groups"
    );
    let warning = evaluation
        .warnings
        .iter()
        .find(|warning| warning.details["feature_id"] == "modifier.attribute_dropped")
        .ok_or("attribute drop warning missing")?;
    assert_eq!(warning.details["modifier_type"], "remesh");
    assert!(
        warning.details["attributes"]
            .as_array()
            .is_some_and(|attributes| attributes.iter().any(|name| name == "vertex_groups"))
    );
    Ok(())
}

// Armature skin export of a mirrored mesh: the mirror copies must carry the side-swapped group
// weights, so the glTF writer does not reject them as unweighted and the importer returns them.

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> TestResult<std::process::Output> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "command failed ({}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(output)
}

fn apply(scene: &Path, operations: &Value) -> TestResult {
    let batch = scene.join("operations.json");
    fs::write(&batch, serde_json::to_vec(operations)?)?;
    run_ok(
        pot()
            .arg("apply")
            .arg(scene)
            .arg("--file")
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}

#[test]
fn mirrored_armature_skin_export_weights_both_halves() -> TestResult {
    let directory = tempdir()?;
    let source = directory.path().join("source");
    let target = directory.path().join("target");
    let gltf = directory.path().join("mirror-skin.glb");
    run_ok(pot().arg("init").arg(&source).arg("--json"))?;
    let vertex_weights: Vec<Value> = (0..8)
        .map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0}))
        .collect();
    apply(
        &source,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
                {"op":"mesh.transform_elements","target":{"id":"body","elements":{"domain":"vertex","ids":["v0","v1","v2","v3","v4","v5","v6","v7"]}},"translation":[2.0,0.0,0.0]},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"arm_left","name":"Arm.L"},
                {"op":"vertex_group.create","target":{"id":"body"},"id":"arm_right","name":"Arm.R"},
                {"op":"vertex_group.assign","target":{"id":"body"},"group_id":"arm_left","weights":vertex_weights},
                {"op":"node.create","id":"rig","name":"Rig","kind":"armature"},
                {"op":"bone.create","target":{"id":"rig"},"id":"bone_left","name":"Arm.L","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]},
                {"op":"bone.create","target":{"id":"rig"},"id":"bone_right","name":"Arm.R","head":[0.0,0.0,0.0],"tail":[0.0,0.0,1.0]},
                {"op":"modifier.create","target":{"id":"body"},"id":"mirror","type":"mirror","params":{"use_axis":[true,false,false]}},
                {"op":"modifier.apply","target":{"id":"body"},"modifier_id":"mirror"},
                {"op":"modifier.create","target":{"id":"body"},"id":"skin","type":"armature","params":{"object":"rig","use_vertex_groups":true}}
            ]
        }),
    )?;
    run_ok(
        pot()
            .arg("export")
            .arg(&source)
            .args(["--format", "glb", "--allow-lossy", "--out"])
            .arg(&gltf)
            .args(["--overwrite", "--json"]),
    )?;
    run_ok(pot().arg("init").arg(&target).arg("--json"))?;
    run_ok(
        pot()
            .arg("import")
            .arg(&target)
            .arg("--file")
            .arg(&gltf)
            .args([
                "--format",
                "glb",
                "--allow-lossy",
                "--base-revision",
                "0",
                "--mode",
                "replace",
                "--json",
            ]),
    )?;

    let doc: SceneDoc = serde_json::from_slice(&fs::read(target.join("scene.json"))?)?;
    let data = doc
        .data_blocks
        .values()
        .find(|data| {
            data.mesh
                .as_ref()
                .is_some_and(|mesh| mesh.vertices.len() == 16)
        })
        .ok_or("mirrored skinned mesh was not imported")?;
    let mesh = data.mesh.as_ref().ok_or("mesh missing")?;
    let group_id = |name: &str| -> TestResult<Id> {
        Ok(data
            .vertex_groups
            .iter()
            .find(|group| group.name == name)
            .ok_or("imported vertex group missing")?
            .id
            .clone())
    };
    let left = group_id("Arm.L")?;
    let right = group_id("Arm.R")?;
    let mut mirrored = 0;
    let mut original = 0;
    for vertex in &mesh.vertices {
        let weights = data
            .vertex_weights
            .get(&vertex.id)
            .ok_or("unweighted vertex")?;
        if vertex.co.x < 0.0 {
            mirrored += 1;
            assert_eq!(
                weights.get(&right),
                Some(&1.0),
                "mirror copy {:?}",
                vertex.co
            );
            assert_eq!(weights.get(&left), None, "mirror copy {:?}", vertex.co);
        } else {
            original += 1;
            assert_eq!(weights.get(&left), Some(&1.0), "original {:?}", vertex.co);
            assert_eq!(weights.get(&right), None, "original {:?}", vertex.co);
        }
    }
    assert_eq!((original, mirrored), (8, 8));
    Ok(())
}
