//! Mesh Sequence Cache reads every UV geom parameter into the layer its `sourceName` names.
//!
//! `two_layers.abc` is a one-sample static cache; `constant_topology.abc` animates points across
//! multiple logical samples while face counts and indices remain constant. `reordered_topology.abc`
//! reverses frame 2 face-corner connectivity without changing counts. Expected UVs are paired with
//! Blender-evaluated corner positions, not a presumed file or face order. Blender stores V up;
//! Potter stores V down.

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use glam::DVec3;
use potter_core::geom::{Edge, Mesh};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Blender 5.2.2 cache corners paired with evaluated positions, converted to Potter's V-down UVs.
type ExpectedCorner = ([f64; 3], [f64; 2]);
const BASE_CORNERS: [ExpectedCorner; 4] = [
    ([-1.0, -1.0, 0.0], [0.137, 0.777]),
    ([1.0, -1.0, 0.0], [0.619, 0.643]),
    ([1.0, 1.0, 0.0], [0.283, 0.119]),
    ([-1.0, 1.0, 0.0], [0.947, 0.431]),
];
const LIGHT_CORNERS: [ExpectedCorner; 4] = [
    ([-1.0, -1.0, 0.0], [0.731, 0.837]),
    ([1.0, -1.0, 0.0], [0.194, 0.093]),
    ([1.0, 1.0, 0.0], [0.853, 0.579]),
    ([-1.0, 1.0, 0.0], [0.367, 0.311]),
];
/// Float32 storage tolerance for the Alembic values.
const TOLERANCE: f64 = 1.0e-6;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mesh_cache_uv/two_layers.abc")
}

fn run_ok(command: &mut Command) -> TestResult<std::process::Output> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(output)
}

fn init(scene: &Path) -> TestResult {
    run_ok(pot().args(["init"]).arg(scene).arg("--json"))?;
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> TestResult {
    let document_path = scene.join("scene.json");
    let document: Value = serde_json::from_slice(&fs::read(&document_path)?)?;
    let batch = scene.with_extension("uv-layer-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": document["revision"],
            "operations": operations,
        }))?,
    )?;
    run_ok(
        pot()
            .args(["apply"])
            .arg(scene)
            .args(["--file"])
            .arg(batch)
            .arg("--json"),
    )?;
    Ok(())
}
/// Creates a plane `body`, packs an Alembic archive as the `cache` resource, and applies `operations`.
fn scene_with_archive(scene: &Path, archive: &Path, operations: Vec<Value>) -> TestResult {
    init(scene)?;
    let mut batch = vec![
        json!({"op":"node.create","id":"body","kind":"plane","params":{}}),
        json!({"op":"resource.pack","id":"cache","uri":archive,"kind":"alembic"}),
    ];
    batch.extend(operations);
    apply(scene, &Value::Array(batch))?;

    let archive_bytes = fs::read(archive)?;
    let archive_hash = potter_core::hash::sha256(&archive_bytes);
    let hash_hex = archive_hash.strip_prefix("sha256:").unwrap_or_default();
    let archive_name = archive
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("cache fixture filename is not UTF-8")?;
    let asset_uri = format!("assets/sha256/{hash_hex}/{archive_name}");
    let asset_path = scene.join(&asset_uri);
    fs::create_dir_all(asset_path.parent().ok_or("asset path has no parent")?)?;
    fs::write(&asset_path, &archive_bytes)?;
    let scene_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    document["resources"]["cache"]["uri"] = json!(asset_uri);
    document["resources"]["cache"]["packed"] = json!(false);
    if let Some(resource) = document["resources"]["cache"].as_object_mut() {
        resource.remove("bytes");
    }
    fs::write(scene_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn scene_with_cache(scene: &Path, operations: Vec<Value>) -> TestResult {
    scene_with_archive(scene, &fixture(), operations)
}
fn replace_body_mesh(scene: &Path, mesh: &Mesh) -> TestResult {
    let scene_path = scene.join("scene.json");
    let mut document: Value = serde_json::from_slice(&fs::read(&scene_path)?)?;
    let data_id = document["nodes"]["body"]["data"]
        .as_str()
        .ok_or("body mesh data block is missing")?
        .to_owned();
    document["data_blocks"][&data_id]["mesh"] = serde_json::to_value(mesh)?;
    fs::write(scene_path, serde_json::to_vec_pretty(&document)?)?;
    Ok(())
}

fn cache_modifier(read_data: &[&str]) -> Value {
    cache_modifier_for(read_data, "/Plane/Plane")
}

fn cache_modifier_for(read_data: &[&str], object_path: &str) -> Value {
    json!({
        "op":"modifier.create",
        "target":{"id":"body"},
        "id":"cache",
        "type":"mesh_sequence_cache",
        "params":{
            "resource":"cache",
            "object_path":object_path,
            "read_data":read_data,
        }
    })
}

fn layer_op(op: &str, name: &str) -> Value {
    json!({"op":op, "target":{"id":"body"}, "name":name})
}

/// A plane has Blender's default `UVMap`; these tests start from a receiver with no UV layers, as
/// they always did, so the layer limit and the layer order they check are the cache's own.
fn without_default_uv() -> Value {
    layer_op("uv.layer_delete", "UVMap")
}

/// The evaluated receiver mesh, UV store, per-face-corner positions, and warnings.
struct Evaluated {
    mesh: Mesh,
    store: Value,
    corner_positions: Vec<[f64; 3]>,
    warnings: Vec<Value>,
}

fn evaluate_body(scene: &Path) -> TestResult<Evaluated> {
    evaluate_body_at(scene, 1.0)
}

fn evaluate_body_at(scene: &Path, frame: f64) -> TestResult<Evaluated> {
    let project = potter_core::store::Project::open(scene)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(frame),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    )?;
    let mesh = snapshot
        .meshes
        .get(&potter_core::model::Id::new("body")?)
        .ok_or("evaluated body mesh is missing")?;
    let corner_positions = mesh
        .faces
        .iter()
        .flat_map(|face| &face.vertices)
        .map(|id| -> TestResult<_> {
            let vertex = mesh
                .vertices
                .iter()
                .find(|vertex| vertex.id == *id)
                .ok_or("evaluated face vertex is missing")?;
            Ok(vertex.co.to_array())
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(Evaluated {
        mesh: mesh.clone(),
        store: mesh
            .attributes
            .get("uv_layers")
            .cloned()
            .unwrap_or(Value::Null),
        corner_positions,
        warnings: snapshot
            .warnings
            .iter()
            .map(|warning| warning.details.clone())
            .collect(),
    })
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    match value
        .as_array()
        .ok_or("UV corner is not an array")?
        .as_slice()
    {
        [u, v] => Ok([
            u.as_f64().ok_or("UV coordinate is not a number")?,
            v.as_f64().ok_or("UV coordinate is not a number")?,
        ]),
        _ => Err("UV corner is not a pair".into()),
    }
}

/// The layers of a store in order, as `(name, corners)`. Each layer must hold exactly one face.
fn layers(store: &Value) -> TestResult<Vec<(String, Vec<[f64; 2]>)>> {
    let mut output = Vec::new();
    for layer in store["layers"]
        .as_array()
        .ok_or("uv_layers has no layers")?
    {
        let name = layer["name"]
            .as_str()
            .ok_or("UV layer has no name")?
            .to_owned();
        let [face] = layer["faces"]
            .as_array()
            .ok_or("UV layer has no faces")?
            .as_slice()
        else {
            return Err(format!("UV layer `{name}` must hold exactly one face").into());
        };
        let corners = face["uv"]
            .as_array()
            .ok_or("UV face has no corners")?
            .iter()
            .map(pair)
            .collect::<TestResult<Vec<_>>>()?;
        output.push((name, corners));
    }
    Ok(output)
}

fn layer_names(layers: &[(String, Vec<[f64; 2]>)]) -> Vec<&str> {
    layers.iter().map(|(name, _)| name.as_str()).collect()
}

fn layer<'a>(layers: &'a [(String, Vec<[f64; 2]>)], name: &str) -> TestResult<&'a [[f64; 2]]> {
    layers
        .iter()
        .find(|(layer, _)| layer == name)
        .map(|(_, corners)| corners.as_slice())
        .ok_or_else(|| format!("layer `{name}` is missing").into())
}

fn assert_corners(actual: &[[f64; 2]], expected: &[[f64; 2]], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: corner count");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for axis in 0..2 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= TOLERANCE,
                "{label}: corner {index} axis {axis} is {}, expected {}",
                actual[axis],
                expected[axis]
            );
        }
    }
}
fn assert_corner_associations(
    positions: &[[f64; 3]],
    uvs: &[[f64; 2]],
    expected: &[ExpectedCorner],
    label: &str,
) {
    assert_eq!(positions.len(), uvs.len(), "{label}: corner count");
    assert_eq!(
        positions.len(),
        expected.len(),
        "{label}: oracle corner count"
    );
    let mut unmatched = expected.to_vec();
    for (position, uv) in positions.iter().zip(uvs) {
        let Some(index) = unmatched
            .iter()
            .position(|(expected_position, expected_uv)| {
                expected_position
                    .iter()
                    .zip(position)
                    .all(|(expected, actual)| (expected - actual).abs() <= TOLERANCE)
                    && expected_uv
                        .iter()
                        .zip(uv)
                        .all(|(expected, actual)| (expected - actual).abs() <= TOLERANCE)
            })
        else {
            panic!(
                "{label}: no Blender cache UV {uv:?} at corner position {position:?}; actual corner sequence: {:?}",
                positions.iter().zip(uvs).collect::<Vec<_>>()
            );
        };
        unmatched.swap_remove(index);
    }
    assert!(
        unmatched.is_empty(),
        "{label}: unmatched Blender cache corners"
    );
}

#[test]
fn cache_restores_every_uv_parameter_into_the_layer_it_names() -> TestResult {
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    scene_with_cache(&scene, vec![without_default_uv()])?;
    apply(&scene, &json!([cache_modifier(&["VERT", "POLY", "UV"])]))?;

    let evaluated = evaluate_body(&scene)?;
    let layers = layers(&evaluated.store)?;
    assert_eq!(layer_names(&layers), ["Light", "Base"]);
    assert_corner_associations(
        &evaluated.corner_positions,
        layer(&layers, "Light")?,
        &LIGHT_CORNERS,
        "Light",
    );
    assert_corner_associations(
        &evaluated.corner_positions,
        layer(&layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    // The first cached layer, in file order, is the active and render layer, as in Blender.
    assert_eq!(evaluated.store["active"], json!("Light"));
    assert_eq!(evaluated.store["render"], json!("Light"));
    assert!(evaluated.warnings.is_empty(), "{:?}", evaluated.warnings);
    Ok(())
}

#[test]
fn cache_keeps_mesh_layers_it_does_not_name_and_keeps_selectors() -> TestResult {
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    scene_with_cache(
        &scene,
        vec![
            without_default_uv(),
            layer_op("uv.layer_create", "Base"),
            layer_op("uv.layer_create", "Extra"),
            layer_op("uv.layer_set_active", "Extra"),
            layer_op("uv.layer_set_render", "Base"),
        ],
    )?;
    let before = evaluate_body(&scene)?;
    apply(&scene, &json!([cache_modifier(&["VERT", "POLY", "UV"])]))?;
    let after = evaluate_body(&scene)?;

    let before_layers = layers(&before.store)?;
    let after_layers = layers(&after.store)?;
    assert_eq!(layer_names(&before_layers), ["Base", "Extra"]);
    assert_eq!(layer_names(&after_layers), ["Base", "Extra", "Light"]);
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Light")?,
        &LIGHT_CORNERS,
        "Light",
    );
    assert_corners(
        layer(&after_layers, "Extra")?,
        layer(&before_layers, "Extra")?,
        "Extra",
    );
    assert_eq!(after.store["active"], json!("Extra"));
    assert_eq!(after.store["render"], json!("Base"));
    assert!(after.warnings.is_empty(), "{:?}", after.warnings);
    Ok(())
}

#[test]
fn cache_writes_named_layers_past_the_first_and_reports_the_one_without_room() -> TestResult {
    let names = ["L0", "Base", "L2", "L3", "L4", "L5", "L6", "L7"];
    let temporary = tempdir()?;
    let scene = temporary.path().join("scene.pot");
    let mut operations: Vec<Value> = vec![without_default_uv()];
    operations.extend(names.iter().map(|name| layer_op("uv.layer_create", name)));
    scene_with_cache(&scene, operations)?;
    let before = evaluate_body(&scene)?;
    // Without topology reads the mesh keeps its own layers, so the limit applies to the store.
    apply(&scene, &json!([cache_modifier(&["UV"])]))?;
    let after = evaluate_body(&scene)?;

    let before_layers = layers(&before.store)?;
    let after_layers = layers(&after.store)?;
    assert_eq!(layer_names(&after_layers), names);
    // `Light` is the canonical cached layer and has no room: it is reported, not written.
    // `Base` is the second layer of the mesh and is overwritten by name, not the first layer.
    assert_corner_associations(
        &after.corner_positions,
        layer(&after_layers, "Base")?,
        &BASE_CORNERS,
        "Base",
    );
    for name in names.iter().filter(|name| **name != "Base") {
        assert_corners(
            layer(&after_layers, name)?,
            layer(&before_layers, name)?,
            name,
        );
    }
    assert_eq!(after.store["active"], json!("L0"));
    assert_eq!(after.store["render"], json!("L0"));
    let [warning] = after.warnings.as_slice() else {
        panic!("expected one warning, found {:?}", after.warnings);
    };
    assert_eq!(warning["feature_id"], json!("uv.layer_limit"));
    assert_eq!(warning["status"], json!("dropped"));
    assert_eq!(warning["layer"], json!("Light"));
    assert_eq!(warning["limit"], json!(8));
    Ok(())
}

#[test]
fn static_vertex_only_cache_preserves_receiver_faces_and_uv_associations() -> TestResult {
    let directory = tempdir()?;
    let source_scene = directory.path().join("source.pot");
    scene_with_cache(&source_scene, vec![cache_modifier(&["VERT", "POLY", "UV"])])?;
    let project = potter_core::store::Project::open(&source_scene)?;
    let snapshot = potter_core::eval::Snapshot::evaluate_with_cache(
        project.doc(),
        &potter_core::eval::EvaluationContext {
            frame: Some(1.0),
            ..potter_core::eval::EvaluationContext::default()
        },
        Some(project.path()),
        Some(project.cache_directory()),
    )?;
    let mut receiver = snapshot
        .meshes
        .get(&potter_core::model::Id::new("body")?)
        .ok_or("cache source mesh is missing")?
        .clone();
    let source_faces: Vec<Vec<u32>> = receiver
        .faces
        .iter()
        .map(|face| face.vertices.clone())
        .collect();
    let source_vertex_count = receiver.vertices.len();
    let source_face_count = receiver.faces.len();
    let source_corner_count = receiver
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    for face in &mut receiver.faces {
        face.id = face.id.checked_add(100).ok_or("face ID overflow")?;
        face.material_index = 1;
        face.vertices.rotate_left(1);
    }
    receiver.next_id.face = receiver
        .faces
        .iter()
        .map(|face| face.id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("face ID counter overflow")?;
    let mut loose_edge_vertices = None;
    'candidate: for (index, first) in receiver.vertices.iter().enumerate() {
        for second in receiver.vertices.iter().skip(index + 1) {
            let endpoints = [first.id.min(second.id), first.id.max(second.id)];
            if !receiver.edges.iter().any(|edge| {
                edge.vertices == endpoints || edge.vertices == [endpoints[1], endpoints[0]]
            }) {
                loose_edge_vertices = Some(endpoints);
                break 'candidate;
            }
        }
    }
    let loose_edge_vertices = loose_edge_vertices.ok_or("receiver mesh has no loose-edge pair")?;
    let loose_edge_id = receiver.next_id.edge;
    receiver.edges.push(Edge {
        id: loose_edge_id,
        vertices: loose_edge_vertices,
    });
    receiver.next_id.edge = loose_edge_id
        .checked_add(1)
        .ok_or("edge ID counter overflow")?;
    let mut probe_values = serde_json::Map::new();
    for vertex in &receiver.vertices {
        probe_values.insert(format!("v{}", vertex.id), json!(42.0));
    }
    receiver.attributes.insert(
        "cache_receiver_probe".to_owned(),
        json!({"domain":"point","type":"float","values":probe_values}),
    );
    let receiver_faces: Vec<Vec<u32>> = receiver
        .faces
        .iter()
        .map(|face| face.vertices.clone())
        .collect();
    assert_eq!(receiver.vertices.len(), source_vertex_count);
    assert_eq!(receiver.faces.len(), source_face_count);
    assert_eq!(
        receiver
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>(),
        source_corner_count
    );
    assert_ne!(
        receiver_faces, source_faces,
        "test receiver must differ from the cache's face-corner connectivity"
    );

    let corner_counts: Vec<usize> = receiver
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .collect();
    let receiver_face_ids: Vec<u32> = receiver.faces.iter().map(|face| face.id).collect();
    let uv_store = receiver
        .attributes
        .get_mut("uv_layers")
        .ok_or("cache source mesh has no UV store")?;
    let uv_layers = uv_store
        .get_mut("layers")
        .and_then(Value::as_array_mut)
        .ok_or("cache source UV store has no layers")?;
    for uv_layer in uv_layers {
        let rows = uv_layer
            .get_mut("faces")
            .and_then(Value::as_array_mut)
            .ok_or("cache source UV layer has no face rows")?;
        assert_eq!(rows.len(), corner_counts.len());
        for (face_index, (row, corner_count)) in rows.iter_mut().zip(&corner_counts).enumerate() {
            row["face_id"] = json!(receiver_face_ids[face_index]);
            let uv = row
                .get_mut("uv")
                .and_then(Value::as_array_mut)
                .ok_or("cache source UV row has no corners")?;
            assert_eq!(uv.len(), *corner_count);
            uv.rotate_left(1);
            if let Some(pinned) = row.get_mut("pinned") {
                let pinned = pinned
                    .as_array_mut()
                    .ok_or("cache source UV pins are not an array")?;
                assert_eq!(pinned.len(), *corner_count);
                pinned.rotate_left(1);
            }
        }
    }
    let input_layers = layers(
        receiver
            .attributes
            .get("uv_layers")
            .ok_or("receiver UV store is missing")?,
    )?;
    let expected_corner_positions = receiver
        .faces
        .iter()
        .flat_map(|face| &face.vertices)
        .map(|id| -> TestResult<[f64; 3]> {
            receiver
                .vertices
                .iter()
                .find(|vertex| vertex.id == *id)
                .map(|vertex| vertex.co.to_array())
                .ok_or_else(|| "receiver face references a missing vertex".into())
        })
        .collect::<TestResult<Vec<_>>>()?;

    let receiver_scene = directory.path().join("receiver.pot");
    scene_with_cache(&receiver_scene, vec![cache_modifier(&["VERT"])])?;
    apply(
        &receiver_scene,
        &json!([
            {"op":"material.create","id":"slot_zero","name":"Slot zero"},
            {"op":"material.create","id":"slot_one","name":"Slot one"},
            {"op":"node.update","target":{"id":"body"},"set":{"materials":["slot_zero","slot_one"]}}
        ]),
    )?;
    replace_body_mesh(&receiver_scene, &receiver)?;
    let evaluated = evaluate_body(&receiver_scene)?;
    assert_eq!(evaluated.corner_positions, expected_corner_positions);
    assert_eq!(evaluated.mesh.faces, receiver.faces);
    assert_eq!(evaluated.mesh.edges, receiver.edges);
    assert_eq!(
        evaluated
            .mesh
            .vertices
            .iter()
            .map(|vertex| vertex.id)
            .collect::<Vec<_>>(),
        receiver
            .vertices
            .iter()
            .map(|vertex| vertex.id)
            .collect::<Vec<_>>()
    );
    assert_eq!(evaluated.mesh.next_id, receiver.next_id);
    assert!(
        evaluated
            .mesh
            .faces
            .iter()
            .all(|face| face.material_index == 1)
    );
    let probe = evaluated
        .mesh
        .attributes
        .get("cache_receiver_probe")
        .ok_or("vertex-only cache dropped a receiver custom attribute")?;
    assert_eq!(probe["domain"], json!("point"));
    let probe_values = probe["values"]
        .as_object()
        .ok_or("receiver custom attribute has no values")?;
    assert_eq!(probe_values.len(), receiver.vertices.len());
    assert!(
        probe_values
            .values()
            .all(|value| value.as_f64() == Some(42.0))
    );
    let output_layers = layers(&evaluated.store)?;
    assert_eq!(layer_names(&output_layers), layer_names(&input_layers));
    for (name, input_uvs) in &input_layers {
        assert_corners(layer(&output_layers, name)?, input_uvs, name);
        let expected = expected_corner_positions
            .iter()
            .copied()
            .zip(input_uvs.iter().copied())
            .collect::<Vec<_>>();
        assert_corner_associations(
            &evaluated.corner_positions,
            layer(&output_layers, name)?,
            &expected,
            name,
        );
    }
    assert!(evaluated.warnings.is_empty(), "{:?}", evaluated.warnings);
    Ok(())
}

#[test]
fn constant_multisample_topology_rebuilds_vertex_only_receiver_connectivity() -> TestResult {
    let directory = tempdir()?;
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mesh_cache_uv");
    let archive = fixture_dir.join("constant_topology.abc");
    let object_path = "/ConstantTopology/ConstantTopology";
    let source_scene = directory.path().join("constant-source.pot");
    scene_with_archive(
        &source_scene,
        &archive,
        vec![cache_modifier_for(&["VERT", "POLY", "UV"], object_path)],
    )?;
    let source = evaluate_body(&source_scene)?;
    let source_faces: Vec<Vec<u32>> = source
        .mesh
        .faces
        .iter()
        .map(|face| face.vertices.clone())
        .collect();
    let source_layers = layers(&source.store)?;
    let mut receiver = source.mesh.clone();
    for face in &mut receiver.faces {
        face.vertices.rotate_left(1);
    }
    let receiver_faces: Vec<Vec<u32>> = receiver
        .faces
        .iter()
        .map(|face| face.vertices.clone())
        .collect();
    assert_ne!(receiver_faces, source_faces);
    let corner_counts: Vec<usize> = receiver
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .collect();
    let uv_store = receiver
        .attributes
        .get_mut("uv_layers")
        .ok_or("cache source mesh has no UV store")?;
    let uv_layers = uv_store
        .get_mut("layers")
        .and_then(Value::as_array_mut)
        .ok_or("cache source UV store has no layers")?;
    for uv_layer in uv_layers {
        let rows = uv_layer
            .get_mut("faces")
            .and_then(Value::as_array_mut)
            .ok_or("cache source UV layer has no face rows")?;
        assert_eq!(rows.len(), corner_counts.len());
        for (row, corner_count) in rows.iter_mut().zip(&corner_counts) {
            let uv = row
                .get_mut("uv")
                .and_then(Value::as_array_mut)
                .ok_or("cache source UV row has no corners")?;
            assert_eq!(uv.len(), *corner_count);
            uv.rotate_left(1);
            if let Some(pinned) = row.get_mut("pinned") {
                let pinned = pinned
                    .as_array_mut()
                    .ok_or("cache source UV pins are not an array")?;
                assert_eq!(pinned.len(), *corner_count);
                pinned.rotate_left(1);
            }
        }
    }

    let receiver_scene = directory.path().join("constant-receiver.pot");
    scene_with_archive(
        &receiver_scene,
        &archive,
        vec![cache_modifier_for(&["VERT"], object_path)],
    )?;
    replace_body_mesh(&receiver_scene, &receiver)?;
    let evaluated = evaluate_body(&receiver_scene)?;
    let evaluated_faces: Vec<Vec<u32>> = evaluated
        .mesh
        .faces
        .iter()
        .map(|face| face.vertices.clone())
        .collect();
    assert_eq!(evaluated_faces, source_faces);
    assert_ne!(evaluated_faces, receiver_faces);
    assert_eq!(evaluated.corner_positions, source.corner_positions);
    let output_layers = layers(&evaluated.store)?;
    assert_eq!(layer_names(&output_layers), layer_names(&source_layers));
    for (name, source_uvs) in &source_layers {
        let output_uvs = layer(&output_layers, name)?;
        assert_corners(output_uvs, source_uvs, name);
        let expected = source
            .corner_positions
            .iter()
            .copied()
            .zip(source_uvs.iter().copied())
            .collect::<Vec<_>>();
        assert_corner_associations(&evaluated.corner_positions, output_uvs, &expected, name);
    }
    assert!(evaluated.warnings.is_empty(), "{:?}", evaluated.warnings);
    Ok(())
}

#[test]
fn topology_change_reloads_uvs_for_same_count_corner_reorder() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("reordered.pot");
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mesh_cache_uv");
    let archive = fixture_dir.join("reordered_topology.abc");
    let oracle: Value =
        serde_json::from_slice(&fs::read(fixture_dir.join("reordered_topology.json"))?)?;
    let frames = oracle["frames"]
        .as_array()
        .ok_or("topology UV oracle frames are missing")?;
    let frame_one = frames
        .iter()
        .find(|frame| frame["frame"] == 1)
        .ok_or("frame-one topology UV oracle is missing")?;
    let frame_two = frames
        .iter()
        .find(|frame| frame["frame"] == 2)
        .ok_or("frame-two topology UV oracle is missing")?;
    let faces_one: Vec<Vec<usize>> = serde_json::from_value(frame_one["faces"].clone())?;
    let faces_two: Vec<Vec<usize>> = serde_json::from_value(frame_two["faces"].clone())?;
    assert_eq!(
        faces_one.iter().map(Vec::len).collect::<Vec<_>>(),
        faces_two.iter().map(Vec::len).collect::<Vec<_>>()
    );
    assert_ne!(
        faces_one, faces_two,
        "the animated archive must reorder face corners without changing counts"
    );

    scene_with_archive(&scene, &archive, Vec::new())?;
    let positions: Vec<[f64; 3]> = serde_json::from_value(frame_one["positions"].clone())?;
    let mut receiver = Mesh::from_positions_and_faces(
        positions.iter().copied().map(DVec3::from_array).collect(),
        faces_one.clone(),
    )?;
    let stale_uv = [[0.91, 0.13], [0.82, 0.24], [0.73, 0.35], [0.64, 0.46]];
    receiver.attributes.insert(
        "uv_layers".to_owned(),
        json!({
            "layers":[{"name":"UVMap","faces":[{"face_id":receiver.faces[0].id,"uv":stale_uv}]}],
            "active":"UVMap",
            "render":"UVMap"
        }),
    );
    replace_body_mesh(&scene, &receiver)?;
    apply(
        &scene,
        &json!([cache_modifier_for(&["VERT"], "/Reordered/Reordered")]),
    )?;

    let before_reorder = evaluate_body_at(&scene, 1.0)?;
    let before_layers = layers(&before_reorder.store)?;
    assert_corners(
        layer(&before_layers, "UVMap")?,
        &stale_uv,
        "unchanged topology keeps input UVs",
    );

    let after_reorder = evaluate_body_at(&scene, 2.0)?;
    let after_layers = layers(&after_reorder.store)?;
    let cached_uvs = layer(&after_layers, "UVMap")?;
    let positions: Vec<[f64; 3]> = serde_json::from_value(frame_two["positions"].clone())?;
    let faces: Vec<Vec<usize>> = serde_json::from_value(frame_two["faces"].clone())?;
    let uv_values = frame_two["uvs"]
        .as_array()
        .ok_or("frame-two UV oracle values are missing")?;
    let mut expected = Vec::with_capacity(uv_values.len());
    let mut corner = 0;
    for face in faces {
        for vertex in face {
            let [u, v] = pair(
                uv_values
                    .get(corner)
                    .ok_or("frame-two UV oracle row is truncated")?,
            )?;
            expected.push((positions[vertex], [u, 1.0 - v]));
            corner += 1;
        }
    }
    assert_eq!(corner, uv_values.len());
    assert_corner_associations(
        &after_reorder.corner_positions,
        cached_uvs,
        &expected,
        "reordered cache UVs",
    );
    Ok(())
}
