//! PR3 UV propagation through mesh-rebuilding modifiers. Each oracle case evaluates a scene whose
//! corner UVs must match Blender 5.2.2 at 1e-6. The Blender corners, and the Potter input that
//! produced them, live in `tests/fixtures/pr3_uv/`, so these assertions need no Blender install.

use std::{collections::BTreeMap, error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::{BoxParams, Mesh},
    model::{DataBlock, Id, Modifier, ModifierRuntime, Node, SceneDoc, Transform},
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "6f1c2a4e-3b5d-4c8e-9a7f-1d2e3f4a5b6c";
const UV_TOLERANCE: f64 = 1.0e-6;
/// Blender writes corner keys at 5 decimals and some subdivision positions differ at 1e-5, so
/// corners are paired by position within this bound and their UVs are then compared exactly.
const POSITION_TOLERANCE: f64 = 1.0e-4;

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

fn fixture(name: &str) -> TestResult<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pr3_uv")
        .join(format!("{name}.json"));
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

/// Names of every fixture in `tests/fixtures/pr3_uv/`, sorted.
fn fixture_names() -> TestResult<Vec<String>> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pr3_uv");
    let mut names = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or("fixture file name")?;
            names.push(stem.to_owned());
        }
    }
    names.sort();
    Ok(names)
}

fn point(value: &Value) -> TestResult<DVec3> {
    let coords = value
        .as_array()
        .ok_or("point must be an array")?
        .iter()
        .map(|coordinate| -> TestResult<f64> {
            Ok(coordinate.as_f64().ok_or("coordinate must be a number")?)
        })
        .collect::<TestResult<Vec<f64>>>()?;
    match coords.as_slice() {
        &[x, y, z] => Ok(DVec3::new(x, y, z)),
        _ => Err("point must have three coordinates".into()),
    }
}

fn uv(value: &Value) -> TestResult<[f64; 2]> {
    match value.as_array().map(Vec::as_slice) {
        Some([u, v]) => Ok([
            u.as_f64().ok_or("UV must be a number")?,
            v.as_f64().ok_or("UV must be a number")?,
        ]),
        _ => Err("UV must be [u, v]".into()),
    }
}

/// Potter mesh from fixture input: positions, faces, and the per-corner `blender_uv_layers`.
fn input_mesh(input: &Value) -> TestResult<Mesh> {
    let positions = input["vertices"]
        .as_array()
        .ok_or("vertices missing")?
        .iter()
        .map(point)
        .collect::<TestResult<Vec<_>>>()?;
    let polygons = input["faces"]
        .as_array()
        .ok_or("faces missing")?
        .iter()
        .map(|face| {
            face.as_array()
                .ok_or("face must be an array")?
                .iter()
                .map(|index| -> TestResult<usize> {
                    Ok(usize::try_from(index.as_u64().ok_or("index")?)?)
                })
                .collect::<TestResult<Vec<usize>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;
    let mut mesh = Mesh::from_positions_and_faces(positions, polygons)?;
    mesh.attributes
        .insert("blender_uv_layers".to_owned(), input["uv_layers"].clone());
    // Loose edges (no face loop) are Screw's profile edges; `edges` lists them as vertex index pairs.
    for edge in input
        .get("edges")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let [first, second] = edge.as_array().ok_or("edge must be an array")?.as_slice() else {
            return Err("edge must have two vertex indices".into());
        };
        let mut ids = [0_u32; 2];
        for (id, index) in ids.iter_mut().zip([first, second]) {
            let index = usize::try_from(index.as_u64().ok_or("edge index must be a number")?)?;
            *id = mesh
                .vertices
                .get(index)
                .ok_or("edge index out of range")?
                .id;
        }
        mesh.insert_edge(ids)?;
    }
    Ok(mesh)
}

/// Moves the layers `moved` selects from `blender_uv_layers` into `uv_map` entries, one per face. A
/// `UVMap` entry is layer-less, as the evaluator reads it; any other layer names its `layer`.
fn move_to_uv_map(mesh: &mut Mesh, moved: impl Fn(&str) -> bool) -> TestResult {
    let layers = mesh
        .attributes
        .remove("blender_uv_layers")
        .and_then(|layers| layers.as_array().cloned())
        .ok_or("UV layers missing")?;
    // A layer without corners has no per-face entry to carry it, so it stays in `blender_uv_layers`.
    let (moved_layers, kept): (Vec<Value>, Vec<Value>) = layers.into_iter().partition(|layer| {
        let has_corners = layer["values"]
            .as_array()
            .is_some_and(|values| !values.is_empty());
        has_corners && moved(layer["name"].as_str().unwrap_or_default())
    });
    let mut entries = Vec::new();
    for layer in &moved_layers {
        let name = layer["name"].as_str().ok_or("layer name missing")?;
        let values = layer["values"].as_array().ok_or("layer values missing")?;
        let mut corner = 0;
        for face in &mesh.faces {
            let end = corner + face.vertices.len();
            let mut entry = json!({"face_id": face.id, "uv": &values[corner..end]});
            if name != "UVMap" {
                entry["layer"] = json!(name);
            }
            entries.push(entry);
            corner = end;
        }
    }
    mesh.attributes
        .insert("blender_uv_layers".to_owned(), Value::Array(kept));
    mesh.attributes
        .insert("uv_map".to_owned(), Value::Array(entries));
    Ok(())
}

fn modifier(id: impl Into<String>, modifier_type: &str, params: &Value) -> TestResult<Modifier> {
    let id = id.into();
    Ok(Modifier {
        id: Id::new(&id)?,
        modifier_type: modifier_type.to_owned(),
        name: id,
        enabled: true,
        params: params
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

/// Scene with a `subject` node carrying `modifiers`, plus `objects` (name, mesh, translation) that
/// the subject's modifiers reference by name, such as array cap objects.
fn scene_doc(
    subject: Mesh,
    modifiers: Vec<Modifier>,
    objects: Vec<(String, Mesh, [f64; 3])>,
) -> TestResult<SceneDoc> {
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    let root = Id::new("collection_root")?;
    let nodes = std::iter::once(("subject".to_owned(), subject, [0.0; 3], modifiers)).chain(
        objects
            .into_iter()
            .map(|(name, mesh, translation)| (name, mesh, translation, Vec::new())),
    );
    for (name, mesh, translation, modifiers) in nodes {
        let node_id = Id::new(&name)?;
        let data_id = Id::new(format!("data_{name}"))?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name,
                kind: "mesh".to_owned(),
                transform: Transform {
                    translation,
                    ..Transform::default()
                },
                modifiers,
                data: Some(data_id),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&root)
            .ok_or("root collection missing")?
            .objects
            .push(node_id);
    }
    doc.validate()?;
    Ok(doc)
}

/// How a fixture's UV layers are stored in a Potter input.
#[derive(Clone, Copy)]
enum UvStorage {
    /// `blender_uv_layers` only, as the fixtures are written.
    Blender,
    /// `UVMap` moved into layer-less `uv_map` entries; other layers stay in `blender_uv_layers`.
    LayerlessUvMap,
    /// Every layer moved into `uv_map` entries that name their layer.
    AllUvMap,
    /// `blender_uv_layers`, plus a UV-named `FLOAT2` `CORNER` `blender_attributes` copy of each layer.
    AttributeCopies,
}

/// Stores `mesh`'s UV layers as `storage` says.
fn store_uv_layers(mesh: &mut Mesh, storage: UvStorage) -> TestResult {
    match storage {
        UvStorage::Blender => {}
        UvStorage::LayerlessUvMap => move_to_uv_map(mesh, |name| name == "UVMap")?,
        UvStorage::AllUvMap => move_to_uv_map(mesh, |_| true)?,
        UvStorage::AttributeCopies => {
            let copies: Vec<Value> = mesh
                .attributes
                .get("blender_uv_layers")
                .and_then(Value::as_array)
                .ok_or("UV layers missing")?
                .iter()
                .map(|layer| {
                    json!({
                        "name": layer["name"],
                        "domain": "CORNER",
                        "data_type": "FLOAT2",
                        "values": layer["values"],
                    })
                })
                .collect();
            let mut attributes = mesh
                .attributes
                .get("blender_attributes")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            attributes.extend(copies);
            mesh.attributes
                .insert("blender_attributes".to_owned(), Value::Array(attributes));
        }
    }
    Ok(())
}

/// Scene document for a fixture's input: the subject stack plus any cap objects, with every
/// object's UV layers stored as `storage` says.
fn fixture_scene(input: &Value, storage: UvStorage) -> TestResult<SceneDoc> {
    let modifiers = input["modifiers"]
        .as_array()
        .ok_or("modifiers missing")?
        .iter()
        .enumerate()
        .map(|(index, stack)| {
            modifier(
                format!("modifier_{index}"),
                stack["type"].as_str().ok_or("modifier type")?,
                &stack["params"],
            )
        })
        .collect::<TestResult<Vec<_>>>()?;
    let objects = input["caps"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, cap)| -> TestResult<_> {
            let mut mesh = input_mesh(cap)?;
            store_uv_layers(&mut mesh, storage)?;
            let loc = &cap["loc"];
            Ok((
                name.clone(),
                mesh,
                [
                    loc[0].as_f64().ok_or("loc")?,
                    loc[1].as_f64().ok_or("loc")?,
                    loc[2].as_f64().ok_or("loc")?,
                ],
            ))
        })
        .collect::<TestResult<Vec<_>>>()?;
    let mut subject = input_mesh(input)?;
    store_uv_layers(&mut subject, storage)?;
    scene_doc(subject, modifiers, objects)
}

fn evaluate(doc: &SceneDoc) -> TestResult<Snapshot> {
    Ok(Snapshot::evaluate(doc, &EvaluationContext::default())?)
}

fn subject_mesh(snapshot: &Snapshot) -> TestResult<&Mesh> {
    Ok(&snapshot.meshes[&Id::new("subject")?])
}

#[derive(Debug)]
struct Corner {
    centroid: DVec3,
    point: DVec3,
    uv: Vec<[f64; 2]>,
}

/// Corners of an evaluated mesh in face order, one entry per loop, with the values of `layers`.
fn corners(mesh: &Mesh, layers: &[String]) -> TestResult<Vec<Corner>> {
    let values = layers
        .iter()
        .map(|name| layer_uvs(mesh, name))
        .collect::<TestResult<Vec<_>>>()?;
    let positions: BTreeMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut output = Vec::new();
    let mut loop_index = 0;
    for face in &mesh.faces {
        let points = face
            .vertices
            .iter()
            .map(|id| positions.get(id).copied().ok_or("face vertex missing"))
            .collect::<Result<Vec<_>, _>>()?;
        let centroid = points.iter().sum::<DVec3>() / f64::from(u32::try_from(points.len())?);
        for (corner, point) in points.iter().enumerate() {
            let uvs = values
                .iter()
                .map(|layer| {
                    layer
                        .get(loop_index + corner)
                        .copied()
                        .ok_or("loop UV missing")
                })
                .collect::<Result<Vec<_>, _>>()?;
            output.push(Corner {
                centroid,
                point: *point,
                uv: uvs,
            });
        }
        loop_index += points.len();
    }
    Ok(output)
}

fn uv_layer_names(mesh: &Mesh) -> Vec<String> {
    mesh.attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| layer["name"].as_str().map(str::to_owned))
        .collect()
}

/// The `uv_map` entries of `mesh`.
fn uv_map_entries(mesh: &Mesh) -> &[Value] {
    mesh.attributes
        .get("uv_map")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
}

/// Layer of a `uv_map` entry; an entry without a `layer` belongs to `UVMap`.
fn uv_map_layer(entry: &Value) -> &str {
    entry["layer"].as_str().unwrap_or("UVMap")
}

/// Names of the UV layers `mesh` stores: those `blender_uv_layers` lists, then `uv_map` layers it
/// does not list.
fn stored_uv_layer_names(mesh: &Mesh) -> Vec<String> {
    let mut names = uv_layer_names(mesh);
    for entry in uv_map_entries(mesh) {
        let name = uv_map_layer(entry);
        if !names.iter().any(|known| known == name) {
            names.push(name.to_owned());
        }
    }
    names
}

/// Values of UV layer `name` in loop order, read as the evaluator reads them: `blender_uv_layers`
/// when it lists the layer, otherwise the layer's `uv_map` entry for each face.
fn layer_uvs(mesh: &Mesh, name: &str) -> TestResult<Vec<[f64; 2]>> {
    let listed = mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|layer| layer["name"].as_str() == Some(name));
    if let Some(layer) = listed {
        return layer["values"]
            .as_array()
            .ok_or("layer values missing")?
            .iter()
            .map(uv)
            .collect();
    }
    let entries = uv_map_entries(mesh);
    let mut output = Vec::new();
    for face in &mesh.faces {
        let entry = entries
            .iter()
            .find(|entry| {
                entry["face_id"].as_u64() == Some(u64::from(face.id)) && uv_map_layer(entry) == name
            })
            .ok_or_else(|| format!("no `{name}` uv_map entry for face {}", face.id))?;
        let row = entry["uv"].as_array().ok_or("uv_map entry without uv")?;
        if row.len() != face.vertices.len() {
            return Err(format!(
                "`{name}` uv_map entry of face {} has the wrong corner count",
                face.id
            )
            .into());
        }
        for value in row {
            output.push(uv(value)?);
        }
    }
    Ok(output)
}

/// Layer names a fixture lists, in fixture order.
fn fixture_layers(fixture: &Value) -> TestResult<Vec<String>> {
    let layers = fixture["layers"].as_array().ok_or("layers missing")?;
    layers
        .iter()
        .map(|name| -> TestResult<String> { Ok(name.as_str().ok_or("layer name")?.to_owned()) })
        .collect()
}

/// Pairs oracle corners with Potter corners at the same face centroid and corner position, and
/// checks each layer's UV within `UV_TOLERANCE`. Repeated positions pair by closest UVs. With
/// `partial`, oracle corners without a Potter counterpart are skipped; otherwise every oracle corner
/// must match and the corner counts must be equal. Returns the number of matched oracle corners.
fn matched_oracle_corners(mesh: &Mesh, fixture: &Value, partial: bool) -> TestResult<usize> {
    let layers = fixture_layers(fixture)?;
    let mut stored = stored_uv_layer_names(mesh);
    stored.sort();
    let mut wanted = layers.clone();
    wanted.sort();
    assert_eq!(stored, wanted, "UV layer names");
    let actual = corners(mesh, &layers)?;
    let rows = fixture["corners"].as_array().ok_or("corners missing")?;
    if !partial {
        assert_eq!(actual.len(), rows.len(), "Potter corner count");
    }
    let mut used = vec![false; actual.len()];
    let mut matched = 0;
    for row in rows {
        let values = row
            .as_array()
            .ok_or("corner row")?
            .iter()
            .map(|value| -> TestResult<f64> { Ok(value.as_f64().ok_or("corner value")?) })
            .collect::<TestResult<Vec<f64>>>()?;
        let (centroid, point) = (
            DVec3::new(values[0], values[1], values[2]),
            DVec3::new(values[3], values[4], values[5]),
        );
        let expected: Vec<[f64; 2]> = values[6..]
            .chunks(2)
            .map(|pair| [pair[0], pair[1]])
            .collect();
        // Candidates: (index, position distance, largest UV gap), closest position first.
        let best = actual
            .iter()
            .enumerate()
            .filter(|(index, _)| !used[*index])
            .map(|(index, corner)| {
                let distance =
                    (corner.centroid - centroid).length() + (corner.point - point).length();
                let gap = corner
                    .uv
                    .iter()
                    .zip(&expected)
                    .flat_map(|(got, want)| (0..2).map(move |axis| (got[axis] - want[axis]).abs()))
                    .fold(0.0_f64, f64::max);
                (index, distance, gap)
            })
            .filter(|(_, distance, _)| *distance < POSITION_TOLERANCE)
            .min_by(|left, right| left.1.total_cmp(&right.1).then(left.2.total_cmp(&right.2)));
        let Some((index, _, _)) = best else {
            assert!(
                partial,
                "no Potter corner at face centroid {centroid:?}, corner {point:?}"
            );
            continue;
        };
        used[index] = true;
        matched += 1;
        for (layer, want) in expected.iter().enumerate() {
            let got = actual[index].uv[layer];
            for axis in 0..2 {
                assert!(
                    (got[axis] - want[axis]).abs() <= UV_TOLERANCE,
                    "{} corner {point:?} UV[{axis}] = {}, Blender {}",
                    layers[layer],
                    got[axis],
                    want[axis]
                );
            }
        }
    }
    Ok(matched)
}

/// Evaluates a fixture's input and checks that every oracle corner matches Blender.
fn check_fixture(name: &str) -> TestResult<usize> {
    let fixture = fixture(name)?;
    let snapshot = evaluate(&fixture_scene(&fixture["input"], UvStorage::Blender)?)?;
    matched_oracle_corners(subject_mesh(&snapshot)?, &fixture, false)
}

/// A `UVMap` layer with one UV per corner, for inputs whose UV values are not compared.
fn uv_layer_for(mesh: &Mesh) -> Value {
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    json!([{"name": "UVMap", "values": vec![json!([0.5, 0.5]); corner_count]}])
}

/// Asserts that the subject lost its UV layers and that the drop is reported against the node.
fn assert_uv_dropped(snapshot: &Snapshot, modifier_type: &str) -> TestResult {
    assert!(
        !subject_mesh(snapshot)?
            .attributes
            .contains_key("blender_uv_layers"),
        "{modifier_type} output must not carry UV layers"
    );
    let warning = snapshot
        .warnings
        .iter()
        .find(|warning| warning.details["feature_id"] == "modifier.attribute_dropped")
        .ok_or("attribute drop warning missing")?;
    assert_eq!(warning.details["modifier_type"], modifier_type);
    assert_eq!(warning.details["node_id"], "subject");
    assert!(
        warning.details["attributes"]
            .as_array()
            .is_some_and(|attributes| attributes.iter().any(|name| name == "blender_uv_layers")),
        "dropped attributes: {}",
        warning.details["attributes"]
    );
    Ok(())
}

// Mirror: flip U/V, offset, UDIM. Oracle: uvcube (touching base), ONE_AXIS merge on.

#[test]
fn mirror_flip_u_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_flip_u")?, 44);
    Ok(())
}

#[test]
fn mirror_flip_uv_with_offsets_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_flip_uv_offset_uv")?, 44);
    Ok(())
}

#[test]
fn mirror_tiled_flip_u_udim_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_tile_flip_u_udim")?, 44);
    Ok(())
}

#[test]
fn mirror_udim_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_udim")?, 44);
    Ok(())
}

// Mirror on X and Y with flip U (uvcube, merge on), and a UDIM reflection of a quad whose corner UVs
// lie outside 0..1.

#[test]
fn mirror_xy_flip_u_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_xy_flip_u")?, 80);
    Ok(())
}

#[test]
fn mirror_udim_half_plane_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("mirror_udim_half_plane")?, 8);
    Ok(())
}

// Subdivision. Default uv_smooth (PRESERVE_BOUNDARIES) on a continuous UV grid at level 2,
// SIMPLE at levels 1 and 2, and SMOOTH_ALL on island UVs (a cube whose faces each own a UV island).

#[test]
fn subdivision_continuous_uv_level2_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("subsurf_cc_l2_cont")?, 256);
    Ok(())
}

#[test]
fn subdivision_simple_level1_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("subsurf_simple")?, 96);
    Ok(())
}

#[test]
fn subdivision_simple_level2_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("subsurf_simple_l2")?, 384);
    Ok(())
}

#[test]
fn subdivision_smooth_all_keeps_island_uvs_matching_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("subsurf_smoothall_cube")?, 96);
    Ok(())
}

// Bevel. Oracle: uvcube, width 0.1, limit NONE.

#[test]
fn bevel_edges_two_segments_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("bevel_cube")?, 216);
    Ok(())
}

#[test]
fn bevel_edges_three_segments_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("bevel_cube_seg3")?, 384);
    Ok(())
}

#[test]
fn bevel_vertices_two_segments_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("bevel_verts_seg2_uv")?, 168);
    Ok(())
}

// Solidify. Oracle: 3x3 grid with a random per-corner UV layer.

#[test]
fn solidify_rim_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("solidify_grid")?, 64);
    Ok(())
}

#[test]
fn solidify_without_rim_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("solidify_grid_norim")?, 32);
    Ok(())
}

// Screw. Oracle: 3x3 grid, three steps around Z with stretched UVs.

#[test]
fn screw_stretch_uv_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("screw_stretch_uv")?, 144);
    Ok(())
}

// Screw over loose edges: a three-point polyline (no faces, so no loops) spun a full turn with
// screw_offset 0.5. The variant also stretches V.

#[test]
fn screw_loose_edges_full_turn_offset_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("screw_loose_edge_offset")?, 24);
    Ok(())
}

#[test]
fn screw_loose_edges_full_turn_offset_stretch_v_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("screw_loose_edge_offset_stretch_v")?, 24);
    Ok(())
}

#[test]
fn screw_keeps_uv_map_entries_equal_to_blender_uv_layers_path() -> TestResult {
    // `screw_stretch_uv` with every layer in `uv_map`: each output face keeps an entry per layer,
    // and the entries carry the values that the `blender_uv_layers` input produces.
    let fixture = fixture("screw_stretch_uv")?;
    let stored = evaluate(&fixture_scene(&fixture["input"], UvStorage::Blender)?)?;
    let mapped = evaluate(&fixture_scene(&fixture["input"], UvStorage::AllUvMap)?)?;
    let (stored, mapped) = (subject_mesh(&stored)?, subject_mesh(&mapped)?);
    assert_eq!(stored.faces.len(), mapped.faces.len(), "face count");
    let layers = fixture_layers(&fixture)?;
    let mut corner = 0;
    for face in &mapped.faces {
        let end = corner + face.vertices.len();
        for layer in &layers {
            let entry = uv_map_entries(mapped)
                .iter()
                .find(|entry| {
                    entry["face_id"].as_u64() == Some(u64::from(face.id))
                        && uv_map_layer(entry) == layer.as_str()
                })
                .ok_or_else(|| format!("no `{layer}` uv_map entry for face {}", face.id))?;
            let want = layer_uvs(stored, layer)?;
            let row = entry["uv"].as_array().ok_or("uv_map entry without uv")?;
            assert_eq!(
                row.len(),
                end - corner,
                "`{layer}` corners of face {}",
                face.id
            );
            for (value, expected) in row.iter().zip(&want[corner..end]) {
                let got = uv(value)?;
                for axis in 0..2 {
                    assert!(
                        (got[axis] - expected[axis]).abs() <= UV_TOLERANCE,
                        "`{layer}` UV[{axis}] = {}, stored {}",
                        got[axis],
                        expected[axis]
                    );
                }
            }
        }
        corner = end;
    }
    assert_eq!(matched_oracle_corners(mapped, &fixture, false)?, 144);
    Ok(())
}

// Decimate (collapse, ratio 0.5). Blender moves 10 of 21 corners to other positions, so only the
// 11 corners present at the same face centroid and corner position are compared.

#[test]
fn decimate_collapse_matches_blender_oracle_on_matched_corners() -> TestResult {
    let fixture = fixture("decimate_uv")?;
    let snapshot = evaluate(&fixture_scene(&fixture["input"], UvStorage::Blender)?)?;
    assert_eq!(
        matched_oracle_corners(subject_mesh(&snapshot)?, &fixture, true)?,
        11
    );
    Ok(())
}

// Array with a start cap (continuous UV grid) and an end cap (tiled cube) that carry UVs.

#[test]
fn array_caps_carry_uvs_matching_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("array_cap_both")?, 112);
    Ok(())
}

#[test]
fn array_merge_with_uv_offset_matches_blender_oracle() -> TestResult {
    assert_eq!(check_fixture("array_merge_uv_offset")?, 64);
    Ok(())
}

/// Checks that two evaluated corner lists agree: same centroid and corner position, same UVs.
fn same_corners(got: &[Corner], want: &[Corner]) -> TestResult {
    if got.len() != want.len() {
        return Err(format!("corner count {}, expected {}", got.len(), want.len()).into());
    }
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        if (got.centroid - want.centroid).length() > UV_TOLERANCE
            || (got.point - want.point).length() > UV_TOLERANCE
        {
            return Err(format!("corner {index} at {:?} moved", got.point).into());
        }
        for (layer, (got_uv, want_uv)) in got.uv.iter().zip(&want.uv).enumerate() {
            for axis in 0..2 {
                if (got_uv[axis] - want_uv[axis]).abs() > UV_TOLERANCE {
                    return Err(format!(
                        "corner {index} layer {layer} UV[{axis}] = {}, expected {}",
                        got_uv[axis], want_uv[axis]
                    )
                    .into());
                }
            }
        }
    }
    Ok(())
}

/// Compares one fixture's layer-less `uv_map` input with its `blender_uv_layers` input.
fn layerless_input_matches_layers(name: &str) -> TestResult {
    let fixture = fixture(name)?;
    let layers = fixture_layers(&fixture)?;
    let stored = evaluate(&fixture_scene(&fixture["input"], UvStorage::Blender)?)?;
    let layerless = evaluate(&fixture_scene(
        &fixture["input"],
        UvStorage::LayerlessUvMap,
    )?)?;
    let (stored, layerless) = (subject_mesh(&stored)?, subject_mesh(&layerless)?);
    same_corners(&corners(layerless, &layers)?, &corners(stored, &layers)?)?;
    let matched = matched_oracle_corners(layerless, &fixture, true)?;
    let expected = matched_oracle_corners(stored, &fixture, true)?;
    if matched != expected {
        return Err(format!("{matched} oracle corners matched, expected {expected}").into());
    }
    Ok(())
}

// Every fixture through layer-less `uv_map` input: the UVMap of the subject and of each cap moves
// into `uv_map` entries, and the corners must equal those of the `blender_uv_layers` input.

#[test]
fn every_fixture_through_layerless_uv_map_input_matches_blender_layers() -> TestResult {
    let mut failures = Vec::new();
    for name in fixture_names()? {
        if let Err(error) = layerless_input_matches_layers(&name) {
            failures.push(format!("{name}: {error}"));
        }
    }
    assert!(
        failures.is_empty(),
        "layer-less uv_map input differs:\n{}",
        failures.join("\n")
    );
    Ok(())
}

// Rebuilding modifiers must keep the UV-named `FLOAT2` `CORNER` copy in `blender_attributes` equal to
// `blender_uv_layers`: mirror, subdivision at level 2, decimate, and array with caps.

#[test]
fn blender_attribute_uv_copies_match_blender_uv_layers_after_rebuilds() -> TestResult {
    for name in [
        "mirror_flip_uv_offset_uv",
        "subsurf_cc_l2_cont",
        "decimate_uv",
        "array_cap_both",
    ] {
        let fixture = fixture(name)?;
        let snapshot = evaluate(&fixture_scene(
            &fixture["input"],
            UvStorage::AttributeCopies,
        )?)?;
        let mesh = subject_mesh(&snapshot)?;
        let attributes = mesh
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .ok_or("blender_attributes missing")?;
        for layer in fixture_layers(&fixture)? {
            let copy = attributes
                .iter()
                .find(|attribute| {
                    attribute["name"].as_str() == Some(layer.as_str())
                        && attribute["domain"].as_str() == Some("CORNER")
                        && attribute["data_type"].as_str() == Some("FLOAT2")
                })
                .ok_or_else(|| format!("{name}: no FLOAT2 CORNER copy of `{layer}`"))?;
            let copied = copy["values"].as_array().ok_or("copy values missing")?;
            let stored = layer_uvs(mesh, &layer)?;
            assert_eq!(copied.len(), stored.len(), "{name}: `{layer}` copy length");
            for (value, want) in copied.iter().zip(&stored) {
                let got = uv(value)?;
                for axis in 0..2 {
                    assert!(
                        (got[axis] - want[axis]).abs() <= UV_TOLERANCE,
                        "{name}: `{layer}` copy UV[{axis}] = {}, layer {}",
                        got[axis],
                        want[axis]
                    );
                }
            }
        }
    }
    Ok(())
}

// Remesh and Skin keep no UV layers in Blender (oracle `remesh_uv` and `skin_uv` both report
// `layers: []`). Potter drops the layer and reports the node-scoped attribute-drop warning.

#[test]
fn remesh_drops_uv_layers_with_warning_naming_the_node() -> TestResult {
    let mut subject = Mesh::box_mesh(BoxParams::default())?;
    subject
        .attributes
        .insert("blender_uv_layers".to_owned(), uv_layer_for(&subject));
    let remesh = modifier(
        "remesh",
        "remesh",
        &json!({"mode": "VOXEL", "voxel_size": 0.2}),
    )?;
    let snapshot = evaluate(&scene_doc(subject, vec![remesh], Vec::new())?)?;
    assert_uv_dropped(&snapshot, "remesh")
}

#[test]
fn skin_drops_uv_layers_with_warning_naming_the_node() -> TestResult {
    let mut subject = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    assert_eq!(subject.edges.len(), 4, "quad face edges drive the skin");
    subject
        .attributes
        .insert("blender_uv_layers".to_owned(), uv_layer_for(&subject));
    let skin = modifier("skin", "skin", &json!({}))?;
    let snapshot = evaluate(&scene_doc(subject, vec![skin], Vec::new())?)?;
    assert_uv_dropped(&snapshot, "skin")
}

// glTF export. Each UV set becomes one TEXCOORD_n accessor. Positions are compared in glTF axes:
// a Potter point (x, y, z) exports as (x, z, -y).

/// Parses a GLB container into its JSON document and binary chunk.
fn glb_parts(bytes: &[u8]) -> TestResult<(Value, Vec<u8>)> {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return Err("export is not a GLB".into());
    }
    let json_length = usize::try_from(u32::from_le_bytes(bytes[12..16].try_into()?))?;
    let json_end = 20 + json_length;
    let json: Value = serde_json::from_slice(&bytes[20..json_end])?;
    let binary_length = usize::try_from(u32::from_le_bytes(
        bytes[json_end..json_end + 4].try_into()?,
    ))?;
    let binary = bytes[json_end + 8..json_end + 8 + binary_length].to_vec();
    Ok((json, binary))
}

/// Float components of a `glTF` VEC2 or VEC3 accessor, in order.
fn accessor_floats(json: &Value, binary: &[u8], index: u64) -> TestResult<Vec<f64>> {
    let accessor = &json["accessors"][usize::try_from(index)?];
    let view = &json["bufferViews"]
        [usize::try_from(accessor["bufferView"].as_u64().ok_or("bufferView")?)?];
    let components = match accessor["type"].as_str() {
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        _ => return Err("expected a VEC2 or VEC3 accessor".into()),
    };
    let count = usize::try_from(accessor["count"].as_u64().ok_or("count")?)?;
    let start = usize::try_from(
        view["byteOffset"].as_u64().unwrap_or(0) + accessor["byteOffset"].as_u64().unwrap_or(0),
    )?;
    (0..count * components)
        .map(|component| -> TestResult<f64> {
            let at = start + component * 4;
            Ok(f64::from(f32::from_le_bytes(
                binary[at..at + 4].try_into()?,
            )))
        })
        .collect()
}

type ExportedVertex = (DVec3, Vec<[f64; 2]>);

/// Exports `doc` through `pot export --format glb` and returns each exported vertex's position
/// with its `TEXCOORD_0..n` values, in glTF axes.
fn exported_glb_vertices(doc: &SceneDoc) -> TestResult<Vec<ExportedVertex>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    run_ok(pot().arg("init").arg(&scene).arg("--json"))?;
    fs::write(scene.join("scene.json"), serde_json::to_vec(doc)?)?;
    let glb = directory.path().join("subject.glb");
    run_ok(
        pot()
            .arg("export")
            .arg(&scene)
            .args(["--format", "glb", "--out"])
            .arg(&glb)
            .arg("--json"),
    )?;
    let (json, binary) = glb_parts(&fs::read(&glb)?)?;
    let mut vertices = Vec::new();
    for primitive in json["meshes"][0]["primitives"]
        .as_array()
        .ok_or("primitives missing")?
    {
        let attributes = &primitive["attributes"];
        let position = attributes["POSITION"].as_u64().ok_or("POSITION missing")?;
        let positions = accessor_floats(&json, &binary, position)?;
        let mut sets = Vec::new();
        for set in 0.. {
            let Some(accessor) = attributes[format!("TEXCOORD_{set}").as_str()].as_u64() else {
                break;
            };
            sets.push(accessor_floats(&json, &binary, accessor)?);
        }
        for (vertex, point) in positions.chunks(3).enumerate() {
            let uvs = sets
                .iter()
                .map(|floats| [floats[vertex * 2], floats[vertex * 2 + 1]])
                .collect();
            vertices.push((DVec3::new(point[0], point[1], point[2]), uvs));
        }
    }
    Ok(vertices)
}

/// Asserts that two (position, UV sets) lists agree: each vertex on either side has a match.
fn assert_same_vertices(exported: &[ExportedVertex], expected: &[ExportedVertex]) {
    let same = |(point, uvs): &ExportedVertex, (other, other_uvs): &ExportedVertex| {
        point.distance(*other) < POSITION_TOLERANCE
            && uvs.len() == other_uvs.len()
            && uvs
                .iter()
                .zip(other_uvs)
                .all(|(uv, want)| (0..2).all(|axis| (uv[axis] - want[axis]).abs() <= UV_TOLERANCE))
    };
    for vertex in exported {
        assert!(
            expected.iter().any(|corner| same(vertex, corner)),
            "exported vertex {vertex:?} does not match any expected corner"
        );
    }
    for corner in expected {
        assert!(
            exported.iter().any(|vertex| same(vertex, corner)),
            "expected corner {corner:?} is missing from the export"
        );
    }
}

#[test]
fn glb_export_texcoords_after_mirror_and_subdivision_match_blender_oracle() -> TestResult {
    let fixture = fixture("mirror_subsurf_uv_quarter")?;
    let exported = exported_glb_vertices(&fixture_scene(&fixture["input"], UvStorage::Blender)?)?;
    // Blender's layers are UVMap then UV2; they export as TEXCOORD_0 then TEXCOORD_1.
    assert!(
        exported.iter().all(|(_, uvs)| uvs.len() == 2),
        "every vertex carries TEXCOORD_0 and TEXCOORD_1"
    );
    let expected = fixture["corners"]
        .as_array()
        .ok_or("corners missing")?
        .iter()
        .map(|row| -> TestResult<ExportedVertex> {
            let values = row
                .as_array()
                .ok_or("corner row")?
                .iter()
                .map(|value| -> TestResult<f64> { Ok(value.as_f64().ok_or("corner value")?) })
                .collect::<TestResult<Vec<f64>>>()?;
            let point = DVec3::new(values[3], values[4], values[5]);
            // The fixture holds Blender loop UVs (V up). glTF stores V down: Blender's own glTF
            // exporter writes 1 - v for the same loop data (verified against Blender 5.2).
            let uvs = values[6..]
                .chunks(2)
                .map(|pair| [pair[0], 1.0 - pair[1]])
                .collect();
            Ok((DVec3::new(point.x, point.z, -point.y), uvs))
        })
        .collect::<TestResult<Vec<_>>>()?;
    assert!(!exported.is_empty(), "export carried no vertices");
    assert_same_vertices(&exported, &expected);
    Ok(())
}

#[test]
fn glb_export_prefers_uv_map_over_blender_uv_layers() -> TestResult {
    // Both representations are present with different values: the `uv_map` entries win, as one set.
    let mut subject = Mesh::from_positions_and_faces(
        vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
        vec![vec![0, 1, 2, 3]],
    )?;
    let face_id = subject.faces[0].id;
    let preferred = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
    subject.attributes.insert(
        "uv_map".to_owned(),
        json!([{"face_id": face_id, "layer": "UVMap", "uv": preferred}]),
    );
    subject.attributes.insert(
        "blender_uv_layers".to_owned(),
        json!([{"name": "UVMap", "values": [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]]}]),
    );
    let points = [DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y];
    // The entry carries a `layer` key, so it is Blender-layer data (V up) that the exporter flips to
    // glTF's V down (`uv_map` shortcut in exchange/gltf.rs, until PR12).
    let expected: Vec<ExportedVertex> = points
        .iter()
        .zip(preferred.map(|[u, v]| [u, 1.0 - v]))
        .map(|(point, uv)| (DVec3::new(point.x, point.z, -point.y), vec![uv]))
        .collect();
    let exported = exported_glb_vertices(&scene_doc(subject, Vec::new(), Vec::new())?)?;
    assert_same_vertices(&exported, &expected);
    Ok(())
}
