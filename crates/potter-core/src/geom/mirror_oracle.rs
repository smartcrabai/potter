//! Mirror parity against Blender 5.2.2. Each `tests/fixtures/pr18b_mirror/*.json` is a stack evaluated by
//! Blender (written by `generate.py` there). Potter must reproduce its vertices, faces, edges, materials,
//! UVs, and vertex weights to `POSITION_TOLERANCE`, and its shading and corner normals through the shading
//! oracle's comparison.

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    fs,
    path::Path,
};

use glam::{DMat4, DVec3};
use serde_json::{Value, json};

use super::{
    Mesh,
    provenance::edge_flags,
    shading_oracle::{compare, point, scene},
    uv_layers,
};
use crate::{
    eval::{EvaluationContext, Snapshot},
    model::Id,
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Positions and UVs match Blender's float32 output to this tolerance.
const POSITION_TOLERANCE: f64 = 1.0e-6;

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pr18b_mirror")
}

fn fixture_paths() -> TestResult<Vec<std::path::PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(fixture_directory())? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Every mismatch between Potter's evaluation of `fixture` and Blender's output, in words.
fn mismatches(fixture: &Value) -> TestResult<Vec<String>> {
    let context = EvaluationContext {
        frame: Some(1.0),
        ..EvaluationContext::default()
    };
    let snapshot = Snapshot::evaluate(&scene(fixture)?, &context)?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("subject mesh is missing")?;
    let oracle = &fixture["output"];
    let mut problems = Vec::new();

    let blender_vertices = points(&oracle["vertices"])?;
    let blender_faces = oracle["faces"].as_array().ok_or("output faces")?;
    let blender_edges = oracle["edges"].as_array().ok_or("output edges")?;
    for (label, potter, blender) in [
        ("vertices", mesh.vertices.len(), blender_vertices.len()),
        ("edges", mesh.edges.len(), blender_edges.len()),
        ("faces", mesh.faces.len(), blender_faces.len()),
    ] {
        if potter != blender {
            problems.push(format!("{label}: potter {potter}, blender {blender}"));
        }
    }
    let report = compare(fixture, mesh)?;
    if !report.exact() {
        problems.push(format!("shading or normals: {report:?}"));
    }

    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut vertex_used = vec![false; positions.len()];
    for (index, blender) in blender_vertices.iter().enumerate() {
        match find_unused(&positions, *blender, &mut vertex_used) {
            Some(_) => {}
            None => problems.push(format!("vertex {index} at {blender:?} has no Potter match")),
        }
    }

    let layers = uv_layers::read_layers(mesh);
    let mut face_used = vec![false; mesh.faces.len()];
    for (blender_index, blender) in blender_faces.iter().enumerate() {
        let corners = points(&blender["corners"])?;
        let found = mesh.faces.iter().enumerate().position(|(index, face)| {
            !face_used[index]
                && face.vertices.len() == corners.len()
                && corners_match(
                    &face
                        .vertices
                        .iter()
                        .map(|id| positions[vertex_index[id]])
                        .collect::<Vec<_>>(),
                    &corners,
                )
        });
        let Some(face_index) = found else {
            problems.push(format!(
                "face {blender_index} {corners:?} has no Potter match"
            ));
            continue;
        };
        face_used[face_index] = true;
        let face = &mesh.faces[face_index];
        let blender_material = blender["material_index"].as_u64();
        if blender_material != Some(u64::from(face.material_index)) {
            problems.push(format!(
                "face {blender_index} material {}, blender {blender_material:?}",
                face.material_index
            ));
        }
        let potter_corners: Vec<DVec3> = face
            .vertices
            .iter()
            .map(|id| positions[vertex_index[id]])
            .collect();
        let corner_of = |point: DVec3| {
            potter_corners.iter().position(|candidate| {
                (*candidate - point).abs().max_element() <= POSITION_TOLERANCE
            })
        };
        for (name, values) in blender["uv"].as_object().into_iter().flatten() {
            let values = values.as_array().ok_or("UV corners")?;
            let Some(layer) = layers.iter().find(|layer| layer.name == *name) else {
                problems.push(format!("face {blender_index}: UV layer {name} missing"));
                continue;
            };
            let Some(potter_uvs) = layer.faces.get(face_index).and_then(Option::as_ref) else {
                problems.push(format!("face {blender_index}: UV {name} has no corners"));
                continue;
            };
            for (corner, point) in corners.iter().enumerate() {
                let Some(at) = corner_of(*point) else {
                    continue;
                };
                let expected = values[corner].as_array().ok_or("UV")?;
                let expected = [
                    expected[0].as_f64().ok_or("UV")?,
                    expected[1].as_f64().ok_or("UV")?,
                ];
                let expected = [expected[0], 1.0 - expected[1]];
                let actual = potter_uvs[at];
                if (actual[0] - expected[0])
                    .abs()
                    .max((actual[1] - expected[1]).abs())
                    > POSITION_TOLERANCE
                {
                    problems.push(format!(
                        "face {blender_index} corner {corner} UV {name}: potter {actual:?}, blender {expected:?}"
                    ));
                }
            }
        }
    }
    if let Some(index) = face_used.iter().position(|used| !used) {
        problems.push(format!("Potter face {index} has no Blender match"));
    }

    let seam_flags = edge_flags(mesh, "seam");
    let mut edge_used = vec![false; mesh.edges.len()];
    for (blender_index, blender) in blender_edges.iter().enumerate() {
        let ends = points(&blender["verts"])?;
        let found = mesh.edges.iter().enumerate().position(|(index, edge)| {
            if edge_used[index] {
                return false;
            }
            let [first, second] = edge.vertices.map(|id| positions[vertex_index[&id]]);
            let close = |a: DVec3, b: DVec3| (a - b).abs().max_element() <= POSITION_TOLERANCE;
            (close(first, ends[0]) && close(second, ends[1]))
                || (close(first, ends[1]) && close(second, ends[0]))
        });
        let Some(edge_index) = found else {
            problems.push(format!("edge {blender_index} {ends:?} has no Potter match"));
            continue;
        };
        edge_used[edge_index] = true;
        let edge = &mesh.edges[edge_index];
        let seam = seam_flags[edge_index];
        let sharp = mesh
            .attributes
            .get("sharp")
            .and_then(|attribute| attribute.get("values"))
            .and_then(|values| values.get(format!("e{}", edge.id)))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if seam != blender["seam"].as_bool().unwrap_or(false) {
            problems.push(format!("edge {blender_index} seam flag differs"));
        }
        if sharp != blender["sharp"].as_bool().unwrap_or(false) {
            problems.push(format!("edge {blender_index} sharp flag differs"));
        }
    }

    problems.extend(weight_mismatches(
        mesh,
        &oracle["vertex_groups"],
        &positions,
    )?);
    Ok(problems)
}

/// Vertex weights, per group. Blender's entries and Potter's vertices pair one-to-one: copies can share a
/// position with their originals, so a position alone does not identify a vertex.
fn weight_mismatches(mesh: &Mesh, blender: &Value, positions: &[DVec3]) -> TestResult<Vec<String>> {
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut potter: Vec<BTreeMap<String, f64>> = vec![BTreeMap::new(); positions.len()];
    for (group, weights) in mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        for (key, weight) in weights.as_object().into_iter().flatten() {
            let id: u32 = key.trim_start_matches('v').parse()?;
            let index = *vertex_index.get(&id).ok_or("weight on a missing vertex")?;
            let weight = weight.as_f64().ok_or("weight")?;
            if weight != 0.0 {
                potter[index].insert(group.clone(), weight);
            }
        }
    }
    let mut problems = Vec::new();
    let mut used = vec![false; positions.len()];
    for entry in blender.as_array().into_iter().flatten() {
        let position = point(&entry["position"])?;
        let mut expected = BTreeMap::new();
        for (group, weight) in entry["weights"].as_object().into_iter().flatten() {
            let weight = weight.as_f64().ok_or("weight")?;
            if weight != 0.0 {
                expected.insert(group.clone(), weight);
            }
        }
        let found = (0..positions.len()).find(|&index| {
            !used[index]
                && (positions[index] - position).abs().max_element() <= POSITION_TOLERANCE
                && same_weights(&potter[index], &expected)
        });
        match found {
            Some(index) => used[index] = true,
            None => problems.push(format!(
                "weighted vertex {position:?} {expected:?} has no Potter match"
            )),
        }
    }
    for (index, weights) in potter.iter().enumerate() {
        if !used[index] && !weights.is_empty() {
            problems.push(format!(
                "Potter vertex {index} weights {weights:?} have no Blender match"
            ));
        }
    }
    Ok(problems)
}

fn same_weights(potter: &BTreeMap<String, f64>, expected: &BTreeMap<String, f64>) -> bool {
    potter.len() == expected.len()
        && expected.iter().all(|(group, weight)| {
            potter
                .get(group)
                .is_some_and(|actual| (actual - weight).abs() <= POSITION_TOLERANCE)
        })
}

fn points(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("point list")?
        .iter()
        .map(point)
        .collect()
}

/// The first unused `positions` entry within `POSITION_TOLERANCE` of `target`, marked as used.
fn find_unused(positions: &[DVec3], target: DVec3, used: &mut [bool]) -> Option<usize> {
    let index = positions
        .iter()
        .enumerate()
        .position(|(index, candidate)| {
            !used[index] && (*candidate - target).abs().max_element() <= POSITION_TOLERANCE
        })?;
    used[index] = true;
    Some(index)
}

/// Whether the two corner lists are the same cyclic polygon, corner by corner within the tolerance.
fn corners_match(potter: &[DVec3], blender: &[DVec3]) -> bool {
    let mut used = vec![false; blender.len()];
    potter
        .iter()
        .all(|point| find_unused(blender, *point, &mut used).is_some())
}

#[test]
fn mirror_fixtures_match_blender() -> TestResult {
    let mut failures = Vec::new();
    let mut checked = 0;
    for path in fixture_paths()? {
        let name = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default()
            .to_owned();
        let fixture: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
        checked += 1;
        let problems = mismatches(&fixture)?;
        if !problems.is_empty() {
            failures.push(format!("{name}:\n  {}", problems.join("\n  ")));
        }
    }
    assert!(checked > 0, "no mirror fixtures found");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn bisect_keeps_blender_face_slots_workstack_order_and_first_loops() -> TestResult {
    for name in [
        "concave_comb_bisect_x",
        "concave_comb_bisect_x_flip",
        "concave_comb_bisect_y",
        "concave_comb_bisect_y_flip",
        "concave_comb_bisect_x_corner_custom",
    ] {
        let fixture: Value = serde_json::from_str(&fs::read_to_string(
            fixture_directory().join(format!("{name}.json")),
        )?)?;
        let snapshot = Snapshot::evaluate(
            &scene(&fixture)?,
            &EvaluationContext {
                frame: Some(1.0),
                ..EvaluationContext::default()
            },
        )?;
        let mesh = &snapshot.meshes[&Id::new("subject")?];
        let positions: Vec<_> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
        let indices: HashMap<_, _> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (v.id, i))
            .collect();
        let expected = fixture["output"]["faces"]
            .as_array()
            .ok_or("output faces")?;
        assert_eq!(mesh.faces.len(), expected.len(), "{name}: face count");
        for (index, (face, blender)) in mesh.faces.iter().zip(expected).enumerate() {
            let actual: Vec<_> = face
                .vertices
                .iter()
                .map(|id| positions[indices[id]])
                .collect();
            let expected = points(&blender["corners"])?;
            assert_eq!(actual.len(), expected.len(), "{name}: face {index} corners");
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (*a - *b).abs().max_element() <= POSITION_TOLERANCE),
                "{name}: Blender face slot/loop order differs at face {index}"
            );
        }
    }
    Ok(())
}

#[test]
fn mirror_codes_survive_reload_through_bisect_and_deformation() -> TestResult {
    let fixture: Value = serde_json::from_str(&fs::read_to_string(
        fixture_directory().join("crosschain_x_y_bisect_then_wave_vertex.json"),
    )?)?;
    let mut document = scene(&fixture)?;
    let subject = Id::new("subject")?;
    let (first_modifier, remaining_modifiers, data_id) = {
        let node = document.nodes.get(&subject).ok_or("subject node")?;
        (
            node.modifiers
                .first()
                .ok_or("first Mirror modifier")?
                .clone(),
            node.modifiers[1..].to_vec(),
            node.data.clone().ok_or("subject data")?,
        )
    };
    let base_mesh = document
        .data_blocks
        .get_mut(&data_id)
        .and_then(|block| block.mesh.as_mut())
        .ok_or("subject mesh")?;
    let mut order_rows = serde_json::Map::new();
    let mut offset = 0;
    for face in &base_mesh.faces {
        let end = offset + face.vertices.len();
        let row: Vec<usize> = (offset..end).collect();
        order_rows.insert(format!("f{}", face.id), json!(row));
        offset = end;
    }
    base_mesh
        .attributes
        .get_mut("custom_normal")
        .and_then(Value::as_object_mut)
        .ok_or("fixture has no custom-normal state")?
        .insert("corner_order".to_owned(), Value::Object(order_rows));
    base_mesh.faces.reverse();

    let source = document
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.mesh.clone())
        .ok_or("subject mesh")?;
    let baked = crate::geom::modifiers::evaluate_modifiers_with_scene_context(
        source,
        std::slice::from_ref(&first_modifier),
        &document.node_groups,
        1.0,
        DMat4::IDENTITY,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
        crate::eval::ModifierEvaluationContext::default(),
        |_, modifier| modifier.enabled,
        |_, _, _| Ok(()),
    )?
    .mesh;
    let codes = super::clnor::corner_codes(&baked)?.ok_or("Mirror lost the code layer")?;
    let order = super::clnor::corner_order(&baked)?;
    assert!(
        order.as_ref().is_some_and(|order| {
            order
                .iter()
                .enumerate()
                .any(|(index, &ordinal)| index != ordinal)
        }),
        "non-bisecting Mirror must preserve the incoming basis"
    );
    let normal_state = baked
        .attributes
        .get("custom_normal")
        .and_then(Value::as_object)
        .ok_or("Mirror lost canonical custom-normal state")?;
    assert_eq!(
        normal_state.get("domain").and_then(Value::as_str),
        Some("corner")
    );
    assert_eq!(
        normal_state.get("type").and_then(Value::as_str),
        Some("normal_code")
    );
    assert!(
        normal_state
            .keys()
            .all(|key| { matches!(key.as_str(), "domain" | "type" | "values" | "corner_order") }),
        "Mirror wrote fields outside the canonical custom-normal object"
    );
    let raw_values = normal_state
        .get("values")
        .cloned()
        .ok_or("Mirror dropped raw normal codes")?;
    document
        .data_blocks
        .get_mut(&data_id)
        .ok_or("subject data")?
        .mesh = Some(baked);
    document
        .nodes
        .get_mut(&subject)
        .ok_or("subject node")?
        .modifiers = remaining_modifiers;

    // This is the native scene save/reload boundary: exact codes and the loop basis must persist.
    let saved = serde_json::to_vec(&document)?;
    let reloaded: crate::model::SceneDoc = serde_json::from_slice(&saved)?;
    let reloaded_mesh = reloaded
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.mesh.as_ref())
        .ok_or("reloaded subject mesh")?;
    assert_eq!(
        &reloaded_mesh.attributes["custom_normal"]["values"],
        &raw_values
    );
    assert_eq!(super::clnor::corner_codes(reloaded_mesh)?, Some(codes));
    assert_eq!(super::clnor::corner_order(reloaded_mesh)?, order);

    let snapshot = Snapshot::evaluate(
        &reloaded,
        &EvaluationContext {
            frame: Some(1.0),
            ..EvaluationContext::default()
        },
    )?;
    let output = snapshot.meshes.get(&subject).ok_or("subject output")?;
    let report = compare(&fixture, output)?;
    assert!(
        report.exact(),
        "second Mirror after reload differs: {report:?}"
    );
    Ok(())
}
