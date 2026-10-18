// SPDX-License-Identifier: GPL-3.0-or-later

//! PR4 shading oracle: smooth flags, sharp edges and corner normals through mesh-rebuilding
//! modifiers, against Blender 5.2.2. `tests/fixtures/pr4_shading/*.json` are written by
//! `generate.py` from Blender; this test replays each fixture's input and stack through the
//! evaluator and matches output elements geometrically (corner positions), never by index.
//! Corner normals are compared at 1e-5 and flags exactly. Fixtures outside [`EXACT`] are classes
//! whose Blender rule is not reproduced yet; [`report_all`] prints their numbers.

use std::{collections::HashMap, error::Error, fs, path::Path};

use glam::{DQuat, DVec3, EulerRot};
use serde_json::{Map, Value, json};

use super::{
    Mesh,
    clnor::{self, NormalCodeOptions, SetterInput},
    normals::{corner_normals, sharp_edge_flags, smooth_face_flags},
    uv_layers::{self, UvLayer},
};
use crate::{
    eval::{EvaluationContext, Snapshot},
    model::{DataBlock, Id, Modifier, ModifierRuntime, Node, SceneDoc, Transform},
};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "3d8b1c0e-9f7a-4b2d-8e61-5c4a2b1f0e9d";
const POSITION_TOLERANCE: f64 = 1.0e-4;
/// Corner normal tolerance without custom normals (Blender computes them in f32).
const NORMAL_TOLERANCE: f64 = 1.0e-5;
/// Corner normal tolerance with custom normals: Blender stores them as 16-bit components
/// (`normals_split_custom_set`), so a copied or interpolated value can differ by about 1e-4.
const QUANTIZED_NORMAL_TOLERANCE: f64 = 2.0e-4;

/// Fixtures whose flags match Blender exactly and whose corner normals are within tolerance.
const EXACT: &[&str] = &[
    "array_cube",
    "array_cube_caps",
    "array_cube_caps_zero_tf",
    "array_cube_merge",
    "bevel_cube",
    "bevel_cube_mark_sharp",
    "build_cube",
    "chain_mirror_subsurf",
    "chain_mirror_subsurf_custom_l1_triangulate",
    "decimate_cube_nocustom",
    "decimate_grid",
    "edge_split_cube",
    "edge_split_sharp_only",
    "mask_grid",
    "mirror_cube",
    "mirror_cube_bisect_flip",
    "mirror_cube_nocustom",
    "mirror_cube_nomerge",
    "mirror_grid",
    "remesh_smooth_mode_grid",
    "remesh_voxel_cube",
    "remesh_voxel_cube_smooth",
    "screw_line",
    "screw_line_smooth",
    "skin_line",
    "skin_line_smooth",
    "solidify_grid",
    "solidify_grid_flip",
    "solidify_grid_nocustom",
    "solidify_grid_norim",
    "solidify_grid_offset0",
    "subsurf_cube_cc",
    "subsurf_cube_cc_custom",
    "subsurf_cube_cc_custom_l2",
    "subsurf_cube_cc_nocustom",
    "subsurf_cube_cc_nocustom_ctrl",
    "subsurf_cube_simple",
    "subsurf_cube_simple_custom",
    "subsurf_grid_cc_custom",
    "subsurf_grid_cc_custom_l2",
    "triangulate_cube_drop",
    "triangulate_cube_drop_ctrl",
    "triangulate_cube_keep",
    "triangulate_cube_keep_jitter",
    "triangulate_grid_drop",
    "triangulate_grid_keep",
    "weld_split_cube",
    "weld_split_cube_connected",
    "weld_split_cube_dense",
    "weld_split_cube_dense_connected",
    "weld_split_cube_reversed",
    "wireframe_cube_nocustom",
    "wireframe_grid",
];

fn fixture(name: &str) -> TestResult<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/pr4_shading")
        .join(format!("{name}.json"));
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn fixture_names() -> TestResult<Vec<String>> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pr4_shading");
    let mut names = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            names.push(
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or("fixture name")?
                    .to_owned(),
            );
        }
    }
    names.sort();
    Ok(names)
}

pub(super) fn point(value: &Value) -> TestResult<DVec3> {
    let [x, y, z] = value.as_array().ok_or("point must be an array")?.as_slice() else {
        return Err("point must have three coordinates".into());
    };
    Ok(DVec3::new(
        x.as_f64().ok_or("coordinate")?,
        y.as_f64().ok_or("coordinate")?,
        z.as_f64().ok_or("coordinate")?,
    ))
}

fn indices(value: &Value) -> TestResult<Vec<usize>> {
    value
        .as_array()
        .ok_or("indices must be an array")?
        .iter()
        .map(|index| Ok(usize::try_from(index.as_u64().ok_or("index")?)?))
        .collect()
}

/// Input mesh of a fixture, with its shading and vertex groups in Potter's attribute layout.
/// Vertex and face IDs equal their fixture indices, since `from_positions_and_faces` assigns them
/// in order.
fn input_mesh(input: &Value) -> TestResult<Mesh> {
    let positions = input["vertices"]
        .as_array()
        .ok_or("vertices")?
        .iter()
        // Blender stores vertex positions as float32; the fixtures' decimal inputs are rounded the same way.
        .map(|value| {
            point(value).map(|p| DVec3::from_array(p.to_array().map(|c| f64::from(c as f32))))
        })
        .collect::<TestResult<Vec<_>>>()?;
    let faces = input["faces"]
        .as_array()
        .ok_or("faces")?
        .iter()
        .map(indices)
        .collect::<TestResult<Vec<_>>>()?;
    let face_count = faces.len();
    let mut mesh = Mesh::from_positions_and_faces(positions, faces)?;
    for edge in input["edges"].as_array().into_iter().flatten() {
        let [first, second] = indices(edge)?[..] else {
            return Err("loose edge needs two vertices".into());
        };
        mesh.insert_edge([u32::try_from(first)?, u32::try_from(second)?])?;
    }
    if let Some(order) = input.get("blender_edge_order") {
        let order = order
            .as_array()
            .ok_or("blender_edge_order must be an array")?;
        let mut by_vertices: HashMap<_, _> = mesh
            .edges
            .drain(..)
            .map(|edge| {
                let [a, b] = edge.vertices;
                ((a.min(b), a.max(b)), edge)
            })
            .collect();
        mesh.edges = order
            .iter()
            .map(|pair| {
                let [a, b] = indices(pair)?[..] else {
                    return Err("Blender edge order needs two vertices".into());
                };
                let (a, b) = (u32::try_from(a)?, u32::try_from(b)?);
                let mut edge = by_vertices
                    .remove(&(a.min(b), a.max(b)))
                    .ok_or("Blender edge order references a missing or duplicate edge")?;
                edge.vertices = [a, b];
                Ok(edge)
            })
            .collect::<TestResult<_>>()?;
        if !by_vertices.is_empty() {
            return Err("Blender edge order omits mesh edges".into());
        }
    }
    let sharp: Vec<(u32, u32)> = input["sharp_edges"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|pair| {
            let [a, b] = indices(pair)?[..] else {
                return Err("sharp edge needs two vertices".into());
            };
            let (a, b) = (u32::try_from(a)?, u32::try_from(b)?);
            Ok((a.min(b), a.max(b)))
        })
        .collect::<TestResult<_>>()?;
    let mut edge_values = Map::new();
    for edge in &mesh.edges {
        let [a, b] = edge.vertices;
        let is_sharp = sharp.contains(&(a.min(b), a.max(b)));
        edge_values.insert(format!("e{}", edge.id), json!(is_sharp));
    }
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain":"edge","type":"bool","values":edge_values}),
    );
    let smooth: Vec<bool> = input["face_smooth"]
        .as_array()
        .ok_or("face_smooth")?
        .iter()
        .map(|flag| flag.as_bool().ok_or("face_smooth must be bool"))
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(smooth.len(), face_count);
    let smooth_values: Map<String, Value> = smooth
        .iter()
        .enumerate()
        .map(|(index, flag)| (format!("f{index}"), json!(flag)))
        .collect();
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":smooth_values}),
    );
    if let Some(seams) = input["seam_edges"].as_array() {
        let seam_pairs: Vec<(u32, u32)> = seams
            .iter()
            .map(|pair| {
                let [a, b] = indices(pair)?[..] else {
                    return Err("seam edge needs two vertices".into());
                };
                let (a, b) = (u32::try_from(a)?, u32::try_from(b)?);
                Ok((a.min(b), a.max(b)))
            })
            .collect::<TestResult<_>>()?;
        let values: Map<String, Value> = mesh
            .edges
            .iter()
            .map(|edge| {
                let [a, b] = edge.vertices;
                let key = (a.min(b), a.max(b));
                (format!("e{}", edge.id), json!(seam_pairs.contains(&key)))
            })
            .collect();
        mesh.attributes.insert(
            "seam".to_owned(),
            json!({"domain":"edge","type":"bool","values":values}),
        );
    }
    if let Some(materials) = input["face_material_index"].as_array() {
        for (face, material) in mesh.faces.iter_mut().zip(materials) {
            face.material_index = u32::try_from(material.as_u64().ok_or("material index")?)?;
        }
    }
    if let Some(layers) = input["uv_layers"].as_object() {
        let layers: Vec<UvLayer> = layers
            .iter()
            .map(|(name, faces)| {
                let faces = faces
                    .as_array()
                    .ok_or("UV faces must be an array")?
                    .iter()
                    .map(|face| {
                        let values = face
                            .as_array()
                            .ok_or("UV corners must be an array")?
                            .iter()
                            .map(|uv| {
                                let uv = uv
                                    .as_array()
                                    .filter(|uv| uv.len() == 2)
                                    .ok_or("UV corner must have two components")?;
                                let u = uv[0].as_f64().ok_or("UV U must be numeric")?;
                                let v = uv[1].as_f64().ok_or("UV V must be numeric")?;
                                Ok([u, 1.0 - v])
                            })
                            .collect::<TestResult<Vec<_>>>()?;
                        Ok(Some(values))
                    })
                    .collect::<TestResult<Vec<_>>>()?;
                Ok(UvLayer::new(name.clone(), faces))
            })
            .collect::<TestResult<_>>()?;
        uv_layers::write_layers(&mut mesh, &layers);
    }
    if let Some(custom) = input["custom_normals"].as_array() {
        // Fixture rows are vectors or null per corner. The shared setter stores them as codes.
        let mut vectors: Vec<Option<DVec3>> = Vec::new();
        for (index, face) in mesh.faces.iter().enumerate() {
            let row: &[Value] = match custom.get(index) {
                None | Some(Value::Null) => &[],
                Some(value) => value
                    .as_array()
                    .ok_or("custom_normals rows must be arrays or null")?
                    .as_slice(),
            };
            if row.len() > face.vertices.len() {
                return Err("custom_normals row exceeds the face corners".into());
            }
            for corner in 0..face.vertices.len() {
                vectors.push(match row.get(corner) {
                    None => None,
                    Some(value) if value.is_null() => None,
                    Some(value) => Some(point(value)?),
                });
            }
        }
        clnor::encode_vectors(
            &mut mesh,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: SetterInput::NormalizeInput,
            },
        )?;
    }
    if let Some(groups) = input["vertex_groups"].as_object() {
        let mut attribute = Map::new();
        for (name, weights) in groups {
            let mut by_id = Map::new();
            for (vertex, weight) in weights.as_object().ok_or("vertex group weights")? {
                by_id.insert(format!("v{vertex}"), weight.clone());
            }
            attribute.insert(name.clone(), Value::Object(by_id));
        }
        mesh.attributes
            .insert("vertex_groups".to_owned(), Value::Object(attribute));
    }
    Ok(mesh)
}

/// Node ID of a fixture object: Potter IDs are lowercase slugs.
fn slug(name: &str) -> String {
    name.to_ascii_lowercase()
}

/// Modifiers of a fixture stack. String params naming a fixture object (cap, operand) use its slug.
fn modifiers(stack: &Value, objects: &Map<String, Value>) -> TestResult<Vec<Modifier>> {
    stack
        .as_array()
        .ok_or("stack")?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let modifier_type = item["type"].as_str().ok_or("modifier type")?.to_owned();
            let id = format!("m{index}_{modifier_type}");
            let mut params = item["params"].as_object().cloned().unwrap_or_default();
            for value in params.values_mut() {
                if let Some(name) = value.as_str().filter(|name| objects.contains_key(*name)) {
                    *value = json!(slug(name));
                }
            }
            Ok(Modifier {
                id: Id::new(&id)?,
                modifier_type,
                name: id,
                enabled: true,
                show_viewport: true,
                show_render: true,
                params,
                binding_data: None,
                runtime: ModifierRuntime::default(),
            })
        })
        .collect()
}

/// Transform of a fixture object or subject: `location`, `rotation_euler` (radians, Blender XYZ),
/// `scale`; a `translation` key is accepted as an alias of `location`.
fn transform_of(value: &Value) -> TestResult<Transform> {
    let vector = |key: &str, default: [f64; 3]| -> TestResult<[f64; 3]> {
        let Some(values) = value.get(key).and_then(Value::as_array) else {
            return Ok(default);
        };
        let [x, y, z] = values.as_slice() else {
            return Err(format!("{key} must have three components").into());
        };
        Ok([
            x.as_f64().ok_or(key)?,
            y.as_f64().ok_or(key)?,
            z.as_f64().ok_or(key)?,
        ])
    };
    let location = match value.get("translation") {
        Some(_) => vector("translation", [0.0; 3])?,
        None => vector("location", [0.0; 3])?,
    };
    let euler = vector("rotation_euler", [0.0; 3])?;
    // Blender's XYZ Euler is the matrix Rz * Ry * Rx, glam's ZYX intrinsic order.
    let rotation = DQuat::from_euler(EulerRot::ZYX, euler[2], euler[1], euler[0]).to_array();
    Ok(Transform {
        translation: location,
        rotation,
        scale: vector("scale", [1.0; 3])?,
        ..Transform::default()
    })
}

/// Scene with the fixture's subject (carrying the stack) and its referenced objects as nodes.
pub(super) fn scene(fixture: &Value) -> TestResult<SceneDoc> {
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    let root = Id::new("collection_root")?;
    let empty = Map::new();
    let objects = fixture["objects"].as_object().unwrap_or(&empty);
    let mut nodes = vec![(
        "subject".to_owned(),
        input_mesh(&fixture["input"])?,
        transform_of(&fixture["subject_transform"])?,
        modifiers(&fixture["stack"], objects)?,
    )];
    for (name, object) in objects {
        // Helper objects without geometry (mirror empties) evaluate to an empty mesh.
        let mesh = if object.get("vertices").is_some() {
            input_mesh(object)?
        } else {
            Mesh::new()
        };
        nodes.push((slug(name), mesh, transform_of(object)?, Vec::new()));
    }
    for (name, mesh, transform, modifiers) in nodes {
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
                transform,
                modifiers,
                data: Some(data_id),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&root)
            .ok_or("root collection")?
            .objects
            .push(node_id);
    }
    doc.validate()?;
    Ok(doc)
}

/// Corner positions of a Blender element, in loop order.
struct OracleFace {
    corners: Vec<DVec3>,
    normals: Vec<DVec3>,
    smooth: bool,
}

struct OracleEdge {
    ends: [DVec3; 2],
    sharp: bool,
}

fn oracle_faces(fixture: &Value) -> TestResult<Vec<OracleFace>> {
    fixture["output"]["faces"]
        .as_array()
        .ok_or("output faces")?
        .iter()
        .map(|face| {
            let corners = face["corners"]
                .as_array()
                .ok_or("corners")?
                .iter()
                .map(point)
                .collect::<TestResult<Vec<_>>>()?;
            let normals = face["normals"]
                .as_array()
                .ok_or("normals")?
                .iter()
                .map(point)
                .collect::<TestResult<Vec<_>>>()?;
            Ok(OracleFace {
                corners,
                normals,
                smooth: face["smooth"].as_bool().ok_or("smooth")?,
            })
        })
        .collect()
}

fn oracle_edges(fixture: &Value) -> TestResult<Vec<OracleEdge>> {
    fixture["output"]["edges"]
        .as_array()
        .ok_or("output edges")?
        .iter()
        .map(|edge| {
            let [first, second] = edge["verts"].as_array().ok_or("verts")?.as_slice() else {
                return Err("edge needs two vertices".into());
            };
            Ok(OracleEdge {
                ends: [point(first)?, point(second)?],
                sharp: edge["sharp"].as_bool().ok_or("sharp")?,
            })
        })
        .collect()
}

/// Whether Potter corner positions and Blender corner positions are the same set, within tolerance.
fn same_corners(potter: &[DVec3], blender: &[DVec3]) -> bool {
    if potter.len() != blender.len() {
        return false;
    }
    let mut used = vec![false; blender.len()];
    potter.iter().all(|point| {
        let found = blender.iter().enumerate().position(|(index, candidate)| {
            !used[index] && (*point - *candidate).abs().max_element() <= POSITION_TOLERANCE
        });
        found.is_some_and(|index| {
            used[index] = true;
            true
        })
    })
}

/// Comparison result of one fixture.
#[derive(Debug, Default)]
pub(super) struct Report {
    faces: usize,
    unmatched_faces: usize,
    unused_oracle_faces: usize,
    smooth_mismatches: usize,
    edges: usize,
    unmatched_edges: usize,
    sharp_mismatches: usize,
    corners: usize,
    unmatched_corners: usize,
    max_normal_error: f64,
    /// Tolerance this fixture's `max_normal_error` is judged against.
    normal_tolerance: f64,
    /// Corners whose normal points opposite to Blender's (a winding difference).
    flipped_corners: usize,
    /// Endpoints of edges whose sharp flag differs from Blender's, for diagnosis.
    sharp_details: Vec<String>,
}

impl Report {
    pub(super) fn exact(&self) -> bool {
        self.unmatched_faces == 0
            && self.unused_oracle_faces == 0
            && self.smooth_mismatches == 0
            && self.unmatched_edges == 0
            && self.sharp_mismatches == 0
            && self.unmatched_corners == 0
            && self.max_normal_error <= self.normal_tolerance
    }
}

pub(super) fn compare(fixture: &Value, mesh: &Mesh) -> TestResult<Report> {
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let oracle = oracle_faces(fixture)?;
    let oracle_edges = oracle_edges(fixture)?;
    let smooth = smooth_face_flags(mesh)?;
    let sharp = sharp_edge_flags(mesh)?;
    let normals = corner_normals(mesh, &positions)?;
    // Blender stores every custom normal as 16-bit codes, so any mesh that carries custom normals is
    // judged at the quantized tolerance, whether they came from the input or the modifier.
    let normal_tolerance = if mesh.attributes.contains_key("custom_normal") {
        QUANTIZED_NORMAL_TOLERANCE
    } else {
        NORMAL_TOLERANCE
    };
    let mut report = Report {
        faces: mesh.faces.len(),
        normal_tolerance,
        ..Report::default()
    };
    let mut used = vec![false; oracle.len()];
    let mut corner = 0;
    for (index, face) in mesh.faces.iter().enumerate() {
        let corners: Vec<DVec3> = face
            .vertices
            .iter()
            .map(|id| positions[vertex_index[id]])
            .collect();
        let found = oracle.iter().enumerate().position(|(candidate, blender)| {
            !used[candidate] && same_corners(&corners, &blender.corners)
        });
        let Some(found) = found else {
            report.unmatched_faces += 1;
            corner += corners.len();
            continue;
        };
        used[found] = true;
        let blender = &oracle[found];
        if smooth[index] != blender.smooth {
            report.smooth_mismatches += 1;
        }
        for (position, point) in corners.iter().enumerate() {
            report.corners += 1;
            let Some(at) = blender.corners.iter().position(|candidate| {
                (*point - *candidate).abs().max_element() <= POSITION_TOLERANCE
            }) else {
                report.unmatched_corners += 1;
                continue;
            };
            let error = (normals[corner + position] - blender.normals[at])
                .abs()
                .max_element();
            if normals[corner + position].dot(blender.normals[at]) < 0.0 {
                report.flipped_corners += 1;
            }
            report.max_normal_error = report.max_normal_error.max(error);
        }
        corner += corners.len();
    }
    report.edges = mesh.edges.len();
    let mut used_edges = vec![false; oracle_edges.len()];
    for (index, edge) in mesh.edges.iter().enumerate() {
        let ends = edge.vertices.map(|id| positions[vertex_index[&id]]);
        let found = oracle_edges
            .iter()
            .enumerate()
            .position(|(candidate, blender)| {
                let forward = (ends[0] - blender.ends[0]).abs().max_element() <= POSITION_TOLERANCE
                    && (ends[1] - blender.ends[1]).abs().max_element() <= POSITION_TOLERANCE;
                let backward = (ends[0] - blender.ends[1]).abs().max_element()
                    <= POSITION_TOLERANCE
                    && (ends[1] - blender.ends[0]).abs().max_element() <= POSITION_TOLERANCE;
                !used_edges[candidate] && (forward || backward)
            });
        match found {
            None => report.unmatched_edges += 1,
            Some(candidate) => {
                used_edges[candidate] = true;
                if oracle_edges[candidate].sharp != sharp[index] {
                    report.sharp_mismatches += 1;
                    report.sharp_details.push(format!(
                        "{:?}-{:?} potter={} blender={}",
                        ends[0], ends[1], sharp[index], oracle_edges[candidate].sharp
                    ));
                }
            }
        }
    }
    report.unused_oracle_faces = used.iter().filter(|used| !**used).count();
    Ok(report)
}

/// Evaluates one fixture and compares it with Blender.
fn evaluate_fixture(name: &str) -> TestResult<Option<Report>> {
    let fixture = fixture(name)?;
    if fixture.get("error").is_some() {
        return Ok(None);
    }
    let doc = scene(&fixture)?;
    // Blender's build fixture is evaluated at frame 4, where the build is complete.
    let frame = if name == "build_cube" { 4.0 } else { 1.0 };
    let context = EvaluationContext {
        frame: Some(frame),
        ..EvaluationContext::default()
    };
    let snapshot = Snapshot::evaluate(&doc, &context)?;
    let mesh = snapshot
        .meshes
        .get(&Id::new("subject")?)
        .ok_or("subject mesh is missing")?;
    Ok(Some(compare(&fixture, mesh)?))
}

/// Prints every fixture's numbers; the exact list is asserted separately.
#[test]
fn report_all_shading_fixtures() -> TestResult {
    for name in fixture_names()? {
        match evaluate_fixture(&name) {
            Ok(Some(report)) => {
                for detail in &report.sharp_details {
                    eprintln!("PR4 {name} sharp: {detail}");
                }
                eprintln!(
                    "PR4 {name}: exact={} potter_faces={} unmatched={} unused_blender={} smooth_mismatch={} potter_edges={} unmatched={} sharp_mismatch={} corners={} unmatched={} flipped={} max_normal_error={:.2e}",
                    report.exact(),
                    report.faces,
                    report.unmatched_faces,
                    report.unused_oracle_faces,
                    report.smooth_mismatches,
                    report.edges,
                    report.unmatched_edges,
                    report.sharp_mismatches,
                    report.corners,
                    report.unmatched_corners,
                    report.flipped_corners,
                    report.max_normal_error,
                );
            }
            Ok(None) => eprintln!("PR4 {name}: oracle refused"),
            Err(error) => eprintln!("PR4 {name}: error {error}"),
        }
    }
    Ok(())
}

#[test]
fn exact_shading_fixtures_match_blender() {
    let mut failures = Vec::new();
    for name in EXACT {
        match evaluate_fixture(name) {
            Ok(Some(report)) if report.exact() => {}
            Ok(Some(report)) => failures.push(format!("{name}: {report:?}")),
            Ok(None) => failures.push(format!("{name}: fixture refused by Blender")),
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
