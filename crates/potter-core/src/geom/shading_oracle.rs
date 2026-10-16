// SPDX-License-Identifier: GPL-3.0-or-later

//! PR4 shading oracle: smooth flags, sharp edges and corner normals through mesh-rebuilding
//! modifiers, against Blender 5.2.2. `tests/fixtures/pr4_shading/*.json` are written by
//! `generate.py` from Blender; this test replays each fixture's input and stack through the
//! evaluator and matches output elements geometrically (corner positions), never by index.
//! Corner normals are compared at 1e-5 and flags exactly. Fixtures outside [`EXACT`] are classes
//! whose Blender rule is not reproduced yet; [`report_all`] prints their numbers.

use std::{collections::HashMap, error::Error, fs, path::Path};

use glam::DVec3;
use serde_json::{Map, Value, json};

use super::{
    Mesh,
    normals::{corner_normals, sharp_edge_flags, smooth_face_flags},
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

/// Fixtures whose stack Potter refuses with a typed feature, as `(fixture, feature_id)`.
const REFUSED: &[(&str, &str)] = &[
    ("mirror_cube_bisect_flip", "modifier.mirror.bisect"),
    (
        "remesh_smooth_mode_grid",
        "modifier.remesh.open_nonmanifold",
    ),
];

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
    "decimate_grid",
    "edge_split_cube",
    "edge_split_sharp_only",
    "mask_grid",
    "mirror_cube",
    "mirror_cube_nocustom",
    "mirror_cube_nomerge",
    "mirror_grid",
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

fn point(value: &Value) -> TestResult<DVec3> {
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
        .map(point)
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
    if let Some(custom) = input["custom_normals"].as_array() {
        let values: Map<String, Value> = custom
            .iter()
            .enumerate()
            .map(|(index, corners)| (format!("f{index}"), corners.clone()))
            .collect();
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({"domain":"corner","type":"float3","values":values}),
        );
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

/// Scene with the fixture's subject (carrying the stack) and its referenced objects as nodes.
fn scene(fixture: &Value) -> TestResult<SceneDoc> {
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    let root = Id::new("collection_root")?;
    let empty = Map::new();
    let objects = fixture["objects"].as_object().unwrap_or(&empty);
    let mut nodes = vec![(
        "subject".to_owned(),
        input_mesh(&fixture["input"])?,
        [0.0; 3],
        modifiers(&fixture["stack"], objects)?,
    )];
    for (name, object) in objects {
        let translation = object["translation"]
            .as_array()
            .map(|values| -> TestResult<[f64; 3]> {
                let [x, y, z] = values.as_slice() else {
                    return Err("translation".into());
                };
                Ok([
                    x.as_f64().ok_or("translation")?,
                    y.as_f64().ok_or("translation")?,
                    z.as_f64().ok_or("translation")?,
                ])
            })
            .transpose()?
            .unwrap_or([0.0; 3]);
        nodes.push((slug(name), input_mesh(object)?, translation, Vec::new()));
    }
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
struct Report {
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
    fn exact(&self) -> bool {
        self.unmatched_faces == 0
            && self.unused_oracle_faces == 0
            && self.smooth_mismatches == 0
            && self.unmatched_edges == 0
            && self.sharp_mismatches == 0
            && self.unmatched_corners == 0
            && self.max_normal_error <= self.normal_tolerance
    }
}

fn compare(fixture: &Value, mesh: &Mesh) -> TestResult<Report> {
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
fn exact_shading_fixtures_match_blender() -> TestResult {
    let mut failures = Vec::new();
    for name in EXACT {
        match evaluate_fixture(name)? {
            Some(report) if report.exact() => {}
            Some(report) => failures.push(format!("{name}: {report:?}")),
            None => failures.push(format!("{name}: fixture refused by Blender")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // Blender's output for these stacks is evaluated by no modifier yet, so Potter refuses them with
    // the feature that names the gap instead of a silent difference.
    for (name, feature) in REFUSED {
        let Err(error) = evaluate_fixture(name) else {
            return Err(format!("{name} must be refused").into());
        };
        let feature_id = error
            .downcast_ref::<crate::error::PotError>()
            .and_then(|error| error.details.get("feature_id"))
            .and_then(serde_json::Value::as_str);
        assert_eq!(feature_id, Some(*feature), "{name}");
    }
    Ok(())
}
