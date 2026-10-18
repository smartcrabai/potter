//! Solidify parity with Blender 5.2.2 over `tests/fixtures/solidify/`, written by `generate.py`.
//!
//! Each case has an input (`inputs/<case>.json`: a `Mesh` in serde shape, the complete Solidify
//! parameter set, and the object's material slot count) and Blender's evaluated mesh for the same
//! geometry (`expected/<case>.json`). Potter evaluates the input through `Snapshot::evaluate`, the
//! scene path the CLI uses, where the node's material list sets the evaluated slot count. Input
//! edges are listed in Blender's own edge order, as a `.blend` import gives Potter, because the
//! neighbour order around a vertex decides where a non-manifold edge group starts.
//!
//! Blender and Potter order vertices, edges, and faces differently, so elements are matched by
//! position: vertices by position and vertex-group weights; edges by endpoint positions (either
//! order), sharp/seam flags, crease, and bevel weight; faces by their position ring in loop order,
//! starting at the same corner, with material, smooth flag, UV corners, named corner colors, and
//! custom-normal corners. Counts are exact.
//!
//! Tolerances: positions and UVs are compared at 1e-6. Group weights, colors, creases, and bevel
//! weights are compared after f32 rounding. Blender's recorded output-normal codec error gets an
//! additional 1e-6 for evaluation rounding.
//!
//! Blender refused no case in the current corpus; a case recorded as `"outcome": "error"` must make
//! Potter return an error.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use crate::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{DataBlock, Id, Material, Modifier, ModifierRuntime, Node, SceneDoc, Transform},
};
use glam::DVec3;
use serde_json::{Map, Value};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "8c2f1d4e-6a3b-4e9f-b1c7-2d5a8e0f3b6c";
const VALUE_TOLERANCE: f64 = 1.0e-6;
const NORMAL_ROUNDING_SLACK: f64 = 1.0e-6;

fn fixture_dir(directory: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/solidify")
        .join(directory)
}

/// Case names of one fixture directory, sorted.
fn case_names(directory: &str) -> TestResult<Vec<String>> {
    let mut names = Vec::new();
    for entry in fs::read_dir(fixture_dir(directory))? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            names.push(
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or("case file name")?
                    .to_owned(),
            );
        }
    }
    names.sort();
    Ok(names)
}

fn load(directory: &str, name: &str) -> TestResult<Value> {
    let path = fixture_dir(directory).join(format!("{name}.json"));
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn number(value: &Value) -> TestResult<f64> {
    Ok(value.as_f64().ok_or("number expected")?)
}

fn vector(value: &Value) -> TestResult<DVec3> {
    match value.as_array().map(Vec::as_slice) {
        Some([x, y, z]) => Ok(DVec3::new(number(x)?, number(y)?, number(z)?)),
        _ => Err("vector must have three coordinates".into()),
    }
}

fn vector_list(value: &Value) -> TestResult<Vec<DVec3>> {
    value
        .as_array()
        .ok_or("vector list expected")?
        .iter()
        .map(vector)
        .collect()
}

fn pair(value: &Value) -> TestResult<[f64; 2]> {
    match value.as_array().map(Vec::as_slice) {
        Some([u, v]) => Ok([number(u)?, number(v)?]),
        _ => Err("UV must be [u, v]".into()),
    }
}
fn rgba(value: &Value) -> TestResult<[f64; 4]> {
    match value.as_array().map(Vec::as_slice) {
        Some([red, green, blue, alpha]) => {
            Ok([number(red)?, number(green)?, number(blue)?, number(alpha)?])
        }
        _ => Err("color must be [red, green, blue, alpha]".into()),
    }
}

fn near(a: DVec3, b: DVec3) -> bool {
    (a - b).abs().max_element() <= VALUE_TOLERANCE
}

fn near_normal(a: DVec3, b: DVec3, tolerance: f64) -> bool {
    (a - b).abs().max_element() <= tolerance
}

fn near_uv(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() <= VALUE_TOLERANCE && (a[1] - b[1]).abs() <= VALUE_TOLERANCE
}

/// Modifier scene: the case's mesh as `subject` with its Solidify stack and `material_slots` materials.
fn scene(input: &Value) -> TestResult<SceneDoc> {
    let mesh: Mesh = serde_json::from_value(input["mesh"].clone())?;
    let modifier = Modifier {
        id: Id::new("solidify")?,
        modifier_type: "solidify".to_owned(),
        name: "solidify".to_owned(),
        enabled: true,
        params: input["modifier"]["params"]
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
        binding_data: None,
        show_viewport: true,
        show_render: true,
        runtime: ModifierRuntime::default(),
    };
    let slot_count = input["material_slots"].as_u64().ok_or("material_slots")?;
    let slot_layout = input["material_slot_layout"]
        .as_array()
        .ok_or("material_slot_layout")?;
    assert_eq!(u64::try_from(slot_layout.len())?, slot_count);
    let materials = slot_layout
        .iter()
        .enumerate()
        .map(|(slot, occupied)| -> TestResult<Option<Id>> {
            if occupied.is_null() {
                Ok(None)
            } else {
                Ok(Some(Id::new(format!("material_{slot}"))?))
            }
        })
        .collect::<TestResult<Vec<_>>>()?;
    let mut doc = SceneDoc::new(SCENE_ID.to_owned());
    for material in materials.iter().flatten() {
        doc.materials.insert(
            material.clone(),
            Material {
                name: material.as_str().to_owned(),
                ..Material::default()
            },
        );
    }
    let subject = Id::new("subject")?;
    let data = Id::new("data_subject")?;
    doc.data_blocks.insert(
        data.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(mesh),
            ..DataBlock::default()
        },
    );
    doc.nodes.insert(
        subject.clone(),
        Node {
            name: "subject".to_owned(),
            kind: "mesh".to_owned(),
            transform: Transform::default(),
            modifiers: vec![modifier],
            data: Some(data),
            materials,
            ..Node::default()
        },
    );
    doc.collections
        .get_mut(&Id::new("collection_root")?)
        .ok_or("root collection missing")?
        .objects
        .push(subject);
    doc.schema_version = 1;
    let mut document = serde_json::to_value(doc)?;
    crate::model::upgrade_scene(&mut document)?;
    let doc: SceneDoc = serde_json::from_value(document)?;
    doc.validate()?;
    Ok(doc)
}

/// The evaluated `subject` mesh, or the error Potter reports for the stack.
fn evaluate(input: &Value) -> TestResult<Mesh> {
    let doc = scene(input)?;
    let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
    Ok(snapshot
        .meshes
        .get(&Id::new("subject")?)
        .cloned()
        .ok_or("subject mesh missing")?)
}

/// Keyed element value of a Potter attribute (`values` map keyed `f{id}`, `e{id}`).
fn keyed<'a>(mesh: &'a Mesh, name: &str, key: &str) -> Option<&'a Value> {
    mesh.attributes.get(name)?.get("values")?.get(key)
}

fn flag(mesh: &Mesh, name: &str, key: &str) -> bool {
    keyed(mesh, name, key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn group_weight(mesh: &Mesh, group: &str, vertex: u32) -> f64 {
    let key = format!("v{vertex}");
    mesh.attributes
        .get("vertex_groups")
        .and_then(|groups| groups.get(group))
        .and_then(|weights| weights.get(key.as_str()))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

/// One face's corners in ring order, with the values Blender records per corner.
struct Corners {
    positions: Vec<DVec3>,
    material: i64,
    smooth: bool,
    uv: BTreeMap<String, Vec<[f64; 2]>>,
    colors: BTreeMap<String, Vec<[f64; 4]>>,
    normals: Option<Vec<DVec3>>,
}

struct Edge {
    ends: [DVec3; 2],
    sharp: bool,
    seam: bool,
    crease: f64,
    bevel_weight: f64,
}

fn potter_corners(mesh: &Mesh) -> TestResult<Vec<Corners>> {
    let vertex_positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let positions: BTreeMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let normals_present = crate::geom::clnor::corner_codes(mesh)?.is_some();
    let normals = if normals_present {
        Some(crate::geom::normals::corner_normals(
            mesh,
            &vertex_positions,
        )?)
    } else {
        None
    };
    let uv_store = crate::geom::uv_layers::read(mesh);
    let color_store = crate::geom::color_layers::read(mesh);
    let mut corners = Vec::new();
    let mut loop_start = 0;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let count = face.vertices.len();
        let uv = uv_store
            .layers
            .iter()
            .map(|layer| -> TestResult<(String, Vec<[f64; 2]>)> {
                let values = layer
                    .faces
                    .get(face_index)
                    .and_then(Option::as_ref)
                    .ok_or("UV face values missing")?;
                Ok((
                    layer.name.clone(),
                    values.iter().map(|[u, v]| [*u, 1.0 - *v]).collect(),
                ))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let colors = color_store
            .layers
            .iter()
            .filter(|layer| layer.domain == crate::geom::color_layers::ColorDomain::Corner)
            .map(|layer| -> TestResult<(String, Vec<[f64; 4]>)> {
                let values = layer
                    .values
                    .get(loop_start..loop_start + count)
                    .ok_or("color layer is shorter than the mesh loops")?;
                Ok((layer.name.clone(), values.to_vec()))
            })
            .collect::<TestResult<BTreeMap<_, _>>>()?;
        let face_normals = normals
            .as_ref()
            .map(|normals| {
                normals
                    .get(loop_start..loop_start + count)
                    .map(<[DVec3]>::to_vec)
                    .ok_or("custom-normal layer is shorter than the mesh loops")
            })
            .transpose()?;
        let key = format!("f{}", face.id);
        corners.push(Corners {
            positions: face
                .vertices
                .iter()
                .map(|id| positions.get(id).copied().ok_or("face vertex missing"))
                .collect::<Result<Vec<_>, _>>()?,
            material: i64::from(face.material_index),
            smooth: flag(mesh, "shade_smooth", &key),
            uv,
            colors,
            normals: face_normals,
        });
        loop_start += count;
    }
    Ok(corners)
}
fn expected_corners(output: &Value, vertices: &[DVec3]) -> TestResult<Vec<Corners>> {
    output["faces"]
        .as_array()
        .ok_or("faces")?
        .iter()
        .map(|face| -> TestResult<Corners> {
            let positions = face["v"]
                .as_array()
                .ok_or("face vertices")?
                .iter()
                .map(|index| -> TestResult<DVec3> {
                    let index = usize::try_from(index.as_u64().ok_or("vertex index")?)?;
                    Ok(*vertices.get(index).ok_or("vertex index out of range")?)
                })
                .collect::<TestResult<Vec<_>>>()?;
            let uv = face["uv"]
                .as_object()
                .ok_or("face uv")?
                .iter()
                .map(|(name, values)| -> TestResult<(String, Vec<[f64; 2]>)> {
                    let values = values
                        .as_array()
                        .ok_or("uv values")?
                        .iter()
                        .map(pair)
                        .collect::<TestResult<Vec<_>>>()?;
                    Ok((name.clone(), values))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()?;
            let colors = face["colors"]
                .as_object()
                .ok_or("face colors")?
                .iter()
                .map(|(name, values)| -> TestResult<(String, Vec<[f64; 4]>)> {
                    let values = values
                        .as_array()
                        .ok_or("color values")?
                        .iter()
                        .map(rgba)
                        .collect::<TestResult<Vec<_>>>()?;
                    Ok((name.clone(), values))
                })
                .collect::<TestResult<BTreeMap<_, _>>>()?;
            Ok(Corners {
                positions,
                material: face["material_index"].as_i64().ok_or("material_index")?,
                smooth: face["smooth"].as_bool().ok_or("smooth")?,
                uv,
                colors,
                normals: face.get("normals").map(vector_list).transpose()?,
            })
        })
        .collect()
}

/// Whether two faces match corner for corner in loop order, including the ring start. Blender's
/// loop start decides each output quad's diagonal, so a rotated ring is a different face here.
fn corners_same(potter: &Corners, expected: &Corners, normal_tolerance: f64) -> bool {
    let count = expected.positions.len();
    if potter.positions.len() != count
        || potter.material != expected.material
        || potter.smooth != expected.smooth
    {
        return false;
    }
    let rings = (0..count).all(|k| near(potter.positions[k], expected.positions[k]));
    let uvs = expected.uv.iter().all(|(name, values)| {
        potter.uv.get(name).is_some_and(|mine| {
            (0..count).all(|k| match (mine.get(k), values.get(k)) {
                (Some(a), Some(b)) => near_uv(*a, *b),
                _ => false,
            })
        })
    });
    let colors = expected.colors.iter().all(|(name, values)| {
        potter.colors.get(name).is_some_and(|mine| {
            (0..count).all(|k| match (mine.get(k), values.get(k)) {
                (Some(a), Some(b)) => a.iter().zip(b).all(|(left, right)| same_f32(*left, *right)),
                _ => false,
            })
        })
    });
    let normals = match (&expected.normals, &potter.normals) {
        (None, _) => true,
        (Some(want), Some(mine)) => (0..count).all(|k| match (mine.get(k), want.get(k)) {
            (Some(a), Some(b)) => near_normal(*a, *b, normal_tolerance),
            _ => false,
        }),
        (Some(_), None) => false,
    };
    rings && uvs && colors && normals
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores group/bevel weights, colors, and creases as f32"
)]
fn same_f32(a: f64, b: f64) -> bool {
    (a as f32).to_bits() == (b as f32).to_bits()
}

fn signature_same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).enumerate().all(|(index, (x, y))| {
            if index < 3 {
                (x - y).abs() <= VALUE_TOLERANCE
            } else {
                same_f32(*x, *y)
            }
        })
}

fn edge_same(a: &Edge, b: &Edge) -> bool {
    let forward = near(a.ends[0], b.ends[0]) && near(a.ends[1], b.ends[1]);
    let backward = near(a.ends[0], b.ends[1]) && near(a.ends[1], b.ends[0]);
    a.sharp == b.sharp
        && a.seam == b.seam
        && same_f32(a.crease, b.crease)
        && same_f32(a.bevel_weight, b.bevel_weight)
        && (forward || backward)
}

/// Index of the first `left` item that cannot take a distinct `right` item, or `None` when every
/// `left` item pairs with its own `right` item. Greedy in order, so identical items pair in order.
fn first_unmatched<A, B>(left: &[A], right: &[B], same: impl Fn(&A, &B) -> bool) -> Option<usize> {
    let mut used = vec![false; right.len()];
    for (index, item) in left.iter().enumerate() {
        let found = right
            .iter()
            .enumerate()
            .position(|(other, candidate)| !used[other] && same(item, candidate));
        match found {
            Some(other) => used[other] = true,
            None => return Some(index),
        }
    }
    None
}

/// Loose edges: mesh edges that no face boundary uses.
fn loose_edge_count(mesh: &Mesh) -> usize {
    let mut on_faces = BTreeSet::new();
    for face in &mesh.faces {
        for (index, id) in face.vertices.iter().enumerate() {
            let next = face.vertices[(index + 1) % face.vertices.len()];
            on_faces.insert(((*id).min(next), (*id).max(next)));
        }
    }
    mesh.edges
        .iter()
        .filter(|edge| {
            let [a, b] = edge.vertices;
            !on_faces.contains(&(a.min(b), a.max(b)))
        })
        .count()
}

/// Problems with `mesh` relative to Blender's record of the case; empty when it matches.
fn compare(expected: &Value, mesh: &Mesh) -> TestResult<Vec<String>> {
    let output = &expected["output"];
    let normal_tolerance = if output["has_custom_normals"].as_bool() == Some(true) {
        number(&expected["output_normal_codec_max_error"])? + NORMAL_ROUNDING_SLACK
    } else {
        0.0
    };
    let mut problems = Vec::new();

    let topology = &output["topology"];
    let loops: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    let counts = [
        ("vertices", mesh.vertices.len()),
        ("edges", mesh.edges.len()),
        ("faces", mesh.faces.len()),
        ("loops", loops),
        ("loose_edges", loose_edge_count(mesh)),
    ];
    for (key, actual) in counts {
        let want = topology[key].as_u64().ok_or("topology count")?;
        if u64::try_from(actual)? != want {
            problems.push(format!("{key}: Potter {actual}, Blender {want}"));
        }
    }

    let mut layer_names: Vec<String> = crate::geom::uv_layers::read(mesh)
        .layers
        .into_iter()
        .map(|layer| layer.name)
        .collect();
    let mut blender_layers: Vec<String> = output["uv_layer_names"]
        .as_array()
        .ok_or("uv_layer_names")?
        .iter()
        .map(|name| -> TestResult<String> { Ok(name.as_str().ok_or("layer name")?.to_owned()) })
        .collect::<TestResult<_>>()?;
    layer_names.sort();
    blender_layers.sort();
    if layer_names != blender_layers {
        problems.push(format!(
            "UV layers: Potter {layer_names:?}, Blender {blender_layers:?}"
        ));
    }
    let mut color_names: Vec<String> = crate::geom::color_layers::read(mesh)
        .layers
        .into_iter()
        .filter(|layer| layer.domain == crate::geom::color_layers::ColorDomain::Corner)
        .map(|layer| layer.name)
        .collect();
    let mut blender_color_names: Vec<String> = output["color_attribute_names"]
        .as_array()
        .ok_or("color_attribute_names")?
        .iter()
        .map(|name| -> TestResult<String> {
            Ok(name.as_str().ok_or("color attribute name")?.to_owned())
        })
        .collect::<TestResult<_>>()?;
    color_names.sort();
    blender_color_names.sort();
    if color_names != blender_color_names {
        problems.push(format!(
            "named corner colors: Potter {color_names:?}, Blender {blender_color_names:?}"
        ));
    }

    let has_custom_normals = crate::geom::clnor::corner_codes(mesh)?.is_some();
    if output["has_custom_normals"].as_bool() != Some(has_custom_normals) {
        problems.push(format!(
            "custom-normal layer: Potter {has_custom_normals}, Blender {}",
            output["has_custom_normals"]
        ));
    }

    let has_bevel_weight_edge = mesh.attributes.contains_key("bevel_weight_edge");
    if output["has_bevel_weight_edge"].as_bool() != Some(has_bevel_weight_edge) {
        problems.push(format!(
            "bevel_weight_edge attribute: Potter {has_bevel_weight_edge}, Blender {}",
            output["has_bevel_weight_edge"]
        ));
    }

    if !problems.is_empty() {
        return Ok(problems);
    }

    // Vertices: position plus the weight of every group Blender recorded, matched as a multiset.
    let vertices = output["vertices"]
        .as_array()
        .ok_or("vertices")?
        .iter()
        .map(vector)
        .collect::<TestResult<Vec<_>>>()?;
    let groups = output["vertex_groups"].as_object().ok_or("vertex_groups")?;
    let group_names: Vec<String> = groups.keys().cloned().collect();
    let mut blender_signatures = Vec::new();
    for (index, position) in vertices.iter().enumerate() {
        let mut signature = vec![position.x, position.y, position.z];
        for name in &group_names {
            let weight = groups[name]
                .as_array()
                .and_then(|weights| weights.get(index))
                .and_then(Value::as_f64)
                .ok_or("group weight")?;
            signature.push(weight);
        }
        blender_signatures.push(signature);
    }
    let potter_signatures: Vec<Vec<f64>> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let mut signature = vec![vertex.co.x, vertex.co.y, vertex.co.z];
            signature.extend(
                group_names
                    .iter()
                    .map(|name| group_weight(mesh, name, vertex.id)),
            );
            signature
        })
        .collect();
    if let Some(index) = first_unmatched(&potter_signatures, &blender_signatures, |a, b| {
        signature_same(a, b)
    }) {
        problems.push(format!(
            "vertex {index} {:?} has no Blender counterpart",
            potter_signatures[index]
        ));
    }

    let positions: BTreeMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let potter_edges = mesh
        .edges
        .iter()
        .map(|edge| -> TestResult<Edge> {
            let end = |id: u32| positions.get(&id).copied().ok_or("edge vertex missing");
            Ok(Edge {
                ends: [end(edge.vertices[0])?, end(edge.vertices[1])?],
                sharp: flag(mesh, "sharp", &format!("e{}", edge.id)),
                seam: flag(mesh, "seam", &format!("e{}", edge.id)),
                crease: keyed(mesh, "crease_edge", &format!("e{}", edge.id))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                bevel_weight: keyed(mesh, "bevel_weight_edge", &format!("e{}", edge.id))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let blender_edges = output["edges"]
        .as_array()
        .ok_or("edges")?
        .iter()
        .map(|edge| -> TestResult<Edge> {
            let ends = edge["v"].as_array().ok_or("edge vertices")?;
            let end = |value: Option<&Value>| -> TestResult<DVec3> {
                let index = usize::try_from(value.and_then(Value::as_u64).ok_or("edge index")?)?;
                Ok(*vertices.get(index).ok_or("edge index out of range")?)
            };
            Ok(Edge {
                ends: [end(ends.first())?, end(ends.get(1))?],
                sharp: edge["sharp"].as_bool().ok_or("sharp")?,
                seam: edge["seam"].as_bool().ok_or("seam")?,
                crease: number(&edge["crease"])?,
                bevel_weight: number(&edge["bevel_weight"])?,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    if let Some(index) = first_unmatched(&potter_edges, &blender_edges, edge_same) {
        let edge = &potter_edges[index];
        problems.push(format!(
            "edge {index} {:?}-{:?} (sharp {}, seam {}, crease {}, bevel weight {}) has no Blender counterpart",
            edge.ends[0], edge.ends[1], edge.sharp, edge.seam, edge.crease, edge.bevel_weight
        ));
    }

    let potter_faces = potter_corners(mesh)?;
    let blender_faces = expected_corners(output, &vertices)?;
    if let Some(index) = first_unmatched(&potter_faces, &blender_faces, |a, b| {
        corners_same(a, b, normal_tolerance)
    }) {
        problems.push(format!(
            "face {index} {:?} has no Blender counterpart",
            potter_faces[index].positions
        ));
    }
    Ok(problems)
}

#[test]
fn solidify_matches_blender_on_every_case() -> TestResult {
    let mut failures = Vec::new();
    for name in case_names("inputs")? {
        let input = load("inputs", &name)?;
        let expected = load("expected", &name)?;
        match (expected["outcome"].as_str(), evaluate(&input)) {
            (Some("error"), Err(_)) => {}
            (Some("error"), Ok(_)) => {
                failures.push(format!(
                    "{name}: Blender refused the case; Potter evaluated it"
                ));
            }
            (Some("ok"), Ok(mesh)) => {
                for problem in compare(&expected, &mesh)? {
                    failures.push(format!("{name}: {problem}"));
                }
            }
            (Some("ok"), Err(error)) => {
                failures.push(format!(
                    "{name}: Potter refused a case Blender evaluates: {error}"
                ));
            }
            _ => return Err(format!("{name}: expected outcome missing").into()),
        }
    }
    assert!(
        failures.is_empty(),
        "{} Solidify mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn solidify_inputs_and_expected_outputs_pair_up() -> TestResult {
    let inputs = case_names("inputs")?;
    assert!(!inputs.is_empty(), "no Solidify cases");
    assert_eq!(inputs, case_names("expected")?);
    for name in &inputs {
        let input = load("inputs", name)?;
        let slot_count = input["material_slots"].as_u64().ok_or("material_slots")?;
        let slot_layout = input["material_slot_layout"]
            .as_array()
            .ok_or("material_slot_layout")?;
        assert_eq!(u64::try_from(slot_layout.len())?, slot_count, "{name}");
        if name == "grid_nm_null_slot_group" {
            assert_eq!(
                slot_layout,
                &vec![Value::Bool(true), Value::Null, Value::Bool(true)]
            );
        }
        let params: &Map<String, Value> = input["modifier"]["params"]
            .as_object()
            .ok_or("modifier params")?;
        assert_eq!(
            input["modifier"]["type"].as_str(),
            Some("solidify"),
            "{name}"
        );
        assert!(
            params.contains_key("solidify_mode"),
            "{name} is missing solidify_mode"
        );
    }
    Ok(())
}
