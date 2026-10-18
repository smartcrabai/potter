// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender 5.2.2 (https://github.com/blender/blender/tree/v5.2.2):
//   source/blender/modifiers/intern/MOD_boolean.cc, SPDX-FileCopyrightText: 2005 Blender Authors.
//   source/blender/bmesh/tools/bmesh_intersect.cc, SPDX-FileCopyrightText: 2023 Blender Authors.
//   Both upstream files are GPL-2.0-or-later.
//! Blender's FLOAT Boolean, ported from `MOD_boolean.cc` and `bmesh_intersect.cc`.
//!
//! The kernel deliberately keeps its predicates, intersection construction, face-group ray tests,
//! and output polygons in `f32`. It folds collection members pairwise, just as the FLOAT modifier
//! does. Intersection defects on coincident and touching geometry are part of this solver's
//! behavior; this is not an exact-predicate fallback.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{PotError, Result};
use glam::{DMat4, DVec3, Mat3, Mat4};
use serde_json::{Value, json};

use crate::geom::{
    Mesh, attribute_keys, clnor,
    color_layers::{self, ColorDomain},
    concat::{self, Part},
    edge_key,
    provenance::{Origin, default_value, remap_attributes},
    triangulate::{BeautyScratch, beautify_with_scratch},
    uv_layers,
};

use super::dissolve::Poly;
use super::imesh::NO_INDEX;
use super::{BooleanOperand, Operation};

type P3 = [f32; 3];
type EdgeKey = (usize, usize);

const FLT_EPSILON: f32 = f32::EPSILON;

#[derive(Clone)]
struct Vertex {
    co: P3,
    /// Joined index in the subject-first output attribute table, or `NO_INDEX` for generated data.
    source: i32,
}

#[derive(Clone)]
struct Face {
    verts: Vec<usize>,
    side: u8,
    source: i32,
}

#[derive(Clone, Copy)]
struct Triangle {
    verts: [usize; 3],
    face: usize,
    side: u8,
}

struct State {
    verts: Vec<Vertex>,
    faces: Vec<Face>,
    tris: Vec<Triangle>,
    /// `BMesh` input boundary edges and their FLOAT output edge provenance.
    edge_orig: HashMap<EdgeKey, i32>,
    edge_start: HashMap<EdgeKey, usize>,
    edge_order: Vec<EdgeKey>,
    input_loose_edge_sources: HashSet<i32>,
    edge_splits: HashMap<EdgeKey, Vec<usize>>,
    face_cuts: HashMap<usize, Vec<EdgeKey>>,
    wire_edges: HashSet<EdgeKey>,
    /// Split-net edges that no output face loops (Blender keeps an edge with no loops as wire).
    dangling_edges: HashSet<EdgeKey>,
    dissolve_verts: Vec<usize>,
    edge_tri_cache: HashMap<[i32; 4], usize>,
}

impl State {
    fn new() -> Self {
        Self {
            verts: Vec::new(),
            faces: Vec::new(),
            tris: Vec::new(),
            edge_orig: HashMap::new(),
            edge_start: HashMap::new(),
            edge_order: Vec::new(),
            input_loose_edge_sources: HashSet::new(),
            edge_splits: HashMap::new(),
            face_cuts: HashMap::new(),
            wire_edges: HashSet::new(),
            dissolve_verts: Vec::new(),
            edge_tri_cache: HashMap::new(),
            dangling_edges: HashSet::new(),
        }
    }

    fn add_vertex(&mut self, co: P3, source: i32) -> usize {
        let id = self.verts.len();
        self.verts.push(Vertex { co, source });
        id
    }
}

/// Blender `BM_mesh_intersect` FLOAT solver. The subject is folded with each evaluated operand in
/// collection order; operand transforms and material remaps are already resolved by the evaluator.
pub(super) fn boolean_float(
    subject: &Mesh,
    operands: &[BooleanOperand],
    operation: Operation,
    double_threshold: f32,
) -> Result<Mesh> {
    if operands.is_empty() {
        return Ok(subject.clone());
    }
    let mut result = subject.clone();
    for operand in operands {
        result = float_pair(&result, operand, operation, double_threshold)?;
    }
    Ok(result)
}

fn remap_material_index(remap: &[u32], local_index: u32) -> u32 {
    if remap.is_empty() {
        local_index
    } else {
        remap.get(local_index as usize).copied().unwrap_or(0)
    }
}

fn float_pair(
    subject: &Mesh,
    operand: &BooleanOperand,
    operation: Operation,
    eps: f32,
) -> Result<Mesh> {
    subject.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid Boolean subject mesh: {error}"))
    })?;
    operand.mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("invalid Boolean operand mesh: {error}"))
    })?;

    let transform = operand.to_subject.as_mat4();
    let flip = match operand.float_flip {
        Some(flip) => flip,
        None => Mat3::from_mat4(transform).determinant() < 0.0,
    };
    let mut operand_frame = operand.mesh.clone();
    if flip {
        reverse_mesh_faces(&mut operand_frame);
        reverse_corner_attributes(&mut operand_frame);
    }
    let operand_local_positions: Vec<P3> = operand_frame
        .vertices
        .iter()
        .map(|vertex| to_f32(vertex.co))
        .collect();
    for (index, vertex) in operand_frame.vertices.iter_mut().enumerate() {
        let co = to_f32(vertex.co);
        let transformed = transform_point(transform, co);
        if !finite3(transformed) {
            return Err(PotError::invalid_argument(format!(
                "Boolean operand vertex {index} is not finite after transform"
            )));
        }
        vertex.co = to_dvec3(transformed);
    }

    let empty_operand_faces = operand.mesh.faces.is_empty();
    let mut subject_frame = Cow::Borrowed(subject);
    if empty_operand_faces {
        for face in &mut subject_frame.to_mut().faces {
            face.material_index =
                remap_material_index(&operand.material_remap, face.material_index);
        }
    }
    for face in &mut operand_frame.faces {
        face.material_index = remap_material_index(&operand.material_remap, face.material_index);
    }
    reduce_operand_metadata(&mut operand_frame);
    reduce_operand_boolean_attributes(&mut operand_frame);

    let mut state = build_state(
        subject_frame.as_ref(),
        &operand_frame,
        &operand_frame,
        &operand_local_positions,
    )?;
    if empty_operand_faces {
        if operand_frame.vertices.is_empty() {
            for (index, vertex) in state.verts.iter_mut().enumerate() {
                let transformed = transform_point(transform, vertex.co);
                if !finite3(transformed) {
                    return Err(PotError::invalid_argument(format!(
                        "Boolean subject vertex {index} is not finite after transform"
                    )));
                }
                vertex.co = transformed;
            }
            for (vertex, state_vertex) in
                subject_frame.to_mut().vertices.iter_mut().zip(&state.verts)
            {
                vertex.co = to_dvec3(state_vertex.co);
            }
        }
        for face in &mut state.faces {
            face.side = 1;
        }
        for triangle in &mut state.tris {
            triangle.side = 1;
        }
    }
    let (bvh_a, bvh_b) = intersect_triangles(&mut state, eps);
    split_input_edges(&mut state);
    split_faces(&mut state);
    classify_faces(&mut state, operation, &bvh_a, &bvh_b);
    dissolve_intersection_vertices(&mut state);

    let has_normal_codes =
        normal_codes_present(&subject_frame) || normal_codes_present(&operand_frame);
    let mut source = joined_boolean_source(&subject_frame, &operand_frame)?;
    source.attributes.remove("custom_normal");
    let FloatOutput {
        mut mesh,
        polygons,
        state_vertex_ids,
        edge_sources,
    } = build_float_output(&subject_frame, &operand_frame, &state)?;
    let transfers = float_face_transfers(
        &subject_frame,
        &operand_frame,
        &state,
        &state_vertex_ids,
        &polygons,
        &mesh,
    )?;
    let origin = float_output_origin(&source, &state, &state_vertex_ids, &polygons, &transfers)?;
    remap_attributes(&source, &mut mesh, &origin)?;
    transfer_float_point_defaults(&source, &mut mesh, &origin.vertices);
    transfer_float_edge_attributes(&source, &mut mesh, &edge_sources);
    transfer_float_corner_attributes(&subject_frame, &operand_frame, &transfers, &mut mesh);
    if has_normal_codes {
        transfer_float_normal_codes(&subject_frame, &operand_frame, &transfers, &mut mesh)?;
    }
    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("Boolean FLOAT output is invalid: {error}"))
    })?;
    Ok(mesh)
}

struct FloatOutput {
    mesh: Mesh,
    polygons: Vec<Poly>,
    state_vertex_ids: Vec<Option<u32>>,
    edge_sources: Vec<Option<usize>>,
}

/// Blender writes each output edge from its `BMEdge` `v1`. An input edge keeps its first vertex through
/// `BM_edge_split`, and the state's `edge_start` records it. `insert_face` instead orients an edge
/// from the first face that uses it, and the next Boolean fold then splits the edge from the wrong
/// end (`edge_verts_sort` runs from `v1`). Each output edge with a recorded start takes that direction.
fn orient_output_edges(mesh: &mut Mesh, state: &State, state_vertex_ids: &[Option<u32>]) {
    let index: HashMap<(u32, u32), usize> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(position, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), position))
        .collect();
    for (&(a, b), &start) in &state.edge_start {
        let (Some(id_a), Some(id_b)) = (
            state_vertex_ids.get(a).copied().flatten(),
            state_vertex_ids.get(b).copied().flatten(),
        ) else {
            continue;
        };
        let Some(&position) = index.get(&edge_key(id_a, id_b)) else {
            continue;
        };
        mesh.edges[position].vertices = if start == a {
            [id_a, id_b]
        } else if start == b {
            [id_b, id_a]
        } else {
            continue;
        };
    }
}

fn build_float_output(subject: &Mesh, operand: &Mesh, state: &State) -> Result<FloatOutput> {
    let mut polygons = Vec::with_capacity(state.faces.len());
    for face in &state.faces {
        if face.verts.len() < 3 {
            continue;
        }
        let edge_orig = face
            .verts
            .iter()
            .enumerate()
            .map(|(corner, &a)| {
                let b = face.verts[(corner + 1) % face.verts.len()];
                state
                    .edge_orig
                    .get(&edge_key_usize(a, b))
                    .copied()
                    .unwrap_or(NO_INDEX)
            })
            .collect();
        polygons.push(Poly {
            verts: face
                .verts
                .iter()
                .map(|&vertex| {
                    u32::try_from(vertex).map_err(|_| {
                        PotError::invalid_argument("Boolean output vertex index overflow")
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            orig: face.source,
            edge_orig,
        });
    }

    let mut mesh = Mesh::new();
    let mut state_vertex_ids = vec![None; state.verts.len()];
    for polygon in &polygons {
        for &index in &polygon.verts {
            let index = index as usize;
            if state_vertex_ids[index].is_none() {
                let id = mesh
                    .insert_vertex(to_dvec3(state.verts[index].co))
                    .map_err(|error| {
                        PotError::invalid_argument(format!("Boolean output vertex failed: {error}"))
                    })?;
                state_vertex_ids[index] = Some(id);
            }
        }
    }
    let mut face_specs = Vec::with_capacity(polygons.len());
    for polygon in &polygons {
        let face_index = usize::try_from(polygon.orig)
            .map_err(|_| PotError::invalid_argument("Boolean output face has no source"))?;
        let material = if face_index < subject.faces.len() {
            subject.faces[face_index].material_index
        } else {
            operand
                .faces
                .get(face_index - subject.faces.len())
                .ok_or_else(|| PotError::invalid_argument("Boolean output face source is missing"))?
                .material_index
        };
        let ring = polygon
            .verts
            .iter()
            .map(|&vertex| {
                state_vertex_ids[vertex as usize].ok_or_else(|| {
                    PotError::invalid_argument("Boolean output face vertex is missing")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        face_specs.push((ring, material));
    }
    mesh.insert_faces(face_specs.into_iter().map(Ok))
        .map_err(|error| {
            PotError::invalid_argument(format!("Boolean output faces failed: {error}"))
        })?;

    let mut face_edges = HashSet::new();
    let mut edge_source_by_output: HashMap<(u32, u32), usize> = HashMap::new();
    for polygon in &polygons {
        for (corner, &vertex) in polygon.verts.iter().enumerate() {
            let vertex = vertex as usize;
            let next = polygon.verts[(corner + 1) % polygon.verts.len()] as usize;
            let state_edge = edge_key_usize(vertex, next);
            face_edges.insert(state_edge);
            if let Ok(source) = usize::try_from(polygon.edge_orig[corner]) {
                let out_a = state_vertex_ids[vertex].ok_or_else(|| {
                    PotError::invalid_argument("Boolean output face vertex is missing")
                })?;
                let out_b = state_vertex_ids[next].ok_or_else(|| {
                    PotError::invalid_argument("Boolean output face vertex is missing")
                })?;
                edge_source_by_output.insert(edge_key(out_a, out_b), source);
            }
        }
    }

    let mut source_face_vertices = HashSet::new();
    let subject_vertex_index: HashMap<u32, usize> = subject
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let operand_vertex_index: HashMap<u32, usize> = operand
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, subject.vertices.len() + index))
        .collect();
    for face in &subject.faces {
        for &vertex in &face.vertices {
            if let Some(&index) = subject_vertex_index.get(&vertex) {
                source_face_vertices.insert(index);
            }
        }
    }
    for face in &operand.faces {
        for &vertex in &face.vertices {
            if let Some(&index) = operand_vertex_index.get(&vertex) {
                source_face_vertices.insert(index);
            }
        }
    }
    let mut loose_edges = Vec::<(EdgeKey, Option<usize>)>::new();
    for &edge in &state.edge_order {
        if face_edges.contains(&edge) {
            continue;
        }
        let Some(&source) = state.edge_orig.get(&edge) else {
            continue;
        };
        if state.input_loose_edge_sources.contains(&source) {
            loose_edges.push((edge, usize::try_from(source).ok()));
        }
    }
    // Cut edges no output face uses, and split-net edges no output face loops, stay in the result
    // as loose edges (Blender keeps an edge that never gets a loop), in sorted order.
    let wire_candidates: HashSet<EdgeKey> = state
        .wire_edges
        .iter()
        .chain(state.dangling_edges.iter())
        .copied()
        .collect();
    let mut leftover_wire: Vec<EdgeKey> = wire_candidates
        .into_iter()
        .filter(|edge| !face_edges.contains(edge) && !loose_edges.iter().any(|(e, _)| e == edge))
        .collect();
    leftover_wire.sort_unstable();
    loose_edges.extend(leftover_wire.into_iter().map(|edge| (edge, None)));

    let mut loose_vertices: HashSet<usize> = state
        .verts
        .iter()
        .enumerate()
        .filter_map(|(state_index, vertex)| {
            usize::try_from(vertex.source)
                .ok()
                .filter(|source| !source_face_vertices.contains(source))
                .map(|_| state_index)
        })
        .collect();
    for (edge, _) in &loose_edges {
        loose_vertices.insert(edge.0);
        loose_vertices.insert(edge.1);
    }
    for (state_index, state_vertex_id) in state_vertex_ids.iter_mut().enumerate() {
        if loose_vertices.contains(&state_index) && state_vertex_id.is_none() {
            let id = mesh
                .insert_vertex(to_dvec3(state.verts[state_index].co))
                .map_err(|error| {
                    PotError::invalid_argument(format!("Boolean loose vertex failed: {error}"))
                })?;
            *state_vertex_id = Some(id);
        }
    }
    let mut output_edge_keys: HashSet<(u32, u32)> = mesh
        .edges
        .iter()
        .map(|edge| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect();
    for (edge, source) in loose_edges {
        let (Some(a), Some(b)) = (state_vertex_ids[edge.0], state_vertex_ids[edge.1]) else {
            continue;
        };
        if a == b {
            continue;
        }
        let output_edge = edge_key(a, b);
        if output_edge_keys.insert(output_edge) {
            mesh.insert_edge([a, b]).map_err(|error| {
                PotError::invalid_argument(format!("Boolean loose edge failed: {error}"))
            })?;
        }
        if let Some(source) = source {
            edge_source_by_output.insert(output_edge, source);
        }
    }
    orient_output_edges(&mut mesh, state, &state_vertex_ids);
    let edge_sources = mesh
        .edges
        .iter()
        .map(|edge| {
            edge_source_by_output
                .get(&edge_key(edge.vertices[0], edge.vertices[1]))
                .copied()
        })
        .collect();
    Ok(FloatOutput {
        mesh,
        polygons,
        state_vertex_ids,
        edge_sources,
    })
}

fn material_slot_count(mesh: &Mesh) -> usize {
    mesh.faces
        .iter()
        .map(|face| face.material_index as usize + 1)
        .max()
        .unwrap_or(1)
        .max(1)
}

fn reduce_operand_metadata(mesh: &mut Mesh) {
    let color_types = mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .cloned();
    if let Some(color_types) = color_types {
        mesh.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"custom_properties": {}, "color_data_types": color_types}),
        );
    } else {
        mesh.attributes.remove("blender_metadata");
    }
}
fn reduce_operand_boolean_attributes(mesh: &mut Mesh) {
    mesh.attributes.retain(|key, value| {
        matches!(
            key.as_str(),
            "blender_attributes"
                | "blender_metadata"
                | "blender_edge_flags"
                | "blender_polygon_smooth"
                | "blender_skin_vertices"
                | "blender_uv_layers"
                | "color_layers"
                | "crease_edge"
                | "custom_normal"
                | "shade_smooth"
                | "seam"
                | "sharp"
                | "skin_radii"
                | "skin_roots"
                | "uv_layers"
                | "uv_map"
                | "vertex_groups"
        ) || (value.get("values").is_some()
            && value
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| {
                    ["point", "edge", "face", "corner"]
                        .into_iter()
                        .any(|candidate| domain.eq_ignore_ascii_case(candidate))
                }))
    });
}

fn joined_boolean_source(subject: &Mesh, operand: &Mesh) -> Result<Mesh> {
    let mut source = concat::concatenate(vec![
        Part {
            mesh: Cow::Borrowed(subject),
            matrix: DMat4::IDENTITY,
            material_offset: 0,
            slot_count: material_slot_count(subject),
        },
        Part {
            mesh: Cow::Borrowed(operand),
            matrix: DMat4::IDENTITY,
            material_offset: 0,
            slot_count: material_slot_count(operand),
        },
    ])?;
    transfer_float_joined_defaults(subject, operand, &mut source);
    // Blender seeds active/render custom-data layers from the subject mesh.
    let subject_uv = uv_layers::read(subject);
    let mut source_uv = uv_layers::read(&source);
    source_uv.active = subject_uv.active;
    source_uv.render = subject_uv.render;
    uv_layers::write(&mut source, &source_uv);
    let subject_colors = color_layers::read(subject);
    let mut source_colors = color_layers::read(&source);
    source_colors.active = subject_colors.active;
    source_colors.render = subject_colors.render;
    color_layers::write(&mut source, &source_colors);
    Ok(source)
}

fn transfer_float_joined_defaults(subject: &Mesh, operand: &Mesh, source: &mut Mesh) {
    let named: Vec<(String, String, String, Value)> = source
        .attributes
        .iter()
        .filter_map(|(key, attribute)| {
            let data_type = attribute
                .get("type")
                .and_then(Value::as_str)
                .map(blender_mixer_type)?;
            let default = match data_type {
                "FLOAT_COLOR" | "BYTE_COLOR" => json!([1.0, 1.0, 1.0, 1.0]),
                "INT16_2D" => json!([0, 0]),
                "FLOAT4" => json!([0.0, 0.0, 0.0, 0.0]),
                "FLOAT4X4" => json!([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]),
                "STRING" => json!(""),
                _ => return None,
            };
            let domain = attribute.get("domain").and_then(Value::as_str)?;
            ["point", "edge", "face", "corner"]
                .into_iter()
                .any(|candidate| domain.eq_ignore_ascii_case(candidate))
                .then(|| {
                    (
                        key.clone(),
                        domain.to_owned(),
                        data_type.to_owned(),
                        default,
                    )
                })
        })
        .collect();
    for (key, domain, data_type, default) in named {
        for slot in 0..2 {
            fill_float_named_attribute_default(
                source, subject, operand, slot, &key, &domain, &data_type, &default,
            );
        }
    }

    let vertex_counts = [subject.vertices.len(), operand.vertices.len()];
    let edge_counts = [subject.edges.len(), operand.edges.len()];
    let face_counts = [subject.faces.len(), operand.faces.len()];
    let corner_counts = [
        subject
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>(),
        operand
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>(),
    ];
    let Some(Value::Array(entries)) = source.attributes.get_mut("blender_attributes") else {
        return;
    };
    for entry in entries {
        let Some(name) = entry.get("name").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        let Some(domain) = entry
            .get("domain")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let Some(data_type) = entry
            .get("data_type")
            .and_then(Value::as_str)
            .map(|value| blender_mixer_type(value).to_owned())
        else {
            continue;
        };
        let data_type = data_type.as_str();
        let default = match data_type {
            "FLOAT_COLOR" | "BYTE_COLOR" => json!([1.0, 1.0, 1.0, 1.0]),
            "FLOAT4" => json!([0.0, 0.0, 0.0, 0.0]),
            "INT16_2D" => json!([0, 0]),
            "FLOAT4X4" => json!([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]),
            "STRING" => json!(""),
            _ => continue,
        };
        for (slot, input) in [subject, operand].into_iter().enumerate() {
            let present = input
                .attributes
                .get("blender_attributes")
                .and_then(Value::as_array)
                .is_some_and(|entries| {
                    entries.iter().any(|candidate| {
                        candidate.get("name").and_then(Value::as_str) == Some(&name)
                            && candidate
                                .get("domain")
                                .and_then(Value::as_str)
                                .is_some_and(|value| value.eq_ignore_ascii_case(&domain))
                            && candidate
                                .get("data_type")
                                .and_then(Value::as_str)
                                .map(blender_mixer_type)
                                .is_some_and(|value| {
                                    if matches!(data_type, "FLOAT_COLOR" | "BYTE_COLOR") {
                                        matches!(value, "FLOAT_COLOR" | "BYTE_COLOR")
                                    } else {
                                        value == data_type
                                    }
                                })
                    })
                });
            if present {
                continue;
            }
            let counts = if domain.eq_ignore_ascii_case("POINT") {
                &vertex_counts
            } else if domain.eq_ignore_ascii_case("EDGE") {
                &edge_counts
            } else if domain.eq_ignore_ascii_case("FACE") {
                &face_counts
            } else if domain.eq_ignore_ascii_case("CORNER") {
                &corner_counts
            } else {
                continue;
            };
            let start = if slot == 1 { counts[0] } else { 0 };
            if let Some(values) = entry.get_mut("values").and_then(Value::as_array_mut) {
                for value in values.iter_mut().skip(start).take(counts[slot]) {
                    *value = default.clone();
                }
            }
        }
    }
}

fn fill_float_named_attribute_default(
    source: &mut Mesh,
    subject: &Mesh,
    operand: &Mesh,
    slot: usize,
    key: &str,
    domain: &str,
    data_type: &str,
    default: &Value,
) {
    let input = if slot == 0 { subject } else { operand };
    let input_attribute = input.attributes.get(key);
    let input_type = input_attribute
        .and_then(|attribute| attribute.get("type"))
        .and_then(Value::as_str)
        .map(blender_mixer_type);
    let input_domain = input_attribute
        .and_then(|attribute| attribute.get("domain"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let color = matches!(data_type, "FLOAT_COLOR" | "BYTE_COLOR");
    let compatible = input_type.is_some_and(|input_type| {
        (if color {
            matches!(input_type, "FLOAT_COLOR" | "BYTE_COLOR")
        } else {
            input_type == data_type
        }) && (input_domain.eq_ignore_ascii_case(domain)
            || (color
                && (domain.eq_ignore_ascii_case("point") || domain.eq_ignore_ascii_case("corner"))
                && (input_domain.eq_ignore_ascii_case("point")
                    || input_domain.eq_ignore_ascii_case("corner"))))
    });
    let input_values = compatible
        .then(|| input_attribute.and_then(|attribute| attribute.get("values")))
        .flatten()
        .and_then(Value::as_object);
    let Mesh {
        vertices,
        edges,
        faces,
        attributes,
        ..
    } = source;
    let Some(output_values) = attributes
        .get_mut(key)
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    if domain.eq_ignore_ascii_case("point") {
        let start = if slot == 1 { subject.vertices.len() } else { 0 };
        for (local, vertex) in input.vertices.iter().enumerate() {
            if input_values
                .and_then(|values| values.get(&format!("v{}", vertex.id)))
                .is_none()
            {
                let output_vertex = &vertices[start + local];
                output_values.insert(format!("v{}", output_vertex.id), default.clone());
            }
        }
    } else if domain.eq_ignore_ascii_case("edge") {
        let start = if slot == 1 { subject.edges.len() } else { 0 };
        for (local, edge) in input.edges.iter().enumerate() {
            if input_values
                .and_then(|values| values.get(&format!("e{}", edge.id)))
                .is_none()
            {
                let output_edge = &edges[start + local];
                output_values.insert(format!("e{}", output_edge.id), default.clone());
            }
        }
    } else if domain.eq_ignore_ascii_case("face") {
        let start = if slot == 1 { subject.faces.len() } else { 0 };
        for (local, face) in input.faces.iter().enumerate() {
            if input_values
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .is_none()
            {
                let output_face = &faces[start + local];
                output_values.insert(format!("f{}", output_face.id), default.clone());
            }
        }
    } else if domain.eq_ignore_ascii_case("corner") {
        let start = if slot == 1 { subject.faces.len() } else { 0 };
        for (local, face) in input.faces.iter().enumerate() {
            let output_face = &faces[start + local];
            let mut row = output_values
                .get(&format!("f{}", output_face.id))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| vec![default.clone(); face.vertices.len()]);
            row.resize(face.vertices.len(), default.clone());
            let source_row = input_values
                .and_then(|values| values.get(&format!("f{}", face.id)))
                .and_then(Value::as_array);
            for (corner, &vertex_id) in face.vertices.iter().enumerate() {
                let present = if input_domain.eq_ignore_ascii_case("corner") {
                    source_row.and_then(|row| row.get(corner)).is_some()
                } else if color && input_domain.eq_ignore_ascii_case("point") {
                    input_values
                        .and_then(|values| values.get(&format!("v{vertex_id}")))
                        .is_some()
                } else {
                    false
                };
                if !present {
                    row[corner] = default.clone();
                }
            }
            output_values.insert(format!("f{}", output_face.id), Value::Array(row));
        }
    }
}

fn float_output_origin(
    source: &Mesh,
    state: &State,
    state_vertex_ids: &[Option<u32>],
    polys: &[Poly],
    transfers: &[FloatFaceTransfer],
) -> Result<Origin> {
    let output_vertex_count = state_vertex_ids.iter().filter(|id| id.is_some()).count();
    let mut vertices = vec![Vec::new(); output_vertex_count];
    for (state_index, id) in state_vertex_ids.iter().enumerate() {
        let Some(id) = id else { continue };
        let output_index = *id as usize;
        if let Ok(source_index) = usize::try_from(state.verts[state_index].source)
            && source_index < source.vertices.len()
        {
            vertices[output_index] = vec![(source_index, 1.0)];
        }
    }
    let mut corner_starts = Vec::with_capacity(source.faces.len());
    let mut offset = 0;
    for face in &source.faces {
        corner_starts.push(offset);
        offset += face.vertices.len();
    }
    let mut corners = Vec::with_capacity(offset);
    for (poly, transfer) in polys.iter().zip(transfers) {
        let face = usize::try_from(poly.orig)
            .map_err(|_| PotError::invalid_argument("Boolean output corner has no source face"))?;
        let start = corner_starts[face];
        for (corner, weights) in transfer.weights.iter().enumerate() {
            if let Some(source_corner) = transfer.copied[corner] {
                corners.push(vec![(start + source_corner, 1.0)]);
            } else {
                corners.push(
                    weights
                        .iter()
                        .enumerate()
                        .filter(|(_, weight)| **weight != 0.0)
                        .map(|(source_corner, weight)| (start + source_corner, f64::from(*weight)))
                        .collect(),
                );
            }
        }
    }
    Ok(Origin {
        vertices,
        faces: polys
            .iter()
            .map(|poly| usize::try_from(poly.orig).ok())
            .collect(),
        corners: Some(corners),
    })
}

fn transfer_float_point_defaults(source: &Mesh, output: &mut Mesh, origins: &[Vec<(usize, f64)>]) {
    #[expect(
        clippy::float_cmp,
        reason = "copied-corner weights are exactly 1.0 by construction; exact match is intended"
    )]
    let source_index = |origin: &[(usize, f64)]| match origin {
        [(index, weight)] if *weight == 1.0 => Some(*index),
        _ => None,
    };
    let supports_type = |data_type: &str| {
        matches!(
            blender_mixer_type(data_type),
            "FLOAT_COLOR" | "BYTE_COLOR" | "INT16_2D" | "FLOAT4X4" | "STRING" | "FLOAT4"
        )
    };
    let named: Vec<(String, String)> = output
        .attributes
        .iter()
        .filter_map(|(key, attribute)| {
            let data_type = attribute.get("type").and_then(Value::as_str)?;
            (supports_type(data_type)
                && attribute
                    .get("domain")
                    .and_then(Value::as_str)
                    .is_some_and(|domain| domain.eq_ignore_ascii_case("point")))
            .then(|| (key.clone(), blender_mixer_type(data_type).to_owned()))
        })
        .collect();
    for (key, data_type) in named {
        let source_values = source
            .attributes
            .get(&key)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object);
        let mut values = serde_json::Map::new();
        for (vertex, origin) in output.vertices.iter().zip(origins) {
            let value = source_index(origin)
                .and_then(|index| source.vertices.get(index))
                .and_then(|vertex| {
                    source_values.and_then(|values| values.get(&format!("v{}", vertex.id)))
                })
                .cloned()
                .unwrap_or_else(|| bmesh_default_value(&data_type));
            values.insert(format!("v{}", vertex.id), value);
        }
        if let Some(attribute) = output
            .attributes
            .get_mut(&key)
            .and_then(Value::as_object_mut)
        {
            attribute.insert("values".to_owned(), Value::Object(values));
        }
    }

    let source_entries = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array);
    if let Some(Value::Array(output_entries)) = output.attributes.get_mut("blender_attributes") {
        for entry in output_entries {
            if !entry
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("POINT"))
            {
                continue;
            }
            let Some(name) = entry.get("name").and_then(Value::as_str).map(str::to_owned) else {
                continue;
            };
            let Some(data_type) = entry
                .get("data_type")
                .and_then(Value::as_str)
                .map(blender_mixer_type)
                .filter(|data_type| supports_type(data_type))
                .map(str::to_owned)
            else {
                continue;
            };
            let source_values = source_entries
                .and_then(|entries| {
                    entries.iter().find(|source| {
                        source.get("name").and_then(Value::as_str) == Some(&name)
                            && source
                                .get("domain")
                                .and_then(Value::as_str)
                                .is_some_and(|domain| domain.eq_ignore_ascii_case("POINT"))
                    })
                })
                .and_then(|source| source.get("values"))
                .and_then(Value::as_array);
            let values: Vec<Value> = origins
                .iter()
                .map(|origin| {
                    source_index(origin)
                        .and_then(|index| source_values.and_then(|values| values.get(index)))
                        .cloned()
                        .unwrap_or_else(|| bmesh_default_value(&data_type))
                })
                .collect();
            if let Some(object) = entry.as_object_mut() {
                object.insert("values".to_owned(), Value::Array(values));
            }
        }
    }

    let source_colors = color_layers::read(source);
    let mut output_colors = color_layers::read(output);
    for layer in &mut output_colors.layers {
        if layer.domain != ColorDomain::Point {
            continue;
        }
        let source_layer = source_colors
            .layers
            .iter()
            .find(|source| source.name == layer.name);
        for (index, origin) in origins.iter().enumerate() {
            layer.values[index] = source_index(origin)
                .and_then(|source| source_layer.and_then(|layer| layer.values.get(source)))
                .copied()
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
        }
    }
    color_layers::write(output, &output_colors);
}

fn transfer_float_edge_attributes(source: &Mesh, output: &mut Mesh, origins: &[Option<usize>]) {
    let source_edge_id = |index: usize| source.edges.get(index).map(|edge| edge.id);
    let named: Vec<String> = source
        .attributes
        .iter()
        .filter(|&(_, value)| {
            value.get("domain").and_then(Value::as_str) == Some("edge")
                && value.get("values").and_then(Value::as_object).is_some()
        })
        .map(|(key, _)| key.clone())
        .collect();
    for key in named {
        let Some(template) = source.attributes.get(&key) else {
            continue;
        };
        let data_type = template
            .get("type")
            .and_then(Value::as_str)
            .map_or("FLOAT", blender_mixer_type);
        let source_values = template.get("values").and_then(Value::as_object);
        let mut values = serde_json::Map::new();
        for (edge, source_index) in output.edges.iter().zip(origins) {
            let value = source_index
                .and_then(source_edge_id)
                .and_then(|id| source_values.and_then(|values| values.get(&format!("e{id}"))))
                .cloned()
                .unwrap_or_else(|| bmesh_default_value(data_type));
            values.insert(format!("e{}", edge.id), value);
        }
        if let Some(attribute) = output
            .attributes
            .get_mut(&key)
            .and_then(Value::as_object_mut)
        {
            attribute.insert("values".to_owned(), Value::Object(values));
        }
    }

    if let Some(Value::Array(source_flags)) = source.attributes.get("blender_edge_flags") {
        let flags = output
            .edges
            .iter()
            .zip(origins)
            .map(|(_, source)| {
                source
                    .and_then(|index| source_flags.get(index))
                    .cloned()
                    .unwrap_or_else(|| json!({"sharp": false, "seam": false}))
            })
            .collect();
        output
            .attributes
            .insert("blender_edge_flags".to_owned(), Value::Array(flags));
    }

    let source_entries = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array);
    if let Some(Value::Array(output_entries)) = output.attributes.get_mut("blender_attributes") {
        for output_entry in output_entries {
            if output_entry.get("domain").and_then(Value::as_str) != Some("EDGE") {
                continue;
            }
            let Some(name) = output_entry.get("name").and_then(Value::as_str) else {
                continue;
            };
            let data_type = output_entry
                .get("data_type")
                .and_then(Value::as_str)
                .map_or("FLOAT", blender_mixer_type);
            let source_values = source_entries
                .and_then(|entries| {
                    entries.iter().find(|entry| {
                        entry.get("name").and_then(Value::as_str) == Some(name)
                            && entry.get("domain").and_then(Value::as_str) == Some("EDGE")
                    })
                })
                .and_then(|entry| entry.get("values"))
                .and_then(Value::as_array);
            let values = origins
                .iter()
                .map(|source| {
                    source
                        .and_then(|index| source_values.and_then(|values| values.get(index)))
                        .cloned()
                        .unwrap_or_else(|| bmesh_default_value(data_type))
                })
                .collect();
            if let Some(entry) = output_entry.as_object_mut() {
                entry.insert("values".to_owned(), Value::Array(values));
            }
        }
    }
}

fn normal_codes_present(mesh: &Mesh) -> bool {
    mesh.attributes
        .get("custom_normal")
        .and_then(|attribute| attribute.get("type"))
        .and_then(Value::as_str)
        == Some("normal_code")
}
struct FloatFaceTransfer {
    source_face: usize,
    copied: Vec<Option<usize>>,
    edge_copied: Vec<Option<usize>>,
    weights: Vec<Vec<f32>>,
}

fn float_face_transfers(
    subject: &Mesh,
    operand: &Mesh,
    state: &State,
    state_output_ids: &[Option<u32>],
    polys: &[Poly],
    output_mesh: &Mesh,
) -> Result<Vec<FloatFaceTransfer>> {
    let subject_vertices: HashMap<u32, usize> = subject
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let operand_vertices: HashMap<u32, usize> = operand
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let output_points: Vec<P3> = output_mesh
        .vertices
        .iter()
        .map(|vertex| to_f32(vertex.co))
        .collect();
    let mut transfers = Vec::with_capacity(polys.len());
    for poly in polys {
        let source_face = usize::try_from(poly.orig)
            .map_err(|_| PotError::invalid_argument("Boolean corner face has no source"))?;
        let (source_mesh, local_face, vertex_index, point_base, edge_base) =
            if source_face < subject.faces.len() {
                (subject, source_face, &subject_vertices, 0, 0)
            } else {
                (
                    operand,
                    source_face - subject.faces.len(),
                    &operand_vertices,
                    subject.vertices.len(),
                    subject.edges.len(),
                )
            };
        let face = source_mesh.faces.get(local_face).ok_or_else(|| {
            PotError::invalid_argument("Boolean corner face references a missing source")
        })?;
        let source_points: Vec<P3> = face
            .vertices
            .iter()
            .map(|id| {
                vertex_index
                    .get(id)
                    .map(|&index| to_f32(source_mesh.vertices[index].co))
                    .ok_or_else(|| {
                        PotError::invalid_argument("Boolean corner source vertex is missing")
                    })
            })
            .collect::<Result<_>>()?;
        let interpolator = FloatPolyInterpolator::new(&source_points);
        let mut copied = Vec::with_capacity(poly.verts.len());
        let mut edge_copied = Vec::with_capacity(poly.verts.len());
        let mut weights = Vec::with_capacity(poly.verts.len());
        for (corner_index, &state_vertex) in poly.verts.iter().enumerate() {
            let state_index = usize::try_from(state_vertex).map_err(|_| {
                PotError::invalid_argument("Boolean corner vertex index is invalid")
            })?;
            let output_vertex = state_output_ids
                .get(state_index)
                .and_then(|id| *id)
                .ok_or_else(|| {
                    PotError::invalid_argument("Boolean corner output vertex is missing")
                })?;
            let output_index = output_vertex as usize;
            let values = interpolator.weights(output_points[output_index]);
            let point_origin = state.verts[state_index].source;
            let copied_corner = face.vertices.iter().enumerate().find_map(|(corner, id)| {
                let local = *vertex_index.get(id)?;
                (point_origin == i32::try_from(point_base + local).ok()?).then_some(corner)
            });
            // Winding reversal can move the split source edge to either adjacent output edge.
            let edge_corner = if copied_corner.is_none() {
                [
                    (corner_index + poly.verts.len() - 1) % poly.verts.len(),
                    corner_index,
                ]
                .into_iter()
                .find_map(|edge_index| {
                    poly.edge_orig
                        .get(edge_index)
                        .copied()
                        .and_then(|source| usize::try_from(source).ok())
                        .and_then(|source| source.checked_sub(edge_base))
                        .and_then(|source| source_mesh.edges.get(source))
                        .and_then(|edge| {
                            face.vertices
                                .iter()
                                .enumerate()
                                .find_map(|(corner, &vertex)| {
                                    let next = face.vertices[(corner + 1) % face.vertices.len()];
                                    ((vertex == edge.vertices[0] && next == edge.vertices[1])
                                        || (vertex == edge.vertices[1] && next == edge.vertices[0]))
                                        .then_some(corner)
                                })
                        })
                })
            } else {
                None
            };
            copied.push(copied_corner);
            edge_copied.push(edge_corner);
            weights.push(values);
        }
        transfers.push(FloatFaceTransfer {
            source_face,
            copied,
            edge_copied,
            weights,
        });
    }
    Ok(transfers)
}

/// Joined custom-normal codes: the subject's stored codes in corner order, then the operand's, with
/// the automatic code where an input stores none. `None` when neither input stores codes. Blender
/// copies these loop codes raw, so they are never decoded, transformed, or negated here.
fn joined_custom_normal_codes(
    subject: &Mesh,
    operand: &Mesh,
) -> Result<Option<Vec<clnor::CornerNormalCode>>> {
    let subject_codes = clnor::corner_codes(subject)?;
    let operand_codes = clnor::corner_codes(operand)?;
    if subject_codes.is_none() && operand_codes.is_none() {
        return Ok(None);
    }
    let mut joined = subject_codes.unwrap_or_else(|| automatic_codes(subject));
    joined.extend(operand_codes.unwrap_or_else(|| automatic_codes(operand)));
    Ok(Some(joined))
}

fn automatic_codes(mesh: &Mesh) -> Vec<clnor::CornerNormalCode> {
    let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    vec![clnor::CornerNormalCode::Auto; corners]
}

fn transfer_float_normal_codes(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) -> Result<()> {
    let joined_codes = joined_custom_normal_codes(subject, operand)?;
    let Some(source_codes) = joined_codes.as_deref() else {
        return Ok(());
    };
    let mut starts = Vec::with_capacity(subject.faces.len() + operand.faces.len());
    let mut start = 0;
    for face in subject.faces.iter().chain(&operand.faces) {
        starts.push(start);
        start += face.vertices.len();
    }
    let mut output_codes = Vec::new();
    for transfer in transfers {
        let start = starts[transfer.source_face];
        for (copied, edge_copied) in transfer.copied.iter().zip(&transfer.edge_copied) {
            // These are raw short2 loop codes: copy the source block, never decode or negate it.
            let source_corner = (*copied).or(*edge_copied);
            output_codes.push(
                source_corner
                    .and_then(|corner| source_codes.get(start + corner).copied())
                    .unwrap_or(clnor::CornerNormalCode::Auto),
            );
        }
    }
    clnor::write_codes(
        output_mesh,
        Some(output_codes.as_slice()),
        clnor::NormalCodeOptions::default(),
    )
}

fn transfer_float_corner_attributes(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    transfer_float_uv_layers(subject, operand, transfers, output_mesh);
    transfer_float_color_layers(subject, operand, transfers, output_mesh);
    transfer_float_blender_uv_layers(subject, operand, transfers, output_mesh);
    transfer_float_uv_map(subject, operand, transfers, output_mesh);
    transfer_float_attribute_entries(subject, operand, transfers, output_mesh);
    transfer_float_named_corner_attributes(subject, operand, transfers, output_mesh);
}

fn transfer_float_uv_layers(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let inputs = [uv_layers::read(subject), uv_layers::read(operand)];
    let mut output = uv_layers::read(output_mesh);
    for layer in &mut output.layers {
        for (output_face, transfer) in transfers.iter().enumerate() {
            let (slot, local_face, _) = select_source_face(subject, operand, transfer.source_face);
            let Some(source_layer) = inputs[slot]
                .layers
                .iter()
                .find(|source| source.name == layer.name)
            else {
                layer.faces[output_face] = Some(vec![[0.0, 1.0]; transfer.weights.len()]);
                layer.pinned[output_face] = None;
                continue;
            };
            let row = source_layer.faces.get(local_face).and_then(Option::as_ref);
            let values = row.map_or_else(
                || vec![[0.0, 1.0]; transfer.weights.len()],
                |values| {
                    transfer
                        .weights
                        .iter()
                        .enumerate()
                        .map(|(output_corner, weights)| {
                            if let Some(source_corner) = transfer.copied[output_corner] {
                                return values.get(source_corner).copied().unwrap_or([0.0, 1.0]);
                            }
                            // Canonical UVs are V-down; Blender interpolates them in V-up.
                            let mut result = [0.0_f32; 2];
                            for (corner, &weight) in weights.iter().enumerate() {
                                if let Some(value) = values.get(corner) {
                                    result[0] += value[0] as f32 * weight;
                                    result[1] += (1.0_f32 - value[1] as f32) * weight;
                                }
                            }
                            [f64::from(result[0]), f64::from(1.0_f32 - result[1])]
                        })
                        .collect()
                },
            );
            layer.faces[output_face] = Some(values);
            layer.pinned[output_face] = source_layer
                .pinned
                .get(local_face)
                .and_then(Option::as_ref)
                .map(|pinned| {
                    transfer
                        .weights
                        .iter()
                        .enumerate()
                        .map(|(output_corner, weights)| {
                            if let Some(source_corner) = transfer.copied[output_corner] {
                                pinned.get(source_corner).copied().unwrap_or(false)
                            } else {
                                weights.iter().enumerate().any(|(corner, &weight)| {
                                    pinned.get(corner).copied().unwrap_or(false) && weight > 0.0
                                })
                            }
                        })
                        .collect()
                });
        }
    }
    uv_layers::write(output_mesh, &output);
}

fn transfer_float_color_layers(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let subject_vertex_index: HashMap<u32, usize> = subject
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let operand_vertex_index: HashMap<u32, usize> = operand
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let vertex_indices = [&subject_vertex_index, &operand_vertex_index];
    let corner_starts = [corner_offsets(subject), corner_offsets(operand)];
    let stores = [color_layers::read(subject), color_layers::read(operand)];
    let mut output = color_layers::read(output_mesh);
    let data_types = output_mesh
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object);
    for layer in &mut output.layers {
        if layer.domain != ColorDomain::Corner {
            continue;
        }
        let byte_color = data_types
            .and_then(|types| types.get(&layer.name))
            .and_then(Value::as_str)
            == Some("BYTE_COLOR");
        let mut values = Vec::new();
        for transfer in transfers {
            let (source_slot, local_face, _) =
                select_source_face(subject, operand, transfer.source_face);
            let source_layer = stores[source_slot]
                .layers
                .iter()
                .find(|source| source.name == layer.name);
            let source_mesh = if source_slot == 0 { subject } else { operand };
            let face = &source_mesh.faces[local_face];
            let corner_start = corner_starts[source_slot][local_face];
            for (output_corner, weights) in transfer.weights.iter().enumerate() {
                let source_color = |corner: usize| {
                    let vertex_id = face.vertices[corner];
                    source_layer.map_or([1.0, 1.0, 1.0, 1.0], |source| match source.domain {
                        ColorDomain::Corner => source
                            .values
                            .get(corner_start + corner)
                            .copied()
                            .unwrap_or([1.0, 1.0, 1.0, 1.0]),
                        ColorDomain::Point => vertex_indices[source_slot]
                            .get(&vertex_id)
                            .and_then(|&index| source.values.get(index).copied())
                            .unwrap_or([1.0, 1.0, 1.0, 1.0]),
                    })
                };
                if let Some(corner) = transfer.copied[output_corner] {
                    values.push(source_color(corner));
                } else {
                    let colors = (0..face.vertices.len()).map(source_color);
                    values.push(mix_color(colors, weights, byte_color));
                }
            }
        }
        layer.values = values;
    }
    color_layers::write(output_mesh, &output);
}

fn transfer_float_blender_uv_layers(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let Some(Value::Array(output_layers)) = output_mesh.attributes.get_mut("blender_uv_layers")
    else {
        return;
    };
    for output_layer in output_layers {
        let Some(name) = output_layer
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let mut values = Vec::new();
        for transfer in transfers {
            let (_, local_face, source_mesh) =
                select_source_face(subject, operand, transfer.source_face);
            let source_layer = source_mesh
                .attributes
                .get("blender_uv_layers")
                .and_then(Value::as_array)
                .and_then(|layers| {
                    layers
                        .iter()
                        .find(|layer| layer.get("name").and_then(Value::as_str) == Some(&name))
                });
            let row = source_layer
                .and_then(|layer| layer.get("values"))
                .and_then(Value::as_array);
            let source_values: Vec<Value> = (0..source_mesh.faces[local_face].vertices.len())
                .map(|corner| {
                    row.and_then(|row| row.get(corner))
                        .cloned()
                        .unwrap_or_else(|| json!([0.0, 0.0]))
                })
                .collect();
            for weights in &transfer.weights {
                values.push(mix_json_values(&source_values, weights, "FLOAT2"));
            }
        }
        if let Some(object) = output_layer.as_object_mut() {
            object.insert("values".to_owned(), Value::Array(values));
        }
    }
}

fn transfer_float_uv_map(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let output_face_indices: HashMap<u32, usize> = output_mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (face.id, index))
        .collect();
    let transfer_by_output: Vec<&FloatFaceTransfer> = transfers.iter().collect();
    let Some(Value::Array(entries)) = output_mesh.attributes.get_mut("uv_map") else {
        return;
    };
    for entry in entries {
        let output_face = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .and_then(|id| output_face_indices.get(&id).copied());
        let Some(output_face) = output_face else {
            continue;
        };
        let Some(transfer) = transfer_by_output.get(output_face) else {
            continue;
        };
        let (source_mesh, local_face) = if transfer.source_face < subject.faces.len() {
            (subject, transfer.source_face)
        } else {
            (operand, transfer.source_face - subject.faces.len())
        };
        let Some(source_face) = source_mesh.faces.get(local_face) else {
            continue;
        };
        let layer = entry.get("layer").and_then(Value::as_str);
        let source_entry = source_mesh
            .attributes
            .get("uv_map")
            .and_then(Value::as_array)
            .and_then(|entries| {
                entries.iter().find(|source| {
                    source.get("face_id").and_then(Value::as_u64) == Some(u64::from(source_face.id))
                        && source.get("layer").and_then(Value::as_str) == layer
                })
            });
        let source_uv = source_entry
            .and_then(|source| source.get("uv"))
            .and_then(Value::as_array);
        let source_uv_values: Vec<Value> = (0..source_face.vertices.len())
            .map(|corner| {
                source_uv
                    .and_then(|uv| uv.get(corner))
                    .cloned()
                    .unwrap_or_else(|| json!([0.0, 1.0]))
            })
            .collect();
        let row: Vec<Value> = transfer
            .weights
            .iter()
            .map(|weights| mix_json_values(&source_uv_values, weights, "FLOAT2"))
            .collect();
        let source_pins = source_entry
            .and_then(|source| source.get("pinned"))
            .and_then(Value::as_array);
        if let Some(object) = entry.as_object_mut() {
            object.insert("uv".to_owned(), Value::Array(row));
            if let Some(source_pins) = source_pins {
                let pins: Vec<Value> = transfer
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(output_corner, weights)| {
                        let pinned = if let Some(source_corner) = transfer.copied[output_corner] {
                            source_pins.get(source_corner).and_then(Value::as_bool) == Some(true)
                        } else {
                            weights.iter().enumerate().any(|(corner, &weight)| {
                                source_pins.get(corner).and_then(Value::as_bool) == Some(true)
                                    && weight > 0.0
                            })
                        };
                        json!(pinned)
                    })
                    .collect();
                object.insert("pinned".to_owned(), Value::Array(pins));
            }
        }
    }
}

fn mix_float_corner_attribute(
    values: &[Value],
    weights: &[f32],
    copied: Option<usize>,
    edge_copied: Option<usize>,
    data_type: &str,
) -> Value {
    let data_type = blender_mixer_type(data_type);
    if let Some(corner) = copied {
        return values
            .get(corner)
            .cloned()
            .unwrap_or_else(|| bmesh_default_value(data_type));
    }
    // CustomData_bmesh_interp skips these types, so the destination keeps its edge copy/default.
    if matches!(
        data_type,
        "INT8" | "INT16_2D" | "INT32_2D" | "FLOAT4X4" | "STRING"
    ) {
        return edge_copied
            .and_then(|corner| values.get(corner).cloned())
            .unwrap_or_else(|| bmesh_default_value(data_type));
    }
    mix_json_values_interpolated(values, weights, data_type)
}
fn transfer_float_attribute_entries(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let Some(Value::Array(output_entries)) = output_mesh.attributes.get_mut("blender_attributes")
    else {
        return;
    };
    let corner_starts = [corner_offsets(subject), corner_offsets(operand)];
    for output_entry in output_entries {
        if output_entry.get("domain").and_then(Value::as_str) != Some("CORNER") {
            continue;
        }
        let Some(name) = output_entry
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_owned)
        else {
            continue;
        };
        let data_type = output_entry
            .get("data_type")
            .and_then(Value::as_str)
            .map_or("FLOAT", blender_mixer_type)
            .to_owned();
        let mut values = Vec::new();
        for transfer in transfers {
            let (slot, local_face, source_mesh) =
                select_source_face(subject, operand, transfer.source_face);
            let row = source_mesh
                .attributes
                .get("blender_attributes")
                .and_then(Value::as_array)
                .and_then(|entries| {
                    entries.iter().find(|entry| {
                        entry.get("name").and_then(Value::as_str) == Some(&name)
                            && entry.get("domain").and_then(Value::as_str) == Some("CORNER")
                    })
                })
                .and_then(|entry| entry.get("values"))
                .and_then(Value::as_array);
            let corner_start = corner_starts[slot][local_face];
            let face_len = source_mesh.faces[local_face].vertices.len();
            let source_values: Vec<Value> = (0..face_len)
                .map(|corner| {
                    row.and_then(|row| row.get(corner_start + corner))
                        .cloned()
                        .unwrap_or_else(|| bmesh_default_value(&data_type))
                })
                .collect();
            values.extend(
                transfer
                    .weights
                    .iter()
                    .zip(&transfer.copied)
                    .zip(&transfer.edge_copied)
                    .map(|((weights, copied), edge_copied)| {
                        mix_float_corner_attribute(
                            &source_values,
                            weights,
                            *copied,
                            *edge_copied,
                            &data_type,
                        )
                    }),
            );
        }
        if let Some(object) = output_entry.as_object_mut() {
            object.insert("values".to_owned(), Value::Array(values));
        }
    }
}

fn transfer_float_named_corner_attributes(
    subject: &Mesh,
    operand: &Mesh,
    transfers: &[FloatFaceTransfer],
    output_mesh: &mut Mesh,
) {
    let color_keys: HashSet<String> = color_layers::read(output_mesh)
        .layers
        .iter()
        .map(|layer| attribute_keys::storage_key(&layer.name).into_owned())
        .collect();
    let keys: Vec<String> = output_mesh
        .attributes
        .iter()
        .filter(|&(key, value)| {
            value.get("domain").and_then(Value::as_str) == Some("corner")
                && value.get("values").and_then(Value::as_object).is_some()
                && !color_keys.contains(key)
        })
        .map(|(key, _)| key.clone())
        .collect();
    for key in keys {
        let stored_type = output_mesh
            .attributes
            .get(&key)
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("FLOAT");
        let data_type = blender_mixer_type(stored_type).to_owned();
        let mut values = serde_json::Map::new();
        for (output_face, transfer) in transfers.iter().enumerate() {
            let (_, local_face, source_mesh) =
                select_source_face(subject, operand, transfer.source_face);
            let source_face = &source_mesh.faces[local_face];
            let source_row = source_mesh
                .attributes
                .get(&key)
                .and_then(|attribute| attribute.get("values"))
                .and_then(Value::as_object)
                .and_then(|rows| rows.get(&format!("f{}", source_face.id)))
                .and_then(Value::as_array);
            let inputs: Vec<Value> = (0..source_face.vertices.len())
                .map(|corner| {
                    source_row
                        .and_then(|row| row.get(corner))
                        .cloned()
                        .unwrap_or_else(|| bmesh_default_value(&data_type))
                })
                .collect();
            let row: Vec<Value> = transfer
                .weights
                .iter()
                .zip(&transfer.copied)
                .zip(&transfer.edge_copied)
                .map(|((weights, copied), edge_copied)| {
                    mix_float_corner_attribute(&inputs, weights, *copied, *edge_copied, &data_type)
                })
                .collect();
            if let Some(face) = output_mesh.faces.get(output_face) {
                values.insert(format!("f{}", face.id), Value::Array(row));
            }
        }
        if let Some(attribute) = output_mesh
            .attributes
            .get_mut(&key)
            .and_then(Value::as_object_mut)
        {
            attribute.insert("values".to_owned(), Value::Object(values));
        }
    }
}

fn select_source_face<'a>(
    subject: &'a Mesh,
    operand: &'a Mesh,
    joined_face: usize,
) -> (usize, usize, &'a Mesh) {
    if joined_face < subject.faces.len() {
        (0, joined_face, subject)
    } else {
        (1, joined_face - subject.faces.len(), operand)
    }
}

fn corner_offsets(mesh: &Mesh) -> Vec<usize> {
    let mut offset = 0;
    mesh.faces
        .iter()
        .map(|face| {
            let start = offset;
            offset += face.vertices.len();
            start
        })
        .collect()
}

fn mix_color(
    colors: impl IntoIterator<Item = [f64; 4]>,
    weights: &[f32],
    byte_color: bool,
) -> [f64; 4] {
    let mut accum = [0.0_f32; 4];
    for (color, &weight) in colors.into_iter().zip(weights) {
        if byte_color {
            let encoded = color_layers::byte_color_encoded_components(color);
            for component in 0..4 {
                let value = (encoded[component] + 0.5).floor().clamp(0.0, 255.0);
                accum[component] += value * weight;
            }
        } else {
            for component in 0..4 {
                accum[component] += color[component] as f32 * weight;
            }
        }
    }
    if byte_color {
        color_layers::byte_color_decoded_components(
            accum.map(|value| (value + 0.5).floor().clamp(0.0, 255.0)),
        )
    } else {
        accum.map(f64::from)
    }
}

fn blender_mixer_type(value: &str) -> &str {
    match value {
        "bool" | "boolean" => "BOOLEAN",
        "int" => "INT",
        "int8" => "INT8",
        "int2" | "int32_2d" => "INT32_2D",
        "short2" | "int16_2d" => "INT16_2D",
        "float4x4" => "FLOAT4X4",
        "string" => "STRING",
        "float" => "FLOAT",
        "float2" => "FLOAT2",
        "float3" => "FLOAT_VECTOR",
        "float4" => "FLOAT4",
        "float_color" | "color" => "FLOAT_COLOR",
        "byte_color" => "BYTE_COLOR",
        "quaternion" => "QUATERNION",
        other => other,
    }
}

fn bmesh_default_value(data_type: &str) -> Value {
    match data_type {
        "FLOAT_COLOR" | "BYTE_COLOR" => json!([1.0, 1.0, 1.0, 1.0]),
        "INT16_2D" => json!([0, 0]),
        "FLOAT4" => json!([0.0, 0.0, 0.0, 0.0]),
        "FLOAT4X4" => json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "STRING" => json!(""),
        _ => default_value(data_type),
    }
}

#[expect(
    clippy::float_cmp,
    reason = "exact 1.0 full-weight test mirrors the Blender port's copy-only path"
)]
fn mix_json_values(values: &[Value], weights: &[f32], data_type: &str) -> Value {
    if let Some(index) = weights.iter().position(|&weight| weight == 1.0)
        && weights
            .iter()
            .enumerate()
            .all(|(i, &weight)| i == index || weight == 0.0)
    {
        return values
            .get(index)
            .cloned()
            .unwrap_or_else(|| bmesh_default_value(data_type));
    }
    mix_json_values_interpolated(values, weights, data_type)
}

fn mix_json_values_interpolated(values: &[Value], weights: &[f32], data_type: &str) -> Value {
    match data_type {
        "BOOLEAN" | "BOOL" => json!(
            values
                .iter()
                .zip(weights)
                .any(|(value, weight)| value.as_bool().unwrap_or(false) && *weight > 0.0)
        ),
        "INT" => {
            let sum = values
                .iter()
                .zip(weights)
                .fold(0.0_f32, |sum, (value, weight)| {
                    sum + value.as_f64().unwrap_or(0.0) as f32 * *weight
                });
            json!(sum.round() as i32)
        }
        "INT8" | "INT16_2D" | "INT32_2D" | "FLOAT4X4" | "STRING" => bmesh_default_value(data_type),
        "QUATERNION" => mix_quaternion(values, weights),
        _ => mix_json_array(values, weights, data_type),
    }
}

fn mix_quaternion(values: &[Value], weights: &[f32]) -> Value {
    let mut expmap = [0.0_f32; 3];
    let mut total = 0.0_f32;
    for (value, &weight) in values.iter().zip(weights) {
        let Some(row) = value.as_array() else {
            continue;
        };
        let x = row.first().and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let y = row.get(1).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let z = row.get(2).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let w = row.get(3).and_then(Value::as_f64).unwrap_or(1.0) as f32;
        let sin_half = ((x * x + y * y) + z * z).sqrt();
        let rotation = if sin_half < 0.0005 {
            [0.0; 3]
        } else {
            let axis = [x / sin_half, y / sin_half, z / sin_half];
            let angle = sin_half.atan2(w) * 2.0;
            [axis[0] * angle, axis[1] * angle, axis[2] * angle]
        };
        expmap[0] += rotation[0] * weight;
        expmap[1] += rotation[1] * weight;
        expmap[2] += rotation[2] * weight;
        total += weight;
    }
    if total <= 0.0 || total.is_nan() {
        return default_value("QUATERNION");
    }
    let inverse = 1.0 / total;
    expmap[0] *= inverse;
    expmap[1] *= inverse;
    expmap[2] *= inverse;
    let length_squared = (expmap[0] * expmap[0] + expmap[1] * expmap[1]) + expmap[2] * expmap[2];
    if length_squared <= 1.0e-35 || length_squared.is_nan() {
        return default_value("QUATERNION");
    }
    let angle = length_squared.sqrt();
    let axis = [expmap[0] / angle, expmap[1] / angle, expmap[2] / angle];
    let half_angle = wrap_radians(angle) / 2.0;
    let sine = half_angle.sin();
    json!([
        axis[0] * sine,
        axis[1] * sine,
        axis[2] * sine,
        half_angle.cos()
    ])
}

fn wrap_radians(angle: f32) -> f32 {
    let pi = std::f32::consts::PI;
    let period = 2.0 * pi;
    let shifted = angle + pi;
    let wrapped = shifted - period * (shifted / period).floor();
    wrapped - pi
}
fn mix_json_array(values: &[Value], weights: &[f32], data_type: &str) -> Value {
    if values.is_empty() {
        return bmesh_default_value(data_type);
    }
    let Some(first) = values.iter().find_map(Value::as_array) else {
        let sum = values
            .iter()
            .zip(weights)
            .fold(0.0_f32, |sum, (value, weight)| {
                sum + value.as_f64().unwrap_or(0.0) as f32 * *weight
            });
        return json!(sum);
    };
    let mut result = Vec::with_capacity(first.len());
    for component in 0..first.len() {
        let sum = values
            .iter()
            .zip(weights)
            .fold(0.0_f32, |sum, (value, weight)| {
                let x = value
                    .as_array()
                    .and_then(|values| values.get(component))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0) as f32;
                sum + x * *weight
            });
        result.push(json!(sum));
    }
    Value::Array(result)
}

struct FloatPolyInterpolator {
    vertices: Vec<[f32; 2]>,
    axis_x: P3,
    axis_y: P3,
}

impl FloatPolyInterpolator {
    fn new(points: &[P3]) -> Self {
        let normal = match points {
            [a, b, c] => normal_tri(*a, *b, *c),
            [a, b, c, d] => normalize(cross(sub(*a, *c), sub(*b, *d))),
            _ => normal_poly(points),
        };
        let f = normal[0] * normal[0] + normal[1] * normal[1];
        let (axis_x, axis_y) = if f > FLT_EPSILON {
            let d = 1.0 / f.sqrt();
            let x = [normal[1] * d, -normal[0] * d, 0.0];
            let y = [
                -normal[2] * x[1],
                normal[2] * x[0],
                normal[0] * x[1] - normal[1] * x[0],
            ];
            (x, y)
        } else {
            (
                [if normal[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            )
        };
        Self {
            vertices: points
                .iter()
                .map(|&point| [dot(axis_x, point), dot(axis_y, point)])
                .collect(),
            axis_x,
            axis_y,
        }
    }

    fn weights(&self, point: P3) -> Vec<f32> {
        let n = self.vertices.len();
        let mut weights = vec![0.0; n];
        if n == 0 {
            return weights;
        }
        let point2 = [dot(self.axis_x, point), dot(self.axis_y, point)];
        let max_value = self.vertices.iter().fold(0.0_f32, |m, v| {
            m.max((v[0] - point2[0]).abs())
                .max((v[1] - point2[1]).abs())
        });
        let eps = 16.0 * FLT_EPSILON * max_value;
        let eps_sq = eps * eps;
        let mut i_curr = n - 1;
        let mut i_next = 0;
        let mut d_curr = double2_len(self.vertices[(n + n - 2) % n], point2);
        let mut d_next = double2_len(self.vertices[i_curr], point2);
        let mut ht_prev = mean_value_half_tan(&d_curr, &d_next);
        let mut total = 0.0_f32;
        let mut intersection = 0_u8;
        while i_next < n {
            if d_next.1 < f64::from(eps) {
                intersection = 1;
                break;
            }
            let current = self.vertices[i_curr];
            let next = self.vertices[i_next];
            if dist_point_segment_sq(point2, current, next) < eps_sq {
                intersection = 2;
                break;
            }
            d_curr = d_next;
            d_next = double2_len(next, point2);
            let ht = mean_value_half_tan(&d_curr, &d_next);
            weights[i_curr] = if d_curr.1 == 0.0 {
                0.0
            } else {
                ((ht_prev + ht) / d_curr.1) as f32
            };
            total += weights[i_curr];
            i_curr = i_next;
            i_next += 1;
            ht_prev = ht;
        }
        match intersection {
            1 => {
                weights.fill(0.0);
                weights[i_curr] = 1.0;
            }
            2 => {
                weights.fill(0.0);
                let a = self.vertices[i_curr];
                let b = self.vertices[i_next];
                let factor = line_factor_2d(point2, a, b).clamp(0.0, 1.0);
                weights[i_curr] = 1.0 - factor;
                weights[i_next] = factor;
            }
            _ if total != 0.0 => {
                for weight in &mut weights {
                    *weight /= total;
                }
            }
            _ => {}
        }
        weights
    }
}

fn double2_len(a: [f32; 2], b: [f32; 2]) -> ([f64; 2], f64) {
    let dir = [
        f64::from(a[0]) - f64::from(b[0]),
        f64::from(a[1]) - f64::from(b[1]),
    ];
    let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
    (dir, len)
}

fn mean_value_half_tan(a: &([f64; 2], f64), b: &([f64; 2], f64)) -> f64 {
    let area = a.0[0] * b.0[1] - a.0[1] * b.0[0];
    if area == 0.0 {
        return 0.0;
    }
    let dot = a.0[0] * b.0[0] + a.0[1] * b.0[1];
    let value = (a.1 * b.1 - dot) / area;
    if value.is_finite() { value } else { 0.0 }
}

fn line_factor_2d(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let denominator = dx * dx + dy * dy;
    if denominator > 0.0 {
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator
    } else {
        0.0
    }
}

fn dist_point_segment_sq(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let factor = line_factor_2d(point, a, b).clamp(0.0, 1.0);
    let x = a[0] + (b[0] - a[0]) * factor;
    let y = a[1] + (b[1] - a[1]) * factor;
    (point[0] - x) * (point[0] - x) + (point[1] - y) * (point[1] - y)
}

fn build_state(
    subject: &Mesh,
    operand_local: &Mesh,
    operand_frame: &Mesh,
    operand_local_positions: &[P3],
) -> Result<State> {
    let subject_vertex: HashMap<u32, usize> = subject
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id, i))
        .collect();
    let operand_vertex: HashMap<u32, usize> = operand_local
        .vertices
        .iter()
        .enumerate()
        .map(|(i, v)| (v.id, i))
        .collect();
    let subject_vertex_base = operand_local.vertices.len();
    let mut state = State::new();
    // MOD_boolean creates BMesh from operand B, then subject A.
    for (i, v) in operand_frame.vertices.iter().enumerate() {
        state.add_vertex(
            to_f32(v.co),
            i32::try_from(subject.vertices.len() + i)
                .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?,
        );
    }
    for (i, v) in subject.vertices.iter().enumerate() {
        state.add_vertex(
            to_f32(v.co),
            i32::try_from(i)
                .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?,
        );
    }
    let max_face_len = operand_local
        .faces
        .iter()
        .chain(&subject.faces)
        .map(|face| face.vertices.len())
        .max()
        .unwrap_or(0);
    let mut local_positions = Vec::with_capacity(max_face_len);
    let mut triangulation_coordinates = Vec::with_capacity(max_face_len);
    let mut face_triangles = Vec::with_capacity(max_face_len.saturating_sub(2));
    let mut triangulation_corners = Vec::with_capacity(max_face_len);
    let mut beauty_scratch = BeautyScratch::default();

    for (i, edge) in operand_local.edges.iter().enumerate() {
        let a = *operand_vertex.get(&edge.vertices[0]).ok_or_else(|| {
            PotError::invalid_argument("Boolean operand edge references a missing vertex")
        })?;
        let b = *operand_vertex.get(&edge.vertices[1]).ok_or_else(|| {
            PotError::invalid_argument("Boolean operand edge references a missing vertex")
        })?;
        let key = edge_key_usize(a, b);
        let src = i32::try_from(subject.edges.len() + i)
            .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?;
        state.edge_orig.insert(key, src);
        state.edge_start.insert(key, a);
        state.edge_order.push(key);
    }
    for (i, edge) in subject.edges.iter().enumerate() {
        let a = subject_vertex_base
            + *subject_vertex.get(&edge.vertices[0]).ok_or_else(|| {
                PotError::invalid_argument("Boolean subject edge references a missing vertex")
            })?;
        let b = subject_vertex_base
            + *subject_vertex.get(&edge.vertices[1]).ok_or_else(|| {
                PotError::invalid_argument("Boolean subject edge references a missing vertex")
            })?;
        let key = edge_key_usize(a, b);
        let src = i32::try_from(i)
            .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?;
        state.edge_orig.insert(key, src);
        state.edge_order.push(key);
        state.edge_start.insert(key, a);
    }

    let mut face_vertex_rings = Vec::with_capacity(operand_local.faces.len() + subject.faces.len());
    for (fi, face) in operand_local.faces.iter().enumerate() {
        let verts = face
            .vertices
            .iter()
            .map(|id| {
                operand_vertex.get(id).copied().ok_or_else(|| {
                    PotError::invalid_argument("Boolean face references a missing vertex")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let source_face = i32::try_from(subject.faces.len() + fi)
            .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?;
        face_vertex_rings.push(verts.clone());
        state.faces.push(Face {
            verts,
            side: 1,
            source: source_face,
        });
    }
    for (fi, face) in subject.faces.iter().enumerate() {
        let verts = face
            .vertices
            .iter()
            .map(|id| {
                subject_vertex
                    .get(id)
                    .map(|&v| subject_vertex_base + v)
                    .ok_or_else(|| {
                        PotError::invalid_argument("Boolean face references a missing vertex")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let source_face = i32::try_from(fi)
            .map_err(|_| PotError::invalid_argument("Boolean input exceeds index range"))?;
        face_vertex_rings.push(verts.clone());
        state.faces.push(Face {
            verts,
            side: 0,
            source: source_face,
        });
    }

    for (face_index, ring) in face_vertex_rings.iter().enumerate() {
        local_positions.clear();
        for &vertex in ring {
            let position = if vertex < subject_vertex_base {
                operand_local_positions[vertex]
            } else {
                to_f32(subject.vertices[vertex - subject_vertex_base].co)
            };
            local_positions.push(position);
        }
        if ring.len() == 4 {
            for verts in beauty_quad_tris(ring, &local_positions) {
                state.tris.push(Triangle {
                    verts,
                    face: face_index,
                    side: state.faces[face_index].side,
                });
            }
        } else {
            beauty_ngon_tris_into(
                ring,
                &local_positions,
                &mut triangulation_coordinates,
                &mut face_triangles,
                &mut triangulation_corners,
                &mut beauty_scratch,
            );
            for &verts in &face_triangles {
                state.tris.push(Triangle {
                    verts,
                    face: face_index,
                    side: state.faces[face_index].side,
                });
            }
        }
    }
    let mut face_edge_keys = HashSet::new();
    for ring in &face_vertex_rings {
        for i in 0..ring.len() {
            face_edge_keys.insert(edge_key_usize(ring[i], ring[(i + 1) % ring.len()]));
        }
    }
    for &edge in &state.edge_order {
        if !face_edge_keys.contains(&edge)
            && let Some(&source) = state.edge_orig.get(&edge)
        {
            state.input_loose_edge_sources.insert(source);
        }
    }
    Ok(state)
}

fn beauty_quad_tris(ring: &[usize], positions: &[P3]) -> [[usize; 3]; 2] {
    let points = [positions[0], positions[1], positions[2], positions[3]];
    if quad_rotate_score(points) < 0.0 {
        [[ring[0], ring[1], ring[2]], [ring[0], ring[2], ring[3]]]
    } else {
        [[ring[0], ring[1], ring[3]], [ring[1], ring[2], ring[3]]]
    }
}
fn quad_rotate_score(p: [P3; 4]) -> f32 {
    let n = normalize(cross(sub(p[0], p[2]), sub(p[1], p[3])));
    let (u, v) = orthogonal_basis(n);
    let projected = p.map(|co| [dot(co, u), dot(co, v)]);
    let cross2 = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let [a, b, c, d] = projected;
    let area123 = cross2(a, b, c);
    let area134 = cross2(a, c, d);
    let area234 = cross2(b, c, d);
    let area241 = cross2(b, d, a);
    if (area123 >= 0.0) != (area134 >= 0.0) || area123.abs() <= 1.0e-12 || area134.abs() <= 1.0e-12
    {
        return f32::MAX;
    }
    if (area234 >= 0.0) != (area241 >= 0.0) || area234.abs() <= 1.0e-12 || area241.abs() <= 1.0e-12
    {
        return -f32::MAX;
    }
    let length = |a: [f32; 2], b: [f32; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let fac24 = area234.abs() / (length(b, c) + length(c, d) + length(b, d))
        + area241.abs() / (length(d, a) + length(a, b) + length(b, d));
    let fac13 = area123.abs() / (length(a, b) + length(b, c) + length(a, c))
        + area134.abs() / (length(c, d) + length(d, a) + length(a, c));
    fac24 - fac13
}

const EAR_CONCAVE: i8 = -1;
const EAR_TANGENTIAL: i8 = 0;
const EAR_CONVEX: i8 = 1;

#[derive(Clone, Copy)]
struct EarCorner {
    previous: usize,
    next: usize,
    sign: i8,
}

struct EarClipper<'a> {
    coordinates: &'a [[f32; 2]],
    corners: &'a mut [EarCorner],
    head: usize,
    count: usize,
    concave: usize,
    triangles: &'a mut Vec<[usize; 3]>,
}

struct EarTriangle {
    edges: [[f32; 2]; 3],
    offsets: [f32; 3],
}

impl EarTriangle {
    fn new(vertices: [[f32; 2]; 3]) -> Self {
        let edges = [
            sub2(vertices[1], vertices[0]),
            sub2(vertices[2], vertices[1]),
            sub2(vertices[0], vertices[2]),
        ];
        let offsets = [
            edges[0][0] * vertices[0][1] - vertices[0][0] * edges[0][1],
            edges[1][0] * vertices[1][1] - vertices[1][0] * edges[1][1],
            edges[2][0] * vertices[2][1] - vertices[2][0] * edges[2][1],
        ];
        Self { edges, offsets }
    }

    fn contains(&self, point: [f32; 2]) -> bool {
        [1, 2, 0].into_iter().all(|i| {
            self.edges[i][1] * point[0] - self.edges[i][0] * point[1] + self.offsets[i] >= 0.0
        })
    }
}

fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn corner_sign(coordinates: &[[f32; 2]], previous: usize, current: usize, next: usize) -> i8 {
    let d2 = sub2(coordinates[current], coordinates[next]);
    let d3 = sub2(coordinates[previous], coordinates[next]);
    let area = d2[0] * d3[1] - d3[0] * d2[1];
    if area > 0.0 {
        EAR_CONVEX
    } else if area >= 0.0 {
        EAR_TANGENTIAL
    } else {
        EAR_CONCAVE
    }
}

fn float_polyfill_with_scratch(
    coordinates: &[[f32; 2]],
    corners: &mut Vec<EarCorner>,
    triangles: &mut Vec<[usize; 3]>,
) {
    let count = coordinates.len();
    corners.clear();
    corners.extend((0..count).map(|i| EarCorner {
        previous: (i + count - 1) % count,
        next: (i + 1) % count,
        sign: EAR_CONVEX,
    }));
    for (i, corner) in corners.iter_mut().enumerate() {
        corner.sign = corner_sign(coordinates, corner.previous, i, corner.next);
    }
    let concave = corners
        .iter()
        .filter(|corner| corner.sign != EAR_CONVEX)
        .count();
    triangles.clear();
    let mut clipper = EarClipper {
        coordinates,
        corners: corners.as_mut_slice(),
        head: 0,
        count,
        concave,
        triangles,
    };
    clipper.clip_all();
}

impl EarClipper<'_> {
    fn is_ear(&self, tip: usize, accepted_sign: i8) -> bool {
        if self.concave == 0 {
            return true;
        }
        let corner = self.corners[tip];
        if corner.sign != accepted_sign {
            return false;
        }
        let triangle = EarTriangle::new([
            self.coordinates[tip],
            self.coordinates[corner.next],
            self.coordinates[corner.previous],
        ]);
        let mut checked = 0;
        let mut current = self.corners[corner.next].next;
        while current != corner.previous {
            if self.corners[current].sign != EAR_CONVEX {
                if triangle.contains(self.coordinates[current]) {
                    return false;
                }
                checked += 1;
                if checked == self.concave {
                    break;
                }
            }
            current = self.corners[current].next;
        }
        true
    }

    fn step(&self, index: usize, reverse: bool) -> usize {
        if reverse {
            self.corners[index].previous
        } else {
            self.corners[index].next
        }
    }

    fn find_ear(&self, init: usize, reverse: bool) -> usize {
        for accepted_sign in [EAR_CONVEX, EAR_TANGENTIAL] {
            let mut ear = init;
            for _ in 0..self.count {
                if self.is_ear(ear, accepted_sign) {
                    return ear;
                }
                ear = self.step(ear, reverse);
            }
        }
        let mut ear = init;
        for _ in 0..self.count {
            if self.corners[ear].sign != EAR_CONCAVE {
                return ear;
            }
            ear = self.corners[ear].next;
        }
        ear
    }

    fn clip_all(&mut self) {
        let mut ear_init = self.head;
        let mut reverse = false;
        while self.count > 3 {
            let ear = self.find_ear(ear_init, reverse);
            if self.corners[ear].sign != EAR_CONVEX {
                self.concave -= 1;
            }
            let corner = self.corners[ear];
            self.triangles.push([corner.previous, ear, corner.next]);
            self.corners[corner.next].previous = corner.previous;
            self.corners[corner.previous].next = corner.next;
            if self.head == ear {
                self.head = corner.next;
            }
            self.count -= 1;
            for neighbor in [corner.previous, corner.next] {
                if self.corners[neighbor].sign != EAR_CONVEX {
                    let c = self.corners[neighbor];
                    self.corners[neighbor].sign =
                        corner_sign(self.coordinates, c.previous, neighbor, c.next);
                    if self.corners[neighbor].sign == EAR_CONVEX {
                        self.concave -= 1;
                    }
                }
            }
            ear_init = if reverse {
                self.corners[corner.previous].previous
            } else {
                self.corners[corner.next].next
            };
            if self.corners[ear_init].sign != EAR_CONVEX {
                ear_init = self.step(ear_init, reverse);
                reverse = !reverse;
            }
        }
        if self.count == 3 {
            let a = self.head;
            let b = self.corners[a].next;
            self.triangles.push([a, b, self.corners[b].next]);
        }
    }
}

fn float_face_tris_into(
    ring: &[usize],
    points: &[P3],
    coordinates: &mut Vec<[f32; 2]>,
    triangles: &mut Vec<[usize; 3]>,
    corners: &mut Vec<EarCorner>,
) {
    triangles.clear();
    if ring.len() == 3 {
        triangles.push([ring[0], ring[1], ring[2]]);
        return;
    }
    let normal = normal_poly(points);
    let (basis_x, basis_y) = orthogonal_basis(scale(normal, -1.0));
    coordinates.clear();
    coordinates.extend(
        points
            .iter()
            .map(|&point| [dot(basis_x, point), dot(basis_y, point)]),
    );
    float_polyfill_with_scratch(coordinates, corners, triangles);
    for triangle in triangles {
        for corner in triangle {
            *corner = ring[*corner];
        }
    }
}
fn beauty_ngon_tris_into(
    ring: &[usize],
    points: &[P3],
    coordinates: &mut Vec<[f32; 2]>,
    triangles: &mut Vec<[usize; 3]>,
    corners: &mut Vec<EarCorner>,
    beauty_scratch: &mut BeautyScratch,
) {
    triangles.clear();
    if ring.len() < 3 {
        return;
    }
    if ring.len() == 3 {
        triangles.push([ring[0], ring[1], ring[2]]);
        return;
    }
    let normal = normal_poly(points);
    let (basis_x, basis_y) = orthogonal_basis(scale(normal, -1.0));
    coordinates.clear();
    coordinates.extend(
        points
            .iter()
            .map(|&point| [dot(basis_x, point), dot(basis_y, point)]),
    );
    float_polyfill_with_scratch(coordinates, corners, triangles);
    beautify_with_scratch(coordinates, triangles, beauty_scratch);
    for triangle in triangles.iter_mut() {
        for corner in triangle {
            *corner = ring[*corner];
        }
    }
}
fn orthogonal_basis(normal: P3) -> (P3, P3) {
    let squared = normal[0] * normal[0] + normal[1] * normal[1];
    if squared > FLT_EPSILON {
        let scale = 1.0 / squared.sqrt();
        let first = [normal[1] * scale, -normal[0] * scale, 0.0];
        let second = [
            -normal[2] * first[1],
            normal[2] * first[0],
            normal[0] * first[1] - normal[1] * first[0],
        ];
        (first, second)
    } else {
        (
            [if normal[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    }
}

struct TriangleIntersectScratch {
    verts_a: Vec<usize>,
    verts_b: Vec<usize>,
    seen_a: HashSet<usize>,
    seen_b: HashSet<usize>,
}

impl TriangleIntersectScratch {
    fn new() -> Self {
        Self {
            verts_a: Vec::with_capacity(8),
            verts_b: Vec::with_capacity(8),
            seen_a: HashSet::with_capacity(8),
            seen_b: HashSet::with_capacity(8),
        }
    }

    fn clear(&mut self) {
        self.verts_a.clear();
        self.verts_b.clear();
        self.seen_a.clear();
        self.seen_b.clear();
    }
}

fn intersect_triangles(state: &mut State, eps: f32) -> (Bvh, Bvh) {
    let eps2x = eps * 2.0;
    let eps2x_sq = eps2x * eps2x;
    let margin = eps2x * 10.0;
    let margin_sq = margin * margin;
    let tree_a = Bvh::from_triangles(&state.tris, &state.verts, 0, margin);
    let tree_b = Bvh::from_triangles(&state.tris, &state.verts, 1, margin);
    let mut overlap = Vec::new();
    tree_b.overlap_pairs(&tree_a, &mut overlap);
    let mut scratch = TriangleIntersectScratch::new();
    for (b_index, a_index) in overlap {
        intersect_tri_pair(
            state,
            b_index,
            a_index,
            eps,
            eps2x,
            eps2x_sq,
            margin,
            margin_sq,
            &mut scratch,
        );
    }
    (tree_a, tree_b)
}

fn intersect_tri_pair(
    state: &mut State,
    a_index: usize,
    b_index: usize,
    eps: f32,
    eps2x: f32,
    eps2x_sq: f32,
    margin: f32,
    margin_sq: f32,
    scratch: &mut TriangleIntersectScratch,
) {
    scratch.clear();
    let TriangleIntersectScratch {
        verts_a,
        verts_b,
        seen_a,
        seen_b,
    } = scratch;
    let a = state.tris[a_index].verts;
    let b = state.tris[b_index].verts;
    if a.iter().any(|v| b.contains(v)) {
        return;
    }
    let face_a = state.tris[a_index].face;
    let face_b = state.tris[b_index].face;
    for &va in &a {
        for &vb in &b {
            if dist_sq(state.verts[va].co, state.verts[vb].co) <= eps2x_sq {
                push_unique(verts_a, seen_a, va);
                push_unique(verts_b, seen_b, vb);
            }
        }
    }
    for &va in &a {
        if seen_a.contains(&va) {
            continue;
        }
        for j in 0..3 {
            let e0 = b[j];
            let e1 = b[(j + 1) % 3];
            if seen_b.contains(&e0) || seen_b.contains(&e1) {
                continue;
            }
            let fac = line_point_factor(state.verts[va].co, state.verts[e0].co, state.verts[e1].co);
            if fac > -eps && fac < 1.0 + eps {
                let ix = interp(state.verts[e0].co, state.verts[e1].co, fac);
                if dist_sq(ix, state.verts[va].co) <= eps2x_sq {
                    push_unique(verts_b, seen_b, va);
                    let edge = edge_key_usize(e0, e1);
                    if state.edge_orig.contains_key(&edge) {
                        state.edge_splits.entry(edge).or_default().push(va);
                    }
                    break;
                }
            }
        }
    }
    for &vb in &b {
        if seen_b.contains(&vb) {
            continue;
        }
        for j in 0..3 {
            let e0 = a[j];
            let e1 = a[(j + 1) % 3];
            if seen_a.contains(&e0) || seen_a.contains(&e1) {
                continue;
            }
            let fac = line_point_factor(state.verts[vb].co, state.verts[e0].co, state.verts[e1].co);
            if fac > -eps && fac < 1.0 + eps {
                let ix = interp(state.verts[e0].co, state.verts[e1].co, fac);
                if dist_sq(ix, state.verts[vb].co) <= eps2x_sq {
                    push_unique(verts_a, seen_a, vb);
                    let edge = edge_key_usize(e0, e1);
                    if state.edge_orig.contains_key(&edge) {
                        state.edge_splits.entry(edge).or_default().push(vb);
                    }
                    break;
                }
            }
        }
    }
    for &va in &a {
        let p = state.verts[va].co;
        let t = scaled_triangle(state, b, eps2x);
        if !seen_a.contains(&va)
            && let Some(ix) = point_triangle(p, t)
            && dist_sq(ix, p) <= eps2x_sq
        {
            push_unique(verts_a, seen_a, va);
            push_unique(verts_b, seen_b, va);
        }
    }
    for &vb in &b {
        let p = state.verts[vb].co;
        let t = scaled_triangle(state, a, eps2x);
        if !seen_b.contains(&vb)
            && let Some(ix) = point_triangle(p, t)
            && dist_sq(ix, p) <= eps2x_sq
        {
            push_unique(verts_a, seen_a, vb);
            push_unique(verts_b, seen_b, vb);
        }
    }
    if verts_a.len() >= 3 && verts_b.len() >= 3 {
        return;
    }

    let na = normal_tri(
        state.verts[a[0]].co,
        state.verts[a[1]].co,
        state.verts[a[2]].co,
    );
    let nb = normal_tri(
        state.verts[b[0]].co,
        state.verts[b[1]].co,
        state.verts[b[2]].co,
    );
    for i in 0..3 {
        let e0 = a[i];
        let e1 = a[(i + 1) % 3];
        if seen_a.contains(&e0) || seen_a.contains(&e1) {
            continue;
        }
        let hit = cached_edge_tri_hit(state, e0, e1, b_index, b).or_else(|| {
            intersect_line_triangle(
                state.verts[e0].co,
                state.verts[e1].co,
                b.map(|v| state.verts[v].co),
                nb,
                eps,
                margin,
                margin_sq,
            )
            .map(|(ix, side)| {
                (
                    cached_edge_tri_vertex(state, e0, e1, b_index, b, side, ix),
                    side,
                )
            })
        });
        if let Some((v, _)) = hit {
            push_unique(verts_a, seen_a, v);
            push_unique(verts_b, seen_b, v);
            if state.edge_orig.contains_key(&edge_key_usize(e0, e1)) {
                state
                    .edge_splits
                    .entry(edge_key_usize(e0, e1))
                    .or_default()
                    .push(v);
            }
        }
    }
    for i in 0..3 {
        let e0 = b[i];
        let e1 = b[(i + 1) % 3];
        if seen_b.contains(&e0) || seen_b.contains(&e1) {
            continue;
        }
        let hit = cached_edge_tri_hit(state, e0, e1, a_index, a).or_else(|| {
            intersect_line_triangle(
                state.verts[e0].co,
                state.verts[e1].co,
                a.map(|v| state.verts[v].co),
                na,
                eps,
                margin,
                margin_sq,
            )
            .map(|(ix, side)| {
                (
                    cached_edge_tri_vertex(state, e0, e1, a_index, a, side, ix),
                    side,
                )
            })
        });
        if let Some((v, _)) = hit {
            push_unique(verts_a, seen_a, v);
            push_unique(verts_b, seen_b, v);
            if state.edge_orig.contains_key(&edge_key_usize(e0, e1)) {
                state
                    .edge_splits
                    .entry(edge_key_usize(e0, e1))
                    .or_default()
                    .push(v);
            }
        }
    }
    if verts_a.len() == 2 {
        let edge = edge_key_usize(verts_a[0], verts_a[1]);
        state.wire_edges.insert(edge);
        if !face_has_edge(&state.faces[face_a], edge) {
            let cuts = state.face_cuts.entry(face_a).or_default();
            if !cuts.contains(&edge) {
                cuts.push(edge);
            }
        }
    }
    if verts_b.len() == 2 {
        let edge = edge_key_usize(verts_b[0], verts_b[1]);
        state.wire_edges.insert(edge);
        if !face_has_edge(&state.faces[face_b], edge) {
            let cuts = state.face_cuts.entry(face_b).or_default();
            if !cuts.contains(&edge) {
                cuts.push(edge);
            }
        }
    }
}

/// The four cache keys of an edge-triangle intersection, as Blender's `k_arr` (`bmesh_intersect.cc`):
/// index 0 is the edge with the triangle, indices 1-3 are the edge with each triangle edge.
fn edge_tri_keys(e0: usize, e1: usize, tri_index: usize, tri: [usize; 3]) -> [[i32; 4]; 4] {
    let (a, b) = if e0 < e1 { (e0, e1) } else { (e1, e0) };
    let mut keys = [[0_i32; 4]; 4];
    keys[0] = [a as i32, b as i32, tri_index as i32, -1];
    for i in 0..3 {
        let mut x = tri[i] as i32;
        let mut y = tri[(i + 1) % 3] as i32;
        if x > y {
            std::mem::swap(&mut x, &mut y);
        }
        let mut key = [a as i32, b as i32, x, y];
        if key[0] > key[2] {
            key.swap(0, 2);
            key.swap(1, 3);
        }
        keys[i + 1] = key;
    }
    keys
}

/// Blender's `bm_isect_edge_tri` cache lookup (bmesh_intersect.cc:395-417), run before the line
/// predicate: the triangle edges are tried in order, then the triangle. Returns the cached vertex
/// and the side it was cached for (3 for the triangle).
fn cached_edge_tri_hit(
    state: &State,
    e0: usize,
    e1: usize,
    tri_index: usize,
    tri: [usize; 3],
) -> Option<(usize, usize)> {
    let keys = edge_tri_keys(e0, e1, tri_index, tri);
    for key_index in [1, 2, 3, 0] {
        if let Some(&id) = state.edge_tri_cache.get(&keys[key_index]) {
            let side = if key_index == 0 { 3 } else { key_index - 1 };
            return Some((id, side));
        }
    }
    None
}

fn cached_edge_tri_vertex(
    state: &mut State,
    e0: usize,
    e1: usize,
    tri_index: usize,
    tri: [usize; 3],
    side: usize,
    co: P3,
) -> usize {
    let (a, b) = if e0 < e1 { (e0, e1) } else { (e1, e0) };
    let keys = edge_tri_keys(e0, e1, tri_index, tri);
    let id = if let Some((id, _)) = cached_edge_tri_hit(state, e0, e1, tri_index, tri) {
        id
    } else {
        let id = state.add_vertex(co, NO_INDEX);
        let key = if side < 3 { keys[side + 1] } else { keys[0] };
        state.edge_tri_cache.insert(key, id);
        // Blender adds the vertex to the dissolve list only when the cut segment has no BMEdge
        // (`bmesh_intersect.cc:440-447`); a point on a real or wire edge is spliced instead.
        let segment = edge_key_usize(a, b);
        if !state.edge_orig.contains_key(&segment) && !state.wire_edges.contains(&segment) {
            state.dissolve_verts.push(id);
        }
        id
    };
    if side < 3 {
        let target_edge = edge_key_usize(tri[side], tri[(side + 1) % 3]);
        if state.edge_orig.contains_key(&target_edge) {
            state.edge_splits.entry(target_edge).or_default().push(id);
        }
    }
    id
}

/// A face corner, the Potter counterpart of Blender's `BMLoop`. `pos` indexes `verts` of `face`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Corner {
    face: usize,
    pos: usize,
}

impl Corner {
    fn vert(self, state: &State) -> usize {
        state.faces[self.face].verts[self.pos]
    }

    fn next(self, state: &State) -> Self {
        let len = state.faces[self.face].verts.len();
        Self {
            face: self.face,
            pos: (self.pos + 1) % len,
        }
    }

    fn prev(self, state: &State) -> Self {
        let len = state.faces[self.face].verts.len();
        Self {
            face: self.face,
            pos: (self.pos + len - 1) % len,
        }
    }
}

/// Every face corner at `vertex` (Blender's loops around a vertex).
fn vertex_corners(state: &State, vertex: usize) -> Vec<Corner> {
    let mut corners = Vec::new();
    for (face, data) in state.faces.iter().enumerate() {
        for (pos, &v) in data.verts.iter().enumerate() {
            if v == vertex {
                corners.push(Corner { face, pos });
            }
        }
    }
    corners
}

/// Vertices joined to `vertex` by an edge: face-ring neighbours and wire (cut) edges.
fn vertex_neighbours(state: &State, vertex: usize) -> HashSet<usize> {
    let mut neighbours = HashSet::new();
    for corner in vertex_corners(state, vertex) {
        neighbours.insert(corner.prev(state).vert(state));
        neighbours.insert(corner.next(state).vert(state));
    }
    for &(a, b) in &state.wire_edges {
        if a == vertex {
            neighbours.insert(b);
        } else if b == vertex {
            neighbours.insert(a);
        }
    }
    neighbours
}

/// Blender `BM_vert_pair_share_face_check`.
fn vertices_share_face(state: &State, a: usize, b: usize) -> bool {
    state
        .faces
        .iter()
        .any(|face| face.verts.contains(&a) && face.verts.contains(&b))
}

/// Blender `BM_vert_splice_check_double_edge`: both vertices reach a common vertex by an edge.
fn splice_double_edge(state: &State, a: usize, b: usize) -> bool {
    let neighbours_a = vertex_neighbours(state, a);
    vertex_neighbours(state, b)
        .iter()
        .any(|vertex| neighbours_a.contains(vertex))
}

/// Blender `BM_face_pair_overlap_check_subset_{same,swap}_winding`: walk both corner runs in step
/// (backwards on `b` when `swap`) and require the same vertex at every step.
fn corner_run_matches(
    state: &State,
    mut a: Corner,
    a_end: Corner,
    mut b: Corner,
    swap: bool,
) -> bool {
    let bound = state.faces[a.face].verts.len() + 1;
    for _ in 0..bound {
        if b.vert(state) != a.vert(state) {
            return false;
        }
        if a == a_end {
            return true;
        }
        a = a.next(state);
        b = if swap { b.prev(state) } else { b.next(state) };
    }
    false
}

/// Blender `BM_vert_splice_check_double_face`: the vertices sit on matching corners of faces of
/// equal length, with the same neighbours (in either winding) and the same run of vertices.
fn splice_double_face(state: &State, a: usize, b: usize) -> bool {
    let corners_a = vertex_corners(state, a);
    let corners_b = vertex_corners(state, b);
    for &la in &corners_a {
        for &lb in &corners_b {
            if state.faces[la.face].verts.len() != state.faces[lb.face].verts.len() {
                continue;
            }
            let (an, ap) = (la.next(state), la.prev(state));
            let (bn, bp) = (lb.next(state), lb.prev(state));
            if an.vert(state) == bn.vert(state) && ap.vert(state) == bp.vert(state) {
                if corner_run_matches(state, an.next(state), ap.prev(state), bn.next(state), false)
                {
                    return true;
                }
            } else if an.vert(state) == bp.vert(state)
                && ap.vert(state) == bn.vert(state)
                && corner_run_matches(state, an.next(state), ap.prev(state), bp.prev(state), true)
            {
                return true;
            }
        }
    }
    false
}

/// Blender `BM_edge_split` on the current segment `tail..end`: `split` joins every face that holds
/// the segment, and a wire segment is replaced by its two halves.
fn insert_split_vertex(state: &mut State, tail: usize, end: usize, split: usize, wire: bool) {
    let segment = edge_key_usize(tail, end);
    for face in &mut state.faces {
        let len = face.verts.len();
        if let Some(i) =
            (0..len).find(|&i| edge_key_usize(face.verts[i], face.verts[(i + 1) % len]) == segment)
        {
            face.verts.insert(i + 1, split);
        }
    }
    if wire {
        state.wire_edges.remove(&segment);
        state.wire_edges.insert(edge_key_usize(tail, split));
        state.wire_edges.insert(edge_key_usize(split, end));
    }
}

/// Blender `BM_vert_splice(bm, keep, split)`: `split`'s faces and edges move to `keep`.
fn splice_split_vertex(state: &mut State, split: usize, keep: usize) {
    for face in &mut state.faces {
        for v in &mut face.verts {
            if *v == split {
                *v = keep;
            }
        }
    }
    let moved: Vec<EdgeKey> = state
        .wire_edges
        .iter()
        .copied()
        .filter(|&(a, b)| a == split || b == split)
        .collect();
    for (a, b) in moved {
        state.wire_edges.remove(&(a, b));
        let other = if a == split { b } else { a };
        state.wire_edges.insert(edge_key_usize(keep, other));
    }
}

/// Blender's `USE_SPLICE` loop (`bmesh_intersect.cc` `BM_mesh_intersect`), run per input edge in
/// order. Each point splits the current segment at a new vertex `split`. The point then replaces
/// `split` when no edge, shared face, double edge or double face joins them; otherwise `split`
/// takes the point's coordinates and stays in the ring, the point is left out of the ring, and
/// the remaining points on this edge are not reached.
fn split_input_edges(state: &mut State) {
    let mut edge_splits = std::mem::take(&mut state.edge_splits);
    let edges: Vec<(EdgeKey, i32, usize, Vec<usize>, bool)> = state
        .edge_order
        .iter()
        .filter_map(|&key| {
            let points = edge_splits.remove(&key)?;
            let source = state.edge_orig.get(&key).copied()?;
            let start = state.edge_start.get(&key).copied().unwrap_or(key.0);
            Some((key, source, start, points, state.wire_edges.contains(&key)))
        })
        .collect();

    for (key, source, start, mut points, wire) in edges {
        let end = if key.0 == start { key.1 } else { key.0 };
        points.sort_by(|&a, &b| {
            dist_sq(state.verts[start].co, state.verts[a].co)
                .total_cmp(&dist_sq(state.verts[start].co, state.verts[b].co))
        });
        points.dedup();
        points.retain(|&v| v != start && v != end);
        if points.is_empty() {
            continue;
        }
        let mut chain = Vec::with_capacity(points.len());
        let mut tail = start;
        for vi in points {
            let split = state.verts.len();
            let point = state.verts[vi].clone();
            state.verts.push(point);
            insert_split_vertex(state, tail, end, split, wire);
            let touching = vertex_neighbours(state, split).contains(&vi);
            if !touching
                && !vertices_share_face(state, split, vi)
                && !splice_double_edge(state, split, vi)
                && !splice_double_face(state, split, vi)
            {
                splice_split_vertex(state, split, vi);
                state.verts.pop();
                chain.push(vi);
                tail = vi;
            } else {
                chain.push(split);
                break;
            }
        }
        state.edge_orig.remove(&key);
        state.edge_start.remove(&key);
        let mut previous = start;
        for &point in chain.iter().chain(std::iter::once(&end)) {
            let segment = edge_key_usize(previous, point);
            state.edge_orig.insert(segment, source);
            state.edge_start.insert(segment, previous);
            state.edge_order.push(segment);
            previous = point;
        }
    }
}

fn split_faces(state: &mut State) {
    let old_faces = std::mem::take(&mut state.faces);
    let old_cuts = std::mem::take(&mut state.face_cuts);
    let positions: Vec<P3> = state.verts.iter().map(|vertex| vertex.co).collect();
    let mut new_faces: Vec<Face> = Vec::new();
    let mut dangling: HashSet<EdgeKey> = HashSet::new();
    for (index, face) in old_faces.iter().enumerate() {
        let Some(cuts) = old_cuts.get(&index) else {
            new_faces.push(face.clone());
            continue;
        };
        // Blender's edge net: the cut edges in stored order (most recent first), then the connect
        // edges that `BM_face_split_edgenet_connect_islands` appends.
        let net: Vec<(usize, usize)> = cuts.iter().rev().copied().collect();
        let loop_edges: Vec<(usize, usize)> = (0..face.verts.len())
            .map(|i| (face.verts[i], face.verts[(i + 1) % face.verts.len()]))
            .collect();
        let points: Vec<P3> = face.verts.iter().map(|&v| state.verts[v].co).collect();
        let normal = normal_poly(&points);
        let connects = super::float_connect::connect_islands(
            face.verts[0],
            normal,
            &loop_edges,
            &net,
            &positions,
        );
        let mut edge_net = net;
        edge_net.extend(connects);
        // The mesh's other faces: the originals not yet split, then the faces already emitted.
        let mut mesh_faces: Vec<Vec<usize>> = old_faces[index + 1..]
            .iter()
            .map(|other| other.verts.clone())
            .collect();
        mesh_faces.extend(new_faces.iter().map(|other| other.verts.clone()));
        let split = super::float_split_net::face_split_edgenet_with_mesh(
            &face.verts,
            &edge_net,
            &positions,
            normal,
            &mesh_faces,
        );
        if split.faces.is_empty() {
            dangling.extend(edge_net.iter().map(|&(a, b)| edge_key_usize(a, b)));
            new_faces.push(face.clone());
        } else {
            dangling.extend(split.dangling.iter().map(|&(a, b)| edge_key_usize(a, b)));
            for ring in split.faces {
                new_faces.push(Face {
                    verts: ring,
                    side: face.side,
                    source: face.source,
                });
            }
        }
    }
    state.faces = new_faces;
    state.dangling_edges = dangling;
}

/// The edges of a face ring, each as a sorted key.
fn ring_edges(verts: &[usize]) -> Vec<EdgeKey> {
    (0..verts.len())
        .map(|i| edge_key_usize(verts[i], verts[(i + 1) % verts.len()]))
        .collect()
}

fn classify_faces(state: &mut State, operation: Operation, bvh_a: &Bvh, bvh_b: &Bvh) {
    let mut edge_faces = HashMap::<EdgeKey, Vec<usize>>::new();
    for (fi, face) in state.faces.iter().enumerate() {
        for i in 0..face.verts.len() {
            let edge = edge_key_usize(face.verts[i], face.verts[(i + 1) % face.verts.len()]);
            edge_faces.entry(edge).or_default().push(fi);
        }
    }
    let mut groups = Vec::with_capacity(state.faces.len());
    let mut group_faces = Vec::with_capacity(state.faces.len());
    let mut seen = vec![false; state.faces.len()];
    let mut queue = VecDeque::with_capacity(state.faces.len());
    for start in 0..state.faces.len() {
        if seen[start] {
            continue;
        }
        let group_start = group_faces.len();
        let side = state.faces[start].side;
        queue.clear();
        queue.push_back(start);
        seen[start] = true;
        while let Some(face) = queue.pop_front() {
            group_faces.push(face);
            for i in 0..state.faces[face].verts.len() {
                let edge = edge_key_usize(
                    state.faces[face].verts[i],
                    state.faces[face].verts[(i + 1) % state.faces[face].verts.len()],
                );
                if state.wire_edges.contains(&edge) {
                    continue;
                }
                if let Some(adjacent) = edge_faces.get(&edge)
                    && adjacent.len() > 1
                {
                    let same_side = adjacent
                        .iter()
                        .all(|&other| state.faces[other].side == side);
                    if same_side {
                        for &other in adjacent {
                            if !seen[other] {
                                seen[other] = true;
                                queue.push_back(other);
                            }
                        }
                    }
                }
            }
        }
        groups.push((group_start, group_faces.len()));
    }
    let mut removed = vec![false; state.faces.len()];
    let mut ray_candidates = Vec::new();
    let mut ray_depths = Vec::new();
    let mut ray_stack = Vec::new();
    let mut flipped = vec![false; state.faces.len()];
    let mut face_scratch = FacePointScratch::default();
    for (start, end) in groups {
        let face_index = group_faces[start];
        let face = &state.faces[face_index];
        let point = point_in_face(face, &state.verts, &mut face_scratch);
        let other_bvh = if face.side == 0 { bvh_b } else { bvh_a };
        let hits = ray_hits(
            point,
            other_bvh,
            &state.tris,
            &state.verts,
            &mut ray_candidates,
            &mut ray_depths,
            &mut ray_stack,
        );
        let odd = hits & 1 == 1;
        let (do_remove, do_flip) = match operation {
            Operation::Intersect => (!odd, false),
            Operation::Union => (odd, false),
            Operation::Difference => (odd == (face.side == 0), face.side == 1),
        };
        for &index in &group_faces[start..end] {
            removed[index] = do_remove;
            flipped[index] = !do_remove && do_flip;
        }
    }
    let mut output = Vec::new();
    let mut removed_edges: HashSet<EdgeKey> = HashSet::new();
    for (index, mut face) in state.faces.drain(..).enumerate() {
        if removed[index] {
            removed_edges.extend(ring_edges(&face.verts));
            continue;
        }
        if flipped[index] && face.verts.len() > 2 {
            face.verts[1..].reverse();
        }
        output.push(face);
    }
    // Blender's `BM_face_kill_loose` (bmesh_core.cc:1001) kills an edge that loses its last loop
    // when its face is removed. An edge with no loop from the start is a chord and survives.
    let mut kept_edges: HashSet<EdgeKey> = HashSet::new();
    for face in &output {
        kept_edges.extend(ring_edges(&face.verts));
    }
    for edge in removed_edges.difference(&kept_edges) {
        state.wire_edges.remove(edge);
        state.dangling_edges.remove(edge);
    }
    state.faces = output;
}

/// Blender `BM_vert_collapse_check_double_face` (`bmesh_core.cc:2290`) for a collapse at `vertex` that
/// joins `a` and `b`. Faces at `vertex` with more than three corners are tested against the faces on
/// the existing edge `a`-`b`; a same-size pair whose corner runs match would become a double face.
/// Returns false when no face loops the edge `a`-`b`, as Blender does for an absent or wire edge.
fn collapse_double_face(state: &State, vertex: usize, a: usize, b: usize) -> bool {
    let mut exists = Vec::new();
    for (face, data) in state.faces.iter().enumerate() {
        let len = data.verts.len();
        for pos in 0..len {
            let (x, y) = (data.verts[pos], data.verts[(pos + 1) % len]);
            if (x == a && y == b) || (x == b && y == a) {
                exists.push(Corner { face, pos });
            }
        }
    }
    if exists.is_empty() {
        return false;
    }
    for collapse in vertex_corners(state, vertex) {
        let collapse_len = state.faces[collapse.face].verts.len();
        if collapse_len <= 3 {
            continue;
        }
        let beg = collapse.next(state);
        let end = collapse.prev(state);
        for &exists_loop in &exists {
            if collapse_len - 1 != state.faces[exists_loop.face].verts.len() {
                continue;
            }
            let overlap = if beg.vert(state) == exists_loop.vert(state) {
                corner_run_matches(
                    state,
                    beg.next(state),
                    end.prev(state),
                    exists_loop.prev(state),
                    true,
                )
            } else {
                corner_run_matches(
                    state,
                    beg.next(state),
                    end.prev(state),
                    exists_loop.next(state),
                    false,
                )
            };
            if overlap {
                return true;
            }
        }
    }
    false
}

fn dissolve_intersection_vertices(state: &mut State) {
    let mut candidates = state.dissolve_verts.clone();
    candidates.sort_unstable();
    candidates.dedup();

    let mut faces_at_vertex = vec![Vec::new(); state.verts.len()];
    let mut edges_at_vertex: Vec<HashSet<EdgeKey>> =
        (0..state.verts.len()).map(|_| HashSet::new()).collect();
    for (face_index, face) in state.faces.iter().enumerate() {
        for i in 0..face.verts.len() {
            let a = face.verts[i];
            let b = face.verts[(i + 1) % face.verts.len()];
            faces_at_vertex[a].push(face_index);
            edges_at_vertex[a].insert(edge_key_usize(a, b));
            edges_at_vertex[b].insert(edge_key_usize(a, b));
        }
    }
    for &edge in state.edge_orig.keys().chain(state.wire_edges.iter()) {
        edges_at_vertex[edge.0].insert(edge);
        edges_at_vertex[edge.1].insert(edge);
    }

    let mut neighbors = Vec::new();
    let mut face_indices = Vec::new();
    for vertex in candidates {
        neighbors.clear();
        neighbors.extend(
            edges_at_vertex[vertex]
                .iter()
                .map(|edge| if edge.0 == vertex { edge.1 } else { edge.0 }),
        );
        neighbors.sort_unstable();
        neighbors.dedup();
        if neighbors.len() != 2 {
            continue;
        }
        std::mem::swap(&mut face_indices, &mut faces_at_vertex[vertex]);
        let has_triangle = face_indices
            .iter()
            .any(|&index| state.faces[index].verts.len() == 3);
        if has_triangle {
            continue;
        }
        let (a, b) = (neighbors[0], neighbors[1]);
        let new_edge = edge_key_usize(a, b);
        // Blender: a collapse onto an existing edge is refused only when it would make a double face
        // (`BM_vert_collapse_check_double_face`, bmesh_core.cc:2290); an absent edge never does.
        if (edges_at_vertex[a].contains(&new_edge) || edges_at_vertex[b].contains(&new_edge))
            && collapse_double_face(state, vertex, a, b)
        {
            continue;
        }

        let mut reduced = Vec::new();
        for &face_index in &face_indices {
            let face = &state.faces[face_index];
            if face.verts.len() < 3 || !face.verts.contains(&vertex) {
                continue;
            }
            let ring: Vec<usize> = face
                .verts
                .iter()
                .copied()
                .filter(|&v| v != vertex)
                .collect();
            if ring.len() >= 3 {
                reduced.push((face_index, ring));
            }
        }

        let source = state
            .edge_orig
            .get(&edge_key_usize(a, vertex))
            .or_else(|| state.edge_orig.get(&edge_key_usize(b, vertex)))
            .copied();
        let wire = state.wire_edges.contains(&edge_key_usize(a, vertex))
            || state.wire_edges.contains(&edge_key_usize(b, vertex));
        for &face_index in &face_indices {
            let old = std::mem::take(&mut state.faces[face_index].verts);
            if old.is_empty() {
                continue;
            }
            let ring: Vec<usize> = old.iter().copied().filter(|&v| v != vertex).collect();
            if ring.len() >= 3 {
                state.faces[face_index].verts = ring;
            } else {
                for old_vertex in old {
                    faces_at_vertex[old_vertex].retain(|&index| index != face_index);
                }
            }
        }
        let edge_a = edge_key_usize(a, vertex);
        let edge_b = edge_key_usize(b, vertex);
        edges_at_vertex[a].remove(&edge_a);
        edges_at_vertex[vertex].remove(&edge_a);
        edges_at_vertex[b].remove(&edge_b);
        edges_at_vertex[vertex].remove(&edge_b);
        edges_at_vertex[a].insert(new_edge);
        edges_at_vertex[b].insert(new_edge);
        edges_at_vertex[vertex].clear();
        state.edge_orig.remove(&edge_a);
        state.edge_orig.remove(&edge_b);
        state.edge_start.remove(&edge_a);
        state.edge_start.remove(&edge_b);
        if let Some(source) = source {
            state.edge_orig.insert(new_edge, source);
            state.edge_start.insert(new_edge, a);
        }
        state.wire_edges.remove(&edge_a);
        state.wire_edges.remove(&edge_b);
        if wire {
            state.wire_edges.insert(new_edge);
        }
        let dangling_a = state.dangling_edges.remove(&edge_a);
        let dangling_b = state.dangling_edges.remove(&edge_b);
        if dangling_a || dangling_b {
            state.dangling_edges.insert(new_edge);
        }
    }
    state.faces.retain(|face| face.verts.len() >= 3);
}

fn ray_hits(
    origin: P3,
    tree: &Bvh,
    tris: &[Triangle],
    verts: &[Vertex],
    candidates: &mut Vec<usize>,
    depths: &mut Vec<f32>,
    stack: &mut Vec<usize>,
) -> usize {
    candidates.clear();
    tree.ray_candidates(origin, candidates, stack);
    depths.clear();
    for &index in candidates.iter() {
        let tri = tris[index];
        let p = tri.verts.map(|v| verts[v].co);
        if let Some(distance) = ray_triangle(origin, p)
            && distance >= 0.0
        {
            depths.push(distance);
        }
    }
    depths.sort_by(f32::total_cmp);
    let mut count = 0;
    let mut last = 0.0;
    for &depth in depths.iter() {
        if count == 0 || depth - last > FLT_EPSILON * 10.0 {
            last = depth;
            count += 1;
        }
    }
    count
}

fn ray_triangle(origin: P3, tri: [P3; 3]) -> Option<f32> {
    let direction = [1.0, 0.0, 0.0];
    let e1 = sub(tri[1], tri[0]);
    let e2 = sub(tri[2], tri[0]);
    let p = cross(direction, e2);
    let a = dot(e1, p);
    if a == 0.0 {
        return None;
    }
    let f = 1.0 / a;
    let s = sub(origin, tri[0]);
    let u = f * dot(s, p);
    if u < -FLT_EPSILON || u > 1.0 + FLT_EPSILON {
        return None;
    }
    let q = cross(s, e1);
    let v = f * dot(direction, q);
    if v < -FLT_EPSILON || u + v > 1.0 + FLT_EPSILON {
        return None;
    }
    let t = f * dot(e2, q);
    (t >= 0.0).then_some(t)
}

#[derive(Clone, Copy)]
struct BvhNode {
    min: P3,
    max: P3,
    start: usize,
    count: usize,
    right: usize,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    entries: Vec<usize>,
}

impl Bvh {
    fn from_triangles(tris: &[Triangle], verts: &[Vertex], side: u8, pad: f32) -> Self {
        let entries: Vec<usize> = tris
            .iter()
            .enumerate()
            .filter_map(|(index, tri)| (tri.side == side).then_some(index))
            .collect();
        let mut bounds = Vec::with_capacity(tris.len());
        for tri in tris {
            let p = tri.verts.map(|v| verts[v].co);
            bounds.push((
                [
                    p[0][0].min(p[1][0]).min(p[2][0]) - pad,
                    p[0][1].min(p[1][1]).min(p[2][1]) - pad,
                    p[0][2].min(p[1][2]).min(p[2][2]) - pad,
                ],
                [
                    p[0][0].max(p[1][0]).max(p[2][0]) + pad,
                    p[0][1].max(p[1][1]).max(p[2][1]) + pad,
                    p[0][2].max(p[1][2]).max(p[2][2]) + pad,
                ],
            ));
        }
        let mut tree = Self {
            nodes: Vec::new(),
            entries,
        };
        let mut boxes = bounds;
        if !tree.entries.is_empty() {
            let len = tree.entries.len();
            tree.build(0, len, &mut boxes);
        }
        tree
    }

    fn build(&mut self, start: usize, end: usize, bounds: &mut [(P3, P3)]) -> usize {
        let index = self.nodes.len();
        self.nodes.push(BvhNode {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            start,
            count: 0,
            right: 0,
        });
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for &entry in &self.entries[start..end] {
            for axis in 0..3 {
                min[axis] = min[axis].min(bounds[entry].0[axis]);
                max[axis] = max[axis].max(bounds[entry].1[axis]);
            }
        }
        if end - start <= 8 {
            self.nodes[index] = BvhNode {
                min,
                max,
                start,
                count: end - start,
                right: 0,
            };
            return index;
        }
        let axis = (0..3)
            .max_by(|&a, &b| (max[a] - min[a]).total_cmp(&(max[b] - min[b])))
            .unwrap_or(0);
        let mid = start + (end - start) / 2;
        self.entries[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            #[expect(
                clippy::manual_midpoint,
                reason = "keeps (min + max) * 0.5 centre arithmetic bit-identical to the FLOAT kernel"
            )]
            let ca = (bounds[a].0[axis] + bounds[a].1[axis]) * 0.5;
            #[expect(
                clippy::manual_midpoint,
                reason = "keeps (min + max) * 0.5 centre arithmetic bit-identical to the FLOAT kernel"
            )]
            let cb = (bounds[b].0[axis] + bounds[b].1[axis]) * 0.5;
            ca.total_cmp(&cb)
        });
        self.build(start, mid, bounds);
        let right = self.build(mid, end, bounds);
        self.nodes[index] = BvhNode {
            min,
            max,
            start: 0,
            count: 0,
            right,
        };
        index
    }

    fn overlap_pairs(&self, other: &Self, out: &mut Vec<(usize, usize)>) {
        if self.nodes.is_empty() || other.nodes.is_empty() {
            return;
        }
        let mut stack = vec![(0_usize, 0_usize)];
        while let Some((a, b)) = stack.pop() {
            let na = self.nodes[a];
            let nb = other.nodes[b];
            if !bounds_overlap(na.min, na.max, nb.min, nb.max) {
                continue;
            }
            if na.count > 0 && nb.count > 0 {
                for &ia in &self.entries[na.start..na.start + na.count] {
                    for &ib in &other.entries[nb.start..nb.start + nb.count] {
                        out.push((ia, ib));
                    }
                }
            } else if na.count == 0 && (nb.count > 0 || box_size(na) >= box_size(nb)) {
                stack.push((na.right, b));
                stack.push((a + 1, b));
            } else {
                stack.push((a, nb.right));
                stack.push((a, b + 1));
            }
        }
    }

    fn ray_candidates(&self, origin: P3, out: &mut Vec<usize>, stack: &mut Vec<usize>) {
        out.clear();
        stack.clear();
        if self.nodes.is_empty() {
            return;
        }
        stack.push(0);
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if node.max[0] < origin[0]
                || origin[1] < node.min[1]
                || origin[1] > node.max[1]
                || origin[2] < node.min[2]
                || origin[2] > node.max[2]
            {
                continue;
            }
            if node.count > 0 {
                out.extend_from_slice(&self.entries[node.start..node.start + node.count]);
            } else {
                stack.push(node.right);
                stack.push(index + 1);
            }
        }
    }
}

fn bounds_overlap(a_min: P3, a_max: P3, b_min: P3, b_max: P3) -> bool {
    (0..3).all(|axis| a_min[axis] <= b_max[axis] && b_min[axis] <= a_max[axis])
}

fn box_size(node: BvhNode) -> f32 {
    (node.max[0] - node.min[0]) * (node.max[1] - node.min[1]) * (node.max[2] - node.min[2])
}

#[derive(Default)]
struct FacePointScratch {
    points: Vec<P3>,
    ring: Vec<usize>,
    coordinates: Vec<[f32; 2]>,
    triangles: Vec<[usize; 3]>,
    corners: Vec<EarCorner>,
}

fn point_in_face(face: &Face, verts: &[Vertex], scratch: &mut FacePointScratch) -> P3 {
    scratch.points.clear();
    scratch
        .points
        .extend(face.verts.iter().map(|&vertex| verts[vertex].co));
    scratch.ring.clear();
    scratch.ring.extend(0..scratch.points.len());
    float_face_tris_into(
        &scratch.ring,
        &scratch.points,
        &mut scratch.coordinates,
        &mut scratch.triangles,
        &mut scratch.corners,
    );
    let mut best = scratch.triangles.first().copied().unwrap_or([0, 1, 2]);
    let mut area_best = -1.0_f32;
    for &triangle in &scratch.triangles {
        let p = triangle.map(|corner| scratch.points[corner]);
        let area = len_sq(cross(sub(p[0], p[1]), sub(p[1], p[2])));
        if area > area_best {
            area_best = area;
            best = triangle;
        }
    }
    let p = best.map(|corner| scratch.points[corner]);
    [
        (p[0][0] + p[1][0] + p[2][0]) / 3.0,
        (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        (p[0][2] + p[1][2] + p[2][2]) / 3.0,
    ]
}

fn face_has_edge(face: &Face, edge: EdgeKey) -> bool {
    (0..face.verts.len())
        .any(|i| edge_key_usize(face.verts[i], face.verts[(i + 1) % face.verts.len()]) == edge)
}

fn edge_key_usize(a: usize, b: usize) -> EdgeKey {
    if a < b { (a, b) } else { (b, a) }
}

fn push_unique(list: &mut Vec<usize>, seen: &mut HashSet<usize>, value: usize) {
    if seen.insert(value) {
        list.push(value);
    }
}

fn finite3(value: P3) -> bool {
    value.iter().all(|component| component.is_finite())
}

fn to_f32(value: DVec3) -> P3 {
    [value.x as f32, value.y as f32, value.z as f32]
}

fn to_dvec3(value: P3) -> DVec3 {
    DVec3::new(
        f64::from(value[0]),
        f64::from(value[1]),
        f64::from(value[2]),
    )
}

fn transform_point(matrix: Mat4, point: P3) -> P3 {
    let columns = matrix.to_cols_array_2d();
    [
        ((columns[0][0] * point[0] + columns[1][0] * point[1]) + columns[2][0] * point[2])
            + columns[3][0],
        ((columns[0][1] * point[0] + columns[1][1] * point[1]) + columns[2][1] * point[2])
            + columns[3][1],
        ((columns[0][2] * point[0] + columns[1][2] * point[1]) + columns[2][2] * point[2])
            + columns[3][2],
    ]
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: P3, factor: f32) -> P3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: P3, b: P3) -> f32 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn len_sq(a: P3) -> f32 {
    dot(a, a)
}

fn dist_sq(a: P3, b: P3) -> f32 {
    len_sq(sub(a, b))
}

fn normalize(vector: P3) -> P3 {
    let length_squared = len_sq(vector);
    // Blender's normalize_v3_v3_length returns zero unless the squared length exceeds this cutoff.
    if length_squared > 1.0e-35 {
        scale(vector, 1.0 / length_squared.sqrt())
    } else {
        [0.0; 3]
    }
}

fn interp(a: P3, b: P3, factor: f32) -> P3 {
    let inverse = 1.0 - factor;
    [
        inverse * a[0] + factor * b[0],
        inverse * a[1] + factor * b[1],
        inverse * a[2] + factor * b[2],
    ]
}

fn line_point_factor(point: P3, a: P3, b: P3) -> f32 {
    let direction = sub(b, a);
    let denominator = len_sq(direction);
    if denominator > 0.0 {
        dot(direction, sub(point, a)) / denominator
    } else {
        0.0
    }
}

fn normal_tri(a: P3, b: P3, c: P3) -> P3 {
    normalize(cross(sub(a, b), sub(b, c)))
}

fn normal_poly(points: &[P3]) -> P3 {
    let mut normal = [0.0; 3];
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
        normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
        normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    normalize(normal)
}

fn scaled_triangle(state: &State, tri: [usize; 3], eps2x: f32) -> [P3; 3] {
    let center = scale(
        add(
            add(state.verts[tri[0]].co, state.verts[tri[1]].co),
            state.verts[tri[2]].co,
        ),
        1.0 / 3.0,
    );
    tri.map(|v| interp(center, state.verts[v].co, 1.0 - eps2x))
}

fn point_triangle(point: P3, triangle: [P3; 3]) -> Option<P3> {
    if !point_in_slice(point, triangle[0], triangle[1], triangle[2])
        || !point_in_slice(point, triangle[1], triangle[2], triangle[0])
        || !point_in_slice(point, triangle[2], triangle[0], triangle[1])
    {
        return None;
    }
    let normal = cross(sub(triangle[0], triangle[1]), sub(triangle[1], triangle[2]));
    let side = dot(normal, sub(point, triangle[0]));
    Some(sub(point, scale(normal, side / len_sq(normal))))
}

fn point_in_slice(point: P3, vertex: P3, line_start: P3, line_end: P3) -> bool {
    let direction = sub(line_end, line_start);
    let closest = add(
        line_start,
        scale(direction, line_point_factor(vertex, line_start, line_end)),
    );
    let toward_line = sub(closest, vertex);
    let length_squared = len_sq(toward_line);
    if length_squared == 0.0 {
        return false;
    }
    let factor = dot(toward_line, sub(point, vertex)) / length_squared;
    (0.0..=1.0).contains(&factor)
}

fn line_line_epsilon(p0: P3, p1: P3, p2: P3, p3: P3, epsilon: f32) -> (u8, P3, P3) {
    let mut c = sub(p2, p0);
    let mut a = sub(p1, p0);
    let mut b = sub(p3, p2);
    let ab = cross(a, b);
    let d = dot(c, ab);
    let div = dot(ab, ab);
    if div == 0.0 {
        return (0, p0, p0);
    }
    if d.abs() <= epsilon {
        let cb = cross(c, b);
        a = scale(a, dot(cb, ab) / div);
        let intersection = add(p0, a);
        return (1, intersection, intersection);
    }
    let mut offset = sub(p0, p2);
    let normal = cross(a, b);
    offset = scale(normal, dot(offset, normal) / dot(normal, normal));
    let p2_shifted = add(p2, offset);
    let p3_shifted = add(p3, offset);
    c = sub(p2_shifted, p0);
    a = sub(p1, p0);
    b = sub(p3_shifted, p2_shifted);
    let ab = cross(a, b);
    let cb = cross(c, b);
    a = scale(a, dot(cb, ab) / dot(ab, ab));
    let intersection = add(p0, a);
    (2, intersection, sub(intersection, offset))
}

fn line_segment_triangle(p0: P3, p1: P3, triangle: [P3; 3]) -> Option<f32> {
    let e1 = sub(triangle[1], triangle[0]);
    let e2 = sub(triangle[2], triangle[0]);
    let direction = sub(p1, p0);
    let p = cross(direction, e2);
    let determinant = dot(e1, p);
    if determinant == 0.0 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let s = sub(p0, triangle[0]);
    let u = inverse * dot(s, p);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = cross(s, e1);
    let v = inverse * dot(direction, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let lambda = inverse * dot(e2, q);
    (0.0..=1.0).contains(&lambda).then_some(lambda)
}

fn intersect_line_triangle(
    p0: P3,
    p1: P3,
    triangle: [P3; 3],
    normal: P3,
    eps: f32,
    margin: f32,
    margin_sq: f32,
) -> Option<(P3, usize)> {
    let direction = normalize(sub(p0, p1));
    for i in 0..3 {
        let j = (i + 1) % 3;
        let edge_dir = normalize(sub(triangle[i], triangle[j]));
        if dot(direction, edge_dir).abs() >= 1.0 - eps {
            continue;
        }
        let (kind, x0, x1) = line_line_epsilon(p0, p1, triangle[i], triangle[j], 0.0);
        if kind == 0 {
            continue;
        }
        let (x0, x1) = if kind == 1 { (x0, x0) } else { (x0, x1) };
        let tri_factor = line_point_factor(x1, triangle[i], triangle[j]);
        let line_factor = line_point_factor(x0, p0, p1);
        if (kind == 1 || dist_sq(x0, x1) <= margin_sq)
            && tri_factor >= margin
            && tri_factor <= 1.0 - margin
            && line_factor >= margin
            && line_factor <= 1.0 - margin
        {
            return Some((x0, i));
        }
    }
    if dot(direction, normal).abs() >= eps
        && let Some(factor) = line_segment_triangle(p0, p1, triangle)
        && factor >= margin
        && factor <= 1.0 - margin
    {
        let intersection = interp(p0, p1, factor);
        let min_dist = triangle
            .iter()
            .map(|&vertex| dist_sq(vertex, intersection))
            .fold(f32::INFINITY, f32::min);
        if min_dist >= margin_sq {
            return Some((intersection, 3));
        }
    }
    None
}

fn reverse_mesh_faces(mesh: &mut Mesh) {
    for face in &mut mesh.faces {
        if face.vertices.len() > 2 {
            face.vertices[1..].reverse();
        }
    }
}

fn reverse_corner_attributes(mesh: &mut Mesh) {
    let faces = mesh.faces.clone();
    let mut offsets = HashMap::new();
    let mut offset = 0;
    for face in &faces {
        offsets.insert(face.id, offset);
        offset += face.vertices.len();
    }
    if let Some(Value::Array(layers)) = mesh.attributes.get_mut("blender_uv_layers") {
        for layer in layers {
            if let Some(values) = layer.get_mut("values").and_then(Value::as_array_mut) {
                reverse_segments(values, &faces, &offsets);
            }
        }
    }
    if let Some(Value::Array(entries)) = mesh.attributes.get_mut("blender_attributes") {
        for entry in entries {
            if entry
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"))
                && let Some(values) = entry.get_mut("values").and_then(Value::as_array_mut)
            {
                reverse_segments(values, &faces, &offsets);
            }
        }
    }
    for (key, attribute) in &mut mesh.attributes {
        if key == "custom_normal" {
            for field in ["values", "corner_order"] {
                if let Some(Value::Object(rows)) = attribute.get_mut(field) {
                    reverse_corner_rows(rows, &faces);
                }
            }
        } else if attribute
            .get("domain")
            .and_then(Value::as_str)
            .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"))
            && let Some(Value::Object(rows)) = attribute.get_mut("values")
        {
            reverse_corner_rows(rows, &faces);
        }
    }
    if let Some(Value::Array(entries)) = mesh.attributes.get_mut("uv_map") {
        let face_sizes: HashMap<u32, usize> = faces
            .iter()
            .map(|face| (face.id, face.vertices.len()))
            .collect();
        for entry in entries {
            let id = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .and_then(|id| u32::try_from(id).ok());
            let Some(size) = id.and_then(|id| face_sizes.get(&id).copied()) else {
                continue;
            };
            if let Some(values) = entry.get_mut("uv").and_then(Value::as_array_mut)
                && values.len() == size
                && size > 1
            {
                values[1..].reverse();
            }
            if let Some(values) = entry.get_mut("pinned").and_then(Value::as_array_mut)
                && values.len() == size
                && size > 1
            {
                values[1..].reverse();
            }
        }
    }
    let mut uv = uv_layers::read(mesh);
    for layer in &mut uv.layers {
        for row in layer.faces.iter_mut().flatten() {
            if row.len() > 1 {
                row[1..].reverse();
            }
        }
        for row in layer.pinned.iter_mut().flatten() {
            if row.len() > 1 {
                row[1..].reverse();
            }
        }
    }
    uv_layers::write(mesh, &uv);
}

fn reverse_corner_rows(rows: &mut serde_json::Map<String, Value>, faces: &[crate::geom::Face]) {
    for face in faces {
        if let Some(Value::Array(row)) = rows.get_mut(&format!("f{}", face.id))
            && row.len() == face.vertices.len()
            && row.len() > 1
        {
            row[1..].reverse();
        }
    }
}

fn reverse_segments(
    values: &mut [Value],
    faces: &[crate::geom::Face],
    offsets: &HashMap<u32, usize>,
) {
    for face in faces {
        if face.vertices.len() < 2 {
            continue;
        }
        if let Some(&start) = offsets.get(&face.id) {
            let end = start + face.vertices.len();
            if end <= values.len() {
                values[start + 1..end].reverse();
            }
        }
    }
}

#[cfg(test)]
mod dangling_edge_tests {
    use super::*;

    fn unit_square_with_centre() -> State {
        let mut state = State::new();
        for co in [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ] {
            state.add_vertex(co, NO_INDEX);
        }
        state.faces.push(Face {
            verts: vec![0, 1, 2, 3],
            side: 0,
            source: 0,
        });
        state
    }

    #[test]
    fn spur_ending_inside_a_face_stays_loose() {
        // The square keeps its ring; the spur to the centre closes no loop, so Blender keeps it as wire.
        let mut state = unit_square_with_centre();
        state.face_cuts.insert(0, vec![edge_key_usize(0, 4)]);
        split_faces(&mut state);
        assert_eq!(state.faces.len(), 1);
        assert_eq!(state.dangling_edges, HashSet::from([edge_key_usize(0, 4)]));
    }

    #[test]
    fn chord_that_closes_two_rings_is_not_dangling() {
        // A diagonal splits the square into two loops, so no split-net edge is dangling.
        let mut state = unit_square_with_centre();
        state.face_cuts.insert(0, vec![edge_key_usize(0, 2)]);
        split_faces(&mut state);
        assert_eq!(state.faces.len(), 2);
        assert!(state.dangling_edges.is_empty());
    }
}
