// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/mesh_normals.cc, SPDX-FileCopyrightText: 2001-2002 NaN Holding BV. All rights reserved., GPL-2.0-or-later.

//! Custom normals across topology rebuilds, as Blender stores them.
//!
//! Blender keeps each custom normal per corner as two 16-bit codes relative to the corner's fan
//! space: an angle from the fan normal and an angle around it. A modifier that rebuilds loops
//! copies the codes, not the vectors, so a copied code is decoded against the space of the
//! corner's new fan. The evaluated normal of a fan is the decode of the mean code over its corners.
//! Ported from Blender 5.2.2 `blenkernel/intern/mesh_normals.cc`: `normals_calc_corners`,
//! `handle_fan_result_and_custom_normals`, `corner_fan_space_define`,
//! `corner_space_custom_normal_to_data`, and `corner_space_custom_data_to_normal`.

use std::collections::{HashMap, HashSet};
use std::f64::consts::TAU;

use glam::DVec3;
use serde_json::{Map, Value, json};

use super::normals::{face_normal, safe_acos_approx, sharp_edge_flags, smooth_face_flags, vector3};
use super::{Mesh, edge_key};
use crate::error::{ErrorCode, PotError, Result};

/// Cosine above which an angle counts as zero (`LNOR_SPACE_TRIGO_THRESHOLD`).
const TRIGO_THRESHOLD: f64 = 1.0 - 1.0e-4;
/// Codes are 16-bit values scaled by `SHRT_MAX`.
const CODE_SCALE: f64 = 32767.0;
/// A custom normal this close to the fan normal per component encodes as the zero code.
const SAME_TOLERANCE: f64 = 1.0e-4;

/// The space of one fan (`CornerNormalSpace`). The default is the invalid space: it decodes to the
/// zero vector.
#[derive(Clone, Copy, Debug, Default)]
struct Space {
    lnor: DVec3,
    vec_ref: DVec3,
    vec_ortho: DVec3,
    ref_alpha: f64,
    ref_beta: f64,
}

/// The angle of a cosine, clamped as Blender's `safe_acos_approx`.
fn angle(cosine: f64) -> f64 {
    safe_acos_approx(cosine)
}

impl Space {
    /// The space of a fan: its normal, the directions of its two boundary edges, and the direction
    /// of every edge of the fan when it has more than one corner (`corner_fan_space_define`).
    fn define(lnor: DVec3, vec_ref: DVec3, vec_other: DVec3, edges: &[DVec3]) -> Self {
        let dtp_ref = vec_ref.dot(lnor);
        let dtp_other = vec_other.dot(lnor);
        if dtp_ref.abs() >= TRIGO_THRESHOLD || dtp_other.abs() >= TRIGO_THRESHOLD {
            return Self::default();
        }
        let ref_alpha = if edges.is_empty() {
            f64::midpoint(angle(dtp_ref), angle(dtp_other))
        } else {
            edges.iter().map(|edge| angle(edge.dot(lnor))).sum::<f64>() / edges.len() as f64
        };
        let vec_ref = (vec_ref - lnor * dtp_ref).normalize_or_zero();
        let vec_ortho = lnor.cross(vec_ref).normalize_or_zero();
        let other = (vec_other - lnor * dtp_other).normalize_or_zero();
        let dtp = vec_ref.dot(other);
        let ref_beta = if dtp < TRIGO_THRESHOLD {
            let beta = angle(dtp);
            if vec_ortho.dot(other) < 0.0 {
                TAU - beta
            } else {
                beta
            }
        } else {
            TAU
        };
        Self {
            lnor,
            vec_ref,
            vec_ortho,
            ref_alpha,
            ref_beta,
        }
    }

    fn is_valid(&self) -> bool {
        self.ref_alpha != 0.0 && self.ref_beta != 0.0
    }

    /// The custom normal a code pair stands for (`corner_space_custom_data_to_normal`).
    fn decode(&self, codes: [i32; 2]) -> DVec3 {
        if codes[0] == 0 || !self.is_valid() {
            return self.lnor;
        }
        let alpha_factor = f64::from(codes[0]) / CODE_SCALE;
        let alpha_range = if alpha_factor > 0.0 {
            self.ref_alpha
        } else {
            TAU - self.ref_alpha
        };
        let alpha = alpha_range * alpha_factor;
        let (sin_alpha, cos_alpha) = alpha.sin_cos();
        let beta_factor = f64::from(codes[1]) / CODE_SCALE;
        if beta_factor == 0.0 {
            return self.lnor * cos_alpha + self.vec_ref * sin_alpha;
        }
        let beta_range = if beta_factor > 0.0 {
            self.ref_beta
        } else {
            TAU - self.ref_beta
        };
        let beta = beta_range * beta_factor;
        self.lnor * cos_alpha
            + self.vec_ref * (sin_alpha * beta.cos())
            + self.vec_ortho * (sin_alpha * beta.sin())
    }

    /// The code pair of a custom normal (`corner_space_custom_normal_to_data`). The zero pair
    /// stands for "no custom normal" and decodes to the fan normal.
    fn encode(&self, custom: DVec3) -> [i32; 2] {
        let same = (self.lnor - custom).abs().max_element() <= SAME_TOLERANCE;
        if custom == DVec3::ZERO || same || !self.is_valid() {
            return [0, 0];
        }
        let cos_alpha = self.lnor.dot(custom);
        let alpha = angle(cos_alpha);
        let alpha_code = if alpha > self.ref_alpha {
            code(-(TAU - alpha) / (TAU - self.ref_alpha))
        } else {
            code(alpha / self.ref_alpha)
        };
        let projected = (self.lnor * -cos_alpha + custom).normalize_or_zero();
        let cos_beta = self.vec_ref.dot(projected);
        let beta_code = if cos_beta < TRIGO_THRESHOLD {
            let mut beta = angle(cos_beta);
            if self.vec_ortho.dot(projected) < 0.0 {
                beta = TAU - beta;
            }
            if beta > self.ref_beta {
                code(-(TAU - beta) / (TAU - self.ref_beta))
            } else {
                code(beta / self.ref_beta)
            }
        } else {
            0
        };
        [alpha_code, beta_code]
    }
}

/// `unit_float_to_short`: a unit value rounded to its 16-bit code.
#[expect(
    clippy::cast_possible_truncation,
    reason = "unit values scale into the 16-bit code range"
)]
fn code(value: f64) -> i32 {
    (value * CODE_SCALE + 0.5).floor() as i32
}

/// The fans of one mesh. `corners[k]` lists the global corners of fan `k` in Blender's traversal
/// order and `spaces[k]` is its space. `fan_of[c]` is the fan of global corner `c`.
struct Fans {
    corners: Vec<Vec<usize>>,
    spaces: Vec<Space>,
    fan_of: Vec<usize>,
}

/// The link of one local edge around a vertex (`EdgeInfo`).
#[derive(Clone, Copy, PartialEq)]
enum Link {
    Unset,
    Sharp,
    One { corner: usize, towards: bool },
    Two { first: usize, second: usize },
}

/// A corner of a face around one vertex, with the local indices of its previous and next edges.
struct FanCorner {
    face: usize,
    corner: usize,
    vert_prev: usize,
    vert_next: usize,
    edge_prev: usize,
    edge_next: usize,
}

/// Fans and spaces of a mesh given by vertex positions, faces (vertex indices), smooth flags, and
/// sharp edges (vertex-index pairs). Corners join a fan through an edge that is not sharp, whose
/// faces are both smooth, and whose faces wind the same way.
fn fans(
    points: &[DVec3],
    faces: &[Vec<usize>],
    smooth: &[bool],
    sharp: &HashSet<(usize, usize)>,
) -> Fans {
    let mut offsets = Vec::with_capacity(faces.len());
    let mut total = 0;
    for face in faces {
        offsets.push(total);
        total += face.len();
    }
    let face_normals: Vec<DVec3> = faces
        .iter()
        .map(|face| {
            let corners: Vec<DVec3> = face.iter().map(|&vertex| points[vertex]).collect();
            face_normal(&corners)
        })
        .collect();
    let mut around: Vec<Vec<(usize, usize)>> = vec![Vec::new(); points.len()];
    for (face_index, face) in faces.iter().enumerate() {
        for (corner, &vertex) in face.iter().enumerate() {
            around[vertex].push((face_index, corner));
        }
    }
    let mut result = Fans {
        corners: Vec::new(),
        spaces: Vec::new(),
        fan_of: vec![0; total],
    };
    for (vertex, uses) in around.iter().enumerate() {
        if uses.is_empty() {
            continue;
        }
        let mut local_vertices: Vec<usize> = Vec::new();
        let corners: Vec<FanCorner> = uses
            .iter()
            .map(|&(face, corner)| {
                let face_vertices = &faces[face];
                let count = face_vertices.len();
                let vert_prev = face_vertices[(corner + count - 1) % count];
                let vert_next = face_vertices[(corner + 1) % count];
                FanCorner {
                    face,
                    corner: offsets[face] + corner,
                    vert_prev,
                    vert_next,
                    edge_prev: local_index(&mut local_vertices, vert_prev),
                    edge_next: local_index(&mut local_vertices, vert_next),
                }
            })
            .collect();
        let origin = points[vertex];
        let directions: Vec<DVec3> = local_vertices
            .iter()
            .map(|&other| (points[other] - origin).normalize_or_zero())
            .collect();
        let links = edge_links(&corners, local_vertices.len(), vertex, smooth, sharp);
        let mut visited = vec![false; corners.len()];
        for start in 0..corners.len() {
            if visited[start] {
                continue;
            }
            let fan = traverse(&corners, &links, start);
            for &local in &fan {
                visited[local] = true;
            }
            let normal = fan_normal(&fan, &corners, &directions, &face_normals);
            let first = &corners[fan[0]];
            let last = &corners[fan[fan.len() - 1]];
            let mut fan_edges = Vec::new();
            if fan.len() > 1 {
                fan_edges.extend(
                    fan.iter()
                        .map(|&local| directions[corners[local].edge_next]),
                );
                if last.edge_prev != first.edge_next {
                    fan_edges.push(directions[last.edge_prev]);
                }
            }
            let space = Space::define(
                normal,
                directions[first.edge_next],
                directions[last.edge_prev],
                &fan_edges,
            );
            let index = result.spaces.len();
            let globals: Vec<usize> = fan.iter().map(|&local| corners[local].corner).collect();
            for &global in &globals {
                result.fan_of[global] = index;
            }
            result.corners.push(globals);
            result.spaces.push(space);
        }
    }
    result
}

/// The local index of `vertex` among the edges of one vertex, numbering new edges in order.
fn local_index(vertices: &mut Vec<usize>, vertex: usize) -> usize {
    if let Some(index) = vertices.iter().position(|&known| known == vertex) {
        return index;
    }
    vertices.push(vertex);
    vertices.len() - 1
}

/// Link each local edge around `vertex` to the corners using it. An edge is sharp when the mesh
/// marks it sharp, when a flat face uses it, or when more than two corners share it.
fn edge_links(
    corners: &[FanCorner],
    local_count: usize,
    vertex: usize,
    smooth: &[bool],
    sharp: &HashSet<(usize, usize)>,
) -> Vec<Link> {
    let mut links = vec![Link::Unset; local_count];
    for (local, corner) in corners.iter().enumerate() {
        if !smooth[corner.face] {
            links[corner.edge_prev] = Link::Sharp;
            links[corner.edge_next] = Link::Sharp;
            continue;
        }
        let prev_sharp = sharp.contains(&edge_key(corner.vert_prev, vertex));
        let next_sharp = sharp.contains(&edge_key(vertex, corner.vert_next));
        connect(&mut links, corner.edge_prev, local, true, prev_sharp);
        connect(&mut links, corner.edge_next, local, false, next_sharp);
    }
    links
}

/// Add a corner to a local edge. The first corner of an edge records its winding; a second corner
/// with the opposite winding joins the edge, and any other use makes it sharp.
fn connect(links: &mut [Link], edge: usize, corner: usize, towards: bool, sharp: bool) {
    links[edge] = match links[edge] {
        Link::Unset if sharp => Link::Sharp,
        Link::Unset => Link::One { corner, towards },
        Link::One {
            corner: first,
            towards: first_towards,
        } if first_towards != towards => Link::Two {
            first,
            second: corner,
        },
        _ => Link::Sharp,
    };
}

/// The corners around one vertex that form a fan with `start`, in Blender's order: forward along
/// next edges, reversed, then backward along previous edges. A closed fan is rotated to begin with
/// its lowest global corner.
fn traverse(corners: &[FanCorner], links: &[Link], start: usize) -> Vec<usize> {
    let mut fan = vec![start];
    let mut current = start;
    let mut cyclic = false;
    while let Link::Two { first, second } = links[corners[current].edge_next] {
        let other = if current == first { second } else { first };
        if other == start {
            cyclic = true;
            break;
        }
        fan.push(other);
        current = other;
    }
    fan.reverse();
    if cyclic {
        let lowest = (0..fan.len())
            .min_by_key(|&index| corners[fan[index]].corner)
            .unwrap_or(0);
        fan.rotate_left(lowest);
        return fan;
    }
    current = start;
    while let Link::Two { first, second } = links[corners[current].edge_prev] {
        let other = if current == first { second } else { first };
        fan.push(other);
        current = other;
    }
    fan
}

/// The fan normal: the face normals weighted by each corner's angle (`accumulate_fan_normal`).
fn fan_normal(
    fan: &[usize],
    corners: &[FanCorner],
    directions: &[DVec3],
    face_normals: &[DVec3],
) -> DVec3 {
    if let [only] = fan {
        return face_normals[corners[*only].face];
    }
    let sum: DVec3 = fan
        .iter()
        .map(|&local| {
            let corner = &corners[local];
            let weight = angle(directions[corner.edge_prev].dot(directions[corner.edge_next]));
            face_normals[corner.face] * weight
        })
        .sum();
    sum.normalize_or_zero()
}

type Topology = (Vec<DVec3>, HashMap<u32, usize>, HashSet<(usize, usize)>);

/// Vertex positions, the vertex-index lookup by ID, and the sharp edges as vertex-index pairs.
fn topology(mesh: &Mesh) -> Result<Topology> {
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut sharp = HashSet::new();
    for (edge, flag) in mesh.edges.iter().zip(sharp_edge_flags(mesh)?) {
        if flag {
            let first = index_by_id[&edge.vertices[0]];
            let second = index_by_id[&edge.vertices[1]];
            sharp.insert(edge_key(first, second));
        }
    }
    let points = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    Ok((points, index_by_id, sharp))
}

/// Fans of `mesh`, with its faces' vertex indices and smooth flags.
fn mesh_fans(mesh: &Mesh) -> Result<Fans> {
    let (points, index_by_id, sharp) = topology(mesh)?;
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    index_by_id.get(id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "custom normal transfer met a face with a missing vertex",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(fans(&points, &faces, &smooth_face_flags(mesh)?, &sharp))
}

/// The authored custom normal of every corner of `mesh`, in `mesh.faces` order. A corner without
/// an authored vector is the zero vector. `None` when the mesh has no `custom_normal` attribute.
fn authored_vectors(mesh: &Mesh) -> Option<Vec<DVec3>> {
    let values = mesh
        .attributes
        .get("custom_normal")?
        .get("values")?
        .as_object()?;
    Some(
        mesh.faces
            .iter()
            .flat_map(|face| {
                let authored = values
                    .get(&format!("f{}", face.id))
                    .and_then(Value::as_array);
                (0..face.vertices.len()).map(move |corner| {
                    authored
                        .and_then(|corners| corners.get(corner))
                        .and_then(vector3)
                        .map_or(DVec3::ZERO, DVec3::normalize_or_zero)
                })
            })
            .collect(),
    )
}

/// The `custom_normal` attribute of `output`, rebuilt from `source` the way Blender rebuilds it
/// after a topology change. `sources[c]` is the global source corner (in `source.faces` order)
/// whose code output corner `c` copies, or `None` for a corner with no source. Returns `None` when
/// `source` has no custom normals.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `sources` does not describe every output corner, or a source
/// corner is out of range.
pub(crate) fn rebuild_custom_normals(
    source: &Mesh,
    output: &Mesh,
    sources: &[Option<usize>],
) -> Result<Option<Value>> {
    let Some(authored) = authored_vectors(source) else {
        return Ok(None);
    };
    let output_corners: usize = output.faces.iter().map(|face| face.vertices.len()).sum();
    if sources.len() != output_corners || sources.iter().flatten().any(|&c| c >= authored.len()) {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "custom normal transfer does not describe the rebuilt corners",
        ));
    }

    let source_fans = mesh_fans(source)?;
    let codes: Vec<[i32; 2]> = authored
        .iter()
        .enumerate()
        .map(|(corner, &vector)| source_fans.spaces[source_fans.fan_of[corner]].encode(vector))
        .collect();

    let output_fans = mesh_fans(output)?;
    let mut vectors = vec![DVec3::ZERO; output_corners];
    for (fan, corners) in output_fans.corners.iter().enumerate() {
        // Blender sums the codes of a fan's corners and divides with integer truncation.
        let mut sum = [0_i32; 2];
        for &corner in corners {
            let pair = sources[corner].map_or([0, 0], |source| codes[source]);
            sum[0] += pair[0];
            sum[1] += pair[1];
        }
        let count = i32::try_from(corners.len()).unwrap_or(i32::MAX);
        let normal = output_fans.spaces[fan].decode([sum[0] / count, sum[1] / count]);
        for &corner in corners {
            vectors[corner] = normal;
        }
    }

    let mut values = Map::new();
    let mut corner = 0;
    for face in &output.faces {
        let count = face.vertices.len();
        let normals: Vec<Value> = vectors[corner..corner + count]
            .iter()
            .map(|normal| json!(normal.to_array()))
            .collect();
        values.insert(format!("f{}", face.id), Value::Array(normals));
        corner += count;
    }
    let mut entry = source
        .attributes
        .get("custom_normal")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    entry.insert("values".to_owned(), Value::Object(values));
    Ok(Some(Value::Object(entry)))
}
