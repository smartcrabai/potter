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
use std::f32::consts::{PI, TAU};

use glam::{DVec3, Vec3};
use serde_json::{Map, Value, json};

use super::normals::{corner_normals, sharp_edge_flags, smooth_face_flags, vector3};
use super::{Mesh, edge_key};
use crate::error::{ErrorCode, PotError, Result};

/// Cosine above which an angle counts as zero (`LNOR_SPACE_TRIGO_THRESHOLD`, single precision).
const TRIGO_THRESHOLD: f32 = 1.0 - 1.0e-4;
/// Codes are 16-bit values scaled by `SHRT_MAX`.
const CODE_SCALE: f32 = 32767.0;
/// A custom normal this close to the fan normal per component encodes as the zero code.
const SAME_TOLERANCE: f32 = 1.0e-4;

/// The space of one fan (`CornerNormalSpace`). The default is the invalid space: it decodes to the
/// zero vector.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Space {
    lnor: Vec3,
    vec_ref: Vec3,
    vec_ortho: Vec3,
    ref_alpha: f32,
    ref_beta: f32,
}

/// `dot_v3v3` and `math::dot`: the products summed in component order.
fn dot(a: Vec3, b: Vec3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// `math::cross`.
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

/// `math::normalize`: divides by the length. A vector too short to have a direction is zero.
pub(crate) fn normalize(v: Vec3) -> Vec3 {
    let length_squared = dot(v, v);
    if length_squared > 1.0e-35 {
        let length = length_squared.sqrt();
        Vec3::new(v.x / length, v.y / length, v.z / length)
    } else {
        Vec3::ZERO
    }
}

/// `normalize_v3`: multiplies by the reciprocal of the length, as Blender's C vector helpers do.
fn normalize_reciprocal(v: Vec3) -> Vec3 {
    let length_squared = dot(v, v);
    if length_squared > 1.0e-35 {
        let scale = 1.0 / length_squared.sqrt();
        Vec3::new(v.x * scale, v.y * scale, v.z * scale)
    } else {
        Vec3::ZERO
    }
}

/// Blender's single-precision `safe_acos_approx`.
fn safe_acos(x: f32) -> f32 {
    let magnitude = x.abs();
    let m = if magnitude < 1.0 {
        1.0 - (1.0 - magnitude)
    } else {
        1.0
    };
    let acos = (1.0 - m).sqrt()
        * (1.5707963267 + m * (-0.213300989 + m * (0.077980478 + m * -0.02164095)));
    if x < 0.0 { PI - acos } else { acos }
}

/// `face_normal_calc`: the unit normal of a face's corners, computed in single precision as Blender
/// does. Quads cross their diagonals, triangles cross two edges, and n-gons use Newell's method from
/// the last corner. A degenerate face points along +Z.
fn face_normal(points: &[Vec3], face: &[usize]) -> Vec3 {
    let normal = match face {
        [i0, i1, i2, i3] => {
            normalize_reciprocal(cross(points[*i0] - points[*i2], points[*i1] - points[*i3]))
        }
        [i0, i1, i2] => normalize(cross(points[*i0] - points[*i1], points[*i1] - points[*i2])),
        _ => {
            let mut normal = Vec3::ZERO;
            let mut previous = points[face[face.len() - 1]];
            for &index in face {
                let current = points[index];
                normal.x += (previous.y - current.y) * (previous.z + current.z);
                normal.y += (previous.z - current.z) * (previous.x + current.x);
                normal.z += (previous.x - current.x) * (previous.y + current.y);
                previous = current;
            }
            normalize_reciprocal(normal)
        }
    };
    if normal == Vec3::ZERO { Vec3::Z } else { normal }
}

impl Space {
    /// The space of a fan: its normal, the directions of its two boundary edges, and the direction
    /// of every edge of the fan when it has more than one corner (`corner_fan_space_define`).
    fn define(lnor: Vec3, vec_ref: Vec3, vec_other: Vec3, edges: &[Vec3]) -> Self {
        let dtp_ref = dot(vec_ref, lnor);
        let dtp_other = dot(vec_other, lnor);
        if dtp_ref.abs() >= TRIGO_THRESHOLD || dtp_other.abs() >= TRIGO_THRESHOLD {
            return Self::default();
        }
        let ref_alpha = if edges.is_empty() {
            (safe_acos(dtp_ref) + safe_acos(dtp_other)) / 2.0
        } else {
            let mut alpha = 0.0;
            for &edge in edges {
                alpha += safe_acos(dot(edge, lnor));
            }
            alpha / edges.len() as f32
        };
        let vec_ref = normalize(vec_ref - lnor * dtp_ref);
        let vec_ortho = normalize(cross(lnor, vec_ref));
        let other = normalize(vec_other - lnor * dtp_other);
        let dtp = dot(vec_ref, other);
        let ref_beta = if dtp < TRIGO_THRESHOLD {
            let beta = safe_acos(dtp);
            if dot(vec_ortho, other) < 0.0 {
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
    pub(crate) fn decode(&self, codes: [i32; 2]) -> Vec3 {
        if codes[0] == 0 || !self.is_valid() {
            return self.lnor;
        }
        let alpha_factor = codes[0] as f32 / CODE_SCALE;
        let alpha_range = if alpha_factor > 0.0 {
            self.ref_alpha
        } else {
            TAU - self.ref_alpha
        };
        let alpha = alpha_range * alpha_factor;
        let sin_alpha = alpha.sin();
        let mut custom = self.lnor * alpha.cos();
        let beta_factor = codes[1] as f32 / CODE_SCALE;
        if beta_factor == 0.0 {
            custom += self.vec_ref * sin_alpha;
            return custom;
        }
        let beta_range = if beta_factor > 0.0 {
            self.ref_beta
        } else {
            TAU - self.ref_beta
        };
        let beta = beta_range * beta_factor;
        custom += self.vec_ref * (sin_alpha * beta.cos());
        custom += self.vec_ortho * (sin_alpha * beta.sin());
        custom
    }

    /// The code pair of a custom normal (`corner_space_custom_normal_to_data`). The zero pair
    /// stands for "no custom normal" and decodes to the fan normal.
    pub(crate) fn encode(&self, custom: Vec3) -> [i32; 2] {
        let same = (self.lnor.x - custom.x).abs() <= SAME_TOLERANCE
            && (self.lnor.y - custom.y).abs() <= SAME_TOLERANCE
            && (self.lnor.z - custom.z).abs() <= SAME_TOLERANCE;
        if custom == Vec3::ZERO || same || !self.is_valid() {
            return [0, 0];
        }
        let cos_alpha = dot(self.lnor, custom);
        let alpha = safe_acos(cos_alpha);
        let alpha_code = if alpha > self.ref_alpha {
            code(-(TAU - alpha) / (TAU - self.ref_alpha))
        } else {
            code(alpha / self.ref_alpha)
        };
        let projected = normalize(self.lnor * -cos_alpha + custom);
        let cos_beta = dot(self.vec_ref, projected);
        let beta_code = if cos_beta < TRIGO_THRESHOLD {
            let mut beta = safe_acos(cos_beta);
            if dot(self.vec_ortho, projected) < 0.0 {
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
fn code(value: f32) -> i32 {
    (value * CODE_SCALE + 0.5).floor() as i32
}

/// The fans of one mesh. `corners[k]` lists the global corners of fan `k` in Blender's traversal
/// order and `spaces[k]` is its space. `fan_of[c]` is the fan of global corner `c`.
pub(crate) struct Fans {
    pub(crate) corners: Vec<Vec<usize>>,
    pub(crate) spaces: Vec<Space>,
    pub(crate) fan_of: Vec<usize>,
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
/// faces are both smooth, and whose faces wind the same way. `numbering[c]` is the number Blender
/// gives global corner `c`; a closed fan begins with its lowest-numbered corner.
pub(crate) fn fans(
    points: &[DVec3],
    faces: &[Vec<usize>],
    smooth: &[bool],
    sharp: &HashSet<(usize, usize)>,
    numbering: &[usize],
) -> Fans {
    let mut offsets = Vec::with_capacity(faces.len());
    let mut total = 0;
    for face in faces {
        offsets.push(total);
        total += face.len();
    }
    let points: Vec<Vec3> = points.iter().map(|point| point.as_vec3()).collect();
    let face_normals: Vec<Vec3> = faces
        .iter()
        .map(|face| face_normal(&points, face))
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
        let directions: Vec<Vec3> = local_vertices
            .iter()
            .map(|&other| normalize(points[other] - origin))
            .collect();
        let links = edge_links(&corners, local_vertices.len(), vertex, smooth, sharp);
        let mut visited = vec![false; corners.len()];
        for start in 0..corners.len() {
            if visited[start] {
                continue;
            }
            let fan = traverse(&corners, &links, start, numbering);
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
/// its lowest-numbered corner.
fn traverse(corners: &[FanCorner], links: &[Link], start: usize, numbering: &[usize]) -> Vec<usize> {
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
            .min_by_key(|&index| numbering[corners[fan[index]].corner])
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
    directions: &[Vec3],
    face_normals: &[Vec3],
) -> Vec3 {
    if let [only] = fan {
        return face_normals[corners[*only].face];
    }
    let mut sum = Vec3::ZERO;
    for &local in fan {
        let corner = &corners[local];
        let factor = safe_acos(dot(directions[corner.edge_prev], directions[corner.edge_next]));
        sum += face_normals[corner.face] * factor;
    }
    normalize(sum)
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

/// Fans of `mesh`, with its faces' vertex indices and smooth flags. `numbering` is as in [`fans`].
fn mesh_fans(mesh: &Mesh, numbering: &[usize]) -> Result<Fans> {
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
    Ok(fans(&points, &faces, &smooth_face_flags(mesh)?, &sharp, numbering))
}

/// The corner numbering of `mesh` as it is stored: global corner `c` is numbered `c`.
pub(crate) fn stored_corner_order(mesh: &Mesh) -> Vec<usize> {
    (0..mesh.faces.iter().map(|face| face.vertices.len()).sum()).collect()
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
                        .map_or(DVec3::ZERO, |vector| vector.as_vec3().as_dvec3())
                })
            })
            .collect(),
    )
}

/// Normalizes caller-supplied custom normals with Blender's single-precision math path.
fn normalize_authored(vectors: &[DVec3]) -> Vec<DVec3> {
    vectors
        .iter()
        .map(|vector| normalize(vector.as_vec3()).as_dvec3())
        .collect()
}

/// The corner normals of `mesh` without its custom normals, at `positions`.
fn plain_corner_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    let mut plain = mesh.clone();
    plain.attributes.remove("custom_normal");
    corner_normals(&plain, positions)
}

/// `authored` where a corner has a custom normal, `computed` elsewhere.
fn fill_custom(authored: &[DVec3], computed: &[DVec3]) -> Vec<DVec3> {
    authored
        .iter()
        .zip(computed)
        .map(|(custom, normal)| {
            if *custom == DVec3::ZERO {
                *normal
            } else {
                *custom
            }
        })
        .collect()
}

/// `mesh` with its `custom_normal` set to `vectors`, one per corner.
fn with_custom_normals(mesh: &Mesh, vectors: &[DVec3]) -> Mesh {
    let mut settled = mesh.clone();
    if let Some(value) = corner_normal_attribute(mesh, vectors, &vec![true; mesh.faces.len()]) {
        settled.attributes.insert("custom_normal".to_owned(), value);
    }
    settled
}

/// Blender's `mesh_normals_corner_custom_set` applied to a mesh with custom normals before they are
/// read: a corner without a custom normal takes its computed corner normal, and a fan whose corners
/// disagree is cut at the edge between them, which becomes sharp. The corners without a custom
/// normal then take the computed normals of the cut mesh. Returns `None` when the mesh has no
/// `custom_normal` attribute.
///
/// # Errors
///
/// Returns the errors of [`corner_normals`] when the mesh is invalid.
pub(crate) fn settle_custom_normals(mesh: &Mesh, numbering: &[usize]) -> Result<Option<Mesh>> {
    settle_custom_normals_inner(mesh, numbering, true)
}

fn settle_custom_normals_inner(
    mesh: &Mesh,
    numbering: &[usize],
    normalize_input: bool,
) -> Result<Option<Mesh>> {
    let Some(authored) = authored_vectors(mesh) else {
        return Ok(None);
    };
    let authored = if normalize_input {
        normalize_authored(&authored)
    } else {
        authored
    };
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let filled = fill_custom(&authored, &plain_corner_normals(mesh, &positions)?);

    let index_of: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let fans = mesh_fans(mesh, numbering)?;
    let mut sharp = sharp_edge_flags(mesh)?;
    let edge_index: HashMap<(usize, usize), usize> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            (
                edge_key(index_of[&edge.vertices[0]], index_of[&edge.vertices[1]]),
                index,
            )
        })
        .collect();
    // Each corner's outgoing and incoming edge, as vertex-index pairs.
    let mut outgoing = Vec::with_capacity(filled.len());
    let mut incoming = Vec::with_capacity(filled.len());
    for face in &mesh.faces {
        let count = face.vertices.len();
        let vertices: Vec<usize> = face.vertices.iter().map(|id| index_of[id]).collect();
        for corner in 0..count {
            outgoing.push(edge_key(vertices[corner], vertices[(corner + 1) % count]));
            incoming.push(edge_key(vertices[(corner + count - 1) % count], vertices[corner]));
        }
    }
    // Blender's `mesh_normals_corner_custom_set` walks each fan from its last corner to its first. A
    // corner whose normal differs from the fan's reference (the normal of the last corner that
    // differed) sharpens the edge it shares with the corner before it and becomes the reference. The
    // pair closing the fan is checked last. Sharpening never clears an edge.
    let mut marked = false;
    let mut sharpen = |previous: usize, corner: usize| {
        let shared = if outgoing[previous] == incoming[corner] {
            outgoing[previous]
        } else {
            outgoing[corner]
        };
        if let Some(&index) = edge_index.get(&shared) {
            sharp[index] = true;
            marked = true;
        }
    };
    for corners in &fans.corners {
        let mut reversed = corners.iter().rev();
        let Some(&first) = reversed.next() else {
            continue;
        };
        let mut reference = filled[first].as_vec3();
        let mut previous = first;
        for &corner in reversed {
            if dot(reference, filled[corner].as_vec3()) < TRIGO_THRESHOLD {
                sharpen(previous, corner);
                reference = filled[corner].as_vec3();
            }
            previous = corner;
        }
        if corners.len() > 1 && dot(reference, filled[first].as_vec3()) < TRIGO_THRESHOLD {
            sharpen(previous, first);
        }
    }
    if !marked {
        return Ok(Some(with_custom_normals(mesh, &filled)));
    }

    let mut cut = mesh.clone();
    let values: Map<String, Value> = mesh
        .edges
        .iter()
        .zip(&sharp)
        .map(|(edge, flag)| (format!("e{}", edge.id), json!(flag)))
        .collect();
    cut.attributes.insert(
        "sharp".to_owned(),
        json!({"domain": "edge", "type": "bool", "values": values}),
    );
    let cut_computed = plain_corner_normals(&cut, &positions)?;
    Ok(Some(with_custom_normals(
        &cut,
        &fill_custom(&authored, &cut_computed),
    )))
}

/// The `custom_normal` attribute of `output` holding `vectors`, one per output corner in face order.
/// A face whose `keep` flag is false has no row, so its corners carry no custom normal. `None` when
/// no face is kept.
pub(crate) fn corner_normal_attribute(
    output: &Mesh,
    vectors: &[DVec3],
    keep: &[bool],
) -> Option<Value> {
    let mut values = Map::new();
    let mut corner = 0;
    for (face, kept) in output.faces.iter().zip(keep) {
        let end = corner + face.vertices.len();
        if *kept {
            let row: Vec<Value> = vectors[corner..end]
                .iter()
                .map(|normal| json!(normal.to_array()))
                .collect();
            values.insert(format!("f{}", face.id), Value::Array(row));
        }
        corner = end;
    }
    (!values.is_empty()).then(|| json!({"domain": "corner", "type": "float3", "values": values}))
}

/// The `custom_normal` attribute of `output`, rebuilt from `source` the way Blender rebuilds it after
/// a topology change. `source` must be settled ([`settle_custom_normals`]). `sources[c]` is the
/// global source corner (in `source.faces` order) whose code output corner `c` copies, or `None` for
/// a corner with no source. A fan whose codes average to zero has no custom normal: its corners
/// decode to the fan normal, and a face with no such fan keeps no row. Returns `None` when `source`
/// has no custom normals or no output fan keeps one.
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
    let authored = normalize_authored(&authored);
    let output_corners: usize = output.faces.iter().map(|face| face.vertices.len()).sum();
    if sources.len() != output_corners || sources.iter().flatten().any(|&c| c >= authored.len()) {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "custom normal transfer does not describe the rebuilt corners",
        ));
    }

    let source_fans = mesh_fans(source, &stored_corner_order(source))?;
    let codes: Vec<[i32; 2]> = authored
        .iter()
        .enumerate()
        .map(|(corner, &vector)| {
            source_fans.spaces[source_fans.fan_of[corner]].encode(vector.as_vec3())
        })
        .collect();

    let output_fans = mesh_fans(output, &stored_corner_order(output))?;
    let mut vectors = vec![DVec3::ZERO; output_corners];
    let mut fan_keeps = vec![false; output_fans.corners.len()];
    for (fan, corners) in output_fans.corners.iter().enumerate() {
        // Blender sums the codes of a fan's corners and divides with integer truncation.
        let mut sum = [0_i32; 2];
        for &corner in corners {
            let pair = sources[corner].map_or([0, 0], |source| codes[source]);
            sum[0] += pair[0];
            sum[1] += pair[1];
        }
        let count = i32::try_from(corners.len()).unwrap_or(i32::MAX);
        let average = [sum[0] / count, sum[1] / count];
        fan_keeps[fan] = average != [0, 0];
        let normal = output_fans.spaces[fan].decode(average).as_dvec3();
        for &corner in corners {
            vectors[corner] = normal;
        }
    }

    let mut keep = Vec::with_capacity(output.faces.len());
    let mut corner = 0;
    for face in &output.faces {
        let end = corner + face.vertices.len();
        keep.push((corner..end).any(|c| fan_keeps[output_fans.fan_of[c]]));
        corner = end;
    }
    Ok(corner_normal_attribute(output, &vectors, &keep))
}

/// Sets `output`'s `custom_normal` from `source` as [`rebuild_custom_normals`] gives it, replacing
/// whatever the provenance copied in: a corner with no fan custom normal is not stored.
///
/// # Errors
///
/// Returns the errors of [`rebuild_custom_normals`].
pub(crate) fn transfer_custom_normals(
    source: &Mesh,
    output: &mut Mesh,
    sources: &[Option<usize>],
) -> Result<()> {
    output.attributes.remove("custom_normal");
    if let Some(value) = rebuild_custom_normals(source, output, sources)? {
        output.attributes.insert("custom_normal".to_owned(), value);
    }
    Ok(())
}

/// Stores `vectors`, one per corner in face order, as `mesh`'s `custom_normal` layer.
pub(crate) fn store_custom_normals(mesh: &mut Mesh, vectors: &[DVec3]) {
    let keep = vec![true; mesh.faces.len()];
    match corner_normal_attribute(mesh, vectors, &keep) {
        Some(value) => {
            mesh.attributes.insert("custom_normal".to_owned(), value);
        }
        None => {
            mesh.attributes.remove("custom_normal");
        }
    }
}

/// The normals Blender reads back when every fan of `mesh` stores the code of its mean vector. The
/// code is encoded in the fan's space and decoded again, so every corner of a fan reads the same
/// vector (`mesh_normals_corner_custom_set`, then `Mesh::corner_normals`).
fn fan_custom_normals(mesh: &Mesh, vectors: &[DVec3], numbering: &[usize]) -> Result<Vec<DVec3>> {
    let fans = mesh_fans(mesh, numbering)?;
    let mut read = vec![DVec3::ZERO; vectors.len()];
    for (fan, corners) in fans.corners.iter().enumerate() {
        let space = &fans.spaces[fan];
        let mut sum = Vec3::ZERO;
        for &corner in corners {
            sum += vectors[corner].as_vec3();
        }
        // Blender scales the sum by the reciprocal of the count (`mul_v3_fl`).
        let average = sum * (1.0 / corners.len() as f32);
        let normal = space.decode(space.encode(average)).as_dvec3();
        for &corner in corners {
            read[corner] = normal;
        }
    }
    Ok(read)
}

/// The custom normals `mesh` reads back from its own `custom_normal` layer. `None` without one.
///
/// # Errors
///
/// Returns the errors of [`mesh_fans`] when the mesh is invalid.
pub(crate) fn read_custom_normals(mesh: &Mesh) -> Result<Option<Vec<DVec3>>> {
    let Some(authored) = authored_vectors(mesh) else {
        return Ok(None);
    };
    let authored = normalize_authored(&authored);
    fan_custom_normals(mesh, &authored, &stored_corner_order(mesh)).map(Some)
}

/// Blender's `mesh_set_custom_normals_normalized`: stores `vectors` as the corners' custom normals.
/// Fans whose vectors diverge get their smooth edges sharpened (`settle_custom_normals`), and every
/// corner then holds the normal its fan reads back. `numbering` gives the corners' Blender numbers,
/// which decide where a closed fan begins.
///
/// # Errors
///
/// Returns the errors of [`settle_custom_normals`] and [`fan_custom_normals`].
pub(crate) fn set_custom_normals(
    mesh: &mut Mesh,
    vectors: &[DVec3],
    numbering: &[usize],
) -> Result<()> {
    store_custom_normals(mesh, vectors);
    if let Some(settled) = settle_custom_normals_inner(mesh, numbering, false)? {
        *mesh = settled;
    }
    let read = fan_custom_normals(mesh, vectors, numbering)?;
    store_custom_normals(mesh, &read);
    Ok(())
}
