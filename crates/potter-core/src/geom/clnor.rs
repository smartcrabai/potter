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

use super::normals::{plain_corner_normals, sharp_edge_flags, smooth_face_flags};
use super::{Mesh, edge_key};
use crate::error::{ErrorCode, PotError, Result};

/// Cosine above which an angle counts as zero (`LNOR_SPACE_TRIGO_THRESHOLD`, single precision).
const TRIGO_THRESHOLD: f32 = 1.0 - 1.0e-4;
/// Codes are 16-bit values scaled by `SHRT_MAX`.
const CODE_SCALE: f32 = 32767.0;
/// A custom normal this close to the fan normal per component encodes as the zero code.
const SAME_TOLERANCE: f32 = 1.0e-4;

/// The space of one fan (`CornerNormalSpace`). The default is the invalid space: it decodes to the
/// fan normal.
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
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// `math::normalize`: divides by the length. A vector too short to have a direction is zero.
pub(crate) fn normalize(vector: Vec3) -> Vec3 {
    let length_squared = dot(vector, vector);
    if length_squared > 1.0e-35 {
        let length = length_squared.sqrt();
        Vec3::new(vector.x / length, vector.y / length, vector.z / length)
    } else {
        Vec3::ZERO
    }
}

/// `normalize_v3`: multiplies by the reciprocal of the length, as Blender's C vector helpers do.
fn normalize_reciprocal(vector: Vec3) -> Vec3 {
    let length_squared = dot(vector, vector);
    if length_squared > 1.0e-35 {
        let scale = 1.0 / length_squared.sqrt();
        Vec3::new(vector.x * scale, vector.y * scale, vector.z * scale)
    } else {
        Vec3::ZERO
    }
}

/// Blender's single-precision `safe_acos_approx`.
fn safe_acos(cosine: f32) -> f32 {
    let magnitude = cosine.abs();
    let clamped = if magnitude < 1.0 {
        1.0 - (1.0 - magnitude)
    } else {
        1.0
    };
    let acos = (1.0 - clamped).sqrt()
        * (1.5707963267
            + clamped * (-0.213300989 + clamped * (0.077980478 + clamped * -0.02164095)));
    if cosine < 0.0 { PI - acos } else { acos }
}

/// `face_normal_calc`: Blender's single-precision quad, triangle, or Newell face normal.
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
    if normal == Vec3::ZERO {
        Vec3::Z
    } else {
        normal
    }
}

/// The semantic authored/automatic state stored in Blender's two short loop codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CornerNormalCode {
    /// Code `(0, 0)`: the corner takes its evaluated fan.
    Auto,
    /// A nonzero Blender fan-space code pair.
    Authored([i16; 2]),
    /// A literal zero vector. It is never reinterpreted as `Auto`.
    ExplicitZero,
}

impl CornerNormalCode {
    /// The code of a stored Blender pair, where `[0, 0]` is the automatic code.
    pub(crate) fn from_pair(pair: [i16; 2]) -> Self {
        if pair == [0, 0] {
            Self::Auto
        } else {
            Self::Authored(pair)
        }
    }
}

/// Which Blender custom-normal setter boundary supplied a vector buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SetterInput {
    /// `mesh.set_custom_normals`: input vectors are normalized first.
    #[default]
    NormalizeInput,
    /// Subdivision's setter: input vectors are already unit length.
    AlreadyNormalized,
}

/// Per-call basis and setter mode. `corner_order` gives each corner's Blender loop ordinal; `None`
/// uses the order persisted with the mesh, or the flattened mesh order when none is persisted.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NormalCodeOptions<'a> {
    pub corner_order: Option<&'a [usize]>,
    pub setter_input: SetterInput,
}

/// The space of a fan: its normal, boundary directions, and its edge directions.
impl Space {
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

    /// The code pair of a custom normal (`corner_space_custom_normal_to_data`).
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
    reason = "unit values scale into Blender's signed-short code range"
)]
fn code(value: f32) -> i32 {
    (value * CODE_SCALE + 0.5).floor() as i32
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender's custom-normal code lies within the signed 16-bit range"
)]
fn short(code: i32) -> i16 {
    code as i16
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

/// Fans and spaces of a mesh, using Blender's single-precision normal math and loop numbering.
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
        let mut local_vertices = Vec::new();
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
/// its lowest global corner.
fn traverse(
    corners: &[FanCorner],
    links: &[Link],
    start: usize,
    numbering: &[usize],
) -> Vec<usize> {
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

/// The fan normal: face normals weighted by each corner's angle.
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
        let factor = safe_acos(dot(
            directions[corner.edge_prev],
            directions[corner.edge_next],
        ));
        sum += face_normals[corner.face] * factor;
    }
    normalize(sum)
}

type Topology = (Vec<DVec3>, HashMap<u32, usize>, HashSet<(usize, usize)>);

/// Vertex positions, the vertex-index lookup by ID, and the sharp edges as vertex-index pairs.
fn topology(mesh: &Mesh, positions: &[DVec3]) -> Result<Topology> {
    if positions.len() != mesh.vertices.len() {
        return Err(normal_state_error(
            "custom normal positions must match the mesh vertices",
        ));
    }
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut sharp = HashSet::new();
    for (edge, flag) in mesh.edges.iter().zip(sharp_edge_flags(mesh)?) {
        if flag {
            let first = vertex_index(&index_by_id, edge.vertices[0])?;
            let second = vertex_index(&index_by_id, edge.vertices[1])?;
            sharp.insert(edge_key(first, second));
        }
    }
    Ok((positions.to_vec(), index_by_id, sharp))
}

fn corner_count(mesh: &Mesh) -> usize {
    mesh.faces.iter().map(|face| face.vertices.len()).sum()
}

/// The corner numbering of `mesh` as it is stored.
pub(crate) fn stored_corner_order(mesh: &Mesh) -> Vec<usize> {
    (0..corner_count(mesh)).collect()
}

fn normal_state_error(message: &'static str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

/// The index of a vertex ID, or an error when the mesh names a vertex it does not store.
fn vertex_index(index_by_id: &HashMap<u32, usize>, id: u32) -> Result<usize> {
    index_by_id
        .get(&id)
        .copied()
        .ok_or_else(|| normal_state_error("custom normal mesh references a missing vertex"))
}

fn validate_corner_order(order: &[usize], corner_count: usize) -> Result<()> {
    if order.len() != corner_count {
        return Err(normal_state_error(
            "custom normal corner_order must match the mesh corners",
        ));
    }
    let mut seen = vec![false; corner_count];
    for &ordinal in order {
        let Some(slot) = seen.get_mut(ordinal) else {
            return Err(normal_state_error(
                "custom normal corner_order must be a loop ordinal permutation",
            ));
        };
        if *slot {
            return Err(normal_state_error(
                "custom normal corner_order must be a loop ordinal permutation",
            ));
        }
        *slot = true;
    }
    Ok(())
}

/// Blender's loop ordinal for every current mesh corner, when the fan-code basis needs an order
/// different from the face storage order.
pub(crate) fn corner_order(mesh: &Mesh) -> Result<Option<Vec<usize>>> {
    let Some(attribute) = mesh.attributes.get("custom_normal") else {
        return Ok(None);
    };
    let Some(order) = attribute.get("corner_order") else {
        return Ok(None);
    };
    let order = order
        .as_object()
        .ok_or_else(|| normal_state_error("custom normal corner_order must be face-keyed"))?;
    let mut flattened = Vec::with_capacity(corner_count(mesh));
    if order.len() != mesh.faces.len() {
        return Err(normal_state_error(
            "custom normal corner_order must contain one row per face",
        ));
    }
    for face in &mesh.faces {
        let row = order
            .get(&format!("f{}", face.id))
            .and_then(Value::as_array)
            .filter(|row| row.len() == face.vertices.len())
            .ok_or_else(|| {
                normal_state_error("custom normal corner_order row has the wrong size")
            })?;
        for value in row {
            let ordinal = value
                .as_u64()
                .and_then(|ordinal| usize::try_from(ordinal).ok())
                .ok_or_else(|| {
                    normal_state_error("custom normal corner_order entries must be ordinals")
                })?;
            flattened.push(ordinal);
        }
    }
    validate_corner_order(&flattened, corner_count(mesh))?;
    Ok(Some(flattened))
}

/// The authoritative custom-normal state, flattened in mesh face/corner order.
pub(crate) fn corner_codes(mesh: &Mesh) -> Result<Option<Vec<CornerNormalCode>>> {
    let Some(attribute) = mesh.attributes.get("custom_normal") else {
        return Ok(None);
    };
    let fields = attribute
        .as_object()
        .ok_or_else(|| normal_state_error("custom_normal must be an object"))?;
    if fields.get("domain").and_then(Value::as_str) != Some("corner")
        || fields.get("type").and_then(Value::as_str) != Some("normal_code")
    {
        return Err(normal_state_error(
            "custom_normal must be a normal_code corner attribute",
        ));
    }
    let Some(raw_rows) = fields.get("values") else {
        let _ = corner_order(mesh)?;
        return Ok(None);
    };
    let rows = raw_rows
        .as_object()
        .ok_or_else(|| normal_state_error("custom_normal values must be an object"))?;
    if rows.len() != mesh.faces.len() {
        return Err(normal_state_error(
            "custom_normal must contain one row per mesh face",
        ));
    }
    let mut codes = Vec::with_capacity(corner_count(mesh));
    for face in &mesh.faces {
        let row = rows
            .get(&format!("f{}", face.id))
            .and_then(Value::as_array)
            .filter(|row| row.len() == face.vertices.len())
            .ok_or_else(|| normal_state_error("custom_normal row must match its face corners"))?;
        for value in row {
            let code = if value.is_null() {
                CornerNormalCode::Auto
            } else if value.as_str() == Some("explicit_zero") {
                CornerNormalCode::ExplicitZero
            } else if let Some(pair) = value.as_array().filter(|pair| pair.len() == 2) {
                let first = pair[0]
                    .as_i64()
                    .and_then(|value| i16::try_from(value).ok())
                    .ok_or_else(|| normal_state_error("custom_normal code must fit i16"))?;
                let second = pair[1]
                    .as_i64()
                    .and_then(|value| i16::try_from(value).ok())
                    .ok_or_else(|| normal_state_error("custom_normal code must fit i16"))?;
                CornerNormalCode::from_pair([first, second])
            } else {
                return Err(normal_state_error(
                    "custom_normal entries must be null, an i16 pair, or explicit_zero",
                ));
            };
            codes.push(code);
        }
    }
    let _ = corner_order(mesh)?;
    Ok(Some(codes))
}

/// Checks the canonical custom-normal attribute and its optional loop-order permutation.
pub(crate) fn validate(mesh: &Mesh) -> Result<()> {
    let _ = corner_codes(mesh)?;
    Ok(())
}

/// Fans of `mesh`, with its faces' vertex indices, smooth flags, sharp edges and loop ordinals.
pub(crate) fn mesh_fans(mesh: &Mesh, positions: &[DVec3], numbering: &[usize]) -> Result<Fans> {
    validate_corner_order(numbering, corner_count(mesh))?;
    let (points, index_by_id, sharp) = topology(mesh, positions)?;
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|&id| vertex_index(&index_by_id, id))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(fans(
        &points,
        &faces,
        &smooth_face_flags(mesh)?,
        &sharp,
        numbering,
    ))
}

fn resolved_corner_order(mesh: &Mesh, provided: Option<&[usize]>) -> Result<Vec<usize>> {
    if let Some(order) = provided {
        validate_corner_order(order, corner_count(mesh))?;
        return Ok(order.to_vec());
    }
    Ok(corner_order(mesh)?.unwrap_or_else(|| (0..corner_count(mesh)).collect()))
}

/// The fan-evaluated loop normals for a typed code buffer, including Auto corners.
pub(crate) fn evaluate_fan_codes(fans: &Fans, codes: &[CornerNormalCode]) -> Result<Vec<Vec3>> {
    if codes.len() != fans.fan_of.len() {
        return Err(normal_state_error(
            "custom normal codes must match the mesh corners",
        ));
    }
    let mut normals = vec![Vec3::ZERO; codes.len()];
    for (fan, corners) in fans.corners.iter().enumerate() {
        let mut sum = [0_i32; 2];
        for &corner in corners {
            if let CornerNormalCode::Authored(pair) = codes[corner] {
                sum[0] += i32::from(pair[0]);
                sum[1] += i32::from(pair[1]);
            }
        }
        let count = i32::try_from(corners.len()).unwrap_or(i32::MAX);
        let average = [sum[0] / count, sum[1] / count];
        let normal = fans.spaces[fan].decode(average);
        for &corner in corners {
            normals[corner] = if codes[corner] == CornerNormalCode::ExplicitZero {
                Vec3::ZERO
            } else {
                normal
            };
        }
    }
    Ok(normals)
}

/// Evaluates one caller-owned code buffer without serializing a temporary mesh attribute.
pub(crate) fn evaluate_code_buffer(
    mesh: &Mesh,
    codes: &[CornerNormalCode],
    positions: &[DVec3],
    order: Option<&[usize]>,
) -> Result<Vec<DVec3>> {
    let numbering = resolved_corner_order(mesh, order)?;
    let fans = mesh_fans(mesh, positions, &numbering)?;
    Ok(evaluate_fan_codes(&fans, codes)?
        .into_iter()
        .map(Vec3::as_dvec3)
        .collect())
}

/// Evaluates the mesh's stored custom-normal codes at `positions`.
pub(crate) fn evaluate_codes(
    mesh: &Mesh,
    codes: &[CornerNormalCode],
    positions: &[DVec3],
    options: NormalCodeOptions<'_>,
) -> Result<Vec<DVec3>> {
    evaluate_code_buffer(mesh, codes, positions, options.corner_order)
}

/// Converts a vector in one already-computed fan space to its canonical stored code.
pub(crate) fn encode_vector_in_space(space: &Space, vector: DVec3) -> CornerNormalCode {
    if vector == DVec3::ZERO {
        return CornerNormalCode::ExplicitZero;
    }
    let [first, second] = space.encode(vector.as_vec3());
    CornerNormalCode::from_pair([short(first), short(second)])
}

/// Authored/Auto view of the canonical code state; automatic entries remain `None`.
pub(crate) fn corner_custom_normals(
    mesh: &Mesh,
    positions: &[DVec3],
    options: NormalCodeOptions<'_>,
) -> Result<Option<Vec<Option<DVec3>>>> {
    let Some(codes) = corner_codes(mesh)? else {
        return Ok(None);
    };
    let normals = evaluate_codes(mesh, &codes, positions, options)?;
    Ok(Some(
        codes
            .iter()
            .zip(normals)
            .map(|(code, normal)| match code {
                CornerNormalCode::Auto => None,
                CornerNormalCode::Authored(_) => Some(normal),
                CornerNormalCode::ExplicitZero => Some(DVec3::ZERO),
            })
            .collect(),
    ))
}

fn code_value(code: CornerNormalCode) -> Value {
    match code {
        CornerNormalCode::Auto | CornerNormalCode::Authored([0, 0]) => Value::Null,
        CornerNormalCode::Authored(pair) => json!(pair),
        CornerNormalCode::ExplicitZero => Value::String("explicit_zero".to_owned()),
    }
}

/// The identity basis is the default, so it is never stored.
fn is_identity(order: &[usize]) -> bool {
    order
        .iter()
        .enumerate()
        .all(|(corner, &ordinal)| corner == ordinal)
}

/// One row per face in `mesh.faces` order, keyed `f<id>`, cut from values in corner order.
fn face_rows(mesh: &Mesh, flat: Vec<Value>) -> Map<String, Value> {
    let mut rows = Map::new();
    let mut start = 0;
    for face in &mesh.faces {
        let end = start + face.vertices.len();
        rows.insert(
            format!("f{}", face.id),
            Value::Array(flat[start..end].to_vec()),
        );
        start = end;
    }
    rows
}

/// Stores the canonical state: `codes` allocates the layer when `Some`, and a non-identity `order`
/// is kept as the basis. With neither, the attribute is removed.
fn store_state(
    mesh: &mut Mesh,
    codes: Option<&[CornerNormalCode]>,
    order: Option<&[usize]>,
) -> Result<()> {
    let count = corner_count(mesh);
    if let Some(codes) = codes
        && codes.len() != count
    {
        return Err(normal_state_error(
            "custom normal codes must match the mesh corners",
        ));
    }
    if let Some(order) = order {
        validate_corner_order(order, count)?;
    }
    let order = order.filter(|order| !is_identity(order));
    if codes.is_none() && order.is_none() {
        mesh.attributes.remove("custom_normal");
        return Ok(());
    }
    let mut attribute = Map::new();
    attribute.insert("domain".to_owned(), json!("corner"));
    attribute.insert("type".to_owned(), json!("normal_code"));
    if let Some(codes) = codes {
        let flat: Vec<Value> = codes.iter().copied().map(code_value).collect();
        attribute.insert("values".to_owned(), Value::Object(face_rows(mesh, flat)));
    }
    if let Some(order) = order {
        let flat: Vec<Value> = order.iter().map(|&ordinal| Value::from(ordinal)).collect();
        attribute.insert(
            "corner_order".to_owned(),
            Value::Object(face_rows(mesh, flat)),
        );
    }
    mesh.attributes
        .insert("custom_normal".to_owned(), Value::Object(attribute));
    Ok(())
}

/// Writes the custom-normal state. `codes` of `None` removes the layer but keeps a non-default
/// basis; `Some` allocates the layer even when every entry is automatic. An unset `corner_order`
/// keeps the persisted basis.
pub(crate) fn write_codes(
    mesh: &mut Mesh,
    codes: Option<&[CornerNormalCode]>,
    options: NormalCodeOptions<'_>,
) -> Result<()> {
    let existing = match options.corner_order {
        Some(_) => None,
        None => corner_order(mesh)?,
    };
    store_state(mesh, codes, options.corner_order.or(existing.as_deref()))
}

/// Writes Blender's sharp edges selected by the custom-normal setter.
fn write_sharp_edges(mesh: &mut Mesh, sharp: &[bool]) {
    let values: Map<String, Value> = mesh
        .edges
        .iter()
        .zip(sharp)
        .map(|(edge, flag)| (format!("e{}", edge.id), json!(flag)))
        .collect();
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain": "edge", "type": "bool", "values": values}),
    );
}

/// Marks the sharp edges where a checked fan's corner vectors disagree, as Blender's custom-normal
/// setter does. `touched` limits the check to flagged fans (`None` checks all). Returns whether any
/// edge changed.
fn mark_custom_normal_edges(
    mesh: &mut Mesh,
    positions: &[DVec3],
    vectors: &[Vec3],
    numbering: &[usize],
    touched: Option<&[bool]>,
) -> Result<bool> {
    let fans = mesh_fans(mesh, positions, numbering)?;
    let (_, index_of, _) = topology(mesh, positions)?;
    let mut edge_lookup = HashMap::new();
    for (index, edge) in mesh.edges.iter().enumerate() {
        let first = vertex_index(&index_of, edge.vertices[0])?;
        let second = vertex_index(&index_of, edge.vertices[1])?;
        edge_lookup.insert(edge_key(first, second), index);
    }
    let mut outgoing = Vec::with_capacity(vectors.len());
    let mut incoming = Vec::with_capacity(vectors.len());
    for face in &mesh.faces {
        let vertices = face
            .vertices
            .iter()
            .map(|&id| vertex_index(&index_of, id))
            .collect::<Result<Vec<usize>>>()?;
        let count = vertices.len();
        for corner in 0..count {
            outgoing.push(edge_key(vertices[corner], vertices[(corner + 1) % count]));
            incoming.push(edge_key(
                vertices[(corner + count - 1) % count],
                vertices[corner],
            ));
        }
    }
    let mut sharp = sharp_edge_flags(mesh)?;
    let mut marked = false;
    for (fan_index, corners) in fans.corners.iter().enumerate() {
        if touched.is_some_and(|touched| !touched[fan_index]) {
            continue;
        }
        let mut reversed = corners.iter().rev();
        let Some(&first) = reversed.next() else {
            continue;
        };
        let mut reference = vectors[first];
        let mut previous = first;
        for &corner in reversed {
            if dot(reference, vectors[corner]) < TRIGO_THRESHOLD {
                let edge = if outgoing[previous] == incoming[corner] {
                    outgoing[previous]
                } else {
                    outgoing[corner]
                };
                if let Some(&index) = edge_lookup.get(&edge)
                    && !sharp[index]
                {
                    sharp[index] = true;
                    marked = true;
                }
                reference = vectors[corner];
            }
            previous = corner;
        }
        if corners.len() > 1 && dot(reference, vectors[first]) < TRIGO_THRESHOLD {
            let edge = if outgoing[previous] == incoming[first] {
                outgoing[previous]
            } else {
                outgoing[first]
            };
            if let Some(&index) = edge_lookup.get(&edge)
                && !sharp[index]
            {
                sharp[index] = true;
                marked = true;
            }
        }
    }
    if marked {
        write_sharp_edges(mesh, &sharp);
    }
    Ok(marked)
}

/// Blender's custom-normal setter over the fans flagged in `touched` (every fan when `None`):
/// missing vectors take their computed normal, edges where a checked fan's vectors disagree are
/// marked sharp, and each fan's mean vector is encoded in its own space.
fn set_fan_codes(
    mesh: &mut Mesh,
    vectors: &[Option<DVec3>],
    numbering: &[usize],
    setter_input: SetterInput,
    touched: Option<&[bool]>,
) -> Result<Vec<CornerNormalCode>> {
    let count = corner_count(mesh);
    if vectors.len() != count {
        return Err(normal_state_error(
            "custom normal vectors must match the mesh corners",
        ));
    }
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let computed = plain_corner_normals(mesh, &positions)?;
    let mut filled = Vec::with_capacity(count);
    for (vector, base) in vectors.iter().zip(&computed) {
        let value = match vector {
            Some(vector) => {
                if !vector.is_finite() {
                    return Err(normal_state_error("custom normal vectors must be finite"));
                }
                // Blender replaces already-normalized zero inputs before edge splitting and fan encoding.
                if *vector == DVec3::ZERO && setter_input == SetterInput::AlreadyNormalized {
                    base.as_vec3()
                } else {
                    let vector = vector.as_vec3();
                    match setter_input {
                        SetterInput::NormalizeInput => normalize(vector),
                        SetterInput::AlreadyNormalized => vector,
                    }
                }
            }
            None => base.as_vec3(),
        };
        filled.push(value);
    }
    if mark_custom_normal_edges(mesh, &positions, &filled, numbering, touched)? {
        // Marked edges change the fans, so automatic corners take their recomputed normal.
        let computed = plain_corner_normals(mesh, &positions)?;
        for (index, vector) in vectors.iter().enumerate() {
            if vector.is_none() {
                filled[index] = computed[index].as_vec3();
            }
        }
    }
    let fans = mesh_fans(mesh, &positions, numbering)?;
    let mut codes = vec![CornerNormalCode::Auto; count];
    for (fan, corners) in fans.corners.iter().enumerate() {
        let mut sum = Vec3::ZERO;
        for &corner in corners {
            sum += filled[corner];
        }
        let average = sum * (1.0 / corners.len() as f32);
        let [first, second] = fans.spaces[fan].encode(average);
        let fan_code = CornerNormalCode::from_pair([short(first), short(second)]);
        for &corner in corners {
            codes[corner] = if vectors[corner] == Some(DVec3::ZERO)
                && setter_input == SetterInput::NormalizeInput
            {
                CornerNormalCode::ExplicitZero
            } else {
                fan_code
            };
        }
    }
    Ok(codes)
}

/// The vector-input boundary for custom normals. `None` removes the layer and keeps a required
/// basis; `Some` allocates the layer, and a missing corner vector means automatic.
pub(crate) fn encode_vectors(
    mesh: &mut Mesh,
    vectors: Option<&[Option<DVec3>]>,
    options: NormalCodeOptions<'_>,
) -> Result<()> {
    let Some(vectors) = vectors else {
        return write_codes(mesh, None, options);
    };
    let numbering = resolved_corner_order(mesh, options.corner_order)?;
    let codes = set_fan_codes(mesh, vectors, &numbering, options.setter_input, None)?;
    write_codes(mesh, Some(codes.as_slice()), options)
}

/// Re-encodes only the fans that hold an updated corner; every other fan keeps its codes bit-exact.
/// `updates` are `(flattened corner, vector)` pairs, and `None` makes that corner automatic.
pub(crate) fn encode_vector_updates(
    mesh: &mut Mesh,
    updates: &[(usize, Option<DVec3>)],
    options: NormalCodeOptions<'_>,
) -> Result<()> {
    if updates.is_empty() {
        return Ok(());
    }
    let count = corner_count(mesh);
    let numbering = resolved_corner_order(mesh, options.corner_order)?;
    let old_codes = corner_codes(mesh)?.unwrap_or_else(|| vec![CornerNormalCode::Auto; count]);
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let old_fans = mesh_fans(mesh, &positions, &numbering)?;
    let old_normals = evaluate_fan_codes(&old_fans, &old_codes)?;
    let mut vectors: Vec<Option<DVec3>> = old_codes
        .iter()
        .zip(old_normals)
        .map(|(code, normal)| match code {
            CornerNormalCode::Auto => None,
            CornerNormalCode::Authored(_) => Some(normal.as_dvec3()),
            CornerNormalCode::ExplicitZero => Some(DVec3::ZERO),
        })
        .collect();
    let mut updated = vec![false; count];
    let mut touched = vec![false; old_fans.corners.len()];
    for &(corner, vector) in updates {
        if corner >= count {
            return Err(normal_state_error(
                "custom normal update corner is out of range",
            ));
        }
        if std::mem::replace(&mut updated[corner], true) {
            return Err(normal_state_error(
                "custom normal update corners must be unique",
            ));
        }
        vectors[corner] = vector;
        touched[old_fans.fan_of[corner]] = true;
    }
    let mut codes = set_fan_codes(
        mesh,
        &vectors,
        &numbering,
        options.setter_input,
        Some(touched.as_slice()),
    )?;
    // The setter can mark an edge that splits an untouched fan. Only a fan whose corners are
    // unchanged keeps its codes.
    let new_fans = mesh_fans(mesh, &positions, &numbering)?;
    for (fan, corners) in old_fans.corners.iter().enumerate() {
        let same_fan = new_fans.corners[new_fans.fan_of[corners[0]]] == *corners;
        if !touched[fan] && same_fan {
            for &corner in corners {
                codes[corner] = old_codes[corner];
            }
        }
    }
    write_codes(mesh, Some(codes.as_slice()), options)
}

/// Canonicalizes the stored custom-normal state in place: `[0, 0]` entries become automatic, an
/// identity basis is dropped, and an empty state is removed. Malformed state is an error.
pub(crate) fn settle_custom_normals(mesh: &mut Mesh) -> Result<()> {
    let codes = corner_codes(mesh)?;
    let order = corner_order(mesh)?;
    store_state(mesh, codes.as_deref(), order.as_deref())
}

/// Typed code state for `output`, copied from the source corner each output corner comes from.
/// Automatic and explicit-zero status stay distinct; evaluation averages the copied codes in the
/// output fan. `None` when the source has no layer.
pub(crate) fn rebuild_custom_normals(
    source: &Mesh,
    output: &Mesh,
    sources: &[Option<usize>],
    options: NormalCodeOptions<'_>,
) -> Result<Option<Vec<CornerNormalCode>>> {
    let Some(source_codes) = corner_codes(source)? else {
        return Ok(None);
    };
    let output_count = corner_count(output);
    if sources.len() != output_count
        || sources
            .iter()
            .flatten()
            .any(|&index| index >= source_codes.len())
    {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "custom normal transfer does not describe the rebuilt corners",
        ));
    }
    if let Some(order) = options.corner_order {
        validate_corner_order(order, output_count)?;
    }
    Ok(Some(
        sources
            .iter()
            .map(|origin| origin.map_or(CornerNormalCode::Auto, |index| source_codes[index]))
            .collect(),
    ))
}

/// Compacts the source loop basis through a corner correspondence.
///
/// Dropped source loops leave gaps, and duplicated source corners become distinct output loops.
fn compact_corner_order(source_order: &[usize], sources: &[Option<usize>]) -> Vec<usize> {
    let mut source_loops: Vec<(bool, usize, usize)> = sources
        .iter()
        .enumerate()
        .map(|(output, source)| match source {
            Some(source) => (false, source_order[*source], output),
            None => (true, 0, output),
        })
        .collect();
    source_loops.sort_unstable();
    let mut order = vec![0; sources.len()];
    for (ordinal, (_, _, output)) in source_loops.into_iter().enumerate() {
        order[output] = ordinal;
    }
    order
}

/// Writes the output's custom-normal state rebuilt from `source`, compacting loop order through
/// the source-corner correspondence when the caller does not provide an output basis.
pub(crate) fn transfer_custom_normals(
    source: &Mesh,
    output: &mut Mesh,
    sources: &[Option<usize>],
    options: NormalCodeOptions<'_>,
) -> Result<()> {
    let source_count = corner_count(source);
    let output_count = corner_count(output);
    if sources.len() != output_count || sources.iter().flatten().any(|&index| index >= source_count)
    {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "custom normal transfer does not describe the rebuilt corners",
        ));
    }
    let order = match options.corner_order {
        Some(order) => {
            validate_corner_order(order, output_count)?;
            order.to_vec()
        }
        None => {
            let source_order = corner_order(source)?.unwrap_or_else(|| (0..source_count).collect());
            compact_corner_order(&source_order, sources)
        }
    };
    let codes = rebuild_custom_normals(
        source,
        output,
        sources,
        NormalCodeOptions {
            corner_order: Some(order.as_slice()),
            setter_input: options.setter_input,
        },
    )?;
    store_state(output, codes.as_deref(), Some(order.as_slice()))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use glam::{DVec3, Vec3};
    use serde_json::{Map, Value, json};

    use super::{
        CornerNormalCode, NormalCodeOptions, SetterInput, corner_codes, corner_count,
        corner_custom_normals, corner_order, encode_vector_updates, encode_vectors, evaluate_codes,
        mesh_fans, rebuild_custom_normals, settle_custom_normals, write_codes,
    };
    use crate::geom::{
        BoxParams, Mesh,
        normals::{corner_normals, shade_faces_smooth},
    };

    /// Authored vectors round-trip through 16-bit codes, so they match to this measured precision.
    const CODEC_TOLERANCE: f64 = 1.0e-3;

    fn smooth_cube() -> Mesh {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).expect("cube");
        shade_faces_smooth(&mut mesh);
        mesh
    }

    fn options() -> NormalCodeOptions<'static> {
        NormalCodeOptions {
            corner_order: None,
            setter_input: SetterInput::NormalizeInput,
        }
    }

    fn positions_of(mesh: &Mesh) -> Vec<DVec3> {
        mesh.vertices.iter().map(|vertex| vertex.co).collect()
    }

    /// One unit direction that no fan normal of the cube lies near.
    fn probe() -> DVec3 {
        DVec3::new(0.3, 0.9, 0.1).normalize()
    }

    #[test]
    fn from_pair_canonicalizes_the_zero_pair() {
        assert_eq!(CornerNormalCode::from_pair([0, 0]), CornerNormalCode::Auto);
        assert_eq!(
            CornerNormalCode::from_pair([3, -4]),
            CornerNormalCode::Authored([3, -4])
        );
    }

    #[test]
    fn code_state_preserves_auto_explicit_zero_and_layer_presence() {
        let mut mesh = smooth_cube();
        let mut codes = vec![CornerNormalCode::Auto; corner_count(&mesh)];
        codes[0] = CornerNormalCode::Authored([12_000, 7_000]);
        codes[1] = CornerNormalCode::ExplicitZero;
        codes[2] = CornerNormalCode::Authored([0, 0]);
        write_codes(&mut mesh, Some(codes.as_slice()), options()).expect("write codes");

        let read = corner_codes(&mesh)
            .expect("read codes")
            .expect("present layer");
        assert_eq!(read[0], CornerNormalCode::Authored([12_000, 7_000]));
        assert_eq!(read[1], CornerNormalCode::ExplicitZero);
        assert_eq!(
            read[2],
            CornerNormalCode::Auto,
            "zero code canonicalizes to Auto"
        );
        assert!(read[3..].iter().all(|code| *code == CornerNormalCode::Auto));
        assert!(mesh.attributes["custom_normal"]["values"].is_object());

        let autos = vec![CornerNormalCode::Auto; corner_count(&mesh)];
        write_codes(&mut mesh, Some(autos.as_slice()), options()).expect("write auto layer");
        assert!(corner_codes(&mesh).expect("read auto layer").is_some());
        write_codes(&mut mesh, None, options()).expect("remove layer");
        assert!(corner_codes(&mesh).expect("read absent layer").is_none());
    }

    #[test]
    fn evaluation_separates_automatic_authored_and_explicit_zero_corners() {
        let mut mesh = smooth_cube();
        let count = corner_count(&mesh);
        let positions = positions_of(&mesh);
        let identity: Vec<usize> = (0..count).collect();
        let fans = mesh_fans(&mesh, &positions, &identity).expect("fans");
        let plain = corner_normals(&mesh, &positions).expect("plain normals");
        let mut codes = vec![CornerNormalCode::Auto; count];
        codes[0] = CornerNormalCode::Authored([12_000, 7_000]);
        codes[1] = CornerNormalCode::ExplicitZero;
        write_codes(&mut mesh, Some(codes.as_slice()), options()).expect("write codes");

        let evaluated = evaluate_codes(&mesh, &codes, &positions, options()).expect("evaluate");
        assert_eq!(evaluated[1], DVec3::ZERO, "explicit zero is literal");
        assert!(
            (evaluated[0].length() - 1.0).abs() < 1.0e-5,
            "an authored code decodes to a unit vector"
        );
        let automatic = (0..count)
            .find(|&corner| {
                codes[corner] == CornerNormalCode::Auto && fans.fan_of[corner] != fans.fan_of[0]
            })
            .expect("an automatic corner outside the authored fan");
        assert!(
            (evaluated[automatic] - plain[automatic]).length() < 1.0e-4,
            "an automatic corner takes its fan normal"
        );
    }

    #[test]
    fn removing_values_keeps_only_a_non_default_order() {
        let mut mesh = smooth_cube();
        let count = corner_count(&mesh);
        let reversed: Vec<usize> = (0..count).rev().collect();
        let autos = vec![CornerNormalCode::Auto; count];
        let ordered = NormalCodeOptions {
            corner_order: Some(reversed.as_slice()),
            setter_input: SetterInput::NormalizeInput,
        };
        write_codes(&mut mesh, Some(autos.as_slice()), ordered).expect("write ordered layer");
        assert!(
            corner_codes(&mesh).expect("read codes").is_some(),
            "all-automatic values keep the layer"
        );

        write_codes(&mut mesh, None, options()).expect("remove values");
        assert_eq!(corner_codes(&mesh).expect("read codes"), None);
        assert_eq!(
            corner_order(&mesh).expect("read order"),
            Some(reversed.clone()),
            "the basis outlives the values"
        );

        let identity: Vec<usize> = (0..count).collect();
        let basis = NormalCodeOptions {
            corner_order: Some(identity.as_slice()),
            setter_input: SetterInput::NormalizeInput,
        };
        write_codes(&mut mesh, None, basis).expect("reset basis");
        assert!(
            !mesh.attributes.contains_key("custom_normal"),
            "an identity basis is never stored"
        );
    }

    #[test]
    fn corner_order_is_validated_and_persisted_with_the_codes() {
        let mut mesh = smooth_cube();
        let codes = vec![CornerNormalCode::Auto; corner_count(&mesh)];
        let mut order = (0..codes.len()).collect::<Vec<_>>();
        order.reverse();
        let ordered = NormalCodeOptions {
            corner_order: Some(order.as_slice()),
            setter_input: SetterInput::NormalizeInput,
        };
        write_codes(&mut mesh, Some(codes.as_slice()), ordered).expect("write ordered codes");
        assert_eq!(corner_order(&mesh).expect("read order"), Some(order));
    }

    #[test]
    fn vector_setter_writes_code_state_and_keeps_explicit_zero_distinct() {
        let mut mesh = smooth_cube();
        let mut vectors = vec![None; corner_count(&mesh)];
        vectors[0] = Some(DVec3::new(0.35, 0.2, 0.1));
        vectors[1] = Some(DVec3::ZERO);
        encode_vectors(&mut mesh, Some(vectors.as_slice()), options()).expect("set normals");

        let codes = corner_codes(&mesh)
            .expect("read codes")
            .expect("custom layer present");
        assert!(
            codes
                .iter()
                .any(|code| matches!(code, CornerNormalCode::Authored(_)))
        );
        assert!(codes.contains(&CornerNormalCode::ExplicitZero));
        let positions = positions_of(&mesh);
        let normals = corner_custom_normals(&mesh, &positions, options())
            .expect("evaluate statuses")
            .expect("custom layer present");
        assert_eq!(normals[1], Some(DVec3::ZERO));
    }

    #[test]
    fn vector_setter_round_trips_within_codec_tolerance() {
        let mut mesh = smooth_cube();
        let count = corner_count(&mesh);
        let vectors = vec![Some(probe()); count];
        encode_vectors(&mut mesh, Some(vectors.as_slice()), options()).expect("set normals");
        let positions = positions_of(&mesh);
        let decoded = corner_custom_normals(&mesh, &positions, options())
            .expect("evaluate")
            .expect("layer present");
        for normal in decoded {
            let normal = normal.expect("authored corner");
            assert!((normal - probe()).length() < CODEC_TOLERANCE, "{normal:?}");
        }
    }

    #[test]
    fn deformation_keeps_codes_and_decodes_against_the_new_fan_space() {
        let mut mesh = smooth_cube();
        let mut vectors = vec![None; corner_count(&mesh)];
        vectors[0] = Some(DVec3::new(0.35, 0.2, 0.1));
        encode_vectors(&mut mesh, Some(vectors.as_slice()), options()).expect("set normals");
        let codes = corner_codes(&mesh)
            .expect("read codes")
            .expect("custom layer present");
        let positions = positions_of(&mesh);
        let before = evaluate_codes(&mesh, &codes, &positions, options()).expect("evaluate");

        mesh.vertices[0].co += DVec3::new(0.2, 0.0, 0.0);
        let positions = positions_of(&mesh);
        let after =
            evaluate_codes(&mesh, &codes, &positions, options()).expect("deformed evaluate");
        assert!(
            codes
                .iter()
                .any(|code| matches!(code, CornerNormalCode::Authored(_)))
        );
        assert_eq!(
            corner_codes(&mesh).expect("codes unchanged"),
            Some(codes.clone())
        );
        assert!(
            (0..codes.len())
                .filter(|&corner| matches!(codes[corner], CornerNormalCode::Authored(_)))
                .any(|corner| (before[corner] - after[corner]).length() > 1.0e-4),
            "authored codes decode in the changed fan space"
        );
    }

    #[test]
    fn vector_updates_reencode_only_the_fan_they_touch() {
        let mut mesh = smooth_cube();
        let count = corner_count(&mesh);
        let vectors = vec![Some(probe()); count];
        encode_vectors(&mut mesh, Some(vectors.as_slice()), options()).expect("set normals");
        let before = corner_codes(&mesh)
            .expect("read codes")
            .expect("layer present");
        let positions = positions_of(&mesh);
        let identity: Vec<usize> = (0..count).collect();
        let fans = mesh_fans(&mesh, &positions, &identity).expect("fans");
        // A small nudge stays inside the sharpness threshold, so the setter marks no edge.
        let nudged = (probe() + DVec3::new(0.0, 0.0, 1.0e-3)).normalize();
        encode_vector_updates(&mut mesh, &[(0, Some(nudged))], options()).expect("update corner 0");
        let after = corner_codes(&mesh)
            .expect("read codes")
            .expect("layer present");
        assert_ne!(after[0], before[0], "the touched fan is re-encoded");
        for corner in (0..count).filter(|&corner| fans.fan_of[corner] != fans.fan_of[0]) {
            assert_eq!(
                after[corner], before[corner],
                "corner {corner} is in an untouched fan"
            );
        }
    }

    #[test]
    fn rebuild_transports_raw_codes_and_keeps_auto_status() {
        let mut source = smooth_cube();
        let mut codes = vec![CornerNormalCode::Auto; corner_count(&source)];
        codes[0] = CornerNormalCode::Authored([15_000, -9_000]);
        codes[1] = CornerNormalCode::ExplicitZero;
        write_codes(&mut source, Some(codes.as_slice()), options()).expect("write source codes");
        settle_custom_normals(&mut source).expect("settle");
        let output = source.clone();
        let sources: Vec<Option<usize>> = (0..corner_count(&source)).map(Some).collect();

        let rebuilt = rebuild_custom_normals(&source, &output, &sources, options())
            .expect("rebuild")
            .expect("layer survives");
        assert_eq!(rebuilt, codes);
    }

    #[test]
    fn settle_canonicalizes_the_state_in_place() {
        let mut mesh = smooth_cube();
        let mut corner = 0;
        let mut rows = Map::new();
        let mut basis = Map::new();
        for face in &mesh.faces {
            let end = corner + face.vertices.len();
            rows.insert(
                format!("f{}", face.id),
                json!(vec![[0, 0]; face.vertices.len()]),
            );
            basis.insert(
                format!("f{}", face.id),
                json!((corner..end).collect::<Vec<usize>>()),
            );
            corner = end;
        }
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({
                "domain": "corner",
                "type": "normal_code",
                "values": rows,
                "corner_order": basis,
            }),
        );
        settle_custom_normals(&mut mesh).expect("settle");
        let attribute = &mesh.attributes["custom_normal"];
        assert!(
            attribute.get("corner_order").is_none(),
            "an identity basis is dropped"
        );
        assert!(
            attribute["values"]
                .as_object()
                .expect("rows")
                .values()
                .all(|row| row.as_array().expect("row").iter().all(Value::is_null)),
            "[0, 0] becomes automatic"
        );
    }

    #[test]
    fn setter_modes_differ_only_for_unnormalized_input() {
        let mut mesh = smooth_cube();
        let count = corner_count(&mesh);
        let positions = positions_of(&mesh);
        let public_setter = NormalCodeOptions {
            corner_order: None,
            setter_input: SetterInput::NormalizeInput,
        };
        let already_normalized = NormalCodeOptions {
            corner_order: None,
            setter_input: SetterInput::AlreadyNormalized,
        };
        let doubled = vec![Some(probe() * 2.0); count];
        encode_vectors(&mut mesh, Some(doubled.as_slice()), public_setter).expect("public setter");
        let decoded = corner_custom_normals(&mesh, &positions, public_setter)
            .expect("evaluate")
            .expect("layer present");
        for normal in decoded {
            let normal = normal.expect("authored corner");
            assert!((normal - probe()).length() < CODEC_TOLERANCE, "{normal:?}");
        }

        encode_vectors(&mut mesh, Some(doubled.as_slice()), already_normalized)
            .expect("subdivision setter");
        let raw = corner_custom_normals(&mesh, &positions, already_normalized)
            .expect("evaluate")
            .expect("layer present");
        assert!(
            raw.iter()
                .flatten()
                .any(|normal| (*normal - probe()).length() > 0.1),
            "the already-normalized setter keeps the doubled input's out-of-range angle"
        );
    }

    #[test]
    fn already_normalized_zero_uses_corner_normal_before_fan_encoding() {
        let source = smooth_cube();
        let positions = positions_of(&source);
        let base_normals = corner_normals(&source, &positions).expect("base normals");
        let identity: Vec<usize> = (0..corner_count(&source)).collect();
        let source_fans = mesh_fans(&source, &positions, &identity).expect("source fans");
        let (fan_index, corners) = source_fans
            .corners
            .iter()
            .enumerate()
            .find(|(_, corners)| corners.len() >= 3)
            .map(|(fan, corners)| (fan, corners.clone()))
            .expect("a smooth cube vertex has a multi-corner fan");
        let zero_corner = corners[0];
        let tilted_corner = corners[1];
        let base = base_normals[zero_corner];
        assert!(
            corners
                .iter()
                .all(|&corner| (base_normals[corner] - base).length() < 1.0e-5),
            "one smooth fan uses one computed corner normal"
        );

        let tangent = if base.cross(DVec3::X).length() > 0.5 {
            base.cross(DVec3::X).normalize()
        } else {
            base.cross(DVec3::Y).normalize()
        };
        let angle: f64 = 0.01;
        let tilted = base * angle.cos() + tangent * angle.sin();
        assert!(base.dot(tilted) > f64::from(super::TRIGO_THRESHOLD));
        let mut vectors: Vec<Option<DVec3>> = base_normals.iter().copied().map(Some).collect();
        vectors[zero_corner] = Some(DVec3::ZERO);
        vectors[tilted_corner] = Some(tilted);

        let mut subdivision_mesh = source.clone();
        let expected_sum = corners.iter().fold(Vec3::ZERO, |sum, &corner| {
            let vector = if corner == zero_corner {
                base
            } else {
                vectors[corner].expect("corner input")
            };
            sum + vector.as_vec3()
        });
        let expected_average = expected_sum * (1.0 / corners.len() as f32);
        let [first, second] = source_fans.spaces[fan_index].encode(expected_average);
        let expected = CornerNormalCode::from_pair([super::short(first), super::short(second)]);
        assert!(
            matches!(expected, CornerNormalCode::Authored(_)),
            "the mixed input's fan average must remain authored"
        );
        encode_vectors(
            &mut subdivision_mesh,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: SetterInput::AlreadyNormalized,
            },
        )
        .expect("Subdivision setter");
        let output_fans = mesh_fans(&subdivision_mesh, &positions, &identity).expect("output fans");
        assert_eq!(
            output_fans.corners[output_fans.fan_of[zero_corner]], corners,
            "a zero input is replaced by the computed corner normal before fan splitting"
        );
        let output_codes = corner_codes(&subdivision_mesh)
            .expect("output codes")
            .expect("custom layer");
        for corner in &corners {
            assert_eq!(
                output_codes[*corner], expected,
                "all corners receive the code of the full fan average"
            );
        }

        let mut public_mesh = source;
        encode_vectors(
            &mut public_mesh,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: SetterInput::NormalizeInput,
            },
        )
        .expect("public setter");
        let public_codes = corner_codes(&public_mesh)
            .expect("public codes")
            .expect("custom layer");
        assert_eq!(
            public_codes[zero_corner],
            CornerNormalCode::ExplicitZero,
            "the native normalized-input zero sentinel remains distinct"
        );
    }
}
