// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_bevel.cc, source/blender/bmesh/intern/bmesh_construct.cc,
//   source/blender/bmesh/intern/bmesh_query.cc, source/blender/blenkernel/intern/mesh_calc_edges.cc,
//   source/blender/blenkernel/intern/mesh_normals.cc, SPDX-FileCopyrightText: 2023 Blender
//   Authors, GPL-2.0-or-later.

//! Shading of bevel output, from Blender's `bmesh_bevel.cc`.
//!
//! Blender builds every new face with `bev_create_ngon`, which copies the face attributes of one
//! representative face, the smooth flag included. Which face that is follows from the `BMesh` edge
//! order: a mesh stores its edges in the order `mesh_calc_edges` first meets them, and each
//! vertex's disk cycle keeps that order.
//!
//! After the faces are built, Blender carries no custom normals through a bevel: every loop is
//! recomputed from the output fans. `harden_normals` then writes the normals of strip and corner
//! loops from their neighbouring faces (`bevel_harden_normals`), and the custom normals of each
//! smooth fan are averaged in that fan's frame (`normals_calc_corners`). `mark_sharp` flags the
//! rung edges of smooth beveled edges (`set_bound_vert_seams`, `bevel_extend_edge_data`).

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde_json::{Map, Value, json};

use super::{BevelEdgeFacesWithMaterial, Mesh, Polygon, edge_key};
use crate::{
    error::Result,
    geom::{
        clnor::{self, NormalCodeOptions, SetterInput},
        normals::{corner_angle, face_normal, smooth_face_flags},
    },
};

/// A beveled edge's side: `(face index, loop's first vertex, loop's second vertex, material)`.
pub(super) type EdgeSide = (usize, usize, usize, u32);

/// Blender's `FKind` for an output polygon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A rebuilt input face (`F_RECON`).
    Source,
    /// A strip along a beveled edge (`F_EDGE`).
    Strip,
    /// A corner polygon at a beveled vertex (`F_VERT`).
    Corner,
}

/// Edges in the order Blender's mesh stores them, each as `(lower, higher)` vertex: the first
/// occurrence over the face corners, taking each corner's edge to its previous corner.
pub(super) fn blender_edge_order(polygons: &[Polygon]) -> Vec<(usize, usize)> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for (face, _) in polygons {
        for corner in 0..face.len() {
            let key = edge_key(face[(corner + face.len() - 1) % face.len()], face[corner]);
            if seen.insert(key) {
                order.push(key);
            }
        }
    }
    order
}

/// Faces on either side of an edge.
fn faces_of(edge_faces: &BevelEdgeFacesWithMaterial, edge: (usize, usize)) -> Vec<usize> {
    edge_faces
        .get(&edge)
        .map(|sides| sides.iter().map(|side| side.0).collect())
        .unwrap_or_default()
}

/// The face whose smooth flag a bevel's corner polygon copies at `vertex` (`bevel_build_poly`
/// takes `boundvert_rep_face` of the vertex's first boundary vertex).
///
/// `BMesh` starts the vertex's edge order at its first beveled edge `first` and walks to the first
/// later edge `second` that shares a face with it. Blender then reverses the order when the walk
/// turned clockwise, so the representative is the face of `first` that `second` does not share.
/// `None` when the vertex has no beveled edge or the topology is not a manifold fan there.
pub(super) fn corner_source(
    vertex: usize,
    order: &[(usize, usize)],
    edge_faces: &BevelEdgeFacesWithMaterial,
    selected_edges: &HashSet<(usize, usize)>,
) -> Option<usize> {
    let incident: Vec<(usize, usize)> = order
        .iter()
        .copied()
        .filter(|edge| edge.0 == vertex || edge.1 == vertex)
        .collect();
    let first = *incident.iter().find(|edge| selected_edges.contains(edge))?;
    let first_faces = faces_of(edge_faces, first);
    let second = incident
        .iter()
        .copied()
        .filter(|edge| *edge != first)
        .find(|edge| {
            faces_of(edge_faces, *edge)
                .iter()
                .any(|face| first_faces.contains(face))
        })?;
    let second_faces = faces_of(edge_faces, second);
    first_faces
        .into_iter()
        .find(|face| !second_faces.contains(face))
}

/// The two faces a beveled edge's strips copy their smooth flags from, as `(before, after)`, given
/// the edge's two sides. `before` is the face whose loop runs from the higher vertex to the lower:
/// Blender's `fprev` at the edge's lower vertex, which `bevel_build_edge_polygons` reads first.
pub(super) fn strip_faces(
    edge: (usize, usize),
    first: EdgeSide,
    second: EdgeSide,
) -> (usize, usize) {
    if (first.1, first.2) == (edge.1, edge.0) {
        (first.0, second.0)
    } else {
        (second.0, first.0)
    }
}

/// The face a bevel strip piece copies its smooth flag from. Blender interpolates a strip from
/// `before` up to its middle piece and from `after` beyond it: for an odd segment count the middle
/// piece is still `before`, for an even count it is `after` (`bevel_build_edge_polygons`).
/// `segment` is zero based.
pub(super) fn strip_face_for_segment(
    before: usize,
    after: usize,
    segments: usize,
    segment: usize,
) -> usize {
    let threshold = segments / 2 + segments % 2;
    if segment < threshold { before } else { after }
}

/// Orients a single-segment corner polygon as Blender builds it. Blender's corner loop runs each
/// rung it shares with a strip against that strip: the rung at the edge's higher vertex runs from
/// the `after` face to the `before` face, and at its lower vertex from `before` to `after`
/// (`strip_faces`). `faces` and `points` are the polygon's face and point at each corner, in
/// boundary order. The polygon is reversed when its first rung runs the other way.
pub(super) fn orient_corner(
    vertex: usize,
    faces: &[usize],
    points: Vec<usize>,
    edge_faces: &BevelEdgeFacesWithMaterial,
) -> Vec<usize> {
    let (Some(&first), Some(&next)) = (faces.first(), faces.get(1)) else {
        return points;
    };
    let shared = edge_faces.iter().find(|(edge, sides)| {
        (edge.0 == vertex || edge.1 == vertex)
            && sides.len() == 2
            && sides.iter().any(|side| side.0 == first)
            && sides.iter().any(|side| side.0 == next)
    });
    let Some((edge, sides)) = shared else {
        return points;
    };
    let (before, after) = strip_faces(*edge, sides[0], sides[1]);
    let from = if vertex == edge.1 { after } else { before };
    if from == first {
        points
    } else {
        points.into_iter().rev().collect()
    }
}

/// Rung edges that `mark_sharp` flags for a single-segment edge bevel, as point pairs.
///
/// A vertex with at least two input-sharp edges (`bevel_extend_edge_data`, with `sharp_len` from
/// `check_edge_data_seam_sharp_edges`) marks the rung of every beveled smooth edge: the corner
/// polygon edge between the two face corners of that edge at the vertex. `corner_point` gives the
/// output point of a face corner. Vertices with a single beveled edge are skipped
/// (`bv->selcount < 2`).
pub(super) fn mark_sharp_rungs(
    edge_faces: &BevelEdgeFacesWithMaterial,
    selected_edges: &HashSet<(usize, usize)>,
    input_sharp: &HashSet<(usize, usize)>,
    corner_point: &HashMap<(usize, usize), usize>,
    vertex_count: usize,
) -> Vec<(usize, usize)> {
    let mut rungs = Vec::new();
    for vertex in 0..vertex_count {
        let incident: Vec<(usize, usize)> = edge_faces
            .keys()
            .copied()
            .filter(|edge| edge.0 == vertex || edge.1 == vertex)
            .collect();
        let beveled = incident
            .iter()
            .filter(|edge| selected_edges.contains(edge))
            .count();
        let sharp = incident
            .iter()
            .filter(|edge| input_sharp.contains(edge))
            .count();
        if beveled < 2 || sharp < 2 {
            continue;
        }
        for edge in incident {
            if !selected_edges.contains(&edge) || input_sharp.contains(&edge) {
                continue;
            }
            if let [first, second] = edge_faces[&edge].as_slice()
                && let (Some(a), Some(b)) = (
                    corner_point.get(&(first.0, vertex)),
                    corner_point.get(&(second.0, vertex)),
                )
            {
                rungs.push((*a, *b));
            }
        }
    }
    rungs
}

/// Output polygons with their loops' neighbours, as Blender's fan walk sees them.
struct Topology {
    polygons: Vec<Vec<usize>>,
    kinds: Vec<Kind>,
    /// Polygons using each edge, by point pair.
    users: HashMap<(usize, usize), Vec<usize>>,
    face_normals: Vec<DVec3>,
    vertex_normals: Vec<DVec3>,
}

/// A loop: `(polygon index, corner index)`.
type Loop = (usize, usize);

impl Topology {
    fn new(points: &[DVec3], polygons: Vec<Vec<usize>>, kinds: &[Kind]) -> Self {
        let mut users: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut face_normals = Vec::with_capacity(polygons.len());
        let mut vertex_normals = vec![DVec3::ZERO; points.len()];
        for (index, polygon) in polygons.iter().enumerate() {
            let corners: Vec<DVec3> = polygon.iter().map(|vertex| points[*vertex]).collect();
            let normal = face_normal(&corners);
            face_normals.push(normal);
            for corner in 0..polygon.len() {
                let next = polygon[(corner + 1) % polygon.len()];
                users
                    .entry(edge_key(polygon[corner], next))
                    .or_default()
                    .push(index);
                vertex_normals[polygon[corner]] += normal * corner_angle(&corners, corner);
            }
        }
        Self {
            polygons,
            kinds: kinds.to_vec(),
            users,
            face_normals,
            vertex_normals: vertex_normals
                .into_iter()
                .map(DVec3::normalize_or_zero)
                .collect(),
        }
    }

    /// The loop of the other face on the loop's outgoing (`outgoing`) or incoming edge, at the
    /// loop's vertex. `None` for a non-manifold or boundary edge.
    fn neighbour(&self, (polygon, corner): Loop, outgoing: bool) -> Option<Loop> {
        let face = &self.polygons[polygon];
        let count = face.len();
        let vertex = face[corner];
        let other = if outgoing {
            face[(corner + 1) % count]
        } else {
            face[(corner + count - 1) % count]
        };
        let users = self.users.get(&edge_key(vertex, other))?;
        if users.len() != 2 {
            return None;
        }
        let next = *users.iter().find(|user| **user != polygon)?;
        let at_vertex = self.polygons[next].iter().position(|id| *id == vertex)?;
        Some((next, at_vertex))
    }

    fn kind(&self, loop_: Option<Loop>) -> Option<Kind> {
        loop_.map(|(polygon, _)| self.kinds[polygon])
    }

    /// Blender's `LONG` loop tag: strip loops whose outgoing edge runs along the beveled edge.
    /// Strip corners 0 and 2 start those edges in the layout `bevel()` emits.
    fn long(&self, (polygon, corner): Loop) -> bool {
        self.kinds[polygon] == Kind::Strip && corner % 2 == 0
    }

    fn previous(&self, (polygon, corner): Loop) -> Loop {
        let count = self.polygons[polygon].len();
        (polygon, (corner + count - 1) % count)
    }
}

/// Blender's `bevel_harden_normals`: the normal each strip and corner loop takes, `None` where no
/// rule applies and the loop keeps its computed fan normal. Source faces keep theirs.
fn harden_normals(topology: &Topology) -> Vec<Vec<Option<DVec3>>> {
    let mut normals = Vec::with_capacity(topology.polygons.len());
    for (polygon, face) in topology.polygons.iter().enumerate() {
        let row = (0..face.len())
            .map(|corner| {
                if topology.kinds[polygon] == Kind::Source {
                    None
                } else {
                    harden_loop(topology, (polygon, corner))
                }
            })
            .collect();
        normals.push(row);
    }
    normals
}

fn harden_loop(topology: &Topology, loop_: Loop) -> Option<DVec3> {
    let (polygon, corner) = loop_;
    let own = topology.kinds[polygon];
    let normal = |other: Loop| topology.face_normals[other.0];
    let prev_loop = topology.previous(loop_);
    let lprev = topology.neighbour(loop_, true);
    let lnext = topology.neighbour(loop_, false);
    let (kprev, knext) = (topology.kind(lprev), topology.kind(lnext));
    match own {
        Kind::Source => None,
        Kind::Strip => {
            if let (Some(prev), true) = (lprev, kprev == Some(Kind::Strip) && topology.long(loop_))
            {
                Some((topology.face_normals[polygon] + normal(prev)).normalize_or_zero())
            } else if let (Some(next), true) = (
                lnext,
                knext == Some(Kind::Strip) && lnext.is_some_and(|next| topology.long(next)),
            ) {
                Some((topology.face_normals[polygon] + normal(next)).normalize_or_zero())
            } else if let (Some(prev), true) =
                (lprev, kprev == Some(Kind::Source) && topology.long(loop_))
            {
                Some(normal(prev))
            } else if let (Some(next), true) = (
                lnext,
                knext == Some(Kind::Source) && topology.long(prev_loop),
            ) {
                Some(normal(next))
            } else {
                None
            }
        }
        Kind::Corner => {
            if kprev == Some(Kind::Corner) && knext == Some(Kind::Corner) {
                Some(topology.vertex_normals[topology.polygons[polygon][corner]])
            } else if let Some(prev) = lprev.filter(|_| kprev == Some(Kind::Source)) {
                Some(normal(prev))
            } else if let Some(next) = lnext.filter(|_| knext == Some(Kind::Source)) {
                Some(normal(next))
            } else {
                let lprevprev = lprev.and_then(|prev| topology.neighbour(prev, true));
                let lnextnext = lnext.and_then(|next| topology.neighbour(next, false));
                let kprevprev = topology.kind(lprevprev);
                let knextnext = topology.kind(lnextnext);
                if let (Some(prevprev), true) = (
                    lprevprev,
                    kprev == Some(Kind::Strip) && kprevprev == Some(Kind::Source),
                ) {
                    Some(normal(prevprev))
                } else if let (Some(prev), Some(prevprev), true) = (
                    lprev,
                    lprevprev,
                    kprev == Some(Kind::Strip)
                        && knext == Some(Kind::Corner)
                        && kprevprev == Some(Kind::Strip),
                ) {
                    Some((normal(prev) + normal(prevprev)).normalize_or_zero())
                } else if let (Some(next), Some(nextnext), true) = (
                    lnext,
                    lnextnext,
                    knext == Some(Kind::Strip)
                        && kprev == Some(Kind::Corner)
                        && knextnext == Some(Kind::Strip),
                ) {
                    Some((normal(next) + normal(nextnext)).normalize_or_zero())
                } else {
                    None
                }
            }
        }
    }
}

/// Blender's `bevel_edges_sharp_boundary`, run when the input has no custom normals: an edge of a
/// smooth rebuilt face that meets a strip or corner polygon is sharp. Point pairs.
fn sharp_boundary(topology: &Topology, smooth: &[bool]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for (polygon, face) in topology.polygons.iter().enumerate() {
        if topology.kinds[polygon] != Kind::Source || !smooth[polygon] {
            continue;
        }
        for corner in 0..face.len() {
            if let Some(other) = topology.neighbour((polygon, corner), true)
                && matches!(topology.kinds[other.0], Kind::Strip | Kind::Corner)
            {
                edges.push(edge_key(face[corner], face[(corner + 1) % face.len()]));
            }
        }
    }
    edges
}

/// Writes the shading of a built bevel result (faces in `kinds` order).
///
/// Blender drops the input custom normals (the loops are rebuilt), so the `custom_normal`
/// attribute is removed. `sharp_points` are the point pairs to mark sharp (see
/// [`mark_sharp_rungs`]). With `harden`, the boundary edges of smooth faces are marked sharp when
/// the input has no custom normals, and each strip and corner loop gets its hardened normal. The
/// hardened codes are averaged over their fan in the fan's frame, so every corner of an affected
/// fan takes the decoded average (Blender's `normals_calc_corners`).
pub(super) fn shade_output(
    mesh: &mut Mesh,
    kinds: &[Kind],
    sharp_points: &[(usize, usize)],
    harden: bool,
    input_custom: bool,
) -> Result<()> {
    let points: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let index_of: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let polygons: Vec<Vec<usize>> = mesh
        .faces
        .iter()
        .map(|face| face.vertices.iter().map(|id| index_of[id]).collect())
        .collect();
    let smooth = smooth_face_flags(mesh)?;
    let topology = Topology::new(&points, polygons.clone(), kinds);
    let mut sharp: Vec<(usize, usize)> = sharp_points.to_vec();
    if harden && !input_custom {
        sharp.extend(sharp_boundary(&topology, &smooth));
    }
    let ids: HashSet<(u32, u32)> = sharp
        .iter()
        .map(|(a, b)| {
            let (a, b) = (mesh.vertices[*a].id, mesh.vertices[*b].id);
            (a.min(b), a.max(b))
        })
        .collect();
    if !ids.is_empty() {
        let mut values = Map::new();
        for edge in &mesh.edges {
            let key = (
                edge.vertices[0].min(edge.vertices[1]),
                edge.vertices[0].max(edge.vertices[1]),
            );
            if ids.contains(&key) {
                values.insert(format!("e{}", edge.id), json!(true));
            }
        }
        let attribute = mesh
            .attributes
            .entry("sharp".to_owned())
            .or_insert_with(|| json!({"domain": "edge", "type": "bool", "values": {}}));
        if let Some(existing) = attribute.get_mut("values").and_then(Value::as_object_mut) {
            existing.extend(values);
        }
    }
    mesh.attributes.remove("custom_normal");
    if !harden {
        return Ok(());
    }
    let rules = harden_normals(&topology);
    if rules.iter().all(|row| row.iter().all(Option::is_none)) {
        return Ok(());
    }
    // Sharp flags are final now; the fans and their frames come from the mesh without custom data.
    let mut starts = Vec::with_capacity(polygons.len());
    let mut total = 0;
    for polygon in &polygons {
        starts.push(total);
        total += polygon.len();
    }
    // Bevel output keeps Potter's storage order as its loop order.
    let numbering: Vec<usize> = (0..total).collect();
    let fans = clnor::mesh_fans(mesh, &points, &numbering)?;
    // Each hardened corner's code in its fan's frame; unhardened corners of a fan contribute zero.
    let mut codes = vec![[0_i32; 2]; total];
    let mut affected = vec![false; fans.corners.len()];
    for (polygon, row) in rules.iter().enumerate() {
        let start = starts[polygon];
        for (corner, rule) in row.iter().enumerate() {
            if let Some(normal) = rule {
                let global = start + corner;
                let fan = fans.fan_of[global];
                codes[global] = fans.spaces[fan].encode(normal.as_vec3());
                affected[fan] = true;
            }
        }
    }
    let mut values = vec![DVec3::ZERO; total];
    for (fan, members) in fans.corners.iter().enumerate() {
        if !affected[fan] {
            continue;
        }
        let mut sum = [0_i32; 2];
        for member in members {
            sum[0] += codes[*member][0];
            sum[1] += codes[*member][1];
        }
        let count = members.len() as i32;
        let average = [sum[0] / count, sum[1] / count];
        let normal = fans.spaces[fan].decode(average).as_dvec3();
        for member in members {
            values[*member] = normal;
        }
    }
    // Corners of unaffected fans have no hardened normal: they stay automatic (`None`).
    let vectors: Vec<Option<DVec3>> = (0..total)
        .map(|global| affected[fans.fan_of[global]].then_some(values[global]))
        .collect();
    clnor::encode_vectors(
        mesh,
        Some(&vectors),
        NormalCodeOptions {
            corner_order: Some(&numbering),
            setter_input: SetterInput::NormalizeInput,
        },
    )
}
