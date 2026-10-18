// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/bmesh/tools/bmesh_bevel.cc, source/blender/bmesh/intern/bmesh_construct.cc,
//   source/blender/bmesh/intern/bmesh_query.cc, source/blender/blenkernel/intern/mesh_calc_edges.cc,
//   source/blender/blenkernel/intern/mesh_normals.cc, SPDX-FileCopyrightText: 2023 Blender
//   Authors, GPL-2.0-or-later.

//! Shading of bevel output, from Blender's `bmesh_bevel.cc` (`bevel_harden_normals`,
//! `bevel_edges_sharp_boundary`).
//!
//! Bevel loops do not carry the input custom-normal codes: each loop starts at the zero code, which
//! decodes to its fan normal. Hardening writes a code only on the strip and corner loops that have a
//! rule, and each fan then averages its codes (zeros included) in the fan's own space, as
//! `normals_calc_corners` does. Blender's rule normals come from the BMesh face and vertex normals,
//! which this module recomputes from the output geometry.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde_json::{Map, Value, json};

use super::{Mesh, edge_key};
use crate::{
    error::Result,
    geom::{
        clnor::{self, NormalCodeOptions},
        normals::{corner_angle, face_normal, smooth_face_flags},
    },
};

/// Blender's `FKind` for an output polygon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A rebuilt input face (`F_RECON`).
    Source,
    /// An untouched input face (`F_ORIG`), retaining its original loops.
    Original,
    /// A strip along a beveled edge (`F_EDGE`).
    Strip,
    /// A corner polygon at a beveled vertex (`F_VERT`).
    Corner,
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
                if matches!(topology.kinds[polygon], Kind::Source | Kind::Original) {
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
    // BM_vert_step_fan_loop crosses the outgoing edge for Blender's `lprev` walk.
    let lprev = topology.neighbour(loop_, true);
    let lnext = topology.neighbour(loop_, false);
    let (kprev, knext) = (topology.kind(lprev), topology.kind(lnext));
    match own {
        Kind::Source | Kind::Original => None,
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

/// Converts a vector in one already-computed fan space to its canonical stored code. Zero is an
/// explicit zero vector, never the automatic code.
fn encode_vector_in_space(space: &clnor::Space, vector: DVec3) -> clnor::CornerNormalCode {
    if vector == DVec3::ZERO {
        return clnor::CornerNormalCode::ExplicitZero;
    }
    let [first, second] = space.encode(vector.as_vec3());
    clnor::CornerNormalCode::from_pair([first as i16, second as i16])
}

/// Applies Blender's loop-local hardening writes without re-encoding other fan corners.
fn write_hardened_codes(
    codes: &mut [clnor::CornerNormalCode],
    fans: &clnor::Fans,
    rules: &[Vec<Option<DVec3>>],
) {
    for (corner, normal) in rules.iter().flatten().enumerate() {
        if let Some(normal) = normal {
            codes[corner] = if *normal == DVec3::ZERO {
                clnor::CornerNormalCode::Auto
            } else {
                let fan = fans.fan_of[corner];
                encode_vector_in_space(&fans.spaces[fan], *normal)
            };
        }
    }
}

/// Writes the shading of a built bevel result (faces in `kinds` order).
///
/// With `harden`, `bevel_harden_normals` sets the custom-normal code of each loop that has a rule,
/// and each fan averages its codes. `input_custom` says the input had custom normals; without them
/// the boundary edges that `bevel_harden_normals` marks sharp are added here.
pub(super) fn shade_output(
    mesh: &mut Mesh,
    kinds: &[Kind],
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
    let sharp: Vec<(usize, usize)> = if harden && !input_custom {
        sharp_boundary(&topology, &smooth)
    } else {
        Vec::new()
    };
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
    if !harden {
        return Ok(());
    }
    let rules = harden_normals(&topology);
    let total: usize = polygons.iter().map(Vec::len).sum();
    let numbering = clnor::corner_order(mesh)?.unwrap_or_else(|| (0..total).collect());
    let fans = clnor::mesh_fans(mesh, &points, &numbering)?;
    let mut codes =
        clnor::corner_codes(mesh)?.unwrap_or_else(|| vec![clnor::CornerNormalCode::Auto; total]);
    write_hardened_codes(&mut codes, &fans, &rules);
    clnor::write_codes(
        mesh,
        Some(&codes),
        NormalCodeOptions {
            corner_order: Some(&numbering),
            ..NormalCodeOptions::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use std::collections::HashSet;

    use crate::geom::clnor::{self, CornerNormalCode};

    use super::{
        Kind, Topology, encode_vector_in_space, harden_normals, sharp_boundary,
        write_hardened_codes,
    };

    /// A source square `[0, 1, 2, 3]` in the plane `z = 0` and a strip `[0, 1, 4, 3]` below its edge
    /// `3 -> 0`. The strip's corners 0 and 2 are its LONG loops.
    fn square_and_strip() -> Topology {
        let points = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(1.0, 0.5, -1.0),
        ];
        Topology::new(
            &points,
            vec![vec![0, 1, 2, 3], vec![0, 1, 4, 3]],
            &[Kind::Source, Kind::Strip],
        )
    }

    fn assert_rule(actual: Option<DVec3>, expected: Option<DVec3>) {
        match (actual, expected) {
            (None, None) => {}
            (Some(actual), Some(expected)) => {
                assert!(
                    (actual - expected).length() < 1e-12,
                    "{actual:?} != {expected:?}"
                );
            }
            _ => panic!("{actual:?} != {expected:?}"),
        }
    }

    #[test]
    fn long_strip_loops_next_to_a_source_face_take_its_normal() {
        let rules = harden_normals(&square_and_strip());
        assert!(
            rules[0].iter().all(Option::is_none),
            "source loops keep their fan normal"
        );
        let up = Some(DVec3::Z);
        assert_rule(rules[1][0], up);
        assert_rule(rules[1][1], None);
        assert_rule(rules[1][2], None);
        assert_rule(rules[1][3], up);
    }

    #[test]
    fn smooth_source_edges_beside_a_corner_are_sharp() {
        let mut topology = square_and_strip();
        topology.kinds = vec![Kind::Source, Kind::Corner];
        // The corner polygon shares the source edges 0-1 and 3-0.
        assert_eq!(
            sharp_boundary(&topology, &[true, false]),
            vec![(0, 1), (0, 3)]
        );
        assert_eq!(
            sharp_boundary(&topology, &[false, false]),
            Vec::<(usize, usize)>::new()
        );
    }

    #[test]
    fn strip_long_loop_uses_the_face_across_its_outgoing_edge() {
        let points = [
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.0, 0.0, -1.0),
            DVec3::new(1.0, 0.0, -1.0),
            DVec3::new(0.0, 1.0, -1.0),
        ];
        let topology = Topology::new(
            &points,
            vec![vec![0, 1, 2], vec![0, 1, 4, 3], vec![3, 0, 5]],
            &[Kind::Source, Kind::Strip, Kind::Corner],
        );
        assert_rule(harden_normals(&topology)[1][0], Some(DVec3::Z));
    }

    #[test]
    fn corner_loop_walks_two_faces_along_the_blender_fan_direction() {
        let points = [
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::new(0.0, 1.0, -1.0),
            DVec3::new(0.0, 0.0, -1.0),
            DVec3::new(1.0, 0.0, -1.0),
        ];
        let topology = Topology::new(
            &points,
            vec![vec![0, 1, 2], vec![1, 0, 4, 3], vec![4, 0, 5]],
            &[Kind::Corner, Kind::Strip, Kind::Source],
        );
        assert_rule(harden_normals(&topology)[0][0], Some(DVec3::Y));
    }

    #[test]
    fn hardening_changes_only_assigned_codes_in_their_output_fan_spaces() {
        let points = [DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
        let faces = [vec![0, 1, 2], vec![1, 0, 3]];
        let numbering = [0, 1, 2, 3, 4, 5];
        let fans = clnor::fans(&points, &faces, &[true, true], &HashSet::new(), &numbering);
        assert_eq!(fans.fan_of[0], fans.fan_of[4]);

        let mut codes = [
            CornerNormalCode::Auto,
            CornerNormalCode::Authored([11, 12]),
            CornerNormalCode::ExplicitZero,
            CornerNormalCode::Auto,
            CornerNormalCode::Authored([21, 22]),
            CornerNormalCode::Authored([31, 32]),
        ];
        let expected = encode_vector_in_space(&fans.spaces[fans.fan_of[0]], DVec3::X);
        write_hardened_codes(
            &mut codes,
            &fans,
            &[
                vec![Some(DVec3::X), None, None],
                vec![None, None, Some(DVec3::ZERO)],
            ],
        );

        assert_eq!(codes[0], expected);
        assert_eq!(codes[1], CornerNormalCode::Authored([11, 12]));
        assert_eq!(codes[2], CornerNormalCode::ExplicitZero);
        assert_eq!(codes[3], CornerNormalCode::Auto);
        assert_eq!(codes[4], CornerNormalCode::Authored([21, 22]));
        assert_eq!(codes[5], CornerNormalCode::Auto);
    }
}
