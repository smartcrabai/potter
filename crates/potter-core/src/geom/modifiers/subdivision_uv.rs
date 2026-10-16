// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_converter_mesh.cc, source/blender/modifiers/intern/MOD_subsurf.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.
//   OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/fvarLevel.cpp, far/primvarRefiner.h, Apache-2.0.

//! Blender's UV layers and the position topology they share. A UV layer's values are Blender's UV
//! vertices: a position vertex splits wherever its UVs differ between faces. Each layer's limit is
//! evaluated by `subdivision_limit::face_varying`, on the leaves of the position limit surface.

use std::collections::{BTreeMap, HashMap};

use glam::{DVec2, DVec3};
use serde_json::{Map, json};

use super::{
    NonManifold, Polygon, blender_edge_crease_sharpness, build_mesh, positions, usize_as_f64,
};
use crate::{
    error::Result,
    geom::{
        Mesh, edge_key,
        provenance::Origin,
        uv_layers::{self, UvLayer},
    },
};

/// Face-varying interpolation of a UV layer, from `uv_smooth` (`OpenSubdiv` `FVarLinearInterpolation`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FvarLinear {
    /// `uv_smooth` NONE: every edge and value is linear, so the UVs interpolate bilinearly.
    All,
    /// `uv_smooth` SMOOTH_ALL: smooth everywhere, UV boundaries included.
    SmoothAll,
    /// `uv_smooth` PRESERVE_CORNERS: UV corners stay put.
    CornersOnly,
    /// `uv_smooth` PRESERVE_CORNERS_AND_JUNCTIONS: corners and junctions stay put.
    CornersPlus1,
    /// `uv_smooth` PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE: corners, junctions, and concave corners.
    CornersPlus2,
    /// `uv_smooth` PRESERVE_BOUNDARIES (Blender's default): UV boundaries stay put.
    Boundaries,
}

impl FvarLinear {
    /// The option for a validated `uv_smooth` value.
    pub(super) fn from_uv_smooth(uv_smooth: &str) -> Self {
        match uv_smooth {
            "NONE" => Self::All,
            "SMOOTH_ALL" => Self::SmoothAll,
            "PRESERVE_CORNERS" => Self::CornersOnly,
            "PRESERVE_CORNERS_AND_JUNCTIONS" => Self::CornersPlus1,
            "PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE" => Self::CornersPlus2,
            _ => Self::Boundaries,
        }
    }
}

/// One UV layer on the base mesh: the value of every base corner, the position vertex of every value,
/// and the UV of every value. A position vertex has one value per UV island it touches.
pub(super) struct ValueMap {
    /// `corners[f][k]` is the value of corner `k` of base face `f`.
    pub(super) corners: Vec<Vec<usize>>,
    /// The position vertex each value belongs to.
    pub(super) vertex: Vec<usize>,
    /// The UV of each value: the UV of its first corner.
    pub(super) uvs: Vec<DVec2>,
}

/// A UV layer's name and value map.
pub(super) struct UvLayerValues {
    pub(super) name: String,
    pub(super) map: ValueMap,
}

/// Position topology of one level, shared by geometry and all face-varying layers.
pub(super) struct Position {
    pub(super) starts: Vec<usize>,
    pub(super) face_of: Vec<usize>,
    pub(super) corner_vertex: Vec<usize>,
    pub(super) edge_starts: Vec<usize>,
    pub(super) edge_sides: Vec<usize>,
    pub(super) edge_ends: Vec<(usize, usize)>,
    pub(super) edge_order: Vec<usize>,
    pub(super) edge_sharp: Vec<f64>,
    pub(super) vertex_starts: Vec<usize>,
    pub(super) vertex_corners: Vec<usize>,
    pub(super) vertex_edge_starts: Vec<usize>,
    pub(super) vertex_edges: Vec<usize>,
    pub(super) non_manifold_corners: Vec<bool>,
}

/// Groups the indices `0..keys.len()` by `keys[index]` into `buckets` buckets: bucket `b` holds
/// `members[starts[b]..starts[b + 1]]`, in ascending index order.
fn group(keys: &[usize], buckets: usize) -> (Vec<usize>, Vec<usize>) {
    let mut starts = vec![0; buckets + 1];
    for &key in keys {
        starts[key + 1] += 1;
    }
    let mut running = 0;
    for start in &mut starts {
        running += *start;
        *start = running;
    }
    let mut cursor = starts.clone();
    let mut members = vec![0; keys.len()];
    for (index, &key) in keys.iter().enumerate() {
        members[cursor[key]] = index;
        cursor[key] += 1;
    }
    (starts, members)
}

impl Position {
    /// Position topology shared by the geometry and every face-varying channel at this level.
    pub(super) fn new(
        polys: &[Polygon],
        loose_edges: &[[usize; 2]],
        creases: &BTreeMap<(usize, usize), f64>,
        vertex_count: usize,
    ) -> Self {
        let non_manifold = NonManifold::new(
            vertex_count,
            polys.iter().map(|(face, _)| face.as_slice()),
        );
        let sharpness = non_manifold.sharpened(creases);
        let total: usize = polys.iter().map(|(face, _)| face.len()).sum();
        let mut starts = Vec::with_capacity(polys.len() + 1);
        let mut face_of = Vec::with_capacity(total);
        let mut corner_vertex = Vec::with_capacity(total);
        starts.push(0);
        for (face_index, (poly, _)) in polys.iter().enumerate() {
            corner_vertex.extend_from_slice(poly);
            face_of.extend(std::iter::repeat_n(face_index, poly.len()));
            starts.push(corner_vertex.len());
        }

        let mut edge_index: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(total / 2 + loose_edges.len());
        let mut edge_ends = Vec::with_capacity(total / 2 + loose_edges.len());
        let mut edge_sharp = Vec::new();
        let mut corner_edge = Vec::with_capacity(total);
        for (poly, _) in polys {
            for corner in 0..poly.len() {
                let key = edge_key(poly[corner], poly[(corner + 1) % poly.len()]);
                let edge = *edge_index.entry(key).or_insert_with(|| {
                    edge_ends.push(key);
                    edge_sharp.push(sharpness.get(&key).copied().unwrap_or(0.0));
                    edge_ends.len() - 1
                });
                corner_edge.push(edge);
            }
        }
        for &[first, second] in loose_edges {
            let key = edge_key(first, second);
            edge_index.entry(key).or_insert_with(|| {
                edge_ends.push(key);
                edge_sharp.push(sharpness.get(&key).copied().unwrap_or(0.0));
                edge_ends.len() - 1
            });
        }
        let (edge_starts, edge_sides) = group(&corner_edge, edge_ends.len());
        let (vertex_starts, vertex_corners) = group(&corner_vertex, vertex_count);
        let endpoints = edge_ends
            .iter()
            .flat_map(|(first, second)| [*first, *second])
            .collect::<Vec<_>>();
        let (vertex_edge_starts, endpoint_slots) = group(&endpoints, vertex_count);
        let vertex_edges: Vec<usize> = endpoint_slots.into_iter().map(|slot| slot / 2).collect();
        let mut edge_order: Vec<usize> = (0..edge_ends.len()).collect();
        edge_order.sort_unstable_by_key(|edge| edge_ends[*edge]);

        Self {
            starts,
            face_of,
            corner_vertex,
            edge_starts,
            edge_sides,
            edge_ends,
            edge_order,
            edge_sharp,
            vertex_starts,
            vertex_corners,
            vertex_edge_starts,
            vertex_edges,
            non_manifold_corners: (0..vertex_count)
                .map(|vertex| non_manifold.corner(vertex))
                .collect(),
        }
    }

    fn corner_count(&self) -> usize {
        self.corner_vertex.len()
    }

    fn corners_at(&self, vertex: usize) -> &[usize] {
        &self.vertex_corners[self.vertex_starts[vertex]..self.vertex_starts[vertex + 1]]
    }
}


/// Blender's UV connection tolerance (`STD_UV_CONNECT_LIMIT`).
const UV_CONNECT_LIMIT: f64 = 0.0001;

/// Whether a face's UVs wind negatively, by the trapezium rule Blender's `cross_poly_v2` uses.
fn uv_winding(uvs: &[DVec2]) -> bool {
    let Some(&last) = uvs.last() else {
        return false;
    };
    let mut previous = last;
    let mut cross = 0.0;
    for uv in uvs {
        cross += (previous.x - uv.x) * (uv.y + previous.y);
        previous = *uv;
    }
    cross < 0.0
}

/// The value map of one layer. Corners at a position vertex share a value when their UVs connect and
/// their face windings agree (Blender's `calc_uv_verts` with winding), grouped greedily in corner
/// order. A value keeps the UV of its first corner.
fn value_map(position: &Position, corner_uv: &[DVec2]) -> ValueMap {
    let winding: Vec<bool> = position
        .starts
        .windows(2)
        .map(|range| uv_winding(&corner_uv[range[0]..range[1]]))
        .collect();
    let mut value_of = vec![usize::MAX; corner_uv.len()];
    let mut vertex = Vec::new();
    let mut uvs = Vec::new();
    for position_vertex in 0..position.vertex_starts.len() - 1 {
        let at_vertex = position.corners_at(position_vertex);
        for (index, &corner) in at_vertex.iter().enumerate() {
            if value_of[corner] != usize::MAX {
                continue;
            }
            let value = uvs.len();
            let uv = corner_uv[corner];
            uvs.push(uv);
            vertex.push(position_vertex);
            value_of[corner] = value;
            for &other in &at_vertex[index + 1..] {
                let other_uv = corner_uv[other];
                if value_of[other] == usize::MAX
                    && (other_uv.x - uv.x).abs() < UV_CONNECT_LIMIT
                    && (other_uv.y - uv.y).abs() < UV_CONNECT_LIMIT
                    && winding[position.face_of[other]] == winding[position.face_of[corner]]
                {
                    value_of[other] = value;
                }
            }
        }
    }
    let corners = position
        .starts
        .windows(2)
        .map(|range| value_of[range[0]..range[1]].to_vec())
        .collect();
    ValueMap {
        corners,
        vertex,
        uvs,
    }
}

/// One layer's corner UVs in global corner order, zero where the layer has no value.
fn corner_uvs(position: &Position, layer: &UvLayer) -> Vec<DVec2> {
    let mut uvs = vec![DVec2::ZERO; position.corner_count()];
    for (face, values) in layer.faces.iter().enumerate() {
        if let Some(values) = values {
            let start = position.starts[face];
            for (corner, uv) in values.iter().enumerate() {
                uvs[start + corner] = DVec2::new(uv[0], uv[1]);
            }
        }
    }
    uvs
}

/// The value maps of every UV layer of `mesh`, over the level-0 position topology.
pub(super) fn value_maps(mesh: &Mesh, position: &Position) -> Vec<UvLayerValues> {
    uv_layers::read_layers(mesh)
        .into_iter()
        .map(|layer| UvLayerValues {
            map: value_map(position, &corner_uvs(position, &layer)),
            name: layer.name,
        })
        .collect()
}

/// Writes one UV layer per entry on the output mesh. Each `loops` lists the UV of every output loop,
/// in face and corner order.
pub(super) fn write_loop_uvs(mesh: &mut Mesh, layers: Vec<(String, Vec<DVec2>)>) {
    let layers: Vec<UvLayer> = layers
        .into_iter()
        .map(|(name, loops)| {
            let mut rest = loops.as_slice();
            let faces = mesh
                .faces
                .iter()
                .map(|face| {
                    let (own, tail) = rest.split_at(face.vertices.len());
                    rest = tail;
                    Some(own.iter().map(|uv| [uv.x, uv.y]).collect())
                })
                .collect();
            UvLayer { name, faces }
        })
        .collect();
    uv_layers::write_layers(mesh, &layers);
}

/// Blender's crease state of one level. Sharpness decays by one level per subdivision and drives the
/// topology. The canonical value is what Blender writes: every descendant of a creased edge carries
/// the input value at every level, even after its sharpness has decayed to zero.
#[derive(Clone, Debug, Default)]
pub(super) struct Creases {
    pub(super) sharpness: BTreeMap<(usize, usize), f64>,
    pub(super) canonical: BTreeMap<(usize, usize), f64>,
}

impl Creases {
    /// Crease state of a mesh from the canonical values (`crease_edge`, in [0, 1]) of its edges.
    pub(super) fn from_values(canonical: BTreeMap<(usize, usize), f64>) -> Creases {
        let sharpness = canonical
            .iter()
            .map(|(&edge, &value)| (edge, blender_edge_crease_sharpness(value)))
            .collect();
        Creases {
            sharpness,
            canonical,
        }
    }

    /// The next level's crease state. Each edge splits at its midpoint into two halves that inherit
    /// the canonical value, and the decayed sharpness of the halves that remain sharp.
    pub(super) fn subdivide(&self, edge_points: &BTreeMap<(usize, usize), usize>) -> Creases {
        let mut next = Creases::default();
        for (&edge, &value) in &self.canonical {
            let Some(&midpoint) = edge_points.get(&edge) else {
                continue;
            };
            next.canonical.insert(edge_key(edge.0, midpoint), value);
            next.canonical.insert(edge_key(midpoint, edge.1), value);
        }
        for (&edge, &sharpness) in &self.sharpness {
            let decayed = if sharpness >= 10.0 {
                10.0
            } else if sharpness > 1.0 {
                sharpness - 1.0
            } else {
                continue;
            };
            let Some(&midpoint) = edge_points.get(&edge) else {
                continue;
            };
            next.sharpness.insert(edge_key(edge.0, midpoint), decayed);
            next.sharpness.insert(edge_key(midpoint, edge.1), decayed);
        }
        next
    }
}

pub(super) fn subdivide_simple_once(
    mesh: &Mesh,
    position: &Position,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    creases: &Creases,
    use_creases: bool,
) -> Result<(Mesh, Creases, BTreeMap<(usize, usize), usize>)> {
    let source_points = positions(mesh);
    let face_points: Vec<DVec3> = source_polygons
        .iter()
        .map(|(face, _)| {
            face.iter()
                .fold(DVec3::ZERO, |sum, index| sum + source_points[*index])
                / usize_as_f64(face.len())
        })
        .collect();
    let mut points = source_points.clone();
    let mut edge_point_indices = BTreeMap::new();
    for &edge_index in &position.edge_order {
        let edge = position.edge_ends[edge_index];
        edge_point_indices.insert(edge, points.len());
        points.push((source_points[edge.0] + source_points[edge.1]) * 0.5);
    }
    points.extend(face_points);
    let next = creases.subdivide(&edge_point_indices);
    let subdivided = subdivided_mesh(
        mesh,
        points,
        source_polygons,
        source_edges,
        &edge_point_indices,
        &next.canonical,
        use_creases,
    )?;
    Ok((subdivided, next, edge_point_indices))
}

/// One subdivision level's mesh from refined points, shared by SIMPLE and Catmull-Clark. `points` holds
/// the refined vertices, then one point per edge in `edge_points` key order, then one per source face.
/// Every source corner becomes one quad `[corner, next edge point, face point, previous edge point]`,
/// faces in source order. Loose edges split at their edge point, and each output vertex records its
/// source origin. `output_creases` maps output edge endpoints to Blender's canonical crease value.
pub(super) fn subdivided_mesh(
    mesh: &Mesh,
    points: Vec<DVec3>,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    edge_points: &BTreeMap<(usize, usize), usize>,
    output_creases: &BTreeMap<(usize, usize), f64>,
    use_creases: bool,
) -> Result<Mesh> {
    let face_point_base = points.len() - source_polygons.len();
    let mut output_polygons =
        Vec::with_capacity(source_polygons.iter().map(|(face, _)| face.len()).sum());
    for (face_index, (face, material)) in source_polygons.iter().enumerate() {
        for corner in 0..face.len() {
            let previous = face[(corner + face.len() - 1) % face.len()];
            let current = face[corner];
            let next = face[(corner + 1) % face.len()];
            output_polygons.push((
                vec![
                    current,
                    edge_points[&edge_key(current, next)],
                    face_point_base + face_index,
                    edge_points[&edge_key(previous, current)],
                ],
                *material,
            ));
        }
    }
    let mut output_loose_edges = Vec::with_capacity(source_edges.len() * 2);
    for &[first, second] in source_edges {
        let midpoint = edge_points[&edge_key(first, second)];
        output_loose_edges.push([first, midpoint]);
        output_loose_edges.push([midpoint, second]);
    }

    let mut vertex_origins: Vec<Vec<(usize, f64)>> = (0..mesh.vertices.len())
        .map(|vertex| vec![(vertex, 1.0)])
        .collect();
    vertex_origins.extend(
        edge_points
            .keys()
            .map(|edge| vec![(edge.0, 0.5), (edge.1, 0.5)]),
    );
    vertex_origins.extend(source_polygons.iter().map(|(face, _)| {
        let share = 1.0 / usize_as_f64(face.len());
        face.iter().map(|&vertex| (vertex, share)).collect()
    }));
    let origin = Origin {
        vertices: vertex_origins,
        faces: source_polygons
            .iter()
            .enumerate()
            .flat_map(|(index, (face, _))| std::iter::repeat_n(Some(index), face.len()))
            .collect(),
        corners: None,
    };
    let mut output = build_mesh(points, output_polygons, output_loose_edges, &origin, mesh)?;
    if use_creases && mesh.attributes.contains_key("crease_edge") {
        let index_of: HashMap<u32, usize> = output
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        let mut values = Map::new();
        for edge in &output.edges {
            let key = edge_key(index_of[&edge.vertices[0]], index_of[&edge.vertices[1]]);
            if let Some(&value) = output_creases.get(&key) {
                values.insert(format!("e{}", edge.id), json!(value));
            }
        }
        output.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain":"edge","type":"float","values":values}),
        );
    }
    Ok(output)
}
