// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/subdiv_converter_mesh.cc, source/blender/modifiers/intern/MOD_subsurf.cc, SPDX-FileCopyrightText: 2018 Blender Authors, GPL-2.0-or-later.
//   OpenSubdiv (https://graphics.pixar.com/opensubdiv) vtr/fvarLevel.cpp, far/primvarRefiner.h, Apache-2.0.

//! Blender's UV layers and the position topology they share. A UV layer's values are Blender's UV
//! vertices: a position vertex splits wherever its UVs differ between faces. Each layer's limit is
//! evaluated by `subdivision_limit::face_varying`, on the leaves of the position limit surface.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use glam::{DVec2, DVec3};
use serde_json::{Map, Value, json};

use super::{
    NonManifold, Polygon, blender_edge_crease_sharpness, build_mesh, positions, usize_as_f64,
};
use crate::{
    color::{linear_to_srgb, srgb_to_linear},
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        color_layers::{self, ColorDomain},
        edge_key,
        provenance::{Origin, corner_sources},
        uv_layers::{self, UvLayer},
    },
};

/// Face-varying interpolation of a UV layer, from `uv_smooth` (`OpenSubdiv` `FVarLinearInterpolation`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum FvarLinear {
    /// `uv_smooth` NONE: every edge and value is linear, so the UVs interpolate bilinearly.
    All,
    /// `uv_smooth` `SMOOTH_ALL`: smooth everywhere, UV boundaries included.
    SmoothAll,
    /// `uv_smooth` `PRESERVE_CORNERS`: UV corners stay put.
    CornersOnly,
    /// `uv_smooth` `PRESERVE_CORNERS_AND_JUNCTIONS`: corners and junctions stay put.
    CornersPlus1,
    /// `uv_smooth` `PRESERVE_CORNERS_JUNCTIONS_AND_CONCAVE`: corners, junctions, and concave corners.
    CornersPlus2,
    /// `uv_smooth` `PRESERVE_BOUNDARIES` (Blender's default): UV boundaries stay put.
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
    /// The UV of each value: that of its last corner in face order, the one Blender's upload
    /// writes last.
    pub(super) uvs: Vec<DVec2>,
}

/// A UV layer's name, value map, and face-local pin state.
pub(super) struct UvLayerValues {
    pub(super) name: String,
    pub(super) map: ValueMap,
    pub(super) pins: Vec<Option<Vec<bool>>>,
}

/// Position topology of one level, shared by geometry and all face-varying layers.
pub(super) struct Position {
    pub(super) starts: Vec<usize>,
    pub(super) face_of: Vec<usize>,
    pub(super) corner_vertex: Vec<usize>,
    pub(super) edge_ends: Vec<(usize, usize)>,
    pub(super) edge_order: Vec<usize>,
    pub(super) edge_sharp: Vec<f64>,
    pub(super) vertex_starts: Vec<usize>,
    pub(super) vertex_corners: Vec<usize>,
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
        let non_manifold =
            NonManifold::new(vertex_count, polys.iter().map(|(face, _)| face.as_slice()));
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

        let mut seen: HashSet<(usize, usize)> =
            HashSet::with_capacity(total / 2 + loose_edges.len());
        let mut edge_ends = Vec::with_capacity(total / 2 + loose_edges.len());
        let mut edge_sharp = Vec::new();
        let mut add_edge = |key: (usize, usize)| {
            if seen.insert(key) {
                edge_ends.push(key);
                edge_sharp.push(sharpness.get(&key).copied().unwrap_or(0.0));
            }
        };
        for (poly, _) in polys {
            for corner in 0..poly.len() {
                add_edge(edge_key(poly[corner], poly[(corner + 1) % poly.len()]));
            }
        }
        for &[first, second] in loose_edges {
            add_edge(edge_key(first, second));
        }
        let (vertex_starts, vertex_corners) = group(&corner_vertex, vertex_count);
        let mut edge_order: Vec<usize> = (0..edge_ends.len()).collect();
        edge_order.sort_unstable_by_key(|edge| edge_ends[*edge]);

        Self {
            starts,
            face_of,
            corner_vertex,
            edge_ends,
            edge_order,
            edge_sharp,
            vertex_starts,
            vertex_corners,
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
/// their face windings agree (Blender's `BKE_mesh_uv_vert_map_create` with winding). Groups seed from
/// the last corner at each vertex, as Blender's prepended vertex list does. A value's UV is that of
/// its last corner in face order, which Blender's face-varying upload writes last.
fn value_map(position: &Position, corner_uv: &[DVec2]) -> ValueMap {
    let winding: Vec<bool> = position
        .starts
        .windows(2)
        .map(|range| uv_winding(&corner_uv[range[0]..range[1]]))
        .collect();
    let mut value_of = vec![usize::MAX; corner_uv.len()];
    let mut vertex = Vec::new();
    for position_vertex in 0..position.vertex_starts.len() - 1 {
        let at_vertex = position.corners_at(position_vertex);
        for (index, &corner) in at_vertex.iter().enumerate().rev() {
            if value_of[corner] != usize::MAX {
                continue;
            }
            let value = vertex.len();
            let uv = corner_uv[corner];
            vertex.push(position_vertex);
            value_of[corner] = value;
            for &other in at_vertex[..index].iter().rev() {
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
    let mut uvs = vec![DVec2::ZERO; vertex.len()];
    for (corner, &value) in value_of.iter().enumerate() {
        uvs[value] = corner_uv[corner];
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

/// One layer's corner UVs in global corner order, zero where the layer has no value. UVs are
/// converted to Blender's V-up convention.
fn corner_uvs(position: &Position, layer: &UvLayer) -> Vec<DVec2> {
    let mut uvs = vec![DVec2::ZERO; position.corner_count()];
    for (face, values) in layer.faces.iter().enumerate() {
        if let Some(values) = values {
            let start = position.starts[face];
            for (corner, uv) in values.iter().enumerate() {
                // The mesh UV store is glTF V-down; Blender's face-varying math is V-up.
                uvs[start + corner] = DVec2::new(uv[0], 1.0 - uv[1]);
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
            pins: layer.pinned,
        })
        .collect()
}

/// Blender's face-local boolean corner interpolation: any source pin with a positive weight pins
/// the child corner. Rows stay in the same face/corner order as subdivision topology refinement.
pub(super) fn refine_pins(
    pins: Vec<Option<Vec<bool>>>,
    starts: &[usize],
) -> Vec<Option<Vec<bool>>> {
    let mut child = Vec::with_capacity(starts.last().copied().unwrap_or_default());
    for (face, row) in pins.into_iter().enumerate() {
        let (start, end) = (starts[face], starts[face + 1]);
        let Some(row) = row else {
            child.extend(std::iter::repeat_n(None, end - start));
            continue;
        };
        let face_pin = row.iter().any(|pinned| *pinned);
        for corner in 0..end - start {
            let next = (corner + 1) % (end - start);
            let previous = if corner == 0 {
                end - start - 1
            } else {
                corner - 1
            };
            child.push(Some(vec![
                row[corner],
                row[corner] || row[next],
                face_pin,
                row[previous] || row[corner],
            ]));
        }
    }
    child
}

type UvLayerInput = (String, Vec<DVec2>, Vec<Option<Vec<bool>>>);

/// Writes one UV layer per entry. UV values are loop-ordered; pin rows are face-ordered.
pub(super) fn write_loop_uvs(mesh: &mut Mesh, layers: Vec<UvLayerInput>) -> Result<()> {
    let mut written_layers = Vec::with_capacity(layers.len());
    for (name, loops, pinned) in layers {
        if pinned.len() != mesh.faces.len() {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "subdivision UV pin rows do not match the output faces",
            ));
        }
        let mut rest = loops.as_slice();
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                let (own, tail) = rest.split_at(face.vertices.len());
                rest = tail;
                Some(own.iter().map(|uv| [uv.x, 1.0 - uv.y]).collect())
            })
            .collect();
        let mut layer = UvLayer::new(name, faces);
        layer.pinned = pinned;
        written_layers.push(layer);
    }
    uv_layers::write_layers(mesh, &written_layers);
    Ok(())
}

/// The pins of layer `name` that the rebuild carried onto `faces`. A face keeps its pins when they
/// cover its corners; otherwise it has none.
pub(super) fn carried_pins(
    carried: &[UvLayer],
    name: &str,
    faces: &[Option<Vec<[f64; 2]>>],
) -> Vec<Option<Vec<bool>>> {
    let known = carried.iter().find(|layer| layer.name == name);
    faces
        .iter()
        .enumerate()
        .map(|(face, uv)| {
            let corners = uv.as_ref()?.len();
            known?
                .pinned
                .get(face)?
                .clone()
                .filter(|pins| pins.len() == corners)
        })
        .collect()
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

pub(super) type SimpleSubdivision = (Mesh, Creases, BTreeMap<(usize, usize), usize>);

pub(super) fn subdivide_simple_once(
    mesh: &Mesh,
    position: &Position,
    source_polygons: &[Polygon],
    source_edges: &[[usize; 2]],
    creases: &Creases,
    use_creases: bool,
) -> Result<SimpleSubdivision> {
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
/// `BYTE_COLOR` layers are re-blended from their sources' bytes (see [`encode_byte_colors`]).
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
    encode_byte_colors(mesh, &origin, &mut output);
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

/// Blender interpolates a `BYTE_COLOR` layer in its stored bytes: RGB holds sRGB-encoded bytes and alpha is
/// linear. Each subdivided value mixes its sources' bytes and truncates the sum to a byte, so the result is
/// not a linear mix. Potter's colour store is linear, so every byte-typed layer is re-blended here from the
/// same origins the attribute remap used, and float layers keep the linear remap.
fn encode_byte_colors(source: &Mesh, origin: &Origin, output: &mut Mesh) {
    let Some(types) = source
        .attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(Value::as_object)
    else {
        return;
    };
    let byte_layers: BTreeSet<&str> = types
        .iter()
        .filter(|(_, data_type)| data_type.as_str() == Some("BYTE_COLOR"))
        .map(|(name, _)| name.as_str())
        .collect();
    if byte_layers.is_empty() {
        return;
    }
    let source_store = color_layers::read(source);
    let mut store = color_layers::read(output);
    let output_vertex_index: HashMap<u32, usize> = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let corners = corner_sources(source, origin, output, &output_vertex_index);
    for layer in &mut store.layers {
        if !byte_layers.contains(layer.name.as_str()) {
            continue;
        }
        let Some(source_layer) = source_store
            .layers
            .iter()
            .find(|candidate| candidate.name == layer.name)
        else {
            continue;
        };
        let sources = match layer.domain {
            ColorDomain::Point => origin.vertices.as_slice(),
            ColorDomain::Corner => corners.as_slice(),
        };
        layer.values = sources
            .iter()
            .map(|sources| mix_byte_color(&source_layer.values, sources))
            .collect();
    }
    color_layers::write(output, &store);
}

/// One output value of a `BYTE_COLOR` layer: each source's stored byte times its origin weight, summed and
/// truncated to a byte per channel, then decoded back to linear. A value with no source is opaque black, as
/// the attribute remap's default.
fn mix_byte_color(values: &[[f64; 4]], sources: &[(usize, f64)]) -> [f64; 4] {
    let mut bytes = [0.0_f64; 4];
    let mut found = false;
    for (index, weight) in sources {
        let Some(value) = values.get(*index) else {
            continue;
        };
        found = true;
        for (channel, component) in value.iter().enumerate() {
            let encoded = if channel < 3 {
                linear_to_srgb(*component)
            } else {
                *component
            };
            bytes[channel] += weight * (encoded * 255.0).round();
        }
    }
    if !found {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let mut color = [0.0; 4];
    for (channel, &byte) in bytes.iter().enumerate() {
        let unit = (byte + 1.0e-9).floor().clamp(0.0, 255.0) / 255.0;
        color[channel] = if channel < 3 {
            srgb_to_linear(unit)
        } else {
            unit
        };
    }
    color
}
