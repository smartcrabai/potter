// SPDX-License-Identifier: GPL-3.0-or-later

//! Element provenance for mesh-rebuilding modifiers.
//!
//! A modifier that rebuilds topology reports where each output vertex and polygon came from.
//! Edges and polygon corners are derived from those origins, so [`remap_attributes`] is the
//! single place that rewrites ID-keyed attributes after a rebuild. Vertex origins are weighted
//! so interpolated data (vertex groups, POINT attributes) blends like Blender's `CustomData`.

use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use glam::DVec3;
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::normals::{face_normal, vector3};
use super::uv_layers::{uv_default, uv_map_entry_layer};
use super::{Mesh, edge_key};

/// Where each element of a rebuilt mesh came from, indexed by output storage order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Origin {
    /// Per output vertex: input vertex storage indices with weights summing to one.
    /// An empty list marks a new vertex without input data.
    pub vertices: Vec<Vec<(usize, f64)>>,
    /// Per output polygon, in the order passed to `build_mesh`: the input polygon storage index,
    /// or `None` for a generated polygon.
    pub faces: Vec<Option<usize>>,
    /// Explicit source corners per output corner, in output order, as `(input corner index, weight)`
    /// with the input corner index counted as in [`corner_offsets`]. `None` derives corner sources
    /// from the vertex origins restricted to each output face's source polygon.
    pub corners: Option<Vec<Vec<(usize, f64)>>>,
    /// Keep blended custom normals unnormalised. Subdivision sets this on every level but normalises
    /// once at the end ([`normalize_custom_normals`]): Blender evaluates all levels from the base
    /// corners, and blending normalised levels would compound the normalisation.
    pub raw_normals: bool,
}

impl Origin {
    /// Origin where each output vertex copies one input vertex, or is new when `None`.
    pub(crate) fn copies(
        vertices: impl IntoIterator<Item = Option<usize>>,
        faces: Vec<Option<usize>>,
    ) -> Self {
        Self {
            vertices: vertices
                .into_iter()
                .map(|source| source.map(|index| vec![(index, 1.0)]).unwrap_or_default())
                .collect(),
            faces,
            corners: None,
            raw_normals: false,
        }
    }
}

/// Rewrite the ID-keyed attributes of `output` (rebuilt from `source`) through `origin`.
///
/// `vertex_groups`, `crease_edge`, skin vertex keys, `uv_map`, every `blender_uv_layers` layer, and
/// every `blender_attributes` entry on the POINT, EDGE, FACE, or CORNER domain are remapped. Corner
/// data blends over the source corners of each output corner (see [`corner_sources`]), so UVs follow
/// the face corner they came from. Shading follows the same origins: `shade_smooth` and
/// `blender_polygon_smooth` copy their source face's flag, `sharp` and `blender_edge_flags.sharp` OR
/// over the source edges, and `custom_normal` blends the custom normals of the source corners. Elements
/// without a source take the default (false, or no custom normal), so no key refers to a stale element.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes(source: &Mesh, output: &mut Mesh, origin: &Origin) -> Result<()> {
    if origin.vertices.len() != output.vertices.len() || origin.faces.len() != output.faces.len() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "modifier provenance does not describe the rebuilt mesh",
        ));
    }
    if source.attributes.is_empty() {
        output.attributes.clear();
        return Ok(());
    }
    let entries = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let needs_edges = source.attributes.contains_key("crease_edge")
        || source.attributes.contains_key("blender_edge_flags")
        || source.attributes.contains_key("sharp")
        || entries.iter().any(|entry| text(entry, "domain") == "EDGE");
    let needs_corners = source.attributes.contains_key("blender_uv_layers")
        || source.attributes.contains_key("uv_map")
        || source.attributes.contains_key("custom_normal")
        || entries
            .iter()
            .any(|entry| text(entry, "domain") == "CORNER");
    let mut edge_origin = Vec::new();
    if needs_edges || needs_corners {
        let output_vertex_index: HashMap<u32, usize> = output
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        if needs_edges {
            edge_origin = edge_sources(source, output, origin, &output_vertex_index);
        }
    }
    let derived_corners;
    let corner_origin: &[Vec<(usize, f64)>] = match &origin.corners {
        Some(corners) => corners,
        None if needs_corners => {
            let output_vertex_index: HashMap<u32, usize> = output
                .vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect();
            derived_corners = corner_sources(source, origin, output, &output_vertex_index);
            &derived_corners
        }
        None => &[],
    };
    let sources = Sources {
        source,
        output,
        origin,
        edge_origin: &edge_origin,
        corner_origin,
        raw_normals: origin.raw_normals,
    };
    let mut attributes = Map::new();
    for (key, value) in &source.attributes {
        if key == "custom_normal" {
            // Dropped when no output corner has an authored normal, which reports attribute_dropped.
            if let Some(remapped) = sources.custom_normal(value) {
                attributes.insert(key.clone(), remapped);
            }
            continue;
        }
        let remapped = match key.as_str() {
            "vertex_groups" => value
                .as_object()
                .map(|groups| Value::Object(sources.vertex_groups(groups))),
            "crease_edge" => Some(sources.keyed_edge_attribute(value)),
            "blender_edge_flags" => value
                .as_array()
                .map(|flags| Value::Array(sources.edge_flags(flags))),
            "blender_uv_layers" => value.as_array().map(|layers| {
                Value::Array(layers.iter().map(|layer| sources.uv_layer(layer)).collect())
            }),
            "uv_map" => value
                .as_array()
                .map(|entries| Value::Array(sources.uv_map(entries))),
            "blender_attributes" => value.as_array().map(|entries| {
                Value::Array(
                    entries
                        .iter()
                        .map(|entry| sources.attribute_entry(entry))
                        .collect(),
                )
            }),
            "skin_radii" | "skin_roots" => value
                .as_object()
                .map(|values| Value::Object(sources.vertex_object(values, "FLOAT"))),
            "blender_skin_vertices" => value.as_array().map(|vertices| {
                Value::Array(sources.vertex_values(vertices, "FLOAT", || json!({})))
            }),
            "shade_smooth" => Some(sources.shade_smooth(value)),
            "sharp" => Some(sources.sharp(value)),
            "blender_polygon_smooth" => value
                .as_array()
                .map(|flags| Value::Array(sources.polygon_smooth(flags))),
            _ => None,
        };
        attributes.insert(key.clone(), remapped.unwrap_or_else(|| value.clone()));
    }
    output.attributes = attributes;
    Ok(())
}

fn corner_offsets(face_sizes: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut offset = 0;
    face_sizes
        .map(|size| {
            let start = offset;
            offset += size;
            start
        })
        .collect()
}

/// The string field `key` of a JSON entry, or `""`.
fn text<'a>(entry: &'a Value, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// `entry` with its `values` replaced; the old values array is not cloned.
fn with_values(entry: &Value, values: Value) -> Value {
    let mut fields = Map::new();
    for (key, field) in entry.as_object().into_iter().flatten() {
        if key.as_str() != "values" {
            fields.insert(key.clone(), field.clone());
        }
    }
    fields.insert("values".to_owned(), values);
    Value::Object(fields)
}

/// Source edges each output edge blends from, with equal weights. Candidates are the distinct input
/// edges joining a source vertex of the output edge's first endpoint to a source vertex of its
/// second. When the endpoints share a source vertex (a subdivision inner edge, for example), a
/// candidate counts only if its two vertices are every source of both endpoints; otherwise the
/// output edge is a new span and gets no source. Blender averages merged duplicate edges, so
/// several candidates blend instead of falling back to default.
fn edge_sources(
    source: &Mesh,
    output: &Mesh,
    origin: &Origin,
    output_vertex_index: &HashMap<u32, usize>,
) -> Vec<Vec<(usize, f64)>> {
    let source_vertex_index: HashMap<u32, usize> = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let source_edge_index: HashMap<(usize, usize), usize> = source
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            let first = *source_vertex_index.get(&edge.vertices[0])?;
            let second = *source_vertex_index.get(&edge.vertices[1])?;
            Some((edge_key(first, second), index))
        })
        .collect();
    output
        .edges
        .iter()
        .map(|edge| {
            let Some(first) = output_vertex_index
                .get(&edge.vertices[0])
                .and_then(|index| origin.vertices.get(*index))
            else {
                return Vec::new();
            };
            let Some(second) = output_vertex_index
                .get(&edge.vertices[1])
                .and_then(|index| origin.vertices.get(*index))
            else {
                return Vec::new();
            };
            let shared = first
                .iter()
                .any(|(a, _)| second.iter().any(|(b, _)| a == b));
            let endpoint_sources: Vec<usize> = first
                .iter()
                .chain(second)
                .map(|(index, _)| *index)
                .collect();
            let mut found: Vec<usize> = Vec::new();
            for (first_source, _) in first {
                for (second_source, _) in second {
                    let Some(index) =
                        source_edge_index.get(&edge_key(*first_source, *second_source))
                    else {
                        continue;
                    };
                    let spans_all = endpoint_sources
                        .iter()
                        .all(|source| source == first_source || source == second_source);
                    if (!shared || spans_all) && !found.contains(index) {
                        found.push(*index);
                    }
                }
            }
            let share = equal_share(found.len());
            found.into_iter().map(|index| (index, share)).collect()
        })
        .collect()
}

/// Source corners each output corner blends from. An output corner of output face `f` (source face
/// `sf`) on output vertex `v` takes the input vertices of `v` that belong to `sf`, weighted by their
/// origin weights renormalised over those vertices, each mapped to its corner in `sf`. A corner with
/// no such vertex, or on a generated face, has no sources and takes the domain default.
fn corner_sources(
    source: &Mesh,
    origin: &Origin,
    output: &Mesh,
    output_vertex_index: &HashMap<u32, usize>,
) -> Vec<Vec<(usize, f64)>> {
    let offsets = corner_offsets(source.faces.iter().map(|face| face.vertices.len()));
    let mut corners = Vec::new();
    for (face, source_face) in output.faces.iter().zip(&origin.faces) {
        let source_face =
            source_face.and_then(|index| source.faces.get(index).map(|face| (index, face)));
        for vertex_id in &face.vertices {
            let vertex_sources = output_vertex_index
                .get(vertex_id)
                .and_then(|index| origin.vertices.get(*index));
            let mut kept = Vec::new();
            if let (Some((index, source_face)), Some(vertex_sources)) =
                (source_face, vertex_sources)
            {
                for (vertex, weight) in vertex_sources {
                    let Some(id) = source.vertices.get(*vertex).map(|vertex| vertex.id) else {
                        continue;
                    };
                    if let Some(position) =
                        source_face.vertices.iter().position(|corner| *corner == id)
                    {
                        kept.push((offsets[index] + position, *weight));
                    }
                }
            }
            let total: f64 = kept.iter().map(|(_, weight)| *weight).sum();
            if total > 0.0 {
                for (_, weight) in &mut kept {
                    *weight /= total;
                }
            } else {
                kept.clear();
            }
            corners.push(kept);
        }
    }
    corners
}

/// Corner sources by Blender's mean-value interpolation. A corner of an output face whose source
/// polygon is `Some(sf)` takes the corners of `sf` weighted by the mean-value coordinates of its
/// output vertex, projected onto `sf`'s plane. Blender interpolates loops this way for vertices
/// created inside a source face (bevel, boolean cuts). A face without a source polygon has no
/// sources, so its corners take the domain default.
pub(crate) fn mean_value_corners<'a>(
    source: &Mesh,
    points: &[DVec3],
    faces: impl IntoIterator<Item = (&'a [usize], Option<usize>)>,
) -> Vec<Vec<(usize, f64)>> {
    let offsets = corner_offsets(source.faces.iter().map(|face| face.vertices.len()));
    let source_index: HashMap<u32, usize> = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut corners = Vec::new();
    for (vertices, source_face) in faces {
        let Some(index) = source_face.filter(|index| *index < source.faces.len()) else {
            corners.extend(vertices.iter().map(|_| Vec::new()));
            continue;
        };
        let polygon: Option<Vec<DVec3>> = source.faces[index]
            .vertices
            .iter()
            .map(|id| {
                source_index
                    .get(id)
                    .map(|position| source.vertices[*position].co)
            })
            .collect();
        for &vertex in vertices {
            let weights = polygon
                .as_deref()
                .and_then(|polygon| mean_value_weights(polygon, points[vertex]))
                .unwrap_or_default();
            corners.push(
                weights
                    .into_iter()
                    .enumerate()
                    .filter(|(_, weight)| *weight != 0.0)
                    .map(|(corner, weight)| (offsets[index] + corner, weight))
                    .collect(),
            );
        }
    }
    corners
}

/// Mean-value coordinates of `point` over `polygon`, projected onto the polygon's plane (Floater,
/// 2003). A point on a polygon vertex or edge takes that vertex's or edge's linear weights. `None`
/// for a degenerate polygon.
fn mean_value_weights(polygon: &[DVec3], point: DVec3) -> Option<Vec<f64>> {
    const COLLINEAR: f64 = 1.0e-9;
    let count = polygon.len();
    let normal = (0..count)
        .fold(DVec3::ZERO, |sum, index| {
            sum + polygon[index].cross(polygon[(index + 1) % count])
        })
        .normalize_or_zero();
    if normal == DVec3::ZERO {
        return None;
    }
    let plane_point = point - normal * (point - polygon[0]).dot(normal);
    let offsets: Vec<DVec3> = polygon.iter().map(|corner| *corner - plane_point).collect();
    let distances: Vec<f64> = offsets.iter().map(|offset| offset.length()).collect();
    let scale = distances.iter().copied().fold(0.0, f64::max);
    if let Some(index) = distances
        .iter()
        .position(|distance| *distance <= COLLINEAR * scale)
    {
        let mut weights = vec![0.0; count];
        weights[index] = 1.0;
        return Some(weights);
    }
    for index in 0..count {
        let next = (index + 1) % count;
        let sine = offsets[index].cross(offsets[next]).dot(normal);
        let cosine = offsets[index].dot(offsets[next]);
        if sine.abs() <= COLLINEAR * distances[index] * distances[next] && cosine < 0.0 {
            let total = distances[index] + distances[next];
            let mut weights = vec![0.0; count];
            weights[index] = distances[next] / total;
            weights[next] = distances[index] / total;
            return Some(weights);
        }
    }
    let half_tangents: Vec<f64> = (0..count)
        .map(|index| {
            let next = (index + 1) % count;
            let angle = offsets[index]
                .cross(offsets[next])
                .dot(normal)
                .atan2(offsets[index].dot(offsets[next]));
            (angle * 0.5).tan()
        })
        .collect();
    let mut weights: Vec<f64> = (0..count)
        .map(|index| {
            let previous = (index + count - 1) % count;
            (half_tangents[previous] + half_tangents[index]) / distances[index]
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total.abs() <= f64::EPSILON {
        return None;
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Some(weights)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "source element counts are far below 2^53"
)]
fn equal_share(count: usize) -> f64 {
    1.0 / count as f64
}

/// Keyed values as `(index, value)` pairs: keys are `{prefix}{index}` in canonical decimal form.
fn keyed_values<'a, T>(values: &'a Map<String, Value>, prefix: &str) -> HashMap<T, &'a Value>
where
    T: FromStr + Display + Hash + Eq,
{
    values
        .iter()
        .filter_map(|(key, value)| {
            let index: T = key.strip_prefix(prefix)?.parse().ok()?;
            (format!("{prefix}{index}") == *key).then_some((index, value))
        })
        .collect()
}

/// One face's `uv_map` entry: its corner UVs and optional corner pins, borrowed from the entry.
type PerFaceUv<'a> = (&'a Vec<Value>, Option<&'a Vec<Value>>);

struct Sources<'a> {
    raw_normals: bool,
    source: &'a Mesh,
    output: &'a Mesh,
    origin: &'a Origin,
    edge_origin: &'a [Vec<(usize, f64)>],
    corner_origin: &'a [Vec<(usize, f64)>],
}

impl Sources<'_> {
    /// Interpolated vertex-group weights keyed `v{output id}` under each group name.
    fn vertex_groups(&self, groups: &Map<String, Value>) -> Map<String, Value> {
        groups
            .iter()
            .map(|(name, weights)| {
                let by_id: HashMap<u32, f64> = weights
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter_map(|(key, value)| {
                        Some((key.strip_prefix('v')?.parse().ok()?, value.as_f64()?))
                    })
                    .collect();
                let mut output = Map::new();
                for (vertex, sources) in self.origin.vertices.iter().enumerate() {
                    let mut found = false;
                    let mut sum = 0.0;
                    for (index, weight) in sources {
                        if let Some(value) = by_id.get(&self.source.vertices[*index].id) {
                            found = true;
                            sum += weight * value;
                        }
                    }
                    if found {
                        output.insert(format!("v{}", self.output.vertices[vertex].id), json!(sum));
                    }
                }
                (name.clone(), Value::Object(output))
            })
            .collect()
    }

    /// An edge-domain float map (`{"domain","type","values":{"e{id}":v}}`) moved to output edge IDs,
    /// each blended over its source edges (see [`blend`]).
    fn keyed_edge_attribute(&self, attribute: &Value) -> Value {
        let by_id: HashMap<u32, &Value> = attribute
            .get("values")
            .and_then(Value::as_object)
            .map(|values| keyed_values(values, "e"))
            .unwrap_or_default();
        let mut output = Map::new();
        for (edge, sources) in self.output.edges.iter().zip(self.edge_origin) {
            let blended = blend(
                sources.iter().filter_map(|(index, weight)| {
                    let id = self.source.edges.get(*index)?.id;
                    by_id.get(&id).map(|value| (*value, *weight))
                }),
                "FLOAT",
            );
            if let Some(value) = blended {
                output.insert(format!("e{}", edge.id), value);
            }
        }
        with_values(attribute, Value::Object(output))
    }

    /// Edge values blended over each output edge's source edges (see [`blend`]).
    fn edge_values(
        &self,
        values: &[Value],
        data_type: &str,
        default: impl Fn() -> Value,
    ) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|sources| blend_indexed(values, sources, data_type).unwrap_or_else(&default))
            .collect()
    }

    /// `blender_edge_flags` per output edge: `sharp` and `seam` are OR-ed over the positive-weight
    /// source edges, so a merged edge keeps a mark from any input edge it came from. An edge with
    /// no source has neither mark.
    fn edge_flags(&self, flags: &[Value]) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|sources| {
                let mut sharp = false;
                let mut seam = false;
                for (index, _) in sources.iter().filter(|(_, weight)| *weight > 0.0) {
                    let Some(flag) = flags.get(*index) else {
                        continue;
                    };
                    sharp |= flag.get("sharp").and_then(Value::as_bool) == Some(true);
                    seam |= flag.get("seam").and_then(Value::as_bool) == Some(true);
                }
                json!({"sharp": sharp, "seam": seam})
            })
            .collect()
    }

    /// `shade_smooth`: each output face copies its source face's flag, and a generated face is flat.
    fn shade_smooth(&self, attribute: &Value) -> Value {
        let values = attribute.get("values").and_then(Value::as_object);
        let mut output = Map::new();
        for (face, source) in self.output.faces.iter().zip(&self.origin.faces) {
            let flag = source
                .and_then(|index| self.source.faces.get(index))
                .and_then(|source| values?.get(&format!("f{}", source.id)))
                .and_then(Value::as_bool)
                == Some(true);
            output.insert(format!("f{}", face.id), json!(flag));
        }
        with_values(attribute, Value::Object(output))
    }

    /// `blender_polygon_smooth`, positional per face: each output face copies its source face's flag.
    fn polygon_smooth(&self, flags: &[Value]) -> Vec<Value> {
        self.origin
            .faces
            .iter()
            .map(|source| {
                let flag = source
                    .and_then(|index| flags.get(index))
                    .and_then(Value::as_bool);
                json!(flag == Some(true))
            })
            .collect()
    }

    /// `sharp`: an output edge is sharp when any positive-weight source edge is.
    fn sharp(&self, attribute: &Value) -> Value {
        let values = attribute.get("values").and_then(Value::as_object);
        let mut output = Map::new();
        for (edge, sources) in self.output.edges.iter().zip(self.edge_origin) {
            let sharp = sources
                .iter()
                .filter(|(_, weight)| *weight > 0.0)
                .any(|(index, _)| {
                    self.source
                        .edges
                        .get(*index)
                        .and_then(|source| values?.get(&format!("e{}", source.id)))
                        .and_then(Value::as_bool)
                        == Some(true)
                });
            output.insert(format!("e{}", edge.id), json!(sharp));
        }
        with_values(attribute, Value::Object(output))
    }

    /// `custom_normal` (`{"domain","type","values":{"f{id}": [[x, y, z]; corners]}}`) moved to output
    /// corners. Each output corner blends the authored normals of its source corners, weights
    /// renormalised over those that have one, and normalises the result unless `raw_normals`. A face
    /// whose corners have no authored source keeps no entry; `None` when no face does.
    fn custom_normal(&self, attribute: &Value) -> Option<Value> {
        let values = attribute.get("values").and_then(Value::as_object)?;
        let source_offsets =
            corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
        let source_corners: usize = self
            .source
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum();
        let mut authored: Vec<Option<DVec3>> = vec![None; source_corners];
        for (index, face) in self.source.faces.iter().enumerate() {
            let Some(corners) = values
                .get(&format!("f{}", face.id))
                .and_then(Value::as_array)
            else {
                continue;
            };
            for (corner, value) in corners.iter().enumerate().take(face.vertices.len()) {
                authored[source_offsets[index] + corner] = vector3(value);
            }
        }
        let output_offsets =
            corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
        let positions: HashMap<u32, DVec3> = self
            .output
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect();
        let mut result = Map::new();
        for (face_index, face) in self.output.faces.iter().enumerate() {
            let corners = &self.corner_origin[output_offsets[face_index]..][..face.vertices.len()];
            let blended: Vec<Option<DVec3>> = corners
                .iter()
                .map(|sources| blend_vectors(sources, &authored, self.raw_normals))
                .collect();
            if blended.iter().all(Option::is_none) {
                continue;
            }
            // A corner whose sources have no authored normal takes the flat normal of its face.
            let points: Vec<DVec3> = face
                .vertices
                .iter()
                .filter_map(|id| positions.get(id).copied())
                .collect();
            let flat = face_normal(&points);
            let rows: Vec<Value> = blended
                .into_iter()
                .map(|normal| json!(normal.unwrap_or(flat).to_array()))
                .collect();
            result.insert(format!("f{}", face.id), Value::Array(rows));
        }
        (!result.is_empty()).then(|| with_values(attribute, Value::Object(result)))
    }

    /// Positional vertex values, blended by origin weights (see [`blend`]).
    fn vertex_values(
        &self,
        values: &[Value],
        data_type: &str,
        default: impl Fn() -> Value,
    ) -> Vec<Value> {
        self.origin
            .vertices
            .iter()
            .map(|sources| blend_indexed(values, sources, data_type).unwrap_or_else(&default))
            .collect()
    }

    /// Keyed vertex values (`"{index}"` keys, storage order) moved to output indices.
    fn vertex_object(&self, values: &Map<String, Value>, data_type: &str) -> Map<String, Value> {
        let by_index: HashMap<usize, &Value> = keyed_values(values, "");
        let mut output = Map::new();
        for (index, sources) in self.origin.vertices.iter().enumerate() {
            let blended = blend(
                sources.iter().filter_map(|(source, weight)| {
                    by_index.get(source).map(|value| (*value, *weight))
                }),
                data_type,
            );
            if let Some(value) = blended {
                output.insert(index.to_string(), value);
            }
        }
        output
    }

    /// One `blender_attributes` entry; POINT, EDGE, FACE, and CORNER entries are remapped and
    /// unknown domains are returned unchanged.
    fn attribute_entry(&self, entry: &Value) -> Value {
        let domain = text(entry, "domain");
        let data_type = text(entry, "data_type");
        let Some(values) = entry.get("values").and_then(Value::as_array) else {
            return entry.clone();
        };
        let default = || default_value(data_type);
        let remapped: Vec<Value> = match domain {
            "POINT" => self.vertex_values(values, data_type, default),
            "EDGE" => self.edge_values(values, data_type, default),
            "FACE" => self
                .origin
                .faces
                .iter()
                .map(|source| {
                    source
                        .and_then(|index| values.get(index).cloned())
                        .unwrap_or_else(default)
                })
                .collect(),
            "CORNER" => self
                .corner_origin
                .iter()
                .map(|sources| blend_indexed(values, sources, data_type).unwrap_or_else(&default))
                .collect(),
            _ => return entry.clone(),
        };
        with_values(entry, Value::Array(remapped))
    }

    /// One `blender_uv_layers` layer: each output corner blends its source corners' UVs.
    fn uv_layer(&self, layer: &Value) -> Value {
        let Some(values) = layer.get("values").and_then(Value::as_array) else {
            return layer.clone();
        };
        let remapped = self
            .corner_origin
            .iter()
            .map(|sources| blend_indexed(values, sources, "FLOAT2").unwrap_or_else(uv_default))
            .collect();
        with_values(layer, Value::Array(remapped))
    }

    /// `uv_map` entries (`{"face_id","layer","uv"}`, plus `"pinned"` when the source has pins)
    /// rebuilt per output face. Each output corner blends the source corners it came from; a source
    /// corner whose face has no entry for the layer reads as `[0, 0]` and keeps its weight (no
    /// renormalisation: Blender layers are complete, so no oracle covers partial layers). An output
    /// face gets an entry when any of its corners has a source with an entry, so a modifier that
    /// leaves face sources empty (screw) keeps its UVs. A corner's pin is the pin of its heaviest
    /// source corner (see [`heaviest_source`]); `pinned` is omitted when no contributing source
    /// corner has pins.
    fn uv_map(&self, entries: &[Value]) -> Vec<Value> {
        let source_faces: HashMap<u64, usize> = self
            .source
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| (u64::from(face.id), index))
            .collect();
        let source_offsets =
            corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
        let output_offsets =
            corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
        let mut layers: Vec<&str> = Vec::new();
        let mut layer_index: HashMap<&str, usize> = HashMap::new();
        // (layer index, source face index) -> that face's corner UVs and pins as written in `uv_map`.
        let mut per_face: HashMap<(usize, usize), PerFaceUv<'_>> = HashMap::new();
        for entry in entries {
            let Some(face) = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .and_then(|id| source_faces.get(&id))
            else {
                continue;
            };
            let Some(uv) = entry.get("uv").and_then(Value::as_array) else {
                continue;
            };
            let layer = uv_map_entry_layer(entry);
            let index = *layer_index.entry(layer).or_insert_with(|| {
                layers.push(layer);
                layers.len() - 1
            });
            let pinned = entry.get("pinned").and_then(Value::as_array);
            per_face.insert((index, *face), (uv, pinned));
        }
        let source_corners: usize = self
            .source
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum();
        let mut output = Vec::new();
        for (index, layer) in layers.iter().enumerate() {
            let mut flat = vec![uv_default(); source_corners];
            // Pin per source corner; `None` where the source entry has no `pinned` array.
            let mut flat_pinned: Vec<Option<bool>> = vec![None; source_corners];
            // Source corners whose face has an entry for this layer.
            let mut filled = vec![false; source_corners];
            for ((_, face), (uv, pinned)) in
                per_face.iter().filter(|((owner, _), _)| *owner == index)
            {
                for (corner, value) in uv.iter().enumerate() {
                    let at = source_offsets[*face] + corner;
                    flat[at] = value.clone();
                    filled[at] = true;
                }
                if let Some(pins) = pinned {
                    for (corner, pin) in pins.iter().enumerate() {
                        flat_pinned[source_offsets[*face] + corner] =
                            Some(pin.as_bool() == Some(true));
                    }
                }
            }
            for (face_index, face) in self.output.faces.iter().enumerate() {
                let corners =
                    &self.corner_origin[output_offsets[face_index]..][..face.vertices.len()];
                if !corners.iter().flatten().any(|(source, _)| filled[*source]) {
                    continue;
                }
                let uv: Vec<Value> = corners
                    .iter()
                    .map(|sources| {
                        blend_indexed(&flat, sources, "FLOAT2").unwrap_or_else(uv_default)
                    })
                    .collect();
                let mut entry = json!({"face_id": face.id, "layer": layer, "uv": uv});
                let has_pins = corners
                    .iter()
                    .flatten()
                    .any(|(source, _)| flat_pinned.get(*source).is_some_and(Option::is_some));
                if has_pins {
                    let pins: Vec<bool> = corners
                        .iter()
                        .map(|sources| {
                            heaviest_source(sources)
                                .and_then(|source| flat_pinned.get(source).copied().flatten())
                                == Some(true)
                        })
                        .collect();
                    entry["pinned"] = json!(pins);
                }
                output.push(entry);
            }
        }
        output
    }
}

/// Blend of the authored custom normals at `sources`, weights renormalised over the sources that
/// have one. Normalised unless `raw`; `None` when no source has an authored normal.
fn blend_vectors(sources: &[(usize, f64)], authored: &[Option<DVec3>], raw: bool) -> Option<DVec3> {
    let mut total = 0.0;
    let mut sum = DVec3::ZERO;
    for (index, weight) in sources {
        if let Some(Some(normal)) = authored.get(*index) {
            sum += *normal * *weight;
            total += *weight;
        }
    }
    if total <= 0.0 {
        return None;
    }
    let blended = sum / total;
    Some(if raw {
        blended
    } else {
        blended.normalize_or_zero()
    })
}

/// Normalises every authored `custom_normal` vector. Subdivision blends unnormalised levels and calls
/// this once on its final mesh.
pub(crate) fn normalize_custom_normals(mesh: &mut Mesh) {
    let Some(values) = mesh
        .attributes
        .get_mut("custom_normal")
        .and_then(|attribute| attribute.get_mut("values"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    for corners in values.values_mut() {
        for value in corners.as_array_mut().into_iter().flatten() {
            if let Some(normal) = vector3(value) {
                *value = json!(normal.normalize_or_zero().to_array());
            }
        }
    }
}

/// Index of the heaviest `(index, weight)` source; ties go to the lowest index.
fn heaviest_source(sources: &[(usize, f64)]) -> Option<usize> {
    sources
        .iter()
        .max_by(|left, right| left.1.total_cmp(&right.1).then(right.0.cmp(&left.0)))
        .map(|(index, _)| *index)
}

/// [`blend`] over `values` at the `(index, weight)` sources; indices past the end are skipped.
fn blend_indexed(values: &[Value], sources: &[(usize, f64)], data_type: &str) -> Option<Value> {
    blend(
        sources
            .iter()
            .filter_map(|(index, weight)| values.get(*index).map(|value| (value, *weight))),
        data_type,
    )
}

/// Blender data types stored as integers; blended results round to the nearest integer.
fn is_integer(data_type: &str) -> bool {
    matches!(data_type, "INT" | "INT8")
}

/// Blend values by weight: numbers and equal-length number arrays interpolate componentwise
/// (rounded for integer types); booleans OR over the positive-weight sources; anything else takes
/// the value of the heaviest source.
/// `None` when there are no sources.
fn blend<'a>(items: impl Iterator<Item = (&'a Value, f64)>, data_type: &str) -> Option<Value> {
    let items: Vec<(&Value, f64)> = items.collect();
    let first = items.first()?.0;
    if items.iter().all(|(value, _)| value.is_number()) {
        let sum: f64 = items
            .iter()
            .filter_map(|(value, weight)| Some(value.as_f64()? * weight))
            .sum();
        return Some(if is_integer(data_type) {
            json!(sum.round() as i64)
        } else {
            json!(sum)
        });
    }
    if let Some(length) = first.as_array().map(Vec::len)
        && items.iter().all(|(value, _)| {
            value
                .as_array()
                .is_some_and(|array| array.len() == length && array.iter().all(Value::is_number))
        })
    {
        let mut sums = vec![0.0; length];
        for (value, weight) in &items {
            for (sum, component) in sums.iter_mut().zip(value.as_array().into_iter().flatten()) {
                *sum += weight * component.as_f64().unwrap_or_default();
            }
        }
        let rounds = data_type == "INT32_2D";
        return Some(Value::Array(
            sums.into_iter()
                .map(|sum| {
                    if rounds {
                        json!(sum.round() as i64)
                    } else {
                        json!(sum)
                    }
                })
                .collect(),
        ));
    }
    if items.iter().all(|(value, _)| value.is_boolean()) {
        return Some(json!(items.iter().any(|(value, weight)| {
            *weight > 0.0 && value.as_bool() == Some(true)
        })));
    }
    items
        .iter()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(value, _)| (*value).clone())
}

/// Value written for an element with no source, by Blender data type.
fn default_value(data_type: &str) -> Value {
    match data_type {
        "BOOLEAN" | "BOOL" => json!(false),
        "INT" | "INT8" => json!(0),
        "FLOAT" => json!(0.0),
        "INT32_2D" => json!([0, 0]),
        "FLOAT2" => json!([0.0, 0.0]),
        "FLOAT_VECTOR" => json!([0.0, 0.0, 0.0]),
        "FLOAT_COLOR" | "BYTE_COLOR" => json!([0.0, 0.0, 0.0, 1.0]),
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use glam::DVec3;
    use serde_json::json;

    use super::*;

    fn triangle() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .expect("triangle")
    }

    #[test]
    fn interpolated_vertex_group_blends_source_weights() {
        let mut source = triangle();
        let ids: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        source.attributes.insert(
            "vertex_groups".to_owned(),
            json!({"Arm": {format!("v{}", ids[0]): 1.0, format!("v{}", ids[1]): 0.0}}),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            faces: vec![Some(0)],
            corners: None,
            raw_normals: false,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_ids: Vec<u32> = output.vertices.iter().map(|vertex| vertex.id).collect();
        let groups = &output.attributes["vertex_groups"]["Arm"];
        assert_eq!(groups[format!("v{}", out_ids[0])], json!(1.0));
        assert_eq!(groups[format!("v{}", out_ids[1])], json!(0.0));
        assert_eq!(groups[format!("v{}", out_ids[2])], json!(0.5));
    }

    /// Subdivision of one quad: edge midpoints take two corners, the face centre takes all four.
    /// Boundary halves keep their source edge's flags; the inner spokes take the default.
    #[test]
    fn subdivision_inner_edges_take_no_source_edge_flags() {
        let corners = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let mut source =
            Mesh::from_positions_and_faces(corners, vec![vec![0, 1, 2, 3]]).expect("quad");
        let flags: Vec<Value> = source
            .edges
            .iter()
            .map(|_| json!({"sharp": true, "seam": true}))
            .collect();
        source
            .attributes
            .insert("blender_edge_flags".to_owned(), Value::Array(flags));
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.5, 0.0, 0.0),
            DVec3::new(1.0, 0.5, 0.0),
            DVec3::new(0.5, 1.0, 0.0),
            DVec3::new(0.0, 0.5, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
        ];
        let mut output = Mesh::from_positions_and_faces(
            positions,
            vec![
                vec![0, 4, 8, 7],
                vec![4, 1, 5, 8],
                vec![8, 5, 2, 6],
                vec![7, 8, 6, 3],
            ],
        )
        .expect("subdivided quad");
        let origin = Origin {
            vertices: vec![
                vec![(0, 1.0)],
                vec![(1, 1.0)],
                vec![(2, 1.0)],
                vec![(3, 1.0)],
                vec![(0, 0.5), (1, 0.5)],
                vec![(1, 0.5), (2, 0.5)],
                vec![(2, 0.5), (3, 0.5)],
                vec![(3, 0.5), (0, 0.5)],
                vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)],
            ],
            faces: vec![Some(0); 4],
            corners: None,
            raw_normals: false,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let centre = output.vertices[8].id;
        assert_eq!(output.edges.len(), 12);
        let flags = output.attributes["blender_edge_flags"]
            .as_array()
            .expect("edge flags");
        for (edge, flag) in output.edges.iter().zip(flags) {
            if edge.vertices.contains(&centre) {
                assert_eq!(flag, &json!({"sharp": false, "seam": false}));
            } else {
                assert_eq!(flag, &json!({"sharp": true, "seam": true}));
            }
        }
    }

    /// Two coincident source edges, one sharp and one seam, merge into one output edge whose
    /// flags OR over both sources.
    #[test]
    fn merged_edge_flags_or_over_coincident_source_edges() {
        let mut source = Mesh::new();
        let first = source.insert_vertex(DVec3::ZERO).expect("vertex");
        let second = source.insert_vertex(DVec3::X).expect("vertex");
        let third = source.insert_vertex(DVec3::ZERO).expect("vertex");
        let fourth = source.insert_vertex(DVec3::X).expect("vertex");
        source.insert_edge([first, second]).expect("edge");
        source.insert_edge([third, fourth]).expect("edge");
        source.attributes.insert(
            "blender_edge_flags".to_owned(),
            json!([{"sharp": true, "seam": false}, {"sharp": false, "seam": true}]),
        );
        let mut output = Mesh::new();
        let near = output.insert_vertex(DVec3::ZERO).expect("vertex");
        let far = output.insert_vertex(DVec3::X).expect("vertex");
        output.insert_edge([near, far]).expect("edge");
        let origin = Origin {
            vertices: vec![vec![(0, 0.5), (2, 0.5)], vec![(1, 0.5), (3, 0.5)]],
            faces: Vec::new(),
            corners: None,
            raw_normals: false,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        assert_eq!(
            output.attributes["blender_edge_flags"],
            json!([{"sharp": true, "seam": true}])
        );
    }

    #[test]
    fn blend_ors_booleans_over_positive_weights() {
        let on = json!(true);
        let off = json!(false);
        assert_eq!(
            blend([(&on, 0.5), (&off, 0.5)].into_iter(), "BOOLEAN"),
            Some(json!(true))
        );
        assert_eq!(
            blend([(&off, 0.5), (&on, 0.0)].into_iter(), "BOOLEAN"),
            Some(json!(false))
        );
        assert_eq!(
            blend([(&off, 0.5), (&off, 0.5)].into_iter(), "BOOLEAN"),
            Some(json!(false))
        );
    }

    /// A rebuilt corner keeps the pin of its heaviest source corner (ties: lowest source index),
    /// and `pinned` stays aligned with the output corners.
    #[test]
    fn uv_pins_follow_heaviest_source_corner() {
        let mut source = triangle();
        let face_id = source.faces[0].id;
        source.attributes.insert(
            "uv_map".to_owned(),
            json!([{
                "face_id": face_id,
                "layer": "UVMap",
                "uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                "pinned": [true, false, false]
            }]),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            faces: vec![Some(0)],
            corners: Some(vec![
                vec![(0, 0.25), (1, 0.75)],
                vec![(0, 0.5), (1, 0.5)],
                vec![(2, 1.0)],
            ]),
            raw_normals: false,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let entry = &output.attributes["uv_map"][0];
        assert_eq!(entry["uv"].as_array().map(Vec::len), Some(3));
        assert_eq!(entry["pinned"], json!([false, true, false]));
    }
}
