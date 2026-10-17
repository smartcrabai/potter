// SPDX-License-Identifier: GPL-3.0-or-later

//! Element provenance for mesh-rebuilding modifiers.
//!
//! A modifier that rebuilds topology reports where each output vertex and polygon came from.
//! Edges and polygon corners are derived from those origins, so [`remap_attributes`] is the
//! single place that rewrites ID-keyed attributes after a rebuild. Vertex origins are weighted
//! so interpolated data (vertex groups, POINT attributes) blends like Blender's `CustomData`.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use glam::DVec3;
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::attribute_keys;
use super::clnor::{self, NormalCodeOptions};
use super::color_layers::{self, ColorDomain, ColorLayer};
use super::uv_layers::{self, UvLayer};
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
        }
    }
}

/// Rewrite the ID-keyed attributes of `output` (rebuilt from `source`) through `origin`.
///
/// `vertex_groups`, `crease_edge`, skin vertex keys, every `uv_layers` layer, every canonical color
/// layer, and every element-domain `mesh.attributes` entry are remapped. Corner data blends over the
/// source corners of each output corner (see [`corner_sources`]), so UVs follow the face corner they
/// came from. Shading flags follow the same origins: `shade_smooth` and `blender_polygon_smooth` copy
/// their source face's flag, and `sharp` and `seam` OR over source edges. Color layers are rebuilt by
/// the canonical store, preserving order and active/render choices. `custom_normal` codes are never
/// interpolated; [`crate::geom::clnor::transfer_custom_normals`] applies the shared code/basis rules.
/// Elements without a source take their type's default, so no key refers to a stale element.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes(source: &Mesh, output: &mut Mesh, origin: &Origin) -> Result<()> {
    remap_attributes_skipping(source, output, origin, &[])
}

/// [`remap_attributes`] without the attributes named in `skipped`, which `output` does not receive.
/// A caller that rebuilds one of them from `source` skips it here, so no work runs on it and it does
/// not make corners need sources.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes_skipping(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    skipped: &[&str],
) -> Result<()> {
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
    let kept = |key: &str| !skipped.contains(&key);
    let has = |key: &str| kept(key) && source.attributes.contains_key(key);
    let entries = if kept("blender_attributes") {
        source
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
    } else {
        &[][..]
    };
    let needs_edges = has("crease_edge")
        || has("blender_edge_flags")
        || has("sharp")
        || entries.iter().any(|entry| text(entry, "domain") == "EDGE")
        || source
            .attributes
            .iter()
            .filter(|(key, _)| kept(key.as_str()))
            .any(|(_, value)| named_domain(value).as_deref() == Some("edge"));
    let needs_corners = has(uv_layers::ATTRIBUTE)
        || has("blender_uv_layers")
        || has("uv_map")
        || has("custom_normal")
        || entries
            .iter()
            .any(|entry| text(entry, "domain") == "CORNER")
        || source
            .attributes
            .iter()
            .filter(|(key, _)| kept(key.as_str()))
            .any(|(_, value)| named_domain(value).as_deref() == Some("corner"));
    let output_vertex_index: HashMap<u32, usize> = if needs_edges || needs_corners {
        output
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect()
    } else {
        HashMap::new()
    };
    let edge_origin = if needs_edges {
        edge_sources(source, output, origin, &output_vertex_index)
    } else {
        Vec::new()
    };
    let derived_corners;
    let corner_origin: &[Vec<(usize, f64)>] = match &origin.corners {
        Some(corners) => corners,
        None if needs_corners => {
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
    };
    let color_store = has(color_layers::ATTRIBUTE).then(|| {
        let mut store = color_layers::read(source);
        store
            .layers
            .retain(|layer| kept(attribute_keys::storage_key(&layer.name).as_ref()));
        store.layers = store
            .layers
            .iter()
            .map(|layer| sources.color_layer(layer))
            .collect();
        store
    });
    let color_layer_keys: HashSet<String> = color_store
        .as_ref()
        .into_iter()
        .flat_map(|store| store.layers.iter())
        .map(|layer| attribute_keys::storage_key(&layer.name).into_owned())
        .collect();
    // Rebuild code state from source corner identity, including a basis-only layer.
    let has_custom_normal_state = kept("custom_normal")
        && (clnor::corner_codes(source)?.is_some() || clnor::corner_order(source)?.is_some());
    let code_sources: Option<Vec<Option<usize>>> = has_custom_normal_state.then(|| {
        corner_origin
            .iter()
            .map(|sources| heaviest_source(sources))
            .collect()
    });
    // The store's layers are rebuilt per corner; their names and active/render choices carry over.
    let uv_store = has(uv_layers::ATTRIBUTE).then(|| {
        let mut store = uv_layers::read(source);
        store.layers = store
            .layers
            .iter()
            .map(|layer| sources.uv_layer(layer))
            .collect();
        store
    });
    let mut attributes = Map::new();
    for (key, value) in &source.attributes {
        if !kept(key.as_str()) || color_layer_keys.contains(key) {
            continue;
        }
        if key == uv_layers::ATTRIBUTE {
            // Rebuilt after the loop from `uv_store`.
            continue;
        }
        if key == "custom_normal" {
            // The code state is rebuilt below from source corner identity, never interpolated.
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
            _ => native_domain(value)
                .map(|_| sources.native_attribute(value))
                .or_else(|| sources.named_attribute(value)),
        };
        attributes.insert(key.clone(), remapped.unwrap_or_else(|| value.clone()));
    }
    output.attributes = attributes;
    if let Some(store) = color_store {
        color_layers::write(output, &store);
    }
    if let Some(store) = uv_store {
        uv_layers::write(output, &store);
    }
    if let Some(sources) = code_sources {
        clnor::transfer_custom_normals(source, output, &sources, NormalCodeOptions::default())?;
    }
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

/// The `domain` of a named element attribute (one with a `values` object), lowercased.
fn named_domain(value: &Value) -> Option<String> {
    value.get("values")?.as_object()?;
    Some(text(value, "domain").to_ascii_lowercase())
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
    // Every source corner keyed by its face and vertex, built once per member, so an output corner
    // finds its source corner by lookup instead of scanning its source face.
    let mut source_corners: HashMap<(usize, u32), usize> = HashMap::new();
    for (index, face) in source.faces.iter().enumerate() {
        for (position, id) in face.vertices.iter().enumerate() {
            source_corners
                .entry((index, *id))
                .or_insert(offsets[index] + position);
        }
    }
    let mut corners = Vec::new();
    for (face, source_face) in output.faces.iter().zip(&origin.faces) {
        for vertex_id in &face.vertices {
            let vertex_sources = output_vertex_index
                .get(vertex_id)
                .and_then(|index| origin.vertices.get(*index));
            let mut kept = Vec::new();
            if let (Some(index), Some(vertex_sources)) = (*source_face, vertex_sources) {
                for (vertex, weight) in vertex_sources {
                    let Some(id) = source.vertices.get(*vertex).map(|vertex| vertex.id) else {
                        continue;
                    };
                    if let Some(&corner) = source_corners.get(&(index, id)) {
                        kept.push((corner, *weight));
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

struct Sources<'a> {
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

    /// Legacy Blender edge flags combined from every source edge.
    fn edge_flags(&self, flags: &[Value]) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|sources| {
                let flag = |name: &str| {
                    sources.iter().any(|(index, weight)| {
                        *weight > 0.0
                            && flags
                                .get(*index)
                                .and_then(|flags| flags.get(name))
                                .and_then(Value::as_bool)
                                == Some(true)
                    })
                };
                json!({"sharp": flag("sharp"), "seam": flag("seam")})
            })
            .collect()
    }

    /// Canonical face smooth flags copied from each output face's source face.
    fn shade_smooth(&self, attribute: &Value) -> Value {
        let Some(values) = attribute.get("values").and_then(Value::as_object) else {
            return attribute.clone();
        };
        let remapped: Map<String, Value> = self
            .output
            .faces
            .iter()
            .zip(&self.origin.faces)
            .map(|(face, source)| {
                let smooth = source
                    .and_then(|index| self.source.faces.get(index))
                    .and_then(|source_face| values.get(&format!("f{}", source_face.id)))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                (format!("f{}", face.id), json!(smooth))
            })
            .collect();
        with_values(attribute, Value::Object(remapped))
    }

    /// Canonical sharp flags OR-ed over every source edge of each output edge.
    fn sharp(&self, attribute: &Value) -> Value {
        if attribute.get("values").and_then(Value::as_object).is_none() {
            return attribute.clone();
        }
        let source_flags = edge_flags(self.source, "sharp");
        let remapped: Map<String, Value> = self
            .output
            .edges
            .iter()
            .zip(self.edge_origin)
            .map(|(edge, sources)| {
                let sharp = sources.iter().any(|(index, weight)| {
                    *weight > 0.0 && source_flags.get(*index) == Some(&true)
                });
                (format!("e{}", edge.id), json!(sharp))
            })
            .collect();
        with_values(attribute, Value::Object(remapped))
    }

    /// Legacy polygon-smooth flags copied from each output face's source face.
    fn polygon_smooth(&self, flags: &[Value]) -> Vec<Value> {
        self.origin
            .faces
            .iter()
            .map(|source| {
                source
                    .and_then(|index| flags.get(index))
                    .cloned()
                    .unwrap_or_else(|| json!(false))
            })
            .collect()
    }

    /// A native element attribute (`{"domain", "type", "values"}`) moved to the output elements of its
    /// domain. Point, edge, and corner values blend over their origins by weight, and a face value copies
    /// its source face's value. Output elements without a source take the type's default. Booleans blend
    /// as OR over the positive-weight sources, and colors blend channel by channel.
    fn native_attribute(&self, attribute: &Value) -> Value {
        let Some(values) = attribute.get("values").and_then(Value::as_object) else {
            return attribute.clone();
        };
        let data_type = blend_type(text(attribute, "type"));
        let default = || default_value(data_type);
        let remapped: Map<String, Value> = match text(attribute, "domain") {
            "point" => {
                let stored: Vec<Option<&Value>> = self
                    .source
                    .vertices
                    .iter()
                    .map(|vertex| values.get(&format!("v{}", vertex.id)))
                    .collect();
                self.output
                    .vertices
                    .iter()
                    .zip(&self.origin.vertices)
                    .map(|(vertex, sources)| {
                        let value =
                            blend_sources(&stored, sources, data_type).unwrap_or_else(default);
                        (format!("v{}", vertex.id), value)
                    })
                    .collect()
            }
            "edge" => {
                let stored: Vec<Option<&Value>> = self
                    .source
                    .edges
                    .iter()
                    .map(|edge| values.get(&format!("e{}", edge.id)))
                    .collect();
                self.output
                    .edges
                    .iter()
                    .zip(self.edge_origin)
                    .map(|(edge, sources)| {
                        let value =
                            blend_sources(&stored, sources, data_type).unwrap_or_else(default);
                        (format!("e{}", edge.id), value)
                    })
                    .collect()
            }
            "face" => self
                .output
                .faces
                .iter()
                .zip(&self.origin.faces)
                .map(|(face, source)| {
                    let value = source
                        .and_then(|index| self.source.faces.get(index))
                        .and_then(|source_face| values.get(&format!("f{}", source_face.id)))
                        .cloned()
                        .unwrap_or_else(default);
                    (format!("f{}", face.id), value)
                })
                .collect(),
            "corner" => {
                let corner_count: usize = self
                    .source
                    .faces
                    .iter()
                    .map(|face| face.vertices.len())
                    .sum();
                let mut stored: Vec<Option<&Value>> = vec![None; corner_count];
                let offsets =
                    corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
                for (face, offset) in self.source.faces.iter().zip(offsets) {
                    let Some(row) = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };
                    for (corner, value) in row.iter().enumerate().take(face.vertices.len()) {
                        stored[offset + corner] = Some(value);
                    }
                }
                let offsets =
                    corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
                self.output
                    .faces
                    .iter()
                    .zip(offsets)
                    .map(|(face, offset)| {
                        let row: Vec<Value> = self.corner_origin[offset..][..face.vertices.len()]
                            .iter()
                            .map(|sources| {
                                blend_sources(&stored, sources, data_type).unwrap_or_else(default)
                            })
                            .collect();
                        (format!("f{}", face.id), Value::Array(row))
                    })
                    .collect()
            }
            _ => return attribute.clone(),
        };
        with_values(attribute, Value::Object(remapped))
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
    /// One canonical color layer rebuilt in storage order and keyed by `color_layers::write`.
    fn color_layer(&self, layer: &ColorLayer) -> ColorLayer {
        let values = match layer.domain {
            ColorDomain::Point => self
                .origin
                .vertices
                .iter()
                .map(|sources| blend_color(&layer.values, sources))
                .collect(),
            ColorDomain::Corner => self
                .corner_origin
                .iter()
                .map(|sources| blend_color(&layer.values, sources))
                .collect(),
        };
        ColorLayer {
            name: layer.name.clone(),
            domain: layer.domain,
            values,
        }
    }
    /// One UV layer rebuilt for the output faces. Each output corner blends the UVs of its source
    /// corners by weight. A source corner whose face has no entry reads as `[0, 1]` and keeps its
    /// weight (no renormalisation). An output face gets an entry when any of its corners has a source
    /// with an entry, so a modifier that leaves face sources empty (screw) keeps its UVs. A corner's
    /// pin is the pin of its heaviest source corner (see [`heaviest_source`]); `pinned` is omitted
    /// when no contributing source corner has pins.
    fn uv_layer(&self, layer: &UvLayer) -> UvLayer {
        let source_offsets =
            corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
        let output_offsets =
            corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
        let source_corners: usize = self
            .source
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum();
        let mut flat = vec![[0.0, 1.0]; source_corners];
        // Pin per source corner; `None` where the source face has no `pinned` array.
        let mut flat_pinned: Vec<Option<bool>> = vec![None; source_corners];
        // Source corners whose face has an entry for this layer.
        let mut filled = vec![false; source_corners];
        for (index, &at) in source_offsets.iter().enumerate() {
            let Some(Some(uv)) = layer.faces.get(index) else {
                continue;
            };
            for (corner, value) in uv.iter().enumerate() {
                flat[at + corner] = *value;
                filled[at + corner] = true;
            }
            if let Some(Some(pins)) = layer.pinned.get(index) {
                for (corner, pin) in pins.iter().enumerate() {
                    flat_pinned[at + corner] = Some(*pin);
                }
            }
        }
        let mut faces: Vec<Option<Vec<[f64; 2]>>> = Vec::with_capacity(self.output.faces.len());
        let mut pinned: Vec<Option<Vec<bool>>> = Vec::with_capacity(self.output.faces.len());
        for (face_index, face) in self.output.faces.iter().enumerate() {
            let corners = &self.corner_origin[output_offsets[face_index]..][..face.vertices.len()];
            if !corners.iter().flatten().any(|(source, _)| filled[*source]) {
                faces.push(None);
                pinned.push(None);
                continue;
            }
            let uv: Vec<[f64; 2]> = corners
                .iter()
                .map(|sources| blend_uv(&flat, sources))
                .collect();
            faces.push(Some(uv));
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
                pinned.push(Some(pins));
            } else {
                pinned.push(None);
            }
        }
        UvLayer {
            name: layer.name.clone(),
            faces,
            pinned,
        }
    }

    /// A named attribute keyed by element ID (`{"domain","type","values":{prefix+id: value}}`), moved to
    /// the output's IDs. Point and edge values blend over their source elements, a face value copies its
    /// source face's, and corner values (one array per face) blend over the source corners. Other domains
    /// are copied unchanged by the caller.
    fn named_attribute(&self, attribute: &Value) -> Option<Value> {
        let domain = named_domain(attribute)?;
        let kind = text(attribute, "type").to_ascii_uppercase();
        let values = attribute.get("values").and_then(Value::as_object)?;
        let mut output = Map::new();
        match domain.as_str() {
            "point" => {
                let by_id: HashMap<u32, &Value> = keyed_values(values, "v");
                for (vertex, sources) in self.output.vertices.iter().zip(&self.origin.vertices) {
                    let blended = blend(
                        sources.iter().filter_map(|(index, weight)| {
                            let id = self.source.vertices.get(*index)?.id;
                            by_id.get(&id).map(|value| (*value, *weight))
                        }),
                        &kind,
                    );
                    if let Some(value) = blended {
                        output.insert(format!("v{}", vertex.id), value);
                    }
                }
            }
            "edge" => {
                let by_id: HashMap<u32, &Value> = keyed_values(values, "e");
                for (edge, sources) in self.output.edges.iter().zip(self.edge_origin) {
                    let blended = blend(
                        sources.iter().filter_map(|(index, weight)| {
                            let id = self.source.edges.get(*index)?.id;
                            by_id.get(&id).map(|value| (*value, *weight))
                        }),
                        &kind,
                    );
                    if let Some(value) = blended {
                        output.insert(format!("e{}", edge.id), value);
                    }
                }
            }
            "face" => {
                for (face, source) in self.output.faces.iter().zip(&self.origin.faces) {
                    let value = source
                        .and_then(|index| self.source.faces.get(index))
                        .and_then(|source| values.get(&format!("f{}", source.id)));
                    if let Some(value) = value {
                        output.insert(format!("f{}", face.id), value.clone());
                    }
                }
            }
            "corner" => {
                let source_offsets =
                    corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
                let source_corners: usize = self
                    .source
                    .faces
                    .iter()
                    .map(|face| face.vertices.len())
                    .sum();
                let mut authored: Vec<Option<&Value>> = vec![None; source_corners];
                for (index, face) in self.source.faces.iter().enumerate() {
                    let Some(corners) = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };
                    for (corner, value) in corners.iter().enumerate().take(face.vertices.len()) {
                        authored[source_offsets[index] + corner] = Some(value);
                    }
                }
                let output_offsets =
                    corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
                for (face_index, face) in self.output.faces.iter().enumerate() {
                    let corners =
                        &self.corner_origin[output_offsets[face_index]..][..face.vertices.len()];
                    let row: Vec<Option<Value>> = corners
                        .iter()
                        .map(|sources| {
                            blend(
                                sources.iter().filter_map(|(source, weight)| {
                                    authored
                                        .get(*source)
                                        .copied()
                                        .flatten()
                                        .map(|value| (value, *weight))
                                }),
                                &kind,
                            )
                        })
                        .collect();
                    if row.iter().all(Option::is_none) {
                        continue;
                    }
                    let row = row
                        .into_iter()
                        .map(|value| value.unwrap_or_else(|| default_value(&kind)))
                        .collect();
                    output.insert(format!("f{}", face.id), Value::Array(row));
                }
            }
            _ => return None,
        }
        Some(with_values(attribute, Value::Object(output)))
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

/// The weighted sum of the UVs at `sources`. Indices past the end of `uvs` are skipped, so a corner
/// with no usable source is `[0, 1]`.
fn blend_uv(uvs: &[[f64; 2]], sources: &[(usize, f64)]) -> [f64; 2] {
    let mut uv = [0.0; 2];
    let mut found = false;
    for (index, weight) in sources {
        if let Some([u, v]) = uvs.get(*index) {
            found = true;
            uv[0] += weight * u;
            uv[1] += weight * v;
        }
    }
    if found { uv } else { [0.0, 1.0] }
}

/// Blend canonical RGBA colors; an output without source values takes the opaque-black default.
fn blend_color(values: &[[f64; 4]], sources: &[(usize, f64)]) -> [f64; 4] {
    let mut color = [0.0; 4];
    let mut found = false;
    for (index, weight) in sources {
        if let Some(source) = values.get(*index) {
            found = true;
            for (channel, value) in color.iter_mut().zip(source) {
                *channel += weight * value;
            }
        }
    }
    if found { color } else { [0.0, 0.0, 0.0, 1.0] }
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
pub(crate) fn default_value(data_type: &str) -> Value {
    match data_type {
        "BOOLEAN" | "BOOL" => json!(false),
        "INT" | "INT8" => json!(0),
        "FLOAT" => json!(0.0),
        "INT32_2D" => json!([0, 0]),
        "FLOAT2" => json!([0.0, 0.0]),
        "FLOAT_VECTOR" => json!([0.0, 0.0, 0.0]),
        "FLOAT_COLOR" | "BYTE_COLOR" => json!([0.0, 0.0, 0.0, 1.0]),
        "QUATERNION" => json!([0.0, 0.0, 0.0, 1.0]),
        _ => Value::Null,
    }
}

/// The Blender data type that [`blend`] and [`default_value`] use for a native attribute type.
fn blend_type(kind: &str) -> &'static str {
    match kind {
        "int" => "INT",
        "bool" => "BOOLEAN",
        "float2" => "FLOAT2",
        "float3" => "FLOAT_VECTOR",
        "quaternion" => "QUATERNION",
        "color" => "FLOAT_COLOR",
        _ => "FLOAT",
    }
}

/// [`blend`] over the `stored` values at the `(index, weight)` sources. Sources without a value are skipped.
fn blend_sources(
    stored: &[Option<&Value>],
    sources: &[(usize, f64)],
    data_type: &str,
) -> Option<Value> {
    blend(
        sources
            .iter()
            .filter_map(|(index, weight)| Some(((*stored.get(*index)?)?, *weight))),
        data_type,
    )
}

/// The element domain of a native attribute (`point`, `edge`, `face`, or `corner`). Other mesh keys, such
/// as `uv_layers`, `color_layers`, `vertex_groups`, and the Blender containers, have no such domain.
fn native_domain(value: &Value) -> Option<&str> {
    value
        .get("domain")
        .and_then(Value::as_str)
        .filter(|domain| matches!(*domain, "point" | "edge" | "face" | "corner"))
}

/// Flags of the edge-domain boolean attribute `name`, one per edge in storage order. An edge without an
/// entry, or a mesh without the attribute, is unflagged.
pub(crate) fn edge_flags(mesh: &Mesh, name: &str) -> Vec<bool> {
    let values = mesh
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    mesh.edges
        .iter()
        .map(|edge| {
            values
                .and_then(|values| values.get(&format!("e{}", edge.id)))
                .and_then(Value::as_bool)
                == Some(true)
        })
        .collect()
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
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_ids: Vec<u32> = output.vertices.iter().map(|vertex| vertex.id).collect();
        let groups = &output.attributes["vertex_groups"]["Arm"];
        assert_eq!(groups[format!("v{}", out_ids[0])], json!(1.0));
        assert_eq!(groups[format!("v{}", out_ids[1])], json!(0.0));
        assert_eq!(groups[format!("v{}", out_ids[2])], json!(0.5));
    }

    /// A JSON object of keyed entries, for attribute fixtures.
    fn keyed(entries: Vec<(String, Value)>) -> Value {
        Value::Object(entries.into_iter().collect())
    }

    /// Subdivision of one quad: edge midpoints take two corners, the face centre takes all four.
    /// Boundary halves keep their source edge's sharp and seam flags; the inner spokes are unflagged.
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
        let flags = keyed(
            source
                .edges
                .iter()
                .map(|edge| (format!("e{}", edge.id), json!(true)))
                .collect(),
        );
        source.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": flags.clone()}),
        );
        source.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": flags}),
        );
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
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let centre = output.vertices[8].id;
        assert_eq!(output.edges.len(), 12);
        let sharp = edge_flags(&output, "sharp");
        let seam = edge_flags(&output, "seam");
        for (edge, (sharp, seam)) in output.edges.iter().zip(sharp.into_iter().zip(seam)) {
            let boundary = !edge.vertices.contains(&centre);
            assert_eq!((sharp, seam), (boundary, boundary));
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
        let (sharp_edge, seam_edge) = (source.edges[0].id, source.edges[1].id);
        source.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": keyed(vec![
                (format!("e{sharp_edge}"), json!(true)),
                (format!("e{seam_edge}"), json!(false)),
            ])}),
        );
        source.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": keyed(vec![
                (format!("e{sharp_edge}"), json!(false)),
                (format!("e{seam_edge}"), json!(true)),
            ])}),
        );
        let mut output = Mesh::new();
        let near = output.insert_vertex(DVec3::ZERO).expect("vertex");
        let far = output.insert_vertex(DVec3::X).expect("vertex");
        output.insert_edge([near, far]).expect("edge");
        let origin = Origin {
            vertices: vec![vec![(0, 0.5), (2, 0.5)], vec![(1, 0.5), (3, 0.5)]],
            faces: Vec::new(),
            corners: None,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        assert_eq!(edge_flags(&output, "sharp"), vec![true]);
        assert_eq!(edge_flags(&output, "seam"), vec![true]);
    }

    /// Native element attributes follow their origins: a point color blends the two sources of a merged
    /// vertex, a corner color follows the corner it came from (blended where the output corner has two
    /// sources), and a face value copies its source face.
    #[test]
    fn native_color_and_generic_attributes_follow_origins() {
        let mut source = triangle();
        let vertices: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        let face = source.faces[0].id;
        source.attributes.insert(
            "Col".to_owned(),
            json!({"domain": "point", "type": "color", "values": keyed(vec![
                (format!("v{}", vertices[0]), json!([1.0, 0.0, 0.0, 1.0])),
                (format!("v{}", vertices[1]), json!([0.0, 1.0, 0.0, 1.0])),
                (format!("v{}", vertices[2]), json!([0.0, 0.0, 1.0, 1.0])),
            ])}),
        );
        source.attributes.insert(
            "Paint".to_owned(),
            json!({"domain": "corner", "type": "color", "values": keyed(vec![(
                format!("f{face}"),
                json!([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            )])}),
        );
        source.attributes.insert(
            "weight".to_owned(),
            json!({"domain": "face", "type": "float", "values": keyed(vec![(
                format!("f{face}"),
                json!(0.75),
            )])}),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            faces: vec![Some(0)],
            corners: None,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_vertex = format!("v{}", output.vertices[2].id);
        let out_face = format!("f{}", output.faces[0].id);
        assert_eq!(
            output.attributes["Col"]["values"][out_vertex.as_str()],
            json!([0.5, 0.5, 0.0, 1.0])
        );
        assert_eq!(
            output.attributes["Paint"]["values"][out_face.as_str()],
            json!([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0]
            ])
        );
        assert_eq!(
            output.attributes["weight"]["values"][out_face.as_str()],
            json!(0.75)
        );
    }

    #[test]
    fn canonical_color_store_layers_follow_origins_and_keep_designations() {
        let mut source = triangle();
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let store = color_layers::ColorStore {
            layers: vec![
                ColorLayer {
                    name: "color_layers".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![red, green, blue],
                },
                ColorLayer {
                    name: "Paint".to_owned(),
                    domain: ColorDomain::Corner,
                    values: vec![red, green, blue],
                },
            ],
            active: Some("Paint".to_owned()),
            render: Some("color_layers".to_owned()),
        };
        color_layers::write(&mut source, &store);
        assert!(
            source
                .attributes
                .contains_key(attribute_keys::storage_key("color_layers").as_ref())
        );

        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            faces: vec![Some(0)],
            corners: None,
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");

        let store = color_layers::read(&output);
        assert_eq!(
            store
                .layers
                .iter()
                .map(|layer| layer.name.as_str())
                .collect::<Vec<_>>(),
            vec!["color_layers", "Paint"]
        );
        assert_eq!(store.layers[0].domain, ColorDomain::Point);
        assert_eq!(
            store.layers[0].values,
            vec![red, green, [0.5, 0.5, 0.0, 1.0]]
        );
        assert_eq!(store.layers[1].domain, ColorDomain::Corner);
        assert_eq!(
            store.layers[1].values,
            vec![red, green, [0.5, 0.5, 0.0, 1.0]]
        );
        assert_eq!(store.active.as_deref(), Some("Paint"));
        assert_eq!(store.render.as_deref(), Some("color_layers"));
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
        uv_layers::write_layers(
            &mut source,
            &[UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                pinned: vec![Some(vec![true, false, false])],
            }],
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
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let layers = uv_layers::read_layers(&output);
        assert_eq!(layers[0].faces[0].as_ref().map(Vec::len), Some(3));
        assert_eq!(layers[0].pinned[0], Some(vec![false, true, false]));
    }

    /// Screw can create a face with no source face while carrying corner provenance from a partial
    /// UV layer; missing and source-less corners take `[0, 1]` without changing populated sums.
    #[test]
    fn screw_partial_uv_provenance_defaults_missing_corners_to_canonical_uv() {
        let mut source = Mesh::from_positions_and_faces(
            vec![
                DVec3::ZERO,
                DVec3::X,
                DVec3::Y,
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(2.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )
        .expect("two source faces");
        uv_layers::write_layers(
            &mut source,
            &[UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]), None],
                pinned: vec![None, None],
            }],
        );
        let mut output = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(0.0, 1.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .expect("generated face");
        let origin = Origin {
            vertices: vec![
                vec![(0, 1.0)],
                vec![(3, 1.0)],
                vec![(1, 0.5), (3, 0.5)],
                Vec::new(),
            ],
            faces: vec![None],
            corners: Some(vec![
                vec![(0, 1.0)],
                vec![(3, 1.0)],
                vec![(1, 0.5), (3, 0.5)],
                Vec::new(),
            ]),
        };

        remap_attributes(&source, &mut output, &origin).expect("remap");

        assert_eq!(
            uv_layers::read_layers(&output)[0].faces[0],
            Some(vec![[2.0, 4.0], [0.0, 1.0], [2.0, 3.5], [0.0, 1.0]])
        );
    }
}
