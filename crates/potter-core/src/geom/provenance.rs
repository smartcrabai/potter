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

use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

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
        }
    }
}

/// Rewrite the ID-keyed attributes of `output` (rebuilt from `source`) through `origin`.
///
/// `vertex_groups`, `crease_edge`, `blender_edge_flags`, skin vertex keys, and every
/// `blender_attributes` entry on the POINT, EDGE, FACE, or non-UV CORNER domain are remapped.
/// UV layers and shading keys are copied unchanged; their remapping belongs to their own pass.
/// Elements without a source take the domain default, so no key refers to a stale element.
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
    let uv_names: HashSet<String> = source
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| layer.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let needs_edges = source.attributes.contains_key("crease_edge")
        || source.attributes.contains_key("blender_edge_flags")
        || entries.iter().any(|entry| text(entry, "domain") == "EDGE");
    let needs_corners = entries
        .iter()
        .any(|entry| text(entry, "domain") == "CORNER" && !uv_names.contains(text(entry, "name")));
    let mut edge_origin = Vec::new();
    let mut corner_origin = Vec::new();
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
        if needs_corners {
            corner_origin = corner_sources(source, origin, output, &output_vertex_index);
        }
    }
    let sources = Sources {
        source,
        output,
        origin,
        edge_origin: &edge_origin,
        corner_origin: &corner_origin,
    };
    let mut attributes = Map::new();
    for (key, value) in &source.attributes {
        let remapped = match key.as_str() {
            "vertex_groups" => value
                .as_object()
                .map(|groups| Value::Object(sources.vertex_groups(groups))),
            "crease_edge" => Some(sources.keyed_edge_attribute(value)),
            "blender_edge_flags" => value.as_array().map(|flags| {
                // Flags are objects, so the heaviest source edge wins rather than a blend.
                Value::Array(sources.edge_values(
                    flags,
                    "",
                    || json!({"sharp": false, "seam": false}),
                ))
            }),
            "blender_attributes" => value.as_array().map(|entries| {
                Value::Array(
                    entries
                        .iter()
                        .map(|entry| sources.attribute_entry(entry, &uv_names))
                        .collect(),
                )
            }),
            "skin_radii" | "skin_roots" => value
                .as_object()
                .map(|values| Value::Object(sources.vertex_object(values, "FLOAT"))),
            "blender_skin_vertices" => value.as_array().map(|vertices| {
                Value::Array(sources.vertex_values(vertices, "FLOAT", || json!({})))
            }),
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

/// Source edges each output edge blends from: every distinct input edge that joins a source vertex
/// of the output edge's first vertex to a source vertex of its second, with equal weights. Blender
/// averages merged duplicate edges, so several candidates blend instead of falling back to default.
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
            let mut found: Vec<usize> = Vec::new();
            for (first_source, _) in first {
                for (second_source, _) in second {
                    if let Some(index) =
                        source_edge_index.get(&edge_key(*first_source, *second_source))
                        && !found.contains(index)
                    {
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

    /// One `blender_attributes` entry; UV layers and unknown domains are returned unchanged.
    fn attribute_entry(&self, entry: &Value, uv_names: &HashSet<String>) -> Value {
        let name = text(entry, "name");
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
            "CORNER" if !uv_names.contains(name) => self
                .corner_origin
                .iter()
                .map(|sources| blend_indexed(values, sources, data_type).unwrap_or_else(&default))
                .collect(),
            _ => return entry.clone(),
        };
        with_values(entry, Value::Array(remapped))
    }
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
    matches!(data_type, "INT" | "INT8" | "INT32")
}

/// Blend values by weight: numbers and equal-length number arrays interpolate componentwise
/// (rounded for integer types); anything else takes the value of the heaviest source.
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
    items
        .iter()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(value, _)| (*value).clone())
}

/// Value written for an element with no source, by Blender data type.
fn default_value(data_type: &str) -> Value {
    match data_type {
        "BOOLEAN" | "BOOL" => json!(false),
        "INT" | "INT8" | "INT32" => json!(0),
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
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_ids: Vec<u32> = output.vertices.iter().map(|vertex| vertex.id).collect();
        let groups = &output.attributes["vertex_groups"]["Arm"];
        assert_eq!(groups[format!("v{}", out_ids[0])], json!(1.0));
        assert_eq!(groups[format!("v{}", out_ids[1])], json!(0.0));
        assert_eq!(groups[format!("v{}", out_ids[2])], json!(0.5));
    }
}
