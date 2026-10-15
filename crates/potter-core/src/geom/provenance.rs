// SPDX-License-Identifier: GPL-3.0-or-later

//! Element provenance for mesh-rebuilding modifiers.
//!
//! A modifier that rebuilds topology reports where each output vertex and polygon came from.
//! Edges and polygon corners are derived from those origins, so [`remap_attributes`] is the
//! single place that rewrites ID-keyed attributes after a rebuild. Vertex origins are weighted
//! so interpolated data (vertex groups, POINT attributes) blends like Blender's `CustomData`.

use std::collections::{HashMap, HashSet};

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
    let source_vertex_index: HashMap<u32, usize> = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let output_vertex_index: HashMap<u32, usize> = output
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
    let edge_sources: Vec<Option<usize>> = output
        .edges
        .iter()
        .map(|edge| {
            let first = single_source(origin, *output_vertex_index.get(&edge.vertices[0])?)?;
            let second = single_source(origin, *output_vertex_index.get(&edge.vertices[1])?)?;
            source_edge_index.get(&edge_key(first, second)).copied()
        })
        .collect();
    let source_corner_offsets = corner_offsets(source.faces.iter().map(|face| face.vertices.len()));
    let mut corner_sources = Vec::new();
    for (face, source_face) in output.faces.iter().zip(&origin.faces) {
        for vertex_id in &face.vertices {
            let source_corner = source_face.and_then(|source_face| {
                let vertex = single_source(origin, *output_vertex_index.get(vertex_id)?)?;
                let position = source.faces[source_face]
                    .vertices
                    .iter()
                    .position(|id| source_vertex_index.get(id) == Some(&vertex))?;
                Some(source_corner_offsets[source_face] + position)
            });
            corner_sources.push(source_corner);
        }
    }
    let uv_names: HashSet<String> = source
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| layer.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let sources = Sources {
        source,
        output,
        origin,
        edge_origin: &edge_sources,
        corner_origin: &corner_sources,
    };
    let mut attributes = source.attributes.clone();
    if let Some(groups) = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    {
        attributes.insert(
            "vertex_groups".to_owned(),
            Value::Object(sources.vertex_groups(groups)),
        );
    }
    if let Some(attribute) = source.attributes.get("crease_edge") {
        attributes.insert(
            "crease_edge".to_owned(),
            sources.keyed_edge_attribute(attribute),
        );
    }
    if let Some(flags) = source
        .attributes
        .get("blender_edge_flags")
        .and_then(Value::as_array)
    {
        let remapped = sources.edge_values(flags, || json!({"sharp": false, "seam": false}));
        attributes.insert("blender_edge_flags".to_owned(), Value::Array(remapped));
    }
    if let Some(entries) = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    {
        let remapped = entries
            .iter()
            .map(|entry| sources.attribute_entry(entry, &uv_names))
            .collect();
        attributes.insert("blender_attributes".to_owned(), Value::Array(remapped));
    }
    for key in ["skin_radii", "skin_roots"] {
        if let Some(values) = source.attributes.get(key).and_then(Value::as_object) {
            let remapped = sources.vertex_object(values, "FLOAT");
            attributes.insert(key.to_owned(), Value::Object(remapped));
        }
    }
    if let Some(vertices) = source
        .attributes
        .get("blender_skin_vertices")
        .and_then(Value::as_array)
    {
        let remapped = sources.vertex_values(vertices, "FLOAT", || json!({}));
        attributes.insert("blender_skin_vertices".to_owned(), Value::Array(remapped));
    }
    output.attributes = attributes;
    Ok(())
}

/// The input vertex a rebuilt vertex copies exactly, if it copies exactly one.
fn single_source(origin: &Origin, output_vertex: usize) -> Option<usize> {
    match origin.vertices.get(output_vertex)?.as_slice() {
        [(index, _)] => Some(*index),
        _ => None,
    }
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

struct Sources<'a> {
    source: &'a Mesh,
    output: &'a Mesh,
    origin: &'a Origin,
    edge_origin: &'a [Option<usize>],
    corner_origin: &'a [Option<usize>],
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

    /// An edge-domain float map (`{"domain","type","values":{"e{id}":v}}`) moved to output edge IDs.
    fn keyed_edge_attribute(&self, attribute: &Value) -> Value {
        let values = attribute.get("values").and_then(Value::as_object);
        let mut output = Map::new();
        for (edge, source) in self.output.edges.iter().zip(self.edge_origin) {
            let Some(source) = source else { continue };
            let Some(source_edge) = self.source.edges.get(*source) else {
                continue;
            };
            if let Some(value) =
                values.and_then(|values| values.get(&format!("e{}", source_edge.id)))
            {
                output.insert(format!("e{}", edge.id), value.clone());
            }
        }
        let mut attribute = attribute.clone();
        attribute["values"] = Value::Object(output);
        attribute
    }

    fn edge_values(&self, values: &[Value], default: impl Fn() -> Value) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|source| {
                source
                    .and_then(|index| values.get(index).cloned())
                    .unwrap_or_else(&default)
            })
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
            .map(|sources| {
                blend(
                    sources.iter().filter_map(|(index, weight)| {
                        values.get(*index).map(|value| (value, *weight))
                    }),
                    data_type,
                )
                .unwrap_or_else(&default)
            })
            .collect()
    }

    /// Keyed vertex values (`"{index}"` keys, storage order) moved to output indices.
    fn vertex_object(&self, values: &Map<String, Value>, data_type: &str) -> Map<String, Value> {
        let mut output = Map::new();
        for (index, sources) in self.origin.vertices.iter().enumerate() {
            let blended = blend(
                sources.iter().filter_map(|(source, weight)| {
                    values
                        .get(&source.to_string())
                        .map(|value| (value, *weight))
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
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let domain = entry
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let data_type = entry
            .get("data_type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(values) = entry.get("values").and_then(Value::as_array) else {
            return entry.clone();
        };
        let default = || default_value(data_type);
        let remapped = match domain {
            "POINT" => self.vertex_values(values, data_type, default),
            "EDGE" => self.edge_values(values, default),
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
                .map(|source| {
                    source
                        .and_then(|index| values.get(index).cloned())
                        .unwrap_or_else(default)
                })
                .collect(),
            _ => return entry.clone(),
        };
        let mut entry = entry.clone();
        entry["values"] = Value::Array(remapped);
        entry
    }
}

/// Blend values by weight: numbers and equal-length number arrays interpolate componentwise;
/// anything else takes the value of the heaviest source. `None` when there are no sources.
fn blend<'a>(items: impl Iterator<Item = (&'a Value, f64)>, data_type: &str) -> Option<Value> {
    let items: Vec<(&Value, f64)> = items.collect();
    let first = items.first()?.0;
    if items.iter().all(|(value, _)| value.is_number()) {
        let sum: f64 = items
            .iter()
            .filter_map(|(value, weight)| Some(value.as_f64()? * weight))
            .sum();
        return Some(if data_type == "INT" {
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
        return Some(Value::Array(
            sums.into_iter().map(|sum| json!(sum)).collect(),
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
        "FLOAT2" | "INT32_2D" => json!([0.0, 0.0]),
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
