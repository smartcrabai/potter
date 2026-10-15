// SPDX-License-Identifier: GPL-3.0-or-later

//! UV layers in mesh attributes.
//!
//! A layer can be stored in three places that must agree: `blender_uv_layers` (`[u, v]` per corner
//! in loop order), `uv_map` (`{"face_id", "layer", "uv"}` entries, `UVMap` when `layer` is absent),
//! and the UV-named `FLOAT2` `CORNER` entries of `blender_attributes` that Blender lists beside
//! `mesh.attributes`. [`read_layers`] is the one reader: `blender_uv_layers` is authoritative for
//! the layers it lists, and `uv_map` supplies the rest. [`write_layers`] is the one writer: it
//! updates every existing store of each named layer and never creates one.

use std::collections::HashMap;

use serde_json::{Value, json};

use super::Mesh;

/// One UV layer: `faces[f]` holds the corner `[u, v]` values of output face `f` in corner order, or
/// `None` where the layer has no value for that face.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct UvLayer {
    pub(crate) name: String,
    pub(crate) faces: Vec<Option<Vec<[f64; 2]>>>,
}

/// Default UV for a corner with no source data.
pub(crate) fn uv_default() -> Value {
    json!([0.0, 0.0])
}

/// Layer name of a `uv_map` entry; an entry without a `layer` belongs to `UVMap`.
pub(crate) fn uv_map_entry_layer(entry: &Value) -> &str {
    entry
        .get("layer")
        .and_then(Value::as_str)
        .unwrap_or("UVMap")
}

/// Whether `mesh` stores any UV layer.
pub(crate) fn has_uv_layers(mesh: &Mesh) -> bool {
    ["blender_uv_layers", "uv_map"].iter().any(|key| {
        mesh.attributes
            .get(*key)
            .and_then(Value::as_array)
            .is_some_and(|entries| !entries.is_empty())
    })
}

/// Parses one `[u, v]` value.
fn uv_pair(value: &Value) -> Option<[f64; 2]> {
    let pair = value.as_array()?;
    match pair.as_slice() {
        [u, v] => Some([u.as_f64()?, v.as_f64()?]),
        _ => None,
    }
}

/// Splits loop-order corner values into per-face rows; `None` when the count is not the mesh's.
fn split_by_face(mesh: &Mesh, values: &[Value]) -> Option<Vec<Option<Vec<[f64; 2]>>>> {
    let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    if values.len() != corners {
        return None;
    }
    let mut next = values.iter();
    mesh.faces
        .iter()
        .map(|face| {
            (0..face.vertices.len())
                .map(|_| next.next().and_then(uv_pair))
                .collect::<Option<Vec<_>>>()
                .map(Some)
        })
        .collect()
}

/// Every UV layer of `mesh`. A `blender_uv_layers` layer is read when its value count matches the
/// corners; a `uv_map` layer that `blender_uv_layers` does not list is read per face entry, and a
/// face without a valid entry is `None`.
pub(crate) fn read_layers(mesh: &Mesh) -> Vec<UvLayer> {
    let mut layers = Vec::new();
    for layer in mesh
        .attributes
        .get("blender_uv_layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let (Some(name), Some(values)) = (
            layer.get("name").and_then(Value::as_str),
            layer.get("values").and_then(Value::as_array),
        ) else {
            continue;
        };
        if let Some(faces) = split_by_face(mesh, values) {
            layers.push(UvLayer {
                name: name.to_owned(),
                faces,
            });
        }
    }
    let listed: Vec<String> = layers.iter().map(|layer| layer.name.clone()).collect();
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    for entry in mesh
        .attributes
        .get("uv_map")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = uv_map_entry_layer(entry);
        if listed.iter().any(|known| known == name) {
            continue;
        }
        let position = layers
            .iter()
            .position(|layer| layer.name == name)
            .unwrap_or_else(|| {
                layers.push(UvLayer {
                    name: name.to_owned(),
                    faces: vec![None; mesh.faces.len()],
                });
                layers.len() - 1
            });
        let layer = &mut layers[position];
        let Some(&face) = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|id| face_index.get(&id))
        else {
            continue;
        };
        let uv: Option<Vec<[f64; 2]>> = entry
            .get("uv")
            .and_then(Value::as_array)
            .and_then(|row| row.iter().map(uv_pair).collect());
        if let Some(uv) = uv.filter(|uv| uv.len() == mesh.faces[face].vertices.len()) {
            layer.faces[face] = Some(uv);
        }
    }
    layers
}

/// Writes `layers` into every existing store of each named layer: its `blender_uv_layers` entry
/// (corner values in loop order, `[0, 0]` for a face without a value), its `uv_map` entries (only
/// those whose face has a value), and its UV-named `FLOAT2` `CORNER` `blender_attributes` entry.
/// A layer that `mesh` does not store is not created.
pub(crate) fn write_layers(mesh: &mut Mesh, layers: &[UvLayer]) {
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    let flat = |layer: &UvLayer| -> Vec<Value> {
        layer
            .faces
            .iter()
            .zip(&mesh.faces)
            .flat_map(|(values, face)| {
                (0..face.vertices.len()).map(move |corner| match values {
                    Some(values) => json!(values[corner]),
                    None => uv_default(),
                })
            })
            .collect()
    };
    for layer in layers
        .iter()
        .filter(|layer| layer.faces.len() == mesh.faces.len())
    {
        if let Some(stored) = mesh
            .attributes
            .get_mut("blender_uv_layers")
            .and_then(Value::as_array_mut)
        {
            for stored in stored
                .iter_mut()
                .filter(|stored| stored.get("name").and_then(Value::as_str) == Some(&layer.name))
            {
                stored["values"] = Value::Array(flat(layer));
            }
        }
        if let Some(entries) = mesh
            .attributes
            .get_mut("uv_map")
            .and_then(Value::as_array_mut)
        {
            for entry in entries
                .iter_mut()
                .filter(|entry| uv_map_entry_layer(entry) == layer.name)
            {
                let Some(&face) = entry
                    .get("face_id")
                    .and_then(Value::as_u64)
                    .and_then(|id| face_index.get(&id))
                else {
                    continue;
                };
                if let Some(values) = &layer.faces[face] {
                    entry["uv"] = json!(values);
                }
            }
        }
        if let Some(entries) = mesh
            .attributes
            .get_mut("blender_attributes")
            .and_then(Value::as_array_mut)
        {
            for entry in entries.iter_mut().filter(|entry| {
                entry.get("name").and_then(Value::as_str) == Some(&layer.name)
                    && entry.get("domain").and_then(Value::as_str) == Some("CORNER")
                    && entry.get("data_type").and_then(Value::as_str) == Some("FLOAT2")
            }) {
                entry["values"] = Value::Array(flat(layer));
            }
        }
    }
}
