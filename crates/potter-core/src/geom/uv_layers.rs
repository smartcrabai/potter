// SPDX-License-Identifier: GPL-3.0-or-later

//! The mesh's UV layers, stored in the single `uv_layers` attribute:
//!
//! `{"layers": [{"name", "faces": [{"face_id", "uv": [[u, v] per corner], "pinned"?: [bool per corner]}]}],
//!   "active": name|null, "render": name|null}`
//!
//! Layers keep mesh order, which is the glTF `TEXCOORD_n` and Blender layer order. `active` is the
//! edit target and `render` the render and export default. Both name a layer, or are null when the
//! mesh has none. Values use Potter's glTF convention (v down); writers that follow Blender's
//! convention flip at their boundary. The store is keyed by persistent face ID, so it survives
//! face reordering. [`validate`] runs at load time; the accessors below assume a valid store.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde_json::{Map, Value, json};

use super::Mesh;
use crate::error::{ErrorCode, PotError, Result};

/// Attribute name of the UV store.
pub(crate) const ATTRIBUTE: &str = "uv_layers";
/// Blender's per-mesh UV layer limit.
pub(crate) const MAX_LAYERS: usize = 8;

/// One UV layer. `faces[f]` holds the corner `[u, v]` values of storage face `f` in corner order,
/// or `None` where the layer has no entry for that face. `pinned` runs parallel to `faces`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct UvLayer {
    pub(crate) name: String,
    pub(crate) faces: Vec<Option<Vec<[f64; 2]>>>,
    pub(crate) pinned: Vec<Option<Vec<bool>>>,
}

impl UvLayer {
    /// A layer with `face_count` faces and no pins.
    pub(crate) fn new(name: impl Into<String>, faces: Vec<Option<Vec<[f64; 2]>>>) -> Self {
        let pinned = vec![None; faces.len()];
        Self {
            name: name.into(),
            faces,
            pinned,
        }
    }
}

/// The whole store: layers in order plus the active and render layer names.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct UvStore {
    pub(crate) layers: Vec<UvLayer>,
    pub(crate) active: Option<String>,
    pub(crate) render: Option<String>,
}

impl UvStore {
    /// Position of the layer called `name`.
    pub(crate) fn index_of(&self, name: &str) -> Option<usize> {
        self.layers.iter().position(|layer| layer.name == name)
    }

    /// Position of the render layer.
    pub(crate) fn render_index(&self) -> Option<usize> {
        self.render.as_deref().and_then(|name| self.index_of(name))
    }

    /// Resolves a layer reference. `None` and `""` name the render layer. A name no layer has
    /// falls back to the render layer and is returned as the second element, so the caller can
    /// report `uv.layer_missing` instead of failing silently.
    pub(crate) fn resolve(&self, reference: Option<&str>) -> (Option<usize>, Option<String>) {
        match reference {
            None | Some("") => (self.render_index(), None),
            Some(name) => match self.index_of(name) {
                Some(index) => (Some(index), None),
                None => (self.render_index(), Some(name.to_owned())),
            },
        }
    }
}

/// Whether `mesh` stores any UV layer.
pub(crate) fn has_uv_layers(mesh: &Mesh) -> bool {
    layer_count(mesh) > 0
}

/// Number of UV layers `mesh` stores.
pub(crate) fn layer_count(mesh: &Mesh) -> usize {
    mesh.attributes
        .get(ATTRIBUTE)
        .and_then(|value| value.get("layers"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

/// Parses one `[u, v]` value.
fn uv_pair(value: &Value) -> Option<[f64; 2]> {
    match value.as_array()?.as_slice() {
        [u, v] => Some([u.as_f64()?, v.as_f64()?]),
        _ => None,
    }
}

/// Reads the store leniently. Malformed rows are skipped; [`validate`] reports them at load time.
pub(crate) fn read(mesh: &Mesh) -> UvStore {
    let Some(object) = mesh.attributes.get(ATTRIBUTE).and_then(Value::as_object) else {
        return UvStore::default();
    };
    let face_index: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(index, face)| (u64::from(face.id), index))
        .collect();
    let layers = object
        .get("layers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|layer| read_layer(mesh, &face_index, layer))
        .collect();
    let name = |key: &str| object.get(key).and_then(Value::as_str).map(str::to_owned);
    UvStore {
        layers,
        active: name("active"),
        render: name("render"),
    }
}

fn read_layer(mesh: &Mesh, face_index: &HashMap<u64, usize>, layer: &Value) -> Option<UvLayer> {
    let name = layer.get("name")?.as_str()?.to_owned();
    let mut result = UvLayer::new(name, vec![None; mesh.faces.len()]);
    for entry in layer.get("faces")?.as_array()? {
        let Some(&face) = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|id| face_index.get(&id))
        else {
            continue;
        };
        let corners = mesh.faces[face].vertices.len();
        let uv: Option<Vec<[f64; 2]>> = entry
            .get("uv")
            .and_then(Value::as_array)
            .and_then(|row| row.iter().map(uv_pair).collect());
        if let Some(uv) = uv.filter(|uv| uv.len() == corners) {
            result.faces[face] = Some(uv);
        }
        result.pinned[face] = entry
            .get("pinned")
            .and_then(Value::as_array)
            .and_then(|flags| flags.iter().map(Value::as_bool).collect::<Option<Vec<_>>>())
            .filter(|flags| flags.len() == corners);
    }
    Some(result)
}

/// The layers of `mesh`, in order.
pub(crate) fn read_layers(mesh: &Mesh) -> Vec<UvLayer> {
    read(mesh).layers
}

/// Drops the rows of faces that `mesh` no longer has, and keeps the active and render names
/// valid. Call after an operation removes or splits off faces. `read` already ignores rows whose
/// face is gone, so writing the store back keeps exactly the surviving corners and pins.
pub(crate) fn prune_missing_faces(mesh: &mut Mesh) {
    if has_uv_layers(mesh) {
        let store = read(mesh);
        write(mesh, &store);
    }
}

/// Writes `store` as the mesh's `uv_layers` attribute. An empty store removes the attribute. A
/// stale `active` or `render` name falls back to the first layer.
pub(crate) fn write(mesh: &mut Mesh, store: &UvStore) {
    if store.layers.is_empty() {
        mesh.attributes.remove(ATTRIBUTE);
        return;
    }
    let layers: Vec<Value> = store
        .layers
        .iter()
        .map(|layer| {
            let faces: Vec<Value> = mesh
                .faces
                .iter()
                .enumerate()
                .filter_map(|(index, face)| {
                    let uv = layer.faces.get(index)?.as_ref()?;
                    let mut entry = json!({"face_id": face.id, "uv": uv});
                    if let Some(Some(pinned)) = layer.pinned.get(index) {
                        entry["pinned"] = json!(pinned);
                    }
                    Some(entry)
                })
                .collect();
            json!({"name": layer.name, "faces": faces})
        })
        .collect();
    let named = |name: &Option<String>| {
        name.as_ref()
            .filter(|name| store.index_of(name).is_some())
            .or(store.layers.first().map(|layer| &layer.name))
            .cloned()
    };
    let mut object = Map::new();
    object.insert("layers".to_owned(), Value::Array(layers));
    object.insert("active".to_owned(), json!(named(&store.active)));
    object.insert("render".to_owned(), json!(named(&store.render)));
    mesh.attributes
        .insert(ATTRIBUTE.to_owned(), Value::Object(object));
}

/// Replaces the layers of `mesh`, keeping its active and render names where they still exist.
pub(crate) fn write_layers(mesh: &mut Mesh, layers: &[UvLayer]) {
    let mut store = read(mesh);
    store.layers = layers.to_vec();
    write(mesh, &store);
}

/// Checks the store of `mesh`. Run at load time for every data-block mesh.
pub(crate) fn validate(mesh: &Mesh) -> Result<()> {
    let Some(value) = mesh.attributes.get(ATTRIBUTE) else {
        return Ok(());
    };
    let invalid = |message: String| PotError::new(ErrorCode::SceneInvalid, message);
    let object = value
        .as_object()
        .ok_or_else(|| invalid("uv_layers must be an object".to_owned()))?;
    let layers = object
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("uv_layers.layers must be an array".to_owned()))?;
    if layers.len() > MAX_LAYERS {
        return Err(invalid(format!(
            "mesh has {} UV layers; the limit is {MAX_LAYERS}",
            layers.len()
        )));
    }
    let corners: HashMap<u64, usize> = mesh
        .faces
        .iter()
        .map(|face| (u64::from(face.id), face.vertices.len()))
        .collect();
    let mut names: Vec<&str> = Vec::new();
    for layer in layers {
        let name = layer
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("UV layer name must be a non-empty string".to_owned()))?;
        if names.contains(&name) {
            return Err(invalid(format!("UV layer name `{name}` is not unique")));
        }
        names.push(name);
        let entries = layer
            .get("faces")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("UV layer `{name}` faces must be an array")))?;
        let mut seen = HashSet::new();
        for entry in entries {
            let face_id = entry
                .get("face_id")
                .and_then(Value::as_u64)
                .filter(|id| corners.contains_key(id))
                .ok_or_else(|| invalid(format!("UV layer `{name}` references a missing face")))?;
            if !seen.insert(face_id) {
                return Err(invalid(format!("UV layer `{name}` repeats a face")));
            }
            let count = corners[&face_id];
            let uv = entry
                .get("uv")
                .and_then(Value::as_array)
                .filter(|uv| uv.len() == count)
                .ok_or_else(|| {
                    invalid(format!(
                        "UV layer `{name}` corner count does not match its face"
                    ))
                })?;
            let finite =
                |pair: &Value| uv_pair(pair).is_some_and(|[u, v]| u.is_finite() && v.is_finite());
            if !uv.iter().all(finite) {
                return Err(invalid(format!(
                    "UV layer `{name}` has an invalid coordinate"
                )));
            }
            if let Some(pinned) = entry.get("pinned") {
                let flags_ok = pinned.as_array().is_some_and(|flags| {
                    flags.len() == count && flags.iter().all(Value::is_boolean)
                });
                if !flags_ok {
                    return Err(invalid(format!(
                        "UV layer `{name}` pinned flags must match its face corners"
                    )));
                }
            }
        }
    }
    for key in ["active", "render"] {
        let value = object.get(key).unwrap_or(&Value::Null);
        let valid = match value {
            Value::Null => names.is_empty(),
            Value::String(name) => names.contains(&name.as_str()),
            _ => false,
        };
        if !valid {
            return Err(invalid(format!("uv_layers.{key} must name a UV layer")));
        }
    }
    Ok(())
}

/// Upgrades one schema-version-1 mesh in place and returns the diagnostics for the legacy data it
/// could not keep. `mesh_id` names the mesh in the diagnostics. `referenced` holds the layer names
/// that material textures use.
///
/// Version 1 kept layers in two places: the `blender_uv_layers` array (Blender's V-up values, one
/// row per corner loop) and `uv_map` entries (Potter's values). The rules follow the version-1
/// reader (`geom/uv_layers.rs` at c4328de):
/// - A Blender layer is read only when it has a unique non-empty name, a `values` array with one
///   `[u, v]` per mesh corner, and no malformed value. Any other Blender layer is skipped with a
///   `uv.legacy_layer_skipped` warning. Kept values are flipped into Potter's convention.
/// - A `uv_map` entry belongs to the layer its `layer` key names, or to `UVMap` when that key is
///   absent or not a string. The layer exists once an entry names it, even if that entry's row is
///   unusable. An entry with a `layer` key came from a Blender-layer code path, so its values are
///   flipped.
/// - A `uv_map` row needs an existing face and one `[u, v]` per corner. Any other row is skipped
///   with a `uv.legacy_row_skipped` warning. When two rows name the same face and layer, the later
///   row replaces the earlier one and the replacement is reported.
/// - A `uv_map` row overrides the same face of the same-named Blender layer. That precedence is the
///   version-1 behaviour and is not reported.
/// - Pins are kept when they are one boolean per corner. Otherwise the row keeps its UVs, its pins
///   are dropped, and a `uv.legacy_pins_dropped` warning is reported.
/// - The first layer that `uv_map` names is the render layer; without `uv_map`, the first layer is.
/// - A valid `blender_uv_layers[].active` flag selects the active layer independently; when none
///   survives, the render layer is active.
/// - At most [`MAX_LAYERS`] layers are kept. Priority: the render layer, then the layers
///   `referenced` names, then the valid legacy active layer, then the rest in order. Kept layers
///   keep their original order. Each dropped layer is reported with a `uv.legacy_layer_dropped` warning.
/// - `blender_attributes` entries that are FLOAT2 `CORNER` attributes named like a version-1 layer
///   are removed, because the store now holds those values.
///
/// Nothing is dropped silently: each skipped or dropped item has a warning, and the original
/// document keeps the bytes.
pub(crate) fn upgrade_mesh_v1(
    mesh: &mut Value,
    mesh_id: &str,
    referenced: &BTreeSet<String>,
) -> Vec<Value> {
    let Some(object) = mesh.as_object_mut() else {
        return Vec::new();
    };
    let faces: Vec<(u64, usize)> = object
        .get("faces")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|face| Some((face.get("id")?.as_u64()?, face.get("v")?.as_array()?.len())))
        .collect();
    let Some(attributes) = object
        .entry("attributes")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
    else {
        return Vec::new();
    };
    let mut warnings = Vec::new();
    let blender_layers = match attributes.remove("blender_uv_layers") {
        Some(Value::Array(layers)) => layers,
        Some(_) => {
            warnings.push(skipped_layer(
                mesh_id,
                None,
                "not_array",
                format!("version-1 blender_uv_layers on mesh `{mesh_id}` is not an array; no Blender UV layer was read"),
            ));
            Vec::new()
        }
        None => Vec::new(),
    };
    let uv_map = match attributes.remove("uv_map") {
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            warnings.push(skipped_row(
                mesh_id,
                None,
                None,
                "not_array",
                format!(
                    "version-1 uv_map on mesh `{mesh_id}` is not an array; no uv_map row was read"
                ),
            ));
            Vec::new()
        }
        None => Vec::new(),
    };

    let corners: usize = faces.iter().map(|(_, count)| count).sum();
    // Every layer name a version-1 record used, kept or not. Their FLOAT2 `blender_attributes`
    // duplicates are removed below.
    let mut names = BTreeSet::new();
    let mut layers: Vec<UvLayer> = Vec::new();
    let mut legacy_active = None;
    for (index, layer) in blender_layers.iter().enumerate() {
        let Some(name) = layer
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
        else {
            warnings.push(skipped_layer(
                mesh_id,
                None,
                "invalid_name",
                format!("version-1 Blender UV layer {index} on mesh `{mesh_id}` has no name; it was skipped"),
            ));
            continue;
        };
        names.insert(name.to_owned());
        let Some(values) = layer.get("values").and_then(Value::as_array) else {
            warnings.push(skipped_layer(
                mesh_id,
                Some(name),
                "missing_values",
                format!("version-1 Blender UV layer `{name}` on mesh `{mesh_id}` has no values array; it was skipped"),
            ));
            continue;
        };
        if values.len() != corners {
            warnings.push(skipped_layer(
                mesh_id,
                Some(name),
                "corner_count",
                format!(
                    "version-1 Blender UV layer `{name}` on mesh `{mesh_id}` has {} values for {corners} corners; it was skipped",
                    values.len()
                ),
            ));
            continue;
        }
        let Some(pairs) = values
            .iter()
            .map(uv_pair)
            .collect::<Option<Vec<[f64; 2]>>>()
        else {
            warnings.push(skipped_layer(
                mesh_id,
                Some(name),
                "invalid_value",
                format!("version-1 Blender UV layer `{name}` on mesh `{mesh_id}` has a value that is not a [u, v] pair; it was skipped"),
            ));
            continue;
        };
        if layers.iter().any(|kept| kept.name == name) {
            warnings.push(skipped_layer(
                mesh_id,
                Some(name),
                "duplicate_name",
                format!("version-1 Blender UV layer `{name}` on mesh `{mesh_id}` repeats an earlier layer of that name; it was skipped"),
            ));
            continue;
        }
        // Blender's V runs up; the store runs down.
        let mut flipped = pairs.into_iter().map(|[u, v]| [u, 1.0 - v]);
        let rows: Vec<Option<Vec<[f64; 2]>>> = faces
            .iter()
            .map(|(_, count)| Some(flipped.by_ref().take(*count).collect()))
            .collect();
        let is_active = layer.get("active").and_then(Value::as_bool) == Some(true);
        let layer_index = layers.len();
        layers.push(UvLayer::new(name, rows));
        if is_active && legacy_active.is_none() {
            legacy_active = Some(layer_index);
        }
    }

    let mut render: Option<String> = None;
    let mut mapped: HashSet<(usize, usize)> = HashSet::new();
    for entry in &uv_map {
        if !entry.is_object() {
            warnings.push(skipped_row(
                mesh_id,
                None,
                None,
                "not_object",
                format!(
                    "version-1 uv_map entry on mesh `{mesh_id}` is not an object; it was skipped"
                ),
            ));
            continue;
        }
        let face_id = entry.get("face_id").and_then(Value::as_u64);
        // A missing or non-string `layer` belongs to `UVMap`, as the version-1 reader had it.
        let layer_key = entry.get("layer");
        let flip = layer_key.is_some();
        let name = layer_key.and_then(Value::as_str).unwrap_or("UVMap");
        if name.is_empty() {
            warnings.push(skipped_row(
                mesh_id,
                None,
                face_id,
                "empty_layer",
                format!("version-1 uv_map entry on mesh `{mesh_id}` has an empty layer name; it was skipped"),
            ));
            continue;
        }
        render.get_or_insert_with(|| name.to_owned());
        names.insert(name.to_owned());
        let position = if let Some(position) = layers.iter().position(|layer| layer.name == name) {
            position
        } else {
            layers.push(UvLayer::new(name, vec![None; faces.len()]));
            layers.len() - 1
        };
        let Some(face_id) = face_id else {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                None,
                "face_id_invalid",
                format!("version-1 uv_map row of layer `{name}` on mesh `{mesh_id}` has no valid face ID; it was skipped"),
            ));
            continue;
        };
        let Some(row) = faces.iter().position(|(id, _)| *id == face_id) else {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                Some(face_id),
                "face_missing",
                format!("version-1 uv_map row for face {face_id} of layer `{name}` on mesh `{mesh_id}` names a face the mesh no longer has; it was skipped"),
            ));
            continue;
        };
        let count = faces[row].1;
        let Some(uv_values) = entry.get("uv").and_then(Value::as_array) else {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                Some(face_id),
                "invalid_uv",
                format!("version-1 uv_map row for face {face_id} of layer `{name}` on mesh `{mesh_id}` has no uv array; it was skipped"),
            ));
            continue;
        };
        if uv_values.len() != count {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                Some(face_id),
                "uv_length",
                format!(
                    "version-1 uv_map row for face {face_id} of layer `{name}` on mesh `{mesh_id}` has {} corner values for {count} corners; it was skipped",
                    uv_values.len()
                ),
            ));
            continue;
        }
        let Some(uv) = uv_values
            .iter()
            .map(uv_pair)
            .collect::<Option<Vec<[f64; 2]>>>()
        else {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                Some(face_id),
                "invalid_uv",
                format!("version-1 uv_map row for face {face_id} of layer `{name}` on mesh `{mesh_id}` has a corner value that is not a [u, v] pair; it was skipped"),
            ));
            continue;
        };
        let uv: Vec<[f64; 2]> = if flip {
            uv.into_iter().map(|[u, v]| [u, 1.0 - v]).collect()
        } else {
            uv
        };
        let pinned = match entry.get("pinned") {
            None | Some(Value::Null) => None,
            Some(flags) => {
                let parsed = flags
                    .as_array()
                    .filter(|flags| flags.len() == count)
                    .and_then(|flags| {
                        flags
                            .iter()
                            .map(Value::as_bool)
                            .collect::<Option<Vec<bool>>>()
                    });
                if parsed.is_none() {
                    warnings.push(json!({
                        "code": "uv.legacy_pins_dropped",
                        "message": format!("version-1 uv_map pins for face {face_id} of layer `{name}` on mesh `{mesh_id}` are not one boolean per corner; the UVs were kept and the pins dropped"),
                        "mesh": mesh_id,
                        "layer": name,
                        "face_id": face_id,
                    }));
                }
                parsed
            }
        };
        if !mapped.insert((position, row)) {
            warnings.push(skipped_row(
                mesh_id,
                Some(name),
                Some(face_id),
                "superseded",
                format!("version-1 uv_map row for face {face_id} of layer `{name}` on mesh `{mesh_id}` repeats an earlier row; the earlier row was replaced"),
            ));
        }
        layers[position].faces[row] = Some(uv);
        layers[position].pinned[row] = pinned;
    }

    if !layers.is_empty() {
        // The first `uv_map` layer is render; a surviving legacy active flag selects active.
        let render_index = render
            .as_deref()
            .and_then(|name| layers.iter().position(|layer| layer.name == name))
            .unwrap_or(0);
        let kept = kept_layers(&layers, render_index, legacy_active, referenced);
        for (index, layer) in layers.iter().enumerate() {
            if !kept.contains(&index) {
                warnings.push(json!({
                    "code": "uv.legacy_layer_dropped",
                    "message": format!("version-1 UV layer `{}` on mesh `{mesh_id}` is past the {MAX_LAYERS}-layer limit; it was dropped from the live store and is kept in the original document", layer.name),
                    "mesh": mesh_id,
                    "layer": layer.name,
                    "reason": "layer_limit",
                    "referenced": referenced.contains(&layer.name),
                }));
            }
        }
        let stored: Vec<Value> = kept
            .iter()
            .map(|&index| layer_json(&layers[index], &faces))
            .collect();
        let render_name = layers[render_index].name.clone();
        let active_name = legacy_active
            .filter(|index| kept.contains(index))
            .map_or_else(|| render_name.clone(), |index| layers[index].name.clone());
        attributes.insert(
            ATTRIBUTE.to_owned(),
            json!({"layers": stored, "active": active_name, "render": render_name}),
        );
    }
    if let Some(entries) = attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
    {
        entries.retain(|entry| {
            !(entry.get("domain").and_then(Value::as_str) == Some("CORNER")
                && entry.get("data_type").and_then(Value::as_str) == Some("FLOAT2")
                && entry
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| names.contains(name)))
        });
    }
    warnings
}

/// A `uv.legacy_layer_skipped` warning for a version-1 Blender layer (or layer entry).
fn skipped_layer(
    mesh_id: &str,
    layer: Option<&str>,
    reason: &str,
    message: impl Into<String>,
) -> Value {
    json!({
        "code": "uv.legacy_layer_skipped",
        "message": message.into(),
        "mesh": mesh_id,
        "layer": layer,
        "reason": reason,
    })
}

/// A `uv.legacy_row_skipped` warning for a version-1 `uv_map` entry.
fn skipped_row(
    mesh_id: &str,
    layer: Option<&str>,
    face_id: Option<u64>,
    reason: &str,
    message: impl Into<String>,
) -> Value {
    json!({
        "code": "uv.legacy_row_skipped",
        "message": message.into(),
        "mesh": mesh_id,
        "layer": layer,
        "face_id": face_id,
        "reason": reason,
    })
}

/// The layers the layer limit keeps, as indices in their original order. Priority: the render
/// layer, then the layers `referenced` names, then the valid legacy active layer, then the rest.
fn kept_layers(
    layers: &[UvLayer],
    render: usize,
    legacy_active: Option<usize>,
    referenced: &BTreeSet<String>,
) -> Vec<usize> {
    let is_referenced = |index: &usize| referenced.contains(&layers[*index].name);
    let is_active = |index: &usize| Some(*index) == legacy_active;
    let others = (0..layers.len()).filter(|index| *index != render);
    let mut order: Vec<usize> = std::iter::once(render)
        .chain(others.clone().filter(is_referenced))
        .chain(
            others
                .clone()
                .filter(|index| is_active(index) && !is_referenced(index)),
        )
        .chain(others.filter(|index| !is_referenced(index) && !is_active(index)))
        .take(MAX_LAYERS)
        .collect();
    order.sort_unstable();
    order
}

/// The stored form of one layer: one entry per face with values, carrying its pins when known.
fn layer_json(layer: &UvLayer, faces: &[(u64, usize)]) -> Value {
    let entries: Vec<Value> = faces
        .iter()
        .zip(layer.faces.iter().zip(&layer.pinned))
        .filter_map(|((id, _), (uv, pinned))| {
            let uv = uv.as_ref()?;
            let mut entry = json!({"face_id": id, "uv": uv});
            if let Some(flags) = pinned {
                entry["pinned"] = json!(flags);
            }
            Some(entry)
        })
        .collect();
    json!({"name": layer.name, "faces": entries})
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "UV migration unit fixtures")]
    use glam::DVec3;
    use serde_json::json;

    use super::*;

    fn two_triangles() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )
        .expect("two triangles")
    }

    /// Version-1 layers: the Blender layer `Light` (V-up, one row per loop) and a `uv_map` face
    /// row for face 0 that names no layer, so it belongs to `UVMap`.
    #[test]
    fn version_one_layers_upgrade_to_one_store() {
        let mesh = two_triangles();
        let face_zero = mesh.faces[0].id;
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        value["attributes"] = json!({
            "blender_uv_layers": [{"name": "Light", "active": true, "values": [
                [0.0, 0.25], [1.0, 0.25], [1.0, 0.75],
                [0.0, 0.75], [0.5, 0.0], [0.5, 1.0]
            ]}],
            "uv_map": [{"face_id": face_zero, "uv": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}],
            "blender_attributes": [
                {"name": "Light", "domain": "CORNER", "data_type": "FLOAT2", "values": []},
                {"name": "_wetness", "domain": "POINT", "data_type": "FLOAT", "values": [0.5]}
            ]
        });

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &BTreeSet::new());
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!(value["attributes"].get("uv_map").is_none());
        assert!(value["attributes"].get("blender_uv_layers").is_none());
        let attributes = value["attributes"]["blender_attributes"]
            .as_array()
            .expect("blender attributes");
        assert_eq!(
            attributes.len(),
            1,
            "UV duplicate dropped, user attribute kept"
        );
        assert_eq!(attributes[0]["name"], json!("_wetness"));

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        validate(&upgraded).expect("upgraded store is valid");
        let store = read(&upgraded);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Light", "UVMap"]);
        assert_eq!(store.render.as_deref(), Some("UVMap"));
        assert_eq!(store.active.as_deref(), Some("Light"));
        // Blender rows are flipped into storage convention; `uv_map` rows are not.
        assert_eq!(
            store.layers[0].faces,
            vec![
                Some(vec![[0.0, 0.75], [1.0, 0.75], [1.0, 0.25]]),
                Some(vec![[0.0, 0.25], [0.5, 1.0], [0.5, 0.0]]),
            ]
        );
        assert_eq!(
            store.layers[1].faces,
            vec![Some(vec![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), None]
        );
    }

    #[test]
    fn store_resolves_missing_names_to_the_render_layer() {
        let store = UvStore {
            layers: vec![
                UvLayer::new("UVMap", vec![None]),
                UvLayer::new("Light", vec![None]),
            ],
            active: Some("UVMap".to_owned()),
            render: Some("Light".to_owned()),
        };
        assert_eq!(store.resolve(None), (Some(1), None));
        assert_eq!(store.resolve(Some("UVMap")), (Some(0), None));
        assert_eq!(
            store.resolve(Some("Gone")),
            (Some(1), Some("Gone".to_owned()))
        );
    }

    /// A stale Blender array (4 values for 6 corners) and `uv_map` rows that name a missing face or
    /// have the wrong length. Each is skipped with its own warning; the valid row survives, and the
    /// `uv_map` entry for the missing face still names its layer.
    #[test]
    fn stale_legacy_data_is_skipped_with_precise_warnings() {
        let mesh = two_triangles();
        let (first, second) = (mesh.faces[0].id, mesh.faces[1].id);
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        value["attributes"] = json!({
            "blender_uv_layers": [{"name": "Light", "active": true, "values": [
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
            ]}],
            "uv_map": [
                {"face_id": 4242, "uv": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]},
                {"face_id": first, "uv": [[0.1, 0.2], [0.3, 0.4]]},
                {"face_id": second, "uv": [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]]}
            ]
        });

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &BTreeSet::new());
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert_eq!(warnings[0]["code"], "uv.legacy_layer_skipped");
        assert_eq!(warnings[0]["layer"], "Light");
        assert_eq!(warnings[0]["reason"], "corner_count");
        assert_eq!(warnings[0]["mesh"], "mesh");
        assert_eq!(warnings[1]["code"], "uv.legacy_row_skipped");
        assert_eq!(warnings[1]["layer"], "UVMap");
        assert_eq!(warnings[1]["face_id"], 4242);
        assert_eq!(warnings[1]["reason"], "face_missing");
        assert_eq!(warnings[2]["code"], "uv.legacy_row_skipped");
        assert_eq!(warnings[2]["face_id"], first);
        assert_eq!(warnings[2]["reason"], "uv_length");

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        validate(&upgraded).expect("upgraded store is valid");
        let store = read(&upgraded);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["UVMap"]);
        assert_eq!(store.active.as_deref(), Some("UVMap"));
        assert_eq!(
            store.layers[0].faces,
            vec![None, Some(vec![[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]]),]
        );
    }

    /// Eight Blender layers `L0`..`L7` and `Extra` from a `uv_map` entry. The first `uv_map` name,
    /// `L2`, is the render layer, and material texture `L7` references a layer. The limit keeps the
    /// render layer, then the referenced layer, then the rest in order, so `Extra` is dropped.
    #[test]
    fn layer_limit_keeps_render_then_referenced_then_original_order() {
        let mesh = two_triangles();
        let face = mesh.faces[0].id;
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        let blender: Vec<Value> = (0..8)
            .map(|index| {
                json!({"name": format!("L{index}"), "values": [
                    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.25, 0.75]
                ]})
            })
            .collect();
        value["attributes"] = json!({
            "blender_uv_layers": blender,
            "uv_map": [
                {"face_id": face, "layer": "L2", "uv": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]},
                {"face_id": face, "layer": "Extra", "uv": [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]}
            ]
        });
        let referenced = BTreeSet::from(["L7".to_owned()]);

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &referenced);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0]["code"], "uv.legacy_layer_dropped");
        assert_eq!(warnings[0]["layer"], "Extra");
        assert_eq!(warnings[0]["referenced"], false);

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        validate(&upgraded).expect("upgraded store is valid");
        let store = read(&upgraded);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]);
        assert_eq!(store.render.as_deref(), Some("L2"));
        assert_eq!(store.active.as_deref(), Some("L2"));
    }

    /// Nine Blender layers and no `uv_map`: the first layer is the render layer. Referenced `L8`
    /// outranks original order, so `L7`, not `L8`, is the layer that is dropped.
    #[test]
    fn referenced_layer_outranks_original_order_under_the_limit() {
        let mesh = two_triangles();
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        let blender: Vec<Value> = (0..9)
            .map(|index| {
                json!({"name": format!("L{index}"), "values": [
                    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.25, 0.75]
                ]})
            })
            .collect();
        value["attributes"] = json!({"blender_uv_layers": blender});
        let referenced = BTreeSet::from(["L8".to_owned()]);

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &referenced);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0]["layer"], "L7");
        assert_eq!(warnings[0]["referenced"], false);

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        let store = read(&upgraded);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L8"]);
        assert_eq!(store.render.as_deref(), Some("L0"));
    }

    /// Eight Blender layers, with `L7` active, and a distinct `uv_map` render layer. The active
    /// layer outranks ordinary source order under the layer limit; stored layers remain ordered.
    #[test]
    fn active_layer_outranks_original_order_under_the_limit() {
        let mesh = two_triangles();
        let face = mesh.faces[0].id;
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        let blender: Vec<Value> = (0..8)
            .map(|index| {
                json!({"name": format!("L{index}"), "active": index == 7, "values": [
                    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.25, 0.75]
                ]})
            })
            .collect();
        value["attributes"] = json!({
            "blender_uv_layers": blender,
            "uv_map": [{"face_id": face, "layer": "Render", "uv": [
                [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]
            ]}]
        });

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &BTreeSet::new());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert_eq!(warnings[0]["code"], "uv.legacy_layer_dropped");
        assert_eq!(warnings[0]["layer"], "L6");

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        let store = read(&upgraded);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["L0", "L1", "L2", "L3", "L4", "L5", "L7", "Render"]);
        assert_eq!(store.active.as_deref(), Some("L7"));
        assert_eq!(store.render.as_deref(), Some("Render"));
    }

    /// Pins of the wrong length are dropped while the row's UVs are kept, and a later row for the
    /// same face and layer replaces the earlier one.
    #[test]
    fn invalid_pins_drop_only_pins_and_later_rows_replace_earlier_ones() {
        let mesh = two_triangles();
        let face = mesh.faces[0].id;
        let mut value = serde_json::to_value(&mesh).expect("mesh json");
        value["attributes"] = json!({
            "uv_map": [
                {"face_id": face, "uv": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], "pinned": [true, true]},
                {"face_id": face, "uv": [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], "pinned": [true, false, true]}
            ]
        });

        let warnings = upgrade_mesh_v1(&mut value, "mesh", &BTreeSet::new());
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert_eq!(warnings[0]["code"], "uv.legacy_pins_dropped");
        assert_eq!(warnings[0]["face_id"], face);
        assert_eq!(warnings[1]["code"], "uv.legacy_row_skipped");
        assert_eq!(warnings[1]["reason"], "superseded");

        let upgraded: Mesh = serde_json::from_value(value).expect("upgraded mesh parses");
        validate(&upgraded).expect("upgraded store is valid");
        let store = read(&upgraded);
        assert_eq!(
            store.layers[0].faces[0],
            Some(vec![[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]])
        );
        assert_eq!(store.layers[0].pinned[0], Some(vec![true, false, true]));
    }
}
