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

use std::collections::{HashMap, HashSet};

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

    /// Resolves a layer reference. `None` names the render layer. A name that no layer has falls
    /// back to the render layer and is returned as the second element, so the caller can report
    /// `uv.layer_missing` instead of failing silently.
    pub(crate) fn resolve(&self, reference: Option<&str>) -> (Option<usize>, Option<String>) {
        match reference {
            None => (self.render_index(), None),
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

/// Upgrades one schema-version-1 mesh in place. Version 1 kept layers in two places: the
/// `blender_uv_layers` array (Blender's V-up values, one row per loop) and the `uv_map` entries
/// (Potter's values). `uv_map` entries override the face row of the same-named Blender layer, as
/// the version-1 reader did. A `uv_map` entry with a `layer` key came from a Blender-layer code path,
/// so its values are flipped. The layer that `uv_map` named becomes the render and active layer.
/// Blender-layer rows are flipped into Potter's convention. Duplicate `blender_attributes` entries
/// for a layer are dropped, because the store now holds those values.
type LegacyLayerRows = (String, Vec<Option<Vec<Value>>>, Vec<Option<Vec<Value>>>);

pub(crate) fn upgrade_mesh_v1(mesh: &mut Value) -> Result<()> {
    let Some(object) = mesh.as_object_mut() else {
        return Ok(());
    };
    let faces: Vec<(u64, usize)> = object
        .get("faces")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|face| Some((face.get("id")?.as_u64()?, face.get("v")?.as_array()?.len())))
        .collect();
    let attributes = object
        .entry("attributes")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(attributes) = attributes.as_object_mut() else {
        return Ok(());
    };
    let blender_layers = match attributes.remove("blender_uv_layers") {
        Some(Value::Array(layers)) => layers,
        Some(_) => {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "version-1 blender_uv_layers must be an array",
            ));
        }
        None => Vec::new(),
    };
    let uv_map = match attributes.remove("uv_map") {
        Some(Value::Array(entries)) => entries,
        Some(_) => {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "version-1 uv_map must be an array",
            ));
        }
        None => Vec::new(),
    };
    if blender_layers.is_empty() && uv_map.is_empty() {
        return Ok(());
    }

    let corners: usize = faces.iter().map(|(_, count)| count).sum();
    let face_row = |id: u64| faces.iter().position(|(face, _)| *face == id);
    let mut layers = Vec::<LegacyLayerRows>::new();
    for layer in blender_layers {
        let name = layer
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "version-1 Blender UV layer has no name",
                )
            })?;
        let values = layer
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    format!("version-1 Blender UV layer `{name}` has no values"),
                )
            })?;
        if values.len() != corners {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                format!("version-1 Blender UV layer `{name}` has the wrong corner count"),
            ));
        }
        let mut next = values.iter();
        let mut rows = Vec::with_capacity(faces.len());
        for (_, count) in &faces {
            let row = (0..*count)
                .map(|_| {
                    let [u, v] = uv_pair(next.next().ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "version-1 UV row is truncated")
                    })?)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            format!("version-1 Blender UV layer `{name}` has an invalid value"),
                        )
                    })?;
                    Ok(json!([u, 1.0 - v]))
                })
                .collect::<Result<Vec<_>>>()?;
            rows.push(Some(row));
        }
        if layers.iter().any(|(existing, _, _)| existing == name) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                format!("version-1 Blender UV layer name `{name}` is duplicated"),
            ));
        }
        layers.push((name.to_owned(), rows, vec![None; faces.len()]));
    }

    let mut render = None;
    for entry in uv_map {
        let flip = entry.get("layer").is_some();
        let name = match entry.get("layer") {
            Some(Value::String(name)) if !name.is_empty() => name.clone(),
            Some(_) => {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "version-1 uv_map layer name is invalid",
                ));
            }
            None => "UVMap".to_owned(),
        };
        render.get_or_insert_with(|| name.clone());
        let position = layers
            .iter()
            .position(|(layer, _, _)| *layer == name)
            .unwrap_or_else(|| {
                layers.push((
                    name.clone(),
                    vec![None; faces.len()],
                    vec![None; faces.len()],
                ));
                layers.len() - 1
            });
        let face_id = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "version-1 UV face ID is invalid")
            })?;
        let row = face_row(face_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "version-1 UV map references a missing face",
            )
        })?;
        let count = faces[row].1;
        let uv_values = entry
            .get("uv")
            .and_then(Value::as_array)
            .filter(|uv| uv.len() == count)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    format!("version-1 UV layer `{name}` has the wrong corner count"),
                )
            })?;
        let uv = uv_values
            .iter()
            .map(|value| {
                let [u, v] = uv_pair(value).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        format!("version-1 UV layer `{name}` has an invalid value"),
                    )
                })?;
                Ok(json!([u, if flip { 1.0 - v } else { v }]))
            })
            .collect::<Result<Vec<_>>>()?;
        let pinned = match entry.get("pinned") {
            Some(Value::Array(flags))
                if flags.len() == count && flags.iter().all(Value::is_boolean) =>
            {
                Some(flags.clone())
            }
            Some(_) => {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    format!("version-1 UV layer `{name}` has invalid pin flags"),
                ));
            }
            None => None,
        };
        layers[position].1[row] = Some(uv);
        layers[position].2[row] = pinned;
    }

    let names: Vec<String> = layers.iter().map(|(name, _, _)| name.clone()).collect();
    if !layers.is_empty() {
        // Version 1 had no active-layer notion. The render layer, the one `uv_map` named, is the
        // edit target; otherwise the first layer is.
        let render = render
            .filter(|name| names.contains(name))
            .unwrap_or_else(|| names[0].clone());
        let stored: Vec<Value> = layers
            .iter()
            .map(|(name, rows, pins)| {
                let faces: Vec<Value> = faces
                    .iter()
                    .enumerate()
                    .filter_map(|(row, (id, _))| {
                        let uv = rows[row].as_ref()?;
                        let mut entry = json!({"face_id": id, "uv": uv});
                        if let Some(flags) = &pins[row] {
                            entry["pinned"] = Value::Array(flags.clone());
                        }
                        Some(entry)
                    })
                    .collect();
                json!({"name": name, "faces": faces})
            })
            .collect();
        attributes.insert(
            ATTRIBUTE.to_owned(),
            json!({"layers": stored, "active": render, "render": render}),
        );
        if let Some(blender_attributes) = attributes
            .get_mut("blender_attributes")
            .and_then(Value::as_array_mut)
        {
            blender_attributes.retain(|entry| {
                !(entry.get("domain").and_then(Value::as_str) == Some("CORNER")
                    && entry.get("data_type").and_then(Value::as_str) == Some("FLOAT2")
                    && entry
                        .get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| names.iter().any(|layer| layer == name)))
            });
        }
    }
    Ok(())
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
            "blender_uv_layers": [{"name": "Light", "values": [
                [0.0, 0.25], [1.0, 0.25], [1.0, 0.75],
                [0.0, 0.75], [0.5, 0.0], [0.5, 1.0]
            ]}],
            "uv_map": [{"face_id": face_zero, "uv": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}],
            "blender_attributes": [
                {"name": "Light", "domain": "CORNER", "data_type": "FLOAT2", "values": []},
                {"name": "_wetness", "domain": "POINT", "data_type": "FLOAT", "values": [0.5]}
            ]
        });

        upgrade_mesh_v1(&mut value).expect("upgrade");
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
        assert_eq!(store.active.as_deref(), Some("UVMap"));
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
}
