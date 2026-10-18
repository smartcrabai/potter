use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Face, Mesh,
        uv_layers::{self, UvLayer, UvStore},
    },
    graph::GraphNode,
    model::{AvatarMtoon, Id, Material, Modifier, SceneDoc, TextureRef},
};

use super::{ChangeKind, Engine, check_fields, resolve_node_targets};

/// Blender's default margin for `uv.pack_islands` (Blender 5.2.2 oracle). Schema and runtime agree.
const DEFAULT_PACK_MARGIN: f64 = 0.001;
/// Blender's name for a new UV map. Unnamed layers take `.001`-style suffixes from it.
const DEFAULT_LAYER_NAME: &str = "UVMap";
/// Modifier params that name a UV layer. The layer-select params of the Blender modifiers are fixed
/// enums (`ALL`, `ACTIVE`, `NAME`, `INDEX`), not layer names, so they are not listed.
const MODIFIER_LAYER_PARAMS: [&str; 2] = ["uv_layer", "mask_tex_uv_layer"];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "uv.unwrap" => unwrap(engine, operation),
        "uv.transform" => transform(engine, operation),
        "uv.pin" => pin(engine, operation),
        "uv.pack" => pack(engine, operation),
        "uv.layer_create" => layer_create(engine, operation),
        "uv.layer_delete" => layer_delete(engine, operation),
        "uv.layer_rename" => layer_rename(engine, operation),
        "uv.layer_set_active" => layer_set_role(engine, operation, LayerRole::Active),
        "uv.layer_set_render" => layer_set_role(engine, operation, LayerRole::Render),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported UV operation `{name}`"),
            "/op",
        )),
    }
}

fn unwrap(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "method", "uv_layer"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let layer = uv_layer_field(engine, operation)?;
    let method = operation
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("box");
    if !["box", "smart", "cube", "angle_based", "conformal"].contains(&method) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "method must be box, smart, cube, angle_based, or conformal",
            "/method",
        ));
    }
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        edit_layer_map(mesh, layer.as_deref(), true, |layer_mesh, uv_map| {
            if matches!(method, "angle_based" | "conformal") {
                unwrap_advanced(layer_mesh, &selected, uv_map, method)?;
            } else {
                let projected = if matches!(method, "box" | "cube") {
                    cube_project(layer_mesh, &selected)?
                } else {
                    layer_mesh
                        .faces
                        .iter()
                        .filter(|face| selected.contains(&face.id))
                        .map(|face| -> Result<(u32, Vec<[f64; 2]>)> {
                            Ok((face.id, smart_project_face(layer_mesh, face)?))
                        })
                        .collect::<Result<BTreeMap<u32, Vec<[f64; 2]>>>>()?
                };
                for (face_id, corners) in projected {
                    let pinned = uv_map
                        .get(&face_id)
                        .and_then(|face_uv| face_uv.pinned.clone())
                        .filter(|pinned| pinned.len() == corners.len());
                    uv_map.insert(
                        face_id,
                        UvFace {
                            uv: corners,
                            pinned,
                        },
                    );
                }
            }
            Ok(())
        })
    })
}

fn transform(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "elements",
            "scope",
            "translation",
            "scale",
            "rotation",
            "uv_layer",
        ],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let layer = uv_layer_field(engine, operation)?;
    let translation = read_vec2(
        engine,
        operation.get("translation"),
        [0.0, 0.0],
        "/translation",
    )?;
    let scale = read_vec2(engine, operation.get("scale"), [1.0, 1.0], "/scale")?;
    let rotation = read_number(engine, operation.get("rotation"), 0.0, "/rotation")?.to_radians();
    let (sin, cos) = rotation.sin_cos();
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        edit_layer_map(mesh, layer.as_deref(), false, |_, uv_map| {
            for (face_id, face_uv) in uv_map.iter_mut() {
                if !selected.contains(face_id) {
                    continue;
                }
                for uv in &mut face_uv.uv {
                    let u = uv[0] * scale[0];
                    let v = uv[1] * scale[1];
                    uv[0] = u.mul_add(cos, -v * sin) + translation[0];
                    uv[1] = u.mul_add(sin, v * cos) + translation[1];
                }
            }
            Ok(())
        })
    })
}

fn pin(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "pinned", "uv_layer"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let layer = uv_layer_field(engine, operation)?;
    let pinned = match operation.get("pinned") {
        None => true,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "pinned must be a boolean",
                "/pinned",
            ));
        }
    };
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        edit_layer_map(mesh, layer.as_deref(), false, |layer_mesh, uv_map| {
            for face_id in selected {
                let face = layer_mesh
                    .faces
                    .iter()
                    .find(|face| face.id == face_id)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::TargetNotFound, "selected UV face was not found")
                    })?;
                let face_uv = uv_map.get_mut(&face_id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("uv.pin requires UV coordinates for face f{face_id}"),
                    )
                })?;
                if face_uv.uv.len() != face.vertices.len() {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("UV corner count does not match face f{face_id}"),
                    ));
                }
                face_uv.pinned = Some(vec![pinned; face_uv.uv.len()]);
            }
            Ok(())
        })
    })
}

fn pack(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "elements", "scope", "margin", "uv_layer"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    let layer = uv_layer_field(engine, operation)?;
    let margin = read_number(
        engine,
        operation.get("margin"),
        DEFAULT_PACK_MARGIN,
        "/margin",
    )?;
    if !(0.0..=0.5).contains(&margin) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "margin must be between 0 and 0.5",
            "/margin",
        ));
    }
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let selected = selected_faces(mesh, elements.as_ref())?;
        edit_layer_map(mesh, layer.as_deref(), false, |_, uv_map| {
            pack_selected_uvs(uv_map, &selected, margin)
        })
    })
}

fn layer_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "name"],
        &["target"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    reject_elements(engine, elements.as_ref())?;
    let requested = operation
        .contains_key("name")
        .then(|| layer_name(engine, operation, "name"))
        .transpose()?;
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let mut store = uv_layers::read(mesh);
        if store.layers.len() >= uv_layers::MAX_LAYERS {
            return Err(layer_limit());
        }
        let name = match &requested {
            Some(name) if store.index_of(name).is_some() => return Err(layer_exists(name)),
            Some(name) => name.clone(),
            None => unnamed_layer_name(&store)?,
        };
        // Blender copies the active layer's values; corners without a source value get [0, 0].
        let source = store
            .active
            .as_deref()
            .and_then(|active| store.index_of(active))
            .and_then(|index| store.layers.get(index))
            .map(|layer| layer.faces.clone());
        let faces: Vec<Option<Vec<[f64; 2]>>> = mesh
            .faces
            .iter()
            .enumerate()
            .map(|(index, face)| {
                let copied = source
                    .as_ref()
                    .and_then(|faces| faces.get(index).cloned().flatten());
                Some(copied.unwrap_or_else(|| vec![[0.0, 0.0]; face.vertices.len()]))
            })
            .collect();
        let first = store.layers.is_empty();
        store.layers.push(UvLayer::new(name.clone(), faces));
        if first {
            store.active = Some(name.clone());
            store.render = Some(name);
        }
        uv_layers::write(mesh, &store);
        Ok(())
    })
}

fn layer_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "name"],
        &["target", "name"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    reject_elements(engine, elements.as_ref())?;
    let name = layer_name(engine, operation, "name")?;
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        false,
    )?;
    let edited = edited_users(engine.doc, &targets, scope);
    let referrers = layer_referrers(engine.doc, &edited, &name);
    if !referrers.is_empty() {
        let mut error = engine.error(
            ErrorCode::InvalidOperation,
            format!("UV layer `{name}` is used by {}", referrers.join(", ")),
            "/name",
        );
        error.details["referrers"] = json!(referrers);
        return Err(feature(error, "uv.layer_in_use"));
    }
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let mut store = uv_layers::read(mesh);
        let index = store
            .index_of(&name)
            .ok_or_else(|| layer_missing(Some(&name)))?;
        // Blender keeps the numeric active and render indices, clamped to the new last layer.
        let active = store
            .active
            .as_deref()
            .and_then(|active| store.index_of(active));
        let render = store
            .render
            .as_deref()
            .and_then(|render| store.index_of(render));
        store.layers.remove(index);
        store.active = clamped_layer_name(&store, active);
        store.render = clamped_layer_name(&store, render);
        uv_layers::write(mesh, &store);
        Ok(())
    })
}

fn layer_rename(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "name", "to"],
        &["target", "name", "to"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    reject_elements(engine, elements.as_ref())?;
    let name = layer_name(engine, operation, "name")?;
    let to = layer_name(engine, operation, "to")?;
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        false,
    )?;
    let edited = edited_users(engine.doc, &targets, scope);
    let conflicts = shared_reference_conflicts(engine.doc, &edited, &name);
    if !conflicts.is_empty() {
        let mut error = engine.error(
            ErrorCode::InvalidOperation,
            format!(
                "UV layer `{name}` cannot be renamed: nodes outside the target also use it: {}",
                conflicts.join(", ")
            ),
            "/name",
        );
        error.details["referrers"] = json!(conflicts);
        return Err(feature(error, "uv.layer_in_use"));
    }
    let changed = edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let mut store = uv_layers::read(mesh);
        let index = store
            .index_of(&name)
            .ok_or_else(|| layer_missing(Some(&name)))?;
        if store
            .index_of(&to)
            .is_some_and(|existing| existing != index)
        {
            return Err(layer_exists(&to));
        }
        if let Some(layer) = store.layers.get_mut(index) {
            layer.name.clone_from(&to);
        }
        if store.active.as_deref() == Some(name.as_str()) {
            store.active = Some(to.clone());
        }
        if store.render.as_deref() == Some(name.as_str()) {
            store.render = Some(to.clone());
        }
        uv_layers::write(mesh, &store);
        Ok(())
    })?;
    let references = rename_layer_references(engine.doc, &edited, &name, &to);
    for id in &references.materials {
        engine.mark("materials", id, ChangeKind::Updated);
    }
    for id in &references.nodes {
        engine.mark("nodes", id, ChangeKind::Updated);
    }
    for id in &references.node_groups {
        engine.mark("node_groups", id, ChangeKind::Updated);
    }
    if references.avatar {
        engine.changes.avatar = true;
    }
    Ok(changed || !references.is_empty())
}

#[derive(Clone, Copy)]
enum LayerRole {
    Active,
    Render,
}

fn layer_set_role(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    role: LayerRole,
) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "name"],
        &["target", "name"],
    )?;
    let (targets, elements) = selected_targets(engine, operation)?;
    reject_elements(engine, elements.as_ref())?;
    let name = layer_name(engine, operation, "name")?;
    edit_targets(engine, targets, operation.get("scope"), |mesh| {
        let mut store = uv_layers::read(mesh);
        if store.index_of(&name).is_none() {
            return Err(layer_missing(Some(&name)));
        }
        match role {
            LayerRole::Active => store.active = Some(name.clone()),
            LayerRole::Render => store.render = Some(name.clone()),
        }
        uv_layers::write(mesh, &store);
        Ok(())
    })
}

/// Tags `error` with the UV-layer contract it reports (`feature_id`).
fn feature(mut error: PotError, feature_id: &str) -> PotError {
    error.details["feature_id"] = json!(feature_id);
    error
}

/// `uv.layer_missing`: the named layer does not exist, or the mesh has no UV layers.
pub(super) fn layer_missing(name: Option<&str>) -> PotError {
    let message = match name {
        Some(name) => format!("UV layer `{name}` was not found"),
        None => "mesh has no UV layers".to_owned(),
    };
    feature(
        PotError::new(ErrorCode::InvalidOperation, message),
        "uv.layer_missing",
    )
}

/// `uv.layer_exists`: a create or rename wanted a name that another layer already has (D-U5).
fn layer_exists(name: &str) -> PotError {
    feature(
        PotError::new(
            ErrorCode::InvalidOperation,
            format!("UV layer `{name}` already exists"),
        ),
        "uv.layer_exists",
    )
}

/// `uv.layer_limit`: the mesh already stores the maximum number of UV layers.
pub(super) fn layer_limit() -> PotError {
    feature(
        PotError::new(
            ErrorCode::LimitExceeded,
            format!(
                "mesh already has {} UV layers, the limit",
                uv_layers::MAX_LAYERS
            ),
        ),
        "uv.layer_limit",
    )
}

/// Blender's name for an unnamed layer: `UVMap`, then the first free `UVMap.001` to `UVMap.999`.
fn unnamed_layer_name(store: &UvStore) -> Result<String> {
    if store.index_of(DEFAULT_LAYER_NAME).is_none() {
        return Ok(DEFAULT_LAYER_NAME.to_owned());
    }
    (1..=999)
        .map(|number| format!("{DEFAULT_LAYER_NAME}.{number:03}"))
        .find(|candidate| store.index_of(candidate).is_none())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "no free UV layer name below UVMap.999",
            )
        })
}

/// The layer at numeric position `position`, clamped to the last layer. `None` once no layer is left.
fn clamped_layer_name(store: &UvStore, position: Option<usize>) -> Option<String> {
    let last = store.layers.len().checked_sub(1)?;
    store
        .layers
        .get(position?.min(last))
        .map(|layer| layer.name.clone())
}

/// A required non-empty layer name field.
fn layer_name(engine: &Engine<'_>, operation: &Map<String, Value>, key: &str) -> Result<String> {
    let pointer = format!("/{key}");
    let name = operation.get(key).and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} must be a string"),
            &pointer,
        )
    })?;
    if name.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidArgument,
            format!("{key} must not be empty"),
            &pointer,
        ));
    }
    Ok(name.to_owned())
}

fn reject_elements(engine: &Engine<'_>, elements: Option<&Value>) -> Result<()> {
    if elements.is_some() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "UV layer operations do not take elements",
            "/elements",
        ));
    }
    Ok(())
}

/// The optional `uv_layer` name of a UV operation. Absent means the active layer. An empty name is
/// rejected: the active layer is selected by omitting the field.
fn uv_layer_field(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Option<String>> {
    match operation.get("uv_layer") {
        None => Ok(None),
        Some(Value::String(name)) if name.is_empty() => Err(engine.error(
            ErrorCode::InvalidOperation,
            "uv_layer must not be empty; omit it to use the active layer",
            "/uv_layer",
        )),
        Some(Value::String(name)) => Ok(Some(name.clone())),
        Some(_) => Err(engine.error(
            ErrorCode::InvalidOperation,
            "uv_layer must be a string",
            "/uv_layer",
        )),
    }
}

fn selected_targets(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<(Vec<Id>, Option<Value>)> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(ErrorCode::InvalidOperation, "target is required", "/target")
    })?;
    let Some(target_object) = target.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            "/target",
        ));
    };
    let mut target_without_elements = target_object.clone();
    let nested_elements = target_without_elements.remove("elements");
    if operation.contains_key("elements") && nested_elements.is_some() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements must appear in either target or operation",
            "/elements",
        ));
    }
    let elements = operation.get("elements").cloned().or(nested_elements);
    let targets = resolve_node_targets(engine, &Value::Object(target_without_elements), true)?;
    Ok((targets, elements))
}

/// Runs `edit` on the face map of one UV layer of `mesh` and stores the result. `requested` names
/// the layer; `None` is the active layer. With `create_default`, a mesh with no layers gets `UVMap`,
/// which becomes active and render.
fn edit_layer_map(
    mesh: &mut Mesh,
    requested: Option<&str>,
    create_default: bool,
    edit: impl FnOnce(&Mesh, &mut UvMap) -> Result<()>,
) -> Result<()> {
    let mut store = uv_layers::read(mesh);
    let index = if create_default && requested.is_none() && store.layers.is_empty() {
        store.layers.push(UvLayer::new(
            DEFAULT_LAYER_NAME,
            vec![None; mesh.faces.len()],
        ));
        store.active = Some(DEFAULT_LAYER_NAME.to_owned());
        store.render = Some(DEFAULT_LAYER_NAME.to_owned());
        0
    } else {
        layer_index(&store, requested)?
    };
    let layer = store
        .layers
        .get_mut(index)
        .ok_or_else(|| layer_missing(requested))?;
    let mut uv_map = layer_uv_map(mesh, layer);
    edit(&*mesh, &mut uv_map)?;
    set_layer_uv_map(mesh, layer, &uv_map);
    uv_layers::write(mesh, &store);
    Ok(())
}

/// Position of the layer `requested` names, or of the active layer when `requested` is `None`.
fn layer_index(store: &UvStore, requested: Option<&str>) -> Result<usize> {
    let index = match requested {
        Some(name) => store.index_of(name),
        None => store
            .active
            .as_deref()
            .and_then(|active| store.index_of(active)),
    };
    index.ok_or_else(|| layer_missing(requested))
}

/// Material and `MToon` texture slots, by field name. The referrer scan and rename share these lists.
fn material_textures(material: &Material) -> [(&'static str, Option<&TextureRef>); 4] {
    [
        ("base_color_texture", material.base_color_texture.as_ref()),
        ("roughness_texture", material.roughness_texture.as_ref()),
        ("metallic_texture", material.metallic_texture.as_ref()),
        ("normal_texture", material.normal_texture.as_ref()),
    ]
}

fn material_textures_mut(material: &mut Material) -> [(&'static str, Option<&mut TextureRef>); 4] {
    [
        ("base_color_texture", material.base_color_texture.as_mut()),
        ("roughness_texture", material.roughness_texture.as_mut()),
        ("metallic_texture", material.metallic_texture.as_mut()),
        ("normal_texture", material.normal_texture.as_mut()),
    ]
}

fn mtoon_textures(mtoon: &AvatarMtoon) -> [(&'static str, Option<&TextureRef>); 6] {
    [
        (
            "shade_multiply_texture",
            mtoon.shade_multiply_texture.as_ref(),
        ),
        (
            "shading_shift_texture",
            mtoon
                .shading_shift_texture
                .as_ref()
                .map(|shift| &shift.texture),
        ),
        ("matcap_texture", mtoon.matcap_texture.as_ref()),
        ("rim_multiply_texture", mtoon.rim_multiply_texture.as_ref()),
        (
            "outline_width_multiply_texture",
            mtoon.outline_width_multiply_texture.as_ref(),
        ),
        (
            "uv_animation_mask_texture",
            mtoon.uv_animation_mask_texture.as_ref(),
        ),
    ]
}

fn mtoon_textures_mut(mtoon: &mut AvatarMtoon) -> [(&'static str, Option<&mut TextureRef>); 6] {
    [
        (
            "shade_multiply_texture",
            mtoon.shade_multiply_texture.as_mut(),
        ),
        (
            "shading_shift_texture",
            mtoon
                .shading_shift_texture
                .as_mut()
                .map(|shift| &mut shift.texture),
        ),
        ("matcap_texture", mtoon.matcap_texture.as_mut()),
        ("rim_multiply_texture", mtoon.rim_multiply_texture.as_mut()),
        (
            "outline_width_multiply_texture",
            mtoon.outline_width_multiply_texture.as_mut(),
        ),
        (
            "uv_animation_mask_texture",
            mtoon.uv_animation_mask_texture.as_mut(),
        ),
    ]
}

/// The nodes whose mesh a UV edit of `targets` changes. A `shared` edit changes the mesh every
/// user of the target shares; any other edit changes only the targets (`single_user` copies the
/// mesh for them first).
fn edited_users(doc: &SceneDoc, targets: &[Id], scope: Option<&str>) -> BTreeSet<Id> {
    let mut edited: BTreeSet<Id> = targets.iter().cloned().collect();
    if scope == Some("shared") {
        let data: BTreeSet<&Id> = targets
            .iter()
            .filter_map(|target| doc.nodes.get(target)?.data.as_ref())
            .collect();
        edited.extend(
            doc.nodes
                .iter()
                .filter(|(_, node)| node.data.as_ref().is_some_and(|id| data.contains(id)))
                .map(|(node_id, _)| node_id.clone()),
        );
    }
    edited
}

/// What a set of nodes reaches: the materials in their slots, the node groups their geometry
/// modifiers instance, and the shader graphs those materials use, including node groups instanced
/// by group nodes, transitively.
#[derive(Default)]
struct Reach {
    materials: BTreeSet<Id>,
    graphs: BTreeSet<Id>,
}

fn reach<'a>(doc: &SceneDoc, nodes: impl IntoIterator<Item = &'a Id>) -> Reach {
    let mut reached = Reach::default();
    let mut pending: Vec<Id> = Vec::new();
    for node_id in nodes {
        if let Some(node) = doc.nodes.get(node_id) {
            let data = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id));
            for slot in 0..node.material_slot_count(data) {
                if let Some(material) = node.material_slot(data, slot) {
                    reached.materials.insert(material.clone());
                }
            }
            pending.extend(node.modifiers.iter().filter_map(modifier_node_group));
        }
    }
    pending.extend(
        reached
            .materials
            .iter()
            .filter_map(|material_id| doc.materials.get(material_id)?.node_tree.clone()),
    );
    while let Some(graph_id) = pending.pop() {
        let Some(graph) = doc.node_groups.get(&graph_id) else {
            continue;
        };
        if !reached.graphs.insert(graph_id) {
            continue;
        }
        for node in graph.nodes.values() {
            if node.node_type != "ShaderNodeGroup" {
                continue;
            }
            let Some(tree) = group_tree_name(node) else {
                continue;
            };
            pending.extend(
                doc.node_groups
                    .iter()
                    .filter(|(_, group)| group.name == tree)
                    .map(|(id, _)| id.clone()),
            );
        }
    }
    reached
}

/// The node group a `ShaderNodeGroup` node instances. Its `node_tree` property names the group, as
/// `{"name": ...}` or as a bare name.
fn group_tree_name(node: &GraphNode) -> Option<&str> {
    let value = node.properties.get("node_tree")?;
    value
        .get("name")
        .and_then(Value::as_str)
        .or_else(|| value.as_str())
}

/// The node group a geometry modifier instances: its `node_group` param, a graph ID.
fn modifier_node_group(modifier: &Modifier) -> Option<Id> {
    Id::new(modifier.params.get("node_group")?.as_str()?).ok()
}

/// The texture and `MToon` slots of material `material_id` that name layer `name`.
fn material_refs(doc: &SceneDoc, material_id: &Id, name: &str) -> Vec<String> {
    let mut refs = Vec::new();
    if let Some(material) = doc.materials.get(material_id) {
        for (field, texture) in material_textures(material) {
            if texture.and_then(|texture| texture.uv_layer.as_deref()) == Some(name) {
                refs.push(format!("material `{material_id}` {field}"));
            }
        }
    }
    if let Some(mtoon) = doc
        .avatar
        .as_ref()
        .and_then(|avatar| avatar.mtoon.get(material_id))
    {
        for (field, texture) in mtoon_textures(mtoon) {
            if texture.and_then(|texture| texture.uv_layer.as_deref()) == Some(name) {
                refs.push(format!("avatar MToon `{material_id}` {field}"));
            }
        }
    }
    refs
}

/// The `ShaderNodeUVMap` nodes of node group `graph_id` that name layer `name`.
fn graph_refs(doc: &SceneDoc, graph_id: &Id, name: &str) -> Vec<String> {
    let Some(graph) = doc.node_groups.get(graph_id) else {
        return Vec::new();
    };
    graph
        .nodes
        .iter()
        .filter(|(_, node)| {
            node.node_type == "ShaderNodeUVMap"
                && node.properties.get("uv_map").and_then(Value::as_str) == Some(name)
        })
        .map(|(node_id, _)| {
            format!("ShaderNodeUVMap node `{node_id}` in node group `{graph_id}` property `uv_map`")
        })
        .collect()
}

/// Describes every reference to the UV layer `name` that the nodes in `edited` reach: material and
/// `MToon` texture slots, modifier params of those nodes, and `ShaderNodeUVMap` nodes of the graphs
/// they reach. Unreached references are not reported.
fn layer_referrers(doc: &SceneDoc, edited: &BTreeSet<Id>, name: &str) -> Vec<String> {
    let reached = reach(doc, edited);
    let mut referrers: Vec<String> = reached
        .materials
        .iter()
        .flat_map(|material_id| material_refs(doc, material_id, name))
        .collect();
    for node_id in edited {
        let Some(node) = doc.nodes.get(node_id) else {
            continue;
        };
        for modifier in &node.modifiers {
            for key in MODIFIER_LAYER_PARAMS {
                if modifier.params.get(key).and_then(Value::as_str) == Some(name) {
                    referrers.push(format!(
                        "modifier `{}` on node `{node_id}` param {key}",
                        modifier.id
                    ));
                }
            }
        }
    }
    for graph_id in &reached.graphs {
        referrers.extend(graph_refs(doc, graph_id, name));
    }
    referrers
}

/// Rename conflicts: the items a rename of layer `name` would rewrite for the nodes in `edited`
/// that a node outside `edited` also reaches. Rewriting such an item would change that node too,
/// so the rename is refused rather than applied.
fn shared_reference_conflicts(doc: &SceneDoc, edited: &BTreeSet<Id>, name: &str) -> Vec<String> {
    let reached = reach(doc, edited);
    let mut conflicts = Vec::new();
    for other in doc
        .nodes
        .keys()
        .filter(|node_id| !edited.contains(*node_id))
    {
        let shared = reach(doc, [other]);
        for material_id in shared.materials.intersection(&reached.materials) {
            if !material_refs(doc, material_id, name).is_empty() {
                conflicts.push(format!(
                    "material `{material_id}` is also used by node `{other}`"
                ));
            }
        }
        for graph_id in shared.graphs.intersection(&reached.graphs) {
            if !graph_refs(doc, graph_id, name).is_empty() {
                conflicts.push(format!(
                    "node group `{graph_id}` is also used by node `{other}`"
                ));
            }
        }
    }
    conflicts
}

/// Objects whose stored layer references a rename changed.
#[derive(Default)]
struct ReferenceChanges {
    materials: Vec<Id>,
    nodes: Vec<Id>,
    node_groups: Vec<Id>,
    avatar: bool,
}

impl ReferenceChanges {
    fn is_empty(&self) -> bool {
        self.materials.is_empty()
            && self.nodes.is_empty()
            && self.node_groups.is_empty()
            && !self.avatar
    }
}

/// Renames every reference to the layer `from` that the nodes in `edited` reach, so that it names
/// `to`. Items the edited nodes do not reach are left alone.
fn rename_layer_references(
    doc: &mut SceneDoc,
    edited: &BTreeSet<Id>,
    from: &str,
    to: &str,
) -> ReferenceChanges {
    let mut changes = ReferenceChanges::default();
    if from == to {
        return changes;
    }
    let reached = reach(&*doc, edited);
    for node_id in edited {
        let Some(node) = doc.nodes.get_mut(node_id) else {
            continue;
        };
        let mut changed = false;
        for modifier in &mut node.modifiers {
            for key in MODIFIER_LAYER_PARAMS {
                if let Some(value) = modifier.params.get_mut(key)
                    && value.as_str() == Some(from)
                {
                    *value = json!(to);
                    changed = true;
                }
            }
        }
        if changed {
            changes.nodes.push(node_id.clone());
        }
    }
    for material_id in &reached.materials {
        if let Some(material) = doc.materials.get_mut(material_id) {
            let mut changed = false;
            for (_, texture) in material_textures_mut(material) {
                if let Some(texture) = texture
                    && texture.uv_layer.as_deref() == Some(from)
                {
                    texture.uv_layer = Some(to.to_owned());
                    changed = true;
                }
            }
            if changed {
                changes.materials.push(material_id.clone());
            }
        }
        if let Some(mtoon) = doc
            .avatar
            .as_mut()
            .and_then(|avatar| avatar.mtoon.get_mut(material_id))
        {
            for (_, texture) in mtoon_textures_mut(mtoon) {
                if let Some(texture) = texture
                    && texture.uv_layer.as_deref() == Some(from)
                {
                    texture.uv_layer = Some(to.to_owned());
                    changes.avatar = true;
                }
            }
        }
    }
    for graph_id in &reached.graphs {
        let Some(graph) = doc.node_groups.get_mut(graph_id) else {
            continue;
        };
        let mut changed = false;
        for node in graph.nodes.values_mut() {
            if node.node_type == "ShaderNodeUVMap"
                && node.properties.get("uv_map").and_then(Value::as_str) == Some(from)
            {
                node.properties.insert("uv_map".to_owned(), json!(to));
                changed = true;
            }
        }

        if changed {
            changes.node_groups.push(graph_id.clone());
        }
    }
    changes
}
fn edit_targets(
    engine: &mut Engine<'_>,
    targets: Vec<Id>,
    scope_value: Option<&Value>,
    mut edit: impl FnMut(&mut Mesh) -> Result<()>,
) -> Result<bool> {
    let scope = super::parse_scope(
        engine,
        scope_value,
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        false,
    )?;
    let mut node_data = Vec::with_capacity(targets.len());
    for target in &targets {
        let data_id = engine
            .doc
            .nodes
            .get(target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                    "/target",
                )
            })?;
        node_data.push((target.clone(), data_id));
    }
    let shared = node_data
        .iter()
        .any(|(_, data_id)| super::data_user_count(engine, data_id) > 1);
    if shared && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry edits require scope",
            "/scope",
        ));
    }
    if scope == Some("single_user") {
        for (target, data_id) in &node_data {
            if super::data_user_count(engine, data_id) <= 1 {
                continue;
            }
            let copy = engine
                .doc
                .data_blocks
                .get(data_id)
                .cloned()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` was not found"),
                    )
                })?;
            let new_id = super::unique_data_id(
                engine,
                target,
                "_mesh_single",
                super::DataIdCollisionHandling::ReserveSuffix,
                "data-block ID suffix exhausted",
            )?;
            engine.doc.data_blocks.insert(new_id.clone(), copy);
            if let Some(node) = engine.doc.nodes.get_mut(target) {
                node.data = Some(new_id.clone());
            }
            engine.mark("data_blocks", &new_id, ChangeKind::Created);
            engine.mark("nodes", target, ChangeKind::Updated);
            engine.map_id("data_blocks", data_id, &new_id);
        }
    }
    let mut changed = false;
    let mut edited_data = Vec::new();
    for target in targets {
        let data_id = engine
            .doc
            .nodes
            .get(&target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data"),
                )
            })?;
        if scope == Some("shared") && edited_data.contains(&data_id) {
            continue;
        }
        let (data_changed, had_descriptor) = {
            let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::TargetNotFound,
                    format!("data block `{data_id}` was not found"),
                )
            })?;
            let mesh = data.mesh.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    format!("data block `{data_id}` has no mesh"),
                )
            })?;
            let before = mesh.clone();
            let had_descriptor = data.descriptor.is_some();
            edit(mesh)?;
            let changed = *mesh != before;
            if changed {
                data.descriptor = None;
            }
            (changed, had_descriptor)
        };
        if data_changed {
            if had_descriptor {
                engine
                    .id_mappings
                    .entry("mesh_descriptor_dropped".to_owned())
                    .or_insert_with(|| json!({}))[data_id.as_str()] = Value::Bool(true);
            }
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            changed = true;
        }
        edited_data.push(data_id);
    }
    Ok(changed)
}

fn selected_faces(mesh: &Mesh, elements: Option<&Value>) -> Result<Vec<u32>> {
    let Some(elements) = elements else {
        return Ok(mesh.faces.iter().map(|face| face.id).collect());
    };
    let object = elements
        .as_object()
        .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "elements must be an object"))?;
    if object.get("domain").and_then(Value::as_str) != Some("face") {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV elements domain must be face",
        ));
    }
    let ids = object.get("ids").and_then(Value::as_array).ok_or_else(|| {
        PotError::new(ErrorCode::InvalidOperation, "elements.ids must be an array")
    })?;
    let mut selected = Vec::with_capacity(ids.len());
    for value in ids {
        let text = value.as_str().ok_or_else(|| {
            PotError::new(ErrorCode::InvalidOperation, "element IDs must be strings")
        })?;
        let Some(id_text) = text.strip_prefix('f') else {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "face element IDs must start with f",
            ));
        };
        let id = id_text
            .parse::<u32>()
            .map_err(|_| PotError::new(ErrorCode::InvalidOperation, "invalid face element ID"))?;
        if !mesh.faces.iter().any(|face| face.id == id) {
            return Err(PotError::new(
                ErrorCode::TargetNotFound,
                format!("face element `{text}` was not found"),
            ));
        }
        if !selected.contains(&id) {
            selected.push(id);
        }
    }
    Ok(selected)
}

#[derive(Clone, Copy)]
struct UnwrapTriangle {
    vertices: [usize; 3],
}

struct UnwrapChart {
    face_indices: Vec<usize>,
    corner_indices: std::collections::HashMap<(usize, usize), usize>,
    vertex_ids: Vec<u32>,
    positions: Vec<DVec3>,
    triangles: Vec<UnwrapTriangle>,
}

fn unwrap_advanced(mesh: &Mesh, selected: &[u32], uv_map: &mut UvMap, method: &str) -> Result<()> {
    let selected_set = selected
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let faces = mesh
        .faces
        .iter()
        .filter(|face| selected_set.contains(&face.id))
        .collect::<Vec<_>>();
    if faces.is_empty() {
        return Ok(());
    }

    // Marked seam edges form chart boundaries; unmarked selected edges join faces.
    let seam_edges = seam_edge_keys(mesh)?;
    let mut edge_faces = std::collections::BTreeMap::<(u32, u32), Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.vertices.len() < 3 {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "UV unwrap encountered a face with fewer than three vertices",
            ));
        }
        for corner in 0..face.vertices.len() {
            let edge = crate::geom::edge_key(
                face.vertices[corner],
                face.vertices[(corner + 1) % face.vertices.len()],
            );
            edge_faces.entry(edge).or_default().push(face_index);
        }
    }
    let mut neighbors = vec![Vec::new(); faces.len()];
    for (edge, incident_faces) in edge_faces {
        if seam_edges.contains(&edge) {
            continue;
        }
        for &left in &incident_faces {
            for &right in &incident_faces {
                if left != right && !neighbors[left].contains(&right) {
                    neighbors[left].push(right);
                }
            }
        }
    }

    let mut visited = vec![false; faces.len()];
    for seed in 0..faces.len() {
        if visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(face_index) = queue.pop_front() {
            component.push(face_index);
            for &neighbor in &neighbors[face_index] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        let chart = build_unwrap_chart(mesh, &faces, component, &seam_edges)?;
        let initial = unfold_chart(&chart)?;
        let pins = chart_pins(&faces, &chart, uv_map)?;
        let coordinates = if method == "conformal" || !pins.is_empty() {
            let anchors = chart_anchors(&chart, &pins, &initial)?;
            solve_lscm(&chart, &anchors)?
        } else {
            initial
        };

        for &face_index in &chart.face_indices {
            let face = faces[face_index];
            let corners = face
                .vertices
                .iter()
                .enumerate()
                .map(|(corner, _)| coordinates[chart.corner_indices[&(face_index, corner)]])
                .collect::<Vec<_>>();
            let pinned = uv_map
                .get(&face.id)
                .and_then(|face_uv| face_uv.pinned.clone());
            uv_map.insert(
                face.id,
                UvFace {
                    uv: corners,
                    pinned,
                },
            );
        }
    }
    Ok(())
}

fn seam_edge_keys(mesh: &Mesh) -> Result<std::collections::HashSet<(u32, u32)>> {
    let Some(attribute) = mesh.attributes.get("seam") else {
        return Ok(std::collections::HashSet::new());
    };
    if attribute.get("domain").and_then(Value::as_str) != Some("edge") {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "seam attribute must use the edge domain",
        ));
    }
    let values = attribute
        .get("values")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "seam attribute values must be an object",
            )
        })?;
    for value in values.values() {
        if !value.is_boolean() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "seam edge values must be boolean",
            ));
        }
    }
    let marked = mesh
        .edges
        .iter()
        .filter_map(|edge| {
            values
                .get(&format!("e{}", edge.id))
                .and_then(Value::as_bool)
                .filter(|is_seam| *is_seam)
                .map(|_| crate::geom::edge_key(edge.vertices[0], edge.vertices[1]))
        })
        .collect();
    Ok(marked)
}

fn corner_component_root(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = index;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

fn join_corner_components(parents: &mut [usize], left: usize, right: usize) {
    let left_root = corner_component_root(parents, left);
    let right_root = corner_component_root(parents, right);
    if left_root != right_root {
        let (root, child) = if left_root < right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parents[child] = root;
    }
}

fn build_unwrap_chart(
    mesh: &Mesh,
    faces: &[&Face],
    face_indices: Vec<usize>,
    seam_edges: &std::collections::HashSet<(u32, u32)>,
) -> Result<UnwrapChart> {
    let mut corner_offsets = std::collections::HashMap::new();
    let mut edge_faces = std::collections::BTreeMap::<(u32, u32), Vec<usize>>::new();
    for &face_index in &face_indices {
        let face = faces[face_index];
        if face.vertices.len() < 3 {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "UV unwrap encountered a face with fewer than three vertices",
            ));
        }
        for (corner, &vertex_id) in face.vertices.iter().enumerate() {
            corner_offsets.insert((face_index, corner), corner_offsets.len());
            edge_faces
                .entry(crate::geom::edge_key(
                    vertex_id,
                    face.vertices[(corner + 1) % face.vertices.len()],
                ))
                .or_default()
                .push(face_index);
        }
    }

    let mut parents = (0..corner_offsets.len()).collect::<Vec<_>>();
    for (edge, incident_faces) in edge_faces {
        if seam_edges.contains(&edge) {
            continue;
        }
        for left_index in 0..incident_faces.len() {
            for right_index in left_index + 1..incident_faces.len() {
                let left_face_index = incident_faces[left_index];
                let right_face_index = incident_faces[right_index];
                for vertex_id in [edge.0, edge.1] {
                    let left_corner = faces[left_face_index]
                        .vertices
                        .iter()
                        .position(|candidate| *candidate == vertex_id)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV edge references a missing face corner",
                            )
                        })?;
                    let right_corner = faces[right_face_index]
                        .vertices
                        .iter()
                        .position(|candidate| *candidate == vertex_id)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV edge references a missing face corner",
                            )
                        })?;
                    join_corner_components(
                        &mut parents,
                        corner_offsets[&(left_face_index, left_corner)],
                        corner_offsets[&(right_face_index, right_corner)],
                    );
                }
            }
        }
    }

    let mut root_indices = std::collections::HashMap::new();
    let mut corner_indices = std::collections::HashMap::new();
    let mut vertex_ids = Vec::new();
    let mut positions = Vec::new();
    for &face_index in &face_indices {
        for (corner, &vertex_id) in faces[face_index].vertices.iter().enumerate() {
            let root = corner_component_root(&mut parents, corner_offsets[&(face_index, corner)]);
            let local_index = if let Some(index) = root_indices.get(&root) {
                *index
            } else {
                let position = mesh
                    .vertex(vertex_id)
                    .map(|vertex| vertex.co)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "face references a missing vertex")
                    })?;
                let index = vertex_ids.len();
                vertex_ids.push(vertex_id);
                positions.push(position);
                root_indices.insert(root, index);
                index
            };
            corner_indices.insert((face_index, corner), local_index);
        }
    }

    let mut triangles = Vec::new();
    for &face_index in &face_indices {
        let face = faces[face_index];
        let local_face = (0..face.vertices.len())
            .map(|corner| corner_indices[&(face_index, corner)])
            .collect::<Vec<_>>();
        for corner in 1..local_face.len() - 1 {
            triangles.push(UnwrapTriangle {
                vertices: [local_face[0], local_face[corner], local_face[corner + 1]],
            });
        }
    }
    Ok(UnwrapChart {
        face_indices,
        corner_indices,
        vertex_ids,
        positions,
        triangles,
    })
}

// Intrinsically unfold each triangle across a spanning traversal of its island.
fn unfold_chart(chart: &UnwrapChart) -> Result<Vec<[f64; 2]>> {
    let Some(seed) = chart.triangles.first() else {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap requires at least one triangle",
        ));
    };
    let mut coordinates = vec![None; chart.vertex_ids.len()];
    place_seed_triangle(chart, seed, &mut coordinates, [0.0, 0.0])?;

    let mut edge_triangles = std::collections::BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (triangle_index, triangle) in chart.triangles.iter().enumerate() {
        for [left, right] in [
            [triangle.vertices[0], triangle.vertices[1]],
            [triangle.vertices[1], triangle.vertices[2]],
            [triangle.vertices[2], triangle.vertices[0]],
        ] {
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            edge_triangles.entry(edge).or_default().push(triangle_index);
        }
    }
    let mut queue = std::collections::VecDeque::from([0usize]);
    let mut visited = vec![false; chart.triangles.len()];
    visited[0] = true;
    while let Some(triangle_index) = queue.pop_front() {
        let triangle = chart.triangles[triangle_index];
        for [left, right] in [
            [triangle.vertices[0], triangle.vertices[1]],
            [triangle.vertices[1], triangle.vertices[2]],
            [triangle.vertices[2], triangle.vertices[0]],
        ] {
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            let Some(adjacent) = edge_triangles.get(&edge) else {
                continue;
            };
            for &neighbor_index in adjacent {
                if neighbor_index == triangle_index || visited[neighbor_index] {
                    continue;
                }
                let neighbor = chart.triangles[neighbor_index];
                let third = neighbor
                    .vertices
                    .iter()
                    .copied()
                    .find(|vertex| *vertex != left && *vertex != right)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "UV triangulation contains a repeated vertex",
                        )
                    })?;
                if coordinates[third].is_none() {
                    let current_third = triangle
                        .vertices
                        .iter()
                        .copied()
                        .find(|vertex| *vertex != left && *vertex != right)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "UV triangulation contains a repeated vertex",
                            )
                        })?;
                    place_unfolded_vertex(
                        chart,
                        left,
                        right,
                        current_third,
                        third,
                        &mut coordinates,
                    )?;
                }
                visited[neighbor_index] = true;
                queue.push_back(neighbor_index);
            }
        }
    }
    if coordinates.iter().any(Option::is_none) || visited.iter().any(|visited| !visited) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV island has disconnected triangulation",
        ));
    }
    Ok(coordinates.into_iter().map(Option::unwrap).collect())
}

fn place_seed_triangle(
    chart: &UnwrapChart,
    triangle: &UnwrapTriangle,
    coordinates: &mut [Option<[f64; 2]>],
    origin: [f64; 2],
) -> Result<()> {
    let [first, second, third] = triangle.vertices;
    let edge = chart.positions[second] - chart.positions[first];
    let edge_length = edge.length();
    let to_third = chart.positions[third] - chart.positions[first];
    let cross_length = edge.cross(to_third).length();
    if !edge_length.is_finite() || !cross_length.is_finite() || edge_length <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let x = to_third.dot(edge / edge_length);
    let height = cross_length / edge_length;
    if !height.is_finite() || height <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    coordinates[first] = Some(origin);
    coordinates[second] = Some([origin[0] + edge_length, origin[1]]);
    coordinates[third] = Some([origin[0] + x, origin[1] + height]);
    Ok(())
}

fn place_unfolded_vertex(
    chart: &UnwrapChart,
    left: usize,
    right: usize,
    current_third: usize,
    next_third: usize,
    coordinates: &mut [Option<[f64; 2]>],
) -> Result<()> {
    let a = coordinates[left].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost an edge endpoint",
        )
    })?;
    let b = coordinates[right].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost an edge endpoint",
        )
    })?;
    let c = coordinates[current_third].ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap lost a neighboring triangle",
        )
    })?;
    let edge = chart.positions[right] - chart.positions[left];
    let edge_length = edge.length();

    let to_next = chart.positions[next_third] - chart.positions[left];
    let cross_length = edge.cross(to_next).length();
    if !edge_length.is_finite() || !cross_length.is_finite() || edge_length <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let x = to_next.dot(edge / edge_length);
    let height = cross_length / edge_length;
    if !height.is_finite() || height <= 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a degenerate triangle",
        ));
    }
    let uv_edge = [b[0] - a[0], b[1] - a[1]];
    let uv_length = uv_edge[0].hypot(uv_edge[1]);
    let side = uv_edge[0] * (c[1] - a[1]) - uv_edge[1] * (c[0] - a[0]);
    if uv_length <= 0.0 || !uv_length.is_finite() || side == 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap encountered a collapsed triangle edge",
        ));
    }
    let unit = [uv_edge[0] / uv_length, uv_edge[1] / uv_length];
    let perpendicular = if side > 0.0 {
        [unit[1], -unit[0]]
    } else {
        [-unit[1], unit[0]]
    };
    coordinates[next_third] = Some([
        a[0] + unit[0] * (x * uv_length / edge_length)
            + perpendicular[0] * (height * uv_length / edge_length),
        a[1] + unit[1] * (x * uv_length / edge_length)
            + perpendicular[1] * (height * uv_length / edge_length),
    ]);

    Ok(())
}

fn chart_pins(
    faces: &[&Face],
    chart: &UnwrapChart,
    uv_map: &UvMap,
) -> Result<std::collections::BTreeMap<usize, [f64; 2]>> {
    let mut pins = std::collections::BTreeMap::<usize, [f64; 2]>::new();
    for &face_index in &chart.face_indices {
        let face = faces[face_index];
        let Some(face_uv) = uv_map.get(&face.id) else {
            continue;
        };
        if face_uv.uv.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV corner count does not match face f{}", face.id),
            ));
        }
        let Some(pinned) = &face_uv.pinned else {
            continue;
        };
        if pinned.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV pin count does not match face f{}", face.id),
            ));
        }
        for (corner, (&vertex_id, &is_pinned)) in
            face.vertices.iter().zip(pinned.iter()).enumerate()
        {
            if !is_pinned {
                continue;
            }
            let vertex_index = chart.corner_indices[&(face_index, corner)];
            let coordinate = face_uv.uv[corner];
            if let Some(previous) = pins.get(&vertex_index) {
                let scale = previous[0]
                    .abs()
                    .max(previous[1].abs())
                    .max(coordinate[0].abs())
                    .max(coordinate[1].abs())
                    .max(1.0);
                if (previous[0] - coordinate[0]).hypot(previous[1] - coordinate[1])
                    > scale * 1.0e-10
                {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        format!("pinned UVs disagree at vertex v{vertex_id}"),
                    ));
                }
            } else {
                pins.insert(vertex_index, coordinate);
            }
        }
    }
    Ok(pins)
}

fn chart_anchors(
    chart: &UnwrapChart,
    pins: &std::collections::BTreeMap<usize, [f64; 2]>,
    initial: &[[f64; 2]],
) -> Result<std::collections::BTreeMap<usize, [f64; 2]>> {
    let mut anchors = pins.clone();
    if anchors.len() >= 2 {
        return Ok(anchors);
    }
    if chart.positions.len() < 2 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV unwrap requires at least two distinct vertices",
        ));
    }
    if let Some((&fixed_index, &fixed_uv)) = anchors.iter().next() {
        let other = (0..chart.positions.len())
            .filter(|index| *index != fixed_index)
            .max_by(|left, right| {
                chart.positions[*left]
                    .distance_squared(chart.positions[fixed_index])
                    .total_cmp(
                        &chart.positions[*right].distance_squared(chart.positions[fixed_index]),
                    )
            })
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidOperation,
                    "UV unwrap could not place an anchor",
                )
            })?;
        anchors.insert(
            other,
            [
                fixed_uv[0] + initial[other][0] - initial[fixed_index][0],
                fixed_uv[1] + initial[other][1] - initial[fixed_index][1],
            ],
        );
    } else {
        let mut pair = (0, 1);
        let mut farthest = 0.0;
        for left in 0..chart.positions.len() {
            for right in left + 1..chart.positions.len() {
                let distance = chart.positions[left].distance_squared(chart.positions[right]);
                if distance > farthest {
                    farthest = distance;
                    pair = (left, right);
                }
            }
        }
        if farthest <= 0.0 || !farthest.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV unwrap requires distinct anchor vertices",
            ));
        }
        anchors.insert(pair.0, initial[pair.0]);
        anchors.insert(pair.1, initial[pair.1]);
    }
    Ok(anchors)
}

struct LinearRow {
    terms: [(usize, f64); 6],
    term_count: usize,
    rhs: f64,
}

fn solve_lscm(
    chart: &UnwrapChart,
    anchors: &std::collections::BTreeMap<usize, [f64; 2]>,
) -> Result<Vec<[f64; 2]>> {
    let mut fixed = vec![None; chart.vertex_ids.len() * 2];
    for (&vertex_index, uv) in anchors {
        fixed[2 * vertex_index] = Some(uv[0]);
        fixed[2 * vertex_index + 1] = Some(uv[1]);
    }
    let mut free_indices = vec![None; fixed.len()];
    let mut free_count = 0;
    for (dof, value) in fixed.iter().enumerate() {
        if value.is_none() {
            free_indices[dof] = Some(free_count);
            free_count += 1;
        }
    }

    // Weighted Cauchy-Riemann residuals define the LSCM least-squares system.
    let mut rows = Vec::with_capacity(chart.triangles.len() * 2);
    for triangle in &chart.triangles {
        let [first, second, third] = triangle.vertices;
        let edge = chart.positions[second] - chart.positions[first];
        let edge_length = edge.length();
        let to_third = chart.positions[third] - chart.positions[first];
        let x = to_third.dot(edge / edge_length);
        let y = edge.cross(to_third).length() / edge_length;
        let area = edge_length * y * 0.5;
        if !edge_length.is_finite() || !y.is_finite() || area <= 0.0 {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "conformal unwrap encountered a degenerate triangle",
            ));
        }
        let gradients = [
            [-1.0 / edge_length, (x - edge_length) / (edge_length * y)],
            [1.0 / edge_length, -x / (edge_length * y)],
            [0.0, 1.0 / y],
        ];
        let weight = area.sqrt();
        let mut first_row = [(0usize, 0.0); 6];
        let mut second_row = [(0usize, 0.0); 6];
        for corner in 0..3 {
            let vertex_index = triangle.vertices[corner];
            first_row[2 * corner] = (2 * vertex_index, gradients[corner][0] * weight);
            first_row[2 * corner + 1] = (2 * vertex_index + 1, -gradients[corner][1] * weight);
            second_row[2 * corner] = (2 * vertex_index, gradients[corner][1] * weight);
            second_row[2 * corner + 1] = (2 * vertex_index + 1, gradients[corner][0] * weight);
        }
        rows.push(reduce_lscm_row(first_row, &fixed, &free_indices));
        rows.push(reduce_lscm_row(second_row, &fixed, &free_indices));
    }
    let mut diagonal = vec![0.0; free_count];
    let mut right_hand_side = vec![0.0; free_count];
    for row in &rows {
        for &(index, coefficient) in &row.terms[..row.term_count] {
            diagonal[index] += coefficient * coefficient;
            right_hand_side[index] += coefficient * row.rhs;
        }
    }
    if diagonal
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap has unconstrained UV coordinates",
        ));
    }
    let solution = conjugate_gradient(&rows, &diagonal, &right_hand_side)?;
    let mut coordinates = vec![[0.0; 2]; chart.vertex_ids.len()];
    for (vertex_index, coordinate) in coordinates.iter_mut().enumerate() {
        for (axis, value) in coordinate.iter_mut().enumerate() {
            let dof = 2 * vertex_index + axis;
            *value = if let Some(value) = fixed[dof] {
                value
            } else {
                solution[free_indices[dof].ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InvalidOperation,
                        "conformal unwrap lost a UV value",
                    )
                })?]
            };
        }
    }
    if coordinates
        .iter()
        .flatten()
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap produced non-finite UV coordinates",
        ));
    }
    Ok(coordinates)
}

fn reduce_lscm_row(
    coefficients: [(usize, f64); 6],
    fixed: &[Option<f64>],
    free_indices: &[Option<usize>],
) -> LinearRow {
    let mut row = LinearRow {
        terms: [(0usize, 0.0); 6],
        term_count: 0,
        rhs: 0.0,
    };
    for (dof, coefficient) in coefficients {
        if let Some(value) = fixed[dof] {
            row.rhs -= coefficient * value;
        } else if let Some(index) = free_indices[dof] {
            row.terms[row.term_count] = (index, coefficient);
            row.term_count += 1;
        }
    }
    row
}

fn conjugate_gradient(
    rows: &[LinearRow],
    diagonal: &[f64],
    right_hand_side: &[f64],
) -> Result<Vec<f64>> {
    let mut solution = vec![0.0; diagonal.len()];
    if diagonal.is_empty() {
        return Ok(solution);
    }
    let mut residual = right_hand_side.to_vec();
    let mut preconditioned = residual
        .iter()
        .zip(diagonal)
        .map(|(value, diagonal)| value / diagonal)
        .collect::<Vec<_>>();
    let mut direction = preconditioned.clone();
    let mut residual_preconditioned = dot(&residual, &preconditioned);
    let rhs_norm = dot(right_hand_side, right_hand_side).sqrt();
    let tolerance = rhs_norm.max(1.0) * 1.0e-11;
    let max_iterations = diagonal.len().saturating_mul(4).clamp(64, 20_000);
    let mut product = vec![0.0; diagonal.len()];
    for _ in 0..max_iterations {
        if dot(&residual, &residual).sqrt() <= tolerance {
            return Ok(solution);
        }
        normal_matrix_product(rows, &direction, &mut product);
        let denominator = dot(&direction, &product);
        if denominator <= 0.0 || !denominator.is_finite() {
            break;
        }
        let alpha = residual_preconditioned / denominator;
        for index in 0..solution.len() {
            solution[index] += alpha * direction[index];
            residual[index] -= alpha * product[index];
            preconditioned[index] = residual[index] / diagonal[index];
        }
        let next_residual_preconditioned = dot(&residual, &preconditioned);
        if !next_residual_preconditioned.is_finite() {
            break;
        }
        let beta = next_residual_preconditioned / residual_preconditioned;
        for index in 0..direction.len() {
            direction[index] = preconditioned[index] + beta * direction[index];
        }
        residual_preconditioned = next_residual_preconditioned;
    }
    if dot(&residual, &residual).sqrt() <= tolerance * 10.0 {
        Ok(solution)
    } else {
        Err(PotError::new(
            ErrorCode::InvalidOperation,
            "conformal unwrap did not converge",
        ))
    }
}

fn normal_matrix_product(rows: &[LinearRow], vector: &[f64], output: &mut [f64]) {
    output.fill(0.0);
    for row in rows {
        let projected = row.terms[..row.term_count]
            .iter()
            .map(|(index, coefficient)| coefficient * vector[*index])
            .sum::<f64>();
        for &(index, coefficient) in &row.terms[..row.term_count] {
            output[index] += coefficient * projected;
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
/// The mesh-local positions of `face`'s corners, in face order.
fn face_points(mesh: &Mesh, face: &Face) -> Result<Vec<DVec3>> {
    face.vertices
        .iter()
        .map(|id| {
            mesh.vertex(*id).map(|vertex| vertex.co).ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "face references a missing vertex")
            })
        })
        .collect()
}

/// The corners of `face` for `smart`: the face projects onto the two axes its normal dominates, and
/// its coordinates normalise to [0, 1].
fn smart_project_face(mesh: &Mesh, face: &Face) -> Result<Vec<[f64; 2]>> {
    let points = face_points(mesh, face)?;
    let normal = points
        .windows(3)
        .find_map(|pair| {
            let cross = (pair[1] - pair[0]).cross(pair[2] - pair[0]);
            (cross.length_squared() > f64::EPSILON).then_some(cross.normalize())
        })
        .unwrap_or(DVec3::Z);
    let axis = if normal.x.abs() > normal.y.abs() && normal.x.abs() > normal.z.abs() {
        0
    } else if normal.y.abs() > normal.z.abs() {
        1
    } else {
        2
    };
    let mut projected = points
        .into_iter()
        .map(|point| match axis {
            0 => [point.y, point.z],
            1 => [point.x, point.z],
            _ => [point.x, point.y],
        })
        .collect::<Vec<_>>();
    normalize_uv(&mut projected);
    Ok(projected)
}

/// Blender's cube projection: `uvedit_unwrap_cube_project` in Blender 5.2.2's `uvedit_unwrap_ops.cc`,
/// as `uv.cube_project` runs it with `cube_size` unset and the default options. The centre and the
/// cube size come from the bounds of the selected faces, the cube size being the largest side (a
/// zero side gives 1). Each face projects onto the two axes its normal dominates. Blender's V is up
/// and Potter stores V down, so the stored V is `1 - v`.
fn cube_project(mesh: &Mesh, selected: &[u32]) -> Result<BTreeMap<u32, Vec<[f64; 2]>>> {
    let mut faces = Vec::new();
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for face in mesh.faces.iter().filter(|face| selected.contains(&face.id)) {
        let points = face_points(mesh, face)?;
        for point in &points {
            min = min.min(*point);
            max = max.max(*point);
        }
        faces.push((face.id, points));
    }
    if faces.is_empty() {
        return Ok(BTreeMap::new());
    }
    let side = (max - min).max_element();
    let cube_size = if crate::float::equal_f64(side, 0.0) {
        1.0
    } else {
        side
    };
    let centre = (min + max) * 0.5;
    Ok(faces
        .into_iter()
        .map(|(face_id, points)| {
            let (axis_u, axis_v) = dominant_axes(blender_face_normal(&points));
            let corners = points
                .iter()
                .map(|point| {
                    let u = 0.5 + (point[axis_u] - centre[axis_u]) / cube_size;
                    let v = 0.5 + (point[axis_v] - centre[axis_v]) / cube_size;
                    [u, 1.0 - v]
                })
                .collect::<Vec<[f64; 2]>>();
            (face_id, corners)
        })
        .collect())
}

/// The face normal used to pick the projection axes: Newell's method, unnormalised, for every face. A
/// box's faces are planar, where it points the same way as Blender's face normal, and a degenerate face
/// gives zero. Blender's triangle and quad normals are not reproduced, so a non-planar quad could tie
/// differently.
fn blender_face_normal(points: &[DVec3]) -> DVec3 {
    let count = points.len();
    let mut normal = DVec3::ZERO;
    for (index, current) in points.iter().enumerate() {
        let previous = points[(index + count - 1) % count];
        normal.x += (previous.y - current.y) * (previous.z + current.z);
        normal.y += (previous.z - current.z) * (previous.x + current.x);
        normal.z += (previous.x - current.x) * (previous.y + current.y);
    }
    normal
}

/// The two axes (0 = X, 1 = Y, 2 = Z) a face projects onto: Blender's `axis_dominant_v3`, which
/// checks Z first, then Y, then X.
fn dominant_axes(normal: DVec3) -> (usize, usize) {
    let (x, y, z) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
    if z >= x && z >= y {
        (0, 1)
    } else if y >= x && y >= z {
        (0, 2)
    } else {
        (1, 2)
    }
}

fn normalize_uv(coords: &mut [[f64; 2]]) {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for uv in coords.iter() {
        for axis in 0..2 {
            min[axis] = min[axis].min(uv[axis]);
            max[axis] = max[axis].max(uv[axis]);
        }
    }
    for uv in coords {
        for axis in 0..2 {
            let extent = max[axis] - min[axis];
            uv[axis] = if extent > f64::EPSILON {
                (uv[axis] - min[axis]) / extent
            } else {
                0.5
            };
        }
    }
}

#[derive(Clone, Debug)]
struct UvFace {
    uv: Vec<[f64; 2]>,
    pinned: Option<Vec<bool>>,
}

type UvMap = std::collections::BTreeMap<u32, UvFace>;

/// The face map of one stored layer, keyed by face ID. Storage face index `i` of `layer` is
/// `mesh.faces[i]`; faces without an entry are absent.
fn layer_uv_map(mesh: &Mesh, layer: &UvLayer) -> UvMap {
    mesh.faces
        .iter()
        .enumerate()
        .filter_map(|(index, face)| {
            let uv = layer.faces.get(index)?.clone()?;
            let pinned = layer.pinned.get(index).cloned().flatten();
            Some((face.id, UvFace { uv, pinned }))
        })
        .collect()
}

/// Stores `uv_map` as the faces of `layer`. Face ID rows go to their storage face index.
fn set_layer_uv_map(mesh: &Mesh, layer: &mut UvLayer, uv_map: &UvMap) {
    for ((face, faces), pinned) in mesh
        .faces
        .iter()
        .zip(layer.faces.iter_mut())
        .zip(layer.pinned.iter_mut())
    {
        let entry = uv_map.get(&face.id);
        *faces = entry.map(|face_uv| face_uv.uv.clone());
        *pinned = entry.and_then(|face_uv| face_uv.pinned.clone());
    }
}

#[derive(Clone, Copy)]
struct UvIsland {
    face_id: u32,
    min: [f64; 2],
    width: f64,
    height: f64,
}

const UV_PACK_MIN_EXTENT: f64 = 1.0e-12;

fn pack_selected_uvs(uv_map: &mut UvMap, selected: &[u32], margin: f64) -> Result<()> {
    if selected.is_empty() {
        return Ok(());
    }
    let mut islands = Vec::with_capacity(selected.len());
    for face_id in selected {
        let face_uv = uv_map.get(face_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!("uv.pack requires UV coordinates for face f{face_id}"),
            )
        })?;
        if face_uv.uv.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                format!("UV face f{face_id} has no corners"),
            ));
        }
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for uv in &face_uv.uv {
            for axis in 0..2 {
                if !uv[axis].is_finite() {
                    return Err(PotError::new(
                        ErrorCode::InvalidOperation,
                        "UV coordinates must be finite to pack",
                    ));
                }
                min[axis] = min[axis].min(uv[axis]);
                max[axis] = max[axis].max(uv[axis]);
            }
        }
        let width = max[0] - min[0];
        let height = max[1] - min[1];
        if !width.is_finite() || !height.is_finite() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV island bounds are too large to pack",
            ));
        }
        islands.push(UvIsland {
            face_id: *face_id,
            min,
            width,
            height,
        });
    }
    islands.sort_by(|left, right| {
        right
            .width
            .max(right.height)
            .total_cmp(&left.width.max(left.height))
            .then_with(|| left.face_id.cmp(&right.face_id))
    });

    let mut placements = Vec::with_capacity(islands.len());
    if !try_shelf_pack(&islands, 1.0, margin, &mut placements) {
        if !try_shelf_pack(&islands, 0.0, margin, &mut placements) {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV islands cannot fit in the unit square with this margin",
            ));
        }
        let mut best = placements.clone();
        let (mut low, mut high) = (0.0, 1.0);
        for _ in 0..64 {
            let scale = f64::midpoint(low, high);
            if try_shelf_pack(&islands, scale, margin, &mut placements) {
                low = scale;
                best.clone_from(&placements);
            } else {
                high = scale;
            }
        }
        placements = best;
        let scale = low * (1.0 - 1.0e-12);
        for (island, placement) in islands.iter().zip(placements) {
            let face_uv = uv_map.get_mut(&island.face_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "packed UV face disappeared")
            })?;
            for uv in &mut face_uv.uv {
                uv[0] = placement[0] + (uv[0] - island.min[0]) * scale;
                uv[1] = placement[1] + (uv[1] - island.min[1]) * scale;
            }
        }
        return Ok(());
    }

    for (island, placement) in islands.iter().zip(placements) {
        let face_uv = uv_map
            .get_mut(&island.face_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "packed UV face disappeared"))?;
        for uv in &mut face_uv.uv {
            uv[0] = placement[0] + (uv[0] - island.min[0]);
            uv[1] = placement[1] + (uv[1] - island.min[1]);
        }
    }
    Ok(())
}

fn try_shelf_pack(
    islands: &[UvIsland],
    scale: f64,
    margin: f64,
    placements: &mut Vec<[f64; 2]>,
) -> bool {
    placements.clear();
    let (mut x, mut y, mut row_height): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for island in islands {
        let width = (island.width * scale).max(UV_PACK_MIN_EXTENT);
        let height = (island.height * scale).max(UV_PACK_MIN_EXTENT);
        if x > 0.0 && x + width > 1.0 {
            y += row_height + margin;
            x = 0.0;
            row_height = 0.0;
        }
        if x + width > 1.0 || y + height > 1.0 {
            placements.clear();
            return false;
        }
        placements.push([x, y]);
        x += width + margin;
        row_height = row_height.max(height);
    }
    true
}

fn read_vec2(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: [f64; 2],
    field: &str,
) -> Result<[f64; 2]> {
    let Some(value) = value else {
        return Ok(default);
    };
    let array = value
        .as_array()
        .filter(|array| array.len() == 2)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must contain two numbers"),
                field,
            )
        })?;
    let x = array[0].as_f64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain numbers"),
            field,
        )
    })?;
    let y = array[1].as_f64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must contain numbers"),
            field,
        )
    })?;
    if !x.is_finite() || !y.is_finite() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be finite"),
            field,
        ));
    }
    Ok([x, y])
}

fn read_number(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: f64,
    field: &str,
) -> Result<f64> {
    let value = value.map_or(Ok(default), |value| {
        value.as_f64().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be numeric"),
                field,
            )
        })
    })?;
    if !value.is_finite() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be finite"),
            field,
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use proptest::prelude::*;

    use super::*;

    /// A `uv_layers` attribute holding one layer named `UVMap`, active and render.
    fn uv_layers_attribute(entries: &[Value]) -> Value {
        json!({"layers":[{"name":"UVMap","faces":entries}],"active":"UVMap","render":"UVMap"})
    }

    fn document_with_uvs(rectangles: &[(f64, f64)]) -> crate::model::SceneDoc {
        let initial = crate::model::SceneDoc::new("scene-test".to_owned());
        let outcome = super::super::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": initial.revision,
                "operations": [
                    {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}}
                ]
            }),
        )
        .unwrap();
        let mut doc = outcome.doc;
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        let entries = mesh
            .faces
            .iter()
            .zip(rectangles.iter().copied())
            .enumerate()
            .map(|(index, (face, (width, height)))| {
                let offset = f64::from(u32::try_from(index).unwrap()) * 2.0;
                json!({
                    "face_id": face.id,
                    "uv": [
                        [offset, 0.0],
                        [offset + width, 0.0],
                        [offset + width, height],
                        [offset, height]
                    ]
                })
            })
            .collect::<Vec<_>>();
        mesh.attributes
            .insert("uv_layers".to_owned(), uv_layers_attribute(&entries));
        doc
    }

    fn document_with_planar_grid() -> crate::model::SceneDoc {
        let mut doc = document_with_uvs(&[(1.0, 1.0); 6]);
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        mesh.vertices = (0..9_u32)
            .map(|id| crate::geom::Vertex {
                id,
                co: DVec3::new(f64::from(id % 3), f64::from(id / 3), 0.0),
            })
            .collect();
        mesh.faces = (0..2_u32)
            .flat_map(|y| {
                (0..2_u32).map(move |x| {
                    let lower_left = y * 3 + x;
                    crate::geom::Face {
                        id: y * 2 + x,
                        vertices: vec![lower_left, lower_left + 1, lower_left + 4, lower_left + 3],
                        material_index: 0,
                    }
                })
            })
            .collect();
        let mut edge_vertices = std::collections::BTreeSet::new();
        for face in &mesh.faces {
            for corner in 0..face.vertices.len() {
                edge_vertices.insert(crate::geom::edge_key(
                    face.vertices[corner],
                    face.vertices[(corner + 1) % face.vertices.len()],
                ));
            }
        }
        mesh.edges = edge_vertices
            .into_iter()
            .enumerate()
            .map(|(index, (left, right))| crate::geom::Edge {
                id: u32::try_from(index).unwrap(),
                vertices: [left, right],
            })
            .collect();
        mesh.next_id = crate::geom::IdCounters {
            vertex: 9,
            edge: u32::try_from(mesh.edges.len()).unwrap(),
            face: 4,
        };
        mesh.attributes.clear();
        doc
    }

    fn run_uv(doc: &mut crate::model::SceneDoc, operation: Value) -> Result<bool> {
        let Value::Object(object) = operation else {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "UV test operation must be an object",
            ));
        };
        let name = object.get("op").and_then(Value::as_str).ok_or_else(|| {
            PotError::new(ErrorCode::InvalidOperation, "UV test operation requires op")
        })?;
        let mut engine = super::super::Engine {
            doc,
            changes: super::super::ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            cache_directory: None,
            pending_assets: std::collections::BTreeMap::new(),
            warnings: Vec::new(),
        };
        apply(&mut engine, name, &object)
    }

    fn uv_entries(doc: &crate::model::SceneDoc) -> Vec<Value> {
        let data_id = Id::new("body_mesh").unwrap();
        doc.data_blocks[&data_id].mesh.as_ref().unwrap().attributes["uv_layers"]["layers"][0]["faces"]
            .as_array()
            .unwrap()
            .clone()
    }

    #[test]
    fn advanced_unwrap_reproduces_a_planar_grid_up_to_similarity() {
        for method in ["angle_based", "conformal"] {
            let mut doc = document_with_planar_grid();
            run_uv(
                &mut doc,
                json!({"op":"uv.unwrap","target":{"id":"body"},"method":method}),
            )
            .unwrap();

            let entries = uv_entries(&doc);
            let data_id = Id::new("body_mesh").unwrap();
            let mesh = doc.data_blocks[&data_id].mesh.as_ref().unwrap();
            let mut uv_by_vertex: Vec<Option<[f64; 2]>> = vec![None; mesh.vertices.len()];
            for face in &mesh.faces {
                let entry = entries
                    .iter()
                    .find(|entry| entry["face_id"].as_u64() == Some(u64::from(face.id)))
                    .unwrap();
                for (corner, &vertex_id) in face.vertices.iter().enumerate() {
                    let uv = [
                        entry["uv"][corner][0].as_f64().unwrap(),
                        entry["uv"][corner][1].as_f64().unwrap(),
                    ];
                    let index = usize::try_from(vertex_id).unwrap();
                    if let Some(previous) = uv_by_vertex[index] {
                        assert!(
                            (previous[0] - uv[0]).hypot(previous[1] - uv[1]) < 1.0e-8,
                            "{method} did not join unseamed face corners at v{vertex_id}"
                        );
                    } else {
                        uv_by_vertex[index] = Some(uv);
                    }
                }
            }

            let uv_by_vertex = uv_by_vertex
                .into_iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>();
            let scale = (uv_by_vertex[0][0] - uv_by_vertex[1][0])
                .hypot(uv_by_vertex[0][1] - uv_by_vertex[1][1]);
            assert!(scale > 0.0 && scale.is_finite());
            for left in 0..uv_by_vertex.len() {
                for right in left + 1..uv_by_vertex.len() {
                    let left_xy = [
                        f64::from(u32::try_from(left).unwrap() % 3),
                        f64::from(u32::try_from(left).unwrap() / 3),
                    ];
                    let right_xy = [
                        f64::from(u32::try_from(right).unwrap() % 3),
                        f64::from(u32::try_from(right).unwrap() / 3),
                    ];
                    let expected =
                        (left_xy[0] - right_xy[0]).hypot(left_xy[1] - right_xy[1]) * scale;
                    let actual = (uv_by_vertex[left][0] - uv_by_vertex[right][0])
                        .hypot(uv_by_vertex[left][1] - uv_by_vertex[right][1]);
                    assert!(
                        (actual - expected).abs() <= scale * 1.0e-7,
                        "{method} changed grid distance {left}-{right}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn advanced_unwrap_preserves_pins_and_splits_seams() {
        let mut doc = document_with_planar_grid();
        let data_id = Id::new("body_mesh").unwrap();
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        let entries = mesh
            .faces
            .iter()
            .map(|face| {
                let mut uv = face
                    .vertices
                    .iter()
                    .map(|vertex_id| {
                        let point = mesh.vertex(*vertex_id).unwrap().co;
                        [point.x, point.y]
                    })
                    .collect::<Vec<_>>();
                let mut entry = json!({"face_id":face.id,"uv":uv});
                if face.id == 0 {
                    uv[0] = [-3.0, 5.0];
                    entry["uv"] = json!(uv);
                    entry["pinned"] = json!([true, false, false, false]);
                }
                entry
            })
            .collect::<Vec<_>>();
        let seam_edge_id = mesh
            .edges
            .iter()
            .find(|edge| crate::geom::edge_key(edge.vertices[0], edge.vertices[1]) == (1, 4))
            .unwrap()
            .id;
        let mut seam_values = serde_json::Map::new();
        seam_values.insert(format!("e{seam_edge_id}"), json!(true));
        mesh.attributes
            .insert("uv_layers".to_owned(), uv_layers_attribute(&entries));
        mesh.attributes.insert(
            "seam".to_owned(),
            json!({"domain":"edge","type":"bool","values":seam_values}),
        );
        run_uv(
            &mut doc,
            json!({"op":"uv.unwrap","target":{"id":"body"},"method":"angle_based"}),
        )
        .unwrap();

        let entries = uv_entries(&doc);
        let pinned_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(0))
            .unwrap();
        assert_eq!(pinned_entry["uv"][0], json!([-3.0, 5.0]));
        assert_eq!(pinned_entry["pinned"][0], json!(true));

        let mesh = doc.data_blocks[&data_id].mesh.as_ref().unwrap();
        let left_face = mesh.faces.iter().find(|face| face.id == 0).unwrap();
        let right_face = mesh.faces.iter().find(|face| face.id == 1).unwrap();
        let left_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(0))
            .unwrap();
        let right_entry = entries
            .iter()
            .find(|entry| entry["face_id"].as_u64() == Some(1))
            .unwrap();
        let split = [1, 4].into_iter().any(|vertex_id| {
            let left_corner = left_face
                .vertices
                .iter()
                .position(|candidate| *candidate == vertex_id)
                .unwrap();
            let right_corner = right_face
                .vertices
                .iter()
                .position(|candidate| *candidate == vertex_id)
                .unwrap();
            left_entry["uv"][left_corner] != right_entry["uv"][right_corner]
        });
        assert!(split, "marked seam did not split its shared UV vertices");
    }

    proptest! {
        #[test]
        fn packed_uv_faces_stay_in_unit_square_and_have_disjoint_bounds(
            rectangles in prop::collection::vec((0.01_f64..1.0, 0.01_f64..1.0), 1..=6),
            margin in 0.0_f64..0.08,
        ) {
            let mut doc = document_with_uvs(&rectangles);
            let selected_ids = uv_entries(&doc)
                .iter()
                .map(|entry| format!("f{}", entry["face_id"].as_u64().unwrap()))
                .collect::<Vec<_>>();
            run_uv(&mut doc, json!({
                "op":"uv.pack",
                "target":{"id":"body"},
                "elements":{"domain":"face","ids":selected_ids},
                "margin":margin
            })).unwrap();

            let packed = uv_entries(&doc);
            let mut bounds = Vec::with_capacity(packed.len());
            for entry in &packed {
                let corners = entry["uv"].as_array().unwrap();
                prop_assert_eq!(corners.len(), 4);
                let mut min = [f64::INFINITY; 2];
                let mut max = [f64::NEG_INFINITY; 2];
                let mut uv = [[0.0; 2]; 4];
                for (index, corner) in corners.iter().enumerate() {
                    for axis in 0..2 {
                        let value = corner[axis].as_f64().unwrap();
                        prop_assert!((0.0..=1.0).contains(&value));
                        uv[index][axis] = value;
                        min[axis] = min[axis].min(value);
                        max[axis] = max[axis].max(value);
                    }
                }
                prop_assert_eq!(uv[0][1], uv[1][1]);
                prop_assert_eq!(uv[1][0], uv[2][0]);
                prop_assert_eq!(uv[2][1], uv[3][1]);
                prop_assert_eq!(uv[3][0], uv[0][0]);
                bounds.push((min, max));
            }

            for left in 0..bounds.len() {
                for right in left + 1..bounds.len() {
                    let (left_min, left_max) = bounds[left];
                    let (right_min, right_max) = bounds[right];
                    let separated = left_max[0] + margin <= right_min[0] + 1.0e-12
                        || right_max[0] + margin <= left_min[0] + 1.0e-12
                        || left_max[1] + margin <= right_min[1] + 1.0e-12
                        || right_max[1] + margin <= left_min[1] + 1.0e-12;
                    prop_assert!(separated, "packed face bounds overlap or violate margin");
                }
            }
        }
    }

    #[test]
    fn pins_survive_face_transforms_and_unwrap() {
        let mut doc = document_with_uvs(&[(1.0, 1.0); 6]);
        let first_face = {
            let data_id = Id::new("body_mesh").unwrap();
            doc.data_blocks[&data_id].mesh.as_ref().unwrap().faces[0].id
        };
        let selector = json!({"domain":"face","ids":[format!("f{first_face}")]});
        run_uv(
            &mut doc,
            json!({
                "op":"uv.pin",
                "target":{"id":"body"},
                "elements":selector,
                "pinned":true
            }),
        )
        .unwrap();

        assert!(
            uv_entries(&doc)
                .iter()
                .filter(|entry| entry["face_id"].as_u64() != Some(u64::from(first_face)))
                .all(|entry| entry.get("pinned").is_none())
        );
        run_uv(
            &mut doc,
            json!({
                "op":"uv.transform",
                "target":{"id":"body"},
                "elements":selector,
                "translation":[0.25,-0.5]
            }),
        )
        .unwrap();
        run_uv(
            &mut doc,
            json!({
                "op":"uv.unwrap",
                "target":{"id":"body"},
                "elements":selector,
                "method":"box"
            }),
        )
        .unwrap();

        let pinned = uv_entries(&doc)
            .into_iter()
            .find(|entry| entry["face_id"].as_u64() == Some(u64::from(first_face)))
            .unwrap()["pinned"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(pinned, vec![json!(true); 4]);

        run_uv(
            &mut doc,
            json!({
                "op":"uv.pin",
                "target":{"id":"body"},
                "elements":selector,
                "pinned":false
            }),
        )
        .unwrap();
        let unpinned = uv_entries(&doc)
            .into_iter()
            .find(|entry| entry["face_id"].as_u64() == Some(u64::from(first_face)))
            .unwrap()["pinned"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(unpinned, vec![json!(false); 4]);
    }

    fn uv_store(doc: &crate::model::SceneDoc) -> Value {
        let data_id = Id::new("body_mesh").unwrap();
        doc.data_blocks[&data_id].mesh.as_ref().unwrap().attributes["uv_layers"].clone()
    }

    fn layer_names(doc: &crate::model::SceneDoc) -> Vec<String> {
        uv_store(doc)["layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|layer| layer["name"].as_str().unwrap().to_owned())
            .collect()
    }

    fn feature_of(error: &PotError) -> Option<&str> {
        error.details["feature_id"].as_str()
    }

    fn create_layer(doc: &mut crate::model::SceneDoc, name: Option<&str>) -> Result<bool> {
        let mut operation = json!({"op":"uv.layer_create","target":{"id":"body"}});
        if let Some(name) = name {
            operation["name"] = json!(name);
        }
        run_uv(doc, operation)
    }

    #[test]
    fn layers_create_rename_and_select_by_name() {
        let mut doc = document_with_planar_grid();
        run_uv(
            &mut doc,
            json!({"op":"uv.unwrap","target":{"id":"body"},"method":"box"}),
        )
        .unwrap();
        assert_eq!(uv_store(&doc)["active"], json!("UVMap"));
        assert_eq!(uv_store(&doc)["render"], json!("UVMap"));

        create_layer(&mut doc, None).unwrap();
        create_layer(&mut doc, None).unwrap();
        create_layer(&mut doc, Some("Detail")).unwrap();
        assert_eq!(
            layer_names(&doc),
            ["UVMap", "UVMap.001", "UVMap.002", "Detail"]
        );
        assert_eq!(
            uv_store(&doc)["active"],
            json!("UVMap"),
            "a later layer leaves the active and render layers unchanged"
        );

        let taken = create_layer(&mut doc, Some("Detail")).unwrap_err();
        assert_eq!(feature_of(&taken), Some("uv.layer_exists"));
        let clash = run_uv(
            &mut doc,
            json!({"op":"uv.layer_rename","target":{"id":"body"},"name":"Detail","to":"UVMap"}),
        )
        .unwrap_err();
        assert_eq!(feature_of(&clash), Some("uv.layer_exists"));

        run_uv(
            &mut doc,
            json!({"op":"uv.layer_rename","target":{"id":"body"},"name":"Detail","to":"Lightmap"}),
        )
        .unwrap();
        run_uv(
            &mut doc,
            json!({"op":"uv.layer_set_render","target":{"id":"body"},"name":"Lightmap"}),
        )
        .unwrap();
        run_uv(
            &mut doc,
            json!({"op":"uv.layer_set_active","target":{"id":"body"},"name":"UVMap.001"}),
        )
        .unwrap();
        assert_eq!(
            layer_names(&doc),
            ["UVMap", "UVMap.001", "UVMap.002", "Lightmap"]
        );
        assert_eq!(uv_store(&doc)["render"], json!("Lightmap"));
        assert_eq!(uv_store(&doc)["active"], json!("UVMap.001"));
    }

    #[test]
    fn layer_delete_refuses_referenced_layers_and_clamps_positions() {
        let initial = crate::model::SceneDoc::new("scene-test".to_owned());
        let outcome = super::super::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": initial.revision,
                "operations": [
                    {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}},
                    {"op":"material.create","id":"surface"}
                ]
            }),
        )
        .unwrap();
        let mut doc = outcome.doc;
        let surface = Id::new("surface").unwrap();
        doc.materials.get_mut(&surface).unwrap().roughness_texture =
            Some(crate::model::TextureRef {
                image: Id::new("paint").unwrap(),
                uv_layer: Some("Mid".to_owned()),
                interpolation: crate::image::ImageInterpolation::Linear,
            });
        run_uv(
            &mut doc,
            json!({"op":"uv.unwrap","target":{"id":"body"},"method":"box"}),
        )
        .unwrap();
        doc.nodes
            .get_mut(&Id::new("body").unwrap())
            .unwrap()
            .materials
            .push(Some(surface.clone()));
        create_layer(&mut doc, Some("Mid")).unwrap();
        create_layer(&mut doc, Some("Top")).unwrap();
        doc.nodes
            .get_mut(&Id::new("body").unwrap())
            .unwrap()
            .modifiers
            .push(crate::model::Modifier {
                id: Id::new("uv_project").unwrap(),
                modifier_type: "uv_project".to_owned(),
                name: "UV Project".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: serde_json::Map::from_iter([("uv_layer".to_owned(), json!("Mid"))]),
                binding_data: None,
                baked_ocean_cache: None,
                runtime: crate::model::ModifierRuntime::default(),
            });
        run_uv(
            &mut doc,
            json!({"op":"uv.layer_set_render","target":{"id":"body"},"name":"Top"}),
        )
        .unwrap();
        run_uv(
            &mut doc,
            json!({"op":"uv.layer_set_active","target":{"id":"body"},"name":"Top"}),
        )
        .unwrap();
        let graph_id = Id::new("surface_graph").unwrap();
        let uv_node_id = Id::new("uv_input").unwrap();
        let mut graph = crate::graph::NodeGroup::new("Surface", crate::graph::GraphKind::Shader);
        let mut uv_node = crate::graph::GraphNode::new("ShaderNodeUVMap");
        uv_node.properties.insert("uv_map".to_owned(), json!("Top"));
        graph.nodes.insert(uv_node_id.clone(), uv_node);
        doc.node_groups.insert(graph_id.clone(), graph);
        doc.materials.get_mut(&surface).unwrap().node_tree = Some(graph_id.clone());

        run_uv(
            &mut doc,
            json!({"op":"uv.layer_rename","target":{"id":"body"},"name":"Mid","to":"Center"}),
        )
        .unwrap();
        assert_eq!(
            doc.materials[&surface]
                .roughness_texture
                .as_ref()
                .unwrap()
                .uv_layer
                .as_deref(),
            Some("Center"),
            "rename updates the texture reference in the same operation"
        );
        assert_eq!(
            doc.nodes[&Id::new("body").unwrap()].modifiers[0].params["uv_layer"],
            json!("Center"),
            "rename updates modifier UV layer references"
        );

        let refused = run_uv(
            &mut doc,
            json!({"op":"uv.layer_delete","target":{"id":"body"},"name":"Center"}),
        )
        .unwrap_err();
        assert_eq!(feature_of(&refused), Some("uv.layer_in_use"));
        assert!(
            refused.details["referrers"][0]
                .as_str()
                .unwrap()
                .contains("surface")
        );
        assert!(
            refused.details["referrers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|referrer| referrer.as_str().unwrap().contains("modifier `uv_project`"))
        );
        let graph_refused = run_uv(
            &mut doc,
            json!({"op":"uv.layer_delete","target":{"id":"body"},"name":"Top"}),
        )
        .unwrap_err();
        assert_eq!(feature_of(&graph_refused), Some("uv.layer_in_use"));
        assert!(
            graph_refused.details["referrers"][0]
                .as_str()
                .unwrap()
                .contains("surface_graph")
        );
        run_uv(
            &mut doc,
            json!({"op":"uv.layer_rename","target":{"id":"body"},"name":"Top","to":"Top2"}),
        )
        .unwrap();
        assert_eq!(
            doc.node_groups[&graph_id].nodes[&uv_node_id].properties["uv_map"],
            json!("Top2")
        );

        run_uv(
            &mut doc,
            json!({"op":"uv.layer_delete","target":{"id":"body"},"name":"UVMap"}),
        )
        .unwrap();
        assert_eq!(layer_names(&doc), ["Center", "Top2"]);
        assert_eq!(
            uv_store(&doc)["active"],
            json!("Top2"),
            "index 2 clamps to 1"
        );
        assert_eq!(
            uv_store(&doc)["render"],
            json!("Top2"),
            "index 2 clamps to 1"
        );
    }

    #[test]
    fn layers_require_existing_names_and_stop_at_eight() {
        let mut doc = document_with_planar_grid();
        let missing = run_uv(
            &mut doc,
            json!({"op":"uv.transform","target":{"id":"body"},"translation":[1.0,0.0]}),
        )
        .unwrap_err();
        assert_eq!(feature_of(&missing), Some("uv.layer_missing"));

        run_uv(
            &mut doc,
            json!({"op":"uv.unwrap","target":{"id":"body"},"method":"box"}),
        )
        .unwrap();
        let unknown = run_uv(
            &mut doc,
            json!({"op":"uv.pack","target":{"id":"body"},"uv_layer":"Nope"}),
        )
        .unwrap_err();
        assert_eq!(feature_of(&unknown), Some("uv.layer_missing"));

        for _ in 1..uv_layers::MAX_LAYERS {
            create_layer(&mut doc, None).unwrap();
        }
        assert_eq!(layer_names(&doc).len(), uv_layers::MAX_LAYERS);
        let limit = create_layer(&mut doc, None).unwrap_err();
        assert!(matches!(limit.code, ErrorCode::LimitExceeded));
        assert_eq!(feature_of(&limit), Some("uv.layer_limit"));
    }
}
