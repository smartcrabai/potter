use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DQuat};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, Result},
    geom::{self, Mesh as GeomMesh},
    model::{
        ArmatureData, CameraData, DataBlock, GreasePencilData, Id, LightData, Node, ParentType,
        PrimitiveDescriptor, Transform, canonicalize_quaternion, normalize_rotation,
    },
};

use super::{
    ChangeKind, Engine, array_strings, check_fields, operation_pointer, parse_id, pointer_escape,
    read_bool, read_id, read_string, resolve_node_targets,
};

#[derive(Default)]
struct TransformUpdate {
    translation: Option<[f64; 3]>,
    scale: Option<[f64; 3]>,
    rotation: Option<[f64; 4]>,
    sets_quaternion_mode: bool,
    rotation_mode: Option<String>,
}

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "node.create" => create(engine, operation),
        "node.update" => update(engine, operation),
        "node.delete" => delete(engine, operation),
        "node.duplicate" => duplicate(engine, operation),
        "node.parent" => parent(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid node operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

pub(super) fn make_single_user(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "data_id"], &["target"])?;
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let targets = resolve_node_targets(engine, target, true)?;
    if targets.len() > 1 && operation.contains_key("data_id") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "data_id is only valid for a single target",
            &operation_pointer(engine.operation_index, "data_id"),
        ));
    }
    let explicit_id = if operation.contains_key("data_id") {
        Some(read_id(engine, operation, "data_id")?)
    } else {
        None
    };
    let mut by_data = BTreeMap::<Id, Vec<Id>>::new();
    for node_id in targets {
        let data_id = engine
            .doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{node_id}` has no data block"),
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
        by_data.entry(data_id).or_default().push(node_id);
    }
    let user_counts: BTreeMap<_, _> = by_data
        .keys()
        .map(|data_id| (data_id.clone(), super::data_user_count(engine, data_id)))
        .collect();
    let mut changed = false;
    for (data_id, node_ids) in by_data {
        let user_count = user_counts.get(&data_id).copied().ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "Data-Block user count is missing",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        for node_id in node_ids {
            if user_count <= 1 {
                continue;
            }
            let new_id = if let Some(id) = explicit_id.clone() {
                id
            } else {
                generated_data_id(engine, &node_id, "_single")?
            };
            if engine.doc.data_blocks.contains_key(&new_id) {
                return Err(engine.error(
                    ErrorCode::IdExists,
                    format!("data-block ID `{new_id}` already exists"),
                    &operation_pointer(
                        engine.operation_index,
                        if explicit_id.is_some() {
                            "data_id"
                        } else {
                            "target"
                        },
                    ),
                ));
            }
            let data = engine
                .doc
                .data_blocks
                .get(&data_id)
                .cloned()
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` was not found"),
                        &operation_pointer(engine.operation_index, "target"),
                    )
                })?;
            engine.doc.data_blocks.insert(new_id.clone(), data);
            if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
                node.data = Some(new_id.clone());
            }
            engine.mark("data_blocks", &new_id, ChangeKind::Created);
            engine.mark("nodes", &node_id, ChangeKind::Updated);
            engine.map_id("data_blocks", &data_id, &new_id);
            changed = true;
        }
    }
    Ok(changed)
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "kind",
            "name",
            "tags",
            "parent",
            "parent_inverse",
            "transform",
            "params",
            "data",
            "material",
            "materials",
            "collection",
            "visible",
            "render_visible",
            "selectable",
        ],
        &["id", "kind"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.nodes.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("node ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let kind_value = read_string(engine, operation, "kind")?;
    let (kind, primitive) = match kind_value.as_str() {
        "group" | "empty" => ("empty".to_owned(), None),
        "sphere" => ("mesh".to_owned(), Some("uv_sphere".to_owned())),
        "box" | "uv_sphere" | "cylinder" | "plane" | "cone" | "torus" | "icosphere" | "circle"
        | "grid" => ("mesh".to_owned(), Some(kind_value)),
        "mesh" => ("mesh".to_owned(), None),
        "camera" | "light" | "armature" | "grease_pencil" | "curve" | "surface" | "text"
        | "metaball" | "lattice" | "pointcloud" | "volume" => (kind_value, None),
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported node kind `{kind_value}`"),
                &operation_pointer(engine.operation_index, "kind"),
            ));
        }
    };
    if let Some(params) = operation.get("params")
        && primitive.is_none()
        && !params.as_object().is_some_and(Map::is_empty)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "non-geometry node params must be an empty object",
            &operation_pointer(engine.operation_index, "params"),
        ));
    }
    let expected_data_type = match kind.as_str() {
        "mesh" => Some("mesh"),
        "camera" => Some("camera"),
        "light" => Some("light"),
        "armature" => Some("armature"),
        "grease_pencil" => Some("grease_pencil"),
        "curve" => Some("curve"),
        "surface" => Some("surface"),
        "text" => Some("text"),
        "metaball" => Some("metaball"),
        "lattice" => Some("lattice"),
        "pointcloud" => Some("pointcloud"),
        "volume" => Some("volume"),
        _ => None,
    };
    let data_id = if let Some(data_value) = operation.get("data") {
        if primitive.is_some()
            || operation
                .get("params")
                .is_some_and(|params| !params.as_object().is_some_and(Map::is_empty))
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "data cannot be combined with primitive params",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        let text = data_value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "data must be a data-block ID",
                &operation_pointer(engine.operation_index, "data"),
            )
        })?;
        let data_id = parse_id(
            engine,
            text,
            &operation_pointer(engine.operation_index, "data"),
        )?;
        let data_block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("data block `{data_id}` was not found"),
                &operation_pointer(engine.operation_index, "data"),
            )
        })?;
        if expected_data_type != Some(data_block.data_type.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("node kind `{kind}` requires a compatible Data-Block"),
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "camera" && data_block.camera.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "camera Data-Block has no camera payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "light" && data_block.light.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "light Data-Block has no light payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "armature" && data_block.armature.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "armature Data-Block has no armature payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "grease_pencil" && data_block.grease_pencil.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Grease Pencil Data-Block has no grease-pencil payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "curve" && data_block.curve.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Curve Data-Block has no curve payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "surface" && data_block.surface.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Surface Data-Block has no surface payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "text" && data_block.text.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Text Data-Block has no text payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "metaball" && data_block.metaball.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Metaball Data-Block has no metaball payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "lattice" && data_block.lattice.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Lattice Data-Block has no lattice payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "pointcloud" && data_block.pointcloud.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Point-cloud Data-Block has no point-cloud payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        if kind == "volume" && data_block.volume.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "Volume Data-Block has no volume payload",
                &operation_pointer(engine.operation_index, "data"),
            ));
        }
        Some(data_id)
    } else if let Some(primitive) = primitive.as_deref() {
        if !operation.contains_key("params") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "primitive node.create requires params",
                &operation_pointer(engine.operation_index, "params"),
            ));
        }
        let params = operation.get("params").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "params is required",
                &operation_pointer(engine.operation_index, "params"),
            )
        })?;
        let (mesh, descriptor_params) =
            build_primitive(engine, primitive, params.clone(), "params")?;
        let data_id = generated_data_id(engine, &id, "_mesh")?;
        if engine.doc.data_blocks.contains_key(&data_id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("data-block ID `{data_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        engine.doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: Some(PrimitiveDescriptor {
                    primitive: primitive.to_owned(),
                    params: descriptor_params,
                }),
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        engine.mark("data_blocks", &data_id, ChangeKind::Created);
        Some(data_id)
    } else if kind == "armature" {
        let data_id = generated_data_id(engine, &id, "_armature")?;
        if engine.doc.data_blocks.contains_key(&data_id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("data-block ID `{data_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        engine.doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "armature".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: None,
                armature: Some(ArmatureData::default()),
                ..DataBlock::default()
            },
        );
        engine.mark("data_blocks", &data_id, ChangeKind::Created);
        Some(data_id)
    } else if kind == "grease_pencil" {
        let data_id = generated_data_id(engine, &id, "_grease_pencil")?;
        if engine.doc.data_blocks.contains_key(&data_id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("data-block ID `{data_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        engine.doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "grease_pencil".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: None,
                grease_pencil: Some(GreasePencilData { layers: Vec::new() }),
                ..DataBlock::default()
            },
        );
        engine.mark("data_blocks", &data_id, ChangeKind::Created);
        Some(data_id)
    } else if kind == "camera" || kind == "light" {
        let suffix = if kind == "camera" {
            "_camera"
        } else {
            "_light"
        };
        let data_id = generated_data_id(engine, &id, suffix)?;
        if engine.doc.data_blocks.contains_key(&data_id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("data-block ID `{data_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        engine.doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: kind.clone(),
                descriptor: None,
                mesh: None,
                camera: (kind == "camera").then(CameraData::default),
                light: (kind == "light").then(LightData::default),
                ..DataBlock::default()
            },
        );
        engine.mark("data_blocks", &data_id, ChangeKind::Created);
        Some(data_id)
    } else {
        None
    };
    if kind == "mesh" && data_id.is_none() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mesh nodes require a mesh Data-Block",
            &operation_pointer(engine.operation_index, "data"),
        ));
    }
    if matches!(
        kind.as_str(),
        "curve" | "surface" | "text" | "metaball" | "lattice" | "pointcloud" | "volume"
    ) && data_id.is_none()
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("node kind `{kind}` requires a compatible Data-Block"),
            &operation_pointer(engine.operation_index, "data"),
        ));
    }
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    let tags = if let Some(value) = operation.get("tags") {
        array_strings(
            engine,
            value,
            &operation_pointer(engine.operation_index, "tags"),
        )?
    } else {
        Vec::new()
    };
    let parent = if operation.get("parent").is_some_and(Value::is_null) {
        None
    } else if operation.contains_key("parent") {
        Some(read_id(engine, operation, "parent")?)
    } else {
        None
    };
    if let Some(parent_id) = &parent
        && !engine.doc.nodes.contains_key(parent_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("parent node `{parent_id}` was not found"),
            &operation_pointer(engine.operation_index, "parent"),
        ));
    }
    let mut transform = Transform::default();
    if let Some(value) = operation.get("transform") {
        let transform_update = parse_transform_update(engine, value, "transform")?;
        apply_transform_update(&mut transform, &transform_update);
    }
    let parent_inverse = if let Some(value) = operation.get("parent_inverse") {
        Some(read_matrix(
            engine,
            value,
            &operation_pointer(engine.operation_index, "parent_inverse"),
        )?)
    } else {
        None
    };
    let materials = read_material_slots(engine, operation)?;
    let node = Node {
        name,
        kind,
        primitive,
        tags,
        parent,
        parent_inverse,
        transform,
        data: data_id,
        materials,
        modifiers: Vec::new(),
        visible: read_bool(engine, operation, "visible", true)?,
        render_visible: read_bool(engine, operation, "render_visible", true)?,
        selectable: read_bool(engine, operation, "selectable", true)?,
        action: None,
        properties: Map::new(),
        rigid_body: None,
        force_field: None,
        ..Node::default()
    };
    engine.doc.nodes.insert(id.clone(), node);
    engine.mark("nodes", &id, ChangeKind::Created);
    let collection_id = if operation.contains_key("collection") {
        read_id(engine, operation, "collection")?
    } else {
        engine
            .doc
            .scenes
            .get(&engine.doc.active_scene)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "active scene is missing",
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?
            .root_collection
            .clone()
    };
    if !engine.doc.collections.contains_key(&collection_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("collection `{collection_id}` was not found"),
            &operation_pointer(
                engine.operation_index,
                if operation.contains_key("collection") {
                    "collection"
                } else {
                    "id"
                },
            ),
        ));
    }
    let changed = engine
        .doc
        .collections
        .get_mut(&collection_id)
        .is_some_and(|collection| {
            if collection.objects.contains(&id) {
                false
            } else {
                collection.objects.push(id.clone());
                collection.objects.sort();
                true
            }
        });
    if changed {
        engine.mark("collections", &collection_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "set"],
        &["target", "set"],
    )?;
    let target_value = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let targets = resolve_node_targets(engine, target_value, true)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    super::check_set_fields_by(
        engine,
        set,
        &[
            "name",
            "tags",
            "transform",
            "params",
            "materials",
            "visible",
            "render_visible",
            "selectable",
            "action",
        ],
        |engine, key| {
            let escaped = super::pointer_escape(key);
            engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown node set field `{key}`"),
                &operation_pointer(engine.operation_index, &format!("set/{escaped}")),
            )
        },
    )?;
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        &operation_pointer(engine.operation_index, "scope"),
        "scope must be a string",
        "scope must be `shared` or `single_user`",
        true,
    )?;
    if scope.is_some() && !set.contains_key("params") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "scope is only valid when updating params",
            &operation_pointer(engine.operation_index, "scope"),
        ));
    }
    let name = set
        .get("name")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    &operation_pointer(engine.operation_index, "set/name"),
                )
            })
        })
        .transpose()?;
    let tags = set
        .get("tags")
        .map(|value| {
            array_strings(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/tags"),
            )
        })
        .transpose()?;
    let transform_update = set
        .get("transform")
        .map(|value| parse_transform_update(engine, value, "set/transform"))
        .transpose()?;
    let materials = set
        .get("materials")
        .map(|value| read_material_array(engine, value, "set/materials"))
        .transpose()?;
    let visible = read_optional_bool(engine, set, "visible")?;
    let render_visible = read_optional_bool(engine, set, "render_visible")?;
    let selectable = read_optional_bool(engine, set, "selectable")?;
    let action = set
        .get("action")
        .map(|value| {
            if value.is_null() {
                return Ok(None);
            }
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "action must be an ID or null",
                    &operation_pointer(engine.operation_index, "set/action"),
                )
            })?;
            let id = parse_id(
                engine,
                text,
                &operation_pointer(engine.operation_index, "set/action"),
            )?;
            if !engine.doc.actions.contains_key(&id) {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    format!("action `{id}` was not found"),
                    &operation_pointer(engine.operation_index, "set/action"),
                ));
            }
            Ok(Some(id))
        })
        .transpose()?;
    let mut changed = if let Some(params) = set.get("params") {
        update_params(engine, &targets, params, scope)?
    } else {
        false
    };
    for node_id in targets {
        let item_changed = if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
            let mut item_changed = false;
            if let Some(name) = name
                && node.name != name
            {
                name.clone_into(&mut node.name);
                item_changed = true;
            }
            if let Some(action) = &action
                && node.action.as_ref() != action.as_ref()
            {
                node.action.clone_from(action);
                item_changed = true;
            }
            if let Some(tags) = &tags
                && node.tags.as_slice() != tags.as_slice()
            {
                node.tags.clone_from(tags);
                item_changed = true;
            }
            if let Some(transform) = &transform_update {
                item_changed |= apply_transform_update(&mut node.transform, transform);
            }
            if let Some(materials) = &materials
                && node.materials.as_slice() != materials.as_slice()
            {
                node.materials.clone_from(materials);
                item_changed = true;
            }
            for (field, value) in [
                ("visible", visible),
                ("render_visible", render_visible),
                ("selectable", selectable),
            ] {
                if let Some(value) = value {
                    let property = match field {
                        "visible" => &mut node.visible,
                        "render_visible" => &mut node.render_visible,
                        _ => &mut node.selectable,
                    };
                    if *property != value {
                        *property = value;
                        item_changed = true;
                    }
                }
            }
            item_changed
        } else {
            false
        };
        if item_changed {
            engine.mark("nodes", &node_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn read_optional_bool(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<bool>> {
    object
        .get(field)
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{field} must be a boolean"),
                    &operation_pointer(engine.operation_index, &format!("set/{field}")),
                )
            })
        })
        .transpose()
}

fn update_params(
    engine: &mut Engine<'_>,
    targets: &[Id],
    params_value: &Value,
    scope: Option<&str>,
) -> Result<bool> {
    let Some(params) = params_value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "params must be an object",
            &operation_pointer(engine.operation_index, "set/params"),
        ));
    };
    let mut by_data = BTreeMap::<Id, Vec<Id>>::new();
    for target in targets {
        let data_id = engine
            .doc
            .nodes
            .get(target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no geometry data to update"),
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
        by_data.entry(data_id).or_default().push(target.clone());
    }
    let shared = by_data
        .keys()
        .any(|data_id| super::data_user_count(engine, data_id) > 1);
    if shared && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry updates require scope",
            &operation_pointer(engine.operation_index, "scope"),
        ));
    }
    let scope = scope.unwrap_or("shared");
    let mut changed = false;
    if scope == "single_user" {
        for (data_id, target_ids) in &by_data {
            let user_count = super::data_user_count(engine, data_id);
            for target in target_ids {
                if user_count <= 1 {
                    continue;
                }
                let new_id = generated_data_id(engine, target, "_single")?;
                if engine.doc.data_blocks.contains_key(&new_id) {
                    return Err(engine.error(
                        ErrorCode::IdExists,
                        format!("data-block ID `{new_id}` already exists"),
                        &operation_pointer(engine.operation_index, "set/params"),
                    ));
                }
                let copy = engine
                    .doc
                    .data_blocks
                    .get(data_id)
                    .cloned()
                    .ok_or_else(|| {
                        engine.error(
                            ErrorCode::TargetNotFound,
                            format!("data block `{data_id}` was not found"),
                            &operation_pointer(engine.operation_index, "set/params"),
                        )
                    })?;
                engine.doc.data_blocks.insert(new_id.clone(), copy);
                if let Some(node) = engine.doc.nodes.get_mut(target) {
                    node.data = Some(new_id.clone());
                }
                engine.mark("data_blocks", &new_id, ChangeKind::Created);
                engine.mark("nodes", target, ChangeKind::Updated);
                engine.map_id("data_blocks", data_id, &new_id);
                changed = true;
            }
        }
    }
    let mut modified = BTreeSet::new();
    for target in targets {
        let data_id = engine
            .doc
            .nodes
            .get(target)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target}` has no data"),
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
        if scope == "shared" && !modified.insert(data_id.clone()) {
            continue;
        }
        let data_block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("data block `{data_id}` was not found"),
                &operation_pointer(engine.operation_index, "set/params"),
            )
        })?;
        let descriptor = data_block.descriptor.as_ref().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("data block `{data_id}` has no editable primitive descriptor"),
                &operation_pointer(engine.operation_index, "set/params"),
            )
        })?;
        let mut merged = descriptor.params.clone();
        for (key, value) in params {
            merged.insert(key.clone(), value.clone());
        }
        let params_value = Value::Object(merged);
        let primitive = descriptor.primitive.clone();
        let (mesh, descriptor_params) =
            build_primitive(engine, &primitive, params_value, "set/params")?;
        if descriptor_params != descriptor.params || data_block.mesh.as_ref() != Some(&mesh) {
            if let Some(data) = engine.doc.data_blocks.get_mut(&data_id) {
                data.descriptor = Some(PrimitiveDescriptor {
                    primitive,
                    params: descriptor_params,
                });
                data.mesh = Some(mesh);
            }
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "recursive",
            "reparent",
            "keep_world",
            "cascade_data",
            "unlink",
        ],
        &["target"],
    )?;
    let targets = resolve_node_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?,
        true,
    )?;
    let recursive = read_bool(engine, operation, "recursive", false)?;
    let cascade_data = read_bool(engine, operation, "cascade_data", false)?;
    let unlink = read_bool(engine, operation, "unlink", false)?;
    let reparent = operation
        .get("reparent")
        .map(|_| read_string(engine, operation, "reparent"))
        .transpose()?;
    if let Some(policy) = &reparent
        && !["to_parent", "to_root"].contains(&policy.as_str())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "reparent must be `to_parent` or `to_root`",
            &operation_pointer(engine.operation_index, "reparent"),
        ));
    }
    if recursive && reparent.is_some() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "recursive and reparent policies are mutually exclusive",
            &operation_pointer(engine.operation_index, "reparent"),
        ));
    }
    if operation.contains_key("keep_world") && reparent.is_none() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "keep_world requires a reparent policy",
            &operation_pointer(engine.operation_index, "keep_world"),
        ));
    }
    let keep_world = if reparent.is_some() {
        read_bool(engine, operation, "keep_world", true)?
    } else {
        false
    };
    let mut deletion: BTreeSet<_> = targets.iter().cloned().collect();
    for target in targets {
        let mut frontier = vec![target];
        while let Some(parent_id) = frontier.pop() {
            let children: Vec<_> = engine
                .doc
                .nodes
                .iter()
                .filter(|(id, node)| {
                    node.parent.as_ref() == Some(&parent_id) && !deletion.contains(*id)
                })
                .map(|(id, _)| id.clone())
                .collect();
            if !children.is_empty() && !recursive && reparent.is_none() {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "node has children; set recursive:true or a reparent policy",
                    &operation_pointer(engine.operation_index, "target"),
                ));
            }
            if recursive {
                for child in children {
                    if deletion.insert(child.clone()) {
                        frontier.push(child);
                    }
                }
            }
        }
    }
    if let Some(policy) = reparent.as_deref() {
        let children_to_reparent: Vec<_> = engine
            .doc
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                let parent = node.parent.as_ref()?;
                (deletion.contains(parent) && !deletion.contains(id))
                    .then(|| (id.clone(), parent.clone()))
            })
            .collect();
        for (child_id, old_parent) in children_to_reparent {
            let new_parent = if policy == "to_root" {
                None
            } else {
                let mut ancestor = Some(old_parent);
                while let Some(parent_id) = ancestor.clone() {
                    if !deletion.contains(&parent_id) {
                        break;
                    }
                    ancestor = engine
                        .doc
                        .nodes
                        .get(&parent_id)
                        .and_then(|node| node.parent.clone());
                }
                ancestor
            };
            let parent_operation = json!({
                "op":"node.parent",
                "target":{"id":child_id},
                "parent":new_parent,
                "keep_world":keep_world,
            });
            let parent_operation = parent_operation.as_object().ok_or_else(|| {
                engine.error(
                    ErrorCode::InternalError,
                    "could not construct reparent operation",
                    &operation_pointer(engine.operation_index, "reparent"),
                )
            })?;
            parent(engine, parent_operation)?;
        }
    }
    let scene_users: Vec<_> = engine
        .doc
        .scenes
        .iter()
        .filter(|(_, scene)| {
            scene
                .camera
                .as_ref()
                .is_some_and(|camera| deletion.contains(camera))
        })
        .map(|(scene_id, _)| scene_id.clone())
        .collect();
    if !scene_users.is_empty() && !unlink {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "node is used as a scene camera; set unlink:true to clear the camera reference",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let affected_data: BTreeSet<_> = deletion
        .iter()
        .filter_map(|node_id| {
            engine
                .doc
                .nodes
                .get(node_id)
                .and_then(|node| node.data.clone())
        })
        .collect();
    let mut changed_collections = BTreeSet::new();
    for node_id in &deletion {
        for (collection_id, collection) in &mut engine.doc.collections {
            if collection.objects.contains(node_id) {
                collection.objects.retain(|object| object != node_id);
                changed_collections.insert(collection_id.clone());
            }
        }
        if engine.doc.nodes.remove(node_id).is_some() {
            engine.mark("nodes", node_id, ChangeKind::Deleted);
        }
    }
    for collection_id in changed_collections {
        engine.mark("collections", &collection_id, ChangeKind::Updated);
    }
    for scene_id in scene_users {
        if let Some(scene) = engine.doc.scenes.get_mut(&scene_id) {
            scene.camera = None;
        }
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    if cascade_data {
        for data_id in affected_data {
            let still_used = engine
                .doc
                .nodes
                .values()
                .any(|node| node.data.as_ref() == Some(&data_id));
            if !still_used && engine.doc.data_blocks.remove(&data_id).is_some() {
                engine.mark("data_blocks", &data_id, ChangeKind::Deleted);
            }
        }
    }
    Ok(!deletion.is_empty())
}

fn duplicate(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "mode", "recursive"],
        &["target", "id"],
    )?;
    let source_id = resolve_node_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?,
        false,
    )?
    .remove(0);
    let new_id = read_id(engine, operation, "id")?;
    if engine.doc.nodes.contains_key(&new_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("node ID `{new_id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mode = if operation.contains_key("mode") {
        read_string(engine, operation, "mode")?
    } else {
        "independent".to_owned()
    };
    if !["independent", "linked"].contains(&mode.as_str()) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mode must be `independent` or `linked`",
            &operation_pointer(engine.operation_index, "mode"),
        ));
    }
    let recursive = read_bool(engine, operation, "recursive", false)?;
    let mut source_ids = vec![source_id.clone()];
    let mut seen_sources = BTreeSet::from([source_id.clone()]);
    if recursive {
        let mut cursor = 0;
        while cursor < source_ids.len() {
            let parent = source_ids[cursor].clone();
            let children: Vec<_> = engine
                .doc
                .nodes
                .iter()
                .filter(|(id, node)| {
                    node.parent.as_ref() == Some(&parent) && !seen_sources.contains(*id)
                })
                .map(|(id, _)| id.clone())
                .collect();
            for child in children {
                seen_sources.insert(child.clone());
                source_ids.push(child);
            }
            cursor += 1;
        }
    }
    let mut mapping = BTreeMap::new();
    mapping.insert(source_id.clone(), new_id.clone());
    for source in source_ids.iter().skip(1) {
        let generated = format!("{new_id}_{source}");
        mapping.insert(
            source.clone(),
            parse_id(
                engine,
                &generated,
                &operation_pointer(engine.operation_index, "id"),
            )?,
        );
    }
    for target in mapping.values() {
        if engine.doc.nodes.contains_key(target) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("node ID `{target}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
    }
    let mut data_mapping = BTreeMap::new();
    for source in &source_ids {
        let old = engine
            .doc
            .nodes
            .get(source)
            .and_then(|node| node.data.clone());
        if mode == "independent"
            && let Some(data_id) = old
            && !data_mapping.contains_key(&data_id)
        {
            let new_node_id = mapping.get(source).ok_or_else(|| {
                engine.error(
                    ErrorCode::InternalError,
                    "duplicate mapping is missing",
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?;
            let new_data_id = generated_data_id(engine, new_node_id, "_mesh")?;
            if engine.doc.data_blocks.contains_key(&new_data_id) {
                return Err(engine.error(
                    ErrorCode::IdExists,
                    format!("data-block ID `{new_data_id}` already exists"),
                    &operation_pointer(engine.operation_index, "id"),
                ));
            }
            let data = engine
                .doc
                .data_blocks
                .get(&data_id)
                .cloned()
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` was not found"),
                        &operation_pointer(engine.operation_index, "target"),
                    )
                })?;
            engine.doc.data_blocks.insert(new_data_id.clone(), data);
            engine.mark("data_blocks", &new_data_id, ChangeKind::Created);
            data_mapping.insert(data_id, new_data_id);
        }
    }
    let mut changed_collections = BTreeSet::new();
    for source in &source_ids {
        let mut copy = engine.doc.nodes.get(source).cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{source}` was not found"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        let new_node_id = mapping.get(source).cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "duplicate mapping is missing",
                &operation_pointer(engine.operation_index, "id"),
            )
        })?;
        copy.name.push_str(" Copy");
        copy.parent = copy
            .parent
            .as_ref()
            .and_then(|parent| mapping.get(parent).cloned())
            .or(copy.parent.clone());
        if let Some(data_id) = copy.data.clone()
            && mode == "independent"
        {
            let new_data_id = data_mapping.get(&data_id).cloned().ok_or_else(|| {
                engine.error(
                    ErrorCode::InternalError,
                    "data duplicate mapping is missing",
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?;
            copy.data = Some(new_data_id.clone());
            engine.map_id("data_blocks", &data_id, &new_data_id);
        }
        engine.doc.nodes.insert(new_node_id.clone(), copy);
        engine.mark("nodes", &new_node_id, ChangeKind::Created);
        engine.map_id("nodes", source, &new_node_id);
        for (collection_id, collection) in &mut engine.doc.collections {
            if collection.objects.contains(source) && !collection.objects.contains(&new_node_id) {
                collection.objects.push(new_node_id.clone());
                collection.objects.sort();
                changed_collections.insert(collection_id.clone());
            }
        }
    }
    for collection_id in changed_collections {
        engine.mark("collections", &collection_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn parent(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "parent",
            "parent_type",
            "parent_bone",
            "keep_world",
        ],
        &["target", "parent"],
    )?;
    let target = resolve_node_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?,
        false,
    )?
    .remove(0);
    let parent = if operation.get("parent").is_some_and(Value::is_null) {
        None
    } else {
        Some(read_id(engine, operation, "parent")?)
    };
    if parent
        .as_ref()
        .is_some_and(|id| !engine.doc.nodes.contains_key(id))
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "parent node was not found",
            &operation_pointer(engine.operation_index, "parent"),
        ));
    }
    let parent_type = if let Some(value) = operation.get("parent_type") {
        match value.as_str() {
            Some("object") => ParentType::Object,
            Some("bone") => ParentType::Bone,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "parent_type must be object or bone",
                    &operation_pointer(engine.operation_index, "parent_type"),
                ));
            }
        }
    } else {
        ParentType::Object
    };
    let parent_bone = match operation.get("parent_bone") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "parent_bone must be a bone ID or null",
                    &operation_pointer(engine.operation_index, "parent_bone"),
                )
            })?;
            Some(parse_id(
                engine,
                text,
                &operation_pointer(engine.operation_index, "parent_bone"),
            )?)
        }
    };
    match (parent_type, parent.as_ref(), parent_bone.as_ref()) {
        (ParentType::Object, _, Some(_)) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "parent_bone is only valid for bone parenting",
                &operation_pointer(engine.operation_index, "parent_bone"),
            ));
        }
        (ParentType::Bone, None, _) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "bone parenting requires a parent Object",
                &operation_pointer(engine.operation_index, "parent"),
            ));
        }
        (ParentType::Bone, Some(parent_id), Some(bone_id)) => {
            let parent_node = engine.doc.nodes.get(parent_id).ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    "armature parent was not found",
                    &operation_pointer(engine.operation_index, "parent"),
                )
            })?;
            let armature_id = parent_node.data.as_ref().ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "armature parent has no Data-Block",
                    &operation_pointer(engine.operation_index, "parent"),
                )
            })?;
            if parent_node.kind != "armature"
                || engine
                    .doc
                    .data_blocks
                    .get(armature_id)
                    .and_then(|data| data.armature.as_ref())
                    .is_none_or(|armature| !armature.bones.contains_key(bone_id))
            {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    "parent bone was not found",
                    &operation_pointer(engine.operation_index, "parent_bone"),
                ));
            }
        }
        (ParentType::Bone, Some(_), None) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "bone parenting requires parent_bone",
                &operation_pointer(engine.operation_index, "parent_bone"),
            ));
        }
        (ParentType::Object, _, None) => {}
    }
    if parent.as_ref() == Some(&target)
        || parent
            .as_ref()
            .is_some_and(|parent| is_descendant(engine, parent, &target))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "parenting would create a node hierarchy cycle",
            &operation_pointer(engine.operation_index, "parent"),
        ));
    }
    let keep_world = read_bool(engine, operation, "keep_world", true)?;
    let current_node = engine.doc.nodes.get(&target).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if current_node.parent == parent
        && current_node.parent_type == parent_type
        && current_node.parent_bone == parent_bone
    {
        return Ok(false);
    }
    let mut root_transform = None;
    let parent_inverse = if keep_world {
        let frame = engine
            .doc
            .scenes
            .get(&engine.doc.active_scene)
            .map_or(1.0, |scene| scene.frame_current);
        let driver_values = crate::eval::rig::evaluate_drivers(engine.doc, frame)?;
        let matrices = crate::eval::rig::evaluate_world_matrices_with_drivers(
            engine.doc,
            frame,
            &driver_values,
        )?;
        let old_world = matrices.get(&target).copied().ok_or_else(|| {
            crate::error::PotError::new(ErrorCode::InternalError, "current world matrix is missing")
        })?;
        let parent_space = if let Some(parent_id) = &parent {
            let parent_world = matrices.get(parent_id).copied().ok_or_else(|| {
                crate::error::PotError::new(
                    ErrorCode::InternalError,
                    "parent world matrix is missing",
                )
            })?;
            if parent_type == ParentType::Bone {
                let bone_id = parent_bone.as_ref().ok_or_else(|| {
                    crate::error::PotError::new(ErrorCode::InternalError, "parent bone is missing")
                })?;
                let bones = crate::eval::rig::evaluate_armature_bone_world_matrices_with_drivers(
                    engine.doc,
                    parent_id,
                    frame,
                    parent_world,
                    &matrices,
                    &driver_values,
                )?;
                bones.get(bone_id).copied().ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        "parent bone was not found",
                        &operation_pointer(engine.operation_index, "parent_bone"),
                    )
                })?
            } else {
                parent_world
            }
        } else {
            DMat4::IDENTITY
        };
        let determinant = parent_space.determinant();
        if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "SINGULAR_TRANSFORM: cannot preserve world transform under a singular parent",
                &operation_pointer(engine.operation_index, "parent"),
            ));
        }
        let animated_local =
            crate::eval::animation::animated_transform(current_node, engine.doc, frame)?.matrix();
        let inverse = parent_space.inverse() * old_world * animated_local.inverse();
        if parent.is_none()
            && current_node.action.is_none()
            && current_node.nla_tracks.is_empty()
            && current_node.drivers.is_empty()
            && current_node.constraints.is_empty()
        {
            if let Some(transform) =
                transform_from_matrix(old_world, &current_node.transform.rotation_mode)
            {
                root_transform = Some(transform);
                None
            } else {
                Some(inverse.to_cols_array())
            }
        } else {
            Some(inverse.to_cols_array())
        }
    } else {
        None
    };
    if let Some(node) = engine.doc.nodes.get_mut(&target) {
        node.parent = parent;
        node.parent_inverse = parent_inverse;
        node.parent_type = parent_type;
        node.parent_bone = parent_bone;
        if let Some(transform) = root_transform {
            node.transform = transform;
        }
    }
    engine.mark("nodes", &target, ChangeKind::Updated);
    Ok(true)
}
fn transform_from_matrix(matrix: DMat4, rotation_mode: &str) -> Option<Transform> {
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    if !scale.is_finite()
        || !rotation.is_finite()
        || rotation.length_squared() <= f64::EPSILON
        || !translation.is_finite()
    {
        return None;
    }
    let rotation = rotation.normalize();
    let reconstructed = DMat4::from_scale_rotation_translation(scale, rotation, translation);
    let original = matrix.to_cols_array();
    let decomposed = reconstructed.to_cols_array();
    if original.into_iter().zip(decomposed).any(|(left, right)| {
        !left.is_finite()
            || !right.is_finite()
            || (left - right).abs() > 1.0e-10 * left.abs().max(right.abs()).max(1.0)
    }) {
        return None;
    }
    Some(Transform {
        translation: translation.to_array(),
        rotation: canonicalize_quaternion([rotation.x, rotation.y, rotation.z, rotation.w]),
        scale: scale.to_array(),
        rotation_mode: rotation_mode.to_owned(),
    })
}

fn is_descendant(engine: &Engine<'_>, possible_descendant: &Id, ancestor: &Id) -> bool {
    let mut current = Some(possible_descendant.clone());
    let mut visited = BTreeSet::new();
    while let Some(id) = current {
        if id == *ancestor {
            return true;
        }
        if !visited.insert(id.clone()) {
            return true;
        }
        current = engine
            .doc
            .nodes
            .get(&id)
            .and_then(|node| node.parent.clone());
    }
    false
}

fn generated_data_id(engine: &Engine<'_>, node_id: &Id, suffix: &str) -> Result<Id> {
    let generated = format!("{node_id}{suffix}");
    parse_id(
        engine,
        &generated,
        &operation_pointer(engine.operation_index, "id"),
    )
}

fn read_material_slots(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Vec<Id>> {
    if operation.contains_key("material") && operation.contains_key("materials") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "material and materials cannot both be specified",
            &operation_pointer(engine.operation_index, "materials"),
        ));
    }
    if let Some(value) = operation.get("material") {
        let text = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "material must be a material ID",
                &operation_pointer(engine.operation_index, "material"),
            )
        })?;
        let id = parse_id(
            engine,
            text,
            &operation_pointer(engine.operation_index, "material"),
        )?;
        if !engine.doc.materials.contains_key(&id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("material `{id}` was not found"),
                &operation_pointer(engine.operation_index, "material"),
            ));
        }
        return Ok(vec![id]);
    }
    if let Some(value) = operation.get("materials") {
        return read_material_array(engine, value, "materials");
    }
    Ok(Vec::new())
}

fn read_material_array(engine: &Engine<'_>, value: &Value, field: &str) -> Result<Vec<Id>> {
    let Some(values) = value.as_array() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "materials must be an array of material IDs",
            &operation_pointer(engine.operation_index, field),
        ));
    };
    let mut ids = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Some(text) = value.as_str() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "material IDs must be strings",
                &operation_pointer(engine.operation_index, &format!("{field}/{index}")),
            ));
        };
        let id = parse_id(
            engine,
            text,
            &operation_pointer(engine.operation_index, &format!("{field}/{index}")),
        )?;
        if !engine.doc.materials.contains_key(&id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("material `{id}` was not found"),
                &operation_pointer(engine.operation_index, &format!("{field}/{index}")),
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

fn read_matrix(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 16]> {
    let Some(values) = value.as_array() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "parent_inverse must be a 16-number array",
            pointer,
        ));
    };
    if values.len() != 16 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "parent_inverse must be a 16-number array",
            pointer,
        ));
    }
    let mut matrix = [0.0; 16];
    for (index, value) in values.iter().enumerate() {
        let Some(number) = value.as_f64() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "parent_inverse components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        };
        if !number.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "parent_inverse components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        }
        matrix[index] = number;
    }
    Ok(matrix)
}

fn parse_transform_update(
    engine: &Engine<'_>,
    value: &Value,
    field: &str,
) -> Result<TransformUpdate> {
    let pointer = operation_pointer(engine.operation_index, field);
    let Some(object) = value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "transform must be an object",
            &pointer,
        ));
    };
    for key in object.keys() {
        if ![
            "translation",
            "rotation",
            "rotation_deg",
            "scale",
            "rotation_mode",
        ]
        .contains(&key.as_str())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown transform field `{key}`"),
                &format!("{pointer}/{}", key.replace('~', "~0").replace('/', "~1")),
            ));
        }
    }
    if object.contains_key("rotation") && object.contains_key("rotation_deg") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rotation and rotation_deg are mutually exclusive",
            &format!("{pointer}/rotation_deg"),
        ));
    }
    let translation = object
        .get("translation")
        .map(|value| read_vec3(engine, value, &format!("{pointer}/translation")))
        .transpose()?;
    let scale = object
        .get("scale")
        .map(|value| read_vec3(engine, value, &format!("{pointer}/scale")))
        .transpose()?;
    let (rotation, sets_quaternion_mode) = if let Some(value) = object.get("rotation") {
        let rotation = read_vec4(engine, value, &format!("{pointer}/rotation"))?;
        let rotation = normalize_rotation(rotation).map_err(|_| {
            engine.error(
                ErrorCode::InvalidOperation,
                "rotation quaternion must be finite and non-zero",
                &format!("{pointer}/rotation"),
            )
        })?;
        (Some(rotation), false)
    } else if let Some(value) = object.get("rotation_deg") {
        let degrees = read_vec3(engine, value, &format!("{pointer}/rotation_deg"))?;
        let x = DQuat::from_rotation_x(degrees[0].to_radians());
        let y = DQuat::from_rotation_y(degrees[1].to_radians());
        let z = DQuat::from_rotation_z(degrees[2].to_radians());
        let q = z * y * x;
        (Some(canonicalize_quaternion([q.x, q.y, q.z, q.w])), true)
    } else {
        (None, false)
    };
    let rotation_mode = if let Some(value) = object.get("rotation_mode") {
        let Some(mode) = value.as_str() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "rotation_mode must be a string",
                &format!("{pointer}/rotation_mode"),
            ));
        };
        if ![
            "quaternion",
            "XYZ",
            "XZY",
            "YXZ",
            "YZX",
            "ZXY",
            "ZYX",
            "AXIS_ANGLE",
        ]
        .contains(&mode)
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported rotation_mode `{mode}`"),
                &format!("{pointer}/rotation_mode"),
            ));
        }
        Some(mode.to_owned())
    } else {
        None
    };
    Ok(TransformUpdate {
        translation,
        scale,
        rotation,
        sets_quaternion_mode,
        rotation_mode,
    })
}

fn apply_transform_update(transform: &mut Transform, update: &TransformUpdate) -> bool {
    let mut changed = false;
    if let Some(translation) = update.translation
        && !crate::float::equal_f64_array(&transform.translation, &translation)
    {
        transform.translation = translation;
        changed = true;
    }
    if let Some(scale) = update.scale
        && !crate::float::equal_f64_array(&transform.scale, &scale)
    {
        transform.scale = scale;
        changed = true;
    }
    if let Some(rotation) = update.rotation
        && !crate::float::equal_f64_array(&transform.rotation, &rotation)
    {
        transform.rotation = rotation;
        changed = true;
    }
    if update.sets_quaternion_mode && transform.rotation_mode != "quaternion" {
        transform.rotation_mode = String::from("quaternion");
        changed = true;
    }
    if let Some(rotation_mode) = &update.rotation_mode
        && transform.rotation_mode != *rotation_mode
    {
        transform.rotation_mode.clone_from(rotation_mode);
        changed = true;
    }
    changed
}

fn read_vec3(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 3]> {
    let Some(values) = value.as_array() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "value must be a three-number array",
            pointer,
        ));
    };
    if values.len() != 3 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "value must be a three-number array",
            pointer,
        ));
    }
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        let Some(number) = value.as_f64() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        };
        if !number.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        }
        result[index] = number;
    }
    Ok(result)
}

fn read_vec4(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 4]> {
    let Some(values) = value.as_array() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rotation must be a four-number array",
            pointer,
        ));
    };
    if values.len() != 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rotation must be a four-number array",
            pointer,
        ));
    }
    let mut result = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        let Some(number) = value.as_f64() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "rotation components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        };
        if !number.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "rotation components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        }
        result[index] = number;
    }
    Ok(result)
}

fn build_primitive(
    engine: &Engine<'_>,
    primitive: &str,
    params: Value,
    path: &str,
) -> Result<(GeomMesh, Map<String, Value>)> {
    let Some(parameter_map) = params.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "params must be an object",
            &operation_pointer(engine.operation_index, path),
        ));
    };
    validate_primitive_params(engine, primitive, parameter_map, path)?;
    let pointer = operation_pointer(engine.operation_index, path);
    let mesh = geom::primitive(primitive, &params).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid `{primitive}` params: {error}"),
            &pointer,
        )
    })?;
    let Value::Object(descriptor_params) = params else {
        return Err(engine.error(
            ErrorCode::InternalError,
            "primitive params are not an object",
            &pointer,
        ));
    };
    Ok((mesh, descriptor_params))
}

fn validate_primitive_params(
    engine: &Engine<'_>,
    primitive: &str,
    params: &Map<String, Value>,
    base: &str,
) -> Result<()> {
    let fields: &[&str] = match primitive {
        "box" | "plane" => &["size"],
        "uv_sphere" => &["segments", "ring_count", "radius"],
        "cylinder" => &["vertices", "radius", "depth", "end_fill_type"],
        "cone" => &["vertices", "radius1", "radius2", "depth", "end_fill_type"],
        "torus" => &[
            "major_radius",
            "minor_radius",
            "abso_major_rad",
            "abso_minor_rad",
            "major_segments",
            "minor_segments",
            "mode",
        ],
        "icosphere" => &["subdivisions", "radius"],
        "circle" => &["vertices", "radius", "fill_type"],
        "grid" => &["size", "x_subdivisions", "y_subdivisions"],
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "unknown primitive kind",
                &operation_pointer(engine.operation_index, base),
            ));
        }
    };
    for field in params.keys() {
        if !fields.contains(&field.as_str()) {
            let suffix = format!("{base}/{}", pointer_escape(field));
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown primitive parameter `{field}`"),
                &operation_pointer(engine.operation_index, &suffix),
            ));
        }
    }
    match primitive {
        "box" | "plane" => {
            validate_scalar_param(engine, params, "size", 2.0, false, base)?;
        }
        "uv_sphere" => {
            validate_integer_param(engine, params, "segments", 3, 100_000, base)?;
            validate_integer_param(engine, params, "ring_count", 3, 100_000, base)?;
            validate_scalar_param(engine, params, "radius", 1.0, false, base)?;
        }
        "cylinder" => {
            validate_integer_param(engine, params, "vertices", 3, 10_000_000, base)?;
            validate_scalar_param(engine, params, "radius", 1.0, false, base)?;
            validate_scalar_param(engine, params, "depth", 2.0, false, base)?;
            validate_end_fill_type(engine, params, base)?;
        }
        "cone" => {
            validate_integer_param(engine, params, "vertices", 3, 10_000_000, base)?;
            let radius1 = validate_scalar_param(engine, params, "radius1", 1.0, true, base)?;
            let radius2 = validate_scalar_param(engine, params, "radius2", 0.0, true, base)?;
            if radius1 == 0.0 && radius2 == 0.0 {
                return Err(primitive_parameter_error(
                    engine,
                    base,
                    "radius1",
                    "at least one cone radius must be positive",
                ));
            }
            validate_scalar_param(engine, params, "depth", 2.0, false, base)?;
            validate_end_fill_type(engine, params, base)?;
        }
        "torus" => {
            let mode = params
                .get("mode")
                .and_then(Value::as_str)
                .unwrap_or("MAJOR_MINOR");
            if !matches!(mode, "MAJOR_MINOR" | "EXT_INT") {
                return Err(primitive_parameter_error(
                    engine,
                    base,
                    "mode",
                    "mode must be `MAJOR_MINOR` or `EXT_INT`",
                ));
            }
            validate_scalar_param(engine, params, "major_radius", 1.0, false, base)?;
            validate_scalar_param(engine, params, "minor_radius", 0.25, false, base)?;
            let exterior =
                validate_scalar_param(engine, params, "abso_major_rad", 1.25, false, base)?;
            let interior =
                validate_scalar_param(engine, params, "abso_minor_rad", 0.75, true, base)?;
            if mode == "EXT_INT" && interior >= exterior {
                return Err(primitive_parameter_error(
                    engine,
                    base,
                    "abso_minor_rad",
                    "interior radius must be less than exterior radius in EXT_INT mode",
                ));
            }
            validate_integer_param(engine, params, "major_segments", 3, 256, base)?;
            validate_integer_param(engine, params, "minor_segments", 3, 256, base)?;
        }
        "icosphere" => {
            validate_integer_param(engine, params, "subdivisions", 1, 10, base)?;
            validate_scalar_param(engine, params, "radius", 1.0, false, base)?;
        }
        "circle" => {
            validate_integer_param(engine, params, "vertices", 3, 10_000_000, base)?;
            validate_scalar_param(engine, params, "radius", 1.0, false, base)?;
            if let Some(fill_type) = params.get("fill_type")
                && !matches!(fill_type.as_str(), Some("NOTHING" | "NGON" | "TRIFAN"))
            {
                return Err(primitive_parameter_error(
                    engine,
                    base,
                    "fill_type",
                    "fill_type must be `NOTHING`, `NGON`, or `TRIFAN`",
                ));
            }
        }
        "grid" => {
            validate_scalar_param(engine, params, "size", 2.0, false, base)?;
            validate_integer_param(engine, params, "x_subdivisions", 1, 10_000_000, base)?;
            validate_integer_param(engine, params, "y_subdivisions", 1, 10_000_000, base)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_end_fill_type(
    engine: &Engine<'_>,
    params: &Map<String, Value>,
    base: &str,
) -> Result<()> {
    if let Some(fill_type) = params.get("end_fill_type")
        && !matches!(fill_type.as_str(), Some("NOTHING" | "NGON" | "TRIFAN"))
    {
        return Err(primitive_parameter_error(
            engine,
            base,
            "end_fill_type",
            "end_fill_type must be `NOTHING`, `NGON`, or `TRIFAN`",
        ));
    }
    Ok(())
}

fn validate_scalar_param(
    engine: &Engine<'_>,
    params: &Map<String, Value>,
    field: &str,
    default: f64,
    allow_zero: bool,
    base: &str,
) -> Result<f64> {
    let Some(value) = params.get(field) else {
        return Ok(default);
    };
    let number = value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            primitive_parameter_error(engine, base, field, "parameter must be a finite number")
        })?;
    if if allow_zero {
        number < 0.0
    } else {
        number <= 0.0
    } {
        return Err(primitive_parameter_error(
            engine,
            base,
            field,
            "parameter is outside its valid range",
        ));
    }
    Ok(number)
}

fn validate_integer_param(
    engine: &Engine<'_>,
    params: &Map<String, Value>,
    field: &str,
    minimum: u64,
    maximum: u64,
    base: &str,
) -> Result<()> {
    let Some(value) = params.get(field) else {
        return Ok(());
    };
    let Some(number) = value.as_u64() else {
        return Err(primitive_parameter_error(
            engine,
            base,
            field,
            "parameter must be an integer within its valid range",
        ));
    };
    if number < minimum || number > maximum {
        return Err(primitive_parameter_error(
            engine,
            base,
            field,
            "parameter must be an integer within its valid range",
        ));
    }
    Ok(())
}

fn primitive_parameter_error(
    engine: &Engine<'_>,
    base: &str,
    field: &str,
    message: &str,
) -> crate::error::PotError {
    let suffix = format!("{base}/{}", pointer_escape(field));
    engine.error(
        ErrorCode::InvalidOperation,
        message,
        &operation_pointer(engine.operation_index, &suffix),
    )
}
