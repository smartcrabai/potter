use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{RenderSettings, Scene, UnitSettings, ViewLayer},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape, read_id,
    read_string,
};
const SCENE_UPDATE_FIELDS: &[&str] = &[
    "name",
    "frame_current",
    "frame_start",
    "frame_end",
    "fps",
    "fps_base",
    "camera",
    "world",
    "active_clip",
    "unit",
];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "scene.create" => create(engine, operation),
        "scene.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid scene operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "root_collection",
            "view_layer",
            "active",
        ],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.scenes.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("scene ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let root_collection = if operation.contains_key("root_collection") {
        let collection_id = read_id(engine, operation, "root_collection")?;
        if !engine.doc.collections.contains_key(&collection_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("root collection `{collection_id}` was not found"),
                &operation_pointer(engine.operation_index, "root_collection"),
            ));
        }
        collection_id
    } else {
        let value = format!("{id}_root");
        let collection_id = parse_id(
            engine,
            &value,
            &operation_pointer(engine.operation_index, "id"),
        )?;
        if engine.doc.collections.contains_key(&collection_id) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("root collection ID `{collection_id}` already exists"),
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        engine.doc.collections.insert(
            collection_id.clone(),
            crate::model::Collection {
                name: format!(
                    "{} Collection",
                    operation
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or(id.as_str())
                ),
                children: Vec::new(),
                objects: Vec::new(),
            },
        );
        engine.mark("collections", &collection_id, ChangeKind::Created);
        collection_id
    };
    let view_layer_id = if operation.contains_key("view_layer") {
        read_id(engine, operation, "view_layer")?
    } else {
        parse_id(
            engine,
            "view_main",
            &operation_pointer(engine.operation_index, "id"),
        )?
    };
    let name = if operation.contains_key("name") {
        read_string(engine, operation, "name")?
    } else {
        id.to_string()
    };
    let scene = Scene {
        name,
        root_collection,
        view_layers: std::collections::BTreeMap::from([(
            view_layer_id,
            ViewLayer {
                name: "View Layer".to_owned(),
                excluded_collections: Vec::new(),
            },
        )]),
        frame_current: 1.0,
        frame_start: 1,
        frame_end: 250,
        fps: 24,
        fps_base: 1.0,
        camera: None,
        world: None,
        active_clip: None,
        unit: UnitSettings::default(),
        render: RenderSettings::default(),
        use_compositing: false,
        compositor: None,
        color_management: crate::color::ColorManagement::default(),
        markers: Vec::new(),
        rigid_body_world: None,
        sequencer: crate::sequencer::Sequencer::default(),
    };
    engine.doc.scenes.insert(id.clone(), scene);
    engine.mark("scenes", &id, ChangeKind::Created);
    if operation
        .get("active")
        .is_some_and(|value| value.as_bool() == Some(true))
    {
        engine.doc.active_scene = id;
    }
    if operation.contains_key("active")
        && operation.get("active").and_then(Value::as_bool).is_none()
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "active must be a boolean",
            &operation_pointer(engine.operation_index, "active"),
        ));
    }
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let target_value = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let target_object = target_value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    for key in target_object.keys() {
        if key != "id" {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown scene target field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("target/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let scene_text = target_object
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "scene target requires an ID",
                &operation_pointer(engine.operation_index, "target/id"),
            )
        })?;
    let scene_id = parse_id(
        engine,
        scene_text,
        &operation_pointer(engine.operation_index, "target/id"),
    )?;
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
    for key in set.keys() {
        if !SCENE_UPDATE_FIELDS.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown scene set field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let current_scene = engine.doc.scenes.get(&scene_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("scene `{scene_id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let name = set
        .get("name")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    &operation_pointer(engine.operation_index, "set/name"),
                )
            })
        })
        .transpose()?;
    let frame_current = set
        .get("frame_current")
        .map(|value| {
            super::finite_number(
                engine,
                Some(value),
                "set/frame_current",
                super::FiniteNumberMessage::FiniteField,
            )
        })
        .transpose()?;
    let frame_start = set
        .get("frame_start")
        .map(|value| read_i32(engine, value, "set/frame_start"))
        .transpose()?;
    let frame_end = set
        .get("frame_end")
        .map(|value| read_i32(engine, value, "set/frame_end"))
        .transpose()?;
    let fps = set
        .get("fps")
        .map(|value| read_fps(engine, value))
        .transpose()?;
    let fps_base = set
        .get("fps_base")
        .map(|value| {
            let number = super::finite_number(
                engine,
                Some(value),
                "set/fps_base",
                super::FiniteNumberMessage::FiniteField,
            )?;
            if number <= 0.0 {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "fps_base must be positive",
                    &operation_pointer(engine.operation_index, "set/fps_base"),
                ));
            }
            Ok(number)
        })
        .transpose()?;
    let read_optional_id = |field: &str| -> Result<Option<Option<crate::model::Id>>> {
        set.get(field)
            .map(|value| {
                if value.is_null() {
                    Ok(None)
                } else {
                    let text = value.as_str().ok_or_else(|| {
                        engine.error(
                            ErrorCode::InvalidOperation,
                            format!("{field} must be an ID or null"),
                            &operation_pointer(engine.operation_index, &format!("set/{field}")),
                        )
                    })?;
                    parse_id(
                        engine,
                        text,
                        &operation_pointer(engine.operation_index, &format!("set/{field}")),
                    )
                    .map(Some)
                }
            })
            .transpose()
    };
    let camera = read_optional_id("camera")?;
    let world = read_optional_id("world")?;
    let active_clip = read_optional_id("active_clip")?;
    let root_collection = set
        .get("root_collection")
        .map(|value| {
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "root_collection must be an ID",
                    &operation_pointer(engine.operation_index, "set/root_collection"),
                )
            })?;
            parse_id(
                engine,
                text,
                &operation_pointer(engine.operation_index, "set/root_collection"),
            )
        })
        .transpose()?;
    if let Some(Some(camera_id)) = &camera
        && engine.doc.nodes.get(camera_id).is_none_or(|node| {
            node.kind != "camera"
                || node
                    .data
                    .as_ref()
                    .and_then(|data_id| engine.doc.data_blocks.get(data_id))
                    .and_then(|data| data.camera.as_ref())
                    .is_none()
        })
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("camera node `{camera_id}` was not found or has no Camera-Data"),
            &operation_pointer(engine.operation_index, "set/camera"),
        ));
    }
    if let Some(Some(world_id)) = &world
        && !engine.doc.worlds.contains_key(world_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("world `{world_id}` was not found"),
            &operation_pointer(engine.operation_index, "set/world"),
        ));
    }
    if let Some(Some(clip_id)) = &active_clip
        && !engine.doc.movie_clips.contains_key(clip_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("movie clip `{clip_id}` was not found"),
            &operation_pointer(engine.operation_index, "set/active_clip"),
        ));
    }
    if let Some(collection_id) = &root_collection
        && !engine.doc.collections.contains_key(collection_id)
    {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("root collection `{collection_id}` was not found"),
            &operation_pointer(engine.operation_index, "set/root_collection"),
        ));
    }
    let unit = set
        .get("unit")
        .map(|value| parse_unit_update(engine, value))
        .transpose()?;
    let next_start = frame_start.unwrap_or(current_scene.frame_start);
    let next_end = frame_end.unwrap_or(current_scene.frame_end);
    if next_start > next_end {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "frame_start must not exceed frame_end",
            &operation_pointer(engine.operation_index, "set/frame_start"),
        ));
    }
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
        crate::error::PotError::new(
            ErrorCode::InternalError,
            "scene disappeared while applying update",
        )
    })?;
    let mut changed = false;
    if let Some(name) = &name
        && scene.name != *name
    {
        scene.name.clone_from(name);
        changed = true;
    }
    if let Some(value) = frame_current
        && !crate::float::equal_f64(scene.frame_current, value)
    {
        scene.frame_current = value;
        changed = true;
    }
    if let Some(value) = frame_start
        && scene.frame_start != value
    {
        scene.frame_start = value;
        changed = true;
    }
    if let Some(value) = frame_end
        && scene.frame_end != value
    {
        scene.frame_end = value;
        changed = true;
    }
    if let Some(value) = fps
        && scene.fps != value
    {
        scene.fps = value;
        changed = true;
    }
    if let Some(value) = fps_base
        && !crate::float::equal_f64(scene.fps_base, value)
    {
        scene.fps_base = value;
        changed = true;
    }
    if let Some(value) = camera
        && scene.camera != value
    {
        scene.camera = value;
        changed = true;
    }
    if let Some(value) = world
        && scene.world != value
    {
        scene.world = value;
        changed = true;
    }
    if let Some(value) = active_clip
        && scene.active_clip != value
    {
        scene.active_clip = value;
        changed = true;
    }
    if let Some(value) = root_collection
        && scene.root_collection != value
    {
        scene.root_collection = value;
        changed = true;
    }
    if let Some(unit) = unit {
        if let Some(system) = unit.system
            && scene.unit.system != system
        {
            scene.unit.system.clone_from(&system);
            changed = true;
        }
        if let Some(scale_length) = unit.scale_length
            && !crate::float::equal_f64(scene.unit.scale_length, scale_length)
        {
            scene.unit.scale_length = scale_length;
            changed = true;
        }
    }
    if changed {
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

struct UnitUpdate {
    system: Option<String>,
    scale_length: Option<f64>,
}

fn parse_unit_update(engine: &Engine<'_>, value: &Value) -> Result<UnitUpdate> {
    let Some(object) = value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "unit must be an object",
            &operation_pointer(engine.operation_index, "set/unit"),
        ));
    };
    for key in object.keys() {
        if !["system", "scale_length"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown unit field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/unit/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let system = object
        .get("system")
        .map(|value| {
            let system = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "unit system must be a string",
                    &operation_pointer(engine.operation_index, "set/unit/system"),
                )
            })?;
            if !["none", "metric", "imperial"].contains(&system) {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "unit system must be none, metric, or imperial",
                    &operation_pointer(engine.operation_index, "set/unit/system"),
                ));
            }
            Ok(system.to_owned())
        })
        .transpose()?;
    let scale_length = object
        .get("scale_length")
        .map(|value| {
            let scale_length = super::finite_number(
                engine,
                Some(value),
                "set/unit/scale_length",
                super::FiniteNumberMessage::FiniteField,
            )?;
            if scale_length <= 0.0 {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "unit scale_length must be positive",
                    &operation_pointer(engine.operation_index, "set/unit/scale_length"),
                ));
            }
            Ok(scale_length)
        })
        .transpose()?;
    Ok(UnitUpdate {
        system,
        scale_length,
    })
}

fn read_i32(engine: &Engine<'_>, value: &Value, field: &str) -> Result<i32> {
    let number = value.as_i64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be an integer"),
            &operation_pointer(engine.operation_index, field),
        )
    })?;
    i32::try_from(number).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} is outside the supported range"),
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn read_fps(engine: &Engine<'_>, value: &Value) -> Result<u32> {
    let fps = u32::try_from(value.as_u64().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "fps must be a positive integer",
            &operation_pointer(engine.operation_index, "set/fps"),
        )
    })?)
    .map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "fps must be a positive integer",
            &operation_pointer(engine.operation_index, "set/fps"),
        )
    })?;
    if fps == 0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "fps must be a positive integer",
            &operation_pointer(engine.operation_index, "set/fps"),
        ));
    }
    Ok(fps)
}
