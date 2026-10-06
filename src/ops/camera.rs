use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{CameraData, CameraProjection, Id, LightData, LightType, World},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, resolve_node_targets};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "camera.create" => create_camera(engine, operation),
        "camera.update" => update_camera(engine, operation),
        "light.create" => create_light(engine, operation),
        "light.update" => update_light(engine, operation),
        "world.create" => create_world(engine, operation),
        "world.update" => update_world(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid camera/light/world operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create_camera(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "tags",
            "collection",
            "parent",
            "parent_inverse",
            "transform",
            "visible",
            "render_visible",
            "selectable",
            "projection",
            "lens_mm",
            "sensor_width_mm",
            "ortho_scale",
            "clip_start",
            "clip_end",
            "shift",
            "panorama_type",
            "dof_enabled",
            "focus_distance",
            "f_stop",
            "aperture_blades",
            "stereo_mode",
            "interocular_distance",
        ],
        &["id"],
    )?;
    let node_operation = node_create_operation(operation, "camera");
    super::node::apply(engine, "node.create", &node_operation)?;
    let node_id = read_id(engine, operation, "id")?;
    let data_id = node_data_id(engine, &node_id)?;
    let mut camera = CameraData::default();
    let camera_values = selected_values(
        operation,
        &[
            "projection",
            "lens_mm",
            "sensor_width_mm",
            "ortho_scale",
            "clip_start",
            "clip_end",
            "shift",
            "panorama_type",
            "dof_enabled",
            "focus_distance",
            "f_stop",
            "aperture_blades",
            "stereo_mode",
            "interocular_distance",
        ],
    );
    apply_camera_values(engine, &camera_values, "", &mut camera)?;
    let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "camera data block disappeared after creation",
        )
    })?;
    data.camera = Some(camera);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn update_camera(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = set_object(engine, operation)?;
    let (_, data_id) = node_data_target(engine, operation, "camera")?;
    let mut camera = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|data| data.camera.clone())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "camera node has no camera payload",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    apply_camera_values(engine, set, "set/", &mut camera)?;
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "camera data block disappeared")
        })?;
    if data.camera.as_ref() == Some(&camera) {
        return Ok(false);
    }
    data.camera = Some(camera);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn apply_camera_values(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    prefix: &str,
    camera: &mut CameraData,
) -> Result<()> {
    check_camera_set_fields(
        engine,
        values,
        &[
            "projection",
            "lens_mm",
            "sensor_width_mm",
            "ortho_scale",
            "clip_start",
            "clip_end",
            "shift",
            "panorama_type",
            "dof_enabled",
            "focus_distance",
            "f_stop",
            "aperture_blades",
            "stereo_mode",
            "interocular_distance",
        ],
        prefix,
    )?;
    if let Some(value) = values.get("projection") {
        camera.projection = match value.as_str() {
            Some("perspective") => CameraProjection::Perspective,
            Some("orthographic") => CameraProjection::Orthographic,
            Some("panorama") => CameraProjection::Panorama,
            Some("fisheye") => CameraProjection::Fisheye,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "projection must be perspective, orthographic, panorama, or fisheye",
                    &operation_pointer(engine.operation_index, &format!("{prefix}projection")),
                ));
            }
        };
    }
    for (field, destination) in [
        ("lens_mm", &mut camera.lens_mm),
        ("sensor_width_mm", &mut camera.sensor_width_mm),
        ("ortho_scale", &mut camera.ortho_scale),
        ("clip_start", &mut camera.clip_start),
        ("clip_end", &mut camera.clip_end),
        ("focus_distance", &mut camera.focus_distance),
        ("f_stop", &mut camera.f_stop),
        ("interocular_distance", &mut camera.interocular_distance),
    ] {
        if let Some(value) = values.get(field) {
            let number = super::finite_number(
                engine,
                Some(value),
                &format!("{prefix}{field}"),
                super::FiniteNumberMessage::Field,
            )?;
            if number <= 0.0 {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{field} must be positive"),
                    &operation_pointer(engine.operation_index, &format!("{prefix}{field}")),
                ));
            }
            *destination = number;
        }
    }
    if camera.clip_end <= camera.clip_start {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "clip_end must exceed clip_start",
            &operation_pointer(engine.operation_index, &format!("{prefix}clip_end")),
        ));
    }
    if let Some(value) = values.get("shift") {
        camera.shift = read_vec2(engine, value, &format!("{prefix}shift"))?;
    }
    if let Some(value) = values.get("panorama_type") {
        camera.panorama_type = match value.as_str() {
            Some("equirectangular" | "fisheye_equidistant") => {
                value.as_str().unwrap_or_default().to_owned()
            }
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "panorama_type must be equirectangular or fisheye_equidistant",
                    &operation_pointer(engine.operation_index, &format!("{prefix}panorama_type")),
                ));
            }
        };
    }
    if let Some(value) = values.get("dof_enabled") {
        camera.dof_enabled = value.as_bool().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "dof_enabled must be a boolean",
                &operation_pointer(engine.operation_index, &format!("{prefix}dof_enabled")),
            )
        })?;
    }
    if let Some(value) = values.get("aperture_blades") {
        let blades = value.as_u64().and_then(|number| u32::try_from(number).ok());
        let Some(blades) = blades.filter(|blades| *blades == 0 || (3..=16).contains(blades)) else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "aperture_blades must be zero or between 3 and 16",
                &operation_pointer(engine.operation_index, &format!("{prefix}aperture_blades")),
            ));
        };
        camera.aperture_blades = blades;
    }
    if let Some(value) = values.get("stereo_mode") {
        camera.stereo_mode = match value.as_str() {
            Some("none" | "side_by_side" | "anaglyph") => {
                value.as_str().unwrap_or_default().to_owned()
            }
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "stereo_mode must be none, side_by_side, or anaglyph",
                    &operation_pointer(engine.operation_index, &format!("{prefix}stereo_mode")),
                ));
            }
        };
    }
    Ok(())
}

fn create_light(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "tags",
            "collection",
            "parent",
            "parent_inverse",
            "transform",
            "visible",
            "render_visible",
            "selectable",
            "light_type",
            "color",
            "energy",
            "radius",
            "spot_size",
            "spot_blend",
        ],
        &["id"],
    )?;
    let node_operation = node_create_operation(operation, "light");
    super::node::apply(engine, "node.create", &node_operation)?;
    let node_id = read_id(engine, operation, "id")?;
    let data_id = node_data_id(engine, &node_id)?;
    let mut light = LightData::default();
    let light_values = selected_values(
        operation,
        &[
            "light_type",
            "color",
            "energy",
            "radius",
            "spot_size",
            "spot_blend",
        ],
    );
    apply_light_values(engine, &light_values, "", &mut light)?;
    let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "light data block disappeared after creation",
        )
    })?;
    data.light = Some(light);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn update_light(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = set_object(engine, operation)?;
    let (_, data_id) = node_data_target(engine, operation, "light")?;
    let mut light = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|data| data.light.clone())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                "light node has no light payload",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    apply_light_values(engine, set, "set/", &mut light)?;
    let data =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "light data block disappeared")
        })?;
    if data.light.as_ref() == Some(&light) {
        return Ok(false);
    }
    data.light = Some(light);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn apply_light_values(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    prefix: &str,
    light: &mut LightData,
) -> Result<()> {
    check_camera_set_fields(
        engine,
        values,
        &[
            "light_type",
            "color",
            "energy",
            "radius",
            "spot_size",
            "spot_blend",
        ],
        prefix,
    )?;
    if let Some(value) = values.get("light_type") {
        light.light_type = match value.as_str() {
            Some("point") => LightType::Point,
            Some("sun") => LightType::Sun,
            Some("spot") => LightType::Spot,
            Some("area") => LightType::Area,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "light_type must be point, sun, spot, or area",
                    &operation_pointer(engine.operation_index, &format!("{prefix}light_type")),
                ));
            }
        };
    }
    if let Some(value) = values.get("color") {
        light.color = read_vec3(engine, value, &format!("{prefix}color"))?;
        if light.color.iter().any(|component| *component < 0.0) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "light color components must be non-negative",
                &operation_pointer(engine.operation_index, &format!("{prefix}color")),
            ));
        }
    }
    if let Some(value) = values.get("energy") {
        light.energy = nonnegative_number(engine, value, &format!("{prefix}energy"))?;
    }
    if let Some(value) = values.get("radius") {
        light.radius = nonnegative_number(engine, value, &format!("{prefix}radius"))?;
    }
    if let Some(value) = values.get("spot_size") {
        light.spot_size = super::finite_number(
            engine,
            Some(value),
            &format!("{prefix}spot_size"),
            super::FiniteNumberMessage::Field,
        )?;
        if light.spot_size <= 0.0 || light.spot_size > std::f64::consts::PI {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "spot_size must be in (0, π] radians",
                &operation_pointer(engine.operation_index, &format!("{prefix}spot_size")),
            ));
        }
    }
    if let Some(value) = values.get("spot_blend") {
        light.spot_blend = super::finite_number(
            engine,
            Some(value),
            &format!("{prefix}spot_blend"),
            super::FiniteNumberMessage::Field,
        )?;
        if !(0.0..=1.0).contains(&light.spot_blend) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "spot_blend must be between 0 and 1",
                &operation_pointer(engine.operation_index, &format!("{prefix}spot_blend")),
            ));
        }
    }
    Ok(())
}

fn create_world(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "color", "strength", "node_tree"],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.worlds.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("world ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mut world = World::default();
    let world_values = selected_values(operation, &["color", "strength", "node_tree"]);
    apply_world_values(engine, &world_values, "", &mut world)?;
    engine.doc.worlds.insert(id.clone(), world);
    engine.mark("worlds", &id, ChangeKind::Created);
    Ok(true)
}

fn update_world(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        super::TargetIdPolicy::Strict {
            object_message: "target must contain an ID",
            shape_message: "target must contain only id",
            id_message: "target id must be a string",
            require_id: true,
        },
    )?;
    let set = set_object(engine, operation)?;
    let mut world = engine.doc.worlds.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("world `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    apply_world_values(engine, set, "set/", &mut world)?;
    if engine.doc.worlds.get(&id) == Some(&world) {
        return Ok(false);
    }
    engine.doc.worlds.insert(id.clone(), world);
    engine.mark("worlds", &id, ChangeKind::Updated);
    Ok(true)
}

fn apply_world_values(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    prefix: &str,
    world: &mut World,
) -> Result<()> {
    check_camera_set_fields(engine, values, &["color", "strength", "node_tree"], prefix)?;
    if let Some(value) = values.get("color") {
        world.color = read_vec3(engine, value, &format!("{prefix}color"))?;
        if world.color.iter().any(|component| *component < 0.0) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "world color components must be non-negative",
                &operation_pointer(engine.operation_index, &format!("{prefix}color")),
            ));
        }
    }
    if let Some(value) = values.get("strength") {
        world.strength = nonnegative_number(engine, value, &format!("{prefix}strength"))?;
    }
    if let Some(value) = values.get("node_tree") {
        world.node_tree = if value.is_null() {
            None
        } else {
            let graph = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "world node_tree must be a node-group ID or null",
                    &operation_pointer(engine.operation_index, &format!("{prefix}node_tree")),
                )
            })?;
            let id = Id::new(graph.to_owned()).map_err(|_| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "world node_tree must be a valid node-group ID",
                    &operation_pointer(engine.operation_index, &format!("{prefix}node_tree")),
                )
            })?;
            if !engine.doc.node_groups.contains_key(&id) {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    format!("world node group `{id}` was not found"),
                    &operation_pointer(engine.operation_index, &format!("{prefix}node_tree")),
                ));
            }
            Some(id)
        };
    }
    Ok(())
}
fn selected_values(values: &Map<String, Value>, fields: &[&str]) -> Map<String, Value> {
    fields
        .iter()
        .filter_map(|field| {
            values
                .get(*field)
                .map(|value| ((*field).to_owned(), value.clone()))
        })
        .collect()
}

fn node_create_operation(operation: &Map<String, Value>, kind: &str) -> Map<String, Value> {
    let mut result = Map::new();
    result.insert("op".to_owned(), Value::String("node.create".to_owned()));
    result.insert("kind".to_owned(), Value::String(kind.to_owned()));
    for field in [
        "id",
        "tags",
        "name",
        "collection",
        "parent",
        "parent_inverse",
        "transform",
        "visible",
        "render_visible",
        "selectable",
    ] {
        if let Some(value) = operation.get(field) {
            result.insert(field.to_owned(), value.clone());
        }
    }
    result
}

fn node_data_target(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
) -> Result<(Id, Id)> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut nodes = resolve_node_targets(engine, target, false)?;
    let node_id = nodes.pop().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let node = engine.doc.nodes.get(&node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if node.kind != kind {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("target node must be a {kind}"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let data_id = node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node has no data block"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    Ok((node_id, data_id))
}

fn node_data_id(engine: &Engine<'_>, node_id: &Id) -> Result<Id> {
    engine
        .doc
        .nodes
        .get(node_id)
        .and_then(|node| node.data.clone())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "new camera/light has no data block",
                &operation_pointer(engine.operation_index, "id"),
            )
        })
}

fn set_object<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
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
    Ok(set)
}

fn check_camera_set_fields(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    super::check_set_fields_by(engine, values, allowed, |engine, key| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown field `{key}`"),
            &operation_pointer(engine.operation_index, &format!("{prefix}{key}")),
        )
    })
}

fn nonnegative_number(engine: &Engine<'_>, value: &Value, field: &str) -> Result<f64> {
    let number = super::finite_number(
        engine,
        Some(value),
        field,
        super::FiniteNumberMessage::Field,
    )?;
    if number < 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be non-negative"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(number)
}

fn read_vec2(engine: &Engine<'_>, value: &Value, field: &str) -> Result<[f64; 2]> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must contain 2 numbers"),
                &operation_pointer(engine.operation_index, field),
            )
        })?;
    Ok([
        super::finite_number(
            engine,
            Some(&values[0]),
            field,
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            Some(&values[1]),
            field,
            super::FiniteNumberMessage::Field,
        )?,
    ])
}

fn read_vec3(engine: &Engine<'_>, value: &Value, field: &str) -> Result<[f64; 3]> {
    let values = value
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must contain 3 numbers"),
                &operation_pointer(engine.operation_index, field),
            )
        })?;
    Ok([
        super::finite_number(
            engine,
            Some(&values[0]),
            field,
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            Some(&values[1]),
            field,
            super::FiniteNumberMessage::Field,
        )?,
        super::finite_number(
            engine,
            Some(&values[2]),
            field,
            super::FiniteNumberMessage::Field,
        )?,
    ])
}

fn read_id(engine: &Engine<'_>, operation: &Map<String, Value>, field: &str) -> Result<Id> {
    let text = operation
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be an ID"),
                &operation_pointer(engine.operation_index, field),
            )
        })?;
    parse_id(
        engine,
        text,
        &operation_pointer(engine.operation_index, field),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use serde_json::json;

    use crate::{model::SceneDoc, ops::apply_batch};

    #[test]
    fn creates_assigns_camera_and_stores_light_energy_and_world_values() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let outcome = apply_batch(
            &doc,
            &json!({"schema_version":1,"base_revision":0,"operations":[
                {"op":"camera.create","id":"camera_main","lens_mm":35.0},
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
                {"op":"light.create","id":"key_light","light_type":"point","energy":240.0},
                {"op":"world.create","id":"world_main","color":[0.1,0.2,0.3],"strength":0.5},
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"world":"world_main"}}
            ]}),
        )
        .unwrap();
        let camera_node = &outcome.doc.nodes[&crate::model::Id::new("camera_main").unwrap()];
        let camera_data = &outcome.doc.data_blocks[camera_node.data.as_ref().unwrap()].camera;
        assert_eq!(camera_data.as_ref().unwrap().lens_mm, 35.0);
        let light_node = &outcome.doc.nodes[&crate::model::Id::new("key_light").unwrap()];
        let light_data = &outcome.doc.data_blocks[light_node.data.as_ref().unwrap()].light;
        assert_eq!(light_data.as_ref().unwrap().energy, 240.0);
        assert_eq!(
            outcome.doc.worlds[&crate::model::Id::new("world_main").unwrap()].color,
            [0.1, 0.2, 0.3]
        );
        assert_eq!(
            outcome.doc.scenes[&crate::model::Id::new("scene_main").unwrap()]
                .camera
                .as_ref()
                .unwrap()
                .as_str(),
            "camera_main"
        );
    }
}
