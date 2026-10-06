use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{ForceField, Id, Modifier, Node, RigidBody, RigidBodyWorld},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer};

const TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::Strict {
    object_message: "target must be an object containing id",
    shape_message: "target must contain only id",
    id_message: "target id must be a string",
    require_id: true,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "physics.cloth.create"
        | "physics.cloth.update"
        | "physics.cloth.delete"
        | "physics.soft_body.create"
        | "physics.soft_body.update"
        | "physics.soft_body.delete"
        | "physics.particle_emitter.create"
        | "physics.particle_emitter.update"
        | "physics.particle_emitter.delete"
        | "physics.fluid.create"
        | "physics.fluid.update"
        | "physics.fluid.delete"
        | "physics.dynamic_paint.create"
        | "physics.dynamic_paint.update"
        | "physics.dynamic_paint.delete"
        | "physics.collision.create"
        | "physics.collision.update"
        | "physics.collision.delete" => apply_physics_system(engine, name, operation),
        "physics.world.update" | "simulation.settings.update" => update_world(engine, operation),
        "physics.rigid_body.create" => create_rigid_body(engine, operation),
        "physics.rigid_body.update" => update_rigid_body(engine, operation),
        "physics.rigid_body.delete" => delete_rigid_body(engine, operation),
        "physics.force_field.create" => create_force_field(engine, operation),
        "physics.force_field.update" => update_force_field(engine, operation),
        "physics.force_field.delete" => delete_force_field(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported physics operation `{name}`"),
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn update_world(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let scene_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let updates = set_object(engine, operation)?;
    check_physics_set_fields(
        engine,
        updates,
        &[
            "enabled",
            "gravity",
            "substeps",
            "solver_iterations",
            "frame_start",
            "frame_end",
            "seed",
        ],
        "set",
    )?;
    let scene = engine.doc.scenes.get(&scene_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "physics Scene was not found",
            "/target/id",
        )
    })?;
    let before = scene.rigid_body_world.clone();
    let mut value = serde_json::to_value(before.clone().unwrap_or_default())
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let object = value.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "rigid body world did not serialize as an object",
        )
    })?;
    object.extend(updates.clone());
    let world = serde_json::from_value::<RigidBodyWorld>(Value::Object(object.clone())).map_err(
        |error| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("invalid rigid body world settings: {error}"),
                "/set",
            )
        },
    )?;
    validate_world(engine, &world)?;
    if before.as_ref() == Some(&world) {
        return Ok(false);
    }
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "physics Scene disappeared during update",
        )
    })?;
    scene.rigid_body_world = Some(world);
    engine.mark("scenes", &scene_id, ChangeKind::Updated);
    Ok(true)
}

fn create_rigid_body(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    const FIELDS: &[&str] = &[
        "op",
        "target",
        "type",
        "mass",
        "friction",
        "restitution",
        "shape",
        "linear_damping",
        "angular_damping",
        "initial_velocity",
    ];
    check_fields(engine, operation, FIELDS, &["target", "type"])?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "rigid body")?;
    if node.rigid_body.is_some() {
        return Err(engine.error(
            ErrorCode::IdExists,
            "Object already has a rigid body",
            "/target/id",
        ));
    }
    validate_rigid_body_node(engine, &node_id, &node)?;
    let body = decode_rigid_body(engine, RigidBody::default(), operation, FIELDS)?;
    validate_body(engine, &body)?;
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.rigid_body = Some(body);
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn update_rigid_body(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    const FIELDS: &[&str] = &[
        "type",
        "mass",
        "friction",
        "restitution",
        "shape",
        "linear_damping",
        "angular_damping",
        "initial_velocity",
    ];
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "rigid body")?;
    let before = node.rigid_body.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "Object has no rigid body",
            "/target/id",
        )
    })?;
    let updates = set_object(engine, operation)?;
    check_physics_set_fields(engine, updates, FIELDS, "set")?;
    let body = decode_rigid_body(engine, before.clone(), updates, FIELDS)?;
    validate_body(engine, &body)?;
    if before == body {
        return Ok(false);
    }
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.rigid_body = Some(body);
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn delete_rigid_body(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "rigid body")?;
    if node.rigid_body.is_none() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "Object has no rigid body",
            "/target/id",
        ));
    }
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.rigid_body = None;
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn create_force_field(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    const FIELDS: &[&str] = &["op", "target", "type", "strength", "falloff"];
    check_fields(engine, operation, FIELDS, &["target", "type"])?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "force field")?;
    if node.kind != "empty" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "force fields require an Empty Object",
            "/target/id",
        ));
    }
    if node.force_field.is_some() {
        return Err(engine.error(
            ErrorCode::IdExists,
            "Object already has a force field",
            "/target/id",
        ));
    }
    let field = decode_force_field(engine, ForceField::default(), operation, FIELDS)?;
    validate_force_field(engine, &field)?;
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.force_field = Some(field);
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn update_force_field(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    const FIELDS: &[&str] = &["type", "strength", "falloff"];
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "force field")?;
    if node.kind != "empty" {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "force fields require an Empty Object",
            "/target/id",
        ));
    }
    let before = node.force_field.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "Object has no force field",
            "/target/id",
        )
    })?;
    let updates = set_object(engine, operation)?;
    check_physics_set_fields(engine, updates, FIELDS, "set")?;
    let field = decode_force_field(engine, before.clone(), updates, FIELDS)?;
    validate_force_field(engine, &field)?;
    if before == field {
        return Ok(false);
    }
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.force_field = Some(field);
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn delete_force_field(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let node_id =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &node_id, "force field")?;
    if node.force_field.is_none() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "Object has no force field",
            "/target/id",
        ));
    }
    if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
        node.force_field = None;
    }
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn decode_rigid_body(
    engine: &Engine<'_>,
    mut body: RigidBody,
    input: &Map<String, Value>,
    fields: &[&str],
) -> Result<RigidBody> {
    let mut values = serde_json::to_value(&body)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let object = values.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "rigid body did not serialize as an object",
        )
    })?;
    for field in fields
        .iter()
        .filter(|field| **field != "op" && **field != "target")
    {
        if let Some(value) = input.get(*field) {
            object.insert((*field).to_owned(), value.clone());
        }
    }
    body = serde_json::from_value::<RigidBody>(values).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid rigid body settings: {error}"),
            "/set",
        )
    })?;
    Ok(body)
}

fn decode_force_field(
    engine: &Engine<'_>,
    mut field: ForceField,
    input: &Map<String, Value>,
    fields: &[&str],
) -> Result<ForceField> {
    let mut values = serde_json::to_value(&field)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let object = values.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "force field did not serialize as an object",
        )
    })?;
    for name in fields
        .iter()
        .filter(|name| **name != "op" && **name != "target")
    {
        if let Some(value) = input.get(*name) {
            object.insert((*name).to_owned(), value.clone());
        }
    }
    field = serde_json::from_value::<ForceField>(values).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid force field settings: {error}"),
            "/set",
        )
    })?;
    Ok(field)
}

fn validate_world(engine: &Engine<'_>, world: &RigidBodyWorld) -> Result<()> {
    if !world.gravity.iter().all(|value| value.is_finite())
        || world.substeps == 0
        || world.substeps > 128
        || world.solver_iterations == 0
        || world.solver_iterations > 128
        || world.frame_start > world.frame_end
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rigid body world settings are outside supported ranges",
            "/set",
        ));
    }
    Ok(())
}

fn validate_body(engine: &Engine<'_>, body: &RigidBody) -> Result<()> {
    if !body.mass.is_finite()
        || body.mass < 0.0
        || (body.body_type == crate::model::RigidBodyType::Active && body.mass == 0.0)
        || !body.friction.is_finite()
        || body.friction < 0.0
        || !body.restitution.is_finite()
        || !(0.0..=1.0).contains(&body.restitution)
        || !body.linear_damping.is_finite()
        || body.linear_damping < 0.0
        || !body.angular_damping.is_finite()
        || body.angular_damping < 0.0
        || !body.initial_velocity.iter().all(|value| value.is_finite())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rigid body values are invalid",
            "/set",
        ));
    }
    Ok(())
}

fn validate_force_field(engine: &Engine<'_>, field: &ForceField) -> Result<()> {
    if !field.strength.is_finite() || !field.falloff.is_finite() || field.falloff < 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "force field values must be finite and falloff non-negative",
            "/set",
        ));
    }
    Ok(())
}

fn physics_node(engine: &Engine<'_>, id: &Id, label: &str) -> Result<Node> {
    engine.doc.nodes.get(id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("{label} Object was not found"),
            "/target/id",
        )
    })
}
fn validate_rigid_body_node(engine: &Engine<'_>, node_id: &Id, node: &Node) -> Result<()> {
    if node.kind != "mesh"
        || node
            .data
            .as_ref()
            .and_then(|data_id| engine.doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .is_none()
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "rigid bodies require a mesh Object with mesh data",
            &format!("/nodes/{node_id}/rigid_body"),
        ));
    }
    Ok(())
}

fn set_object<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
    operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| engine.error(ErrorCode::InvalidOperation, "set must be an object", "/set"))
}

fn check_physics_set_fields(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    super::check_set_fields_by(engine, set, allowed, |engine, field| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown physics set field `{field}`"),
            &format!("/{prefix}/{field}"),
        )
    })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &format!("/{prefix}"),
        ));
    }
    Ok(())
}
fn apply_physics_system(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    let (system, action) = name
        .strip_prefix("physics.")
        .and_then(|suffix| suffix.split_once('.'))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "invalid physics operation",
                "/op",
            )
        })?;
    let (property, canonical_system) = match system {
        "cloth" => ("physics_cloth", "cloth"),
        "soft_body" => ("physics_soft_body", "soft_body"),
        "particle_emitter" => ("physics_particle_emitter", "particle_emitter"),
        "fluid" => ("physics_fluid", "fluid"),
        "dynamic_paint" => ("physics_dynamic_paint", "dynamic_paint"),
        "collision" => ("physics_collision", "collision"),
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported physics system `{system}`"),
                "/op",
            ));
        }
    };
    let target =
        super::target_id_value(engine, operation.get("target"), "/target", TARGET_ID_POLICY)?;
    let node = physics_node(engine, &target, "physics")?;
    validate_system_target(engine, &target, &node)?;
    let current = node.properties.get(property);

    match action {
        "create" => {
            let allowed = if canonical_system == "particle_emitter" {
                &["op", "target", "settings", "seed"][..]
            } else {
                &["op", "target", "settings"][..]
            };
            check_fields(engine, operation, allowed, &["target"])?;
            if current.is_some() {
                return Err(engine.error(
                    ErrorCode::IdExists,
                    format!("{system} is already configured on this Object"),
                    "/target/id",
                ));
            }
            let mut settings = operation
                .get("settings")
                .map(|value| {
                    value.as_object().cloned().ok_or_else(|| {
                        engine.error(
                            ErrorCode::InvalidOperation,
                            "settings must be an object",
                            "/settings",
                        )
                    })
                })
                .transpose()?
                .unwrap_or_default();
            if let Some(seed) = operation.get("seed") {
                settings.insert("seed".to_owned(), seed.clone());
            }
            validate_physics_settings(engine, canonical_system, &settings)?;
            validate_physics_references(engine, canonical_system, &target, &settings)?;
            let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "physics target disappeared")
            })?;
            node.properties
                .insert(property.to_owned(), Value::Object(settings));
            ensure_physics_stack_modifier(engine, &target, canonical_system)?;
            engine.mark("nodes", &target, ChangeKind::Updated);
            Ok(true)
        }
        "update" => {
            check_fields(
                engine,
                operation,
                &["op", "target", "set"],
                &["target", "set"],
            )?;
            let updates = set_object(engine, operation)?;
            if updates.is_empty() {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "set must not be empty",
                    "/set",
                ));
            }
            let mut settings = current.and_then(Value::as_object).cloned().ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!("{system} is not configured on this Object"),
                    "/target/id",
                )
            })?;
            settings.extend(updates.clone());
            validate_physics_settings(engine, canonical_system, &settings)?;
            validate_physics_references(engine, canonical_system, &target, &settings)?;
            let mut changed = current != Some(&Value::Object(settings.clone()));
            if changed {
                let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "physics target disappeared")
                })?;
                node.properties
                    .insert(property.to_owned(), Value::Object(settings));
            }
            changed |= ensure_physics_stack_modifier(engine, &target, canonical_system)?;
            if !changed {
                return Ok(false);
            }
            engine.mark("nodes", &target, ChangeKind::Updated);
            Ok(true)
        }
        "delete" => {
            check_fields(engine, operation, &["op", "target"], &["target"])?;
            if current.is_none() {
                return Err(engine.error(
                    ErrorCode::TargetNotFound,
                    format!("{system} is not configured on this Object"),
                    "/target/id",
                ));
            }
            let node = engine.doc.nodes.get_mut(&target).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "physics target disappeared")
            })?;
            node.properties.remove(property);
            if canonical_system == "particle_emitter" {
                node.properties.remove("physics_particle_systems");
                node.properties
                    .remove("physics_particle_emitter_modifier_id");
            }
            let modifier_type = physics_modifier_type(canonical_system);
            node.modifiers
                .retain(|modifier| modifier.modifier_type != modifier_type);
            engine.mark("nodes", &target, ChangeKind::Updated);
            Ok(true)
        }
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported physics operation `{name}`"),
            "/op",
        )),
    }
}

fn physics_modifier_type(system: &str) -> &'static str {
    match system {
        "cloth" => "cloth",
        "soft_body" => "soft_body",
        "particle_emitter" => "particle_system",
        "fluid" => "fluid",
        "dynamic_paint" => "dynamic_paint",
        "collision" => "collision",
        _ => "unknown",
    }
}

fn physics_system_for_modifier(modifier_type: &str) -> Option<&'static str> {
    match modifier_type {
        "cloth" => Some("cloth"),
        "soft_body" => Some("soft_body"),
        "collision" => Some("collision"),
        "dynamic_paint" => Some("dynamic_paint"),
        "fluid" => Some("fluid"),
        "particle_system" => Some("particle_emitter"),
        _ => None,
    }
}

fn physics_property(system: &str) -> &'static str {
    match system {
        "cloth" => "physics_cloth",
        "soft_body" => "physics_soft_body",
        "collision" => "physics_collision",
        "dynamic_paint" => "physics_dynamic_paint",
        "fluid" => "physics_fluid",
        "particle_emitter" => "physics_particle_emitter",
        _ => "physics_unknown",
    }
}

pub(super) fn create_linked_modifier_settings(
    engine: &mut Engine<'_>,
    target: &Id,
    modifier_id: &Id,
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let Some(system) = physics_system_for_modifier(modifier_type) else {
        return Ok(params.clone());
    };
    let mut settings = params.clone();
    if let Some(settings_id) = settings.remove("settings_id")
        && settings_id.as_str() != Some(modifier_id.as_str())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "physics modifier settings_id must equal the modifier ID",
            "/params/settings_id",
        ));
    }
    let node = physics_node(engine, target, "modifier physics")?;
    validate_system_target(engine, target, &node)?;
    validate_physics_settings(engine, system, &settings)?;
    validate_physics_references(engine, system, target, &settings)?;
    let property = physics_property(system);
    let node = engine.doc.nodes.get_mut(target).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier physics target disappeared",
        )
    })?;
    if system == "particle_emitter" && node.properties.contains_key(property) {
        let registry = node
            .properties
            .entry("physics_particle_systems".to_owned())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "physics_particle_systems must be an object keyed by settings ID",
                )
            })?;
        if registry.contains_key(modifier_id.as_str()) {
            return Err(engine.error(
                ErrorCode::IdExists,
                "particle system settings ID already exists",
                "/id",
            ));
        }
        registry.insert(modifier_id.to_string(), Value::Object(settings));
    } else {
        if node.properties.contains_key(property) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("{system} is already configured on this Object"),
                "/target/id",
            ));
        }
        node.properties
            .insert(property.to_owned(), Value::Object(settings));
        if system == "particle_emitter" {
            node.properties.insert(
                "physics_particle_emitter_modifier_id".to_owned(),
                Value::String(modifier_id.to_string()),
            );
        }
    }
    engine.mark("nodes", target, ChangeKind::Updated);
    Ok(Map::from_iter([(
        "settings_id".to_owned(),
        Value::String(modifier_id.to_string()),
    )]))
}

pub(super) fn update_linked_modifier_settings(
    engine: &mut Engine<'_>,
    target: &Id,
    modifier_id: &Id,
    modifier_type: &str,
    params: &Map<String, Value>,
) -> Result<Map<String, Value>> {
    let Some(system) = physics_system_for_modifier(modifier_type) else {
        return Ok(params.clone());
    };
    if params
        .get("settings_id")
        .is_some_and(|settings_id| settings_id.as_str() != Some(modifier_id.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "physics modifier settings_id cannot be changed",
            "/set/params/settings_id",
        ));
    }
    let mut settings = params.clone();
    settings.remove("settings_id");
    let property = physics_property(system);
    let node = physics_node(engine, target, "modifier physics")?;
    let is_particle_registry_entry = system == "particle_emitter"
        && node
            .properties
            .get("physics_particle_systems")
            .and_then(Value::as_object)
            .is_some_and(|registry| registry.contains_key(modifier_id.as_str()));
    let settings_exist = if is_particle_registry_entry {
        true
    } else {
        node.properties.contains_key(property)
    };
    if !settings_exist {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("{system} has no linked physics settings"),
            "/modifier_id",
        ));
    }
    validate_physics_settings(engine, system, &settings)?;
    validate_physics_references(engine, system, target, &settings)?;
    let node = engine.doc.nodes.get_mut(target).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier physics target disappeared",
        )
    })?;
    if is_particle_registry_entry {
        let registry = node
            .properties
            .get_mut("physics_particle_systems")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "particle settings registry is invalid",
                )
            })?;
        registry.insert(modifier_id.to_string(), Value::Object(settings));
    } else {
        node.properties
            .insert(property.to_owned(), Value::Object(settings));
        if system == "particle_emitter" {
            node.properties.insert(
                "physics_particle_emitter_modifier_id".to_owned(),
                Value::String(modifier_id.to_string()),
            );
        }
    }
    engine.mark("nodes", target, ChangeKind::Updated);
    Ok(Map::from_iter([(
        "settings_id".to_owned(),
        Value::String(modifier_id.to_string()),
    )]))
}

pub(super) fn delete_linked_modifier_settings(
    engine: &mut Engine<'_>,
    target: &Id,
    modifier: &Modifier,
) -> Result<()> {
    let Some(system) = physics_system_for_modifier(&modifier.modifier_type) else {
        return Ok(());
    };
    let expected = modifier.params.get("settings_id").and_then(Value::as_str);
    if expected.is_some_and(|settings_id| settings_id != modifier.id.as_str()) {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            "physics modifier settings_id does not match its modifier ID",
            "/modifier_id",
        ));
    }
    let node = engine.doc.nodes.get_mut(target).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "modifier physics target disappeared",
        )
    })?;
    let mut changed = false;
    if system == "particle_emitter"
        && let Some(registry) = node
            .properties
            .get_mut("physics_particle_systems")
            .and_then(Value::as_object_mut)
        && registry.remove(modifier.id.as_str()).is_some()
    {
        changed = true;
        if registry.is_empty() {
            node.properties.remove("physics_particle_systems");
        }
    } else if node.properties.remove(physics_property(system)).is_some() {
        changed = true;
        if system == "particle_emitter" {
            node.properties
                .remove("physics_particle_emitter_modifier_id");
        }
    }
    if changed {
        engine.mark("nodes", target, ChangeKind::Updated);
    }
    Ok(())
}

fn ensure_physics_stack_modifier(
    engine: &mut Engine<'_>,
    target: &Id,
    system: &str,
) -> Result<bool> {
    let modifier_type = physics_modifier_type(system);
    let node = engine
        .doc
        .nodes
        .get_mut(target)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "physics target disappeared"))?;
    if let Some(modifier) = node
        .modifiers
        .iter_mut()
        .find(|modifier| modifier.modifier_type == modifier_type)
    {
        let settings_id = modifier.id.to_string();
        let mut changed = false;
        if modifier.params.get("settings_id").and_then(Value::as_str) != Some(settings_id.as_str())
        {
            modifier
                .params
                .insert("settings_id".to_owned(), Value::String(settings_id.clone()));
            changed = true;
        }
        if system == "particle_emitter"
            && node.properties.contains_key("physics_particle_emitter")
            && node
                .properties
                .get("physics_particle_emitter_modifier_id")
                .and_then(Value::as_str)
                != Some(settings_id.as_str())
        {
            node.properties.insert(
                "physics_particle_emitter_modifier_id".to_owned(),
                Value::String(settings_id),
            );
            changed = true;
        }
        return Ok(changed);
    }
    let base = format!("physics_{system}");
    let mut suffix = 0_u32;
    let modifier_id = loop {
        let candidate = if suffix == 0 {
            base.clone()
        } else {
            format!("{base}_{suffix}")
        };
        if !node
            .modifiers
            .iter()
            .any(|modifier| modifier.id.as_str() == candidate)
        {
            break Id::new(candidate).map_err(|error| PotError::invalid_argument(error.message))?;
        }
        suffix = suffix.checked_add(1).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "physics modifier ID space is exhausted",
            )
        })?;
    };
    let mut params = Map::new();
    let settings_id = modifier_id.to_string();
    params.insert("settings_id".to_owned(), Value::String(settings_id.clone()));
    node.modifiers.push(Modifier {
        id: modifier_id,
        modifier_type: modifier_type.to_owned(),
        name: format!("{system} Physics"),
        enabled: true,
        params,
        binding_data: None,
        runtime: crate::model::ModifierRuntime::default(),
    });
    if system == "particle_emitter" {
        node.properties.insert(
            "physics_particle_emitter_modifier_id".to_owned(),
            Value::String(settings_id),
        );
    }
    Ok(true)
}

fn validate_system_target(engine: &Engine<'_>, id: &Id, node: &Node) -> Result<()> {
    if node.kind != "mesh"
        || node
            .data
            .as_ref()
            .and_then(|data_id| engine.doc.data_blocks.get(data_id))
            .and_then(|data| data.mesh.as_ref())
            .is_none()
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "physics systems require a mesh Object with mesh data",
            &format!("/nodes/{id}"),
        ));
    }
    Ok(())
}

fn validate_physics_references(
    engine: &Engine<'_>,
    system: &str,
    target: &Id,
    settings: &Map<String, Value>,
) -> Result<()> {
    if system == "particle_emitter"
        && (settings.get("render_as").and_then(Value::as_str) == Some("object")
            || settings.get("render_type").and_then(Value::as_str) == Some("OBJECT"))
    {
        let value = settings
            .get("instance_object")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "object particle rendering requires instance_object",
                    "/settings/instance_object",
                )
            })?;
        let instance_id = Id::new(value.to_owned()).map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                error.message,
                "/settings/instance_object",
            )
        })?;
        if &instance_id == target {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "particle instances cannot reference their own emitter",
                "/settings/instance_object",
            ));
        }
        let instance = physics_node(engine, &instance_id, "particle instance")?;
        validate_system_target(engine, &instance_id, &instance)?;
    }
    if system == "particle_emitter"
        && settings.get("render_type").and_then(Value::as_str) == Some("COLLECTION")
    {
        let collection_text = settings
            .get("instance_collection")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "collection particle rendering requires instance_collection",
                    "/settings/instance_collection",
                )
            })?;
        let collection_id = Id::new(collection_text.to_owned()).map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                error.message,
                "/settings/instance_collection",
            )
        })?;
        if !engine.doc.collections.contains_key(&collection_id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "particle instance collection was not found",
                "/settings/instance_collection",
            ));
        }
    }
    if system == "dynamic_paint"
        && let Some(brushes) = settings.get("brushes").and_then(Value::as_array)
    {
        for (index, brush) in brushes.iter().enumerate() {
            let value = brush.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "dynamic-paint brush IDs must be strings",
                    &format!("/settings/brushes/{index}"),
                )
            })?;
            let brush_id = Id::new(value.to_owned()).map_err(|error| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    error.message,
                    &format!("/settings/brushes/{index}"),
                )
            })?;
            if &brush_id == target {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "dynamic-paint canvas cannot also be one of its brushes",
                    &format!("/settings/brushes/{index}"),
                ));
            }
            let brush = physics_node(engine, &brush_id, "dynamic-paint brush")?;
            if brush
                .properties
                .get("physics_dynamic_paint")
                .and_then(|settings| settings.get("role"))
                .and_then(Value::as_str)
                != Some("brush")
            {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "dynamic-paint brush target is not configured as a brush",
                    &format!("/settings/brushes/{index}"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_physics_settings(
    engine: &Engine<'_>,
    system: &str,
    settings: &Map<String, Value>,
) -> Result<()> {
    let allowed: &[&str] = match system {
        "cloth" => &[
            "stiffness",
            "structural_stiffness",
            "tension_stiffness",
            "compression_stiffness",
            "shear_stiffness",
            "bend_stiffness",
            "bending_stiffness",
            "mass",
            "air_drag",
            "air_damping",
            "drag",
            "damping",
            "substeps",
            "iterations",
            "quality",
            "pin_group",
            "vertex_group_mass",
            "collision_enabled",
            "use_collision",
            "collision_distance",
            "distance_min",
            "use_self_collision",
            "self_collision_distance",
        ],
        "soft_body" => &[
            "stiffness",
            "edge_stiffness",
            "volume_stiffness",
            "goal_group",
            "vertex_group_goal",
            "goal_strength",
            "goal_stiffness",
            "goal_spring",
            "goal_friction",
            "goal_default",
            "goal_min",
            "goal_max",
            "mass",
            "friction",
            "speed",
            "plastic",
            "bend",
            "substeps",
            "iterations",
            "quality",
            "air_drag",
            "drag",
            "damping",
            "restitution",
        ],
        "particle_emitter" => &[
            "seed",
            "count",
            "rate",
            "emission_rate",
            "lifetime",
            "lifetime_random",
            "speed",
            "normal_factor",
            "factor_random",
            "random_velocity",
            "source",
            "emit_from",
            "velocity",
            "normal_velocity",
            "physics_type",
            "mass",
            "render_as",
            "render_type",
            "instance_object",
            "hair_length",
            "particle_size",
            "size_random",
            "child_nbr",
            "child_radius",
            "child_type",
            "start_frame",
            "end_frame",
            "frame_start",
            "frame_end",
            "instance_collection",
        ],
        "fluid" => &[
            "type",
            "resolution",
            "particle_radius",
            "smoothing_length",
            "pressure_stiffness",
            "viscosity",
            "inflow_rate",
            "seed",
        ],
        "dynamic_paint" => &[
            "role",
            "radius",
            "color",
            "strength",
            "brushes",
            "surface_format",
        ],
        "collision" => &[
            "thickness",
            "thickness_outer",
            "thickness_inner",
            "damping",
            "damping_factor",
            "damping_random",
            "permeability",
            "stickiness",
            "friction_factor",
            "friction_random",
            "use_culling",
            "use_normal",
            "use_particle_kill",
        ],
        _ => &[],
    };
    for (field, value) in settings {
        if !allowed.contains(&field.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown {system} setting `{field}`"),
                &format!("/settings/{field}"),
            ));
        }
        let valid = match field.as_str() {
            "thickness"
            | "thickness_outer"
            | "thickness_inner"
            | "collision_distance"
            | "self_collision_distance"
            | "distance_min"
            | "child_radius" => value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number >= 0.0),
            "permeability" | "stickiness" | "damping_factor" | "damping_random"
            | "friction_factor" | "friction_random" | "lifetime_random" | "size_random"
            | "restitution" => value
                .as_f64()
                .is_some_and(|number| number.is_finite() && (0.0..=1.0).contains(&number)),
            "use_particle_kill" | "collision_enabled" | "use_collision" | "use_self_collision"
            | "use_culling" | "use_normal" => value.is_boolean(),
            "seed" => value
                .as_u64()
                .is_some_and(|seed| u32::try_from(seed).is_ok()),
            "substeps" | "quality" | "collision_quality" => value
                .as_u64()
                .is_some_and(|count| (1..=32).contains(&count)),
            "iterations" => value
                .as_u64()
                .is_some_and(|count| (1..=64).contains(&count)),
            "resolution" => value
                .as_u64()
                .is_some_and(|count| (1..=10).contains(&count)),
            "count" | "child_nbr" => value.as_u64().is_some_and(|count| count <= 1_000_000),
            "rate" | "emission_rate" => value
                .as_f64()
                .is_some_and(|rate| rate.is_finite() && (0.0..=1_000_000.0).contains(&rate)),
            "lifetime" => value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number > 0.0 && number <= 1_000_000.0),
            "start_frame" | "end_frame" | "frame_start" | "frame_end" | "goal_min" | "goal_max" => {
                value.as_f64().is_some_and(f64::is_finite)
            }
            "particle_radius" | "smoothing_length" | "radius" | "hair_length" | "particle_size"
            | "mass" => value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number > 0.0),
            "stiffness"
            | "structural_stiffness"
            | "tension_stiffness"
            | "compression_stiffness"
            | "shear_stiffness"
            | "bend_stiffness"
            | "bending_stiffness"
            | "edge_stiffness"
            | "volume_stiffness"
            | "goal_strength"
            | "goal_stiffness"
            | "goal_spring"
            | "goal_friction"
            | "goal_default"
            | "air_drag"
            | "air_damping"
            | "drag"
            | "damping"
            | "pressure_stiffness"
            | "viscosity"
            | "inflow_rate"
            | "strength"
            | "speed"
            | "normal_factor"
            | "normal_velocity"
            | "factor_random"
            | "friction"
            | "plastic"
            | "bend" => value
                .as_f64()
                .is_some_and(|number| number.is_finite() && number >= 0.0),
            "pin_group"
            | "vertex_group_mass"
            | "goal_group"
            | "vertex_group"
            | "instance_object"
            | "vertex_group_goal"
            | "instance_collection" => value.as_str().is_some_and(|name| !name.is_empty()),
            "source" => value
                .as_str()
                .is_some_and(|source| matches!(source, "faces" | "vertices")),
            "emit_from" => value
                .as_str()
                .is_some_and(|source| matches!(source, "VERT" | "FACE" | "VOLUME")),
            "render_as" => value
                .as_str()
                .is_some_and(|mode| matches!(mode, "points" | "object" | "hair_curves")),
            "render_type" => value
                .as_str()
                .is_some_and(|mode| matches!(mode, "HALO" | "PATH" | "OBJECT" | "COLLECTION")),
            "physics_type" => value
                .as_str()
                .is_some_and(|kind| matches!(kind, "NEWTON" | "NO")),
            "child_type" => value
                .as_str()
                .is_some_and(|kind| matches!(kind, "NONE" | "SIMPLE" | "INTERPOLATED")),
            "type" => value
                .as_str()
                .is_some_and(|kind| matches!(kind, "liquid" | "smoke" | "fire")),
            "role" => value
                .as_str()
                .is_some_and(|role| matches!(role, "canvas" | "brush")),
            "surface_format" => value
                .as_str()
                .is_some_and(|format| matches!(format, "color" | "weight")),
            "color" => value.as_array().is_some_and(|color| {
                color.len() == 4
                    && color
                        .iter()
                        .all(|channel| channel.as_f64().is_some_and(f64::is_finite))
            }),
            "velocity" => value.as_array().is_some_and(|velocity| {
                velocity.len() == 3
                    && velocity
                        .iter()
                        .all(|component| component.as_f64().is_some_and(f64::is_finite))
            }),
            "random_velocity" => {
                value
                    .as_f64()
                    .is_some_and(|number| number.is_finite() && number >= 0.0)
                    || value.as_array().is_some_and(|values| {
                        values.len() == 3
                            && values.iter().all(|component| {
                                component
                                    .as_f64()
                                    .is_some_and(|number| number.is_finite() && number >= 0.0)
                            })
                    })
            }
            "brushes" => value
                .as_array()
                .is_some_and(|brushes| brushes.iter().all(|brush| brush.as_str().is_some())),
            _ => false,
        };
        if !valid {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("invalid value for {system} setting `{field}`"),
                &format!("/settings/{field}"),
            ));
        }
    }
    if system == "fluid"
        && settings
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "liquid")
    {
        return Err(unsupported_physics(
            "physics.fluid.smoke_fire",
            "only liquid SPH simulation is supported; smoke and fire are not supported",
        ));
    }
    Ok(())
}

fn unsupported_physics(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":feature_id,"reason":reason,"status":"not_supported"}),
    )
}
