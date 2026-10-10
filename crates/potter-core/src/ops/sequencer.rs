use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::Id,
    sequencer::{Sequencer, Strip},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape, read_id,
};

const TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::Strict {
    object_message: "target must be an object",
    shape_message: "target must contain only id",
    id_message: "target.id must be a string",
    require_id: true,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "sequencer.strip_create" => strip_create(engine, operation),
        "sequencer.strip_update" => strip_update(engine, operation),
        "sequencer.strip_delete" => strip_delete(engine, operation),
        "sequencer.update" => sequencer_update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported sequencer operation `{name}`"),
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn strip_create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, STRIP_FIELDS, &["op", "id", "type"])?;
    let scene_id = scene_target(engine, operation)?;
    let mut fields = operation.clone();
    fields.remove("op");
    fields.remove("target");
    let mut strip: Strip = serde_json::from_value(Value::Object(fields)).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid sequencer strip: {error}"),
            &operation_pointer(engine.operation_index, ""),
        )
    })?;
    if strip.name.is_empty() {
        strip.name.clone_from(&strip.id);
    }
    validate_strip(engine, &strip)?;
    validate_input_strips(engine, &scene_id, &strip, "inputs")?;
    parse_id(
        engine,
        &strip.id,
        &operation_pointer(engine.operation_index, "id"),
    )?;
    let missing_scene = engine.error(
        ErrorCode::TargetNotFound,
        "sequencer scene does not exist",
        &operation_pointer(engine.operation_index, "target/id"),
    );
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
    if scene
        .sequencer
        .strips
        .iter()
        .any(|entry| entry.id == strip.id)
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            "sequencer strip ID already exists",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    if strip.channel > scene.sequencer.channels {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "sequencer strip channel exceeds the scene channel count",
            &operation_pointer(engine.operation_index, "channel"),
        ));
    }
    scene.sequencer.strips.push(strip);
    sort_strips(&mut scene.sequencer);
    engine.mark("scenes", &scene_id, ChangeKind::Updated);
    Ok(true)
}

fn strip_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "set"],
        &["id", "set"],
    )?;
    let scene_id = scene_target(engine, operation)?;
    let strip_id = read_id(engine, operation, "id")?;
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
        if !STRIP_SET_FIELDS.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown strip field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let (index, mut current, channel_count) = {
        let scene = engine.doc.scenes.get(&scene_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "sequencer scene does not exist",
                json!({"scene_id":scene_id}),
            )
        })?;
        let Some(index) = scene
            .sequencer
            .strips
            .iter()
            .position(|strip| strip.id == strip_id.as_str())
        else {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "sequencer strip does not exist",
                &operation_pointer(engine.operation_index, "id"),
            ));
        };
        let current = serde_json::to_value(&scene.sequencer.strips[index])
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        (index, current, scene.sequencer.channels)
    };
    let current_fields = current.as_object_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "serialized strip is not an object",
        )
    })?;
    for (key, value) in set {
        current_fields.insert(key.clone(), value.clone());
    }
    let updated: Strip = serde_json::from_value(current).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid sequencer strip update: {error}"),
            &operation_pointer(engine.operation_index, "set"),
        )
    })?;
    validate_strip(engine, &updated)?;
    if updated.id != strip_id.as_str() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "strip ID cannot be changed",
            &operation_pointer(engine.operation_index, "set/id"),
        ));
    }
    validate_input_strips(engine, &scene_id, &updated, "set/inputs")?;
    if updated.channel > channel_count {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "sequencer strip channel exceeds the scene channel count",
            &operation_pointer(engine.operation_index, "set/channel"),
        ));
    }
    let missing_scene = PotError::with_details(
        ErrorCode::TargetNotFound,
        "sequencer scene does not exist",
        json!({"scene_id":scene_id}),
    );
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
    let changed = scene.sequencer.strips[index] != updated;
    if changed {
        scene.sequencer.strips[index] = updated;
        sort_strips(&mut scene.sequencer);
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn strip_delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "id"], &["id"])?;
    let scene_id = scene_target(engine, operation)?;
    let strip_id = read_id(engine, operation, "id")?;
    let missing_scene = engine.error(
        ErrorCode::TargetNotFound,
        "sequencer scene does not exist",
        &operation_pointer(engine.operation_index, "target/id"),
    );
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
    let Some(index) = scene
        .sequencer
        .strips
        .iter()
        .position(|strip| strip.id == strip_id.as_str())
    else {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "sequencer strip does not exist",
            &operation_pointer(engine.operation_index, "id"),
        ));
    };
    if scene
        .sequencer
        .strips
        .iter()
        .enumerate()
        .any(|(candidate, strip)| {
            candidate != index && strip.inputs.iter().any(|input| input == strip_id.as_str())
        })
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "cannot delete a strip while other strips reference it",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    scene.sequencer.strips.remove(index);
    engine.mark("scenes", &scene_id, ChangeKind::Updated);
    Ok(true)
}

fn sequencer_update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "set"], &["set"])?;
    let scene_id = scene_target(engine, operation)?;
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
    if set.len() != 1 || !set.contains_key("channels") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "sequencer.update accepts only channels",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let channels = set["channels"]
        .as_u64()
        .filter(|value| (1..=256).contains(value))
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "channels must be an integer from 1 to 256",
                &operation_pointer(engine.operation_index, "set/channels"),
            )
        })?;
    let missing_scene = engine.error(
        ErrorCode::TargetNotFound,
        "sequencer scene does not exist",
        &operation_pointer(engine.operation_index, "target/id"),
    );
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or(missing_scene)?;
    if scene
        .sequencer
        .strips
        .iter()
        .any(|strip| strip.channel > channels)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "cannot reduce channels below an occupied strip channel",
            &operation_pointer(engine.operation_index, "set/channels"),
        ));
    }
    let changed = scene.sequencer.channels != channels;
    if changed {
        scene.sequencer.channels = channels;
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn scene_target(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let Some(target) = operation.get("target") else {
        return Ok(engine.doc.active_scene.clone());
    };
    super::target_id_value(
        engine,
        Some(target),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )
}

fn validate_strip(engine: &Engine<'_>, strip: &Strip) -> Result<()> {
    strip.validate().map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, ""),
        )
    })
}

fn validate_input_strips(
    engine: &Engine<'_>,
    scene_id: &Id,
    strip: &Strip,
    path: &str,
) -> Result<()> {
    let scene = engine.doc.scenes.get(scene_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "sequencer scene does not exist",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    for (index, input) in strip.inputs.iter().enumerate() {
        if !scene
            .sequencer
            .strips
            .iter()
            .any(|source| source.id == input.as_str())
        {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                "sequencer input strip does not exist",
                &operation_pointer(engine.operation_index, &format!("{path}/{index}")),
            ));
        }
    }
    let mut pending: Vec<&str> = strip.inputs.iter().map(String::as_str).collect();
    let mut visited = BTreeSet::new();
    while let Some(input_id) = pending.pop() {
        if input_id == strip.id.as_str() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "sequencer strip inputs form a dependency cycle",
                &operation_pointer(engine.operation_index, path),
            ));
        }
        if !visited.insert(input_id) {
            continue;
        }
        if let Some(input_strip) = scene
            .sequencer
            .strips
            .iter()
            .find(|source| source.id == input_id)
        {
            pending.extend(input_strip.inputs.iter().map(String::as_str));
        }
    }
    Ok(())
}

fn sort_strips(sequencer: &mut Sequencer) {
    sequencer.strips.sort_by(|first, second| {
        first
            .channel
            .cmp(&second.channel)
            .then_with(|| first.frame_start.total_cmp(&second.frame_start))
            .then_with(|| first.id.cmp(&second.id))
    });
}

const STRIP_FIELDS: &[&str] = &[
    "op",
    "target",
    "id",
    "name",
    "type",
    "channel",
    "frame_start",
    "frame_offset_start",
    "frame_offset_end",
    "length",
    "blend_type",
    "opacity",
    "mute",
    "retiming_keys",
    "modifiers",
    "sound_volume",
    "sound_pan",
    "sound_pitch",
    "source",
    "color",
    "text",
    "transition",
    "inputs",
    "effect",
];
const STRIP_SET_FIELDS: &[&str] = &[
    "type",
    "channel",
    "name",
    "frame_start",
    "frame_offset_start",
    "frame_offset_end",
    "length",
    "blend_type",
    "opacity",
    "mute",
    "retiming_keys",
    "modifiers",
    "sound_volume",
    "sound_pan",
    "sound_pitch",
    "source",
    "color",
    "text",
    "transition",
    "inputs",
    "effect",
];
