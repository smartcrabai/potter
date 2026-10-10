use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{ActionSlot, Id, NlaBlendType, NlaExtrapolation, NlaStrip, NlaTrack, TimelineMarker},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_bool, read_id, read_string,
};

const TARGET_ID_POLICY: super::TargetIdPolicy = super::TargetIdPolicy::NamedField("target");

const TRACK_SET_FIELDS: &[&str] = &["name", "mute", "solo"];
const STRIP_FIELDS: &[&str] = &[
    "id",
    "action",
    "frame_start",
    "frame_end",
    "action_frame_start",
    "action_frame_end",
    "scale",
    "repeat",
    "blend_type",
    "influence",
    "extrapolation",
    "blend_in",
    "blend_out",
];

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "nla.track_create" => create_track(engine, operation),
        "nla.track_update" => update_track(engine, operation),
        "nla.track_delete" => delete_track(engine, operation),
        "nla.strip_create" => create_strip(engine, operation),
        "nla.strip_update" => update_strip(engine, operation),
        "nla.strip_delete" => delete_strip(engine, operation),
        "nla.push_down" => push_down(engine, operation),
        "action.slot_create" => create_action_slot(engine, operation),
        "scene.marker_add" => add_marker(engine, operation),
        "scene.marker_remove" => remove_marker(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid NLA operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}
pub(super) fn action_is_referenced(engine: &Engine<'_>, action_id: &Id) -> bool {
    engine
        .doc
        .actions
        .get(action_id)
        .is_some_and(|action| !action.slots.is_empty())
        || engine.doc.nodes.values().any(|node| {
            node.nla_tracks
                .iter()
                .any(|track| track.strips.iter().any(|strip| &strip.action == action_id))
        })
}

fn create_track(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "name", "mute", "solo"],
        &["target"],
    )?;
    let node_id = node_target(engine, operation)?;
    let id = match operation.get("id") {
        Some(_) => read_id(engine, operation, "id")?,
        None => unique_local_id(&format!("{node_id}_nla_track"), |candidate| {
            engine
                .doc
                .nodes
                .get(&node_id)
                .is_some_and(|node| node.nla_tracks.iter().any(|track| &track.id == candidate))
        })?,
    };
    if engine
        .doc
        .nodes
        .get(&node_id)
        .is_some_and(|node| node.nla_tracks.iter().any(|track| track.id == id))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("NLA track ID `{id}` already exists on node `{node_id}`"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let name = match operation.get("name") {
        Some(_) => read_string(engine, operation, "name")?,
        None => id.to_string(),
    };
    let mute = read_bool(engine, operation, "mute", false)?;
    let solo = read_bool(engine, operation, "solo", false)?;
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    node.nla_tracks.push(NlaTrack {
        id,
        name,
        mute,
        solo,
        strips: Vec::new(),
    });
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn update_track(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "track", "set"],
        &["target", "track", "set"],
    )?;
    let node_id = node_target(engine, operation)?;
    let track_id = read_id(engine, operation, "track")?;
    let set = set_object(engine, operation)?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "track set cannot be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    reject_fields(engine, set, TRACK_SET_FIELDS, "set")?;
    let name = set
        .get("name")
        .map(|value| string_value(engine, value, "set/name"))
        .transpose()?;
    let mute = set
        .get("mute")
        .map(|value| bool_value(engine, value, "set/mute"))
        .transpose()?;
    let solo = set
        .get("solo")
        .map(|value| bool_value(engine, value, "set/solo"))
        .transpose()?;
    let index = track_index(engine, &node_id, &track_id)?;
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    let track = node.nla_tracks.get_mut(index).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA track disappeared")
    })?;
    let before = track.clone();
    if let Some(name) = name {
        track.name = name;
    }
    if let Some(mute) = mute {
        track.mute = mute;
    }
    if let Some(solo) = solo {
        track.solo = solo;
    }
    let changed = *track != before;
    if changed {
        engine.mark("nodes", &node_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn delete_track(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "track"],
        &["target", "track"],
    )?;
    let node_id = node_target(engine, operation)?;
    let track_id = read_id(engine, operation, "track")?;
    let index = track_index(engine, &node_id, &track_id)?;
    if engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.nla_tracks.get(index))
        .is_some_and(|track| !track.strips.is_empty())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("NLA track `{track_id}` is not empty"),
            &operation_pointer(engine.operation_index, "track"),
        ));
    }
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    node.nla_tracks.remove(index);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn create_strip(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "track",
            "id",
            "action",
            "frame_start",
            "frame_end",
            "action_frame_start",
            "action_frame_end",
            "scale",
            "repeat",
            "blend_type",
            "influence",
            "extrapolation",
            "blend_in",
            "blend_out",
        ],
        &[
            "target",
            "track",
            "action",
            "frame_start",
            "frame_end",
            "action_frame_start",
            "action_frame_end",
            "scale",
            "repeat",
            "blend_type",
            "influence",
            "extrapolation",
            "blend_in",
            "blend_out",
        ],
    )?;
    let node_id = node_target(engine, operation)?;
    let track_id = read_id(engine, operation, "track")?;
    let track_index = track_index(engine, &node_id, &track_id)?;
    let action_id = read_id(engine, operation, "action")?;
    require_action(engine, &action_id, "action")?;
    let id = match operation.get("id") {
        Some(_) => read_id(engine, operation, "id")?,
        None => unique_local_id(&format!("{track_id}_strip"), |candidate| {
            engine
                .doc
                .nodes
                .get(&node_id)
                .and_then(|node| node.nla_tracks.get(track_index))
                .is_some_and(|track| track.strips.iter().any(|strip| &strip.id == candidate))
        })?,
    };
    if engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.nla_tracks.get(track_index))
        .is_some_and(|track| track.strips.iter().any(|strip| strip.id == id))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("NLA strip ID `{id}` already exists in track `{track_id}`"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mut strip = NlaStrip {
        id,
        action: action_id,
        frame_start: number_value(
            engine,
            required_value(engine, operation, "frame_start")?,
            "frame_start",
        )?,
        frame_end: number_value(
            engine,
            required_value(engine, operation, "frame_end")?,
            "frame_end",
        )?,
        action_frame_start: number_value(
            engine,
            required_value(engine, operation, "action_frame_start")?,
            "action_frame_start",
        )?,
        action_frame_end: number_value(
            engine,
            required_value(engine, operation, "action_frame_end")?,
            "action_frame_end",
        )?,
        scale: 1.0,
        repeat: 1.0,
        blend_type: NlaBlendType::Replace,
        influence: 1.0,
        extrapolation: NlaExtrapolation::Hold,
        blend_in: 0.0,
        blend_out: 0.0,
    };
    apply_strip_values(engine, &mut strip, operation, "")?;
    validate_strip(engine, &strip, "")?;
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    let track = node.nla_tracks.get_mut(track_index).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA track disappeared")
    })?;
    track.strips.push(strip);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn update_strip(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "track", "strip", "set"],
        &["target", "track", "strip", "set"],
    )?;
    let node_id = node_target(engine, operation)?;
    let track_id = read_id(engine, operation, "track")?;
    let strip_id = read_id(engine, operation, "strip")?;
    let track_index = track_index(engine, &node_id, &track_id)?;
    let strip_index = strip_index(engine, &node_id, track_index, &strip_id)?;
    let set = set_object(engine, operation)?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "strip set cannot be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    reject_fields(engine, set, STRIP_FIELDS, "set")?;
    let mut updated = engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.nla_tracks.get(track_index))
        .and_then(|track| track.strips.get(strip_index))
        .cloned()
        .ok_or_else(|| {
            crate::error::PotError::new(ErrorCode::InternalError, "NLA strip disappeared")
        })?;
    apply_strip_values(engine, &mut updated, set, "set/")?;
    validate_strip(engine, &updated, "set/")?;
    if updated.id != strip_id
        && engine
            .doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.nla_tracks.get(track_index))
            .is_some_and(|track| {
                track
                    .strips
                    .iter()
                    .enumerate()
                    .any(|(index, strip)| index != strip_index && strip.id == updated.id)
            })
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!(
                "NLA strip ID `{}` already exists in track `{track_id}`",
                updated.id
            ),
            &operation_pointer(engine.operation_index, "set/id"),
        ));
    }
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    let strip = node
        .nla_tracks
        .get_mut(track_index)
        .and_then(|track| track.strips.get_mut(strip_index))
        .ok_or_else(|| {
            crate::error::PotError::new(ErrorCode::InternalError, "NLA strip disappeared")
        })?;
    let changed = *strip != updated;
    if changed {
        *strip = updated;
        engine.mark("nodes", &node_id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn delete_strip(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "track", "strip"],
        &["target", "track", "strip"],
    )?;
    let node_id = node_target(engine, operation)?;
    let track_id = read_id(engine, operation, "track")?;
    let strip_id = read_id(engine, operation, "strip")?;
    let track_index = track_index(engine, &node_id, &track_id)?;
    let strip_index = strip_index(engine, &node_id, track_index, &strip_id)?;
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    node.nla_tracks
        .get_mut(track_index)
        .ok_or_else(|| {
            crate::error::PotError::new(ErrorCode::InternalError, "NLA track disappeared")
        })?
        .strips
        .remove(strip_index);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn push_down(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target"], &["target"])?;
    let node_id = node_target(engine, operation)?;
    let action_id = engine
        .doc
        .nodes
        .get(&node_id)
        .and_then(|node| node.action.clone())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "node must have an active action to push down",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    let action = engine.doc.actions.get(&action_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("action `{action_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut start = f64::INFINITY;
    let mut end = f64::NEG_INFINITY;
    for key in action.fcurves.iter().flat_map(|curve| &curve.keyframes) {
        if !key.frame.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "action keyframe frames must be finite",
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        start = start.min(key.frame);
        end = end.max(key.frame);
    }
    if !start.is_finite() || !end.is_finite() || start >= end {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "active action must have keyframes spanning a nondegenerate range",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let track_id = unique_local_id(&format!("nla_{node_id}_{action_id}_track"), |candidate| {
        engine
            .doc
            .nodes
            .get(&node_id)
            .is_some_and(|node| node.nla_tracks.iter().any(|track| &track.id == candidate))
    })?;
    let strip_id = unique_local_id(&format!("nla_{node_id}_{action_id}_strip"), |candidate| {
        engine.doc.nodes.get(&node_id).is_some_and(|node| {
            node.nla_tracks
                .iter()
                .any(|track| track.strips.iter().any(|strip| &strip.id == candidate))
        })
    })?;
    let strip = NlaStrip {
        id: strip_id,
        action: action_id,
        frame_start: start,
        frame_end: end,
        action_frame_start: start,
        action_frame_end: end,
        scale: 1.0,
        repeat: 1.0,
        blend_type: NlaBlendType::Replace,
        influence: 1.0,
        extrapolation: NlaExtrapolation::Hold,
        blend_in: 0.0,
        blend_out: 0.0,
    };
    validate_strip(engine, &strip, "")?;
    let name = engine
        .doc
        .actions
        .get(&strip.action)
        .map_or_else(|| strip.action.to_string(), |action| action.name.clone());
    let node = engine.doc.nodes.get_mut(&node_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "NLA node disappeared")
    })?;
    node.nla_tracks.push(NlaTrack {
        id: track_id,
        name,
        mute: false,
        solo: false,
        strips: vec![strip],
    });
    node.action = None;
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    Ok(true)
}

fn create_action_slot(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "node"],
        &["target", "id", "node"],
    )?;
    let action_id = target_id(engine, operation, "actions")?;
    let slot_id = read_id(engine, operation, "id")?;
    let node_id = read_id(engine, operation, "node")?;
    if !engine.doc.nodes.contains_key(&node_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "node"),
        ));
    }
    if engine
        .doc
        .actions
        .get(&action_id)
        .is_some_and(|action| action.slots.iter().any(|slot| slot.id == slot_id))
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("action slot ID `{slot_id}` already exists on action `{action_id}`"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let action = engine.doc.actions.get_mut(&action_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "action disappeared")
    })?;
    action.slots.push(ActionSlot {
        id: slot_id,
        node: node_id,
    });
    engine.mark("actions", &action_id, ChangeKind::Updated);
    Ok(true)
}

fn add_marker(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "scene", "id", "frame", "name"],
        &["id", "frame"],
    )?;
    let scene_id = scene_id(engine, operation)?;
    let id = read_id(engine, operation, "id")?;
    let frame = number_value(engine, required_value(engine, operation, "frame")?, "frame")?;
    let name = match operation.get("name") {
        Some(_) => read_string(engine, operation, "name")?,
        None => id.to_string(),
    };
    let scene = engine.doc.scenes.get(&scene_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("scene `{scene_id}` was not found"),
            &operation_pointer(engine.operation_index, "scene"),
        )
    })?;
    if scene.markers.iter().any(|marker| marker.id == id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("marker ID `{id}` already exists in scene `{scene_id}`"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "scene disappeared")
    })?;
    scene.markers.push(TimelineMarker { id, name, frame });
    engine.mark("scenes", &scene_id, ChangeKind::Updated);
    Ok(true)
}

fn remove_marker(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "scene", "target", "id"], &[])?;
    if operation.contains_key("target") == operation.contains_key("id") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "provide exactly one of target or id",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let scene_id = scene_id(engine, operation)?;
    let marker_id = if operation.contains_key("target") {
        super::target_id_value(
            engine,
            operation.get("target"),
            &operation_pointer(engine.operation_index, "target"),
            TARGET_ID_POLICY,
        )?
    } else {
        read_id(engine, operation, "id")?
    };
    let index = engine
        .doc
        .scenes
        .get(&scene_id)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("scene `{scene_id}` was not found"),
                &operation_pointer(engine.operation_index, "scene"),
            )
        })?
        .markers
        .iter()
        .position(|marker| marker.id == marker_id)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("marker `{marker_id}` was not found in scene `{scene_id}`"),
                &operation_pointer(
                    engine.operation_index,
                    if operation.contains_key("target") {
                        "target/id"
                    } else {
                        "id"
                    },
                ),
            )
        })?;
    let scene = engine.doc.scenes.get_mut(&scene_id).ok_or_else(|| {
        crate::error::PotError::new(ErrorCode::InternalError, "scene disappeared")
    })?;
    scene.markers.remove(index);
    engine.mark("scenes", &scene_id, ChangeKind::Updated);
    Ok(true)
}

fn node_target(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )?;
    if !engine.doc.nodes.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok(id)
}

fn target_id(engine: &Engine<'_>, operation: &Map<String, Value>, registry: &str) -> Result<Id> {
    let id = super::target_id_value(
        engine,
        operation.get("target"),
        &operation_pointer(engine.operation_index, "target"),
        TARGET_ID_POLICY,
    )?;
    let found = match registry {
        "actions" => engine.doc.actions.contains_key(&id),
        _ => false,
    };
    if !found {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("{registry} target `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok(id)
}

fn scene_id(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    match operation.get("scene") {
        Some(_) => read_id(engine, operation, "scene"),
        None => Ok(engine.doc.active_scene.clone()),
    }
}

fn track_index(engine: &Engine<'_>, node_id: &Id, track_id: &Id) -> Result<usize> {
    engine
        .doc
        .nodes
        .get(node_id)
        .and_then(|node| {
            node.nla_tracks
                .iter()
                .position(|track| &track.id == track_id)
        })
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("NLA track `{track_id}` was not found on node `{node_id}`"),
                &operation_pointer(engine.operation_index, "track"),
            )
        })
}

fn strip_index(
    engine: &Engine<'_>,
    node_id: &Id,
    track_index: usize,
    strip_id: &Id,
) -> Result<usize> {
    engine
        .doc
        .nodes
        .get(node_id)
        .and_then(|node| node.nla_tracks.get(track_index))
        .and_then(|track| track.strips.iter().position(|strip| &strip.id == strip_id))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("NLA strip `{strip_id}` was not found"),
                &operation_pointer(engine.operation_index, "strip"),
            )
        })
}

fn require_action(engine: &Engine<'_>, action_id: &Id, field: &str) -> Result<()> {
    if !engine.doc.actions.contains_key(action_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("action `{action_id}` was not found"),
            &operation_pointer(engine.operation_index, field),
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
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })
}

fn reject_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown {prefix} field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("{prefix}/{}", key.replace('~', "~0").replace('/', "~1")),
                ),
            ));
        }
    }
    Ok(())
}

fn required_value<'a>(
    engine: &Engine<'_>,
    object: &'a Map<String, Value>,
    key: &str,
) -> Result<&'a Value> {
    object.get(key).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("missing required field `{key}`"),
            &operation_pointer(engine.operation_index, key),
        )
    })
}

fn string_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<String> {
    value.as_str().map(str::to_owned).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a string"),
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn bool_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a boolean"),
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn number_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a finite number"),
                &operation_pointer(engine.operation_index, field),
            )
        })
}

fn apply_strip_values(
    engine: &Engine<'_>,
    strip: &mut NlaStrip,
    values: &Map<String, Value>,
    prefix: &str,
) -> Result<()> {
    let field = |name: &str| format!("{prefix}{name}");
    if let Some(value) = values.get("id") {
        let name = field("id");
        let text = string_value(engine, value, &name)?;
        strip.id = parse_id(
            engine,
            &text,
            &operation_pointer(engine.operation_index, &name),
        )?;
    }
    if let Some(value) = values.get("action") {
        let name = field("action");
        let text = string_value(engine, value, &name)?;
        strip.action = parse_id(
            engine,
            &text,
            &operation_pointer(engine.operation_index, &name),
        )?;
        require_action(engine, &strip.action, &name)?;
    }
    for name in [
        "frame_start",
        "frame_end",
        "action_frame_start",
        "action_frame_end",
        "scale",
        "repeat",
        "influence",
        "blend_in",
        "blend_out",
    ] {
        if let Some(value) = values.get(name) {
            let full_name = field(name);
            let number = number_value(engine, value, &full_name)?;
            match name {
                "frame_start" => strip.frame_start = number,
                "frame_end" => strip.frame_end = number,
                "action_frame_start" => strip.action_frame_start = number,
                "action_frame_end" => strip.action_frame_end = number,
                "scale" => strip.scale = number,
                "repeat" => strip.repeat = number,
                "influence" => strip.influence = number,
                "blend_in" => strip.blend_in = number,
                "blend_out" => strip.blend_out = number,
                _ => unreachable!(),
            }
        }
    }
    if let Some(value) = values.get("blend_type") {
        let name = field("blend_type");
        strip.blend_type = match value.as_str() {
            Some("replace") => NlaBlendType::Replace,
            Some("add") => NlaBlendType::Add,
            Some("combine") => NlaBlendType::Combine,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "blend_type must be replace, add, or combine",
                    &operation_pointer(engine.operation_index, &name),
                ));
            }
        };
    }
    if let Some(value) = values.get("extrapolation") {
        let name = field("extrapolation");
        strip.extrapolation = match value.as_str() {
            Some("hold") => NlaExtrapolation::Hold,
            Some("hold_forward") => NlaExtrapolation::HoldForward,
            Some("nothing") => NlaExtrapolation::Nothing,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "extrapolation must be hold, hold_forward, or nothing",
                    &operation_pointer(engine.operation_index, &name),
                ));
            }
        };
    }
    Ok(())
}

fn validate_strip(engine: &Engine<'_>, strip: &NlaStrip, prefix: &str) -> Result<()> {
    require_action(engine, &strip.action, &format!("{prefix}action"))?;
    let duration = strip.frame_end - strip.frame_start;
    let action_duration = strip.action_frame_end - strip.action_frame_start;
    if !duration.is_finite() || duration <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "frame_end must exceed frame_start by a finite amount",
            &operation_pointer(engine.operation_index, &format!("{prefix}frame_end")),
        ));
    }
    if !action_duration.is_finite() || action_duration <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "action_frame_end must exceed action_frame_start by a finite amount",
            &operation_pointer(engine.operation_index, &format!("{prefix}action_frame_end")),
        ));
    }
    if !strip.scale.is_finite() || strip.scale <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "scale must be positive and finite",
            &operation_pointer(engine.operation_index, &format!("{prefix}scale")),
        ));
    }
    if !strip.repeat.is_finite() || strip.repeat <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "repeat must be positive and finite",
            &operation_pointer(engine.operation_index, &format!("{prefix}repeat")),
        ));
    }
    if !strip.influence.is_finite() || !(0.0..=1.0).contains(&strip.influence) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "influence must be between 0 and 1",
            &operation_pointer(engine.operation_index, &format!("{prefix}influence")),
        ));
    }
    if !strip.blend_in.is_finite() || strip.blend_in < 0.0 || strip.blend_in > duration {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "blend_in must be nonnegative and no longer than the strip",
            &operation_pointer(engine.operation_index, &format!("{prefix}blend_in")),
        ));
    }
    if !strip.blend_out.is_finite()
        || strip.blend_out < 0.0
        || strip.blend_out > duration
        || strip.blend_in + strip.blend_out > duration
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "blend_out must be nonnegative and fades must fit within the strip",
            &operation_pointer(engine.operation_index, &format!("{prefix}blend_out")),
        ));
    }
    Ok(())
}

fn unique_local_id(base: &str, mut exists: impl FnMut(&Id) -> bool) -> Result<Id> {
    for suffix in 0_u32.. {
        let tail = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let prefix = base
            .chars()
            .take(64_usize.saturating_sub(tail.len()))
            .collect::<String>();
        let candidate = format!("{prefix}{tail}");
        if let Ok(id) = Id::new(candidate)
            && !exists(&id)
        {
            return Ok(id);
        }
    }
    Err(crate::error::PotError::new(
        ErrorCode::InvalidOperation,
        "could not generate a unique ID",
    ))
}
