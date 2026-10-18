use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path};

use glam::{DQuat, EulerRot};
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{Snapshot, animation::animated_pose_bones},
    model::{
        Action, ArmatureData, Bone, DataBlock, Extrapolation, FCurve, Id, Interpolation, Keyframe,
        Node, Registry, SceneDoc,
    },
};

use super::{ImportedGraph, import_error};

const MAX_BVH_FRAMES: usize = 1_000_000;
const BLENDER_TO_BVH: DQuat = DQuat::from_xyzw(
    std::f64::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    std::f64::consts::FRAC_1_SQRT_2,
);

#[derive(Clone)]
struct JointExport {
    name: String,
    bone: Bone,
    children: Vec<Id>,
}

/// Export armature rest bones and sampled pose channels as BVH.
pub(crate) fn export(doc: &SceneDoc, _snapshot: &Snapshot) -> Result<Vec<u8>> {
    let armatures = doc
        .nodes
        .iter()
        .filter(|(_, node)| node.kind == "armature")
        .collect::<Vec<_>>();
    if armatures.len() != 1 {
        return Err(unsupported(
            "BVH currently requires exactly one armature object".to_owned(),
        ));
    }
    let (armature_id, node) = armatures[0];
    let data_id = node
        .data
        .as_ref()
        .ok_or_else(|| unsupported("armature data is missing".to_owned()))?;
    let armature = doc
        .data_blocks
        .get(data_id)
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| unsupported("armature data is missing".to_owned()))?;
    if armature.bones.is_empty() {
        return Err(unsupported("BVH requires at least one bone".to_owned()));
    }
    let scene = doc
        .scenes
        .get(&doc.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "active scene does not exist"))?;
    let frame_count = frame_count(scene.frame_start, scene.frame_end)?;
    let frame_time = scene.fps_base / f64::from(scene.fps);
    if !frame_time.is_finite() || frame_time <= 0.0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "scene frame rate must be positive",
        ));
    }
    let mut joints = BTreeMap::<Id, JointExport>::new();
    let mut names = BTreeMap::<String, Id>::new();
    for (id, bone) in &armature.bones {
        let unique_name = unique_joint_name(&bone.name, &names);
        names.insert(unique_name.clone(), id.clone());
        joints.insert(
            id.clone(),
            JointExport {
                name: unique_name,
                bone: bone.clone(),
                children: Vec::new(),
            },
        );
    }
    let ids = joints.keys().cloned().collect::<Vec<_>>();
    let mut roots = Vec::new();
    for id in ids {
        let parent = joints.get(&id).and_then(|joint| joint.bone.parent.clone());
        if let Some(parent) = parent {
            if !joints.contains_key(&parent) {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "armature bone references a missing parent",
                    json!({"armature": armature_id, "bone": id, "parent": parent}),
                ));
            }
            let parent_joint = joints.get_mut(&parent).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "validated parent joint is missing",
                )
            })?;
            parent_joint.children.push(id);
        } else {
            roots.push(id);
        }
    }
    if roots.is_empty() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "armature bone hierarchy has no root",
        ));
    }
    if roots.len() != 1 {
        return Err(unsupported("BVH requires exactly one root bone".to_owned()));
    }
    let mut output = String::from("HIERARCHY\n");
    for id in &roots {
        write_joint(&mut output, &joints, id, None, 0)?;
    }
    output.push_str("MOTION\n");
    write!(
        &mut output,
        "Frames: {frame_count}\nFrame Time: {}\n",
        number(frame_time)?
    )
    .map_err(|_| PotError::new(ErrorCode::InternalError, "failed to format BVH output"))?;
    let ordered = roots
        .iter()
        .flat_map(|root| preorder(&joints, root))
        .collect::<Vec<_>>();
    if ordered.len() != joints.len() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "armature bone hierarchy contains an unreachable cycle",
        ));
    }
    for frame_offset in 0..frame_count {
        let frame_offset = u32::try_from(frame_offset)
            .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "BVH frame index is too large"))?;
        let frame = f64::from(scene.frame_start) + f64::from(frame_offset);
        let poses = animated_pose_bones(node, doc, frame)?;
        let mut values = Vec::with_capacity(ordered.len() * 3 + roots.len() * 3);
        for joint_id in &ordered {
            let joint = joints.get(joint_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "BVH output joint is missing")
            })?;
            let pose = poses.get(joint_id).copied().unwrap_or_default();
            if joint.bone.parent.is_none() {
                let translated = blender_to_bvh_vec(pose.translation);
                values.extend(translated);
            }
            let rotation = blender_to_bvh_rotation(pose.rotation)?;
            let (z, x, y) = rotation.to_euler(EulerRot::ZXY);
            values.extend([z.to_degrees(), x.to_degrees(), y.to_degrees()]);
        }
        let line = values
            .into_iter()
            .map(number)
            .collect::<Result<Vec<_>>>()?
            .join(" ");
        output.push_str(&line);
        output.push('\n');
    }
    Ok(output.into_bytes())
}

/// Import BVH hierarchy, offsets, channel order, frame rate, and sampled motion into an
/// armature object with a sampled Action.
pub(crate) fn import(file: &Path, scene_id: String) -> Result<ImportedGraph> {
    let text = fs::read_to_string(file).map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            format!("BVH file is not readable text: {error}"),
            json!({"path": file}),
        )
    })?;
    let parsed = parse_bvh(&text)?;
    let mut doc = SceneDoc::new(scene_id);
    let node_id = Id::new("armature")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let data_id = Id::new("armature_data")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let action_id = Id::new("bvh_motion")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let mut armature = ArmatureData::default();
    let mut joint_ids = BTreeMap::<String, Id>::new();
    let mut id_counts = BTreeMap::<String, usize>::new();
    for root in &parsed.roots {
        collect_bones(
            root,
            [0.0; 3],
            None,
            &mut armature.bones,
            &mut joint_ids,
            &mut id_counts,
        )?;
    }
    let mut node = Node {
        name: "BVH Armature".to_owned(),
        kind: "armature".to_owned(),
        data: Some(data_id.clone()),
        ..Node::default()
    };
    let mut curves = BTreeMap::<(String, String, u32), Vec<Keyframe>>::new();
    for (frame_index, frame_values) in parsed.frames.iter().enumerate() {
        let mut cursor = 0_usize;
        let frame_number = u32::try_from(frame_index)
            .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "BVH frame index is too large"))?;
        let frame = f64::from(frame_number) + 1.0;
        for root in &parsed.roots {
            collect_frame_curves(
                root,
                &joint_ids,
                frame_values,
                &mut cursor,
                frame,
                &mut curves,
            )?;
        }
        if cursor != frame_values.len() {
            return Err(import_error(
                "BVH frame has an unexpected number of channels",
            ));
        }
    }
    let fcurves = curves
        .into_iter()
        .map(|((bone, channel, index), keyframes)| FCurve {
            path: format!("pose.bones[\"{bone}\"].{channel}"),
            index,
            keyframes,
            extrapolation: Extrapolation::Constant,
        })
        .collect::<Vec<_>>();
    if !fcurves.is_empty() {
        node.action = Some(action_id.clone());
        doc.actions.insert(
            action_id,
            Action {
                name: "BVH Motion".to_owned(),
                fcurves,
                slots: Vec::new(),
            },
        );
    }
    let data = DataBlock {
        data_type: "armature".to_owned(),
        mesh: None,
        armature: Some(armature),
        ..DataBlock::default()
    };
    doc.data_blocks.insert(data_id, data);
    doc.nodes.insert(node_id.clone(), node);
    let root_collection = doc
        .collections
        .get_mut(&Id::from_static("collection_root"))
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "default collection is missing"))?;
    root_collection.objects.push(node_id);
    let scene = doc
        .scenes
        .get_mut(&doc.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "default scene is missing"))?;
    scene.frame_end = i32::try_from(parsed.frames.len().max(1))
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "BVH frame range is too large"))?;
    let rate = 1.0 / parsed.frame_time;
    if !rate.is_finite() || rate <= 0.0 || rate > f64::from(u32::MAX) {
        return Err(import_error(
            "BVH frame time is outside the supported range",
        ));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the positive frame rate is rounded and clamped to the u32 range"
    )]
    let fps = rate.round().clamp(1.0, f64::from(u32::MAX)) as u32;
    scene.fps = fps;
    scene.fps_base = f64::from(fps) / rate;
    doc.validate()?;
    Ok(ImportedGraph {
        doc,
        losses: Vec::new(),
        id_mappings: json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source: json!({"format":"bvh", "fps":rate, "frames":parsed.frames.len()}),
    })
}

#[derive(Clone, Debug)]
struct BvhJoint {
    name: String,
    offset: [f64; 3],
    channels: Vec<String>,
    children: Vec<BvhJoint>,
    end_site: Option<[f64; 3]>,
}

#[derive(Clone, Debug)]
struct BvhFile {
    roots: Vec<BvhJoint>,
    frames: Vec<Vec<f64>>,
    frame_time: f64,
}

fn parse_bvh(text: &str) -> Result<BvhFile> {
    let tokens = tokenize(text);
    let mut cursor = 0;
    expect(&tokens, &mut cursor, "HIERARCHY")?;
    let mut roots = Vec::new();
    while peek(&tokens, cursor).is_some_and(|token| token == "ROOT") {
        roots.push(parse_joint(&tokens, &mut cursor, true)?);
    }
    if roots.is_empty() {
        return Err(import_error("BVH hierarchy has no ROOT joint"));
    }
    expect(&tokens, &mut cursor, "MOTION")?;
    expect(&tokens, &mut cursor, "Frames:")?;
    let frame_count = parse_usize(next(&tokens, &mut cursor, "frame count")?, "frame count")?;
    if frame_count > MAX_BVH_FRAMES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "BVH frame count exceeds the limit",
        ));
    }
    expect(&tokens, &mut cursor, "Frame")?;
    expect(&tokens, &mut cursor, "Time:")?;
    let frame_time = parse_number(next(&tokens, &mut cursor, "frame time")?, "frame time")?;
    if frame_time <= 0.0 {
        return Err(import_error("BVH frame time must be positive"));
    }
    let channel_count = roots.iter().map(joint_channel_count).sum::<usize>();
    let expected = frame_count.checked_mul(channel_count).ok_or_else(|| {
        PotError::new(ErrorCode::LimitExceeded, "BVH motion sample count overflow")
    })?;
    if tokens.len().saturating_sub(cursor) != expected {
        return Err(import_error(
            "BVH motion sample count does not match its hierarchy",
        ));
    }
    let mut frames = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        let mut values = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            values.push(parse_number(
                next(&tokens, &mut cursor, "motion channel")?,
                "motion channel",
            )?);
        }
        frames.push(values);
    }
    Ok(BvhFile {
        roots,
        frames,
        frame_time,
    })
}

fn parse_joint(tokens: &[String], cursor: &mut usize, root: bool) -> Result<BvhJoint> {
    expect(tokens, cursor, if root { "ROOT" } else { "JOINT" })?;
    let name = next(tokens, cursor, "joint name")?.to_owned();
    expect(tokens, cursor, "{")?;
    expect(tokens, cursor, "OFFSET")?;
    let offset = read_vec3(tokens, cursor, "joint offset")?;
    expect(tokens, cursor, "CHANNELS")?;
    let channel_count = parse_usize(next(tokens, cursor, "channel count")?, "channel count")?;
    if channel_count > 6 {
        return Err(import_error("BVH joint has more than six channels"));
    }
    let mut channels = Vec::with_capacity(channel_count);
    for _ in 0..channel_count {
        channels.push(next(tokens, cursor, "channel name")?.to_owned());
    }
    let mut children = Vec::new();
    let mut end_site = None;
    loop {
        match peek(tokens, *cursor) {
            Some("JOINT") => children.push(parse_joint(tokens, cursor, false)?),
            Some("End") => {
                expect(tokens, cursor, "End")?;
                expect(tokens, cursor, "Site")?;
                expect(tokens, cursor, "{")?;
                expect(tokens, cursor, "OFFSET")?;
                end_site = Some(read_vec3(tokens, cursor, "end site offset")?);
                expect(tokens, cursor, "}")?;
            }
            Some("}") => {
                expect(tokens, cursor, "}")?;
                break;
            }
            _ => return Err(import_error("malformed BVH joint hierarchy")),
        }
    }
    Ok(BvhJoint {
        name,
        offset,
        channels,
        children,
        end_site,
    })
}

fn collect_bones(
    joint: &BvhJoint,
    parent_head: [f64; 3],
    parent: Option<Id>,
    bones: &mut Registry<Bone>,
    ids: &mut BTreeMap<String, Id>,
    id_counts: &mut BTreeMap<String, usize>,
) -> Result<()> {
    let id_seed = bone_id_seed(&joint.name);
    let ordinal = id_counts.entry(id_seed.clone()).or_default();
    let id_text = if *ordinal == 0 {
        id_seed
    } else {
        format!("{id_seed}_{}", *ordinal)
    };
    *ordinal += 1;
    let id = Id::new(id_text).map_err(|error| import_error(error.to_string()))?;
    let head_bvh = add(parent_head, joint.offset);
    let tail_bvh = joint
        .end_site
        .or_else(|| joint.children.first().map(|child| child.offset))
        .unwrap_or([0.0, 0.1, 0.0]);
    let tail_bvh = add(head_bvh, tail_bvh);
    let head = bvh_to_blender_vec(head_bvh);
    let tail = bvh_to_blender_vec(tail_bvh);
    let bone = Bone {
        name: joint.name.clone(),
        parent,
        head,
        tail,
        roll: 0.0,
        deform: true,
        inherit_rotation: true,
        use_connect: false,
        custom_shape: None,
        envelope_distance: 0.25,
        envelope_weight: 1.0,
        head_radius: 0.1,
        tail_radius: 0.1,
        bbone_settings: BTreeMap::new(),
    };
    bones.insert(id.clone(), bone);
    ids.insert(joint.name.clone(), id.clone());
    for child in &joint.children {
        collect_bones(child, head_bvh, Some(id.clone()), bones, ids, id_counts)?;
    }
    Ok(())
}

fn collect_frame_curves(
    joint: &BvhJoint,
    ids: &BTreeMap<String, Id>,
    values: &[f64],
    cursor: &mut usize,
    frame: f64,
    curves: &mut BTreeMap<(String, String, u32), Vec<Keyframe>>,
) -> Result<()> {
    let id = ids
        .get(&joint.name)
        .ok_or_else(|| import_error("BVH motion references an unknown joint"))?;
    let mut translation = [0.0; 3];
    let mut rotations = DQuat::IDENTITY;
    for channel in &joint.channels {
        let value = *values
            .get(*cursor)
            .ok_or_else(|| import_error("BVH motion channel is missing"))?;
        *cursor = cursor
            .checked_add(1)
            .ok_or_else(|| import_error("BVH channel index overflow"))?;
        match channel.as_str() {
            "Xposition" => translation[0] = value,
            "Yposition" => translation[1] = value,
            "Zposition" => translation[2] = value,
            "Xrotation" => rotations *= DQuat::from_rotation_x(value.to_radians()),
            "Yrotation" => rotations *= DQuat::from_rotation_y(value.to_radians()),
            "Zrotation" => rotations *= DQuat::from_rotation_z(value.to_radians()),
            _ => return Err(import_error("BVH uses an unsupported joint channel")),
        }
    }
    let blender_rotation = (BLENDER_TO_BVH * rotations * BLENDER_TO_BVH.inverse()).normalize();
    let quaternion = blender_rotation.to_array();
    let id_string = id.as_str().to_owned();
    for (index, value) in quaternion.into_iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| import_error("BVH rotation component index is out of range"))?;
        curves
            .entry((id_string.clone(), "rotation".to_owned(), index))
            .or_default()
            .push(keyframe(frame, value));
    }
    if joint
        .channels
        .iter()
        .any(|channel| channel.ends_with("position"))
    {
        let converted = bvh_to_blender_vec(translation);
        for (index, value) in converted.into_iter().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| import_error("BVH position component index is out of range"))?;
            curves
                .entry((id_string.clone(), "location".to_owned(), index))
                .or_default()
                .push(keyframe(frame, value));
        }
    }
    for child in &joint.children {
        collect_frame_curves(child, ids, values, cursor, frame, curves)?;
    }
    Ok(())
}

fn write_joint(
    output: &mut String,
    joints: &BTreeMap<Id, JointExport>,
    id: &Id,
    parent: Option<&Bone>,
    depth: usize,
) -> Result<()> {
    let joint = joints
        .get(id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "joint is missing"))?;
    let label = if parent.is_some() { "JOINT" } else { "ROOT" };
    let offset = match parent {
        Some(parent) => sub(joint.bone.head, parent.head),
        None => joint.bone.head,
    };
    let offset = blender_to_bvh_vec(offset);
    line(output, depth, &format!("{label} {}", joint.name));
    line(output, depth, "{");
    line(output, depth + 1, &format!("OFFSET {}", vec3(offset)?));
    if parent.is_some() {
        line(
            output,
            depth + 1,
            "CHANNELS 3 Zrotation Xrotation Yrotation",
        );
    } else {
        line(
            output,
            depth + 1,
            "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        );
    }
    for child in &joint.children {
        write_joint(output, joints, child, Some(&joint.bone), depth + 1)?;
    }
    if joint.children.is_empty() {
        let end = blender_to_bvh_vec(sub(joint.bone.tail, joint.bone.head));
        line(output, depth + 1, "End Site");
        line(output, depth + 1, "{");
        line(output, depth + 2, &format!("OFFSET {}", vec3(end)?));
        line(output, depth + 1, "}");
    }
    line(output, depth, "}");
    Ok(())
}

fn preorder<'a>(joints: &'a BTreeMap<Id, JointExport>, root: &'a Id) -> Vec<&'a Id> {
    let mut result = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        result.push(id);
        if let Some(joint) = joints.get(id) {
            stack.extend(joint.children.iter().rev());
        }
    }
    result
}

fn joint_channel_count(joint: &BvhJoint) -> usize {
    joint.channels.len()
        + joint
            .children
            .iter()
            .map(joint_channel_count)
            .sum::<usize>()
}

fn frame_count(start: i32, end: i32) -> Result<usize> {
    let frames = i64::from(end)
        .checked_sub(i64::from(start))
        .and_then(|span| span.checked_add(1))
        .filter(|count| *count > 0)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "scene frame range is invalid"))?;
    let count = usize::try_from(frames)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "BVH frame count is too large"))?;
    if count > MAX_BVH_FRAMES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "BVH frame count exceeds the limit",
        ));
    }
    Ok(count)
}

fn unique_joint_name(name: &str, names: &BTreeMap<String, Id>) -> String {
    let base = bvh_name(name);
    if !names.contains_key(&base) {
        return base;
    }
    let mut index = 2_u32;
    loop {
        let candidate = format!("{base}_{index}");
        if !names.contains_key(&candidate) {
            return candidate;
        }
        index += 1;
    }
}

fn bvh_name(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if sanitized.is_empty() {
        "Joint".to_owned()
    } else {
        sanitized
    }
}

fn bone_id_seed(value: &str) -> String {
    let mut id = value
        .chars()
        .map(|character| character.to_ascii_lowercase())
        .filter(|character| {
            character.is_ascii_alphanumeric() || *character == '_' || *character == '-'
        })
        .collect::<String>();
    if !id.starts_with(|character: char| character.is_ascii_lowercase()) {
        id.insert_str(0, "bone_");
    }
    if id.len() > 55 {
        id.truncate(55);
    }
    if id.is_empty() { "bone".to_owned() } else { id }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .flat_map(|token| {
            let mut chunks = Vec::new();
            let mut start = 0;
            for (index, character) in token.char_indices() {
                if matches!(character, '{' | '}') {
                    if start < index {
                        chunks.push(token[start..index].to_owned());
                    }
                    chunks.push(character.to_string());
                    start = index + character.len_utf8();
                }
            }
            if start < token.len() {
                chunks.push(token[start..].to_owned());
            }
            chunks
        })
        .collect()
}

fn expect<'a>(tokens: &'a [String], cursor: &mut usize, wanted: &str) -> Result<&'a str> {
    let value = next(tokens, cursor, wanted)?;
    if value != wanted {
        return Err(import_error(format!(
            "expected '{wanted}' in BVH, found '{value}'"
        )));
    }
    Ok(value)
}

fn next<'a>(tokens: &'a [String], cursor: &mut usize, label: &str) -> Result<&'a str> {
    let value = tokens
        .get(*cursor)
        .ok_or_else(|| import_error(format!("missing BVH {label}")))?;
    *cursor += 1;
    Ok(value)
}

fn peek(tokens: &[String], cursor: usize) -> Option<&str> {
    tokens.get(cursor).map(String::as_str)
}

fn read_vec3(tokens: &[String], cursor: &mut usize, label: &str) -> Result<[f64; 3]> {
    Ok([
        parse_number(next(tokens, cursor, label)?, label)?,
        parse_number(next(tokens, cursor, label)?, label)?,
        parse_number(next(tokens, cursor, label)?, label)?,
    ])
}

fn parse_number(value: &str, label: &str) -> Result<f64> {
    let number = value
        .parse::<f64>()
        .map_err(|_| import_error(format!("invalid BVH {label}")))?;
    if !number.is_finite() {
        return Err(import_error(format!("non-finite BVH {label}")));
    }
    Ok(number)
}

fn parse_usize(value: &str, label: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| import_error(format!("invalid BVH {label}")))
}

fn blender_to_bvh_rotation(rotation: [f64; 4]) -> Result<DQuat> {
    if rotation.iter().any(|component| !component.is_finite()) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "armature pose contains a non-finite rotation",
        ));
    }
    let q = DQuat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    if q.length_squared() <= f64::EPSILON {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "armature pose contains a zero rotation",
        ));
    }
    Ok((BLENDER_TO_BVH.inverse() * q.normalize() * BLENDER_TO_BVH).normalize())
}

fn blender_to_bvh_vec(value: [f64; 3]) -> [f64; 3] {
    [value[0], value[2], -value[1]]
}

fn bvh_to_blender_vec(value: [f64; 3]) -> [f64; 3] {
    [value[0], -value[2], value[1]]
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn vec3(value: [f64; 3]) -> Result<String> {
    Ok(format!(
        "{} {} {}",
        number(value[0])?,
        number(value[1])?,
        number(value[2])?
    ))
}

fn number(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "BVH value is non-finite",
        ));
    }
    Ok(format!("{value:.9}"))
}

fn line(output: &mut String, depth: usize, value: &str) {
    output.push_str(&"  ".repeat(depth));
    output.push_str(value);
    output.push('\n');
}

fn keyframe(frame: f64, value: f64) -> Keyframe {
    Keyframe {
        frame,
        value,
        interpolation: Interpolation::Linear,
        ..Keyframe::default()
    }
}

fn unsupported(message: String) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id":"format.bvh", "status":"not_supported"}),
    )
}

#[cfg(test)]
mod tests {

    use super::{blender_to_bvh_vec, bvh_to_blender_vec};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn axis_conversion_is_an_involution(point in prop::array::uniform3(-1.0e6_f64..1.0e6)) {
            let round_trip = bvh_to_blender_vec(blender_to_bvh_vec(point));
            for axis in 0..3 {
                prop_assert!((round_trip[axis] - point[axis]).abs() <= 1.0e-9);
            }
        }
    }

    #[test]
    fn motion_values_follow_declared_channel_order() -> crate::error::Result<()> {
        use std::collections::BTreeMap;

        use super::{BLENDER_TO_BVH, collect_frame_curves, import_error, parse_bvh};
        use crate::model::Id;
        use glam::DQuat;

        let source = "HIERARCHY\nROOT Root {\nOFFSET 0 0 0\nCHANNELS 6 Zposition Yrotation Xposition Xrotation Yposition Zrotation\nEnd Site { OFFSET 0 1 0 }\n}\nMOTION\nFrames: 1\nFrame Time: 0.0333333333\n3 45 2 90 4 30\n";
        let parsed = parse_bvh(source)?;
        let root = parsed
            .roots
            .first()
            .ok_or_else(|| import_error("BVH hierarchy has no root"))?;
        let frame = parsed
            .frames
            .first()
            .ok_or_else(|| import_error("BVH motion has no sample"))?;
        let ids = BTreeMap::from([("Root".to_owned(), Id::new("root")?)]);
        let mut cursor = 0;
        let mut curves = BTreeMap::new();
        collect_frame_curves(root, &ids, frame, &mut cursor, 1.0, &mut curves)?;
        assert_eq!(cursor, 6);
        for (component, value) in [2.0, -3.0, 4.0].into_iter().enumerate() {
            let index = u32::try_from(component).map_err(|_| {
                crate::error::PotError::new(
                    crate::error::ErrorCode::InternalError,
                    "index overflow",
                )
            })?;
            let key = ("root".to_owned(), "location".to_owned(), index);
            assert_eq!(
                curves
                    .get(&key)
                    .and_then(|keyframes| keyframes.first())
                    .map(|keyframe| keyframe.value),
                Some(value)
            );
        }
        let bvh_rotation = DQuat::from_rotation_y(45_f64.to_radians())
            * DQuat::from_rotation_x(90_f64.to_radians())
            * DQuat::from_rotation_z(30_f64.to_radians());
        let expected_rotation =
            (BLENDER_TO_BVH * bvh_rotation * BLENDER_TO_BVH.inverse()).to_array();
        for (component, expected) in expected_rotation.into_iter().enumerate() {
            let index =
                u32::try_from(component).map_err(|_| import_error("component index overflow"))?;
            let key = ("root".to_owned(), "rotation".to_owned(), index);
            let actual = curves
                .get(&key)
                .and_then(|keyframes| keyframes.first())
                .map(|keyframe| keyframe.value)
                .ok_or_else(|| import_error("rotation channel is missing"))?;
            assert!((actual - expected).abs() < 1.0e-12);
        }
        Ok(())
    }
}
