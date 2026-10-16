//! VRM spring bones. VRM 1.0 `VRMC_springBone` inverts `vrm1_spring_bone` and `spring_chain_v1`.
//! VRM 0.x `secondaryAnimation` inverts `vrm0_spring_bone` and `legacy_collider_groups`. VRM 0.x cannot
//! name collider groups, order their colliders, or mark a chain's terminal joint, so those Potter-only
//! facts are read from root `extras.potter.vrm0SpringBone` when present; without them the VRM 0.x
//! structure is decoded directly.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::{
    error::Result,
    exchange::gltf,
    model::{
        AvatarNodeRef, AvatarSpring, AvatarSpringBone, AvatarSpringCollider,
        AvatarSpringColliderGroup, AvatarSpringColliderShape, AvatarSpringJoint, AvatarVersion,
        validate_spring,
    },
};

use super::{
    Decoder,
    expression::{index_of, items, number_array},
};

const SUGGESTION: &str = "use --allow-lossy to import the file without this spring bone data";
const COLLIDER_FEATURE: &str = "vrm.spring.collider";
const COLLIDER_GROUP_FEATURE: &str = "vrm.spring.collider_group";

/// Decodes the spring bones; `None` when the file has no spring bone extension.
pub(super) fn decode(decoder: &mut Decoder<'_>) -> Result<Option<AvatarSpringBone>> {
    let index = decoder.index;
    let extension: &Value = decoder.extension;
    match decoder.version {
        AvatarVersion::V1 => {
            let spring = &index.root["extensions"]["VRMC_springBone"];
            if !spring.is_object() {
                return Ok(None);
            }
            decode_v1(decoder, spring).map(Some)
        }
        AvatarVersion::V0 => {
            let secondary = &extension["secondaryAnimation"];
            if !secondary.is_object() {
                return Ok(None);
            }
            let meta = &index.root["extras"]["potter"]["vrm0SpringBone"];
            decode_v0(decoder, secondary, meta).map(Some)
        }
    }
}

/// A spring chain in source form, before Potter's invariants are checked.
struct Candidate {
    feature: String,
    name: Option<String>,
    center: Option<AvatarNodeRef>,
    joints: Vec<AvatarSpringJoint>,
    /// Source collider group indices, mapped to Potter groups by [`assemble`].
    groups: Vec<usize>,
}

/// A collider group; `colliders` are Potter collider indices.
struct RawGroup {
    name: String,
    colliders: Vec<usize>,
}

/// Potter colliders and collider groups. `group_map` maps each source collider group index to its
/// Potter group, or `None` when the group was dropped.
struct Colliders {
    items: Vec<AvatarSpringCollider>,
    groups: Vec<AvatarSpringColliderGroup>,
    group_map: Vec<Option<usize>>,
}

fn decode_v1(decoder: &mut Decoder<'_>, spring: &Value) -> Result<AvatarSpringBone> {
    let colliders = v1_colliders(decoder, spring)?;
    let mut candidates = Vec::new();
    for (index, chain) in items(&spring["springs"]).iter().enumerate() {
        if let Some(candidate) = v1_chain(decoder, index, chain)? {
            candidates.push(candidate);
        }
    }
    Ok(assemble(decoder, colliders, candidates))
}

/// Inverts `vrm1_spring_bone` colliders and collider groups.
fn v1_colliders(decoder: &mut Decoder<'_>, spring: &Value) -> Result<Colliders> {
    let mut colliders = Vec::new();
    let mut collider_map: Vec<Option<usize>> = Vec::new();
    for entry in items(&spring["colliders"]) {
        match v1_collider(decoder, entry)? {
            Some(collider) => {
                collider_map.push(Some(colliders.len()));
                colliders.push(collider);
            }
            None => collider_map.push(None),
        }
    }
    let mut raw = Vec::new();
    for group in items(&spring["colliderGroups"]) {
        let mut members = Vec::new();
        for member in items(&group["colliders"]) {
            let Some(index) = index_of(member) else {
                decoder.loss(
                    COLLIDER_GROUP_FEATURE,
                    "collider group member is not an index",
                    SUGGESTION,
                );
                continue;
            };
            match collider_map.get(index) {
                Some(Some(mapped)) => members.push(*mapped),
                // The dropped collider was already reported.
                Some(None) => {}
                None => decoder.loss(
                    COLLIDER_GROUP_FEATURE,
                    format!("collider group references missing collider {index}"),
                    SUGGESTION,
                ),
            }
        }
        raw.push(RawGroup {
            name: group["name"].as_str().unwrap_or_default().to_owned(),
            colliders: members,
        });
    }
    let source = (0..raw.len()).map(Some).collect();
    Ok(finish_colliders(decoder, colliders, raw, source))
}

/// Inverts one `vrm1_spring_bone` collider; `None` (with a loss) when it is unrepresentable.
fn v1_collider(decoder: &mut Decoder<'_>, entry: &Value) -> Result<Option<AvatarSpringCollider>> {
    let Some(node_index) = index_of(&entry["node"]) else {
        decoder.loss(COLLIDER_FEATURE, "collider has no node", SUGGESTION);
        return Ok(None);
    };
    let node = decoder.node_ref(node_index)?;
    let shape_json = &entry["shape"];
    let shape = if shape_json["sphere"].is_object() {
        let sphere = &shape_json["sphere"];
        let (Some(offset), Some(radius)) = (
            number_array::<3>(&sphere["offset"]),
            sphere["radius"].as_f64(),
        ) else {
            decoder.loss(COLLIDER_FEATURE, "sphere collider is malformed", SUGGESTION);
            return Ok(None);
        };
        let Some(offset) = point_for(decoder, COLLIDER_FEATURE, &node, offset) else {
            return Ok(None);
        };
        let Some(radius) = radius_for(decoder, COLLIDER_FEATURE, &node, radius) else {
            return Ok(None);
        };
        AvatarSpringColliderShape::Sphere { offset, radius }
    } else if shape_json["capsule"].is_object() {
        let capsule = &shape_json["capsule"];
        let (Some(offset), Some(tail), Some(radius)) = (
            number_array::<3>(&capsule["offset"]),
            number_array::<3>(&capsule["tail"]),
            capsule["radius"].as_f64(),
        ) else {
            decoder.loss(
                COLLIDER_FEATURE,
                "capsule collider is malformed",
                SUGGESTION,
            );
            return Ok(None);
        };
        let Some(offset) = point_for(decoder, COLLIDER_FEATURE, &node, offset) else {
            return Ok(None);
        };
        let Some(tail) = point_for(decoder, COLLIDER_FEATURE, &node, tail) else {
            return Ok(None);
        };
        let Some(radius) = radius_for(decoder, COLLIDER_FEATURE, &node, radius) else {
            return Ok(None);
        };
        AvatarSpringColliderShape::Capsule {
            offset,
            tail,
            radius,
        }
    } else {
        decoder.loss(
            COLLIDER_FEATURE,
            "collider shape is neither a sphere nor a capsule",
            SUGGESTION,
        );
        return Ok(None);
    };
    Ok(Some(AvatarSpringCollider { node, shape }))
}

/// Inverts one `vrm1_spring_bone` spring. `None` (with a loss) when a joint is unrepresentable.
fn v1_chain(decoder: &mut Decoder<'_>, index: usize, chain: &Value) -> Result<Option<Candidate>> {
    let name = chain["name"].as_str().map(str::to_owned);
    let feature = chain_feature(name.as_deref(), index);
    let mut joints = Vec::new();
    for joint in items(&chain["joints"]) {
        let Some(node_index) = index_of(&joint["node"]) else {
            decoder.loss(&feature, "spring joint has no node", SUGGESTION);
            return Ok(None);
        };
        let node = decoder.node_ref(node_index)?;
        let Some(hit_radius) = radius_for(
            decoder,
            &feature,
            &node,
            joint["hitRadius"].as_f64().unwrap_or(0.0),
        ) else {
            return Ok(None);
        };
        // `spring_chain_v1` writes gravityDir with `axis_to_gltf` and no frame offset.
        let gravity_dir = number_array::<3>(&joint["gravityDir"]).unwrap_or([0.0, -1.0, 0.0]);
        joints.push(AvatarSpringJoint {
            node,
            hit_radius,
            stiffness: joint["stiffness"].as_f64().unwrap_or(1.0),
            gravity_power: joint["gravityPower"].as_f64().unwrap_or(0.0),
            gravity_dir: gltf::axis_from_gltf(gravity_dir),
            drag_force: joint["dragForce"].as_f64().unwrap_or(0.5),
        });
    }
    let center = match &chain["center"] {
        // A center of -1 means the chain has none (UniVRM and the VRM Add-on write it).
        Value::Null => None,
        value if value.as_i64() == Some(-1) => None,
        value => {
            let Some(center) = index_of(value) else {
                decoder.loss(&feature, "spring center is not a node index", SUGGESTION);
                return Ok(None);
            };
            Some(decoder.node_ref(center)?)
        }
    };
    let groups = index_list(decoder, &feature, &chain["colliderGroups"]);
    Ok(Some(Candidate {
        feature,
        name,
        center,
        joints,
        groups,
    }))
}

fn decode_v0(
    decoder: &mut Decoder<'_>,
    secondary: &Value,
    meta: &Value,
) -> Result<AvatarSpringBone> {
    let colliders = v0_colliders(decoder, secondary, meta)?;
    let bone_groups = items(&secondary["boneGroups"]);
    let meta_chains = items(&meta["boneGroups"]);
    let meta_chains = if meta_chains.len() == bone_groups.len() {
        Some(meta_chains)
    } else {
        if !meta.is_null() {
            decoder.loss(
                "vrm.spring",
                "Potter spring metadata does not match the VRM 0.x bone groups; joint paths are inferred",
                SUGGESTION,
            );
        }
        None
    };
    let mut candidates = Vec::new();
    for (index, group) in bone_groups.iter().enumerate() {
        let listed = meta_chains.and_then(|chains| chains.get(index));
        candidates.extend(v0_chains(decoder, index, group, listed)?);
    }
    Ok(assemble(decoder, colliders, candidates))
}

/// Recovers Potter colliders and collider groups from the VRM 0.x `colliderGroups`.
///
/// Each VRM 0.x group holds one node's spheres. With Potter metadata, the exporter's group names,
/// member order and per-sphere Potter indices restore the original structure. Without it, each VRM
/// 0.x group becomes one Potter group named by its position and each sphere a new collider.
fn v0_colliders(decoder: &mut Decoder<'_>, secondary: &Value, meta: &Value) -> Result<Colliders> {
    let groups = items(&secondary["colliderGroups"]);
    let total: usize = groups
        .iter()
        .map(|group| items(&group["colliders"]).len())
        .sum();
    // `pending` groups use Potter collider indices; `item_index` gives each VRM 0.x sphere's Potter index.
    let (pending, source, item_index) = if let Some(parsed) = v0_meta_colliders(groups, meta, total)
    {
        parsed
    } else {
        if !meta.is_null() {
            decoder.loss(
                "vrm.spring",
                "Potter spring metadata does not match the VRM 0.x collider groups; collider groups are inferred",
                SUGGESTION,
            );
        }
        default_v0_colliders(groups)
    };

    let mut slots: Vec<Option<AvatarSpringCollider>> = vec![None; total];
    let mut position = 0;
    for group in groups {
        let node = index_of(&group["node"]);
        for item in items(&group["colliders"]) {
            let potter = item_index[position];
            position += 1;
            if slots[potter].is_some() {
                continue;
            }
            let Some(node) = node else {
                decoder.loss(COLLIDER_FEATURE, "collider group has no node", SUGGESTION);
                continue;
            };
            slots[potter] = v0_collider(decoder, node, item)?;
        }
    }

    let mut remap: Vec<Option<usize>> = vec![None; total];
    let mut colliders = Vec::new();
    for (old, slot) in slots.into_iter().enumerate() {
        if let Some(collider) = slot {
            remap[old] = Some(colliders.len());
            colliders.push(collider);
        }
    }

    let mut resolved = Vec::with_capacity(pending.len());
    for group in pending {
        let mut members = Vec::with_capacity(group.colliders.len());
        for member in group.colliders {
            match remap.get(member).copied().flatten() {
                Some(index) => members.push(index),
                None => decoder.loss(
                    COLLIDER_GROUP_FEATURE,
                    format!(
                        "collider group `{}` references a sphere that is not in the file",
                        group.name
                    ),
                    SUGGESTION,
                ),
            }
        }
        resolved.push(RawGroup {
            name: group.name,
            colliders: members,
        });
    }
    Ok(finish_colliders(decoder, colliders, resolved, source))
}

type MetaColliders = (Vec<RawGroup>, Vec<Option<usize>>, Vec<usize>);

/// Reads the Potter collider structure from `extras.potter.vrm0SpringBone`, or `None` when it is
/// absent or does not match the VRM 0.x groups. Returns the pending groups (Potter collider indices),
/// the source-group to pending-group map, and the Potter index of every VRM 0.x sphere in file order.
fn v0_meta_colliders(groups: &[Value], meta: &Value, total: usize) -> Option<MetaColliders> {
    let mut pending = Vec::new();
    for group in items(&meta["colliderGroups"]) {
        let name = group["name"].as_str()?.to_owned();
        let colliders = items(&group["colliders"])
            .iter()
            .map(index_of)
            .collect::<Option<Vec<_>>>()?;
        pending.push(RawGroup { name, colliders });
    }
    let legacy = items(&meta["legacyColliderGroups"]);
    if legacy.len() != groups.len() {
        return None;
    }
    let mut source = Vec::with_capacity(groups.len());
    let mut item_index = Vec::with_capacity(total);
    for (group, entry) in groups.iter().zip(legacy) {
        let group_index = index_of(&entry["group"])?;
        if group_index >= pending.len() {
            return None;
        }
        let potter = items(&entry["colliders"])
            .iter()
            .map(index_of)
            .collect::<Option<Vec<_>>>()?;
        if potter.len() != items(&group["colliders"]).len()
            || potter.iter().any(|index| *index >= total)
        {
            return None;
        }
        source.push(Some(group_index));
        item_index.extend(potter);
    }
    Some((pending, source, item_index))
}

/// The VRM 0.x structure alone: one Potter group per VRM 0.x group, one collider per sphere.
fn default_v0_colliders(groups: &[Value]) -> (Vec<RawGroup>, Vec<Option<usize>>, Vec<usize>) {
    let mut pending = Vec::with_capacity(groups.len());
    let mut source = Vec::with_capacity(groups.len());
    let mut item_index = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        let start = item_index.len();
        let count = items(&group["colliders"]).len();
        item_index.extend(start..start + count);
        pending.push(RawGroup {
            name: format!("vrm0 collider group {group_index}"),
            colliders: (start..start + count).collect(),
        });
        source.push(Some(group_index));
    }
    (pending, source, item_index)
}

/// Inverts one `vrm0_spring_bone` sphere; `None` (with a loss) when it is malformed or unrepresentable.
fn v0_collider(
    decoder: &mut Decoder<'_>,
    node: usize,
    item: &Value,
) -> Result<Option<AvatarSpringCollider>> {
    let reference = decoder.node_ref(node)?;
    let (Some(stored), Some(radius)) = (legacy_vec3(&item["offset"]), item["radius"].as_f64())
    else {
        decoder.loss(
            COLLIDER_FEATURE,
            "VRM 0.x collider is malformed",
            SUGGESTION,
        );
        return Ok(None);
    };
    // `legacy_z_reflected_vec3` negates the z component of the emitted offset.
    let emitted = [stored[0], stored[1], -stored[2]];
    let Some(offset) = point_for(decoder, COLLIDER_FEATURE, &reference, emitted) else {
        return Ok(None);
    };
    let Some(radius) = radius_for(decoder, COLLIDER_FEATURE, &reference, radius) else {
        return Ok(None);
    };
    Ok(Some(AvatarSpringCollider {
        node: reference,
        shape: AvatarSpringColliderShape::Sphere { offset, radius },
    }))
}

/// Inverts one `vrm0_spring_bone` bone group into one chain per root bone. Each root's chain follows
/// the glTF subtree below it (`VRoid` hair lists many roots in one group). Potter's `listed` joint path
/// applies only to a single-root group, which is how the exporter writes every group.
fn v0_chains(
    decoder: &mut Decoder<'_>,
    index: usize,
    group: &Value,
    listed: Option<&Value>,
) -> Result<Vec<Candidate>> {
    let name = group["comment"].as_str().map(str::to_owned);
    let feature = chain_feature(name.as_deref(), index);
    let Some(roots) = items(&group["bones"])
        .iter()
        .map(index_of)
        .collect::<Option<Vec<usize>>>()
        .filter(|roots| !roots.is_empty())
    else {
        decoder.loss(&feature, "spring has no root bone", SUGGESTION);
        return Ok(Vec::new());
    };
    // Writers list either independent roots (VRoid hair) or every bone of one chain, whose roots nest.
    // A nested root is already in its ancestor root's chain, so only top-level roots start chains.
    let roots = top_level_roots(decoder, &roots)?;
    let listed = if roots.len() == 1 { listed } else { None };
    // A center of -1 means the chain has none (UniVRM and the VRM Add-on write it).
    let center = match &group["center"] {
        Value::Null => None,
        value if value.as_i64() == Some(-1) => None,
        value => {
            let Some(center) = index_of(value) else {
                decoder.loss(&feature, "spring center is not a node index", SUGGESTION);
                return Ok(Vec::new());
            };
            Some(decoder.node_ref(center)?)
        }
    };
    let groups = index_list(decoder, &feature, &group["colliderGroups"]);
    // `vrm0_spring_bone` stores z-reflect(Rot_Y180(g)) with g = `axis_to_gltf(gravity_dir)`, so
    // g = [-x, y, z] of the stored vector.
    let gravity_dir = legacy_vec3(&group["gravityDir"]).map_or([0.0, 0.0, -1.0], |stored| {
        gltf::axis_from_gltf([-stored[0], stored[1], stored[2]])
    });
    let stiffness = group["stiffiness"].as_f64().unwrap_or(1.0);
    let gravity_power = group["gravityPower"].as_f64().unwrap_or(0.0);
    let drag_force = group["dragForce"].as_f64().unwrap_or(0.5);
    let mut chains = Vec::with_capacity(roots.len());
    for root in roots {
        let Some(joints) = v0_joint_path(decoder, &feature, root, listed)? else {
            continue;
        };
        let root_ref = decoder.node_ref(root)?;
        let Some(hit_radius) = radius_for(
            decoder,
            &feature,
            &root_ref,
            group["hitRadius"].as_f64().unwrap_or(0.0),
        ) else {
            continue;
        };
        // VRM 0.x has one parameter set per chain, so every joint takes the root's values.
        let mut joint_refs = Vec::with_capacity(joints.len());
        for joint in joints {
            joint_refs.push(AvatarSpringJoint {
                node: decoder.node_ref(joint)?,
                hit_radius,
                stiffness,
                gravity_power,
                gravity_dir,
                drag_force,
            });
        }
        chains.push(Candidate {
            feature: feature.clone(),
            name: name.clone(),
            center: center.clone(),
            joints: joint_refs,
            groups: groups.clone(),
        });
    }
    Ok(chains)
}

/// The glTF node indices of the chain, root first. The exporter's listed path is checked to be a
/// parent-child chain from the root; without it, the path follows the single child at each step and
/// stops at a leaf. A branch makes the path non-unique, which is reported and skipped.
fn v0_joint_path(
    decoder: &mut Decoder<'_>,
    feature: &str,
    root: usize,
    listed: Option<&Value>,
) -> Result<Option<Vec<usize>>> {
    if let Some(listed) = listed {
        let Some(joints) = items(&listed["joints"])
            .iter()
            .map(index_of)
            .collect::<Option<Vec<usize>>>()
        else {
            decoder.loss(
                feature,
                "Potter spring metadata lists a malformed joint path",
                SUGGESTION,
            );
            return Ok(None);
        };
        let mut linked = joints.first() == Some(&root);
        for pair in joints.windows(2) {
            linked = linked && children(decoder, pair[0])?.contains(&pair[1]);
        }
        if !linked {
            decoder.loss(
                feature,
                "Potter joint path is not a parent-child chain from the root bone",
                SUGGESTION,
            );
            return Ok(None);
        }
        return Ok(Some(joints));
    }
    let mut joints = vec![root];
    let mut current = root;
    loop {
        let next = children(decoder, current)?;
        match next.as_slice() {
            [] => return Ok(Some(joints)),
            [child] if !joints.contains(child) => {
                joints.push(*child);
                current = *child;
            }
            [_] => {
                decoder.loss(feature, "spring joint path contains a cycle", SUGGESTION);
                return Ok(None);
            }
            _ => {
                decoder.loss(
                    feature,
                    "spring root branches, so its simulated joint path is not unique",
                    SUGGESTION,
                );
                return Ok(None);
            }
        }
    }
}

/// Builds Potter's collider groups from the decoded colliders, dropping groups that are unnamed or
/// empty. `source` maps each source group index to its entry in `raw`.
fn finish_colliders(
    decoder: &mut Decoder<'_>,
    colliders: Vec<AvatarSpringCollider>,
    raw: Vec<RawGroup>,
    source: Vec<Option<usize>>,
) -> Colliders {
    let mut groups = Vec::new();
    let mut raw_map = Vec::with_capacity(raw.len());
    for group in raw {
        let mut indices = Vec::new();
        for index in group.colliders {
            if !indices.contains(&index) {
                indices.push(index);
            }
        }
        if group.name.trim().is_empty() || indices.is_empty() {
            decoder.loss(
                COLLIDER_GROUP_FEATURE,
                format!(
                    "collider group `{}` has no name or no colliders",
                    group.name
                ),
                SUGGESTION,
            );
            raw_map.push(None);
            continue;
        }
        raw_map.push(Some(groups.len()));
        groups.push(AvatarSpringColliderGroup {
            name: group.name,
            colliders: indices,
        });
    }
    let group_map = source
        .into_iter()
        .map(|entry| entry.and_then(|raw_index| raw_map.get(raw_index).copied().flatten()))
        .collect();
    Colliders {
        items: colliders,
        groups,
        group_map,
    }
}

/// Accepts each candidate chain that keeps the spring bone valid under Potter's invariants, in file
/// order. A rejected chain is reported and skipped.
fn assemble(
    decoder: &mut Decoder<'_>,
    colliders: Colliders,
    candidates: Vec<Candidate>,
) -> AvatarSpringBone {
    let Colliders {
        items: colliders,
        groups,
        group_map,
    } = colliders;
    let mut spring_bone = AvatarSpringBone {
        colliders,
        collider_groups: groups,
        springs: Vec::new(),
    };
    for candidate in candidates {
        if candidate.joints.is_empty() {
            decoder.loss(&candidate.feature, "spring has no joints", SUGGESTION);
            continue;
        }
        let mut collider_groups = Vec::new();
        for source in candidate.groups {
            match group_map.get(source).copied().flatten() {
                Some(group) => {
                    if !collider_groups.contains(&group) {
                        collider_groups.push(group);
                    }
                }
                None => decoder.loss(
                    &candidate.feature,
                    "spring references a collider group that was not imported",
                    SUGGESTION,
                ),
            }
        }
        spring_bone.springs.push(AvatarSpring {
            name: candidate.name,
            center: candidate.center,
            joints: candidate.joints,
            collider_groups,
        });
        if let Err(error) = validate_spring(decoder.doc, &spring_bone) {
            spring_bone.springs.pop();
            decoder.loss(
                &candidate.feature,
                format!("spring is not representable: {}", error.message),
                SUGGESTION,
            );
        }
    }
    spring_bone
}

/// Maps a point stored in `reference`'s emitted frame to Potter source coordinates.
fn point_for(
    decoder: &mut Decoder<'_>,
    feature: &str,
    reference: &AvatarNodeRef,
    point: [f64; 3],
) -> Option<[f64; 3]> {
    let mapped = decoder.source_point(reference, point);
    frame_mapped(decoder, feature, mapped)
}

/// Maps a radius stored in `reference`'s emitted frame to Potter source units.
fn radius_for(
    decoder: &mut Decoder<'_>,
    feature: &str,
    reference: &AvatarNodeRef,
    radius: f64,
) -> Option<f64> {
    if radius < 0.0 {
        decoder.loss(feature, "radius is negative", SUGGESTION);
        return None;
    }
    let mapped = decoder.source_radius(reference, radius);
    frame_mapped(decoder, feature, mapped)
}

/// Turns a failed frame mapping (singular or non-uniformly scaled) into a loss.
fn frame_mapped<T>(decoder: &mut Decoder<'_>, feature: &str, mapped: Result<T>) -> Option<T> {
    match mapped {
        Ok(value) => Some(value),
        Err(error) => {
            decoder.loss(feature, error.message, SUGGESTION);
            None
        }
    }
}

/// Collider group references in a VRM 0.x or VRM 1.0 array; non-index entries are reported.
fn index_list(decoder: &mut Decoder<'_>, feature: &str, value: &Value) -> Vec<usize> {
    let mut indices = Vec::new();
    for item in items(value) {
        match index_of(item) {
            Some(index) => indices.push(index),
            None => decoder.loss(
                feature,
                "collider group reference is not an index",
                SUGGESTION,
            ),
        }
    }
    indices
}

/// Inverts `legacy_vec3`: a VRM 0.x `{x, y, z}` object.
fn legacy_vec3(value: &Value) -> Option<[f64; 3]> {
    Some([
        value["x"].as_f64()?,
        value["y"].as_f64()?,
        value["z"].as_f64()?,
    ])
}

fn chain_feature(name: Option<&str>, index: usize) -> String {
    match name {
        Some(name) if !name.trim().is_empty() => format!("vrm.spring.{name}"),
        _ => format!("vrm.spring.{index}"),
    }
}

/// The glTF node indices of `node`'s children.
fn children(decoder: &Decoder<'_>, node: usize) -> Result<Vec<usize>> {
    Ok(items(&decoder.gltf_node(node)?["children"])
        .iter()
        .filter_map(index_of)
        .collect())
}

/// The roots that start a chain: each listed root not below another listed root, once each, in order.
fn top_level_roots(decoder: &Decoder<'_>, roots: &[usize]) -> Result<Vec<usize>> {
    let mut kept = Vec::new();
    for &root in roots {
        let mut nested = false;
        for &other in roots {
            if other != root && descendants(decoder, other)?.contains(&root) {
                nested = true;
                break;
            }
        }
        if !nested && !kept.contains(&root) {
            kept.push(root);
        }
    }
    Ok(kept)
}

/// The glTF node indices below `node`, `node` excluded.
fn descendants(decoder: &Decoder<'_>, node: usize) -> Result<BTreeSet<usize>> {
    let mut found = BTreeSet::new();
    let mut pending = children(decoder, node)?;
    while let Some(next) = pending.pop() {
        if found.insert(next) {
            pending.extend(children(decoder, next)?);
        }
    }
    Ok(found)
}
