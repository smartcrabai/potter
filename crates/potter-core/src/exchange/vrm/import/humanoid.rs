//! Humanoid decoding. VRM 1.0 `humanoid.humanBones` maps role to `{node}`; VRM 0.x uses an array of
//! `{bone, node}` with legacy thumb names. Inverts `human_bones` and `legacy_humanoid_role`.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::{
    error::Result,
    model::{AvatarVersion, Id, is_humanoid_role},
};

use super::Decoder;

pub(super) fn decode(dec: &mut Decoder<'_>) -> Result<Option<(Id, BTreeMap<String, Id>)>> {
    let extension = dec.extension;
    let version = dec.version;
    // (file label, Potter role, armature node, bone) for every entry that maps to a skinned joint.
    let mut mapped: Vec<(String, String, Id, Id)> = Vec::new();
    for (label, node) in human_bone_entries(extension, version) {
        // Inverse of `legacy_humanoid_role` for VRM 0.x. VRM 0.x has no metacarpal bone.
        let role = match potter_role(version, &label) {
            Some(role) if is_humanoid_role(role) => role.to_owned(),
            _ => {
                dec.loss(
                    &format!("vrm.humanoid.{label}"),
                    format!("VRM humanoid role `{label}` is not a Potter humanoid role"),
                    "remove the role from the VRM file or use --allow-lossy to import without it",
                );
                continue;
            }
        };
        let Some(node) = node else {
            dec.loss(
                &format!("vrm.humanoid.{label}"),
                format!("VRM humanoid role `{label}` has no node"),
                "set the node of the role in the VRM file",
            );
            continue;
        };
        let reference = dec.node_ref(node)?;
        match reference.bone {
            Some(bone) => mapped.push((label, role, reference.node, bone)),
            None => dec.loss(
                &format!("vrm.humanoid.{label}"),
                format!("VRM humanoid role `{label}` node is not a skinned joint"),
                "skin the bone to an exported mesh or remove the role",
            ),
        }
    }
    // The armature is the `hips` bone's armature when mapped, else the first mapped bone's.
    let armature = mapped
        .iter()
        .find(|entry| entry.1 == "hips")
        .or_else(|| mapped.first())
        .map(|entry| entry.2.clone());
    let Some(armature) = armature else {
        dec.loss(
            "vrm.humanoid",
            "VRM humanoid maps no bone to a skinned joint",
            "use --allow-lossy to import the file without avatar settings",
        );
        return Ok(None);
    };
    let mut humanoid: BTreeMap<String, Id> = BTreeMap::new();
    for (label, role, node, bone) in mapped {
        if node != armature {
            dec.loss(
                &format!("vrm.humanoid.{label}"),
                format!("VRM humanoid bone belongs to armature `{node}`, not `{armature}`"),
                "move the bone under the armature that owns the other humanoid bones",
            );
            continue;
        }
        if humanoid.contains_key(&role) {
            dec.loss(
                &format!("vrm.humanoid.{label}"),
                format!("VRM humanoid role `{role}` is assigned more than once"),
                "keep one node per humanoid role in the VRM file",
            );
            continue;
        }
        humanoid.insert(role, bone);
    }
    Ok(Some((armature, humanoid)))
}

/// The `(role label, glTF node index)` pairs of the humanoid object, in file order.
fn human_bone_entries(extension: &Value, version: AvatarVersion) -> Vec<(String, Option<usize>)> {
    let bones = &extension["humanoid"]["humanBones"];
    match version {
        AvatarVersion::V1 => bones.as_object().map_or_else(Vec::new, |object| {
            object
                .iter()
                .map(|(role, bone)| (role.clone(), node_index(&bone["node"])))
                .collect()
        }),
        AvatarVersion::V0 => bones.as_array().map_or_else(Vec::new, |items| {
            items
                .iter()
                .map(|item| {
                    (
                        item["bone"].as_str().unwrap_or_default().to_owned(),
                        node_index(&item["node"]),
                    )
                })
                .collect()
        }),
    }
}

/// The Potter role for a file label. VRM 0.x thumb names are one step behind Potter's
/// (`leftThumbProximal` is Potter's `leftThumbMetacarpal`); `None` marks a name VRM 0.x does not define.
fn potter_role(version: AvatarVersion, label: &str) -> Option<&str> {
    match version {
        AvatarVersion::V1 => Some(label),
        AvatarVersion::V0 => match label {
            "leftThumbProximal" => Some("leftThumbMetacarpal"),
            "leftThumbIntermediate" => Some("leftThumbProximal"),
            "rightThumbProximal" => Some("rightThumbMetacarpal"),
            "rightThumbIntermediate" => Some("rightThumbProximal"),
            "leftThumbMetacarpal" | "rightThumbMetacarpal" => None,
            _ => Some(label),
        },
    }
}

fn node_index(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|index| usize::try_from(index).ok())
}
