//! First-person and look-at decoding. VRM 1.0 stores them under `firstPerson` and `lookAt`. VRM 0.x stores
//! both under `firstPerson`: mesh annotations by mesh index, look-at range maps, and the head offset.
//! Inverts `vrm1_first_person`, `vrm1_look_at`, `vrm0_first_person`, `map_head_offset` and
//! `legacy_z_reflected_vec3`.

use serde_json::Value;

use crate::{
    error::Result,
    model::{
        AvatarFirstPerson, AvatarLookAt, AvatarLookAtType, AvatarMeshAnnotation,
        AvatarMeshAnnotationType, AvatarNodeRef, AvatarRangeMap, AvatarVersion, Id,
    },
};

use super::Decoder;

pub(super) fn first_person(dec: &mut Decoder<'_>) -> Result<Option<AvatarFirstPerson>> {
    match dec.version {
        AvatarVersion::V1 => v1_first_person(dec),
        AvatarVersion::V0 => v0_first_person(dec),
    }
}

pub(super) fn look_at(dec: &mut Decoder<'_>) -> Result<Option<AvatarLookAt>> {
    match dec.version {
        AvatarVersion::V1 => v1_look_at(dec),
        AvatarVersion::V0 => v0_look_at(dec),
    }
}

/// Inverts `vrm1_first_person`: the `firstPerson` object exists exactly when Potter exported first-person settings.
fn v1_first_person(dec: &mut Decoder<'_>) -> Result<Option<AvatarFirstPerson>> {
    let extension = dec.extension;
    let first_person = &extension["firstPerson"];
    if !first_person.is_object() {
        return Ok(None);
    }
    check_first_person_bone(dec, first_person)?;
    if let Some(offset) = vec3_array(&first_person["firstPersonBoneOffset"])
        && offset.iter().any(|value| *value != 0.0)
    {
        dec.loss(
            "vrm.first_person.first_person_bone_offset",
            "VRM 1.0 firstPersonBoneOffset has no Potter field",
            "use --allow-lossy to import the file without this offset",
        );
    }
    let mut mesh_annotations = Vec::new();
    for annotation in first_person["meshAnnotations"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
    {
        let Some(node) = node_index(&annotation["node"]) else {
            dec.loss(
                "vrm.first_person.node",
                "VRM 1.0 first-person mesh annotation has no node",
                "set the node of the annotation in the VRM file",
            );
            continue;
        };
        let Some(annotation_type) = v1_annotation_type(&annotation["type"]) else {
            dec.loss(
                "vrm.first_person.type",
                "VRM 1.0 first-person mesh annotation type is unsupported",
                "correct the annotation type in the VRM file",
            );
            continue;
        };
        if let Some(node) = annotation_node(dec, node)? {
            mesh_annotations.push(AvatarMeshAnnotation {
                node,
                annotation_type,
            });
        }
    }
    Ok(Some(AvatarFirstPerson { mesh_annotations }))
}

/// Inverts the first-person half of `vrm0_first_person`. `meshAnnotations` name glTF meshes; the look-at
/// settings share the same object and are recognized by `lookAtTypeName`.
fn v0_first_person(dec: &mut Decoder<'_>) -> Result<Option<AvatarFirstPerson>> {
    let extension = dec.extension;
    let first_person = &extension["firstPerson"];
    if !first_person.is_object() {
        return Ok(None);
    }
    check_first_person_bone(dec, first_person)?;
    let mut mesh_annotations = Vec::new();
    for annotation in first_person["meshAnnotations"]
        .as_array()
        .map_or(&[][..], Vec::as_slice)
    {
        let Some(mesh) = node_index(&annotation["mesh"]) else {
            dec.loss(
                "vrm.first_person.mesh",
                "VRM 0.x first-person mesh annotation has no mesh index",
                "set the mesh of the annotation in the VRM file",
            );
            continue;
        };
        let Some(annotation_type) = v0_annotation_type(&annotation["firstPersonFlag"]) else {
            dec.loss(
                "vrm.first_person.type",
                "VRM 0.x firstPersonFlag is unsupported",
                "correct the firstPersonFlag in the VRM file",
            );
            continue;
        };
        // The exporter writes the mesh index of the annotated node; the node is recovered only when unique.
        let nodes = dec.nodes_using_mesh(mesh);
        match nodes.as_slice() {
            [node] => {
                if let Some(node) = annotation_node(dec, *node)? {
                    mesh_annotations.push(AvatarMeshAnnotation {
                        node,
                        annotation_type,
                    });
                }
            }
            [] => dec.loss(
                "vrm.first_person.mesh",
                format!("VRM 0.x first-person mesh {mesh} is not instanced by any node"),
                "remove the annotation from the VRM file",
            ),
            _ => dec.loss(
                "vrm.first_person.shared_mesh",
                format!("VRM 0.x first-person mesh {mesh} is instanced by several nodes"),
                "give the annotated node its own mesh, or use --allow-lossy to import without annotations",
            ),
        }
    }
    // Without look-at, the exporter writes `firstPerson` only for first-person settings, so an empty
    // object is an empty `Some`. With look-at, the object is the look-at carrier and empty first-person
    // settings cannot be told apart from none, so `None` is chosen.
    if mesh_annotations.is_empty() && v0_has_look_at(first_person) {
        return Ok(None);
    }
    Ok(Some(AvatarFirstPerson { mesh_annotations }))
}

/// Inverts `vrm1_look_at`. `offsetFromHeadBone` is a vector in the head bone frame (`map_head_offset`).
fn v1_look_at(dec: &mut Decoder<'_>) -> Result<Option<AvatarLookAt>> {
    let extension = dec.extension;
    let look = &extension["lookAt"];
    if look.is_null() {
        return Ok(None);
    }
    let look_at_type = match look["type"].as_str() {
        Some("bone") => AvatarLookAtType::Bone,
        Some("expression") => AvatarLookAtType::Expression,
        _ => {
            return Ok(malformed_look_at(
                dec,
                "VRM 1.0 lookAt type is missing or unsupported",
            ));
        }
    };
    let offset = match &look["offsetFromHeadBone"] {
        Value::Null => [0.0; 3],
        value => match vec3_array(value) {
            Some(offset) => offset,
            None => {
                return Ok(malformed_look_at(
                    dec,
                    "VRM 1.0 lookAt offsetFromHeadBone is not a vector",
                ));
            }
        },
    };
    let (Some(horizontal_inner), Some(horizontal_outer), Some(vertical_down), Some(vertical_up)) = (
        range_map(
            &look["rangeMapHorizontalInner"],
            "inputMaxValue",
            "outputScale",
        ),
        range_map(
            &look["rangeMapHorizontalOuter"],
            "inputMaxValue",
            "outputScale",
        ),
        range_map(
            &look["rangeMapVerticalDown"],
            "inputMaxValue",
            "outputScale",
        ),
        range_map(&look["rangeMapVerticalUp"], "inputMaxValue", "outputScale"),
    ) else {
        return Ok(malformed_look_at(
            dec,
            "VRM 1.0 lookAt is missing a range map",
        ));
    };
    let Some(head) = head_reference(dec) else {
        return Ok(malformed_look_at(
            dec,
            "VRM look-at settings require a mapped humanoid head bone",
        ));
    };
    let offset_from_head_bone = dec.source_vector(&head, offset)?;
    Ok(Some(AvatarLookAt {
        look_at_type,
        offset_from_head_bone,
        range_map_horizontal_inner: horizontal_inner,
        range_map_horizontal_outer: horizontal_outer,
        range_map_vertical_down: vertical_down,
        range_map_vertical_up: vertical_up,
    }))
}

/// Inverts the look-at half of `vrm0_first_person`. `lookAtTypeName` marks look-at settings. The offset is
/// written as `legacy_z_reflected_vec3`, so the z component is reflected back before undoing the head frame.
fn v0_look_at(dec: &mut Decoder<'_>) -> Result<Option<AvatarLookAt>> {
    let extension = dec.extension;
    let first_person = &extension["firstPerson"];
    if !v0_has_look_at(first_person) {
        return Ok(None);
    }
    let look_at_type = match first_person["lookAtTypeName"].as_str() {
        Some("Bone") => AvatarLookAtType::Bone,
        Some("BlendShape") => AvatarLookAtType::Expression,
        _ => {
            return Ok(malformed_look_at(
                dec,
                "VRM 0.x lookAtTypeName is missing or unsupported",
            ));
        }
    };
    let offset = match &first_person["firstPersonBoneOffset"] {
        Value::Null => [0.0; 3],
        value => match vec3_object(value) {
            Some(offset) => offset,
            None => {
                return Ok(malformed_look_at(
                    dec,
                    "VRM 0.x firstPersonBoneOffset is not a vector",
                ));
            }
        },
    };
    let (Some(horizontal_inner), Some(horizontal_outer), Some(vertical_down), Some(vertical_up)) = (
        range_map(&first_person["lookAtHorizontalInner"], "xRange", "yRange"),
        range_map(&first_person["lookAtHorizontalOuter"], "xRange", "yRange"),
        range_map(&first_person["lookAtVerticalDown"], "xRange", "yRange"),
        range_map(&first_person["lookAtVerticalUp"], "xRange", "yRange"),
    ) else {
        return Ok(malformed_look_at(
            dec,
            "VRM 0.x look-at settings are missing a range map",
        ));
    };
    let Some(head) = head_reference(dec) else {
        return Ok(malformed_look_at(
            dec,
            "VRM look-at settings require a mapped humanoid head bone",
        ));
    };
    // Undo `legacy_z_reflected_vec3`: the stored z is the negated emitted head-frame z.
    let emitted = [offset[0], offset[1], -offset[2]];
    let offset_from_head_bone = dec.source_vector(&head, emitted)?;
    Ok(Some(AvatarLookAt {
        look_at_type,
        offset_from_head_bone,
        range_map_horizontal_inner: horizontal_inner,
        range_map_horizontal_outer: horizontal_outer,
        range_map_vertical_down: vertical_down,
        range_map_vertical_up: vertical_up,
    }))
}

/// Records look-at data that cannot be represented and imports no look-at settings.
fn malformed_look_at(dec: &mut Decoder<'_>, reason: &str) -> Option<AvatarLookAt> {
    dec.loss(
        "vrm.look_at",
        reason,
        "use --allow-lossy to import the file without look-at settings",
    );
    None
}

/// Checks `firstPersonBone`. The exporter always uses the humanoid head; another bone has no Potter field.
fn check_first_person_bone(dec: &mut Decoder<'_>, first_person: &Value) -> Result<()> {
    let Some(node) = node_index(&first_person["firstPersonBone"]) else {
        return Ok(());
    };
    let reference = dec.node_ref(node)?;
    if head_reference(dec).as_ref() != Some(&reference) {
        dec.loss(
            "vrm.first_person.first_person_bone",
            "first-person bone is not the humanoid head bone",
            "use --allow-lossy to import the file with the head bone as first-person bone",
        );
    }
    Ok(())
}

/// A first-person mesh annotation targets a glTF node that instances a mesh and is not a skinned joint.
fn annotation_node(dec: &mut Decoder<'_>, node: usize) -> Result<Option<Id>> {
    let reference = dec.node_ref(node)?;
    if reference.bone.is_some() || dec.gltf_node(node)?["mesh"].is_null() {
        dec.loss(
            "vrm.first_person.node",
            format!("first-person annotation node {node} does not instance a mesh"),
            "annotate a node that instances a mesh",
        );
        return Ok(None);
    }
    Ok(Some(reference.node))
}

/// The humanoid head bone that look-at offsets and first-person bones are expressed in.
fn head_reference(dec: &Decoder<'_>) -> Option<AvatarNodeRef> {
    let armature = dec.armature.clone()?;
    let bone = dec.humanoid.get("head")?.clone();
    Some(AvatarNodeRef {
        node: armature,
        bone: Some(bone),
    })
}

/// VRM 1.0 `type` defaults to `auto`; VRM 0.x `firstPersonFlag` defaults to `Auto`.
fn v1_annotation_type(value: &Value) -> Option<AvatarMeshAnnotationType> {
    match value.as_str() {
        None | Some("auto") => Some(AvatarMeshAnnotationType::Auto),
        Some("both") => Some(AvatarMeshAnnotationType::Both),
        Some("thirdPersonOnly") => Some(AvatarMeshAnnotationType::ThirdPersonOnly),
        Some("firstPersonOnly") => Some(AvatarMeshAnnotationType::FirstPersonOnly),
        Some(_) => None,
    }
}

fn v0_annotation_type(value: &Value) -> Option<AvatarMeshAnnotationType> {
    match value.as_str() {
        None | Some("Auto") => Some(AvatarMeshAnnotationType::Auto),
        Some("Both") => Some(AvatarMeshAnnotationType::Both),
        Some("ThirdPersonOnly") => Some(AvatarMeshAnnotationType::ThirdPersonOnly),
        Some("FirstPersonOnly") => Some(AvatarMeshAnnotationType::FirstPersonOnly),
        Some(_) => None,
    }
}

/// Whether the VRM 0.x `firstPerson` object carries look-at settings (the exporter writes them together).
fn v0_has_look_at(first_person: &Value) -> bool {
    !first_person["lookAtTypeName"].is_null()
}

/// A range map from an object with the given input and output keys.
fn range_map(value: &Value, input_key: &str, output_key: &str) -> Option<AvatarRangeMap> {
    Some(AvatarRangeMap {
        input_max_value: value[input_key].as_f64()?,
        output_scale: value[output_key].as_f64()?,
    })
}

/// A three-element JSON array of numbers.
fn vec3_array(value: &Value) -> Option<[f64; 3]> {
    match value.as_array()?.as_slice() {
        [x, y, z] => Some([x.as_f64()?, y.as_f64()?, z.as_f64()?]),
        _ => None,
    }
}

/// A VRM 0.x `{x, y, z}` object of numbers.
fn vec3_object(value: &Value) -> Option<[f64; 3]> {
    Some([
        value["x"].as_f64()?,
        value["y"].as_f64()?,
        value["z"].as_f64()?,
    ])
}

fn node_index(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|index| usize::try_from(index).ok())
}
