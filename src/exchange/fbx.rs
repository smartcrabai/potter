use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Value, json};

use serde::de::DeserializeOwned;

use super::{ImportedGraph, Loss, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::Mesh,
    model::{
        Action, ArmatureData, Bone, DataBlock, Extrapolation, FCurve, Id, Image, ImageAlphaMode,
        ImageColorspace, ImageSource, Interpolation, Keyframe, Material, Node, SceneDoc,
        TextureRef, Transform,
    },
};

const FBX_TICKS_PER_SECOND: f64 = 46_186_158_000.0;
const DEFAULT_FPS: f64 = 24.0;
#[derive(Clone, Copy, Debug)]
struct ObjectIds {
    model: i64,
    geometry: Option<i64>,
}

#[derive(Clone, Debug)]
struct CurveExport {
    id: i64,
    curve_node_id: i64,
    component: &'static str,
    keys: Vec<Keyframe>,
    rotation: bool,
}

#[derive(Clone, Debug)]
struct ClusterExport {
    id: i64,
    bone_model: i64,
    indexes: Vec<usize>,
    weights: Vec<f64>,
    mesh_bind: DMat4,
    bone_bind: DMat4,
}

#[derive(Clone, Debug)]
struct SkinExport {
    id: i64,
    geometry: i64,
    clusters: Vec<ClusterExport>,
}

#[derive(Clone, Debug)]
struct MorphExport {
    blend_id: i64,
    channel_id: i64,
    shape_id: i64,
    geometry: i64,
    name: String,
    value: f64,
    indices: Vec<usize>,
    offsets: Vec<[f64; 3]>,
}

#[derive(Clone, Debug)]
struct TextureExport {
    id: i64,
    material_id: Id,
    property: &'static str,
    path: String,
    uv_map: Option<String>,
}

#[derive(Clone, Debug)]
struct BoneExport {
    model_id: i64,
    parent_model: i64,
    bone_id: Id,
    bone: Bone,
    local: DMat4,
}

/// Export the evaluated mesh and scene transforms as ASCII FBX 7.4.
pub(crate) fn export(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<u8>> {
    export_with_root(doc, snapshot, Path::new("."))
}

/// Export ASCII FBX with the project root available for linked and packed images.
pub(crate) fn export_with_root(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    project_root: &Path,
) -> Result<Vec<u8>> {
    let _ = project_root;
    doc.validate()?;
    let scene = doc
        .scenes
        .get(&doc.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "active scene is missing"))?;
    let fps = f64::from(scene.fps) / scene.fps_base;
    if !fps.is_finite() || fps <= 0.0 {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "FBX animation frame rate must be finite and positive",
        ));
    }

    let mut next_id = 1_000_001_i64;
    let mut ids = BTreeMap::<Id, ObjectIds>::new();
    for (node_id, node) in &doc.nodes {
        let geometry_id = match &node.data {
            Some(data_id) => {
                let data = doc.data_blocks.get(data_id).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::SceneInvalid,
                        "node references a missing data block",
                        json!({ "node_id": node_id, "data_id": data_id }),
                    )
                })?;
                if data.data_type == "mesh" {
                    if !snapshot.meshes.contains_key(node_id) {
                        return Err(PotError::with_details(
                            ErrorCode::EvaluationFailed,
                            "evaluated mesh is missing for FBX export",
                            json!({ "node_id": node_id }),
                        ));
                    }
                    Some(take_object_id(&mut next_id)?)
                } else if data.data_type == "armature" {
                    if data.armature.is_none() {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "FBX armature Data-Block has no bone hierarchy",
                        ));
                    }
                    None
                } else {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX adapter does not represent this data-block type",
                        json!({ "feature_id": format!("format.fbx.{}", data.data_type), "node_id": node_id }),
                    ));
                }
            }
            None => None,
        };
        let model_id = take_object_id(&mut next_id)?;
        ids.insert(
            node_id.clone(),
            ObjectIds {
                model: model_id,
                geometry: geometry_id,
            },
        );
    }

    let (bone_models, bone_model_ids) = build_bone_models(doc, &ids, &mut next_id)?;
    let (skin_exports, morph_exports) =
        build_deformers(doc, snapshot, &ids, &bone_model_ids, &mut next_id)?;
    let mut material_ids = BTreeMap::<Id, i64>::new();
    for material_id in doc.materials.keys() {
        material_ids.insert(material_id.clone(), take_object_id(&mut next_id)?);
    }
    let texture_exports = build_texture_exports(doc, project_root, &mut next_id)?;

    let mut stacks = BTreeMap::<Id, (i64, i64)>::new();
    let mut curve_exports = Vec::<CurveExport>::new();
    let mut curve_node_ids = BTreeMap::<(Id, Id, String), i64>::new();
    for (node_id, node) in &doc.nodes {
        let Some(action_id) = &node.action else {
            continue;
        };
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "node references a missing action",
                json!({ "node_id": node_id, "action_id": action_id }),
            )
        })?;
        if !stacks.contains_key(action_id) {
            stacks.insert(
                action_id.clone(),
                (take_object_id(&mut next_id)?, take_object_id(&mut next_id)?),
            );
        }
        for curve in &action.fcurves {
            if curve.keyframes.is_empty() {
                continue;
            }
            let (property, component, rotation) = match curve.path.as_str() {
                "transform.translation" => ("Lcl Translation", component_name(curve.index)?, false),
                "transform.scale" => ("Lcl Scaling", component_name(curve.index)?, false),
                "transform.rotation_euler" => ("Lcl Rotation", component_name(curve.index)?, true),
                "transform.rotation_quaternion" | "transform.rotation" => {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX local rotation animation uses Euler component curves",
                        json!({ "feature_id": "format.fbx.quaternion_animation", "node_id": node_id, "action_id": action_id }),
                    ));
                }
                path => {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX adapter supports transform animation curves only",
                        json!({ "feature_id": format!("format.fbx.animation.{path}"), "node_id": node_id, "action_id": action_id }),
                    ));
                }
            };
            let mut keys = curve.keyframes.clone();
            keys.sort_by(|left, right| left.frame.total_cmp(&right.frame));
            if keys
                .iter()
                .any(|key| !key.frame.is_finite() || !key.value.is_finite())
                || keys
                    .windows(2)
                    .any(|pair| crate::float::equal_f64(pair[0].frame, pair[1].frame))
            {
                return Err(PotError::with_details(
                    ErrorCode::ExportFailed,
                    "FBX animation keys must have unique finite frame and value pairs",
                    json!({ "node_id": node_id, "action_id": action_id, "path": curve.path, "index": curve.index }),
                ));
            }
            let curve_node_key = (action_id.clone(), node_id.clone(), property.to_owned());
            let curve_node_id = if let Some(id) = curve_node_ids.get(&curve_node_key) {
                *id
            } else {
                let id = take_object_id(&mut next_id)?;
                curve_node_ids.insert(curve_node_key, id);
                id
            };
            curve_exports.push(CurveExport {
                id: take_object_id(&mut next_id)?,
                curve_node_id,
                component,
                keys,
                rotation,
            });
        }
    }

    let mut output = String::with_capacity(32_768);
    output.push_str(
        "; FBX 7.4.0 project file\nFBXHeaderExtension:  {\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n\tCreationTimeStamp:  {\n\t\tVersion: 1000\n\t}\n}\n",
    );
    writeln!(
        output,
        "Documents:  {{\n\tCount: 1\n\tDocument: 1000000, \"{}\", \"Scene\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"SourceObject\", \"object\", \"\", \"\"\n\t\t\tP: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"AnimStack::\"\n\t\t}}\n\t\tRootNode: 0\n\t}}\n}}\nReferences:  {{\n}}\n",
        quote(&scene.name)
    )
    .map_err(write_error)?;
    writeln!(
        output,
        "GlobalSettings:  {{\n\tVersion: 1000\n\tProperties70:  {{\n\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",2\n\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",1\n\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",-1\n\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n\t\tP: \"OriginalUpAxis\", \"int\", \"Integer\", \"\",2\n\t\tP: \"OriginalUpAxisSign\", \"int\", \"Integer\", \"\",1\n\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",100\n\t\tP: \"CustomFrameRate\", \"double\", \"Number\", \"\",{}\n\t}}\n}}",
        fmt_num(fps)
    )
    .map_err(write_error)?;

    let deformer_count = skin_exports.len()
        + skin_exports
            .iter()
            .map(|skin| skin.clusters.len())
            .sum::<usize>()
        + morph_exports.len() * 2;
    let object_count = ids.len()
        + bone_models.len()
        + ids.values().filter(|ids| ids.geometry.is_some()).count()
        + morph_exports.len()
        + material_ids.len()
        + texture_exports.len()
        + deformer_count
        + stacks.len() * 2
        + curve_node_ids.len()
        + curve_exports.len();
    writeln!(
        output,
        "Definitions:  {{\n\tVersion: 100\n\tCount: {object_count}\n\tObjectType: \"Model\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"Geometry\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"Material\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"Texture\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"Deformer\" {{\n\t\tCount: {deformer_count}\n\t}}\n\tObjectType: \"AnimationStack\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"AnimationLayer\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"AnimationCurveNode\" {{\n\t\tCount: {}\n\t}}\n\tObjectType: \"AnimationCurve\" {{\n\t\tCount: {}\n\t}}\n}}\n",
        ids.len() + bone_models.len(),
        ids.values().filter(|ids| ids.geometry.is_some()).count() + morph_exports.len(),
        material_ids.len(),
        texture_exports.len(),
        stacks.len(),
        stacks.len(),
        curve_node_ids.len(),
        curve_exports.len()
    )
    .map_err(write_error)?;

    output.push_str("Objects:  {\n");
    for (node_id, node) in &doc.nodes {
        let object_ids = ids.get(node_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "FBX model ID was not allocated")
        })?;
        if let Some(geometry_id) = object_ids.geometry {
            let data_block = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id));
            let has_deformations = data_block.is_some_and(|data| {
                data.shape_keys
                    .as_ref()
                    .is_some_and(|keys| !keys.keys.is_empty())
                    || !data.vertex_groups.is_empty()
                    || !data.vertex_weights.is_empty()
            });
            let mesh = if has_deformations {
                data_block
                    .and_then(|data| data.mesh.as_ref())
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "FBX deformation source mesh is missing",
                        )
                    })?
            } else {
                snapshot.meshes.get(node_id).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "evaluated mesh disappeared during FBX export",
                    )
                })?
            };
            write_geometry(&mut output, geometry_id, node, mesh, data_block)?;
        }
        let mesh_model = object_ids.geometry.is_some();
        let rotation = DQuat::from_xyzw(
            node.transform.rotation[0],
            node.transform.rotation[1],
            node.transform.rotation[2],
            node.transform.rotation[3],
        );
        if node
            .transform
            .rotation
            .iter()
            .any(|value| !value.is_finite())
            || rotation.length_squared() < f64::EPSILON
        {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "node rotation is not a finite nonzero quaternion",
                json!({ "node_id": node_id }),
            ));
        }
        let (rx, ry, rz) = rotation.to_euler(EulerRot::XYZ);
        let armature_metadata = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.armature.as_ref())
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| {
                PotError::with_details(
                    ErrorCode::ExportFailed,
                    "could not encode FBX armature hierarchy",
                    json!({"reason":error.to_string()}),
                )
            })?
            .unwrap_or_default();
        writeln!(
            output,
            "\tModel: {}, \"{}\", \"{}\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Visibility\", \"Visibility\", \"\",\"A\",{}\n\t\t\tP: \"potter:id\", \"KString\", \"\", \"\", \"{}\"\n\t\t\tP: \"potter:armature\", \"KString\", \"\", \"\", \"{}\"\n\t\t}}\n\t\tShading: T\n\t\tCulling: \"CullingOff\"\n\t}}",
            object_ids.model,
            quote(&format!("Model::{}", node.name)),
            if mesh_model { "Mesh" } else { "Null" },
            fmt_num(node.transform.translation[0]),
            fmt_num(node.transform.translation[1]),
            fmt_num(node.transform.translation[2]),
            fmt_num(rx.to_degrees()),
            fmt_num(ry.to_degrees()),
            fmt_num(rz.to_degrees()),
            fmt_num(node.transform.scale[0]),
            fmt_num(node.transform.scale[1]),
            fmt_num(node.transform.scale[2]),
            i32::from(node.visible && node.render_visible),
            quote(node_id.as_str()),
            quote(&armature_metadata),
        )
        .map_err(write_error)?;
    }

    for bone in &bone_models {
        write_bone_model(&mut output, bone)?;
    }
    for skin in &skin_exports {
        write_skin_export(&mut output, skin)?;
    }
    for morph in &morph_exports {
        write_morph_export(&mut output, morph)?;
    }

    for (material_id, material) in &doc.materials {
        let object_id = material_ids.get(material_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "FBX material ID was not allocated",
            )
        })?;
        writeln!(
            output,
            "\tMaterial: {}, \"{}\", \"\" {{\n\t\tVersion: 102\n\t\tShadingModel: \"phong\"\n\t\tMultiLayer: 0\n\t\tProperties70:  {{\n\t\t\tP: \"DiffuseColor\", \"Color\", \"\",\"A\",{},{},{}\n\t\t\tP: \"DiffuseFactor\", \"Number\", \"\",\"A\",1\n\t\t\tP: \"Opacity\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:id\", \"KString\", \"\", \"\", \"{}\"\n\t\t\tP: \"potter:metallic\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:roughness\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:emission_color\", \"Color\", \"\",\"A\",{},{},{}\n\t\t\tP: \"potter:emission_strength\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:transmission\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:ior\", \"Number\", \"\",\"A\",{}\n\t\t\tP: \"potter:double_sided\", \"bool\", \"Boolean\", \"\",{}\n\t\t}}\n\t}}",
            object_id,
            quote(&format!("Material::{}", material.name)),
            fmt_num(material.base_color[0]),
            fmt_num(material.base_color[1]),
            fmt_num(material.base_color[2]),
            fmt_num(material.base_color[3]),
            quote(material_id.as_str()),
            fmt_num(material.metallic),
            fmt_num(material.roughness),
            fmt_num(material.emission_color[0]),
            fmt_num(material.emission_color[1]),
            fmt_num(material.emission_color[2]),
            fmt_num(material.emission_strength),
            fmt_num(material.transmission),
            fmt_num(material.ior),
            i32::from(material.double_sided),
        )
        .map_err(write_error)?;
    }

    for texture in &texture_exports {
        let material = doc.materials.get(&texture.material_id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "FBX texture material is missing")
        })?;
        writeln!(
            output,
            "\tTexture: {}, \"Texture::{}\", \"TextureVideoClip\" {{\n\t\tType: \"TextureVideoClip\"\n\t\tVersion: 202\n\t\tTextureName: \"Texture::{}\"\n\t\tProperties70:  {{\n\t\t\tP: \"UVSet\", \"KString\", \"\", \"\", \"{}\"\n\t\t\tP: \"potter:property\", \"KString\", \"\", \"\", \"{}\"\n\t\t}}\n\t\tFileName: \"{}\"\n\t\tRelativeFilename: \"{}\"\n\t\tModelUVTranslation: 0,0\n\t\tModelUVScaling: 1,1\n\t\tTexture_Alpha_Source: \"None\"\n\t\tCropping: 0,0,0,0\n\t}}",
            texture.id,
            quote(&material.name),
            quote(&material.name),
            quote(texture.uv_map.as_deref().unwrap_or("")),
            texture.property,
            quote(&texture.path),
            quote(&texture.path),
        )
        .map_err(write_error)?;
    }

    for (action_id, (stack_id, layer_id)) in &stacks {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "FBX animation action disappeared")
        })?;
        let mut range = None::<(i64, i64)>;
        for key in action.fcurves.iter().flat_map(|curve| &curve.keyframes) {
            let time = frame_to_ticks(key.frame, fps)?;
            range = Some(match range {
                Some((start, stop)) => (start.min(time), stop.max(time)),
                None => (time, time),
            });
        }
        let (start, stop) = range.unwrap_or((0, 0));
        writeln!(
            output,
            "\tAnimationStack: {}, \"{}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",{}\n\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",{}\n\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",{}\n\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",{}\n\t\t\tP: \"potter:id\", \"KString\", \"\", \"\", \"{}\"\n\t\t}}\n\t}}\n\tAnimationLayer: {}, \"{}\", \"\" {{\n\t\tVersion: 100\n\t\tWeight: 100\n\t\tMute: 0\n\t\tSolo: 0\n\t\tLock: 0\n\t\tBlendMode: 0\n\t}}",
            stack_id,
            quote(&format!("AnimationStack::{}", action.name)),
            start,
            stop,
            start,
            stop,
            quote(action_id.as_str()),
            layer_id,
            quote("AnimationLayer::BaseLayer")
        )
        .map_err(write_error)?;
    }
    for ((_, _, property), curve_node_id) in &curve_node_ids {
        let name = match property.as_str() {
            "Lcl Translation" => "T",
            "Lcl Rotation" => "R",
            "Lcl Scaling" => "S",
            _ => {
                return Err(PotError::new(
                    ErrorCode::InternalError,
                    "unknown FBX TRS property",
                ));
            }
        };
        writeln!(
            output,
            "\tAnimationCurveNode: {curve_node_id}, \"AnimCurveNode::{name}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"d|X\", \"Number\", \"\",\"A\",0\n\t\t\tP: \"d|Y\", \"Number\", \"\",\"A\",0\n\t\t\tP: \"d|Z\", \"Number\", \"\",\"A\",0\n\t\t}}\n\t}}"
        )
        .map_err(write_error)?;
    }
    for curve in &curve_exports {
        write_animation_curve(&mut output, curve, fps)?;
    }
    output.push_str("}\nConnections:  {\n");
    for (node_id, node) in &doc.nodes {
        let object_ids = ids.get(node_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "FBX model ID was not allocated")
        })?;
        if let Some(geometry_id) = object_ids.geometry {
            writeln!(output, "\tC: \"OO\",{geometry_id},{}", object_ids.model)
                .map_err(write_error)?;
        }
        if let Some(parent_id) = &node.parent {
            let parent = ids.get(parent_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "node parent was not included in FBX export",
                    json!({ "node_id": node_id, "parent_id": parent_id }),
                )
            })?;
            writeln!(output, "\tC: \"OO\",{},{}", object_ids.model, parent.model)
                .map_err(write_error)?;
        } else {
            writeln!(output, "\tC: \"OO\",{},0", object_ids.model).map_err(write_error)?;
        }
        for material_id in &node.materials {
            let material = material_ids.get(material_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "node references a material absent from the FBX export",
                    json!({ "node_id": node_id, "material_id": material_id }),
                )
            })?;
            writeln!(output, "\tC: \"OO\",{material},{}", object_ids.model).map_err(write_error)?;
        }
    }
    for bone in &bone_models {
        writeln!(
            output,
            "\tC: \"OO\",{},{}",
            bone.model_id, bone.parent_model
        )
        .map_err(write_error)?;
    }
    for skin in &skin_exports {
        writeln!(output, "\tC: \"OO\",{},{}", skin.id, skin.geometry).map_err(write_error)?;
        for cluster in &skin.clusters {
            writeln!(output, "\tC: \"OO\",{},{}", cluster.id, skin.id).map_err(write_error)?;
            writeln!(output, "\tC: \"OO\",{},{}", cluster.id, cluster.bone_model)
                .map_err(write_error)?;
        }
    }
    for morph in &morph_exports {
        writeln!(output, "\tC: \"OO\",{},{}", morph.blend_id, morph.geometry)
            .map_err(write_error)?;
        writeln!(
            output,
            "\tC: \"OO\",{},{}",
            morph.channel_id, morph.blend_id
        )
        .map_err(write_error)?;
        writeln!(
            output,
            "\tC: \"OO\",{},{}",
            morph.shape_id, morph.channel_id
        )
        .map_err(write_error)?;
    }
    for texture in &texture_exports {
        let material_id = material_ids.get(&texture.material_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "FBX texture material ID is missing",
            )
        })?;
        writeln!(
            output,
            "\tC: \"OP\",{}, {},\"{}\"",
            texture.id, material_id, texture.property
        )
        .map_err(write_error)?;
    }
    for (action_id, (stack_id, layer_id)) in &stacks {
        writeln!(output, "\tC: \"OO\",{layer_id},{stack_id}").map_err(write_error)?;
        for ((curve_action, node_id, property), curve_node_id) in &curve_node_ids {
            if curve_action != action_id {
                continue;
            }
            let layer = *layer_id;
            writeln!(output, "\tC: \"OO\",{curve_node_id},{layer}").map_err(write_error)?;
            let model = ids
                .get(node_id)
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "FBX animation model is missing")
                })?
                .model;
            writeln!(output, "\tC: \"OP\",{curve_node_id},{model},\"{property}\"")
                .map_err(write_error)?;
        }
    }
    for curve in &curve_exports {
        writeln!(
            output,
            "\tC: \"OP\",{}, {},\"d|{}\"",
            curve.id, curve.curve_node_id, curve.component
        )
        .map_err(write_error)?;
    }
    output.push_str("}\n\nTakes:  {\n\tCurrent: \"\"\n}\n");
    Ok(output.into_bytes())
}

fn build_texture_exports(
    doc: &SceneDoc,
    root: &Path,
    next_id: &mut i64,
) -> Result<Vec<TextureExport>> {
    let mut output = Vec::new();
    for (material_id, material) in &doc.materials {
        if material
            .node_tree
            .as_ref()
            .is_some_and(|graph_id| !super::is_default_material_graph(doc, graph_id))
        {
            return Err(PotError::with_details(
                ErrorCode::UnrepresentableFeature,
                "FBX cannot preserve arbitrary material node trees",
                json!({"feature_id":"format.fbx.material_node_tree","material_id":material_id}),
            ));
        }
        for (texture, property) in [
            (&material.base_color_texture, "DiffuseColor"),
            (&material.roughness_texture, "Shininess"),
            (&material.metallic_texture, "ReflectionColor"),
            (&material.normal_texture, "NormalMap"),
        ] {
            let Some(texture) = texture else {
                continue;
            };
            let image = doc.images.get(&texture.image).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "FBX material texture references a missing image",
                    json!({"material_id":material_id,"image_id":texture.image}),
                )
            })?;
            if !image.tiles.is_empty() {
                return Err(PotError::with_details(
                    ErrorCode::UnrepresentableFeature,
                    "FBX texture export does not support image tiles",
                    json!({"feature_id":"format.fbx.texture_udim","material_id":material_id}),
                ));
            }
            crate::image::load_image_data(image, root, texture.interpolation)?;
            let path = match image.source {
                ImageSource::File => {
                    let source = image.source_path.as_deref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::UnrepresentableFeature,
                            "FBX texture requires a file-backed or packed image",
                            json!({"feature_id":"format.fbx.texture_missing_path","image_id":texture.image}),
                        )
                    })?;
                    let source = Path::new(source);
                    if source.is_absolute() {
                        source.to_path_buf()
                    } else {
                        root.join(source)
                    }
                }
                ImageSource::Packed => {
                    let blob = image.blob.as_deref().ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::UnrepresentableFeature,
                            "FBX packed texture is missing its asset blob",
                            json!({"feature_id":"format.fbx.texture_missing_blob","image_id":texture.image}),
                        )
                    })?;
                    let digest = blob.strip_prefix("sha256:").ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "FBX image blob hash is invalid")
                    })?;
                    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "FBX image blob hash is invalid",
                        ));
                    }
                    root.join("assets").join("sha256").join(digest).join("blob")
                }
                ImageSource::Generated => {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX cannot reference generated pixel-store images as external texture files",
                        json!({"feature_id":"format.fbx.texture_generated_image","image_id":texture.image}),
                    ));
                }
            };
            let path = fs::canonicalize(path).map_err(|error| PotError::io(&error))?;
            output.push(TextureExport {
                id: take_object_id(next_id)?,
                material_id: material_id.clone(),
                property,
                path: path.to_string_lossy().into_owned(),
                uv_map: texture.uv_map.clone(),
            });
        }
    }
    Ok(output)
}

type BoneModelIds = BTreeMap<(Id, Id), i64>;

fn build_bone_models(
    doc: &SceneDoc,
    object_ids: &BTreeMap<Id, ObjectIds>,
    next_id: &mut i64,
) -> Result<(Vec<BoneExport>, BoneModelIds)> {
    let mut output = Vec::new();
    let mut model_ids = BTreeMap::<(Id, Id), i64>::new();
    for (node_id, node) in &doc.nodes {
        let Some(data_id) = &node.data else {
            continue;
        };
        let Some(armature) = doc
            .data_blocks
            .get(data_id)
            .and_then(|data| data.armature.as_ref())
        else {
            continue;
        };
        let object_model = object_ids
            .get(node_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "FBX armature model is missing")
            })?
            .model;
        for bone_id in armature.bones.keys() {
            model_ids.insert((node_id.clone(), bone_id.clone()), take_object_id(next_id)?);
        }
        let matrices = crate::eval::rig::evaluate_bone_matrices(armature, &BTreeMap::new())?;
        for (bone_id, bone) in &armature.bones {
            let rest = matrices
                .get(bone_id)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "FBX rest bone matrix is missing",
                    )
                })?
                .rest;
            let local = if let Some(parent_id) = &bone.parent {
                let parent = matrices
                    .get(parent_id)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "FBX bone parent is missing")
                    })?
                    .rest;
                parent.inverse() * rest
            } else {
                rest
            };
            let parent_model = if let Some(parent_id) = &bone.parent {
                *model_ids
                    .get(&(node_id.clone(), parent_id.clone()))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "FBX parent bone model is missing")
                    })?
            } else {
                object_model
            };
            output.push(BoneExport {
                model_id: *model_ids
                    .get(&(node_id.clone(), bone_id.clone()))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "FBX bone model ID is missing")
                    })?,
                parent_model,
                bone_id: bone_id.clone(),
                bone: bone.clone(),
                local,
            });
        }
    }
    Ok((output, model_ids))
}

fn write_bone_model(output: &mut String, bone: &BoneExport) -> Result<()> {
    let (scale, rotation, translation) = bone.local.to_scale_rotation_translation();
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::UnrepresentableFeature,
            "FBX bone local transform is not finite",
            json!({"feature_id":"format.fbx.skin_bone_transform","bone_id":bone.bone_id}),
        ));
    }
    let (rx, ry, rz) = rotation.to_euler(EulerRot::XYZ);
    writeln!(
        output,
        "\tModel: {}, \"{}\", \"LimbNode\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\",\"A\",{},{},{}\n\t\t\tP: \"Size\", \"double\", \"Number\", \"\",{}\n\t\t}}\n\t\tShading: T\n\t\tCulling: \"CullingOff\"\n\t}}",
        bone.model_id,
        quote(&format!("Model::{}", bone.bone.name)),
        fmt_num(translation.x),
        fmt_num(translation.y),
        fmt_num(translation.z),
        fmt_num(rx.to_degrees()),
        fmt_num(ry.to_degrees()),
        fmt_num(rz.to_degrees()),
        fmt_num(scale.x),
        fmt_num(scale.y),
        fmt_num(scale.z),
        fmt_num(DVec3::from_array(bone.bone.head).distance(DVec3::from_array(bone.bone.tail))),
    )
    .map_err(write_error)
}

fn build_deformers(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    object_ids: &BTreeMap<Id, ObjectIds>,
    bone_model_ids: &BTreeMap<(Id, Id), i64>,
    next_id: &mut i64,
) -> Result<(Vec<SkinExport>, Vec<MorphExport>)> {
    let mut skins = Vec::new();
    let mut morphs = Vec::new();
    for (node_id, node) in &doc.nodes {
        let Some(object) = object_ids.get(node_id) else {
            continue;
        };
        let Some(geometry) = object.geometry else {
            continue;
        };
        let Some(data_id) = &node.data else {
            continue;
        };
        let data = doc.data_blocks.get(data_id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "FBX mesh Data-Block is missing")
        })?;
        let source_mesh = data.mesh.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "FBX mesh Data-Block has no source mesh",
            )
        })?;
        let enabled = node
            .modifiers
            .iter()
            .filter(|modifier| modifier.enabled)
            .collect::<Vec<_>>();
        let skin_modifiers = enabled
            .iter()
            .filter(|modifier| modifier.modifier_type == "armature")
            .copied()
            .collect::<Vec<_>>();
        if !skin_modifiers.is_empty() {
            if skin_modifiers.len() != 1
                || enabled
                    .iter()
                    .any(|modifier| modifier.modifier_type != "armature")
            {
                return Err(PotError::with_details(
                    ErrorCode::UnrepresentableFeature,
                    "FBX skin export requires one armature modifier and no other enabled modifiers",
                    json!({"feature_id":"format.fbx.skin_modifier_stack","node_id":node_id}),
                ));
            }
            let armature_raw = skin_modifiers[0]
                .params
                .get("object")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "FBX armature modifier has no target",
                    )
                })?;
            let armature_id = Id::new(armature_raw).map_err(|_| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "FBX armature modifier target is invalid",
                )
            })?;
            let armature_node = doc.nodes.get(&armature_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "FBX armature modifier target is missing",
                )
            })?;
            let armature_data_id = armature_node.data.as_ref().ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "FBX armature object has no Data-Block",
                )
            })?;
            let armature = doc
                .data_blocks
                .get(armature_data_id)
                .and_then(|armature_data| armature_data.armature.as_ref())
                .ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "FBX armature hierarchy is missing")
                })?;
            let bone_matrices =
                crate::eval::rig::evaluate_bone_matrices(armature, &BTreeMap::new())?;
            let mesh_world = snapshot
                .nodes
                .get(node_id)
                .map(|state| DMat4::from_cols_array(&state.world_matrix))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "FBX mesh world transform is missing",
                    )
                })?;
            let armature_world = snapshot
                .nodes
                .get(&armature_id)
                .map(|state| DMat4::from_cols_array(&state.world_matrix))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "FBX armature world transform is missing",
                    )
                })?;
            let valid_groups = data
                .vertex_groups
                .iter()
                .map(|group| (group.id.clone(), group.name.as_str()))
                .collect::<BTreeMap<_, _>>();
            let mut clusters = Vec::new();
            for (bone_id, bone) in &armature.bones {
                if !bone.deform {
                    continue;
                }
                let matching_groups = data
                    .vertex_groups
                    .iter()
                    .filter(|group| group.name == bone.name)
                    .collect::<Vec<_>>();
                if matching_groups.len() > 1 {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX skin cannot map duplicate vertex-group names to a unique joint",
                        json!({"feature_id":"format.fbx.skin_duplicate_group","node_id":node_id,"bone_id":bone_id}),
                    ));
                }
                let Some(group) = matching_groups.first() else {
                    continue;
                };
                let mut indexes = Vec::new();
                let mut weights = Vec::new();
                for (index, vertex) in source_mesh.vertices.iter().enumerate() {
                    let weight = data
                        .vertex_weights
                        .get(&vertex.id)
                        .and_then(|vertex_weights| vertex_weights.get(&group.id))
                        .copied()
                        .unwrap_or(0.0);
                    if !weight.is_finite() || weight < 0.0 {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "FBX skin weight must be finite and nonnegative",
                        ));
                    }
                    if weight > 0.0 {
                        indexes.push(index);
                        weights.push(weight);
                    }
                }
                for vertex_weights in data.vertex_weights.values() {
                    for (group_id, weight) in vertex_weights {
                        if !valid_groups.contains_key(group_id) {
                            return Err(PotError::new(
                                ErrorCode::SceneInvalid,
                                "FBX skin weight references a missing vertex group",
                            ));
                        }
                        if !weight.is_finite() || *weight < 0.0 {
                            return Err(PotError::new(
                                ErrorCode::SceneInvalid,
                                "FBX skin weight must be finite and nonnegative",
                            ));
                        }
                    }
                }
                if indexes.is_empty() {
                    continue;
                }
                let bone_model = *bone_model_ids
                    .get(&(armature_id.clone(), bone_id.clone()))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "FBX deform bone model is missing")
                    })?;
                let rest = bone_matrices
                    .get(bone_id)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "FBX deform bone rest matrix is missing",
                        )
                    })?
                    .rest;
                clusters.push(ClusterExport {
                    id: take_object_id(next_id)?,
                    bone_model,
                    indexes,
                    weights,
                    mesh_bind: mesh_world,
                    bone_bind: armature_world * rest,
                });
            }
            skins.push(SkinExport {
                id: take_object_id(next_id)?,
                geometry,
                clusters,
            });
        }
        if let Some(shape_keys) = data
            .shape_keys
            .as_ref()
            .filter(|shape_keys| !shape_keys.keys.is_empty())
        {
            if enabled
                .iter()
                .any(|modifier| modifier.modifier_type != "armature")
            {
                return Err(PotError::with_details(
                    ErrorCode::UnrepresentableFeature,
                    "FBX shape-key export cannot preserve shape keys through a non-armature modifier",
                    json!({"feature_id":"format.fbx.shape_key_modifier_stack","node_id":node_id}),
                ));
            }
            for (key_id, key) in &shape_keys.keys {
                if key.relative_key.is_some() {
                    return Err(PotError::with_details(
                        ErrorCode::UnrepresentableFeature,
                        "FBX blend shapes cannot preserve non-basis relative keys",
                        json!({"feature_id":"format.fbx.shape_key_relative","node_id":node_id,"key_id":key_id}),
                    ));
                }
                let mut indices = Vec::new();
                let mut offsets = Vec::new();
                for (index, vertex) in source_mesh.vertices.iter().enumerate() {
                    let base = shape_keys
                        .basis
                        .get(&vertex.id)
                        .copied()
                        .unwrap_or(vertex.co.to_array());
                    let target = key.positions.get(&vertex.id).copied().unwrap_or(base);
                    let mut offset = DVec3::from_array(target) - DVec3::from_array(base);
                    if let Some(group_id) = &key.vertex_group {
                        let weight = data
                            .vertex_weights
                            .get(&vertex.id)
                            .and_then(|weights| weights.get(group_id))
                            .copied()
                            .unwrap_or(0.0);
                        if !weight.is_finite() || weight < 0.0 {
                            return Err(PotError::new(
                                ErrorCode::SceneInvalid,
                                "FBX shape key vertex-group weight is invalid",
                            ));
                        }
                        offset *= weight;
                    }
                    if !offset.is_finite() {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "FBX shape key offset is not finite",
                        ));
                    }
                    if offset.length_squared() > 0.0 {
                        indices.push(index);
                        offsets.push(offset.to_array());
                    }
                }
                morphs.push(MorphExport {
                    blend_id: take_object_id(next_id)?,
                    channel_id: take_object_id(next_id)?,
                    shape_id: take_object_id(next_id)?,
                    geometry,
                    name: key.name.clone(),
                    value: key.value,
                    indices,
                    offsets,
                });
            }
        }
    }
    Ok((skins, morphs))
}

fn write_skin_export(output: &mut String, skin: &SkinExport) -> Result<()> {
    writeln!(
        output,
        "\tDeformer: {}, \"Deformer::Skin\", \"Skin\" {{\n\t\tVersion: 101\n\t\tLink_DeformAcuracy: 50\n\t}}",
        skin.id
    )
    .map_err(write_error)?;
    for cluster in &skin.clusters {
        writeln!(
            output,
            "\tDeformer: {}, \"SubDeformer::Cluster\", \"Cluster\" {{\n\t\tVersion: 100\n\t\tUserData: \"\", \"\"\n\t\tIndexes: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tWeights: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tTransform: *16 {{\n\t\t\ta: {}\n\t\t}}\n\t\tTransformLink: *16 {{\n\t\t\ta: {}\n\t\t}}\n\t}}",
            cluster.id,
            cluster.indexes.len(),
            cluster.indexes.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
            cluster.weights.len(),
            cluster.weights.iter().map(|weight| fmt_num(*weight)).collect::<Vec<_>>().join(","),
            cluster.mesh_bind.to_cols_array().iter().map(|value| fmt_num(*value)).collect::<Vec<_>>().join(","),
            cluster.bone_bind.to_cols_array().iter().map(|value| fmt_num(*value)).collect::<Vec<_>>().join(","),
        )
        .map_err(write_error)?;
    }
    Ok(())
}

fn write_morph_export(output: &mut String, morph: &MorphExport) -> Result<()> {
    writeln!(
        output,
        "\tDeformer: {}, \"Deformer::{}\", \"BlendShape\" {{\n\t\tVersion: 100\n\t}}",
        morph.blend_id,
        quote(&morph.name),
    )
    .map_err(write_error)?;
    writeln!(
        output,
        "\tDeformer: {}, \"SubDeformer::{}\", \"BlendShapeChannel\" {{\n\t\tVersion: 100\n\t\tDeformPercent: {}\n\t\tFullWeights: *1 {{\n\t\t\ta: 100\n\t\t}}\n\t}}",
        morph.channel_id,
        quote(&morph.name),
        fmt_num(morph.value * 100.0),
    )
    .map_err(write_error)?;
    write!(
        output,
        "\tGeometry: {}, \"Geometry::{}\", \"Shape\" {{\n\t\tVersion: 100\n\t\tIndexes: *{} {{\n\t\t\ta: ",
        morph.shape_id,
        quote(&morph.name),
        morph.indices.len(),
    )
    .map_err(write_error)?;
    output.push_str(
        &morph
            .indices
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    write!(
        output,
        "\n\t\t}}\n\t\tVertices: *{} {{\n\t\t\ta: ",
        morph.offsets.len().saturating_mul(3),
    )
    .map_err(write_error)?;
    let deltas = morph
        .offsets
        .iter()
        .flatten()
        .map(|value| fmt_num(*value))
        .collect::<Vec<_>>()
        .join(",");
    output.push_str(&deltas);
    output.push_str("\n\t\t}\n\t}\n");
    Ok(())
}

fn write_geometry(
    output: &mut String,
    id: i64,
    node: &Node,
    mesh: &Mesh,
    data_block: Option<&DataBlock>,
) -> Result<()> {
    if !mesh.attributes.is_empty() {
        return Err(PotError::with_details(
            ErrorCode::UnrepresentableFeature,
            "FBX adapter does not represent mesh custom attributes",
            json!({ "feature_id": "format.fbx.mesh_attributes", "node_id": node.name }),
        ));
    }
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    writeln!(
        output,
        "\tGeometry: {id}, \"{}\", \"Mesh\" {{\n\t\tGeometryVersion: 124",
        quote(&format!("Geometry::{}", node.name))
    )
    .map_err(write_error)?;
    if let Some(data) = data_block.filter(|data| {
        data.shape_keys
            .as_ref()
            .is_some_and(|keys| !keys.keys.is_empty())
            || !data.vertex_groups.is_empty()
            || !data.vertex_weights.is_empty()
            || node
                .modifiers
                .iter()
                .any(|modifier| modifier.enabled && modifier.modifier_type == "armature")
    }) {
        let metadata = serde_json::to_string(&json!({
            "shape_keys": &data.shape_keys,
            "vertex_groups": &data.vertex_groups,
            "vertex_weights": &data.vertex_weights,
        }))
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "could not encode FBX rig and morph metadata",
                json!({"reason": error.to_string()}),
            )
        })?;
        writeln!(
            output,
            "\t\tProperties70:  {{\n\t\t\tP: \"potter:deformation\", \"KString\", \"\", \"\", \"{}\"\n\t\t}}",
            quote(&metadata)
        )
        .map_err(write_error)?;
    }
    write!(
        output,
        "\t\tVertices: *{} {{\n\t\t\ta: ",
        mesh.vertices.len().saturating_mul(3)
    )
    .map_err(write_error)?;
    let mut first = true;
    for vertex in &mesh.vertices {
        for coordinate in vertex.co.to_array() {
            if !coordinate.is_finite() {
                return Err(PotError::new(
                    ErrorCode::ExportFailed,
                    "mesh position is not finite",
                ));
            }
            if !first {
                output.push(',');
            }
            write!(output, "{}", fmt_num(coordinate)).map_err(write_error)?;
            first = false;
        }
    }
    output.push_str("\n\t\t}\n");
    let polygon_vertex_count = mesh
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "FBX polygon index count overflow")
        })?;
    write!(
        output,
        "\t\tPolygonVertexIndex: *{polygon_vertex_count} {{\n\t\t\ta: "
    )
    .map_err(write_error)?;
    let mut first = true;
    for face in &mesh.faces {
        if face.vertices.len() < 3 {
            return Err(PotError::with_details(
                ErrorCode::ExportFailed,
                "FBX polygons need at least three vertices",
                json!({"face_id":face.id}),
            ));
        }
        for (face_index, vertex_id) in face.vertices.iter().enumerate() {
            let index = *vertex_indices.get(vertex_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "mesh face references a missing vertex",
                    json!({ "face_id": face.id, "vertex_id": vertex_id }),
                )
            })?;
            let index = i64::try_from(index).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "FBX vertex index exceeds signed range",
                )
            })?;
            let encoded = if face_index + 1 == face.vertices.len() {
                index
                    .checked_neg()
                    .and_then(|value| value.checked_sub(1))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::LimitExceeded, "FBX polygon index overflow")
                    })?
            } else {
                index
            };
            if !first {
                output.push(',');
            }
            write!(output, "{encoded}").map_err(write_error)?;
            first = false;
        }
    }
    output.push_str("\n\t\t}\n");
    if !node.materials.is_empty() {
        write!(output, "\t\tLayerElementMaterial: 0 {{\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n\t\t\tMappingInformationType: \"ByPolygon\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n\t\t\tMaterials: *{} {{\n\t\t\t\ta: ", mesh.faces.len()).map_err(write_error)?;
        for (index, face) in mesh.faces.iter().enumerate() {
            if usize::try_from(face.material_index)
                .map_or(true, |slot| slot >= node.materials.len())
            {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "mesh face material slot is outside the node material list",
                    json!({"face_id":face.id,"slot":face.material_index,"material_count":node.materials.len()}),
                ));
            }
            if index != 0 {
                output.push(',');
            }
            write!(output, "{}", face.material_index).map_err(write_error)?;
        }
        output.push_str("\n\t\t\t}\n\t\t}\n\t\tLayer: 0 {\n\t\t\tVersion: 100\n\t\t\tLayerElement: {\n\t\t\t\tType: \"LayerElementMaterial\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n\t\t}\n");
    }
    output.push_str("\t}\n");
    Ok(())
}

fn write_animation_curve(output: &mut String, curve: &CurveExport, fps: f64) -> Result<()> {
    let keys = &curve.keys;
    write!(output, "\tAnimationCurve: {}, \"AnimCurve::{}\", \"\" {{\n\t\tDefault: 0\n\t\tKeyVer: 4008\n\t\tKeyTime: *{} {{\n\t\t\ta: ", curve.id, curve.component, keys.len()).map_err(write_error)?;
    for (index, key) in keys.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let time = frame_to_ticks(key.frame, fps)?;
        write!(output, "{time}").map_err(write_error)?;
    }
    output.push_str("\n\t\t}\n");
    write!(output, "\t\tKeyValueFloat: *{} {{\n\t\t\ta: ", keys.len()).map_err(write_error)?;
    for (index, key) in keys.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let value = if curve.rotation {
            key.value.to_degrees()
        } else {
            key.value
        };
        write!(output, "{}", fmt_num(value)).map_err(write_error)?;
    }
    output.push_str("\n\t\t}\n");
    write!(output, "\t\tKeyAttrFlags: *{} {{\n\t\t\ta: ", keys.len()).map_err(write_error)?;
    for (index, key) in keys.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        let flag = match key.interpolation {
            Interpolation::Constant => 2,
            Interpolation::Linear => 4,
            Interpolation::Bezier => 8,
        };
        write!(output, "{flag}").map_err(write_error)?;
    }
    output.push_str("\n\t\t}\n");
    write!(
        output,
        "\t\tKeyAttrDataFloat: *{} {{\n\t\t\ta: ",
        keys.len().saturating_mul(4)
    )
    .map_err(write_error)?;
    for index in 0..keys.len() {
        if index != 0 {
            output.push(',');
        }
        output.push_str("0,0,0,0");
    }
    output.push_str("\n\t\t}\n");
    write!(output, "\t\tKeyAttrRefCount: *{} {{\n\t\t\ta: ", keys.len()).map_err(write_error)?;
    for index in 0..keys.len() {
        if index != 0 {
            output.push(',');
        }
        output.push('1');
    }
    output.push_str("\n\t\t}\n\t}\n");
    Ok(())
}

/// Import an ASCII FBX scene into a Potter scene graph.
pub(crate) fn import(path: &Path, scene_id: String) -> Result<ImportedGraph> {
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    import_bytes_with_base(
        &bytes,
        scene_id,
        path.parent().unwrap_or_else(|| Path::new(".")),
    )
}

pub(crate) fn import_bytes(bytes: &[u8], scene_id: String) -> Result<ImportedGraph> {
    import_bytes_with_base(bytes, scene_id, Path::new("."))
}

fn import_bytes_with_base(
    bytes: &[u8],
    scene_id: String,
    source_directory: &Path,
) -> Result<ImportedGraph> {
    if bytes.starts_with(b"Kaydara FBX Binary") {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "binary FBX input is not supported; import an ASCII FBX file",
            json!({ "feature_id": "format.fbx_binary" }),
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| import_error("FBX input is not valid UTF-8 ASCII"))?;
    let tokens = lex(text)?;
    let root = parse_nodes(&tokens)?;
    let objects = root
        .iter()
        .find(|node| node.name == "Objects")
        .ok_or_else(|| import_error("FBX Objects section is missing"))?;
    let connections_node = root.iter().find(|node| node.name == "Connections");
    let connections = parse_connections(connections_node)?;

    let mut models = BTreeMap::<i64, ImportedModel>::new();
    let mut geometries = BTreeMap::<i64, ImportedGeometry>::new();
    let mut materials = BTreeMap::<i64, ImportedMaterial>::new();
    let mut textures = BTreeMap::<i64, ImportedTexture>::new();
    let mut losses = Vec::<Loss>::new();
    let mut has_deformers = false;
    for object in &objects.children {
        match object.name.as_str() {
            "Model" => {
                let id = object_id(object)?;
                if models.insert(id, parse_model(object, id)?).is_some() {
                    return Err(import_error(
                        "FBX Objects section contains a duplicate object ID",
                    ));
                }
            }
            "Geometry" => {
                let id = object_id(object)?;
                let kind = object.args.get(2).map_or("Mesh", String::as_str);
                if kind == "Shape" {
                    continue;
                }
                if kind != "Mesh" {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        "FBX geometry type is not supported",
                        json!({ "feature_id": format!("format.fbx.geometry.{kind}") }),
                    ));
                }
                let geometry = parse_geometry(object)?;
                for attribute in [
                    "LayerElementNormal",
                    "LayerElementUV",
                    "LayerElementTangent",
                    "LayerElementBinormal",
                    "LayerElementColor",
                    "LayerElementSmoothing",
                ] {
                    if object.child(attribute).is_some() {
                        losses.push(Loss {
                            feature_id: format!(
                                "format.fbx.{}",
                                attribute
                                    .trim_start_matches("LayerElement")
                                    .to_ascii_lowercase()
                            ),
                            data_id: Some(id.to_string()),
                            reason: format!(
                                "FBX {attribute} data is not represented in the Potter mesh model"
                            ),
                            suggestion: None,
                        });
                    }
                }
                if geometries.insert(id, geometry).is_some() {
                    return Err(import_error(
                        "FBX Objects section contains a duplicate object ID",
                    ));
                }
            }
            "Material" => {
                let id = object_id(object)?;
                if materials.insert(id, parse_material(object, id)?).is_some() {
                    return Err(import_error(
                        "FBX Objects section contains a duplicate object ID",
                    ));
                }
            }
            "Texture" => {
                let id = object_id(object)?;
                if textures.insert(id, parse_texture(object, id)?).is_some() {
                    return Err(import_error(
                        "FBX Objects section contains a duplicate texture ID",
                    ));
                }
            }
            "Deformer" => has_deformers = true,
            "AnimationStack" | "AnimationLayer" | "AnimationCurveNode" | "AnimationCurve" => {}
            other => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX object type is not supported",
                    json!({ "feature_id": format!("format.fbx.object_type.{other}") }),
                ));
            }
        }
    }
    if has_deformers
        && geometries
            .values()
            .all(|geometry| geometry.deformation.is_none())
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "FBX deformer data has no supported Potter feature metadata",
            json!({"feature_id":"format.fbx.deformer"}),
        ));
    }
    if models.is_empty() {
        return Err(import_error("FBX file contains no Model objects"));
    }

    let fps = find_custom_frame_rate(&root)?.unwrap_or(DEFAULT_FPS);
    if !fps.is_finite() || fps <= 0.0 {
        return Err(import_error(
            "FBX CustomFrameRate must be finite and positive",
        ));
    }
    let mut doc = SceneDoc::new(scene_id);
    if let Some(scene) = doc.scenes.get_mut(&doc.active_scene) {
        scene.fps = fps.round().clamp(1.0, f64::from(u32::MAX)) as u32;
        scene.fps_base = f64::from(scene.fps) / fps;
        if !scene.fps_base.is_finite() || scene.fps_base <= 0.0 {
            return Err(import_error(
                "FBX CustomFrameRate cannot be represented in a Potter scene",
            ));
        }
    }
    let mut used_node_ids = BTreeSet::<Id>::new();
    let mut model_ids = BTreeMap::<i64, Id>::new();
    let mut id_mappings = serde_json::Map::new();
    for model in models.values() {
        let seed = model.potter_id.as_deref().unwrap_or(&model.name);
        let node_id = unique_import_id(seed, "fbx_node", model.numeric_id, &mut used_node_ids)?;
        id_mappings.insert(model.numeric_id.to_string(), json!(node_id.as_str()));
        model_ids.insert(model.numeric_id, node_id);
    }
    let mut used_material_ids = BTreeSet::<Id>::new();
    let mut material_ids = BTreeMap::<i64, Id>::new();
    for material in materials.values() {
        let seed = material.potter_id.as_deref().unwrap_or(&material.name);
        let material_id = unique_import_id(
            seed,
            "fbx_material",
            material.numeric_id,
            &mut used_material_ids,
        )?;
        id_mappings.insert(
            format!("material:{}", material.numeric_id),
            json!(material_id.as_str()),
        );
        material_ids.insert(material.numeric_id, material_id.clone());
        doc.materials.insert(
            material_id,
            Material {
                name: material.name.clone(),
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                emission_color: material.emission_color,
                emission_strength: material.emission_strength,
                transmission: material.transmission,
                ior: material.ior,
                double_sided: false,
                ..Material::default()
            },
        );
    }

    let mut geometry_by_model = BTreeMap::<i64, i64>::new();
    let mut parent_by_model = BTreeMap::<i64, i64>::new();
    let mut material_links = BTreeMap::<i64, Vec<i64>>::new();
    for connection in &connections {
        if connection.kind != "OO" {
            continue;
        }
        if models.contains_key(&connection.parent) && geometries.contains_key(&connection.child) {
            geometry_by_model.insert(connection.parent, connection.child);
        } else if models.contains_key(&connection.child) && models.contains_key(&connection.parent)
        {
            parent_by_model.insert(connection.child, connection.parent);
        } else if models.contains_key(&connection.parent)
            && materials.contains_key(&connection.child)
        {
            material_links
                .entry(connection.parent)
                .or_default()
                .push(connection.child);
        }
    }

    let mut used_image_ids = BTreeSet::<Id>::new();
    let mut images_by_source = BTreeMap::<(String, bool), Id>::new();
    for connection in connections.iter().filter(|connection| {
        connection.kind == "OP"
            && textures.contains_key(&connection.child)
            && materials.contains_key(&connection.parent)
    }) {
        let texture = textures
            .get(&connection.child)
            .ok_or_else(|| import_error("FBX material texture connection is missing"))?;
        let source = Path::new(&texture.file_name);
        let source = if source.is_absolute() {
            source.to_path_buf()
        } else {
            source_directory.join(source)
        };
        let source = fs::canonicalize(&source).map_err(|error| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "FBX referenced texture file is unavailable",
                json!({"feature_id":"format.fbx.texture_external_file","path":source.to_string_lossy(),"reason":error.to_string()}),
            )
        })?;
        let property = connection
            .property
            .as_deref()
            .or(texture.property.as_deref())
            .ok_or_else(|| import_error("FBX material texture channel is missing"))?;
        let colorspace = if property == "NormalMap" || property == "Bump" {
            ImageColorspace::NonColor
        } else {
            ImageColorspace::Srgb
        };
        let source_key = (
            source.to_string_lossy().into_owned(),
            colorspace == ImageColorspace::NonColor,
        );
        let image_id = if let Some(image_id) = images_by_source.get(&source_key) {
            image_id.clone()
        } else {
            let bytes = fs::read(&source).map_err(|error| PotError::io(&error))?;
            let (width, height, _) = crate::image::decode_pixels(&bytes, colorspace).map_err(|error| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX referenced texture encoding is unsupported",
                    json!({"feature_id":"format.fbx.texture_codec","path":source.to_string_lossy(),"reason":error.message}),
                )
            })?;
            let seed = source
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("image");
            let image_id =
                unique_import_id(seed, "fbx_image", texture.numeric_id, &mut used_image_ids)?;
            doc.images.insert(
                image_id.clone(),
                Image {
                    name: seed.to_owned(),
                    source: ImageSource::File,
                    colorspace,
                    width,
                    height,
                    tiles: Vec::new(),
                    blob: None,
                    source_path: Some(source.to_string_lossy().into_owned()),
                    source_hash: Some(crate::hash::sha256(&bytes)),
                    alpha_mode: ImageAlphaMode::Straight,
                },
            );
            images_by_source.insert(source_key, image_id.clone());
            image_id
        };
        let material_id = material_ids
            .get(&connection.parent)
            .ok_or_else(|| import_error("FBX texture material mapping is missing"))?;
        let material = doc
            .materials
            .get_mut(material_id)
            .ok_or_else(|| import_error("FBX texture material was not imported"))?;
        let texture_ref = TextureRef {
            image: image_id,
            uv_map: texture.uv_map.clone(),
            interpolation: crate::image::ImageInterpolation::Linear,
        };
        match property {
            "DiffuseColor" => material.base_color_texture = Some(texture_ref),
            "Shininess" => material.roughness_texture = Some(texture_ref),
            "ReflectionColor" => material.metallic_texture = Some(texture_ref),
            "NormalMap" | "Bump" => material.normal_texture = Some(texture_ref),
            other => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX material texture channel is unsupported",
                    json!({"feature_id":format!("format.fbx.texture_channel.{other}")}),
                ));
            }
        }
    }
    let root_collection = Id::from_static("collection_root");
    let mut root_nodes = Vec::new();
    for (numeric_id, model) in &models {
        let node_id = model_ids
            .get(numeric_id)
            .cloned()
            .ok_or_else(|| import_error("FBX model ID mapping is missing"))?;
        let parent = parent_by_model
            .get(numeric_id)
            .and_then(|parent| model_ids.get(parent).cloned());
        let mut node_materials = Vec::<Id>::new();
        if let Some(material_numbers) = material_links.get(numeric_id) {
            for material_number in material_numbers {
                if let Some(id) = material_ids.get(material_number)
                    && !node_materials.contains(id)
                {
                    node_materials.push(id.clone());
                }
            }
        }
        let mut data_id = None;
        if let Some(geometry_number) = geometry_by_model.get(numeric_id) {
            let geometry = geometries
                .get(geometry_number)
                .ok_or_else(|| import_error("FBX model references missing mesh geometry"))?;
            let mut mesh = Mesh::from_positions_and_faces(
                geometry
                    .positions
                    .iter()
                    .map(|point| glam::DVec3::from_array(*point))
                    .collect(),
                geometry.faces.clone(),
            )
            .map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "FBX mesh topology is invalid",
                    json!({"reason":error.to_string(),"geometry_id":geometry_number}),
                )
            })?;
            for (face_index, face) in mesh.faces.iter_mut().enumerate() {
                let slot = geometry
                    .material_indices
                    .get(face_index)
                    .copied()
                    .unwrap_or(0);
                if (node_materials.is_empty() && slot != 0)
                    || (!node_materials.is_empty() && slot as usize >= node_materials.len())
                {
                    return Err(PotError::with_details(
                        ErrorCode::ImportFailed,
                        "FBX face references a missing material slot",
                        json!({"model_id":numeric_id,"face_index":face_index,"slot":slot}),
                    ));
                }
                face.material_index = slot;
            }
            let mut data_block = DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            };
            data_block.shape_keys =
                decode_deformation_field(geometry.deformation.as_ref(), "shape_keys")?;
            data_block.vertex_groups =
                decode_deformation_field(geometry.deformation.as_ref(), "vertex_groups")?
                    .unwrap_or_default();
            data_block.vertex_weights =
                decode_deformation_field(geometry.deformation.as_ref(), "vertex_weights")?
                    .unwrap_or_default();
            let next_data_id = unique_data_id(&doc, &node_id)?;
            doc.data_blocks.insert(next_data_id.clone(), data_block);
            data_id = Some(next_data_id);
        }
        if data_id.is_none()
            && let Some(armature) = &model.armature
        {
            let armature_data_id = unique_data_id(&doc, &node_id)?;
            doc.data_blocks.insert(
                armature_data_id.clone(),
                DataBlock {
                    data_type: "armature".to_owned(),
                    descriptor: None,
                    mesh: None,
                    camera: None,
                    light: None,
                    armature: Some(armature.clone()),
                    ..DataBlock::default()
                },
            );
            data_id = Some(armature_data_id);
        }
        let node = Node {
            name: model.name.clone(),
            kind: if model.armature.is_some() {
                "armature".to_owned()
            } else if data_id.is_some() {
                "mesh".to_owned()
            } else {
                "empty".to_owned()
            },
            primitive: None,
            tags: Vec::new(),
            parent: parent.clone(),
            parent_inverse: None,
            transform: model.transform.clone(),
            data: data_id,
            materials: node_materials,
            modifiers: Vec::new(),
            visible: model.visible,
            render_visible: model.visible,
            selectable: true,
            action: None,
            properties: serde_json::Map::new(),
            ..Node::default()
        };
        if parent.is_none() {
            root_nodes.push(node_id.clone());
        }
        doc.nodes.insert(node_id, node);
    }
    doc.collections
        .get_mut(&root_collection)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "root collection is missing"))?
        .objects = root_nodes;
    import_animations(&root, &connections, &models, &model_ids, fps, &mut doc)?;
    doc.validate()?;
    Ok(ImportedGraph {
        doc,
        losses,
        id_mappings: Value::Object(id_mappings),
        assets: Vec::new(),
        compat_blobs: Vec::new(),
        source: json!({ "format": "fbx", "version": "7.4" }),
    })
}

#[derive(Clone, Debug)]
struct ImportedModel {
    numeric_id: i64,
    name: String,
    potter_id: Option<String>,
    transform: Transform,
    visible: bool,
    armature: Option<ArmatureData>,
}

#[derive(Clone, Debug)]
struct ImportedGeometry {
    positions: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    material_indices: Vec<u32>,
    deformation: Option<Value>,
}

fn decode_deformation_field<T: DeserializeOwned>(
    metadata: Option<&Value>,
    key: &str,
) -> Result<Option<T>> {
    metadata
        .and_then(|value| value.get(key))
        .cloned()
        .map(|value| {
            serde_json::from_value(value).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "FBX rig and morph metadata has an invalid field",
                    json!({"field":key,"reason":error.to_string()}),
                )
            })
        })
        .transpose()
}

#[derive(Clone, Debug)]
struct ImportedMaterial {
    numeric_id: i64,
    name: String,
    potter_id: Option<String>,
    base_color: [f64; 4],
    metallic: f64,
    roughness: f64,
    emission_color: [f64; 3],
    emission_strength: f64,
    transmission: f64,
    ior: f64,
}

#[derive(Clone, Debug)]
struct ImportedTexture {
    numeric_id: i64,
    file_name: String,
    uv_map: Option<String>,
    property: Option<String>,
}
#[derive(Clone, Debug)]
struct Connection {
    kind: String,
    child: i64,
    parent: i64,
    property: Option<String>,
}

fn parse_model(node: &FbxNode, numeric_id: i64) -> Result<ImportedModel> {
    let object_name = node
        .args
        .get(1)
        .ok_or_else(|| import_error("FBX Model name is missing"))?;
    let name = strip_object_prefix(object_name, "Model").to_owned();
    let properties = node.child("Properties70");
    let translation = read_vec3_property(properties, "Lcl Translation", [0.0; 3])?;
    let rotation_degrees = read_vec3_property(properties, "Lcl Rotation", [0.0; 3])?;
    let scale = read_vec3_property(properties, "Lcl Scaling", [1.0; 3])?;
    let rotation_order = read_scalar_property(properties, "RotationOrder", 0.0)?;
    if !rotation_order.is_finite()
        || rotation_order.fract() != 0.0
        || !(0.0..=5.0).contains(&rotation_order)
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "FBX Euler rotation order is unsupported",
            json!({ "feature_id": "format.fbx.rotation_order", "rotation_order": rotation_order }),
        ));
    }
    let rotation_order = match rotation_order as u32 {
        0 => EulerRot::XYZ,
        1 => EulerRot::XZY,
        2 => EulerRot::YZX,
        3 => EulerRot::YXZ,
        4 => EulerRot::ZXY,
        5 => EulerRot::ZYX,
        _ => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "FBX Euler rotation order is unsupported",
                json!({ "feature_id": "format.fbx.rotation_order", "rotation_order": rotation_order }),
            ));
        }
    };
    for property in [
        "PreRotation",
        "PostRotation",
        "RotationOffset",
        "RotationPivot",
        "ScalingOffset",
        "ScalingPivot",
        "GeometricTranslation",
        "GeometricRotation",
        "GeometricScaling",
    ] {
        let expected = if property == "GeometricScaling" {
            [1.0; 3]
        } else {
            [0.0; 3]
        };
        if !crate::float::equal_f64_array(
            &read_vec3_property(properties, property, expected)?,
            &expected,
        ) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "FBX pivot or geometric transform cannot be represented without changing mesh coordinates",
                json!({ "feature_id": "format.fbx.pivot_transform", "property": property }),
            ));
        }
    }
    let rotation = DQuat::from_euler(
        rotation_order,
        rotation_degrees[0].to_radians(),
        rotation_degrees[1].to_radians(),
        rotation_degrees[2].to_radians(),
    )
    .normalize();
    let visible = read_scalar_property(properties, "Visibility", 1.0)? != 0.0;
    let potter_id = read_string_property(properties, "potter:id");
    let armature = read_string_property(properties, "potter:armature")
        .filter(|encoded| !encoded.is_empty())
        .map(|encoded| {
            serde_json::from_str(&encoded).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "FBX armature metadata is invalid",
                    json!({"reason":error.to_string()}),
                )
            })
        })
        .transpose()?;
    Ok(ImportedModel {
        numeric_id,
        name,
        potter_id,
        transform: Transform {
            translation,
            rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
            scale,
            rotation_mode: "quaternion".to_owned(),
        },
        visible,
        armature,
    })
}

fn parse_geometry(node: &FbxNode) -> Result<ImportedGeometry> {
    let vertex_values = numeric_array(
        node.child("Vertices")
            .ok_or_else(|| import_error("FBX geometry has no Vertices array"))?,
    )?;
    if vertex_values.len() % 3 != 0 || vertex_values.iter().any(|value| !value.is_finite()) {
        return Err(import_error(
            "FBX Vertices array must contain finite xyz triples",
        ));
    }
    let positions = vertex_values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|point| [point[0], point[1], point[2]])
        .collect::<Vec<_>>();
    let encoded_indices = integer_array(
        node.child("PolygonVertexIndex")
            .ok_or_else(|| import_error("FBX geometry has no PolygonVertexIndex array"))?,
    )?;
    let mut faces = Vec::<Vec<usize>>::new();
    let mut face = Vec::<usize>::new();
    for encoded in encoded_indices {
        let last = encoded < 0;
        let index_i64 = if last {
            encoded
                .checked_neg()
                .and_then(|index| index.checked_sub(1))
                .ok_or_else(|| import_error("FBX polygon index overflows"))?
        } else {
            encoded
        };
        let index = usize::try_from(index_i64)
            .map_err(|_| import_error("FBX polygon index is negative or too large"))?;
        if index >= positions.len() {
            return Err(import_error(
                "FBX polygon index is outside the vertex array",
            ));
        }
        face.push(index);
        if last {
            if face.len() < 3 {
                return Err(import_error(
                    "FBX polygon contains fewer than three vertices",
                ));
            }
            faces.push(std::mem::take(&mut face));
        }
    }
    if !face.is_empty() {
        return Err(import_error(
            "FBX polygon index array ends before a polygon terminator",
        ));
    }
    let mut material_indices = Vec::new();
    if let Some(layer) = node.child("LayerElementMaterial") {
        let mapping = layer
            .child("MappingInformationType")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        if mapping != "ByPolygon" {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "FBX material mapping is unsupported",
                json!({"feature_id":"format.fbx.material_mapping","mapping":mapping}),
            ));
        }
        if let Some(values) = layer.child("Materials") {
            material_indices = integer_array(values)?
                .into_iter()
                .map(|value| {
                    u32::try_from(value)
                        .map_err(|_| import_error("FBX material index is negative or too large"))
                })
                .collect::<Result<Vec<_>>>()?;
        }
        if !material_indices.is_empty() && material_indices.len() != faces.len() {
            return Err(import_error(
                "FBX material index count does not match polygon count",
            ));
        }
    }
    let deformation = read_string_property(node.child("Properties70"), "potter:deformation")
        .map(|encoded| {
            serde_json::from_str(&encoded).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "FBX rig and morph metadata is invalid",
                    json!({"reason":error.to_string()}),
                )
            })
        })
        .transpose()?;
    Ok(ImportedGeometry {
        positions,
        faces,
        material_indices,
        deformation,
    })
}

fn parse_material(node: &FbxNode, numeric_id: i64) -> Result<ImportedMaterial> {
    let object_name = node
        .args
        .get(1)
        .ok_or_else(|| import_error("FBX Material name is missing"))?;
    let name = strip_object_prefix(object_name, "Material").to_owned();
    let properties = node.child("Properties70");
    let diffuse = read_vec3_property(properties, "DiffuseColor", [0.6; 3])?;
    let diffuse_factor = read_scalar_property(properties, "DiffuseFactor", 1.0)?;
    let opacity = read_scalar_property(properties, "Opacity", 1.0)?;
    Ok(ImportedMaterial {
        numeric_id,
        name,
        potter_id: read_string_property(properties, "potter:id"),
        base_color: [
            diffuse[0] * diffuse_factor,
            diffuse[1] * diffuse_factor,
            diffuse[2] * diffuse_factor,
            opacity.clamp(0.0, 1.0),
        ],
        metallic: read_scalar_property(properties, "potter:metallic", 0.0)?.clamp(0.0, 1.0),
        roughness: read_scalar_property(properties, "potter:roughness", 0.8)?.clamp(0.0, 1.0),
        emission_color: read_vec3_property(properties, "potter:emission_color", [0.0; 3])?,
        emission_strength: read_scalar_property(properties, "potter:emission_strength", 0.0)?,
        transmission: read_scalar_property(properties, "potter:transmission", 0.0)?.clamp(0.0, 1.0),
        ior: read_scalar_property(properties, "potter:ior", 1.45)?,
    })
}

fn parse_texture(node: &FbxNode, numeric_id: i64) -> Result<ImportedTexture> {
    let file_name = node
        .child("FileName")
        .and_then(FbxNode::first_arg)
        .or_else(|| node.child("RelativeFilename").and_then(FbxNode::first_arg))
        .ok_or_else(|| import_error("FBX texture has no file path"))?
        .to_owned();
    let properties = node.child("Properties70");
    let uv_map = read_string_property(properties, "UVSet").filter(|value| !value.is_empty());
    Ok(ImportedTexture {
        numeric_id,
        file_name,
        uv_map,
        property: read_string_property(properties, "potter:property"),
    })
}

fn parse_connections(node: Option<&FbxNode>) -> Result<Vec<Connection>> {
    let Some(node) = node else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for connection in node.children.iter().filter(|child| child.name == "C") {
        if connection.args.len() < 3 {
            return Err(import_error(
                "FBX connection record has fewer than three values",
            ));
        }
        result.push(Connection {
            kind: connection.args[0].clone(),
            child: parse_i64(&connection.args[1], "connection child ID")?,
            parent: parse_i64(&connection.args[2], "connection parent ID")?,
            property: connection.args.get(3).cloned(),
        });
    }
    Ok(result)
}

fn import_animations(
    root: &[FbxNode],
    connections: &[Connection],
    models: &BTreeMap<i64, ImportedModel>,
    model_ids: &BTreeMap<i64, Id>,
    fps: f64,
    doc: &mut SceneDoc,
) -> Result<()> {
    let Some(objects) = root.iter().find(|node| node.name == "Objects") else {
        return Ok(());
    };
    let mut stacks = BTreeMap::<i64, (String, Option<String>)>::new();
    let mut layers = BTreeMap::<i64, i64>::new();
    let mut curve_nodes = BTreeMap::<i64, String>::new();
    let mut curves = BTreeMap::<i64, ImportedCurve>::new();
    for object in &objects.children {
        match object.name.as_str() {
            "AnimationStack" => {
                let id = object_id(object)?;
                let name = object.args.get(1).map_or_else(
                    || "Action".to_owned(),
                    |value| strip_object_prefix(value, "AnimationStack").to_owned(),
                );
                stacks.insert(
                    id,
                    (
                        name,
                        read_string_property(object.child("Properties70"), "potter:id"),
                    ),
                );
            }
            "AnimationLayer" => {
                layers.insert(object_id(object)?, 0);
            }
            "AnimationCurveNode" => {
                let id = object_id(object)?;
                curve_nodes.insert(
                    id,
                    object
                        .args
                        .get(1)
                        .map(|name| strip_object_prefix(name, "AnimCurveNode").to_owned())
                        .unwrap_or_default(),
                );
            }
            "AnimationCurve" => {
                let id = object_id(object)?;
                curves.insert(id, parse_animation_curve(object, id)?);
            }
            _ => {}
        }
    }
    if stacks.is_empty() || curves.is_empty() {
        return Ok(());
    }
    let mut layer_to_stack = BTreeMap::<i64, i64>::new();
    let mut curve_node_to_layer = BTreeMap::<i64, i64>::new();
    let mut curve_node_to_model = BTreeMap::<i64, (i64, String)>::new();
    let mut curve_to_curve_node = BTreeMap::<i64, (i64, String)>::new();
    for connection in connections {
        if connection.kind == "OO" {
            if layers.contains_key(&connection.child) && stacks.contains_key(&connection.parent) {
                layer_to_stack.insert(connection.child, connection.parent);
            } else if curve_nodes.contains_key(&connection.child)
                && layers.contains_key(&connection.parent)
            {
                curve_node_to_layer.insert(connection.child, connection.parent);
            }
        } else if connection.kind == "OP" {
            if curve_nodes.contains_key(&connection.child)
                && models.contains_key(&connection.parent)
            {
                curve_node_to_model.insert(
                    connection.child,
                    (
                        connection.parent,
                        connection.property.clone().unwrap_or_default(),
                    ),
                );
            } else if curves.contains_key(&connection.child)
                && curve_nodes.contains_key(&connection.parent)
            {
                curve_to_curve_node.insert(
                    connection.child,
                    (
                        connection.parent,
                        connection.property.clone().unwrap_or_default(),
                    ),
                );
            }
        }
    }
    let mut actions = BTreeMap::<i64, Action>::new();
    let mut stack_models = BTreeMap::<i64, BTreeSet<i64>>::new();
    for (curve_id, (curve_node_id, component_name)) in curve_to_curve_node {
        let Some((model_id, property)) = curve_node_to_model.get(&curve_node_id) else {
            continue;
        };
        let Some(layer_id) = curve_node_to_layer.get(&curve_node_id) else {
            continue;
        };
        let Some(stack_id) = layer_to_stack.get(layer_id) else {
            continue;
        };
        let Some(curve) = curves.get(&curve_id) else {
            continue;
        };
        let component = match component_name.as_str() {
            "d|X" => 0,
            "d|Y" => 1,
            "d|Z" => 2,
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX animation component is unsupported",
                    json!({"feature_id":"format.fbx.animation_component","component":component_name}),
                ));
            }
        };
        let (path, rotation) = match property.as_str() {
            "Lcl Translation" => ("transform.translation", false),
            "Lcl Scaling" => ("transform.scale", false),
            "Lcl Rotation" => ("transform.rotation_euler", true),
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX animation property is not a transform channel",
                    json!({"feature_id":"format.fbx.animation_property","property":property}),
                ));
            }
        };
        let keyframes = curve
            .times
            .iter()
            .zip(&curve.values)
            .zip(&curve.interpolations)
            .map(|((time, value), interpolation)| Keyframe {
                frame: *time / FBX_TICKS_PER_SECOND * fps,
                value: if rotation { value.to_radians() } else { *value },
                interpolation: *interpolation,
                ..Keyframe::default()
            })
            .collect::<Vec<_>>();
        let action = actions.entry(*stack_id).or_insert_with(|| Action {
            name: stacks
                .get(stack_id)
                .map_or_else(|| "Action".to_owned(), |entry| entry.0.clone()),
            fcurves: Vec::new(),
            ..Action::default()
        });
        action.fcurves.push(FCurve {
            path: path.to_owned(),
            index: component,
            keyframes,
            extrapolation: Extrapolation::Constant,
        });
        stack_models.entry(*stack_id).or_default().insert(*model_id);
    }
    let mut used = doc.actions.keys().cloned().collect::<BTreeSet<_>>();
    for (stack_id, action) in actions {
        let (name, custom_id) = stacks
            .get(&stack_id)
            .cloned()
            .unwrap_or_else(|| ("Action".to_owned(), None));
        let action_id = unique_import_id(
            custom_id.as_deref().unwrap_or(&name),
            "fbx_action",
            stack_id,
            &mut used,
        )?;
        doc.actions.insert(action_id.clone(), action);
        if let Some(model_ids_for_action) = stack_models.get(&stack_id) {
            for model_number in model_ids_for_action {
                if let Some(node_id) = model_ids.get(model_number)
                    && let Some(node) = doc.nodes.get_mut(node_id)
                {
                    if node.action.is_some() {
                        return Err(PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            "a model connected to multiple FBX animation stacks cannot map to one Potter Action",
                            json!({"feature_id":"format.fbx.multiple_actions","node_id":node_id}),
                        ));
                    }
                    node.action = Some(action_id.clone());
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ImportedCurve {
    times: Vec<f64>,
    values: Vec<f64>,
    interpolations: Vec<Interpolation>,
}

fn parse_animation_curve(node: &FbxNode, _numeric_id: i64) -> Result<ImportedCurve> {
    let time_values = integer_array(
        node.child("KeyTime")
            .ok_or_else(|| import_error("FBX animation curve has no KeyTime array"))?,
    )?;
    let values = numeric_array(
        node.child("KeyValueFloat")
            .ok_or_else(|| import_error("FBX animation curve has no KeyValueFloat array"))?,
    )?;
    if time_values.len() != values.len() || values.iter().any(|value| !value.is_finite()) {
        return Err(import_error(
            "FBX animation key-time and value arrays are inconsistent",
        ));
    }
    if time_values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(import_error(
            "FBX animation key times must be strictly increasing",
        ));
    }
    let flags = if let Some(array) = node.child("KeyAttrFlags") {
        let flags = integer_array(array)?;
        if flags.len() != time_values.len() {
            return Err(import_error(
                "FBX animation interpolation flag count does not match key count",
            ));
        }
        flags
    } else {
        vec![4; time_values.len()]
    };
    let interpolations = flags
        .into_iter()
        .map(|flag| {
            if flag & 2 != 0 {
                Interpolation::Constant
            } else if flag & 8 != 0 {
                Interpolation::Bezier
            } else {
                Interpolation::Linear
            }
        })
        .collect();
    Ok(ImportedCurve {
        times: time_values.into_iter().map(|time| time as f64).collect(),
        values,
        interpolations,
    })
}

#[derive(Clone, Debug)]
struct FbxNode {
    name: String,
    args: Vec<String>,
    children: Vec<FbxNode>,
}

impl FbxNode {
    fn child(&self, name: &str) -> Option<&FbxNode> {
        self.children.iter().find(|child| child.name == name)
    }
    fn first_arg(&self) -> Option<&str> {
        self.args.first().map(String::as_str)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Atom(String),
    String(String),
    Colon,
    Comma,
    Open,
    Close,
    Star,
    Newline,
}

fn lex(text: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\n' | '\r' => {
                if character == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                tokens.push(Token::Newline);
            }
            ' ' | '\t' => {}
            ';' => {
                for next in chars.by_ref() {
                    if next == '\n' {
                        tokens.push(Token::Newline);
                        break;
                    }
                }
            }
            ':' => tokens.push(Token::Colon),
            ',' => tokens.push(Token::Comma),
            '{' => tokens.push(Token::Open),
            '}' => tokens.push(Token::Close),
            '*' => tokens.push(Token::Star),
            '"' => {
                let mut string = String::new();
                let mut terminated = false;
                while let Some(next) = chars.next() {
                    match next {
                        '"' => {
                            terminated = true;
                            break;
                        }
                        '\\' => match chars.next() {
                            Some('"') => string.push('"'),
                            Some('\\') => string.push('\\'),
                            Some(other) => {
                                string.push('\\');
                                string.push(other);
                            }
                            None => return Err(import_error("FBX string ends with an escape")),
                        },
                        other => string.push(other),
                    }
                }
                if !terminated {
                    return Err(import_error("FBX string is not terminated"));
                }
                tokens.push(Token::String(string));
            }
            other => {
                let mut atom = String::new();
                atom.push(other);
                while let Some(next) = chars.peek().copied() {
                    if next.is_whitespace()
                        || matches!(next, ':' | ',' | '{' | '}' | '*' | ';' | '"')
                    {
                        break;
                    }
                    atom.push(next);
                    chars.next();
                }
                tokens.push(Token::Atom(atom));
            }
        }
    }
    Ok(tokens)
}

fn parse_nodes(tokens: &[Token]) -> Result<Vec<FbxNode>> {
    let mut index = 0;
    let nodes = parse_node_list(tokens, &mut index, false)?;
    if index != tokens.len() {
        return Err(import_error(
            "FBX input contains trailing unmatched block delimiters",
        ));
    }
    Ok(nodes)
}

fn parse_node_list(tokens: &[Token], index: &mut usize, in_block: bool) -> Result<Vec<FbxNode>> {
    let mut nodes = Vec::new();
    loop {
        while matches!(tokens.get(*index), Some(Token::Newline)) {
            *index += 1;
        }
        match tokens.get(*index) {
            Some(Token::Close) if in_block => {
                *index += 1;
                return Ok(nodes);
            }
            None if !in_block => return Ok(nodes),
            None => return Err(import_error("FBX block is not terminated")),
            Some(Token::Close) => {
                return Err(import_error(
                    "FBX input contains an unmatched closing brace",
                ));
            }
            _ => {}
        }
        let name = match tokens.get(*index) {
            Some(Token::Atom(name) | Token::String(name)) => name.clone(),
            _ => return Err(import_error("FBX node name is invalid")),
        };
        *index += 1;
        if !matches!(tokens.get(*index), Some(Token::Colon)) {
            return Err(import_error("FBX node has no colon separator"));
        }
        *index += 1;
        let mut args = Vec::new();
        let mut opens_block = false;
        loop {
            match tokens.get(*index) {
                Some(Token::Newline) => {
                    *index += 1;
                    break;
                }
                Some(Token::Open) => {
                    *index += 1;
                    opens_block = true;
                    break;
                }
                Some(Token::Close) | None => break,
                Some(Token::Comma | Token::Star) => {
                    *index += 1;
                }
                Some(Token::Atom(value) | Token::String(value)) => {
                    args.push(value.clone());
                    *index += 1;
                }
                Some(Token::Colon) => {
                    return Err(import_error("FBX node contains an unexpected colon"));
                }
            }
        }
        let children = if opens_block {
            parse_node_list(tokens, index, true)?
        } else {
            Vec::new()
        };
        nodes.push(FbxNode {
            name,
            args,
            children,
        });
    }
}

fn numeric_array(node: &FbxNode) -> Result<Vec<f64>> {
    let values = node.child("a").unwrap_or(node);
    values
        .args
        .iter()
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|_| import_error("FBX numeric array contains an invalid number"))
        })
        .collect()
}

fn integer_array(node: &FbxNode) -> Result<Vec<i64>> {
    let values = node.child("a").unwrap_or(node);
    values
        .args
        .iter()
        .map(|value| {
            value
                .parse::<i64>()
                .map_err(|_| import_error("FBX integer array contains an invalid integer"))
        })
        .collect()
}

fn read_vec3_property(
    properties: Option<&FbxNode>,
    name: &str,
    default: [f64; 3],
) -> Result<[f64; 3]> {
    let Some(property) = property_node(properties, name) else {
        return Ok(default);
    };
    let values = property.args.get(4..).unwrap_or(&[]);
    if values.len() < 3 {
        return Err(import_error(
            "FBX vector property has fewer than three components",
        ));
    }
    let mut result = [0.0; 3];
    for index in 0..3 {
        result[index] = values[index]
            .parse::<f64>()
            .map_err(|_| import_error("FBX vector property contains an invalid number"))?;
        if !result[index].is_finite() {
            return Err(import_error("FBX vector property is not finite"));
        }
    }
    Ok(result)
}

fn read_scalar_property(properties: Option<&FbxNode>, name: &str, default: f64) -> Result<f64> {
    let Some(property) = property_node(properties, name) else {
        return Ok(default);
    };
    let value = property
        .args
        .get(4)
        .ok_or_else(|| import_error("FBX scalar property has no value"))?
        .parse::<f64>()
        .map_err(|_| import_error("FBX scalar property contains an invalid number"))?;
    if !value.is_finite() {
        return Err(import_error("FBX scalar property is not finite"));
    }
    Ok(value)
}

fn read_string_property(properties: Option<&FbxNode>, name: &str) -> Option<String> {
    property_node(properties, name).and_then(|property| property.args.get(4).cloned())
}

fn property_node<'a>(properties: Option<&'a FbxNode>, name: &str) -> Option<&'a FbxNode> {
    properties?.children.iter().find(|property| {
        property.name == "P" && property.args.first().is_some_and(|value| value == name)
    })
}

fn find_custom_frame_rate(root: &[FbxNode]) -> Result<Option<f64>> {
    let Some(settings) = root.iter().find(|node| node.name == "GlobalSettings") else {
        return Ok(None);
    };
    let Some(properties) = settings.child("Properties70") else {
        return Ok(None);
    };
    if property_node(Some(properties), "CustomFrameRate").is_none() {
        return Ok(None);
    }
    read_scalar_property(Some(properties), "CustomFrameRate", DEFAULT_FPS).map(Some)
}

fn object_id(node: &FbxNode) -> Result<i64> {
    parse_i64(
        node.args
            .first()
            .ok_or_else(|| import_error("FBX object ID is missing"))?,
        "object ID",
    )
}

fn parse_i64(value: &str, label: &str) -> Result<i64> {
    value.parse::<i64>().map_err(|_| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            format!("FBX {label} is invalid"),
            json!({"value":value}),
        )
    })
}

fn strip_object_prefix<'a>(value: &'a str, prefix: &str) -> &'a str {
    value.strip_prefix(&format!("{prefix}::")).unwrap_or(value)
}

fn unique_import_id(raw: &str, prefix: &str, index: i64, used: &mut BTreeSet<Id>) -> Result<Id> {
    let candidate = if Id::new(raw.to_owned()).is_ok() {
        raw.to_owned()
    } else {
        sanitize_id(raw, prefix, index)
    };
    let id = Id::new(candidate.clone())?;
    if let Some(id) = super::claim_import_id(id, used) {
        return Ok(id);
    }
    let stem = candidate.chars().take(52).collect::<String>();
    let mut suffix = 2_u32;
    loop {
        let id = Id::new(format!("{stem}_{suffix}"))?;
        if let Some(id) = super::claim_import_id(id, used) {
            return Ok(id);
        }
        suffix = suffix.checked_add(1).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "FBX ID uniqueness counter overflow",
            )
        })?;
    }
}

fn sanitize_id(raw: &str, prefix: &str, index: i64) -> String {
    let mut value = String::with_capacity(64);
    let prefix = prefix
        .chars()
        .filter(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || *character == '_'
        })
        .collect::<String>();
    value.push_str(if prefix.is_empty() { "fbx" } else { &prefix });
    value.push('_');
    for character in raw.chars() {
        if value.len() >= 64 {
            break;
        }
        let character = character.to_ascii_lowercase();
        value.push(
            if character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '_'
                || character == '-'
            {
                character
            } else {
                '_'
            },
        );
    }
    if value.len() < 64 {
        value.push('_');
        value.push_str(&index.unsigned_abs().to_string());
    }
    value.truncate(64);
    value
}

fn unique_data_id(doc: &SceneDoc, node_id: &Id) -> Result<Id> {
    let mut base = format!("{}_mesh", node_id.as_str());
    base.truncate(64);
    let first = Id::new(base.clone())?;
    super::unique_data_id(
        first,
        |id| doc.data_blocks.contains_key(id),
        |suffix| {
            let suffix = format!("_{suffix}");
            let prefix = &base[..base.len().min(64 - suffix.len())];
            Id::new(format!("{prefix}{suffix}"))
        },
        3,
        "FBX data ID uniqueness counter overflow",
    )
}

fn component_name(index: u32) -> Result<&'static str> {
    match index {
        0 => Ok("X"),
        1 => Ok("Y"),
        2 => Ok("Z"),
        _ => Err(PotError::with_details(
            ErrorCode::UnrepresentableFeature,
            "FBX vector animation component must be X, Y, or Z",
            json!({"feature_id":"format.fbx.animation_component","index":index}),
        )),
    }
}

fn frame_to_ticks(frame: f64, fps: f64) -> Result<i64> {
    let time = frame / fps * FBX_TICKS_PER_SECOND;
    if !time.is_finite() || time < i64::MIN as f64 || time > i64::MAX as f64 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "FBX animation time exceeds representable range",
        ));
    }
    Ok(time.round() as i64)
}

fn take_object_id(next: &mut i64) -> Result<i64> {
    let id = *next;
    *next = next
        .checked_add(1)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "FBX object ID range exhausted"))?;
    Ok(id)
}

fn fmt_num(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.17}")
    }
}

fn quote(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn write_error(_: std::fmt::Error) -> PotError {
    PotError::new(ErrorCode::InternalError, "writing FBX text failed")
}
