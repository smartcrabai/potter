use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Value, json};

use serde::de::DeserializeOwned;

use super::{
    ExchangeUvLayer, ImportedGraph, Loss, decode_byte_color, encode_byte_color, import_error,
    import_uv_layers, shading, unique_uv_name, uv_import_loss,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationPurpose, SlotBinding, Snapshot},
    geom::{
        Mesh,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        normals::corner_normals,
        uv_layers,
    },
    model::{
        Action, ArmatureData, Bone, DataBlock, Extrapolation, FCurve, Id, Image, ImageColorspace,
        ImageSource, Interpolation, Keyframe, Material, Node, SceneDoc, TextureRef, Transform,
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
struct MaterialExport {
    id: i64,
    material_id: Id,
    uv_layers: [Option<String>; 4],
}

type MaterialBindings = BTreeMap<(Id, usize), i64>;
type MaterialSlotIndices = BTreeMap<Id, Vec<Option<u32>>>;

#[derive(Clone, Debug)]
struct TextureExport {
    id: i64,
    material_id: Id,
    material_object_id: i64,
    property: &'static str,
    path: String,
    uv_layer: Option<String>,
    /// Blender's exporter writes `WrapModeU` and `WrapModeV` as clamp (1) for any extension but REPEAT (0).
    clamp: bool,
}

#[derive(Clone, Debug)]
struct BoneExport {
    model_id: i64,
    parent_model: i64,
    bone_id: Id,
    bone: Bone,
    local: DMat4,
}

/// Export ASCII FBX 7.4 with texture assets resolved under `project_root`, optionally dropping
/// attributes reported by the lossy-export gate. Color layers are written as `LayerElementColor`;
/// generic attributes are not written, and `exchange::attribute_losses` reports them.
pub(crate) fn export_with_root_allow_lossy(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    project_root: &Path,
    allow_lossy: bool,
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

    let (bone_models, bone_model_ids) = build_bone_models(doc, snapshot, &ids, &mut next_id)?;
    let (skin_exports, morph_exports) =
        build_deformers(doc, snapshot, &ids, &bone_model_ids, &mut next_id)?;
    let (material_exports, material_bindings, material_slot_indices) =
        build_material_exports(doc, snapshot, &mut next_id)?;
    let texture_exports =
        build_texture_exports(doc, project_root, &material_exports, &mut next_id)?;

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
        + material_exports.len()
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
        material_exports.len(),
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
            let mesh = if writes_source_mesh(data_block) {
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
            let slot_indices = material_slot_indices.get(node_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "FBX material slot indices are missing",
                )
            })?;
            write_geometry(
                &mut output,
                geometry_id,
                node_id,
                node,
                mesh,
                data_block,
                fbx_material_slots(doc, snapshot, node_id, node).len(),
                slot_indices,
                allow_lossy,
                snapshot.purpose,
            )?;
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

    for material_export in &material_exports {
        let material = doc
            .materials
            .get(&material_export.material_id)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "FBX material is missing"))?;
        writeln!(
            output,
            "\tMaterial: {}, \"{}\", \"\" {{\n\t\tVersion: 102\n\t\tShadingModel: \"phong\"\n\t\tMultiLayer: 0\n\t\tProperties70:  {{\n\t\t\tP: \"DiffuseColor\", \"Color\",\"\",\"A\",{},{},{}\n\t\t\tP: \"DiffuseFactor\", \"Number\",\"\",\"A\",1\n\t\t\tP: \"Opacity\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:id\", \"KString\", \"\", \"\", \"{}\"\n\t\t\tP: \"potter:metallic\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:roughness\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:emission_color\", \"Color\",\"\",\"A\",{},{},{}\n\t\t\tP: \"potter:emission_strength\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:transmission\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:ior\", \"Number\",\"\",\"A\",{}\n\t\t\tP: \"potter:double_sided\", \"bool\", \"Boolean\", \"\",{}\n\t\t}}\n\t}}",
            material_export.id,
            quote(&format!("Material::{}", material.name)),
            fmt_num(material.base_color[0]),
            fmt_num(material.base_color[1]),
            fmt_num(material.base_color[2]),
            fmt_num(material.base_color[3]),
            quote(material_export.material_id.as_str()),
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
        let uv_set_property = texture
            .uv_layer
            .as_deref()
            .map_or_else(String::new, |layer| {
                format!(
                    "\t\t\tP: \"UVSet\", \"KString\", \"\", \"\", \"{}\"\n",
                    quote(layer)
                )
            });
        let default_name_property = texture
            .uv_layer
            .as_deref()
            .filter(|layer| *layer == "default")
            .map_or_else(String::new, |layer| {
                format!(
                    "\t\t\tP: \"potter:uv_layer_name\", \"KString\", \"\", \"\", \"{}\"\n",
                    quote(layer)
                )
            });
        let wrap_mode = i32::from(texture.clamp);
        writeln!(
            output,
            "\tTexture: {}, \"Texture::{}\", \"TextureVideoClip\" {{\n\t\tType: \"TextureVideoClip\"\n\t\tVersion: 202\n\t\tTextureName: \"Texture::{}\"\n\t\tProperties70:  {{\n{}{default_name_property}\t\t\tP: \"WrapModeU\", \"enum\", \"\", \"\", {wrap_mode}\n\t\t\tP: \"WrapModeV\", \"enum\", \"\", \"\", {wrap_mode}\n\t\t\tP: \"potter:property\", \"KString\", \"\", \"\", \"{}\"\n\t\t}}\n\t\tFileName: \"{}\"\n\t\tRelativeFilename: \"{}\"\n\t\tModelUVTranslation: 0,0\n\t\tModelUVScaling: 1,1\n\t\tTexture_Alpha_Source: \"None\"\n\t\tCropping: 0,0,0,0\n\t}}",
            texture.id,
            quote(&material.name),
            quote(&material.name),
            uv_set_property,
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
        for (slot, material_id) in fbx_material_slots(doc, snapshot, node_id, node)
            .iter()
            .enumerate()
        {
            let Some(material_id) = material_id else {
                continue;
            };
            let material_object_id =
                material_bindings
                    .get(&(node_id.clone(), slot))
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::SceneInvalid,
                            "node material UV variant is missing",
                            json!({ "node_id": node_id, "slot": slot, "material_id": material_id }),
                        )
                    })?;
            writeln!(
                output,
                "\tC: \"OO\",{material_object_id},{}",
                object_ids.model
            )
            .map_err(write_error)?;
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
            writeln!(output, "\tC: \"OO\",{},{}", cluster.bone_model, cluster.id)
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
        writeln!(
            output,
            "\tC: \"OP\",{}, {},\"{}\"",
            texture.id, texture.material_object_id, texture.property
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

fn material_texture_refs(material: &Material) -> [Option<&TextureRef>; 4] {
    [
        material.base_color_texture.as_ref(),
        material.roughness_texture.as_ref(),
        material.metallic_texture.as_ref(),
        material.normal_texture.as_ref(),
    ]
}

const TEXTURE_PROPERTIES: [&str; 4] = ["DiffuseColor", "Shininess", "ReflectionColor", "NormalMap"];

#[derive(Clone, Debug)]
struct TextureUvSelection {
    uv_layer: Option<String>,
    missing: Option<String>,
}

fn resolve_texture_uv_selection(
    store: Option<&uv_layers::UvStore>,
    binding: Option<&SlotBinding>,
    reference: Option<&str>,
) -> TextureUvSelection {
    let reference = reference.filter(|name| !name.is_empty());
    let Some(store) = store else {
        return TextureUvSelection {
            uv_layer: reference.map(str::to_owned),
            missing: None,
        };
    };
    let render_index = match binding {
        Some(binding) => binding.uv.render_index.map(usize::from),
        None => store.render_index(),
    }
    .filter(|index| store.layers.get(*index).is_some());
    let (index, missing) = match reference {
        Some(name) => match store.index_of(name) {
            Some(index)
                if binding.is_none_or(|binding| {
                    index < 8 && binding.uv.available_mask & (1_u8 << index) != 0
                }) =>
            {
                (Some(index), None)
            }
            _ => (render_index, Some(name.to_owned())),
        },
        None => (render_index, None),
    };
    TextureUvSelection {
        uv_layer: index
            .and_then(|index| store.layers.get(index))
            .map(|layer| layer.name.clone()),
        missing,
    }
}

fn material_uv_layers(
    material: &Material,
    store: Option<&uv_layers::UvStore>,
    binding: Option<&SlotBinding>,
) -> [Option<String>; 4] {
    let textures = material_texture_refs(material);
    std::array::from_fn(|index| {
        textures[index].and_then(|texture| {
            resolve_texture_uv_selection(store, binding, texture.uv_layer.as_deref()).uv_layer
        })
    })
}

fn export_mesh_for_node<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    node_id: &Id,
    node: &'a Node,
) -> Option<&'a Mesh> {
    let data_block = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    if writes_source_mesh(data_block) {
        data_block.and_then(|data| data.mesh.as_ref())
    } else {
        snapshot.meshes.get(node_id)
    }
}

fn add_material_export(
    material_id: &Id,
    uv_layers: [Option<String>; 4],
    variants: &mut BTreeMap<(Id, [Option<String>; 4]), i64>,
    output: &mut Vec<MaterialExport>,
    next_id: &mut i64,
) -> Result<i64> {
    let key = (material_id.clone(), uv_layers.clone());
    if let Some(id) = variants.get(&key) {
        return Ok(*id);
    }
    let id = take_object_id(next_id)?;
    variants.insert(key, id);
    output.push(MaterialExport {
        id,
        material_id: material_id.clone(),
        uv_layers,
    });
    Ok(id)
}

fn build_material_exports(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    next_id: &mut i64,
) -> Result<(Vec<MaterialExport>, MaterialBindings, MaterialSlotIndices)> {
    let mut output = Vec::new();
    let mut variants = BTreeMap::<(Id, [Option<String>; 4]), i64>::new();
    let mut bindings = BTreeMap::new();
    let mut slot_indices = BTreeMap::new();
    let mut used_materials = BTreeSet::new();
    for (node_id, node) in &doc.nodes {
        let data_block = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        let store = export_mesh_for_node(doc, snapshot, node_id, node).map(uv_layers::read);
        let slots = fbx_material_slots(doc, snapshot, node_id, node);
        let slot_count = slots.len();
        let mut node_slot_indices = data_block
            .is_some_and(|data| data.data_type == "mesh")
            .then(|| Vec::with_capacity(slot_count));
        let mut next_connection_index = 0_usize;
        for (slot, material_id) in slots.iter().enumerate() {
            let Some(material_id) = material_id else {
                if let Some(indices) = &mut node_slot_indices {
                    indices.push(None);
                }
                continue;
            };
            let material = doc.materials.get(material_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "node references a material absent from the FBX export",
                    json!({"node_id":node_id,"material_id":material_id}),
                )
            })?;
            let binding = snapshot.slot_binding(node_id, node, slot);
            let uv_layers = material_uv_layers(material, store.as_ref(), binding);
            let object_id =
                add_material_export(material_id, uv_layers, &mut variants, &mut output, next_id)?;
            if let Some(indices) = &mut node_slot_indices {
                let connection_index = u32::try_from(next_connection_index).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "FBX material slot index exceeds u32",
                    )
                })?;
                next_connection_index = next_connection_index.checked_add(1).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "FBX material slot count overflows",
                    )
                })?;
                indices.push(Some(connection_index));
            }
            bindings.insert((node_id.clone(), slot), object_id);
            used_materials.insert(material_id.clone());
        }
        if let Some(node_slot_indices) = node_slot_indices {
            slot_indices.insert(node_id.clone(), node_slot_indices);
        }
    }
    for (material_id, material) in &doc.materials {
        if !used_materials.contains(material_id) {
            let uv_layers = material_texture_refs(material)
                .map(|texture| texture.and_then(|texture| texture.uv_layer.clone()));
            add_material_export(material_id, uv_layers, &mut variants, &mut output, next_id)?;
        }
    }
    Ok((output, bindings, slot_indices))
}
fn build_texture_exports(
    doc: &SceneDoc,
    root: &Path,
    materials: &[MaterialExport],
    next_id: &mut i64,
) -> Result<Vec<TextureExport>> {
    let mut paths = BTreeMap::<(Id, &'static str), String>::new();
    for (material_id, material) in &doc.materials {
        for (texture, property) in material_texture_refs(material)
            .into_iter()
            .zip(TEXTURE_PROPERTIES)
        {
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
            paths.insert(
                (material_id.clone(), property),
                path.to_string_lossy().into_owned(),
            );
        }
    }

    let mut output = Vec::new();
    for material_export in materials {
        let material = doc
            .materials
            .get(&material_export.material_id)
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "FBX material is missing"))?;
        for (index, (texture, property)) in material_texture_refs(material)
            .into_iter()
            .zip(TEXTURE_PROPERTIES)
            .enumerate()
        {
            if texture.is_none() {
                continue;
            }
            let path = paths
                .get(&(material_export.material_id.clone(), property))
                .ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "FBX texture path is missing")
                })?;
            output.push(TextureExport {
                id: take_object_id(next_id)?,
                material_id: material_export.material_id.clone(),
                material_object_id: material_export.id,
                property,
                path: path.clone(),
                uv_layer: material_export.uv_layers[index].clone(),
                clamp: texture.as_ref().is_some_and(|texture| {
                    texture.extension != crate::image::ImageExtension::Repeat
                }),
            });
        }
    }
    Ok(output)
}

fn missing_texture_uv_layers(
    doc: &SceneDoc,
    snapshot: &Snapshot,
) -> BTreeMap<(Id, Option<String>, Id, Option<String>), usize> {
    let mut missing = BTreeMap::new();
    for (node_id, node) in &doc.nodes {
        let Some(mesh) = export_mesh_for_node(doc, snapshot, node_id, node) else {
            continue;
        };
        let store = uv_layers::read(mesh);
        for (slot, material_id) in fbx_material_slots(doc, snapshot, node_id, node)
            .iter()
            .enumerate()
        {
            let Some(material_id) = material_id else {
                continue;
            };
            let Some(material) = doc.materials.get(material_id) else {
                continue;
            };
            let binding = snapshot.slot_binding(node_id, node, slot);
            for texture in material_texture_refs(material).into_iter().flatten() {
                let requested = texture.uv_layer.as_deref().filter(|name| !name.is_empty());
                let selection = resolve_texture_uv_selection(Some(&store), binding, requested);
                if selection.missing.is_some() || selection.uv_layer.is_none() {
                    let missing_layer = selection.missing.or_else(|| requested.map(str::to_owned));
                    missing
                        .entry((
                            material_id.clone(),
                            missing_layer,
                            node_id.clone(),
                            selection.uv_layer,
                        ))
                        .or_insert(slot);
                }
            }
        }
    }
    missing
}

/// FBX losses for texture wrap extensions and texture references to unavailable UV layers.
pub(crate) fn uv_texture_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<Loss> {
    let mut losses: Vec<Loss> = missing_texture_uv_layers(doc, snapshot)
        .into_iter()
        .filter_map(|((material, layer, node, render_layer), slot)| {
            render_layer.is_none().then_some(Loss {
                feature_id: "uv.layer_missing".to_owned(),
                data_id: Some(material.to_string()),
                reason: match layer {
                    Some(layer) => format!(
                        "material `{material}` samples UV layer `{layer}`, which node `{node}` does not have for slot {slot} and has no render layer to fall back to"
                    ),
                    None => format!(
                        "material `{material}` uses the render UV layer, but node `{node}` has no render layer for slot {slot}"
                    ),
                },
                suggestion: Some(
                    "add the requested UV layer or a render UV layer to the mesh".to_owned(),
                ),
            })
        })
        .collect();
    losses.extend(texture_extension_losses(doc, snapshot));
    losses
}

/// FBX has no mirror or clip wrap. Blender's exporter writes both as clamp, so each texture field on an exported
/// material with one of those extensions reports `format.fbx.texture_extension`.
pub(crate) fn texture_extension_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<Loss> {
    let mut losses = Vec::new();
    let mut seen = BTreeSet::<Id>::new();
    for (node_id, node) in &doc.nodes {
        if !snapshot.meshes.contains_key(node_id) {
            continue;
        }
        for material_id in fbx_material_slots(doc, snapshot, node_id, node)
            .into_iter()
            .flatten()
        {
            if !seen.insert(material_id.clone()) {
                continue;
            }
            let Some(material) = doc.materials.get(&material_id) else {
                continue;
            };
            for (texture, property) in material_texture_refs(material)
                .into_iter()
                .zip(TEXTURE_PROPERTIES)
            {
                let Some(texture) = texture else {
                    continue;
                };
                if matches!(
                    texture.extension,
                    crate::image::ImageExtension::Mirror | crate::image::ImageExtension::Clip
                ) {
                    losses.push(Loss {
                        feature_id: "format.fbx.texture_extension".to_owned(),
                        data_id: Some(material_id.to_string()),
                        reason: format!(
                            "material `{material_id}` samples its {property} texture with a mirror or clip extension, which FBX cannot represent; FBX writes it as clamp"
                        ),
                        suggestion: Some(
                            "use the repeat or extend extension for FBX export".to_owned(),
                        ),
                    });
                }
            }
        }
    }
    losses
}

/// Empty material slots cannot be represented by FBX material connections.
pub(crate) fn material_slot_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<Loss> {
    let mut losses = Vec::new();
    for (node_id, node) in &doc.nodes {
        let is_mesh = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .is_some_and(|data| data.data_type == "mesh");
        if !is_mesh {
            continue;
        }
        if !snapshot.meshes.contains_key(node_id) {
            continue;
        }
        let empty_slots = fbx_material_slots(doc, snapshot, node_id, node)
            .iter()
            .enumerate()
            .filter_map(|(slot, material)| material.is_none().then_some(slot))
            .collect::<Vec<_>>();
        if empty_slots.is_empty() {
            continue;
        }
        losses.push(Loss {
            feature_id: "format.fbx.material_slot_empty".to_owned(),
            data_id: node.data.as_ref().map(ToString::to_string),
            reason: format!(
                "node `{node_id}` has empty material slots {empty_slots:?}, which FBX cannot preserve"
            ),
            suggestion: Some("assign materials to empty slots or export as .blend".to_owned()),
        });
    }
    losses
}

/// FBX writes each material's mirror fields instead of its node graph, so every non-default graph is
/// a loss: strict export refuses it, and `--allow-lossy` writes the mirror fields.
pub(crate) fn material_graph_losses(doc: &SceneDoc) -> Vec<Loss> {
    doc.materials
        .iter()
        .filter(|(_, material)| {
            material
                .node_tree
                .as_ref()
                .is_some_and(|graph_id| !super::is_default_material_graph(doc, graph_id))
        })
        .map(|(material_id, _)| Loss {
            feature_id: "format.fbx.material_node_tree".to_owned(),
            data_id: Some(material_id.to_string()),
            reason:
                "FBX cannot preserve arbitrary material node trees; it writes the mirror fields"
                    .to_owned(),
            suggestion: Some("export as blend or glb to keep the node tree".to_owned()),
        })
        .collect()
}

/// Blender's FBX add-on stores vertex groups only as skin clusters bound to deform bones. Groups
/// without a matching bone/positive-weight cluster survive Potter's metadata but are not imported
/// by Blender, so report them for strict export and validation.
pub(crate) fn vertex_group_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<Loss>> {
    let visible_nodes = super::render_visible_nodes(doc, snapshot)?;
    let mut losses = Vec::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible || !node.render_visible || !visible_nodes.contains(node_id) {
            continue;
        }
        let Some(data_id) = node.data.as_ref() else {
            continue;
        };
        let Some(data) = doc
            .data_blocks
            .get(data_id)
            .filter(|data| data.data_type == "mesh")
        else {
            continue;
        };
        if data.vertex_groups.is_empty() || !snapshot.meshes.contains_key(node_id) {
            continue;
        }
        let enabled = node
            .modifiers
            .iter()
            .filter(|modifier| modifier.is_active(snapshot.purpose))
            .collect::<Vec<_>>();
        let skin_modifiers = enabled
            .iter()
            .filter(|modifier| modifier.modifier_type == "armature")
            .copied()
            .collect::<Vec<_>>();
        let mut deform_bones = BTreeSet::<String>::new();
        if skin_modifiers.len() == 1 && enabled.len() == 1 {
            let armature_id = skin_modifiers[0]
                .params
                .get("object")
                .and_then(Value::as_str)
                .and_then(|raw| Id::new(raw).ok());
            let armature = armature_id
                .as_ref()
                .and_then(|id| doc.nodes.get(id))
                .and_then(|node| node.data.as_ref())
                .and_then(|id| doc.data_blocks.get(id))
                .and_then(|data| data.armature.as_ref());
            if let Some(armature) = armature {
                for bone in armature.bones.values() {
                    if bone.deform {
                        deform_bones.insert(bone.name.clone());
                    }
                }
            }
        }
        for group in &data.vertex_groups {
            let has_positive_weight = data.vertex_weights.values().any(|weights| {
                weights
                    .get(&group.id)
                    .is_some_and(|weight| weight.is_finite() && *weight > 0.0)
            });
            if deform_bones.contains(&group.name) && has_positive_weight {
                continue;
            }
            losses.push(Loss {
                feature_id: "mesh.vertex_group".to_owned(),
                data_id: Some(node_id.to_string()),
                reason: format!(
                    "FBX cannot write vertex group `{}` as a Blender skin cluster without a matching deform bone and positive weights",
                    group.name
                ),
                suggestion: Some(
                    "bind the group to a deform bone or export as .blend or usda".to_owned(),
                ),
            });
        }
    }
    Ok(losses)
}

/// Warnings for explicit texture UV names that fall back to an evaluated mesh's render layer.
pub(crate) fn uv_texture_warnings(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<PotError> {
    missing_texture_uv_layers(doc, snapshot)
        .into_iter()
        .filter_map(|((material, layer, node, render_layer), slot)| {
            Some((material, layer?, node, slot, render_layer?))
        })
        .map(|(material, layer, node, slot, render_layer)| {
            PotError::with_details(
                ErrorCode::UnrepresentableFeature,
                format!(
                    "material `{material}` samples UV layer `{layer}`, which node `{node}` does not have for slot {slot}; render layer `{render_layer}` is used instead"
                ),
                json!({
                    "feature_id": "uv.layer_missing",
                    "status": "fallback",
                    "material_id": material,
                    "node_id": node,
                    "slot": slot,
                    "uv_layer": layer,
                    "render_uv_layer": render_layer,
                }),
            )
        })
        .collect()
}
type BoneModelIds = BTreeMap<(Id, Id), i64>;
/// Each bone's posed matrix in armature space. The snapshot's evaluated bone world matrices already
/// include pose drivers, animation, IK and constraints, so the armature world is divided out of
/// them. An armature the snapshot did not evaluate falls back to its forward-kinematics pose;
/// `posed_armature_losses` reports that case when the armature has pose effects.
fn posed_bone_matrices(
    snapshot: &Snapshot,
    node_id: &Id,
    node: &Node,
    armature: &ArmatureData,
) -> Result<BTreeMap<Id, DMat4>> {
    if let (Some(bones), Some(world)) = (
        snapshot.bone_matrices.get(node_id),
        snapshot.transform_world_matrices.get(node_id),
    ) {
        let inverse = world.inverse();
        return Ok(bones
            .iter()
            .map(|(bone_id, matrix)| (bone_id.clone(), inverse * DMat4::from_cols_array(matrix)))
            .collect());
    }
    Ok(
        crate::eval::rig::evaluate_bone_matrices(armature, &node.pose)?
            .into_iter()
            .map(|(bone_id, matrices)| (bone_id, matrices.pose))
            .collect(),
    )
}

/// Armatures with pose-bone constraints or pose drivers that the snapshot did not evaluate. Their
/// FBX bones carry only the forward-kinematics pose, so the omission must be reported.
pub(crate) fn posed_armature_losses(doc: &SceneDoc, snapshot: &Snapshot) -> Vec<Loss> {
    doc.nodes
        .iter()
        .filter(|(node_id, node)| {
            let is_armature = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .is_some_and(|data| data.armature.is_some());
            let pose_effects = node.constraints.iter().any(|constraint| constraint.owner_bone.is_some())
                || node.drivers.iter().any(|driver| driver.path.starts_with("pose."));
            is_armature && pose_effects && !snapshot.bone_matrices.contains_key(*node_id)
        })
        .map(|(node_id, _)| Loss {
            feature_id: "format.fbx.pose_effects".to_owned(),
            data_id: Some(node_id.to_string()),
            reason: "FBX writes the armature's forward-kinematics pose; its pose-bone constraints and drivers were not evaluated for this export".to_owned(),
            suggestion: Some("export as blend or glb to keep the pose effects".to_owned()),
        })
        .collect()
}

fn build_bone_models(
    doc: &SceneDoc,
    snapshot: &Snapshot,
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
        // Blender's LimbNode `Lcl` is the posed `matrix_local` (`export_fbx_bin.py`,
        // `fbx_data_armature_elements`, via `fbx_object_matrix`); only the bind data uses rest.
        let posed_matrices = posed_bone_matrices(snapshot, node_id, node, armature)?;
        for (bone_id, bone) in &armature.bones {
            let posed = posed_matrices.get(bone_id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "FBX posed bone matrix is missing",
                )
            })?;
            let local = if let Some(parent_id) = &bone.parent {
                let parent = posed_matrices.get(parent_id).copied().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "FBX bone parent is missing")
                })?;
                parent.inverse() * posed
            } else {
                posed
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
            .filter(|modifier| modifier.is_active(snapshot.purpose))
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
            // Blender stores `Transform` in bone space: `inv(bone_world) @ mesh_world`
            // (`export_fbx_bin.py`, `fbx_data_armature_elements`).
            (cluster.bone_bind.inverse() * cluster.mesh_bind)
                .to_cols_array()
                .iter()
                .map(|value| fmt_num(*value))
                .collect::<Vec<_>>()
                .join(","),
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

/// Whether FBX writes a node's document source mesh rather than its evaluated mesh. Deformers
/// (shape keys, vertex groups, vertex weights) are written against the source mesh.
fn writes_source_mesh(data_block: Option<&DataBlock>) -> bool {
    data_block.is_some_and(|data| {
        data.shape_keys
            .as_ref()
            .is_some_and(|keys| !keys.keys.is_empty())
            || !data.vertex_groups.is_empty()
            || !data.vertex_weights.is_empty()
    })
}

/// The material table the FBX mesh of `node` addresses, one entry per face slot; `None` is an empty
/// slot. A source mesh written for its deformers (`writes_source_mesh`) addresses its own DATA slots,
/// as its face indices do; any other mesh addresses the evaluated slots (a TRANSFER Boolean may have
/// replaced them with its pool).
fn fbx_material_slots(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node_id: &Id,
    node: &Node,
) -> Vec<Option<Id>> {
    let data_block = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    if writes_source_mesh(data_block) {
        let data_materials = data_block.map_or(&[][..], |data| data.materials.as_slice());
        return crate::eval::effective_material_slots(node, data_materials);
    }
    (0..snapshot.slot_count(node_id, node, data_block))
        .map(|slot| {
            snapshot
                .slot_material(node_id, node, data_block, slot)
                .cloned()
        })
        .collect()
}

fn write_geometry(
    output: &mut String,
    id: i64,
    node_id: &Id,
    node: &Node,
    mesh: &Mesh,
    data_block: Option<&DataBlock>,
    material_slot_count: usize,
    material_slot_indices: &[Option<u32>],
    allow_lossy: bool,
    purpose: EvaluationPurpose,
) -> Result<()> {
    if material_slot_indices.len() != material_slot_count {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "FBX material slot index table has the wrong length",
        ));
    }
    if !allow_lossy && let Some(slot) = material_slot_indices.iter().position(Option::is_none) {
        return Err(PotError::with_details(
            ErrorCode::UnrepresentableFeature,
            "FBX cannot preserve empty material slots",
            json!({
                "feature_id": "format.fbx.material_slot_empty",
                "node_id": node_id,
                "slot": slot,
            }),
        ));
    }
    let fallback_material_index = material_slot_indices.iter().copied().flatten().next();
    let has_material_layer = fallback_material_index.is_some();
    // Shading attributes, UV layers, and color layers have their own FBX layers. Unsupported
    // attributes are skipped only for lossy export and are reported by `commands::export::export_losses`.
    let colors = color_layers::read(mesh);
    if !allow_lossy && !super::mesh_attribute_losses("fbx", &node.name, mesh).is_empty() {
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
    // FBX has no active or render flag on its UV layers, so Potter records both by layer name.
    let uv = uv_layers::read(mesh);
    let mut properties = Vec::new();
    for (property, name) in [
        ("potter:uv_active", &uv.active),
        ("potter:uv_render", &uv.render),
    ] {
        if let Some(name) = name {
            properties.push(format!(
                "\t\t\tP: \"{property}\", \"KString\", \"\", \"\", \"{}\"",
                quote(name)
            ));
        }
    }
    if let Some(data) = data_block.filter(|data| {
        data.shape_keys
            .as_ref()
            .is_some_and(|keys| !keys.keys.is_empty())
            || !data.vertex_groups.is_empty()
            || !data.vertex_weights.is_empty()
            || !colors.layers.is_empty()
            || node
                .modifiers
                .iter()
                .any(|modifier| modifier.is_active(purpose) && modifier.modifier_type == "armature")
    }) {
        let metadata = serde_json::to_string(&json!({
            "shape_keys": &data.shape_keys,
            "vertex_groups": &data.vertex_groups,
            "vertex_weights": &data.vertex_weights,
            "color_roles": {"active": colors.active, "render": colors.render},
        }))
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "could not encode FBX rig and morph metadata",
                json!({"reason": error.to_string()}),
            )
        })?;
        properties.push(format!(
            "\t\t\tP: \"potter:deformation\", \"KString\", \"\", \"\", \"{}\"",
            quote(&metadata)
        ));
    }
    if !properties.is_empty() {
        writeln!(
            output,
            "\t\tProperties70:  {{\n{}\n\t\t}}",
            properties.join("\n")
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
    write_normal_layer(output, mesh)?;
    write_color_layers(output, mesh, &colors)?;
    write_uv_layers(output, mesh)?;
    // Blender's FBX exporter sends a face index at or past the slot count to the first exported
    // material (`export_fbx_bin.py`, `def_me_blmaterial_idx`). `.get` returns `None` for such an index
    // and the fallback below is that first exported slot.
    if has_material_layer {
        write!(output, "\t\tLayerElementMaterial: 0 {{\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n\t\t\tMappingInformationType: \"ByPolygon\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n\t\t\tMaterials: *{} {{\n\t\t\t\ta: ", mesh.faces.len()).map_err(write_error)?;
        for (index, face) in mesh.faces.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            let slot = usize::try_from(face.material_index).unwrap_or(usize::MAX);
            let output_slot = material_slot_indices
                .get(slot)
                .copied()
                .flatten()
                .or(fallback_material_index)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "FBX material slot fallback is missing",
                    )
                })?;
            write!(output, "{output_slot}").map_err(write_error)?;
        }
        output.push_str("\n\t\t\t}\n\t\t}\n");
    }
    output.push_str("\t\tLayer: 0 {\n\t\t\tVersion: 100\n\t\t\tLayerElement: {\n\t\t\t\tType: \"LayerElementNormal\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n");
    if has_material_layer {
        output.push_str("\t\t\tLayerElement: {\n\t\t\t\tType: \"LayerElementMaterial\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n");
    }
    for index in 0..colors.layers.len() {
        writeln!(
            output,
            "\t\t\tLayerElement: {{\n\t\t\t\tType: \"LayerElementColor\"\n\t\t\t\tTypedIndex: {index}\n\t\t\t}}"
        )
        .map_err(write_error)?;
    }
    // Blender keeps the first UV layer in layer 0 and gives each further layer its own layer.
    let uv_count = uv_layers::layer_count(mesh);
    if uv_count > 0 {
        output.push_str("\t\t\tLayerElement: {\n\t\t\t\tType: \"LayerElementUV\"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n");
    }
    output.push_str("\t\t}\n");
    for index in 1..uv_count {
        write!(
            output,
            "\t\tLayer: {index} {{\n\t\t\tVersion: 100\n\t\t\tLayerElement: {{\n\t\t\t\tType: \"LayerElementUV\"\n\t\t\t\tTypedIndex: {index}\n\t\t\t}}\n\t\t}}\n"
        )
        .map_err(write_error)?;
    }
    output.push_str("\t}\n");
    Ok(())
}

/// Writes `LayerElementNormal` in Blender's default layout: mesh-local corner normals, stored
/// as unique values and indexed per polygon vertex. `LayerElementSmoothing` is never written,
/// matching Blender's default export.
fn write_normal_layer(output: &mut String, mesh: &Mesh) -> Result<()> {
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let mut unique = BTreeMap::<[u64; 3], usize>::new();
    let mut values = Vec::<[f64; 3]>::new();
    let mut indices = Vec::<usize>::new();
    for corner in corner_normals(mesh, &points)? {
        // Zero normals are written as they are; Blender writes them for degenerate faces too.
        let value = corner.to_array();
        let key = value.map(|component| (component + 0.0).to_bits());
        let index = *unique.entry(key).or_insert_with(|| {
            values.push(value);
            values.len() - 1
        });
        indices.push(index);
    }
    write!(
        output,
        "\t\tLayerElementNormal: 0 {{\n\t\t\tVersion: 101\n\t\t\tName: \"\"\n\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n\t\t\tNormals: *{} {{\n\t\t\t\ta: ",
        values.len().saturating_mul(3)
    )
    .map_err(write_error)?;
    let components = values
        .iter()
        .flatten()
        .map(|value| fmt_num(*value))
        .collect::<Vec<_>>();
    output.push_str(&components.join(","));
    write!(
        output,
        "\n\t\t\t}}\n\t\t\tNormalsIndex: *{} {{\n\t\t\t\ta: ",
        indices.len()
    )
    .map_err(write_error)?;
    let encoded = indices.iter().map(usize::to_string).collect::<Vec<_>>();
    output.push_str(&encoded.join(","));
    output.push_str("\n\t\t\t}\n\t\t}\n");
    Ok(())
}

fn fbx_encode_color(value: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|channel| encode_byte_color(value[channel], channel == 3))
}

fn fbx_decode_color(value: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|channel| decode_byte_color(value[channel], channel == 3))
}
/// Writes FBX `BYTE_COLOR` values: RGB is sRGB encoded, alpha is linear, and all channels are
/// nearest-byte quantized. Blender imports all color layers into CORNER (`import_fbx.py`:
/// `blen_read_geom_layer_color`); `ByVertice`/`Direct` expands point values onto loops. POINT
/// domain loss is reported by `element_data_losses`. CORNER colors use ByPolygonVertex/IndexToDirect.
fn write_color_layers(output: &mut String, mesh: &Mesh, store: &ColorStore) -> Result<()> {
    for (layer_index, layer) in store.layers.iter().enumerate() {
        let expected = match layer.domain {
            ColorDomain::Point => mesh.vertices.len(),
            ColorDomain::Corner => mesh
                .faces
                .iter()
                .try_fold(0_usize, |count, face| {
                    count.checked_add(face.vertices.len())
                })
                .ok_or_else(|| {
                    PotError::new(ErrorCode::LimitExceeded, "FBX color corner count overflow")
                })?,
        };
        if layer.values.len() != expected
            || layer
                .values
                .iter()
                .flatten()
                .any(|component| !component.is_finite())
        {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "FBX color layer has invalid values for its mapping",
                json!({"layer": layer.name, "expected": expected, "actual": layer.values.len()}),
            ));
        }
        let (mapping, reference, values, indices) = match layer.domain {
            ColorDomain::Point => (
                "ByVertice",
                "Direct",
                layer
                    .values
                    .iter()
                    .copied()
                    .map(fbx_encode_color)
                    .collect::<Vec<_>>(),
                None,
            ),
            ColorDomain::Corner => {
                let mut unique = BTreeMap::<[u64; 4], usize>::new();
                let mut values = Vec::<[f64; 4]>::new();
                let mut indices = Vec::<usize>::with_capacity(layer.values.len());
                for value in &layer.values {
                    let encoded = fbx_encode_color(*value);
                    let key = encoded.map(f64::to_bits);
                    let index = *unique.entry(key).or_insert_with(|| {
                        values.push(encoded);
                        values.len() - 1
                    });
                    indices.push(index);
                }
                ("ByPolygonVertex", "IndexToDirect", values, Some(indices))
            }
        };
        let component_count = values.len().checked_mul(4).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "FBX color component count overflow",
            )
        })?;
        write!(
            output,
            "\t\tLayerElementColor: {layer_index} {{\n\t\t\tVersion: 101\n\t\t\tName: \"{}\"\n\t\t\tMappingInformationType: \"{mapping}\"\n\t\t\tReferenceInformationType: \"{reference}\"\n\t\t\tColors: *{component_count} {{\n\t\t\t\ta: ",
            quote(&layer.name)
        )
        .map_err(write_error)?;
        output.push_str(
            &values
                .iter()
                .flatten()
                .map(|value| fmt_num(*value))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push_str("\n\t\t\t}\n");
        if let Some(indices) = indices {
            write!(
                output,
                "\t\t\tColorIndex: *{} {{\n\t\t\t\ta: ",
                indices.len()
            )
            .map_err(write_error)?;
            output.push_str(
                &indices
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            output.push_str("\n\t\t\t}\n");
        }
        output.push_str("\t\t}\n");
    }
    Ok(())
}

/// Writes one `LayerElementUV` per UV layer, named by the layer, in Blender's layout: each distinct
/// value once, indexed per polygon vertex. FBX stores V up, as Blender writes it, so `v` is flipped
/// from storage.
fn write_uv_layers(output: &mut String, mesh: &Mesh) -> Result<()> {
    let store = uv_layers::read(mesh);
    for (layer_index, layer) in store.layers.iter().enumerate() {
        let mut unique = BTreeMap::<[u64; 2], usize>::new();
        let mut values = Vec::<[f64; 2]>::new();
        let mut indices = Vec::<usize>::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let entry = layer.faces.get(face_index).and_then(Option::as_ref);
            for corner in 0..face.vertices.len() {
                let [u, v] = entry
                    .and_then(|corners| corners.get(corner))
                    .copied()
                    .unwrap_or([0.0, 0.0]);
                let value = [u, 1.0 - v];
                let key = value.map(|component| (component + 0.0).to_bits());
                let index = *unique.entry(key).or_insert_with(|| {
                    values.push(value);
                    values.len() - 1
                });
                indices.push(index);
            }
        }
        write!(
            output,
            "\t\tLayerElementUV: {layer_index} {{\n\t\t\tVersion: 101\n\t\t\tName: \"{}\"\n\t\t\tMappingInformationType: \"ByPolygonVertex\"\n\t\t\tReferenceInformationType: \"IndexToDirect\"\n\t\t\tUV: *{} {{\n\t\t\t\ta: ",
            quote(&layer.name),
            values.len().saturating_mul(2)
        )
        .map_err(write_error)?;
        let components = values
            .iter()
            .flatten()
            .map(|value| fmt_num(*value))
            .collect::<Vec<_>>();
        output.push_str(&components.join(","));
        write!(
            output,
            "\n\t\t\t}}\n\t\t\tUVIndex: *{} {{\n\t\t\t\ta: ",
            indices.len()
        )
        .map_err(write_error)?;
        let encoded = indices.iter().map(usize::to_string).collect::<Vec<_>>();
        output.push_str(&encoded.join(","));
        output.push_str("\n\t\t\t}\n\t\t}\n");
    }
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
    let mut model_order = Vec::<i64>::new();
    let mut geometries = BTreeMap::<i64, ImportedGeometry>::new();
    let mut materials = BTreeMap::<i64, ImportedMaterial>::new();
    let mut textures = BTreeMap::<i64, ImportedTexture>::new();
    let mut losses = Vec::<Loss>::new();
    let mut has_deformers = false;
    for object in &objects.children {
        match object.name.as_str() {
            "Model" => {
                let id = object_id(object)?;
                let model = parse_model(object, id)?;
                if models.insert(id, model).is_some() {
                    return Err(import_error(
                        "FBX Objects section contains a duplicate object ID",
                    ));
                }
                model_order.push(id);
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
                let geometry = parse_geometry(object, id, &mut losses)?;
                for attribute in ["LayerElementTangent", "LayerElementBinormal"] {
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

    // Blender shares one mesh material table per Geometry object. Connections are processed in
    // Model object order; later models can add trailing mesh slots, and a model with fewer links
    // still sees those slots on the shared mesh.
    let mut geometry_material_slots = BTreeMap::<i64, Vec<i64>>::new();
    for model_id in &model_order {
        let Some(geometry_id) = geometry_by_model.get(model_id) else {
            continue;
        };
        let model_slots = material_links.get(model_id).map_or(&[][..], Vec::as_slice);
        let shared_slots = geometry_material_slots.entry(*geometry_id).or_default();
        if model_slots.len() > shared_slots.len() {
            shared_slots.extend_from_slice(&model_slots[shared_slots.len()..]);
        }
    }
    let mut effective_material_links = BTreeMap::<i64, Vec<i64>>::new();
    for model_id in &model_order {
        let Some(geometry_id) = geometry_by_model.get(model_id) else {
            continue;
        };
        let mut model_slots = material_links.get(model_id).cloned().unwrap_or_default();
        if let Some(shared_slots) = geometry_material_slots.get(geometry_id)
            && model_slots.len() < shared_slots.len()
        {
            model_slots.extend_from_slice(&shared_slots[model_slots.len()..]);
        }
        effective_material_links.insert(*model_id, model_slots);
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
                    alpha_mode: crate::image::storage::default_alpha_mode(&bytes),
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
            uv_layer: texture.uv_layer.clone(),
            interpolation: crate::image::ImageInterpolation::Linear,
            extension: texture.extension,
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
        let mut node_materials = Vec::<Option<Id>>::new();
        if let Some(material_numbers) = effective_material_links.get(numeric_id) {
            for material_number in material_numbers {
                if let Some(id) = material_ids.get(material_number) {
                    node_materials.push(Some(id.clone()));
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
            if let Some(normals) = &geometry.normals {
                shading::import_shading(&mut mesh, normals)?;
            }
            import_uv_layers(
                &mut mesh,
                &geometry.uv_layers,
                geometry.uv_active,
                geometry.uv_render,
            );
            color_layers::write(&mut mesh, &geometry.colors);
            for (face_index, face) in mesh.faces.iter_mut().enumerate() {
                let raw = geometry
                    .material_indices
                    .get(face_index)
                    .copied()
                    .unwrap_or(0);
                face.material_index = resolve_material_slot(raw, node_materials.len());
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
    material_indices: Vec<i64>,
    deformation: Option<Value>,
    /// Mesh-local corner normals in `faces` order, unit length; `None` when absent.
    normals: Option<Vec<DVec3>>,
    /// UV layers in file order; empty when the geometry has none.
    uv_layers: Vec<ExchangeUvLayer>,
    /// Index of the active UV layer; `None` when the geometry has no UV layer.
    uv_active: Option<usize>,
    /// Index of the render UV layer; `None` when the geometry has no UV layer.
    uv_render: Option<usize>,
    /// Color layers in file order; empty when the geometry has none.
    colors: ColorStore,
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
    uv_layer: Option<String>,
    property: Option<String>,
    /// `EXTEND` when either wrap mode is a clamp, as Blender's importer sets it (`import_fbx.py` `texture_mapping_set`).
    extension: crate::image::ImageExtension,
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

/// Blender's `validate_material_indices` (`mesh_validate.cc`) resets a negative or past-the-end
/// index to 0, the first connected material; a mesh with no materials still counts one slot.
fn resolve_material_slot(raw: i64, material_count: usize) -> u32 {
    usize::try_from(raw)
        .ok()
        .filter(|slot| *slot < material_count.max(1))
        .and_then(|slot| u32::try_from(slot).ok())
        .unwrap_or(0)
}

fn parse_geometry(
    node: &FbxNode,
    geometry_id: i64,
    losses: &mut Vec<Loss>,
) -> Result<ImportedGeometry> {
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
        let reference = layer
            .child("ReferenceInformationType")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        match (mapping, reference) {
            ("ByPolygon", _) => {
                if let Some(values) = layer.child("Materials") {
                    material_indices = integer_array(values)?;
                }
                if !material_indices.is_empty() && material_indices.len() != faces.len() {
                    return Err(import_error(
                        "FBX material index count does not match polygon count",
                    ));
                }
            }
            // Blender's `blen_read_geom_array_mapped_polygon` reads `Materials[0]` for every polygon
            // of an `AllSame` layer, and only with `IndexToDirect`. Blender writes this layer when
            // one material is exported (`export_fbx_bin.py`, `LayerElementMaterial` block).
            ("AllSame", "IndexToDirect") => {
                let first = layer
                    .child("Materials")
                    .map(integer_array)
                    .transpose()?
                    .and_then(|values| values.first().copied())
                    .ok_or_else(|| {
                        import_error("FBX AllSame material layer has no material index")
                    })?;
                material_indices = vec![first; faces.len()];
            }
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "FBX material mapping is unsupported",
                    json!({"feature_id":"format.fbx.material_mapping","mapping":mapping}),
                ));
            }
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
    let normals = parse_normal_layer(node, positions.len(), &faces, geometry_id, losses)?;
    if normals.is_none() && node.child("LayerElementSmoothing").is_some() {
        losses.push(Loss {
            feature_id: "format.fbx.smoothing".to_owned(),
            data_id: Some(geometry_id.to_string()),
            reason: "FBX LayerElementSmoothing without LayerElementNormal data is not represented in the Potter mesh model".to_owned(),
            suggestion: None,
        });
    }
    let layers = parse_uv_layers(node, positions.len(), &faces, geometry_id, losses)?;
    // A geometry without a recorded selection, or one naming no layer, selects the first layer.
    let select = |property: &str| {
        if layers.is_empty() {
            return None;
        }
        let named = read_string_property(node.child("Properties70"), property)
            .and_then(|name| layers.iter().position(|layer| layer.name == name));
        Some(named.unwrap_or(0))
    };
    let uv_active = select("potter:uv_active");
    let uv_render = select("potter:uv_render");
    let colors = parse_color_layers(
        node,
        positions.len(),
        &faces,
        geometry_id,
        deformation.as_ref(),
        losses,
    )?;
    Ok(ImportedGeometry {
        positions,
        faces,
        material_indices,
        deformation,
        normals,
        uv_layers: layers,
        uv_active,
        uv_render,
        colors,
    })
}

/// Reads `LayerElementNormal` as mesh-local corner normals in `faces` order.
///
/// Normals map per polygon vertex (`ByPolygonVertex`) or per control point (`ByVertice` or
/// `ByVertex`), stored directly (`Direct`) or through `NormalsIndex` (`IndexToDirect`). Zero
/// normals stay zero, as Blender writes them. Other mappings are reported as losses and yield no
/// normals.
fn parse_normal_layer(
    node: &FbxNode,
    point_count: usize,
    faces: &[Vec<usize>],
    geometry_id: i64,
    losses: &mut Vec<Loss>,
) -> Result<Option<Vec<DVec3>>> {
    let Some(layer) = node.child("LayerElementNormal") else {
        return Ok(None);
    };
    let mapping = layer
        .child("MappingInformationType")
        .and_then(FbxNode::first_arg)
        .unwrap_or("");
    let reference = layer
        .child("ReferenceInformationType")
        .and_then(FbxNode::first_arg)
        .unwrap_or("");
    let by_corner = match mapping {
        "ByPolygonVertex" => true,
        "ByVertice" | "ByVertex" => false,
        _ => {
            losses.push(normal_loss(
                geometry_id,
                format!("FBX LayerElementNormal mapping `{mapping}` is not represented in the Potter mesh model"),
            ));
            return Ok(None);
        }
    };
    let indexed = match reference {
        "Direct" => false,
        "IndexToDirect" => true,
        _ => {
            losses.push(normal_loss(
                geometry_id,
                format!("FBX LayerElementNormal reference `{reference}` is not represented in the Potter mesh model"),
            ));
            return Ok(None);
        }
    };
    let values = numeric_array(
        layer
            .child("Normals")
            .ok_or_else(|| import_error("FBX LayerElementNormal has no Normals array"))?,
    )?;
    if values.len() % 3 != 0 || values.iter().any(|value| !value.is_finite()) {
        return Err(import_error(
            "FBX Normals array must contain finite xyz triples",
        ));
    }
    let vectors = values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| DVec3::from_array([value[0], value[1], value[2]]))
        .collect::<Vec<_>>();
    let indices = if indexed {
        Some(integer_array(layer.child("NormalsIndex").ok_or_else(
            || import_error("FBX LayerElementNormal has no NormalsIndex array"),
        )?)?)
    } else {
        None
    };
    let polygon_points = faces.iter().flatten().copied().collect::<Vec<_>>();
    let element_count = indices.as_ref().map_or(vectors.len(), Vec::len);
    let expected = if by_corner {
        polygon_points.len()
    } else {
        point_count
    };
    if element_count != expected {
        return Err(import_error(
            "FBX LayerElementNormal count does not match its mapping",
        ));
    }
    let mut normals = Vec::with_capacity(polygon_points.len());
    for (corner, point) in polygon_points.iter().enumerate() {
        let element = if by_corner { corner } else { *point };
        let value_index = match &indices {
            Some(indices) => {
                let raw = indices
                    .get(element)
                    .ok_or_else(|| import_error("FBX LayerElementNormal index is missing"))?;
                usize::try_from(*raw)
                    .map_err(|_| import_error("FBX LayerElementNormal index is negative"))?
            }
            None => element,
        };
        let vector = vectors
            .get(value_index)
            .ok_or_else(|| import_error("FBX LayerElementNormal index is outside Normals"))?;
        normals.push(vector.normalize_or_zero());
    }
    Ok(Some(normals))
}

/// Reads every `LayerElementUV` of a geometry as a UV layer, in file order, with corners in
/// `faces` order in storage convention. FBX stores V up, as Blender does, so `v` is flipped. A
/// layer whose mapping is unsupported is a loss, and layers past the limit are losses too.
fn parse_uv_layers(
    node: &FbxNode,
    point_count: usize,
    faces: &[Vec<usize>],
    geometry_id: i64,
    losses: &mut Vec<Loss>,
) -> Result<Vec<ExchangeUvLayer>> {
    let mut layers = Vec::<ExchangeUvLayer>::new();
    for layer in node
        .children
        .iter()
        .filter(|child| child.name == "LayerElementUV")
    {
        let raw_name = layer
            .child("Name")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        if layers.len() == uv_layers::MAX_LAYERS {
            losses.push(uv_import_loss(
                &geometry_id.to_string(),
                format!(
                    "FBX geometry has more than {} UV layers; layer `{raw_name}` is not represented",
                    uv_layers::MAX_LAYERS
                ),
            ));
            continue;
        }
        let Some(values) =
            parse_uv_values(layer, point_count, faces, geometry_id, raw_name, losses)?
        else {
            continue;
        };
        let mut corners = Vec::with_capacity(faces.len());
        let mut start = 0;
        for face in faces {
            corners.push(values[start..start + face.len()].to_vec());
            start += face.len();
        }
        let name = unique_uv_name(raw_name, &layers);
        let pinned = corners.iter().map(|face| vec![false; face.len()]).collect();
        layers.push(ExchangeUvLayer {
            name,
            corners,
            pinned,
        });
    }
    Ok(layers)
}

/// Values per polygon corner, or `None` after a loss when the layer's mapping is unsupported.
fn parse_uv_values(
    layer: &FbxNode,
    point_count: usize,
    faces: &[Vec<usize>],
    geometry_id: i64,
    name: &str,
    losses: &mut Vec<Loss>,
) -> Result<Option<Vec<[f64; 2]>>> {
    let mapping = layer
        .child("MappingInformationType")
        .and_then(FbxNode::first_arg)
        .unwrap_or("");
    let reference = layer
        .child("ReferenceInformationType")
        .and_then(FbxNode::first_arg)
        .unwrap_or("");
    let by_corner = match mapping {
        "ByPolygonVertex" => true,
        "ByVertice" | "ByVertex" => false,
        _ => {
            losses.push(uv_import_loss(
                &geometry_id.to_string(),
                format!("FBX LayerElementUV `{name}` mapping `{mapping}` is not represented in the Potter mesh model"),
            ));
            return Ok(None);
        }
    };
    let indexed = match reference {
        "Direct" => false,
        "IndexToDirect" => true,
        _ => {
            losses.push(uv_import_loss(
                &geometry_id.to_string(),
                format!("FBX LayerElementUV `{name}` reference `{reference}` is not represented in the Potter mesh model"),
            ));
            return Ok(None);
        }
    };
    let raw = numeric_array(
        layer
            .child("UV")
            .ok_or_else(|| import_error("FBX LayerElementUV has no UV array"))?,
    )?;
    if raw.len() % 2 != 0 || raw.iter().any(|value| !value.is_finite()) {
        return Err(import_error("FBX UV array must contain finite uv pairs"));
    }
    let vectors = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| [pair[0], 1.0 - pair[1]])
        .collect::<Vec<_>>();
    let indices = if indexed {
        Some(integer_array(layer.child("UVIndex").ok_or_else(|| {
            import_error("FBX LayerElementUV has no UVIndex array")
        })?)?)
    } else {
        None
    };
    let polygon_points = faces.iter().flatten().copied().collect::<Vec<_>>();
    let element_count = indices.as_ref().map_or(vectors.len(), Vec::len);
    let expected = if by_corner {
        polygon_points.len()
    } else {
        point_count
    };
    if element_count != expected {
        return Err(import_error(
            "FBX LayerElementUV count does not match its mapping",
        ));
    }
    let mut values = Vec::with_capacity(polygon_points.len());
    for (corner, point) in polygon_points.iter().enumerate() {
        let element = if by_corner { corner } else { *point };
        let value_index = match &indices {
            Some(indices) => {
                let raw = indices
                    .get(element)
                    .ok_or_else(|| import_error("FBX LayerElementUV index is missing"))?;
                usize::try_from(*raw)
                    .map_err(|_| import_error("FBX LayerElementUV index is negative"))?
            }
            None => element,
        };
        values.push(
            *vectors
                .get(value_index)
                .ok_or_else(|| import_error("FBX LayerElementUV index is outside the UV array"))?,
        );
    }
    Ok(Some(values))
}

/// Reads the `LayerElementColor` layers of a geometry as color layers, in file order. The active and
/// render layers come from the `color_roles` that the export records in `potter:deformation`; without
/// them, the first layer is both. A mapping or reference Potter cannot represent is reported as a loss.
fn parse_color_layers(
    node: &FbxNode,
    point_count: usize,
    faces: &[Vec<usize>],
    geometry_id: i64,
    deformation: Option<&Value>,
    losses: &mut Vec<Loss>,
) -> Result<ColorStore> {
    let corner_count = faces.iter().map(Vec::len).sum::<usize>();
    let mut layers = Vec::<ColorLayer>::new();
    for layer in node
        .children
        .iter()
        .filter(|child| child.name == "LayerElementColor")
    {
        let raw_name = layer
            .child("Name")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        let mapping = layer
            .child("MappingInformationType")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        let domain = match mapping {
            "ByPolygonVertex" => ColorDomain::Corner,
            "ByVertice" | "ByVertex" | "ByControlPoint" => ColorDomain::Point,
            _ => {
                losses.push(color_import_loss(
                    geometry_id,
                    format!("FBX color layer `{raw_name}` mapping `{mapping}` is not represented"),
                ));
                continue;
            }
        };
        let reference = layer
            .child("ReferenceInformationType")
            .and_then(FbxNode::first_arg)
            .unwrap_or("");
        let indexed = match reference {
            "Direct" => false,
            "IndexToDirect" => true,
            _ => {
                losses.push(color_import_loss(
                    geometry_id,
                    format!(
                        "FBX color layer `{raw_name}` reference `{reference}` is not represented"
                    ),
                ));
                continue;
            }
        };
        let raw_values = numeric_array(
            layer
                .child("Colors")
                .ok_or_else(|| import_error("FBX LayerElementColor has no Colors array"))?,
        )?;
        if raw_values.len() % 4 != 0 || raw_values.iter().any(|value| !value.is_finite()) {
            return Err(import_error(
                "FBX Colors array must contain finite rgba tuples",
            ));
        }
        let direct = raw_values.as_chunks::<4>().0.to_vec();
        let indices = if indexed {
            Some(integer_array(layer.child("ColorIndex").ok_or_else(
                || import_error("FBX LayerElementColor has no ColorIndex array"),
            )?)?)
        } else {
            None
        };
        let element_count = indices.as_ref().map_or(direct.len(), Vec::len);
        let expected = match domain {
            ColorDomain::Point => point_count,
            ColorDomain::Corner => corner_count,
        };
        if element_count != expected {
            return Err(import_error(
                "FBX LayerElementColor count does not match its mapping",
            ));
        }
        let mut values = Vec::with_capacity(expected);
        for element in 0..expected {
            let direct_index = match &indices {
                Some(indices) => usize::try_from(indices[element])
                    .map_err(|_| import_error("FBX LayerElementColor index is negative"))?,
                None => element,
            };
            let encoded = *direct
                .get(direct_index)
                .ok_or_else(|| import_error("FBX LayerElementColor index is outside Colors"))?;
            values.push(fbx_decode_color(encoded));
        }
        let raw_name = if raw_name.is_empty() {
            "Color"
        } else {
            raw_name
        };
        let name = unique_color_name(raw_name, &layers);
        layers.push(ColorLayer {
            name,
            domain,
            values,
        });
    }
    let first_name = layers.first().map(|layer| layer.name.clone());
    let role = |key: &str| {
        let Some(roles) = deformation.and_then(|value| value.get("color_roles")) else {
            return first_name.clone();
        };
        match roles.get(key) {
            Some(Value::Null) => None,
            Some(Value::String(name)) if layers.iter().any(|layer| layer.name.as_str() == name) => {
                Some(name.clone())
            }
            Some(_) | None => first_name.clone(),
        }
    };
    Ok(ColorStore {
        active: role("active"),
        render: role("render"),
        layers,
    })
}

/// `name`, or `name.001`-style with the first suffix that no earlier layer uses.
fn unique_color_name(name: &str, layers: &[ColorLayer]) -> String {
    if !layers.iter().any(|layer| layer.name == name) {
        return name.to_owned();
    }
    let mut suffix = 1_usize;
    loop {
        let candidate = format!("{name}.{suffix:03}");
        if !layers.iter().any(|layer| layer.name == candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn color_import_loss(geometry_id: i64, reason: String) -> Loss {
    Loss {
        feature_id: "mesh.color_attribute".to_owned(),
        data_id: Some(geometry_id.to_string()),
        reason,
        suggestion: Some(
            "use --allow-lossy to import the file without these color layers".to_owned(),
        ),
    }
}

fn normal_loss(geometry_id: i64, reason: impl Into<String>) -> Loss {
    Loss {
        feature_id: "mesh.normals".to_owned(),
        data_id: Some(geometry_id.to_string()),
        reason: reason.into(),
        suggestion: None,
    }
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
        ior: read_scalar_property(properties, "potter:ior", 1.5)?,
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
    let uv_layer = read_string_property(properties, "potter:uv_layer_name").or_else(|| {
        read_string_property(properties, "UVSet")
            .filter(|value| !value.is_empty() && value != "default")
    });
    // Blender's importer sets EXTEND when either wrap mode is a clamp (1); REPEAT (0) keeps the default.
    let clamp = read_scalar_property(properties, "WrapModeU", 0.0)? != 0.0
        || read_scalar_property(properties, "WrapModeV", 0.0)? != 0.0;
    Ok(ImportedTexture {
        numeric_id,
        file_name,
        uv_layer,
        property: read_string_property(properties, "potter:property"),
        extension: if clamp {
            crate::image::ImageExtension::Extend
        } else {
            crate::image::ImageExtension::Repeat
        },
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

#[cfg(test)]
mod tests {
    use super::{
        export_with_root_allow_lossy, import_bytes, lex, material_slot_losses, numeric_array,
        parse_color_layers, parse_geometry, parse_nodes, parse_uv_layers, resolve_material_slot,
        uv_texture_losses, uv_texture_warnings, vertex_group_losses, write_color_layers,
        write_geometry, write_uv_layers,
    };
    use crate::{
        error::{ErrorCode, PotError, Result},
        eval::{EvaluationContext, Snapshot},
        exchange::Loss,
        geom::{
            Mesh,
            color_layers::{ColorDomain, ColorLayer, ColorStore},
            uv_layers::{self, UvLayer, UvStore},
        },
        image::{ImageExtension, ImageInterpolation},
        model::{
            DataBlock, Id, Image, ImageColorspace, ImageSource, Material, Node, SceneDoc,
            TextureRef, VertexGroup,
        },
    };
    use glam::DVec3;
    use std::{fs, path::Path};

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn triangle() -> Result<Mesh> {
        Mesh::from_positions_and_faces(vec![DVec3::ZERO, DVec3::X, DVec3::Y], vec![vec![0, 1, 2]])
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
    }

    fn assert_close(actual: &[[f64; 2]], expected: &[[f64; 2]]) {
        assert_eq!(actual.len(), expected.len());
        for (left, right) in actual.iter().zip(expected) {
            assert!(
                (left[0] - right[0]).abs() < 1.0e-12 && (left[1] - right[1]).abs() < 1.0e-12,
                "{left:?} != {right:?}"
            );
        }
    }

    #[test]
    fn blender_written_byte_color_decodes_srgb_rgb_and_linear_alpha() -> TestResult<()> {
        // The FBX add-on writes BYTE_COLOR as normalized bytes. Its 128 RGB byte decodes from
        // sRGB to ~0.21586; alpha byte 128 remains linear 128/255.
        let text = r#"Geometry: 1, "Geometry::Triangle", "Mesh" {
    LayerElementColor: 0 {
        Version: 101
        Name: "Paint"
        MappingInformationType: "ByVertice"
        ReferenceInformationType: "Direct"
        Colors: *12 {
            a: 0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255,0.5019607843137255
        }
    }
}
"#;
        let nodes = parse_nodes(&lex(text)?)?;
        let geometry = nodes.first().ok_or("FBX test has no geometry")?;
        let mut losses = Vec::new();
        let colors = parse_color_layers(geometry, 3, &[vec![0, 1, 2]], 1, None, &mut losses)?;
        assert!(losses.is_empty());
        assert_eq!(colors.layers[0].domain, ColorDomain::Point);
        assert!((colors.layers[0].values[0][0] - 0.215_860_500_113_899_26).abs() < 1.0e-12);
        assert!((colors.layers[0].values[0][3] - 128.0 / 255.0).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn point_color_writer_uses_byvertice_direct_and_blender_byte_encoding() -> TestResult<()> {
        let mesh = triangle()?;
        let store = ColorStore {
            layers: vec![ColorLayer {
                name: "Paint".to_owned(),
                domain: ColorDomain::Point,
                values: vec![[0.5, 0.5, 0.5, 0.5]; 3],
            }],
            active: Some("Paint".to_owned()),
            render: Some("Paint".to_owned()),
        };
        let mut text = String::new();
        write_color_layers(&mut text, &mesh, &store)?;
        assert!(text.contains("MappingInformationType: \"ByVertice\""));
        assert!(text.contains("ReferenceInformationType: \"Direct\""));
        assert!(!text.contains("ColorIndex"));
        let wrapped = format!("Geometry: 1, \"Geometry::Triangle\", \"Mesh\" {{\n{text}}}\n");
        let nodes = parse_nodes(&lex(&wrapped)?)?;
        let geometry = nodes.first().ok_or("FBX test has no geometry")?;
        let raw = numeric_array(
            geometry
                .child("LayerElementColor")
                .and_then(|layer| layer.child("Colors"))
                .ok_or("FBX color array missing")?,
        )?;
        assert!((raw[0] - 188.0 / 255.0).abs() < 1.0e-12);
        assert!((raw[3] - 128.0 / 255.0).abs() < 1.0e-12);
        let mut losses = Vec::new();
        let colors = parse_color_layers(geometry, 3, &[vec![0, 1, 2]], 1, None, &mut losses)?;
        assert!((colors.layers[0].values[0][0] - 0.502_886_458_032_568_7).abs() < 1.0e-9);
        assert!((colors.layers[0].values[0][3] - 128.0 / 255.0).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn unbound_vertex_groups_report_their_blender_fbx_loss() -> TestResult<()> {
        use std::collections::BTreeMap;

        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let group_id = Id::new("top_group")?;
        let data_id = Id::new("weighted_mesh")?;
        let node_id = Id::new("weighted_node")?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(triangle()?),
                vertex_groups: vec![VertexGroup {
                    id: group_id.clone(),
                    name: "Top".to_owned(),
                }],
                vertex_weights: BTreeMap::from([(0, BTreeMap::from([(group_id, 0.5)]))]),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Weighted".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = vertex_group_losses(&doc, &snapshot)?;
        assert_eq!(losses.len(), 1);
        assert_eq!(losses[0].feature_id, "mesh.vertex_group");
        assert!(losses[0].reason.contains("`Top`"));
        Ok(())
    }

    #[test]
    fn layer_element_uv_round_trips_names_order_and_storage_values() -> Result<()> {
        let mut mesh = triangle()?;
        let corners = |values: [[f64; 2]; 3]| vec![Some(values.to_vec())];
        uv_layers::write(
            &mut mesh,
            &UvStore {
                layers: vec![
                    UvLayer::new("UVMap", corners([[0.1, 0.25], [0.5, 0.75], [0.9, 0.0]])),
                    UvLayer::new("Second", corners([[0.2, 0.1], [0.6, 0.3], [0.4, 0.0]])),
                ],
                active: Some("UVMap".to_owned()),
                render: Some("UVMap".to_owned()),
            },
        );
        let mut text = String::from("Geometry: 1, \"Geometry::Triangle\", \"Mesh\" {\n");
        write_uv_layers(&mut text, &mesh)?;
        text.push_str("}\n");
        let nodes = parse_nodes(&lex(&text)?)?;
        let geometry = nodes
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "FBX test has no geometry"))?;
        let mut losses = Vec::<Loss>::new();
        let layers = parse_uv_layers(geometry, 3, &[vec![0, 1, 2]], 1, &mut losses)?;
        assert!(losses.is_empty());
        let names = layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["UVMap", "Second"]);
        assert_close(
            &layers[0].corners[0],
            &[[0.1, 0.25], [0.5, 0.75], [0.9, 0.0]],
        );
        assert_close(&layers[1].corners[0], &[[0.2, 0.1], [0.6, 0.3], [0.4, 0.0]]);
        Ok(())
    }

    #[test]
    fn color_and_generic_attributes_are_not_refused_by_the_geometry_writer() -> Result<()> {
        let mesh = crate::exchange::test_support::triangle_with_attributes()?;
        let node = crate::model::Node {
            name: "Triangle".to_owned(),
            ..crate::model::Node::default()
        };
        let mut output = String::new();
        let node_id = Id::new("triangle")?;
        write_geometry(
            &mut output,
            1,
            &node_id,
            &node,
            &mesh,
            None,
            0,
            &[],
            true,
            crate::eval::EvaluationPurpose::Preview,
        )?;
        assert!(output.contains("Geometry: 1, \"Geometry::Triangle\", \"Mesh\""));
        // The losses for these attributes come from `exchange::attribute_losses`.
        let losses = crate::exchange::mesh_attribute_losses("fbx", "triangle", &mesh);
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.color_attribute")
        );
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "mesh.attribute")
        );
        Ok(())
    }

    #[test]
    fn empty_material_slots_do_not_shift_later_fbx_bindings() -> TestResult<()> {
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let data_id = Id::new("mesh_data")?;
        let node_id = Id::new("sheet")?;
        let mut mesh = triangle()?;
        mesh.faces[0].material_index = 2;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials
            .insert(first_material.clone(), Material::default());
        doc.materials
            .insert(second_material.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                materials: vec![Some(first_material), None, Some(second_material)],
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id.clone()),
                ..Node::default()
            },
        );
        let member_collection_id = Id::new("members")?;
        let instance_id = Id::new("instance")?;
        doc.collections.insert(
            member_collection_id.clone(),
            crate::model::Collection {
                name: "Members".to_owned(),
                children: Vec::new(),
                objects: vec![node_id.clone()],
            },
        );
        let mut instance_properties = serde_json::Map::new();
        instance_properties.insert(
            "instance_collection".to_owned(),
            serde_json::json!(member_collection_id),
        );
        doc.nodes.insert(
            instance_id.clone(),
            Node {
                name: "Instance".to_owned(),
                kind: "collection_instance".to_owned(),
                properties: instance_properties,
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id.clone());
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(instance_id.clone());

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        assert!(snapshot.meshes.contains_key(&instance_id));
        let losses = material_slot_losses(&doc, &snapshot);
        assert_eq!(losses.len(), 1);
        assert_eq!(losses[0].feature_id, "format.fbx.material_slot_empty");
        let Err(strict_error) =
            export_with_root_allow_lossy(&doc, &snapshot, Path::new("."), false)
        else {
            return Err("strict FBX export must reject an empty material slot".into());
        };
        assert_eq!(
            strict_error.details["feature_id"].as_str(),
            Some("format.fbx.material_slot_empty")
        );

        let exported = export_with_root_allow_lossy(&doc, &snapshot, Path::new("."), true)?;
        let text = String::from_utf8(exported)?;
        assert!(text.contains("Materials: *1 {\n\t\t\t\ta: 1"));
        let mut null_face_doc = doc.clone();
        let face = null_face_doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .and_then(|mesh| mesh.faces.first_mut())
            .ok_or("FBX test face is missing")?;
        face.material_index = 1;
        let null_face_snapshot = Snapshot::evaluate(&null_face_doc, &EvaluationContext::default())?;
        let null_face_export = export_with_root_allow_lossy(
            &null_face_doc,
            &null_face_snapshot,
            Path::new("."),
            true,
        )?;
        let null_face_text = String::from_utf8(null_face_export)?;
        assert!(null_face_text.contains("Materials: *1 {\n\t\t\t\ta: 0"));

        let mut no_slot_doc = doc.clone();
        let data = no_slot_doc
            .data_blocks
            .get_mut(&data_id)
            .ok_or("FBX test data is missing")?;
        data.materials.clear();
        data.mesh
            .as_mut()
            .and_then(|mesh| mesh.faces.first_mut())
            .ok_or("FBX test face is missing")?
            .material_index = 0;
        let no_slot_snapshot = Snapshot::evaluate(&no_slot_doc, &EvaluationContext::default())?;
        let no_slot_export =
            export_with_root_allow_lossy(&no_slot_doc, &no_slot_snapshot, Path::new("."), true)?;
        assert!(!String::from_utf8(no_slot_export)?.contains("LayerElementMaterial: 0"));
        Ok(())
    }

    /// One triangle whose base colour texture has `extension`.
    fn textured_triangle_scene(extension: ImageExtension, root: &Path) -> TestResult<SceneDoc> {
        let image_path = root.join("texture.pixels");
        let pixels = crate::image::encode_pixels(1, 1, &[[1.0, 1.0, 1.0, 1.0]])?;
        fs::write(&image_path, &pixels)?;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let image_id = Id::new("image")?;
        doc.images.insert(
            image_id.clone(),
            Image {
                name: "Image".to_owned(),
                source: ImageSource::File,
                colorspace: ImageColorspace::Srgb,
                width: 1,
                height: 1,
                tiles: Vec::new(),
                blob: None,
                source_path: Some(image_path.to_string_lossy().into_owned()),
                source_hash: Some(crate::hash::sha256(&pixels)),
                alpha_mode: crate::model::ImageAlphaMode::default(),
            },
        );
        let material_id = Id::new("textured")?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                name: "Textured".to_owned(),
                base_color: [1.0; 4],
                base_color_texture: Some(TextureRef {
                    image: image_id,
                    uv_layer: None,
                    interpolation: ImageInterpolation::Linear,
                    extension,
                }),
                ..Material::default()
            },
        );
        let data_id = Id::new("triangle_data")?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(triangle()?),
                materials: vec![Some(material_id)],
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            Id::new("sheet")?,
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                ..Node::default()
            },
        );
        Ok(doc)
    }

    fn textured_triangle_fbx(extension: ImageExtension, root: &Path) -> TestResult<Vec<u8>> {
        let doc = textured_triangle_scene(extension, root)?;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        Ok(export_with_root_allow_lossy(&doc, &snapshot, root, true)?)
    }

    fn textured_extension(doc: &SceneDoc) -> TestResult<ImageExtension> {
        doc.materials
            .values()
            .find_map(|material| material.base_color_texture.as_ref())
            .map(|texture| texture.extension)
            .ok_or_else(|| "FBX test texture is missing".into())
    }

    #[test]
    fn mirror_and_clip_texture_extensions_report_fbx_losses() -> TestResult {
        let root = tempfile::tempdir()?;
        for extension in [
            ImageExtension::Repeat,
            ImageExtension::Extend,
            ImageExtension::Mirror,
            ImageExtension::Clip,
        ] {
            let doc = textured_triangle_scene(extension, root.path())?;
            let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
            let has_extension_loss = uv_texture_losses(&doc, &snapshot)
                .iter()
                .any(|loss| loss.feature_id == "format.fbx.texture_extension");
            assert_eq!(
                has_extension_loss,
                matches!(extension, ImageExtension::Mirror | ImageExtension::Clip),
                "unexpected FBX texture-extension loss for {extension:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn texture_wrap_modes_export_as_blenders_exporter_writes_them() -> TestResult {
        // Blender's exporter writes WrapModeU and WrapModeV as clamp (1) for any extension but REPEAT (0).
        let root = tempfile::tempdir()?;
        for (extension, expected) in [
            (ImageExtension::Extend, 1),
            (ImageExtension::Mirror, 1),
            (ImageExtension::Repeat, 0),
        ] {
            let text = String::from_utf8(textured_triangle_fbx(extension, root.path())?)?;
            assert!(
                text.contains(&format!(
                    "P: \"WrapModeU\", \"enum\", \"\", \"\", {expected}\n"
                )),
                "WrapModeU is {expected} for {extension:?}"
            );
            assert!(
                text.contains(&format!(
                    "P: \"WrapModeV\", \"enum\", \"\", \"\", {expected}\n"
                )),
                "WrapModeV is {expected} for {extension:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn texture_wrap_modes_import_as_blenders_importer_reads_them() -> TestResult {
        // Blender's importer sets EXTEND when either wrap mode is a clamp; REPEAT only when both are 0.
        let root = tempfile::tempdir()?;
        let scene_id = "00000000-0000-4000-8000-000000000001".to_owned();
        let clamp = textured_triangle_fbx(ImageExtension::Extend, root.path())?;
        let imported = import_bytes(&clamp, scene_id.clone())?;
        assert_eq!(textured_extension(&imported.doc)?, ImageExtension::Extend);

        let clamp_on_v_only = String::from_utf8(clamp)?.replace(
            "P: \"WrapModeU\", \"enum\", \"\", \"\", 1",
            "P: \"WrapModeU\", \"enum\", \"\", \"\", 0",
        );
        let imported = import_bytes(clamp_on_v_only.as_bytes(), scene_id.clone())?;
        assert_eq!(textured_extension(&imported.doc)?, ImageExtension::Extend);

        let repeat = textured_triangle_fbx(ImageExtension::Repeat, root.path())?;
        let imported = import_bytes(&repeat, scene_id)?;
        assert_eq!(textured_extension(&imported.doc)?, ImageExtension::Repeat);
        Ok(())
    }

    #[test]
    fn out_of_range_face_material_exports_as_the_first_exported_slot() -> TestResult<()> {
        // Blender's FBX exporter sends an index past the slot count to the first exported material
        // (`export_fbx_bin.py`, `def_me_blmaterial_idx`), and validation keeps such a face legal.
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let data_id = Id::new("mesh_data")?;
        let node_id = Id::new("sheet")?;
        let mut mesh = triangle()?;
        mesh.faces[0].material_index = 5;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials
            .insert(first_material.clone(), Material::default());
        doc.materials
            .insert(second_material.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                materials: vec![Some(first_material), Some(second_material)],
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);

        let report = crate::validate::validate(&doc, Some("fbx"))?;
        assert!(!report.has_errors, "{}", report.value);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let exported = String::from_utf8(export_with_root_allow_lossy(
            &doc,
            &snapshot,
            Path::new("."),
            false,
        )?)?;
        // Slot 0 is FBX material index 0, and the out-of-range face takes it.
        assert!(
            exported.contains("Materials: *1 {\n\t\t\t\ta: 0"),
            "{exported}"
        );
        Ok(())
    }

    #[test]
    fn invalid_material_index_resolves_to_the_first_connected_material() {
        // Blender's `validate_material_indices` (`mesh_validate.cc`) resets an index outside
        // `0..max(count, 1)` to 0, and `import_fbx.py` runs it after linking the materials.
        assert_eq!(resolve_material_slot(-1, 2), 0);
        assert_eq!(resolve_material_slot(2, 2), 0);
        assert_eq!(resolve_material_slot(1, 2), 1);
        assert_eq!(resolve_material_slot(-1, 0), 0);
        assert_eq!(resolve_material_slot(4, 0), 0);
    }

    #[test]
    fn blender_written_material_layer_reads_its_index_array() -> TestResult<()> {
        // Blender writes a `ByPolygon` `IndexToDirect` material layer of FBX material indices
        // (`export_fbx_bin.py`, `LayerElementMaterial` block, lines 1462-1520). Two triangles of a quad.
        let text = r#"Geometry: 1, "Geometry::Plane", "Mesh" {
    Vertices: *12 {
        a: 0,0,0,1,0,0,0,1,0,1,1,0
    }
    PolygonVertexIndex: *6 {
        a: 0,1,-3,1,3,-3
    }
    LayerElementMaterial: 0 {
        Version: 101
        Name: ""
        MappingInformationType: "ByPolygon"
        ReferenceInformationType: "IndexToDirect"
        Materials: *2 {
            a: 1,0
        }
    }
}
"#;
        let nodes = parse_nodes(&lex(text)?)?;
        let geometry = nodes.first().ok_or("FBX test has no geometry")?;
        let mut losses = Vec::<Loss>::new();
        let imported = parse_geometry(geometry, 1, &mut losses)?;
        assert!(losses.is_empty());
        assert_eq!(imported.faces, vec![vec![0, 1, 2], vec![1, 3, 2]]);
        assert_eq!(imported.material_indices, vec![1, 0]);
        Ok(())
    }

    #[test]
    fn blender_written_all_same_material_layer_applies_to_every_face() -> TestResult<()> {
        // Blender writes `AllSame` / `IndexToDirect` with one `Materials` entry when one material
        // is exported (`export_fbx_bin.py`, `LayerElementMaterial` block, `all_same_idx`), and
        // `import_fbx.py` `blen_read_geom_array_foreach_set_allsame` broadcasts that entry to every
        // polygon. Same quad as the `ByPolygon` test above.
        let text = r#"Geometry: 1, "Geometry::Plane", "Mesh" {
    Vertices: *12 {
        a: 0,0,0,1,0,0,0,1,0,1,1,0
    }
    PolygonVertexIndex: *6 {
        a: 0,1,-3,1,3,-3
    }
    LayerElementMaterial: 0 {
        Version: 101
        Name: ""
        MappingInformationType: "AllSame"
        ReferenceInformationType: "IndexToDirect"
        Materials: *1 {
            a: 0
        }
    }
}
"#;
        let nodes = parse_nodes(&lex(text)?)?;
        let geometry = nodes.first().ok_or("FBX test has no geometry")?;
        let mut losses = Vec::<Loss>::new();
        let imported = parse_geometry(geometry, 1, &mut losses)?;
        assert!(losses.is_empty());
        assert_eq!(imported.faces, vec![vec![0, 1, 2], vec![1, 3, 2]]);
        assert_eq!(imported.material_indices, vec![0, 0]);
        Ok(())
    }

    #[test]
    fn imported_invalid_face_material_index_resolves_to_the_first_material() -> TestResult<()> {
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let data_id = Id::new("mesh_data")?;
        let node_id = Id::new("sheet")?;
        let mut mesh = triangle()?;
        mesh.faces[0].material_index = 1;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials
            .insert(first_material.clone(), Material::default());
        doc.materials
            .insert(second_material.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                materials: vec![Some(first_material.clone()), Some(second_material.clone())],
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let exported = String::from_utf8(export_with_root_allow_lossy(
            &doc,
            &snapshot,
            Path::new("."),
            false,
        )?)?;
        let face_line = "Materials: *1 {\n\t\t\t\ta: 1";
        assert!(exported.contains(face_line), "{exported}");

        // The unmodified export keeps face 0 on the second material.
        let round_trip = import_bytes(
            exported.as_bytes(),
            "00000000-0000-4000-8000-000000000001".to_owned(),
        )?;
        assert_eq!(imported_face_slots(&round_trip.doc)?, vec![1]);
        assert_eq!(
            resolved_material_id(&round_trip.doc, "Sheet", 1)?,
            second_material
        );

        // Blender resets -1 and past-the-end indices to 0, the first connected material. The import
        // never produces an empty slot for them and reports no material loss.
        for invalid in ["-1", "7"] {
            let text = exported.replace(
                face_line,
                &format!("Materials: *1 {{\n\t\t\t\ta: {invalid}"),
            );
            let imported = import_bytes(
                text.as_bytes(),
                "00000000-0000-4000-8000-000000000001".to_owned(),
            )?;
            assert_eq!(imported_face_slots(&imported.doc)?, vec![0], "{invalid}");
            assert_eq!(
                resolved_material_id(&imported.doc, "Sheet", 0)?,
                first_material
            );
            assert!(
                imported
                    .losses
                    .iter()
                    .all(|loss| !loss.feature_id.contains("material")),
                "{invalid}"
            );
        }
        Ok(())
    }

    #[test]
    fn missing_material_link_creates_no_slot_and_faces_resolve_to_connected_materials()
    -> TestResult<()> {
        // Blender's `connection_filter_reverse` (`import_fbx.py`) keeps only Model connections whose
        // target exists and is a `Material`, in file order, and `mats_to_append` builds the mesh's
        // material list from those alone. A dropped link therefore leaves no placeholder slot, and
        // `validate_material_indices` resets an index past the connected list to 0.
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let data_id = Id::new("mesh_data")?;
        let node_id = Id::new("sheet")?;
        let mut mesh = triangle()?;
        mesh.faces[0].material_index = 1;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials
            .insert(first_material.clone(), Material::default());
        doc.materials
            .insert(second_material.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                materials: vec![Some(first_material.clone()), Some(second_material.clone())],
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let exported = String::from_utf8(export_with_root_allow_lossy(
            &doc,
            &snapshot,
            Path::new("."),
            false,
        )?)?;

        // Find the FBX object id of the second material from its `potter:id` property.
        let mut current_object = None::<&str>;
        let mut second_object = None::<&str>;
        for line in exported.lines() {
            if let Some(rest) = line.strip_prefix("\tMaterial: ") {
                current_object = rest.split(',').next();
            } else if line.ends_with("\"second_material\"") && line.contains("\"potter:id\"") {
                second_object = current_object;
            }
        }
        let second_object = second_object.ok_or("FBX export has no second material object")?;

        // Drop the second material's Model connection, leaving its Material object in the file.
        let link = format!("\tC: \"OO\",{second_object},");
        let before = exported
            .lines()
            .filter(|line| line.starts_with(&link))
            .count();
        assert_eq!(before, 1, "{exported}");
        let mut text = exported
            .lines()
            .filter(|line| !line.starts_with(&link))
            .collect::<Vec<_>>()
            .join("\n");
        text.push('\n');

        let imported = import_bytes(
            text.as_bytes(),
            "00000000-0000-4000-8000-000000000001".to_owned(),
        )?;
        // Face 0 pointed at FBX slot 1 (second material). With only the first material connected,
        // the index is past the list and resolves to the first connected material.
        assert_eq!(imported_face_slots(&imported.doc)?, vec![0]);
        assert_eq!(
            resolved_material_id(&imported.doc, "Sheet", 0)?,
            first_material
        );
        let sheet = imported
            .doc
            .nodes
            .values()
            .find(|node| node.name == "Sheet")
            .ok_or("imported FBX has no Sheet node")?;
        assert_eq!(
            sheet.materials.len(),
            1,
            "a dropped link must not create an empty slot"
        );
        Ok(())
    }

    #[test]
    fn shared_geometry_inherits_trailing_material_slots_in_model_order() -> TestResult<()> {
        // Blender's importer builds a shared mesh material table in Model processing order. The
        // second model has one connection but still sees the first model's trailing slot on shared
        // geometry, so face index 1 continues to name the second material.
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let first_node = Id::new("first_node")?;
        let second_node = Id::new("second_node")?;
        let data_id = Id::new("shared_mesh")?;
        let first = Material {
            name: "FirstMaterial".to_owned(),
            ..Material::default()
        };
        let second = Material {
            name: "SecondMaterial".to_owned(),
            ..Material::default()
        };
        let mut mesh = triangle()?;
        mesh.faces[0].material_index = 1;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials.insert(first_material.clone(), first);
        doc.materials.insert(second_material.clone(), second);
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            first_node.clone(),
            Node {
                name: "First".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id.clone()),
                materials: vec![Some(first_material.clone()), Some(second_material.clone())],
                ..Node::default()
            },
        );
        doc.nodes.insert(
            second_node.clone(),
            Node {
                name: "Second".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                materials: vec![Some(first_material)],
                ..Node::default()
            },
        );
        let root = doc
            .collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?;
        root.objects.extend([first_node, second_node]);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let exported = String::from_utf8(export_with_root_allow_lossy(
            &doc,
            &snapshot,
            Path::new("."),
            false,
        )?)?;
        let object_id = |kind: &str, name: &str| -> TestResult<String> {
            let prefix = format!("\t{kind}: ");
            let marker = format!("::{name}\"");
            let line = exported
                .lines()
                .find(|line| line.starts_with(&prefix) && line.contains(&marker))
                .ok_or("FBX object is missing")?;
            let object_id = line
                .strip_prefix(&prefix)
                .ok_or("FBX object line has no ID")?
                .split(',')
                .next()
                .ok_or("FBX object ID is missing")?;
            Ok(object_id.to_owned())
        };
        let first_geometry = object_id("Geometry", "First")?;
        let second_geometry = object_id("Geometry", "Second")?;
        let second_model = object_id("Model", "Second")?;
        let original_link = format!("\tC: \"OO\",{second_geometry},{second_model}");
        assert!(exported.contains(&original_link), "{exported}");
        let shared_link = format!("\tC: \"OO\",{first_geometry},{second_model}");
        let shared = exported.replacen(&original_link, &shared_link, 1);
        let imported = import_bytes(
            shared.as_bytes(),
            "00000000-0000-4000-8000-000000000001".to_owned(),
        )?;
        let second_node = imported
            .doc
            .nodes
            .values()
            .find(|node| node.name == "Second")
            .ok_or("second FBX node is missing")?;
        assert_eq!(second_node.materials.len(), 2);
        let data = imported
            .doc
            .data_blocks
            .get(
                second_node
                    .data
                    .as_ref()
                    .ok_or("second mesh data is missing")?,
            )
            .ok_or("second mesh data is missing")?;
        let mesh = data.mesh.as_ref().ok_or("second mesh is missing")?;
        assert_eq!(mesh.faces[0].material_index, 1);
        let second_slot = second_node.materials[1]
            .as_ref()
            .ok_or("inherited material slot is empty")?;
        assert_eq!(imported.doc.materials[second_slot].name, "SecondMaterial");
        Ok(())
    }

    #[test]
    fn unassigned_and_out_of_range_faces_export_the_first_exported_material_index() -> TestResult<()>
    {
        // Blender's exporter gives an empty or past-the-end face the first exported material's FBX
        // index (`export_fbx_bin.py`, `def_me_blmaterial_idx`), never -1. With slots
        // [empty, A, B], A is FBX index 0 and B is FBX index 1.
        let first_material = Id::new("first_material")?;
        let second_material = Id::new("second_material")?;
        let data_id = Id::new("mesh_data")?;
        let node_id = Id::new("sheet")?;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.materials
            .insert(first_material.clone(), Material::default());
        doc.materials
            .insert(second_material.clone(), Material::default());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(triangle()?),
                materials: vec![None, Some(first_material), Some(second_material)],
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Sheet".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id.clone()),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);

        // (face slot, expected FBX index): the empty slot 0, A at slot 1, B at slot 2, past the end at 7.
        for (face_slot, expected) in [(0_u32, 0_u32), (1, 0), (2, 1), (7, 0)] {
            let mut face_doc = doc.clone();
            face_doc
                .data_blocks
                .get_mut(&data_id)
                .and_then(|data| data.mesh.as_mut())
                .and_then(|mesh| mesh.faces.first_mut())
                .ok_or("FBX test face is missing")?
                .material_index = face_slot;
            let snapshot = Snapshot::evaluate(&face_doc, &EvaluationContext::default())?;
            let exported = String::from_utf8(export_with_root_allow_lossy(
                &face_doc,
                &snapshot,
                Path::new("."),
                true,
            )?)?;
            assert!(
                exported.contains(&format!("Materials: *1 {{\n\t\t\t\ta: {expected}")),
                "slot {face_slot}: {exported}"
            );
        }

        // The empty slot still refuses strict export and is still reported as a loss.
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let losses = material_slot_losses(&doc, &snapshot);
        assert_eq!(losses.len(), 1);
        assert_eq!(losses[0].feature_id, "format.fbx.material_slot_empty");
        let Err(strict_error) =
            export_with_root_allow_lossy(&doc, &snapshot, Path::new("."), false)
        else {
            return Err("strict FBX export must reject an empty material slot".into());
        };
        assert_eq!(
            strict_error.details["feature_id"].as_str(),
            Some("format.fbx.material_slot_empty")
        );
        Ok(())
    }

    fn imported_face_slots(doc: &SceneDoc) -> TestResult<Vec<u32>> {
        let mesh = doc
            .data_blocks
            .values()
            .find_map(|data| data.mesh.as_ref())
            .ok_or("FBX test mesh is missing")?;
        Ok(mesh.faces.iter().map(|face| face.material_index).collect())
    }

    fn add_member_mesh(
        doc: &mut SceneDoc,
        node_name: &str,
        first_layer: &str,
        render_layer: &str,
        first_uv: [f64; 2],
        detail_uv: [f64; 2],
        render_uv: [f64; 2],
        material_id: &Id,
    ) -> TestResult<()> {
        let node_id = Id::new(node_name)?;
        let data_id = Id::new(format!("{node_name}_data"))?;
        let mut mesh = triangle()?;
        mesh.faces
            .first_mut()
            .ok_or("FBX test mesh has no face")?
            .material_index = 1;
        let corners = |uv| vec![Some(vec![uv; 3])];
        uv_layers::write(
            &mut mesh,
            &UvStore {
                layers: vec![
                    UvLayer::new(first_layer, corners(first_uv)),
                    UvLayer::new("Detail", corners(detail_uv)),
                    UvLayer::new(render_layer, corners(render_uv)),
                ],
                active: Some(render_layer.to_owned()),
                render: Some(render_layer.to_owned()),
            },
        );
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: node_name.to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                materials: vec![Some(material_id.clone()); 2],
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("FBX test root collection is missing")?
            .objects
            .push(node_id);
        Ok(())
    }

    fn resolved_material_id(doc: &SceneDoc, node_name: &str, slot: usize) -> TestResult<Id> {
        let (node_id, node) = doc
            .nodes
            .iter()
            .find(|(_, node)| node.name == node_name)
            .ok_or("FBX test mesh is missing")?;
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let data_block = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        snapshot
            .slot_material(node_id, node, data_block, slot)
            .cloned()
            .ok_or_else(|| "FBX test mesh has no material".into())
    }

    fn texture_on_node<'a>(doc: &'a SceneDoc, node_name: &str) -> TestResult<&'a TextureRef> {
        let material_id = resolved_material_id(doc, node_name, 0)?;
        doc.materials
            .get(&material_id)
            .and_then(|material| material.base_color_texture.as_ref())
            .ok_or_else(|| "FBX test material texture is missing".into())
    }

    fn sample_node_texture(doc: &SceneDoc, node_name: &str, root: &Path) -> TestResult<[f64; 4]> {
        let node = doc
            .nodes
            .values()
            .find(|node| node.name == node_name)
            .ok_or("FBX test mesh is missing")?;
        let data_id = node.data.as_ref().ok_or("FBX test mesh has no data")?;
        let mesh = doc
            .data_blocks
            .get(data_id)
            .and_then(|data| data.mesh.as_ref())
            .ok_or("FBX test mesh data is missing")?;
        let material_id = resolved_material_id(doc, node_name, 0)?;
        let material = doc
            .materials
            .get(&material_id)
            .ok_or("FBX test material is missing")?;
        let texture = material
            .base_color_texture
            .as_ref()
            .ok_or("FBX test material texture is missing")?;
        let image = doc
            .images
            .get(&texture.image)
            .ok_or("FBX test image is missing")?;
        let mut images = std::collections::BTreeMap::new();
        images.insert(
            texture.image.to_string(),
            crate::image::load_image_data(image, root, texture.interpolation)?,
        );

        let store = uv_layers::read(mesh);
        let names = store
            .layers
            .iter()
            .map(|layer| layer.name.clone())
            .collect::<Vec<_>>();
        let corners = store
            .layers
            .iter()
            .map(|layer| -> TestResult<[[f64; 2]; 3]> {
                layer
                    .faces
                    .first()
                    .and_then(Option::as_ref)
                    .ok_or("FBX test UV face is missing")?
                    .as_slice()
                    .try_into()
                    .map_err(|_| "FBX test UV face is not a triangle".into())
            })
            .collect::<TestResult<Vec<_>>>()?;
        let hit_uvs = crate::shader::HitUvLayers {
            names: &names,
            corners: &corners,
            positions: [DVec3::ZERO, DVec3::X, DVec3::Y],
            barycentric: [1.0, 0.0, 0.0],
            render: store.render_index(),
            available_mask: None,
        };
        let hit = crate::shader::HitContext::new(&doc.node_groups)
            .with_images(&images)
            .with_uv_layers(hit_uvs);
        Ok(crate::shader::evaluate_surface(material, &hit)?.base_color)
    }

    fn assert_pixel(actual: [f64; 4], expected: [f64; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }

    #[test]
    fn shared_material_uv_variants_round_trip_and_sample_member_uvs() -> TestResult {
        let root = tempfile::tempdir()?;
        let image_path = root.path().join("texture.pixels");
        let pixels = crate::image::encode_pixels(
            4,
            1,
            &[
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        )?;
        fs::write(&image_path, &pixels)?;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let image_id = Id::new("image")?;
        doc.images.insert(
            image_id.clone(),
            Image {
                name: "Image".to_owned(),
                source: ImageSource::File,
                colorspace: ImageColorspace::Srgb,
                width: 4,
                height: 1,
                tiles: Vec::new(),
                blob: None,
                source_path: Some(image_path.to_string_lossy().into_owned()),
                source_hash: Some(crate::hash::sha256(&pixels)),
                alpha_mode: crate::model::ImageAlphaMode::default(),
            },
        );
        let material_id = Id::new("shared")?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                name: "Shared".to_owned(),
                base_color: [1.0; 4],
                base_color_texture: Some(TextureRef {
                    image: image_id,
                    uv_layer: None,
                    interpolation: ImageInterpolation::Linear,
                    extension: crate::image::ImageExtension::Repeat,
                }),
                ..Material::default()
            },
        );
        add_member_mesh(
            &mut doc,
            "left",
            "LeftFirst",
            "LeftRender",
            [0.875, 0.5],
            [0.375, 0.5],
            [0.125, 0.5],
            &material_id,
        )?;
        add_member_mesh(
            &mut doc,
            "right",
            "RightFirst",
            "RightRender",
            [0.125, 0.5],
            [0.625, 0.5],
            [0.875, 0.5],
            &material_id,
        )?;

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        assert!(uv_texture_losses(&doc, &snapshot).is_empty());
        let bytes = export_with_root_allow_lossy(&doc, &snapshot, root.path(), false)?;
        let imported = import_bytes(&bytes, "00000000-0000-4000-8000-000000000001".to_owned())?;
        assert_ne!(
            resolved_material_id(&imported.doc, "left", 0)?,
            resolved_material_id(&imported.doc, "right", 0)?
        );
        assert_eq!(
            resolved_material_id(&imported.doc, "left", 0)?,
            resolved_material_id(&imported.doc, "left", 1)?
        );
        assert_eq!(
            texture_on_node(&imported.doc, "left")?.uv_layer.as_deref(),
            Some("LeftRender")
        );
        assert_eq!(
            texture_on_node(&imported.doc, "right")?.uv_layer.as_deref(),
            Some("RightRender")
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "left", root.path())?,
            [1.0, 0.0, 0.0, 1.0],
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "right", root.path())?,
            [0.0, 0.0, 1.0, 1.0],
        );

        doc.materials
            .get_mut(&material_id)
            .and_then(|material| material.base_color_texture.as_mut())
            .ok_or("FBX test shared texture disappeared")?
            .uv_layer = Some("Detail".to_owned());
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        assert!(uv_texture_losses(&doc, &snapshot).is_empty());
        let bytes = export_with_root_allow_lossy(&doc, &snapshot, root.path(), false)?;
        let imported = import_bytes(&bytes, "00000000-0000-4000-8000-000000000002".to_owned())?;
        assert_eq!(
            texture_on_node(&imported.doc, "left")?.uv_layer.as_deref(),
            Some("Detail")
        );
        assert_eq!(
            texture_on_node(&imported.doc, "right")?.uv_layer.as_deref(),
            Some("Detail")
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "left", root.path())?,
            [0.0, 1.0, 0.0, 1.0],
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "right", root.path())?,
            [0.0, 1.0, 1.0, 1.0],
        );

        let right_id = Id::new("right")?;
        let right_data_id = doc
            .nodes
            .get(&right_id)
            .and_then(|node| node.data.clone())
            .ok_or("FBX test right mesh data is missing")?;
        let right_mesh = doc
            .data_blocks
            .get_mut(&right_data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or("FBX test right mesh is missing")?;
        let mut right_uv = uv_layers::read(right_mesh);
        right_uv.layers.retain(|layer| layer.name != "Detail");
        uv_layers::write(right_mesh, &right_uv);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        assert!(uv_texture_losses(&doc, &snapshot).is_empty());
        let warnings = uv_texture_warnings(&doc, &snapshot);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].details["feature_id"], "uv.layer_missing");
        assert_eq!(warnings[0].details["status"], "fallback");
        assert_eq!(warnings[0].details["node_id"], "right");
        assert_eq!(warnings[0].details["render_uv_layer"], "RightRender");
        let bytes = export_with_root_allow_lossy(&doc, &snapshot, root.path(), false)?;
        let imported = import_bytes(&bytes, "00000000-0000-4000-8000-000000000003".to_owned())?;
        assert_eq!(
            texture_on_node(&imported.doc, "left")?.uv_layer.as_deref(),
            Some("Detail")
        );
        assert_eq!(
            texture_on_node(&imported.doc, "right")?.uv_layer.as_deref(),
            Some("RightRender")
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "left", root.path())?,
            [0.0, 1.0, 0.0, 1.0],
        );
        assert_pixel(
            sample_node_texture(&imported.doc, "right", root.path())?,
            [0.0, 0.0, 1.0, 1.0],
        );
        Ok(())
    }
}
