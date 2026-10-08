use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use super::{ExchangeMesh, ImportedGraph, evaluated_meshes, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{Snapshot, animation::animated_transform},
    geom::Mesh,
    model::{
        Action, ArmatureData, CameraData, CameraProjection, DataBlock, FCurve, Id, Interpolation,
        Keyframe, LightData, LightType, Material, Modifier, Node, PoseBone, SceneDoc, ShapeKeyData,
        Transform, VertexGroup,
    },
};
use glam::{DMat4, DQuat, DVec3, EulerRot};
use serde_json::{Value, json};

const USDA_HEADER: &str = "#usda 1.0";

pub(crate) fn export_usda(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<u8>> {
    export_usda_with_root(doc, snapshot, Path::new("."))
}

pub(crate) fn export_usda_with_root(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    _project_root: &Path,
) -> Result<Vec<u8>> {
    if doc
        .scenes
        .get(&snapshot.scene_id)
        .is_some_and(|scene| scene.rigid_body_world.is_some())
    {
        return Err(unsupported(
            "physics.rigid_body_world",
            "USD export does not represent the scene rigid-body world".to_owned(),
        ));
    }
    let meshes = evaluated_meshes(doc, snapshot)?;
    let evaluated = meshes
        .into_iter()
        .map(|mesh| (mesh.id.clone(), mesh))
        .collect::<BTreeMap<_, _>>();
    let mut children = BTreeMap::<Option<Id>, Vec<Id>>::new();
    for (id, node) in &doc.nodes {
        if let Some(parent) = &node.parent
            && !doc.nodes.contains_key(parent)
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "USD node parent is missing",
            ));
        }
        children
            .entry(node.parent.clone())
            .or_default()
            .push(id.clone());
    }
    let mut text = String::from(USDA_HEADER);
    text.push_str("\n(\n    defaultPrim = \"Scene\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n");
    if let Some(scene) = doc.scenes.get(&snapshot.scene_id) {
        let rate = f64::from(scene.fps) / scene.fps_base;
        if rate.is_finite() && rate > 0.0 {
            line(
                &mut text,
                &format!("    timeCodesPerSecond = {}", number(rate)?),
            );
        }
        line(
            &mut text,
            &format!("    startTimeCode = {}", scene.frame_start),
        );
        line(&mut text, &format!("    endTimeCode = {}", scene.frame_end));
    }
    text.push_str(")\n\n");
    let scene_root_kind = if doc.nodes.values().any(|node| node.kind == "armature") {
        "SkelRoot"
    } else {
        "Xform"
    };
    text.push_str("def ");
    text.push_str(scene_root_kind);
    text.push_str(" \"Scene\"\n{\n");
    line(
        &mut text,
        &format!(
            "    custom string potter:scene_id = {}",
            quote(&snapshot.scene_id.to_string())?
        ),
    );
    line(
        &mut text,
        &format!(
            "    custom string potter:hash = {}",
            quote(&snapshot.scene_hash)?
        ),
    );
    line(
        &mut text,
        &format!("    custom int potter:revision = {}", snapshot.revision),
    );
    line(
        &mut text,
        &format!(
            "    custom double potter:frame = {}",
            number(snapshot.frame)?
        ),
    );
    if let Some(scene) = doc.scenes.get(&snapshot.scene_id) {
        line(
            &mut text,
            &format!("    custom int potter:fps = {}", scene.fps),
        );
        line(
            &mut text,
            &format!(
                "    custom double potter:fps_base = {}",
                number(scene.fps_base)?
            ),
        );
    }
    let collections_json = serde_json::to_string(&doc.collections).map_err(|error| {
        PotError::with_details(
            ErrorCode::ExportFailed,
            "could not encode USD collection metadata",
            json!({"reason":error.to_string()}),
        )
    })?;
    line(
        &mut text,
        &format!(
            "    custom string potter:collections = {}",
            quote(&collections_json)?
        ),
    );
    text.push_str("    double3 xformOp:translate = (0, 0, 0)\n    quatd xformOp:orient = (1, 0, 0, 0)\n    double3 xformOp:scale = (1, 1, 1)\n    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]\n");
    if let Some(roots) = children.get(&None) {
        for id in roots {
            write_node(&mut text, doc, snapshot, id, &children, &evaluated, 1)?;
        }
    }
    write_collection_sources(&mut text, doc, 2)?;
    text.push_str("    def Scope \"Materials\"\n    {\n");
    for (id, material) in &doc.materials {
        write_material(&mut text, id, material, 2)?;
    }
    text.push_str("    }\n}");
    Ok(text.into_bytes())
}

pub(crate) fn export_usdz(doc: &SceneDoc, snapshot: &Snapshot, root_name: &str) -> Result<Vec<u8>> {
    export_usdz_with_root(doc, snapshot, root_name, Path::new("."))
}

pub(crate) fn export_usdz_with_root(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    root_name: &str,
    project_root: &Path,
) -> Result<Vec<u8>> {
    let layer = export_usda_with_root(doc, snapshot, project_root)?;
    let filename = root_layer_name(root_name);
    make_stored_zip(&filename, &layer)
}

pub(crate) fn import_usda(path: &Path, scene_id: String) -> Result<ImportedGraph> {
    let bytes = fs::read(path).map_err(|error| PotError::io(&error))?;
    let text = std::str::from_utf8(&bytes).map_err(|_| import_error("USDA layer is not UTF-8"))?;
    let doc = parse_usda(text, scene_id)?;
    Ok(ImportedGraph {
        doc,
        losses: Vec::new(),
        id_mappings: json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source: json!({ "format": "usda", "path": path.display().to_string() }),
    })
}

pub(crate) fn import_usdz(path: &Path, scene_id: String) -> Result<ImportedGraph> {
    let archive = fs::read(path).map_err(|error| PotError::io(&error))?;
    let (name, layer) = read_usdz_root(&archive)?;
    let text =
        std::str::from_utf8(layer).map_err(|_| import_error("USDZ root layer is not UTF-8"))?;
    let doc = parse_usda(text, scene_id)?;
    Ok(ImportedGraph {
        doc,
        losses: Vec::new(),
        id_mappings: json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source: json!({ "format": "usdz", "path": path.display().to_string(), "root_layer": name }),
    })
}

fn write_node(
    output: &mut String,
    doc: &SceneDoc,
    snapshot: &Snapshot,
    id: &Id,
    children: &BTreeMap<Option<Id>, Vec<Id>>,
    evaluated: &BTreeMap<String, ExchangeMesh>,
    depth: usize,
) -> Result<()> {
    let node = doc.nodes.get(id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "USD node disappeared during export",
        )
    })?;
    if node.rigid_body.is_some() {
        return Err(unsupported(
            "physics.rigid_body",
            format!("USD export does not represent rigid body `{id}`"),
        ));
    }
    if node.force_field.is_some() {
        return Err(unsupported(
            "physics.force_field",
            format!("USD export does not represent force field `{id}`"),
        ));
    }
    let prim_name = format!("N_{}", prim_identifier(id.as_str()));
    let instance_target = node
        .properties
        .get("instance_collection")
        .and_then(serde_json::Value::as_str);
    if let Some(target) = instance_target {
        let target_id = Id::new(target).map_err(|_| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "USD collection instance target is invalid",
            )
        })?;
        if !doc.collections.contains_key(&target_id) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "USD collection instance target does not exist",
            ));
        }
        indent(output, depth);
        line(
            output,
            &format!(
                "def Xform {} (\n{}references = </Scene/CollectionSources/C_{}>\n{}instanceable = true\n{})",
                quote(&prim_name)?,
                spaces(depth + 1),
                prim_identifier(target),
                spaces(depth + 1),
                spaces(depth),
            ),
        );
    } else {
        indent(output, depth);
        line(output, &format!("def Xform {}", quote(&prim_name)?));
    }
    indent(output, depth);
    line(output, "{");
    let child_depth = depth + 1;
    let prop = |key: &str, value: String| {
        format!(
            "{}custom string potter:{} = {}",
            spaces(child_depth),
            key,
            value
        )
    };
    line(output, &prop("id", quote(id.as_str())?));
    line(output, &prop("name", quote(&node.name)?));
    line(output, &prop("kind", quote(&node.kind)?));
    line(
        output,
        &format!(
            "{}custom string[] potter:tags = {}",
            spaces(child_depth),
            string_array(&node.tags)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom string potter:hash = {}",
            spaces(child_depth),
            quote(&snapshot.scene_hash)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom int potter:revision = {}",
            spaces(child_depth),
            snapshot.revision
        ),
    );
    line(
        output,
        &format!(
            "{}custom bool potter:visible = {}",
            spaces(child_depth),
            node.visible
        ),
    );
    line(
        output,
        &format!(
            "{}custom bool potter:renderVisible = {}",
            spaces(child_depth),
            node.render_visible
        ),
    );
    line(
        output,
        &format!(
            "{}custom bool potter:selectable = {}",
            spaces(child_depth),
            node.selectable
        ),
    );
    line(
        output,
        &format!(
            "{}custom string[] potter:materials = {}",
            spaces(child_depth),
            string_array(
                &node
                    .materials
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            )?
        ),
    );
    if let Some(target) = instance_target {
        line(output, &prop("instanceCollection", quote(target)?));
    }
    if !node.modifiers.is_empty() {
        let encoded = serde_json::to_string(&node.modifiers).map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "could not encode USD modifier metadata",
                json!({"node_id":id,"reason":error.to_string()}),
            )
        })?;
        line(output, &prop("modifiers", quote(&encoded)?));
    }
    if !node.pose.is_empty() {
        let encoded = serde_json::to_string(&node.pose).map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "could not encode USD armature pose metadata",
                json!({"node_id":id,"reason":error.to_string()}),
            )
        })?;
        line(output, &prop("pose", quote(&encoded)?));
    }
    if let Some(parent_inverse) = node.parent_inverse {
        let values = parent_inverse
            .iter()
            .map(|value| number(*value))
            .collect::<Result<Vec<_>>>()?;
        line(
            output,
            &format!(
                "{}custom double[] potter:parentInverse = [{}]",
                spaces(child_depth),
                values.join(", ")
            ),
        );
    }
    line(
        output,
        &format!(
            "{}double3 xformOp:translate = {}",
            spaces(child_depth),
            vec3(node.transform.translation)?
        ),
    );
    let [qx, qy, qz, qw] = node.transform.rotation;
    line(
        output,
        &format!(
            "{}quatd xformOp:orient = ({}, {}, {}, {})",
            spaces(child_depth),
            number(qw)?,
            number(qx)?,
            number(qy)?,
            number(qz)?
        ),
    );
    line(
        output,
        &format!(
            "{}double3 xformOp:scale = {}",
            spaces(child_depth),
            vec3(node.transform.scale)?
        ),
    );
    if let Some(parent_inverse) = node.parent_inverse {
        let matrix = DMat4::from_cols_array(&parent_inverse) * node.transform.matrix();
        line(
            output,
            &format!(
                "{}matrix4d xformOp:transform = {}",
                spaces(child_depth),
                matrix4d(matrix)?
            ),
        );
        line(
            output,
            &format!(
                "{}uniform token[] xformOpOrder = [\"xformOp:transform\"]",
                spaces(child_depth)
            ),
        );
    } else {
        line(
            output,
            &format!(
                "{}uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:orient\", \"xformOp:scale\"]",
                spaces(child_depth)
            ),
        );
    }
    write_animation(output, doc, node, child_depth)?;
    if let Some(action_id) = &node.action {
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "USD node references a missing Action",
            )
        })?;
        let encoded = serde_json::to_string(action).map_err(|error| {
            PotError::with_details(
                ErrorCode::ExportFailed,
                "could not encode USD animation metadata",
                json!({"reason": error.to_string()}),
            )
        })?;
        line(
            output,
            &format!(
                "{}custom string potter:actionId = {}",
                spaces(child_depth),
                quote(action_id.as_str())?
            ),
        );
        line(
            output,
            &format!(
                "{}custom string potter:action = {}",
                spaces(child_depth),
                quote(&encoded)?
            ),
        );
    }
    if let Some(exchange) = evaluated.get(id.as_str()) {
        let evaluated_mesh = snapshot.meshes.get(id).ok_or_else(|| {
            PotError::new(ErrorCode::EvaluationFailed, "USD evaluated mesh is missing")
        })?;
        let data_block = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        let preserves_source_features = data_block.is_some_and(|data| {
            data.shape_keys
                .as_ref()
                .is_some_and(|keys| !keys.keys.is_empty())
                || !data.vertex_groups.is_empty()
                || !data.vertex_weights.is_empty()
                || node
                    .modifiers
                    .iter()
                    .any(|modifier| modifier.enabled && modifier.modifier_type == "armature")
        });
        if preserves_source_features
            && node
                .modifiers
                .iter()
                .any(|modifier| modifier.enabled && modifier.modifier_type != "armature")
        {
            return Err(unsupported(
                "mesh.rig_modifier_stack",
                format!("USD cannot preserve rig data through an enabled modifier stack on `{id}`"),
            ));
        }
        let positions = if preserves_source_features {
            data_block
                .and_then(|data| data.mesh.as_ref())
                .ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "USD rig mesh source is missing")
                })?
                .vertices
                .iter()
                .map(|vertex| vertex.co.to_array())
                .collect::<Vec<_>>()
        } else {
            evaluated_mesh
                .vertices
                .iter()
                .map(|vertex| vertex.co.to_array())
                .collect::<Vec<_>>()
        };
        write_mesh(
            output,
            doc,
            snapshot,
            node,
            exchange,
            &positions,
            child_depth,
        )?;
    }
    if let Some(data_id) = &node.data
        && let Some(data) = doc.data_blocks.get(data_id)
    {
        if let Some(armature) = &data.armature {
            write_skeleton(output, armature, child_depth)?;
        }
        if let Some(camera) = &data.camera {
            write_camera(output, camera, child_depth)?;
        }
        if let Some(light) = &data.light {
            write_light(output, light, child_depth)?;
        }
    }
    if let Some(nested) = children.get(&Some(id.clone())) {
        for nested_id in nested {
            write_node(
                output,
                doc,
                snapshot,
                nested_id,
                children,
                evaluated,
                child_depth,
            )?;
        }
    }
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn write_mesh(
    output: &mut String,
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node: &Node,
    exchange: &ExchangeMesh,
    positions: &[[f64; 3]],
    depth: usize,
) -> Result<()> {
    if positions.len() != exchange.positions.len() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "USD mesh vertex count changed during export",
        ));
    }
    let has_skel_binding = node
        .modifiers
        .iter()
        .any(|modifier| modifier.enabled && modifier.modifier_type == "armature")
        || node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|data| data.shape_keys.as_ref())
            .is_some_and(|shape_keys| !shape_keys.keys.is_empty());
    if has_skel_binding {
        indent(output, depth);
        line(output, "def Mesh \"Geometry\" (");
        indent(output, depth + 1);
        line(output, "prepend apiSchemas = [\"SkelBindingAPI\"]");
        indent(output, depth);
        line(output, ")");
    } else {
        indent(output, depth);
        line(output, "def Mesh \"Geometry\"");
    }
    indent(output, depth);
    line(output, "{");
    let d = depth + 1;
    let points = positions
        .iter()
        .map(|value| vec3(*value))
        .collect::<Result<Vec<_>>>()?;
    line(
        output,
        &format!("{}point3f[] points = [{}]", spaces(d), points.join(", ")),
    );
    let counts = exchange
        .faces
        .iter()
        .map(|face| face.len().to_string())
        .collect::<Vec<_>>();
    let flat_indices = exchange
        .faces
        .iter()
        .flatten()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if flat_indices
        .iter()
        .any(|index| index.parse::<usize>().is_err())
    {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "USD mesh index is invalid",
        ));
    }
    line(
        output,
        &format!(
            "{}int[] faceVertexCounts = [{}]",
            spaces(d),
            counts.join(", ")
        ),
    );
    line(
        output,
        &format!(
            "{}int[] faceVertexIndices = [{}]",
            spaces(d),
            flat_indices.join(", ")
        ),
    );
    let normals = exchange
        .faces
        .iter()
        .map(|face| face_normal(face, positions))
        .collect::<Result<Vec<_>>>()?;
    let normals = normals
        .iter()
        .map(|normal| vec3(*normal))
        .collect::<Result<Vec<_>>>()?;
    line(
        output,
        &format!(
            "{}normal3f[] normals = [{}] (interpolation = \"uniform\")",
            spaces(d),
            normals.join(", ")
        ),
    );
    let uv = generated_uvs(positions);
    let uv = uv
        .iter()
        .map(|value| -> Result<String> {
            Ok(format!("({}, {})", number(value[0])?, number(value[1])?))
        })
        .collect::<Result<Vec<_>>>()?;
    line(
        output,
        &format!(
            "{}texCoord2f[] primvars:st = [{}] (interpolation = \"vertex\")",
            spaces(d),
            uv.join(", ")
        ),
    );
    line(
        output,
        &format!(
            "{}custom string potter:id = {}",
            spaces(d),
            quote(&exchange.id)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom string potter:name = {}",
            spaces(d),
            quote(&exchange.name)?
        ),
    );
    write_deformation(output, doc, snapshot, node, &exchange.id, d)?;
    if let Some(material_id) = node.materials.first() {
        if !doc.materials.contains_key(material_id) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "USD mesh references a missing material",
            ));
        }
        line(
            output,
            &format!(
                "{}rel material:binding = </Scene/Materials/M_{}>",
                spaces(d),
                prim_identifier(material_id.as_str())
            ),
        );
    }
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn write_deformation(
    output: &mut String,
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node: &Node,
    node_id: &str,
    depth: usize,
) -> Result<()> {
    let Some(data_id) = &node.data else {
        return Ok(());
    };
    let Some(data) = doc.data_blocks.get(data_id) else {
        return Ok(());
    };
    let Some(mesh) = data.mesh.as_ref() else {
        return Ok(());
    };
    let has_shapes = data
        .shape_keys
        .as_ref()
        .is_some_and(|shape_keys| !shape_keys.keys.is_empty());
    let has_skin = node
        .modifiers
        .iter()
        .any(|modifier| modifier.enabled && modifier.modifier_type == "armature");
    if !has_shapes && !has_skin && data.vertex_groups.is_empty() && data.vertex_weights.is_empty() {
        return Ok(());
    }
    let encoded = serde_json::to_string(&json!({
        "shape_keys": &data.shape_keys,
        "vertex_groups": &data.vertex_groups,
        "vertex_weights": &data.vertex_weights,
        "vertex_ids": mesh.vertices.iter().map(|vertex| vertex.id).collect::<Vec<_>>(),
    }))
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ExportFailed,
            "could not encode USD rig and shape-key data",
            json!({"node_id":node_id,"reason":error.to_string()}),
        )
    })?;
    line(
        output,
        &format!(
            "{}custom string potter:deformation = {}",
            spaces(depth),
            quote(&encoded)?
        ),
    );
    if has_skin {
        write_skin_binding(output, doc, snapshot, node, node_id, data, depth)?;
    }
    if has_shapes {
        write_shape_targets(output, doc, node_id, mesh, data, depth)?;
    }
    Ok(())
}

fn write_skin_binding(
    output: &mut String,
    doc: &SceneDoc,
    snapshot: &Snapshot,
    node: &Node,
    node_id: &str,
    data: &DataBlock,
    depth: usize,
) -> Result<()> {
    let active = node
        .modifiers
        .iter()
        .filter(|modifier| modifier.enabled)
        .collect::<Vec<_>>();
    let skin_modifiers = active
        .iter()
        .filter(|modifier| modifier.modifier_type == "armature")
        .copied()
        .collect::<Vec<_>>();
    if skin_modifiers.len() != 1
        || active
            .iter()
            .any(|modifier| modifier.modifier_type != "armature")
    {
        return Err(unsupported(
            "rig.skin_modifier_stack",
            "USD skin export requires exactly one enabled armature modifier and no other enabled modifiers".to_owned(),
        ));
    }
    let raw_target = skin_modifiers[0]
        .params
        .get("object")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "USD armature modifier has no target",
            )
        })?;
    let armature_id = Id::new(raw_target)
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "USD armature target ID is invalid"))?;
    let armature_node = doc.nodes.get(&armature_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "USD armature modifier target does not exist",
        )
    })?;
    let armature = armature_node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.armature.as_ref())
        .ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "USD armature hierarchy is missing")
        })?;
    let mesh = data.mesh.as_ref().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "USD skin Data-Block has no source mesh",
        )
    })?;
    let joint_ids = ordered_bone_ids(armature)?;
    let joint_indices = joint_ids
        .iter()
        .enumerate()
        .map(|(index, bone_id)| (bone_id.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let mut group_bones = BTreeMap::<Id, usize>::new();
    for group in &data.vertex_groups {
        let matches = armature
            .bones
            .iter()
            .filter(|(_, bone)| bone.deform && bone.name == group.name)
            .collect::<Vec<_>>();
        if matches.len() > 1 {
            return Err(unsupported(
                "rig.skin_duplicate_group",
                format!(
                    "USD vertex group `{}` maps ambiguously to armature joints",
                    group.name
                ),
            ));
        }
        if let Some((bone_id, _)) = matches.first()
            && let Some(index) = joint_indices.get(*bone_id)
        {
            group_bones.insert(group.id.clone(), *index);
        }
    }
    let group_ids = data
        .vertex_groups
        .iter()
        .map(|group| group.id.clone())
        .collect::<BTreeSet<_>>();
    let mut all_weights = Vec::<Vec<(usize, f64)>>::with_capacity(mesh.vertices.len());
    let mut element_size = 1_usize;
    for vertex in &mesh.vertices {
        let mut weights_for_vertex = Vec::new();
        if let Some(weights) = data.vertex_weights.get(&vertex.id) {
            for (group_id, weight) in weights {
                if !group_ids.contains(group_id) || !weight.is_finite() || *weight < 0.0 {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "USD skin weights must be finite, nonnegative, and reference known groups",
                    ));
                }
                if *weight > 0.0
                    && let Some(joint_index) = group_bones.get(group_id)
                {
                    weights_for_vertex.push((*joint_index, *weight));
                }
            }
        }
        weights_for_vertex.sort_by_key(|(joint_index, _)| *joint_index);
        element_size = element_size.max(weights_for_vertex.len());
        all_weights.push(weights_for_vertex);
    }
    let mut joint_values = Vec::new();
    let mut weight_values = Vec::new();
    for vertex_weights in all_weights {
        for slot in 0..element_size {
            let (joint_index, weight) = vertex_weights.get(slot).copied().unwrap_or((0, 0.0));
            joint_values.push(joint_index.to_string());
            weight_values.push(number(weight)?);
        }
    }
    let skeleton_path = format!("{}/Skeleton", node_prim_path(doc, &armature_id)?);
    line(
        output,
        &format!("{}rel skel:skeleton = <{}>", spaces(depth), skeleton_path),
    );
    line(
        output,
        &format!(
            "{}int[] primvars:skel:jointIndices = [{}] (",
            spaces(depth),
            joint_values.join(", ")
        ),
    );
    line(
        output,
        &format!("{}elementSize = {}", spaces(depth + 1), element_size),
    );
    line(
        output,
        &format!("{}interpolation = \"vertex\"", spaces(depth + 1)),
    );
    line(output, &format!("{})", spaces(depth)));
    line(
        output,
        &format!(
            "{}float[] primvars:skel:jointWeights = [{}] (",
            spaces(depth),
            weight_values.join(", ")
        ),
    );
    line(
        output,
        &format!("{}elementSize = {}", spaces(depth + 1), element_size),
    );
    line(
        output,
        &format!("{}interpolation = \"vertex\"", spaces(depth + 1)),
    );
    line(output, &format!("{})", spaces(depth)));
    let armature_world = snapshot
        .nodes
        .get(&armature_id)
        .map(|state| DMat4::from_cols_array(&state.world_matrix))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "USD armature world matrix is missing",
            )
        })?;
    let mesh_id = Id::new(node_id)
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "USD mesh node ID is invalid"))?;
    let mesh_world = snapshot
        .nodes
        .get(&mesh_id)
        .map(|state| DMat4::from_cols_array(&state.world_matrix))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "USD mesh world matrix is missing",
            )
        })?;
    line(
        output,
        &format!(
            "{}matrix4d primvars:skel:geomBindTransform = {}",
            spaces(depth),
            matrix4d(armature_world.inverse() * mesh_world)?
        ),
    );
    Ok(())
}

fn write_shape_targets(
    output: &mut String,
    doc: &SceneDoc,
    node_id: &str,
    mesh: &Mesh,
    data: &DataBlock,
    depth: usize,
) -> Result<()> {
    let Some(shape_keys) = data.shape_keys.as_ref() else {
        return Ok(());
    };
    let node_id = Id::new(node_id)
        .map_err(|_| PotError::new(ErrorCode::SceneInvalid, "USD shape-key node ID is invalid"))?;
    let mesh_path = node_prim_path(doc, &node_id)?;
    let mut names = Vec::new();
    let mut targets = Vec::new();
    let mut values = Vec::new();
    for (key_id, key) in &shape_keys.keys {
        if key.relative_key.is_some() {
            return Err(unsupported(
                "mesh.shape_key_relative",
                format!("USD BlendShape `{}` has a non-basis relative key", key.name),
            ));
        }
        let name = format!("Blend_{}", prim_identifier(key_id.as_str()));
        names.push(format!("\"{name}\""));
        targets.push(format!("<{mesh_path}/Geometry/{name}>"));
        values.push(number(key.value)?);
        let mut indices = Vec::new();
        let mut offsets = Vec::new();
        for (index, vertex) in mesh.vertices.iter().enumerate() {
            let basis = shape_keys
                .basis
                .get(&vertex.id)
                .copied()
                .unwrap_or(vertex.co.to_array());
            let target = key.positions.get(&vertex.id).copied().unwrap_or(basis);
            let mut offset = DVec3::from_array(target) - DVec3::from_array(basis);
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
                        "USD shape-key vertex-group weight is invalid",
                    ));
                }
                offset *= weight;
            }
            if !offset.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "USD shape-key offset is not finite",
                ));
            }
            if offset.length_squared() > 0.0 {
                indices.push(index.to_string());
                offsets.push(vec3(offset.to_array())?);
            }
        }
        indent(output, depth);
        line(output, &format!("def BlendShape {}", quote(&name)?));
        indent(output, depth);
        line(output, "{");
        line(
            output,
            &format!(
                "{}custom string potter:id = {}",
                spaces(depth + 1),
                quote(key_id.as_str())?
            ),
        );
        line(
            output,
            &format!(
                "{}custom double potter:value = {}",
                spaces(depth + 1),
                number(key.value)?
            ),
        );
        line(
            output,
            &format!(
                "{}int[] pointIndices = [{}]",
                spaces(depth + 1),
                indices.join(", ")
            ),
        );
        line(
            output,
            &format!(
                "{}vector3f[] offsets = [{}]",
                spaces(depth + 1),
                offsets.join(", ")
            ),
        );
        indent(output, depth);
        line(output, "}");
    }
    if !names.is_empty() {
        line(
            output,
            &format!(
                "{}token[] skel:blendShapes = [{}]",
                spaces(depth),
                names.join(", ")
            ),
        );
        line(
            output,
            &format!(
                "{}float[] skel:blendShapeWeights = [{}]",
                spaces(depth),
                values.join(", ")
            ),
        );
        line(
            output,
            &format!(
                "{}rel skel:blendShapeTargets = [{}]",
                spaces(depth),
                targets.join(", ")
            ),
        );
    }
    Ok(())
}

fn ordered_bone_ids(armature: &ArmatureData) -> Result<Vec<Id>> {
    fn visit(bone_id: &Id, children: &BTreeMap<Id, Vec<Id>>, ordered: &mut Vec<Id>) {
        ordered.push(bone_id.clone());
        if let Some(child_ids) = children.get(bone_id) {
            for child_id in child_ids {
                visit(child_id, children, ordered);
            }
        }
    }
    let mut children = BTreeMap::<Id, Vec<Id>>::new();
    let mut roots = Vec::new();
    for (bone_id, bone) in &armature.bones {
        if let Some(parent_id) = &bone.parent {
            if !armature.bones.contains_key(parent_id) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "USD armature parent bone is missing",
                ));
            }
            children
                .entry(parent_id.clone())
                .or_default()
                .push(bone_id.clone());
        } else {
            roots.push(bone_id.clone());
        }
    }
    let mut ordered = Vec::with_capacity(armature.bones.len());
    for root_id in roots {
        visit(&root_id, &children, &mut ordered);
    }
    if ordered.len() != armature.bones.len() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "USD armature contains a bone cycle",
        ));
    }
    Ok(ordered)
}

fn write_skeleton(output: &mut String, armature: &ArmatureData, depth: usize) -> Result<()> {
    let matrices = crate::eval::rig::evaluate_bone_matrices(armature, &BTreeMap::new())?;
    let joint_ids = ordered_bone_ids(armature)?;
    let mut paths = BTreeMap::<Id, String>::new();
    for bone_id in &joint_ids {
        joint_path(armature, bone_id, &mut paths, &mut BTreeSet::new())?;
    }
    let joints = joint_ids
        .iter()
        .map(|bone_id| {
            paths
                .get(bone_id)
                .map(|path| format!("\"{path}\""))
                .ok_or_else(|| PotError::new(ErrorCode::InternalError, "USD joint path is missing"))
        })
        .collect::<Result<Vec<_>>>()?;
    let rest = joint_ids
        .iter()
        .map(|bone_id| {
            let bone = armature
                .bones
                .get(bone_id)
                .ok_or_else(|| PotError::new(ErrorCode::InternalError, "USD bone disappeared"))?;
            let rest = matrices
                .get(bone_id)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "USD bone rest matrix is missing",
                    )
                })?
                .rest;
            let local = if let Some(parent_id) = &bone.parent {
                let parent = matrices
                    .get(parent_id)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "USD parent bone is missing")
                    })?
                    .rest;
                parent.inverse() * rest
            } else {
                rest
            };
            matrix4d(local)
        })
        .collect::<Result<Vec<_>>>()?;
    let bind = joint_ids
        .iter()
        .map(|bone_id| {
            let matrix = matrices.get(bone_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "USD bone bind transform is missing",
                )
            })?;
            matrix4d(matrix.rest.inverse())
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = serde_json::to_string(armature).map_err(|error| {
        PotError::with_details(
            ErrorCode::ExportFailed,
            "could not encode USD armature hierarchy",
            json!({"reason":error.to_string()}),
        )
    })?;
    indent(output, depth);
    line(output, "def Skeleton \"Skeleton\"");
    indent(output, depth);
    line(output, "{");
    let skeleton_depth = depth + 1;
    line(
        output,
        &format!(
            "{}uniform token[] joints = [{}]",
            spaces(skeleton_depth),
            joints.join(", ")
        ),
    );
    line(
        output,
        &format!(
            "{}matrix4d[] restTransforms = [{}]",
            spaces(skeleton_depth),
            rest.join(", ")
        ),
    );
    line(
        output,
        &format!(
            "{}matrix4d[] bindTransforms = [{}]",
            spaces(skeleton_depth),
            bind.join(", ")
        ),
    );
    line(
        output,
        &format!(
            "{}custom string potter:armature = {}",
            spaces(skeleton_depth),
            quote(&metadata)?
        ),
    );
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn joint_path(
    armature: &ArmatureData,
    bone_id: &Id,
    paths: &mut BTreeMap<Id, String>,
    active: &mut BTreeSet<Id>,
) -> Result<String> {
    if let Some(path) = paths.get(bone_id) {
        return Ok(path.clone());
    }
    if !active.insert(bone_id.clone()) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "USD armature contains a bone cycle",
        ));
    }
    let bone = armature
        .bones
        .get(bone_id)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "USD armature bone is missing"))?;
    let local = format!("J_{}", prim_identifier(bone_id.as_str()));
    let path = if let Some(parent_id) = &bone.parent {
        format!(
            "{}/{}",
            joint_path(armature, parent_id, paths, active)?,
            local
        )
    } else {
        local
    };
    active.remove(bone_id);
    paths.insert(bone_id.clone(), path.clone());
    Ok(path)
}

fn write_collection_sources(output: &mut String, doc: &SceneDoc, depth: usize) -> Result<()> {
    let root_collection = doc
        .scenes
        .get(&doc.active_scene)
        .map(|scene| scene.root_collection.clone())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "USD active scene has no root collection",
            )
        })?;
    indent(output, depth);
    line(output, "def Scope \"CollectionSources\"");
    indent(output, depth);
    line(output, "{");
    for (collection_id, collection) in &doc.collections {
        if collection_id == &root_collection {
            continue;
        }
        let source_depth = depth + 1;
        let source_name = format!("C_{}", prim_identifier(collection_id.as_str()));
        indent(output, source_depth);
        line(output, &format!("def Xform {}", quote(&source_name)?));
        indent(output, source_depth);
        line(output, "{");
        let nested_depth = source_depth + 1;
        line(
            output,
            &format!(
                "{}custom string potter:collectionId = {}",
                spaces(nested_depth),
                quote(collection_id.as_str())?
            ),
        );
        for object_id in &collection.objects {
            if !doc.nodes.contains_key(object_id) {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "USD collection references a missing object",
                    json!({"collection_id":collection_id,"node_id":object_id}),
                ));
            }
            let path = node_prim_path(doc, object_id)?;
            let alias = format!("Object_{}", prim_identifier(object_id.as_str()));
            indent(output, nested_depth);
            line(
                output,
                &format!(
                    "def Xform {} (references = <{}> instanceable = true)",
                    quote(&alias)?,
                    path
                ),
            );
            indent(output, nested_depth);
            line(output, "{");
            indent(output, nested_depth);
            line(output, "}");
        }
        for child_id in &collection.children {
            if !doc.collections.contains_key(child_id) {
                return Err(PotError::with_details(
                    ErrorCode::SceneInvalid,
                    "USD collection has a missing child collection",
                    json!({"collection_id":collection_id,"child_id":child_id}),
                ));
            }
            let alias = format!("Collection_{}", prim_identifier(child_id.as_str()));
            indent(output, nested_depth);
            line(
                output,
                &format!(
                    "def Xform {} (references = </Scene/CollectionSources/C_{}> instanceable = true)",
                    quote(&alias)?,
                    prim_identifier(child_id.as_str())
                ),
            );
            indent(output, nested_depth);
            line(output, "{");
            indent(output, nested_depth);
            line(output, "}");
        }
        indent(output, source_depth);
        line(output, "}");
    }
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn node_prim_path(doc: &SceneDoc, node_id: &Id) -> Result<String> {
    let mut chain = Vec::<Id>::new();
    let mut cursor = Some(node_id.clone());
    let mut seen = BTreeSet::new();
    while let Some(id) = cursor.take() {
        if !seen.insert(id.clone()) {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "USD node parent cycle detected",
            ));
        }
        let node = doc.nodes.get(&id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "USD collection node is missing")
        })?;
        chain.push(id);
        cursor.clone_from(&node.parent);
    }
    chain.reverse();
    let mut path = String::from("/Scene");
    for id in chain {
        path.push_str("/N_");
        path.push_str(&prim_identifier(id.as_str()));
    }
    Ok(path)
}

fn write_material(output: &mut String, id: &Id, material: &Material, depth: usize) -> Result<()> {
    let name = format!("M_{}", prim_identifier(id.as_str()));
    indent(output, depth);
    line(output, &format!("def Material {}", quote(&name)?));
    indent(output, depth);
    line(output, "{");
    let d = depth + 1;
    line(
        output,
        &format!(
            "{}custom string potter:id = {}",
            spaces(d),
            quote(id.as_str())?
        ),
    );
    line(
        output,
        &format!(
            "{}custom string potter:name = {}",
            spaces(d),
            quote(&material.name)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom bool potter:doubleSided = {}",
            spaces(d),
            material.double_sided
        ),
    );
    line(
        output,
        &format!(
            "{}token outputs:surface.connect = </Scene/Materials/{name}/Preview.outputs:surface>",
            spaces(d)
        ),
    );
    indent(output, d);
    line(output, "def Shader \"Preview\"");
    indent(output, d);
    line(output, "{");
    let shader = d + 1;
    line(
        output,
        &format!(
            "{}uniform token info:id = \"UsdPreviewSurface\"",
            spaces(shader)
        ),
    );
    line(
        output,
        &format!(
            "{}color3f inputs:diffuseColor = ({}, {}, {})",
            spaces(shader),
            number(material.base_color[0])?,
            number(material.base_color[1])?,
            number(material.base_color[2])?
        ),
    );
    line(
        output,
        &format!(
            "{}float inputs:opacity = {}",
            spaces(shader),
            number(material.base_color[3])?
        ),
    );
    line(
        output,
        &format!(
            "{}float inputs:metallic = {}",
            spaces(shader),
            number(material.metallic)?
        ),
    );
    line(
        output,
        &format!(
            "{}float inputs:roughness = {}",
            spaces(shader),
            number(material.roughness)?
        ),
    );
    line(output, &format!("{}token outputs:surface", spaces(shader)));
    indent(output, d);
    line(output, "}");
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn write_camera(output: &mut String, camera: &CameraData, depth: usize) -> Result<()> {
    indent(output, depth);
    line(output, "def Camera \"Camera\"");
    indent(output, depth);
    line(output, "{");
    let d = depth + 1;
    let projection = if camera.projection == CameraProjection::Orthographic {
        "orthographic"
    } else {
        "perspective"
    };
    line(
        output,
        &format!("{}token projection = \"{projection}\"", spaces(d)),
    );
    line(
        output,
        &format!(
            "{}float focalLength = {}",
            spaces(d),
            number(camera.lens_mm)?
        ),
    );
    line(
        output,
        &format!(
            "{}float horizontalAperture = {}",
            spaces(d),
            number(camera.sensor_width_mm)?
        ),
    );
    line(
        output,
        &format!(
            "{}float2 clippingRange = ({}, {})",
            spaces(d),
            number(camera.clip_start)?,
            number(camera.clip_end)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom float potter:orthoScale = {}",
            spaces(d),
            number(camera.ortho_scale)?
        ),
    );
    line(
        output,
        &format!(
            "{}custom float2 potter:shift = ({}, {})",
            spaces(d),
            number(camera.shift[0])?,
            number(camera.shift[1])?
        ),
    );
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn write_light(output: &mut String, light: &LightData, depth: usize) -> Result<()> {
    let schema = match light.light_type {
        LightType::Sun => "DistantLight",
        LightType::Area => "DiskLight",
        LightType::Point | LightType::Spot => "SphereLight",
    };
    indent(output, depth);
    line(output, &format!("def {schema} \"Light\""));
    indent(output, depth);
    line(output, "{");
    let d = depth + 1;
    let kind = match light.light_type {
        LightType::Point => "point",
        LightType::Sun => "sun",
        LightType::Spot => "spot",
        LightType::Area => "area",
    };
    line(
        output,
        &format!("{}custom string potter:lightType = \"{kind}\"", spaces(d)),
    );
    line(
        output,
        &format!(
            "{}color3f inputs:color = ({}, {}, {})",
            spaces(d),
            number(light.color[0])?,
            number(light.color[1])?,
            number(light.color[2])?
        ),
    );
    line(
        output,
        &format!(
            "{}float inputs:intensity = {}",
            spaces(d),
            number(light.energy)?
        ),
    );
    line(
        output,
        &format!(
            "{}float inputs:radius = {}",
            spaces(d),
            number(light.radius)?
        ),
    );
    if light.light_type == LightType::Spot {
        line(
            output,
            &format!(
                "{}custom float potter:spotSize = {}",
                spaces(d),
                number(light.spot_size)?
            ),
        );
        line(
            output,
            &format!(
                "{}custom float potter:spotBlend = {}",
                spaces(d),
                number(light.spot_blend)?
            ),
        );
    }
    indent(output, depth);
    line(output, "}");
    Ok(())
}

fn write_animation(output: &mut String, doc: &SceneDoc, node: &Node, depth: usize) -> Result<()> {
    let Some(action_id) = &node.action else {
        return Ok(());
    };
    let action = doc.actions.get(action_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "USD node references a missing Action",
        )
    })?;
    let mut frames = BTreeSet::<u64>::new();
    let mut channels = BTreeSet::<&str>::new();
    for curve in &action.fcurves {
        let channel = match curve.path.as_str() {
            "transform.translation" => "translation",
            "transform.scale" => "scale",
            "transform.rotation" | "transform.rotation_quaternion" | "transform.rotation_euler" => {
                "rotation"
            }
            path => {
                return Err(unsupported(
                    "animation.fcurve",
                    format!("USD cannot represent animation path `{path}`"),
                ));
            }
        };
        let limit = if channel == "rotation" && curve.path != "transform.rotation_euler" {
            4
        } else {
            3
        };
        if curve.index >= limit {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "USD animation component index is out of range",
            ));
        }
        channels.insert(channel);
        for key in &curve.keyframes {
            if !key.frame.is_finite() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "USD animation frame is not finite",
                ));
            }
            frames.insert(key.frame.to_bits());
        }
    }
    let mut frames = frames.into_iter().map(f64::from_bits).collect::<Vec<_>>();
    frames.sort_by(f64::total_cmp);
    if frames.is_empty() {
        return Ok(());
    }
    for channel in channels {
        let mut samples = Vec::with_capacity(frames.len());
        for frame in &frames {
            let value = animated_transform(node, doc, *frame)?;
            let sample = match channel {
                "translation" => vec3(value.translation)?,
                "scale" => vec3(value.scale)?,
                _ => {
                    let [x, y, z, w] = value.rotation;
                    format!(
                        "({}, {}, {}, {})",
                        number(w)?,
                        number(x)?,
                        number(y)?,
                        number(z)?
                    )
                }
            };
            samples.push(format!("{}: {}", number(*frame)?, sample));
        }
        let (ty, prop) = match channel {
            "translation" => ("double3", "xformOp:translate"),
            "scale" => ("double3", "xformOp:scale"),
            _ => ("quatd", "xformOp:orient"),
        };
        line(
            output,
            &format!(
                "{}{} {}.timeSamples = {{ {} }}",
                spaces(depth),
                ty,
                prop,
                samples.join(", ")
            ),
        );
    }
    Ok(())
}

fn parse_usda(text: &str, scene_id: String) -> Result<SceneDoc> {
    if !text.starts_with(USDA_HEADER) {
        return Err(import_error("input is not a USDA 1.0 layer"));
    }
    let tokens = lex_usda(text)?;
    let roots = parse_prims(&tokens, 0, tokens.len())?;
    let root = roots
        .iter()
        .find(|prim| (prim.kind == "Xform" || prim.kind == "SkelRoot") && prim.name == "Scene");
    let mut doc = SceneDoc::new(scene_id);
    let mut used_nodes = BTreeSet::<Id>::new();
    let mut used_data = BTreeSet::<Id>::new();
    let mut used_materials = BTreeSet::<Id>::new();
    let mut used_actions = BTreeSet::<Id>::new();
    let mut materials_by_name = BTreeMap::<String, Id>::new();
    for material in roots
        .iter()
        .flat_map(|prim| std::iter::once(prim).chain(prim.descendants()))
        .filter(|prim| prim.kind == "Material")
    {
        let raw_id = property_string(material, "potter:id");
        let id = import_id(raw_id.as_deref(), &material.name, &mut used_materials)?;
        let shader = material
            .descendants()
            .into_iter()
            .find(|prim| prim.kind == "Shader");
        let mut color = [0.6, 0.6, 0.6, 1.0];
        let mut metallic = 0.0;
        let mut roughness = 0.8;
        if let Some(shader) = shader {
            if let Some(shader_id) = property_string(shader, "info:id")
                && shader_id != "UsdPreviewSurface"
            {
                return Err(unsupported(
                    "material.usd_shader",
                    format!("USD shader `{shader_id}` is unsupported"),
                ));
            }
            if shader
                .properties
                .keys()
                .any(|name| name.ends_with(".connect"))
            {
                return Err(unsupported(
                    "material.usd_shader_network",
                    "USD connected material inputs are not representable without texture support"
                        .to_owned(),
                ));
            }
            if let Some(values) = property_numbers(shader, "inputs:diffuseColor") {
                if values.len() < 3 {
                    return Err(import_error("USD material diffuseColor is invalid"));
                }
                color[..3].copy_from_slice(&values[..3]);
            }
            if let Some(value) = property_numbers(shader, "inputs:opacity")
                .and_then(|values| values.first().copied())
            {
                color[3] = value;
            }
            if let Some(value) = property_numbers(shader, "inputs:metallic")
                .and_then(|values| values.first().copied())
            {
                metallic = value;
            }
            if let Some(value) = property_numbers(shader, "inputs:roughness")
                .and_then(|values| values.first().copied())
            {
                roughness = value;
            }
        }
        let name =
            property_string(material, "potter:name").unwrap_or_else(|| material.name.clone());
        doc.materials.insert(
            id.clone(),
            Material {
                name,
                base_color: color,
                metallic,
                roughness,
                double_sided: property_bool(material, "potter:doubleSided").unwrap_or(false),
                ..Material::default()
            },
        );
        materials_by_name.insert(material.name.clone(), id.clone());
        materials_by_name.insert(format!("M_{}", prim_identifier(id.as_str())), id);
    }
    if let Some(root) = root {
        let source_hash = property_string(root, "potter:hash");
        let source_scene_id = property_string(root, "potter:scene_id");
        if source_hash.is_some() || source_scene_id.is_some() {
            doc.compatibility.insert(
                "usd".to_owned(),
                json!({
                    "scene_hash": source_hash,
                    "scene_id": source_scene_id,
                }),
            );
        }
        doc.revision = property_numbers(root, "potter:revision")
            .and_then(|items| items.first().copied())
            .map_or(0, |value| value.max(0.0) as u64);
        if doc.revision > crate::model::MAX_REVISION {
            return Err(import_error("USD revision exceeds the supported range"));
        }
        if let Some(frame) =
            property_numbers(root, "potter:frame").and_then(|items| items.first().copied())
            && let Some(scene) = doc.scenes.get_mut(&doc.active_scene)
        {
            scene.frame_current = frame;
        }
        if let Some(fps) =
            property_numbers(root, "potter:fps").and_then(|items| items.first().copied())
            && fps >= 1.0
            && fps <= f64::from(u32::MAX)
            && let Some(scene) = doc.scenes.get_mut(&doc.active_scene)
        {
            scene.fps = fps as u32;
        }
        if let Some(fps_base) =
            property_numbers(root, "potter:fps_base").and_then(|items| items.first().copied())
            && fps_base.is_finite()
            && fps_base > 0.0
            && let Some(scene) = doc.scenes.get_mut(&doc.active_scene)
        {
            scene.fps_base = fps_base;
        }
        for child in &root.children {
            import_node_tree(
                child,
                None,
                &mut doc,
                &mut used_nodes,
                &mut used_data,
                &mut used_actions,
                &materials_by_name,
            )?;
        }
    } else {
        for prim in &roots {
            if prim.kind == "Material" || prim.kind == "Scope" {
                continue;
            }
            import_node_tree(
                prim,
                None,
                &mut doc,
                &mut used_nodes,
                &mut used_data,
                &mut used_actions,
                &materials_by_name,
            )?;
        }
    }
    let collection = doc
        .scenes
        .get(&doc.active_scene)
        .map(|scene| scene.root_collection.clone())
        .ok_or_else(|| import_error("USD scene root collection is missing"))?;
    let mut roots = doc
        .nodes
        .iter()
        .filter(|(_, node)| node.parent.is_none())
        .map(|(id, _)| id.clone())
        .collect::<Vec<_>>();
    roots.sort();
    if let Some(collection) = doc.collections.get_mut(&collection) {
        collection.objects = roots;
    }
    if doc.nodes.is_empty() {
        return Err(import_error("USD layer contains no supported scene prims"));
    }
    doc.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "imported USD scene is invalid",
            json!({"reason": error.message}),
        )
    })?;
    Ok(doc)
}

fn import_node_tree(
    prim: &Prim,
    parent: Option<Id>,
    doc: &mut SceneDoc,
    used_nodes: &mut BTreeSet<Id>,
    used_data: &mut BTreeSet<Id>,
    used_actions: &mut BTreeSet<Id>,
    materials: &BTreeMap<String, Id>,
) -> Result<()> {
    if prim.kind == "Scope" && prim.name == "Materials" {
        return Ok(());
    }
    if prim.kind != "Xform"
        && prim.kind != "Mesh"
        && prim.kind != "Camera"
        && !prim.kind.ends_with("Light")
    {
        for child in &prim.children {
            import_node_tree(
                child,
                parent.clone(),
                doc,
                used_nodes,
                used_data,
                used_actions,
                materials,
            )?;
        }
        return Ok(());
    }
    let raw_id = property_string(prim, "potter:id");
    let id = import_id(raw_id.as_deref(), &prim.name, used_nodes)?;
    let mesh_prim = if prim.kind == "Mesh" {
        Some(prim)
    } else {
        prim.children.iter().find(|child| child.kind == "Mesh")
    };
    let camera_prim = if prim.kind == "Camera" {
        Some(prim)
    } else {
        prim.children.iter().find(|child| child.kind == "Camera")
    };
    let light_prim = if prim.kind.ends_with("Light") {
        Some(prim)
    } else {
        prim.children
            .iter()
            .find(|child| child.kind.ends_with("Light"))
    };
    let kind = property_string(prim, "potter:kind").unwrap_or_else(|| {
        if mesh_prim.is_some() {
            "mesh".to_owned()
        } else if camera_prim.is_some() {
            "camera".to_owned()
        } else if light_prim.is_some() {
            "light".to_owned()
        } else {
            "empty".to_owned()
        }
    });
    let mut data_id = None;
    let mut node_materials = Vec::new();
    if let Some(mesh_prim) = mesh_prim {
        let (positions, faces) = import_mesh(mesh_prim)?;
        let mesh = Mesh::from_positions_and_faces(
            positions.iter().copied().map(DVec3::from_array).collect(),
            faces,
        )
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "USD mesh topology is invalid",
                json!({"reason": error.to_string()}),
            )
        })?;
        let mesh_id = import_id(None, &format!("{}_mesh", id.as_str()), used_data)?;
        let (shape_keys, vertex_groups, vertex_weights) = import_deformation(mesh_prim, &mesh)?;
        doc.data_blocks.insert(
            mesh_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                shape_keys,
                vertex_groups,
                vertex_weights,
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        data_id = Some(mesh_id);
        if let Some(raw) = mesh_prim.properties.get("material:binding") {
            let path = raw
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            let name = path
                .trim_start_matches('<')
                .trim_end_matches('>')
                .rsplit('/')
                .next()
                .ok_or_else(|| import_error("USD material binding path is malformed"))?;
            let material = materials
                .get(name)
                .ok_or_else(|| import_error("USD mesh references a missing material"))?;
            node_materials.push(material.clone());
        }
        if let Some(list) = property_strings(prim, "potter:materials") {
            for raw in list {
                let material =
                    Id::new(raw).map_err(|_| import_error("USD material ID is invalid"))?;
                if !doc.materials.contains_key(&material) {
                    return Err(import_error("USD node references a missing material"));
                }
                if !node_materials.contains(&material) {
                    node_materials.push(material);
                }
            }
        }
    } else if let Some(camera_prim) = camera_prim {
        let camera = import_camera(camera_prim)?;
        let block_id = import_id(None, &format!("{}_camera", id.as_str()), used_data)?;
        doc.data_blocks.insert(
            block_id.clone(),
            DataBlock {
                data_type: "camera".to_owned(),
                descriptor: None,
                mesh: None,
                camera: Some(camera),
                light: None,
                ..DataBlock::default()
            },
        );
        data_id = Some(block_id);
        if let Some(scene) = doc.scenes.get_mut(&doc.active_scene)
            && scene.camera.is_none()
        {
            scene.camera = Some(id.clone());
        }
    } else if let Some(light_prim) = light_prim {
        let light = import_light(light_prim)?;
        let block_id = import_id(None, &format!("{}_light", id.as_str()), used_data)?;
        doc.data_blocks.insert(
            block_id.clone(),
            DataBlock {
                data_type: "light".to_owned(),
                descriptor: None,
                mesh: None,
                camera: None,
                light: Some(light),
                ..DataBlock::default()
            },
        );
        data_id = Some(block_id);
    }
    if kind == "armature" {
        let encoded = prim
            .descendants()
            .into_iter()
            .find(|child| child.kind == "Skeleton")
            .and_then(|skeleton| property_string(skeleton, "potter:armature"))
            .ok_or_else(|| import_error("USD armature has no Potter skeleton metadata"))?;
        let armature = serde_json::from_str::<ArmatureData>(&encoded).map_err(|error| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "USD armature metadata is invalid",
                json!({"reason":error.to_string()}),
            )
        })?;
        let armature_data_id = import_id(None, &format!("{}_armature", id.as_str()), used_data)?;
        doc.data_blocks.insert(
            armature_data_id.clone(),
            DataBlock {
                data_type: "armature".to_owned(),
                armature: Some(armature),
                ..DataBlock::default()
            },
        );
        data_id = Some(armature_data_id);
    }
    let transform = import_transform(prim)?;
    let parent_inverse = if let Some(raw) = prim.properties.get("potter:parentInverse") {
        let values = scan_numbers(raw)?;
        if values.len() != 16 {
            return Err(import_error(
                "USD parent-inverse matrix must contain 16 values",
            ));
        }
        let mut matrix = [0.0; 16];
        matrix.copy_from_slice(&values);
        Some(matrix)
    } else {
        None
    };
    let modifiers = property_string(prim, "potter:modifiers")
        .map(|encoded| {
            serde_json::from_str::<Vec<Modifier>>(&encoded).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "USD modifier metadata is invalid",
                    json!({"reason":error.to_string()}),
                )
            })
        })
        .transpose()?
        .unwrap_or_default();
    let pose = property_string(prim, "potter:pose")
        .map(|encoded| {
            serde_json::from_str::<BTreeMap<Id, PoseBone>>(&encoded).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "USD armature pose metadata is invalid",
                    json!({"reason":error.to_string()}),
                )
            })
        })
        .transpose()?
        .unwrap_or_default();
    let tags = property_strings(prim, "potter:tags").unwrap_or_default();
    let mut node = Node {
        name: property_string(prim, "potter:name").unwrap_or_else(|| prim.name.clone()),
        kind,
        primitive: None,
        tags,
        parent,
        parent_inverse,
        transform,
        data: data_id,
        materials: node_materials,
        modifiers,
        visible: property_bool(prim, "potter:visible").unwrap_or(true),
        render_visible: property_bool(prim, "potter:renderVisible").unwrap_or(true),
        selectable: property_bool(prim, "potter:selectable").unwrap_or(true),
        action: None,
        properties: serde_json::Map::new(),
        pose,
        ..Node::default()
    };
    if let Some(encoded) = property_string(prim, "potter:action") {
        let action: Action = serde_json::from_str(&encoded).map_err(|error| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "USD action metadata is invalid",
                json!({"reason": error.to_string()}),
            )
        })?;
        let action_id = import_id(
            property_string(prim, "potter:actionId").as_deref(),
            &format!("{}_action", id.as_str()),
            used_actions,
        )?;
        doc.actions.insert(action_id.clone(), action);
        node.action = Some(action_id);
    } else {
        let action = import_time_samples(prim)?;
        if !action.fcurves.is_empty() {
            let action_id = import_id(None, &format!("{}_action", id.as_str()), used_actions)?;
            doc.actions.insert(action_id.clone(), action);
            node.action = Some(action_id);
        }
    }
    doc.nodes.insert(id.clone(), node);
    for child in &prim.children {
        if child.kind == "Mesh"
            || child.kind == "Camera"
            || child.kind.ends_with("Light")
            || child.kind == "Shader"
        {
            continue;
        }
        import_node_tree(
            child,
            Some(id.clone()),
            doc,
            used_nodes,
            used_data,
            used_actions,
            materials,
        )?;
    }
    Ok(())
}

type ImportedMesh = (Vec<[f64; 3]>, Vec<Vec<usize>>);
type UsdDeformation = (
    Option<ShapeKeyData>,
    Vec<VertexGroup>,
    BTreeMap<u32, BTreeMap<Id, f64>>,
);

fn import_mesh(prim: &Prim) -> Result<ImportedMesh> {
    let point_values = property_numbers(prim, "points")
        .ok_or_else(|| import_error("USD mesh points are missing"))?;
    if point_values.len() % 3 != 0 {
        return Err(import_error("USD mesh point array has an invalid size"));
    }
    let points = point_values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| [value[0], value[1], value[2]])
        .collect::<Vec<_>>();
    let counts = property_numbers(prim, "faceVertexCounts")
        .ok_or_else(|| import_error("USD mesh faceVertexCounts are missing"))?
        .into_iter()
        .map(integral_index)
        .collect::<Result<Vec<_>>>()?;
    let indices = property_numbers(prim, "faceVertexIndices")
        .ok_or_else(|| import_error("USD mesh faceVertexIndices are missing"))?
        .into_iter()
        .map(integral_index)
        .collect::<Result<Vec<_>>>()?;
    let total = counts
        .iter()
        .try_fold(0_usize, |sum, count| sum.checked_add(*count))
        .ok_or_else(|| import_error("USD mesh index count overflows"))?;
    if total != indices.len() {
        return Err(import_error(
            "USD mesh face indices do not match face counts",
        ));
    }
    let mut faces = Vec::with_capacity(counts.len());
    let mut offset = 0_usize;
    for count in counts {
        if count < 3 {
            return Err(import_error("USD mesh face has fewer than three vertices"));
        }
        let end = offset
            .checked_add(count)
            .ok_or_else(|| import_error("USD mesh index count overflow"))?;
        let face = indices
            .get(offset..end)
            .ok_or_else(|| import_error("USD mesh face indices are truncated"))?
            .to_vec();
        if face.iter().any(|index| *index >= points.len()) {
            return Err(import_error("USD mesh references an absent point"));
        }
        faces.push(face);
        offset = end;
    }
    Ok((points, faces))
}
fn import_deformation(prim: &Prim, mesh: &Mesh) -> Result<UsdDeformation> {
    let Some(encoded) = property_string(prim, "potter:deformation") else {
        return Ok((None, Vec::new(), BTreeMap::new()));
    };
    let metadata = serde_json::from_str::<Value>(&encoded).map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "USD deformation metadata is invalid",
            json!({"reason":error.to_string()}),
        )
    })?;
    let source_vertex_ids = metadata["vertex_ids"]
        .as_array()
        .ok_or_else(|| import_error("USD deformation vertex IDs are missing"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| import_error("USD deformation vertex ID is invalid"))
        })
        .collect::<Result<Vec<_>>>()?;
    if source_vertex_ids.len() != mesh.vertices.len() {
        return Err(import_error(
            "USD deformation vertex IDs do not match mesh point count",
        ));
    }
    let mut vertex_id_map = BTreeMap::new();
    for (source_id, vertex) in source_vertex_ids.iter().zip(&mesh.vertices) {
        if vertex_id_map.insert(*source_id, vertex.id).is_some() {
            return Err(import_error("USD deformation vertex IDs are duplicated"));
        }
    }
    let remap_positions = |positions: BTreeMap<u32, [f64; 3]>| {
        positions
            .into_iter()
            .map(|(source_id, position)| {
                vertex_id_map
                    .get(&source_id)
                    .copied()
                    .map(|vertex_id| (vertex_id, position))
                    .ok_or_else(|| import_error("USD deformation references a missing vertex"))
            })
            .collect::<Result<BTreeMap<_, _>>>()
    };
    let mut shape_keys = serde_json::from_value::<Option<ShapeKeyData>>(
        metadata["shape_keys"].clone(),
    )
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "USD shape-key metadata is invalid",
            json!({"reason":error.to_string()}),
        )
    })?;
    if let Some(shape_keys) = &mut shape_keys {
        shape_keys.basis = remap_positions(std::mem::take(&mut shape_keys.basis))?;
        for key in shape_keys.keys.values_mut() {
            key.positions = remap_positions(std::mem::take(&mut key.positions))?;
        }
    }
    let vertex_groups = serde_json::from_value::<Vec<VertexGroup>>(
        metadata["vertex_groups"].clone(),
    )
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "USD vertex-group metadata is invalid",
            json!({"reason":error.to_string()}),
        )
    })?;
    let source_weights = serde_json::from_value::<BTreeMap<u32, BTreeMap<Id, f64>>>(
        metadata["vertex_weights"].clone(),
    )
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "USD vertex-weight metadata is invalid",
            json!({"reason":error.to_string()}),
        )
    })?;
    let vertex_weights = source_weights
        .into_iter()
        .map(|(source_id, weights)| {
            vertex_id_map
                .get(&source_id)
                .copied()
                .map(|vertex_id| (vertex_id, weights))
                .ok_or_else(|| import_error("USD weights reference a missing vertex"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok((shape_keys, vertex_groups, vertex_weights))
}

fn import_camera(prim: &Prim) -> Result<CameraData> {
    let projection =
        property_string(prim, "projection").unwrap_or_else(|| "perspective".to_owned());
    let mut camera = CameraData::default();
    camera.projection = match projection.as_str() {
        "orthographic" => CameraProjection::Orthographic,
        "perspective" => CameraProjection::Perspective,
        _ => return Err(import_error("USD camera projection is unsupported")),
    };
    camera.lens_mm = property_number(prim, "focalLength").unwrap_or(camera.lens_mm);
    camera.sensor_width_mm =
        property_number(prim, "horizontalAperture").unwrap_or(camera.sensor_width_mm);
    if let Some(values) = property_numbers(prim, "clippingRange")
        && values.len() == 2
    {
        camera.clip_start = values[0];
        camera.clip_end = values[1];
    }
    camera.ortho_scale = property_number(prim, "potter:orthoScale").unwrap_or(camera.ortho_scale);
    if let Some(values) = property_numbers(prim, "potter:shift")
        && values.len() == 2
    {
        camera.shift = [values[0], values[1]];
    }
    Ok(camera)
}

fn import_light(prim: &Prim) -> Result<LightData> {
    let raw = property_string(prim, "potter:lightType");
    let inferred = match prim.kind.as_str() {
        "DistantLight" => "sun",
        "DiskLight" | "RectLight" => "area",
        "SphereLight" => "point",
        _ if raw.is_some() => "point",
        _ => {
            return Err(unsupported(
                "light.usd_schema",
                format!("USD light schema `{}` is unsupported", prim.kind),
            ));
        }
    };
    let light_type = match raw.as_deref().unwrap_or(inferred) {
        "point" => LightType::Point,
        "sun" => LightType::Sun,
        "area" => LightType::Area,
        "spot" => LightType::Spot,
        _ => return Err(import_error("USD light type is unsupported")),
    };
    let mut light = LightData {
        light_type,
        ..LightData::default()
    };
    if let Some(color) = property_numbers(prim, "inputs:color")
        && color.len() == 3
    {
        light.color = [color[0], color[1], color[2]];
    }
    light.energy = property_number(prim, "inputs:intensity").unwrap_or(light.energy);
    light.radius = property_number(prim, "inputs:radius").unwrap_or(light.radius);
    light.spot_size = property_number(prim, "potter:spotSize").unwrap_or(light.spot_size);
    light.spot_blend = property_number(prim, "potter:spotBlend").unwrap_or(light.spot_blend);
    Ok(light)
}

fn import_transform(prim: &Prim) -> Result<Transform> {
    let order = property_strings(prim, "xformOpOrder");
    let matrix_active = order
        .as_ref()
        .is_some_and(|ops| ops.iter().any(|op| op == "xformOp:transform"))
        || order.is_none() && prim.properties.contains_key("xformOp:transform");
    if matrix_active && !prim.properties.contains_key("potter:parentInverse") {
        let ops = order.unwrap_or_else(|| vec!["xformOp:transform".to_owned()]);
        if ops.len() != 1 {
            return Err(unsupported(
                "transform.xform_ops",
                "USD matrix transforms combined with other xform ops are unsupported".to_owned(),
            ));
        }
        let values = numbers_array::<16>(prim, "xformOp:transform")?
            .ok_or_else(|| import_error("USD matrix transform is missing"))?;
        let matrix = DMat4::from_cols_array(&values);
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
            return Err(import_error("USD matrix transform is not finite"));
        }
        let transform = Transform::from_rotation_quat(
            translation.to_array(),
            [rotation.x, rotation.y, rotation.z, rotation.w],
            scale.to_array(),
        )
        .map_err(|error| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "USD matrix transform is invalid",
                json!({"reason": error.message}),
            )
        })?;
        let reconstructed = transform.matrix().to_cols_array();
        if matrix
            .to_cols_array()
            .iter()
            .zip(reconstructed)
            .any(|(original, value)| (*original - value).abs() > 1.0e-9 * (1.0 + original.abs()))
        {
            return Err(unsupported(
                "transform.matrix_shear",
                "USD matrix transform contains shear not representable as TRS".to_owned(),
            ));
        }
        return Ok(transform);
    }
    if let Some(ops) = &order {
        let mut rotations = 0;
        for op in ops {
            if !matches!(
                op.as_str(),
                "xformOp:translate"
                    | "xformOp:orient"
                    | "xformOp:scale"
                    | "xformOp:rotateXYZ"
                    | "xformOp:transform"
            ) {
                return Err(unsupported(
                    "transform.xform_ops",
                    format!("USD transform operation `{op}` is unsupported"),
                ));
            }
            if matches!(op.as_str(), "xformOp:orient" | "xformOp:rotateXYZ") {
                rotations += 1;
            }
        }
        if rotations > 1 {
            return Err(unsupported(
                "transform.xform_ops",
                "USD transform has multiple rotation operations".to_owned(),
            ));
        }
    }
    let translation = numbers_array::<3>(prim, "xformOp:translate")?.unwrap_or([0.0; 3]);
    let scale = numbers_array::<3>(prim, "xformOp:scale")?.unwrap_or([1.0; 3]);
    let rotation = if let Some([real, x, y, z]) = numbers_array::<4>(prim, "xformOp:orient")? {
        [x, y, z, real]
    } else if let Some([x, y, z]) = numbers_array::<3>(prim, "xformOp:rotateXYZ")? {
        let rotation = DQuat::from_euler(
            EulerRot::XYZ,
            x.to_radians(),
            y.to_radians(),
            z.to_radians(),
        );
        [rotation.x, rotation.y, rotation.z, rotation.w]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    };
    Transform::from_rotation_quat(translation, rotation, scale).map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "USD TRS transform is invalid",
            json!({"reason": error.message}),
        )
    })
}

fn import_time_samples(prim: &Prim) -> Result<Action> {
    let mut curves = Vec::new();
    for (property, path, width, quaternion) in [
        (
            "xformOp:translate.timeSamples",
            "transform.translation",
            3,
            false,
        ),
        ("xformOp:scale.timeSamples", "transform.scale", 3, false),
        (
            "xformOp:orient.timeSamples",
            "transform.rotation_quaternion",
            4,
            true,
        ),
    ] {
        let Some(raw) = prim.properties.get(property) else {
            continue;
        };
        let numbers = scan_numbers(raw)?;
        let stride = width + 1;
        if numbers.len() % stride != 0 {
            return Err(import_error("USD transform timeSamples are malformed"));
        }
        for component in 0..width {
            let keyframes = numbers
                .chunks_exact(stride)
                .map(|sample| {
                    let value = if quaternion {
                        match component {
                            0 => sample[2],
                            1 => sample[3],
                            2 => sample[4],
                            _ => sample[1],
                        }
                    } else {
                        sample[component + 1]
                    };
                    Keyframe {
                        frame: sample[0],
                        value,
                        interpolation: Interpolation::Linear,
                        ..Keyframe::default()
                    }
                })
                .collect::<Vec<_>>();
            curves.push(FCurve {
                path: path.to_owned(),
                index: component as u32,
                keyframes,
                extrapolation: crate::model::Extrapolation::default(),
            });
        }
    }
    Ok(Action {
        name: "USD Animation".to_owned(),
        fcurves: curves,
        ..Action::default()
    })
}

fn lex_usda(text: &str) -> Result<Vec<Token>> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut line_no = 1;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'\n' {
            line_no += 1;
            i += 1;
            continue;
        }
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if byte == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        let start_line = line_no;
        if byte == b'"' {
            i += 1;
            let mut escaped = false;
            while i < bytes.len() {
                if bytes[i] == b'\n' {
                    line_no += 1;
                }
                if escaped {
                    escaped = false;
                    i += 1;
                    continue;
                }
                if bytes[i] == b'\\' {
                    escaped = true;
                    i += 1;
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            if i > bytes.len() || bytes.get(i.saturating_sub(1)) != Some(&b'"') {
                return Err(import_error("unterminated string in USDA layer"));
            }
        } else if b"{}[](),=<>".contains(&byte) {
            i += 1;
        } else {
            i += 1;
            while i < bytes.len()
                && !bytes[i].is_ascii_whitespace()
                && !b"{}[](),=<>\"#".contains(&bytes[i])
            {
                i += 1;
            }
        }
        let value = text
            .get(start..i)
            .ok_or_else(|| import_error("USDA token boundary is invalid"))?
            .to_owned();
        tokens.push(Token {
            value,
            line: start_line,
        });
    }
    Ok(tokens)
}

#[derive(Clone)]
struct Token {
    value: String,
    line: usize,
}

#[derive(Clone)]
struct Prim {
    kind: String,
    name: String,
    properties: BTreeMap<String, String>,
    children: Vec<Prim>,
}

impl Prim {
    fn descendants(&self) -> Vec<&Prim> {
        fn collect<'a>(prim: &'a Prim, output: &mut Vec<&'a Prim>) {
            for child in &prim.children {
                output.push(child);
                collect(child, output);
            }
        }
        let mut output = Vec::new();
        collect(self, &mut output);
        output
    }
}
fn parse_prims(tokens: &[Token], start: usize, end: usize) -> Result<Vec<Prim>> {
    let mut result = Vec::new();
    let mut index = start;
    while index < end {
        if tokens[index].value == "def" {
            let (prim, next) = parse_prim(tokens, index, end)?;
            result.push(prim);
            index = next;
        } else {
            index += 1;
        }
    }
    Ok(result)
}

fn parse_prim(tokens: &[Token], start: usize, end: usize) -> Result<(Prim, usize)> {
    let kind = tokens
        .get(start + 1)
        .ok_or_else(|| import_error("USDA prim type is missing"))?
        .value
        .clone();
    let name_token = tokens
        .get(start + 2)
        .ok_or_else(|| import_error("USDA prim name is missing"))?;
    let name = decode_string(&name_token.value).unwrap_or_else(|| name_token.value.clone());
    let mut index = start + 3;
    if tokens.get(index).is_some_and(|token| token.value == "(") {
        index = skip_balanced(tokens, index, end, "(", ")")?;
    }
    if tokens.get(index).is_none_or(|token| token.value != "{") {
        return Err(import_error("USDA prim body is missing"));
    }
    let body_start = index + 1;
    let body_end = matching_close(tokens, index, end, "{", "}")?;
    let mut properties = BTreeMap::new();
    let mut children = Vec::new();
    index = body_start;
    while index < body_end {
        if tokens[index].value == "def" {
            let (child, next) = parse_prim(tokens, index, body_end)?;
            children.push(child);
            index = next;
            continue;
        }
        if tokens[index].value == "=" {
            let key = tokens
                .get(index.wrapping_sub(1))
                .map(|token| token.value.clone())
                .ok_or_else(|| import_error("USDA attribute name is missing"))?;
            let value_start = index + 1;
            let value_end = attribute_value_end(tokens, value_start, body_end);
            let raw = tokens[value_start..value_end]
                .iter()
                .map(|token| token.value.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            properties.insert(key, raw);
            index = value_end.max(index + 1);
            continue;
        }
        index += 1;
    }
    Ok((
        Prim {
            kind,
            name,
            properties,
            children,
        },
        body_end + 1,
    ))
}

fn attribute_value_end(tokens: &[Token], start: usize, end: usize) -> usize {
    if start >= end {
        return end;
    }
    let start_line = tokens[start].line;
    let mut stack = Vec::new();
    let mut index = start;
    while index < end {
        let token = tokens[index].value.as_str();
        if stack.is_empty() && index > start && tokens[index].line > start_line {
            break;
        }
        match token {
            "[" => stack.push("]"),
            "{" => stack.push("}"),
            "(" => stack.push(")"),
            "<" => stack.push(">"),
            "]" | "}" | ")" | ">" => {
                stack.pop();
            }
            _ => {}
        }
        index += 1;
    }
    index
}

fn matching_close(
    tokens: &[Token],
    open: usize,
    end: usize,
    left: &str,
    right: &str,
) -> Result<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate().take(end).skip(open) {
        if token.value == left {
            depth += 1;
        }
        if token.value == right {
            depth = depth
                .checked_sub(1)
                .ok_or_else(|| import_error("unbalanced USDA delimiters"))?;
            if depth == 0 {
                return Ok(index);
            }
        }
    }
    Err(import_error("unterminated USDA prim"))
}

fn skip_balanced(
    tokens: &[Token],
    open: usize,
    end: usize,
    left: &str,
    right: &str,
) -> Result<usize> {
    matching_close(tokens, open, end, left, right).map(|close| close + 1)
}

fn property_string(prim: &Prim, name: &str) -> Option<String> {
    prim.properties.get(name).and_then(|raw| first_string(raw))
}
fn property_strings(prim: &Prim, name: &str) -> Option<Vec<String>> {
    prim.properties.get(name).map(|raw| all_strings(raw))
}
fn property_numbers(prim: &Prim, name: &str) -> Option<Vec<f64>> {
    prim.properties
        .get(name)
        .and_then(|raw| scan_numbers(raw).ok())
}
fn property_number(prim: &Prim, name: &str) -> Option<f64> {
    property_numbers(prim, name).and_then(|values| values.first().copied())
}
fn property_bool(prim: &Prim, name: &str) -> Option<bool> {
    prim.properties
        .get(name)
        .and_then(|value| match value.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        })
}
fn first_string(raw: &str) -> Option<String> {
    let start = raw.find('"')?;
    let end = quoted_end(raw, start)?;
    decode_string(&raw[start..=end])
}
fn all_strings(raw: &str) -> Vec<String> {
    let bytes = raw.as_bytes();
    let mut strings = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'"' {
            if let Some(end) = quoted_end(raw, index) {
                if let Some(value) = decode_string(&raw[index..=end]) {
                    strings.push(value);
                }
                index = end + 1;
            } else {
                break;
            }
        } else {
            index += 1;
        }
    }
    strings
}
fn quoted_end(raw: &str, start: usize) -> Option<usize> {
    let bytes = raw.as_bytes();
    let mut escaped = false;
    for (offset, byte) in bytes.iter().enumerate().skip(start + 1) {
        if escaped {
            escaped = false;
            continue;
        }
        if *byte == b'\\' {
            escaped = true;
            continue;
        }
        if *byte == b'"' {
            return Some(offset);
        }
    }
    None
}
fn decode_string(raw: &str) -> Option<String> {
    if raw.starts_with('"') {
        serde_json::from_str(raw).ok()
    } else {
        None
    }
}

fn scan_numbers(raw: &str) -> Result<Vec<f64>> {
    let bytes = raw.as_bytes();
    let mut values = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_digit()
            || ((bytes[index] == b'-' || bytes[index] == b'+')
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
            || (bytes[index] == b'.' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_digit()
                    || matches!(bytes[index], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                index += 1;
            }
            let value = raw
                .get(start..index)
                .and_then(|value| value.parse::<f64>().ok())
                .ok_or_else(|| import_error("USDA numeric value is malformed"))?;
            if !value.is_finite() {
                return Err(import_error("USDA numeric value is not finite"));
            }
            values.push(value);
        } else {
            index += 1;
        }
    }
    Ok(values)
}
fn numbers_array<const N: usize>(prim: &Prim, name: &str) -> Result<Option<[f64; N]>> {
    let Some(raw) = prim.properties.get(name) else {
        return Ok(None);
    };
    let values = scan_numbers(raw)?;
    let values: [f64; N] = values
        .try_into()
        .map_err(|_| import_error("USD transform attribute has an invalid component count"))?;
    Ok(Some(values))
}

fn import_id(raw: Option<&str>, fallback: &str, used: &mut BTreeSet<Id>) -> Result<Id> {
    let base = if let Some(raw) = raw {
        Id::new(raw.to_owned())
            .map_err(|_| import_error("USD potter:id is not a valid Potter ID"))?
    } else {
        Id::new(id_seed(fallback))
            .map_err(|_| import_error("USD prim name cannot be converted to a Potter ID"))?
    };
    if used.insert(base.clone()) {
        return Ok(base);
    }
    let source = base.as_str();
    for suffix in 2_u32.. {
        let suffix = format!("_{suffix}");
        let keep = 64_usize.saturating_sub(suffix.len());
        let candidate = format!("{}{}", &source[..source.len().min(keep)], suffix);
        let id = Id::new(candidate)?;
        if used.insert(id.clone()) {
            return Ok(id);
        }
    }
    Err(PotError::new(
        ErrorCode::LimitExceeded,
        "USD ID uniqueness space is exhausted",
    ))
}

fn id_seed(raw: &str) -> String {
    let mut value = String::with_capacity(raw.len().min(64));
    for character in raw.chars() {
        if value.len() >= 64 {
            break;
        }
        let character = character.to_ascii_lowercase();
        let character = if character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || character == '_'
            || character == '-'
        {
            character
        } else {
            '_'
        };
        if value.is_empty() && !character.is_ascii_lowercase() {
            value.push('n');
        }
        value.push(character);
    }
    if value.is_empty() {
        "usd_import".to_owned()
    } else {
        value
    }
}

fn integral_index(value: f64) -> Result<usize> {
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(import_error("USD mesh index is invalid"));
    }
    Ok(value as usize)
}

fn face_normal(face: &[usize], positions: &[[f64; 3]]) -> Result<[f64; 3]> {
    let a = positions
        .get(
            *face
                .first()
                .ok_or_else(|| import_error("USD face is empty"))?,
        )
        .ok_or_else(|| import_error("USD face references an absent point"))?;
    let b = positions
        .get(
            *face
                .get(1)
                .ok_or_else(|| import_error("USD face has fewer than three vertices"))?,
        )
        .ok_or_else(|| import_error("USD face references an absent point"))?;
    let c = positions
        .get(
            *face
                .get(2)
                .ok_or_else(|| import_error("USD face has fewer than three vertices"))?,
        )
        .ok_or_else(|| import_error("USD face references an absent point"))?;
    let normal = (DVec3::from_array(*b) - DVec3::from_array(*a))
        .cross(DVec3::from_array(*c) - DVec3::from_array(*a))
        .normalize_or_zero();
    Ok(normal.to_array())
}

fn generated_uvs(points: &[[f64; 3]]) -> Vec<[f64; 2]> {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for point in points {
        for axis in 0..2 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    points
        .iter()
        .map(|point| {
            std::array::from_fn(|axis| {
                let span = max[axis] - min[axis];
                if span > 0.0 {
                    (point[axis] - min[axis]) / span
                } else {
                    0.5
                }
            })
        })
        .collect()
}

fn matrix4d(matrix: DMat4) -> Result<String> {
    let values = matrix.to_cols_array();
    let rows = values
        .as_chunks::<4>()
        .0
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| number(*value))
                .collect::<Result<Vec<_>>>()
                .map(|row| format!("({})", row.join(", ")))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("({})", rows.join(", ")))
}
fn vector_format(values: &[f64]) -> Result<String> {
    Ok(format!(
        "({})",
        values
            .iter()
            .map(|value| number(*value))
            .collect::<Result<Vec<_>>>()?
            .join(", ")
    ))
}
fn vec3(values: [f64; 3]) -> Result<String> {
    vector_format(&values)
}
fn number(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "USD cannot represent a non-finite number",
        ));
    }
    Ok(value.to_string())
}
fn quote(value: &str) -> Result<String> {
    serde_json::to_string(value).map_err(|error| {
        PotError::with_details(
            ErrorCode::ExportFailed,
            "could not quote USD string",
            json!({"reason": error.to_string()}),
        )
    })
}
fn string_array(values: &[String]) -> Result<String> {
    Ok(format!(
        "[{}]",
        values
            .iter()
            .map(|value| quote(value))
            .collect::<Result<Vec<_>>>()?
            .join(", ")
    ))
}
fn prim_identifier(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 1);
    for (index, byte) in value.bytes().enumerate() {
        let character = char::from(byte);
        if character.is_ascii_alphanumeric() || character == '_' {
            if index == 0 && character.is_ascii_digit() {
                result.push('_');
            }
            result.push(character);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() {
        result.push('_');
    }
    result
}
fn line(output: &mut String, value: &str) {
    output.push_str(value);
    output.push('\n');
}
fn indent(output: &mut String, depth: usize) {
    output.push_str(&spaces(depth));
}
fn spaces(depth: usize) -> String {
    "    ".repeat(depth)
}

fn unsupported(feature: &str, message: String) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id": feature}),
    )
}

fn root_layer_name(root_name: &str) -> String {
    let basename = Path::new(root_name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("scene");
    let stem = basename
        .strip_suffix(".usdz")
        .or_else(|| basename.strip_suffix(".usda"))
        .unwrap_or(basename);
    let mut clean = stem
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if clean.is_empty() {
        clean.push_str("scene");
    }
    clean.push_str(".usda");
    clean
}

fn make_stored_zip(filename: &str, contents: &[u8]) -> Result<Vec<u8>> {
    let name = filename.as_bytes();
    let name_len = u16::try_from(name.len())
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "USDZ root layer name is too long"))?;
    let size = u32::try_from(contents.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "USDZ root layer exceeds ZIP32 size",
        )
    })?;
    let remainder = (64 - ((30 + name.len()) % 64)) % 64;
    let extra_len = if remainder == 0 {
        0
    } else if remainder < 4 {
        remainder + 64
    } else {
        remainder
    };
    let extra_len_u16 = u16::try_from(extra_len).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "USDZ alignment padding is too large",
        )
    })?;
    let crc = crc32(contents);
    let mut zip = Vec::with_capacity(contents.len().saturating_add(160));
    put_u32(&mut zip, 0x0403_4b50);
    put_u16(&mut zip, 20);
    put_u16(&mut zip, 0x0800);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u32(&mut zip, crc);
    put_u32(&mut zip, size);
    put_u32(&mut zip, size);
    put_u16(&mut zip, name_len);
    put_u16(&mut zip, extra_len_u16);
    zip.extend_from_slice(name);
    if extra_len > 0 {
        put_u16(&mut zip, 0xffff);
        put_u16(&mut zip, (extra_len - 4) as u16);
        zip.resize(zip.len() + extra_len - 4, 0);
    }
    let root_offset = u32::try_from(zip.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "USDZ local header exceeds ZIP32 range",
        )
    })?;
    if root_offset % 64 != 0 {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "USDZ root layer alignment calculation failed",
        ));
    }
    zip.extend_from_slice(contents);
    let central_offset = u32::try_from(zip.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "USDZ central directory exceeds ZIP32 range",
        )
    })?;
    put_u32(&mut zip, 0x0201_4b50);
    put_u16(&mut zip, 20);
    put_u16(&mut zip, 20);
    put_u16(&mut zip, 0x0800);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u32(&mut zip, crc);
    put_u32(&mut zip, size);
    put_u32(&mut zip, size);
    put_u16(&mut zip, name_len);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u32(&mut zip, 0);
    put_u32(&mut zip, 0);
    zip.extend_from_slice(name);
    let central_size = u32::try_from(zip.len())
        .map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "USDZ central directory is too large",
            )
        })?
        .checked_sub(central_offset)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "USDZ central directory size underflow",
            )
        })?;
    put_u32(&mut zip, 0x0605_4b50);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 0);
    put_u16(&mut zip, 1);
    put_u16(&mut zip, 1);
    put_u32(&mut zip, central_size);
    put_u32(&mut zip, central_offset);
    put_u16(&mut zip, 0);
    Ok(zip)
}

fn read_usdz_root(bytes: &[u8]) -> Result<(String, &[u8])> {
    if bytes.len() < 22 {
        return Err(import_error("USDZ ZIP archive is truncated"));
    }
    let search_start = bytes.len().saturating_sub(65_557);
    let eocd = bytes[search_start..]
        .windows(4)
        .rposition(|window| window == b"PK\x05\x06")
        .map(|offset| search_start + offset)
        .ok_or_else(|| import_error("USDZ ZIP end record is missing"))?;
    let disk = get_u16(bytes, eocd + 4)?;
    let central_disk = get_u16(bytes, eocd + 6)?;
    let disk_entries = get_u16(bytes, eocd + 8)?;
    let total_entries = get_u16(bytes, eocd + 10)?;
    if disk != 0 || central_disk != 0 || disk_entries != total_entries {
        return Err(unsupported(
            "format.usdz_multidisk",
            "multi-disk USDZ archives are unsupported".to_owned(),
        ));
    }
    if total_entries == u16::MAX {
        return Err(unsupported(
            "format.usdz_zip64",
            "ZIP64 USDZ archives are unsupported".to_owned(),
        ));
    }
    let central_size = get_u32(bytes, eocd + 12)? as usize;
    let central_offset = get_u32(bytes, eocd + 16)? as usize;
    let central_end = central_offset
        .checked_add(central_size)
        .ok_or_else(|| import_error("USDZ central directory range overflows"))?;
    if central_end > eocd || central_end > bytes.len() {
        return Err(import_error("USDZ central directory is truncated"));
    }
    let mut cursor = central_offset;
    let mut root: Option<(String, &[u8])> = None;
    for _ in 0..total_entries {
        if get_u32(bytes, cursor)? != 0x0201_4b50 {
            return Err(import_error("USDZ central directory entry is malformed"));
        }
        let flags = get_u16(bytes, cursor + 8)?;
        let method = get_u16(bytes, cursor + 10)?;
        let crc = get_u32(bytes, cursor + 16)?;
        let compressed_size = get_u32(bytes, cursor + 20)? as usize;
        let uncompressed_size = get_u32(bytes, cursor + 24)? as usize;
        let name_len = get_u16(bytes, cursor + 28)? as usize;
        let extra_len = get_u16(bytes, cursor + 30)? as usize;
        let comment_len = get_u16(bytes, cursor + 32)? as usize;
        let local_offset = get_u32(bytes, cursor + 42)? as usize;
        let name_start = cursor
            .checked_add(46)
            .ok_or_else(|| import_error("USDZ entry range overflows"))?;
        let name_end = name_start
            .checked_add(name_len)
            .ok_or_else(|| import_error("USDZ filename range overflows"))?;
        let end = name_end
            .checked_add(extra_len)
            .and_then(|value| value.checked_add(comment_len))
            .ok_or_else(|| import_error("USDZ directory entry range overflows"))?;
        let name = std::str::from_utf8(
            bytes
                .get(name_start..name_end)
                .ok_or_else(|| import_error("USDZ filename is truncated"))?,
        )
        .map_err(|_| import_error("USDZ filename is not UTF-8"))?
        .to_owned();
        if flags & 1 != 0 {
            return Err(unsupported(
                "format.usdz_encryption",
                "encrypted USDZ entries are unsupported".to_owned(),
            ));
        }
        if method != 0 {
            return Err(unsupported(
                "format.usdz_compression",
                format!("USDZ compression method {method} is unsupported"),
            ));
        }
        if compressed_size != uncompressed_size {
            return Err(import_error("stored USDZ entry has inconsistent sizes"));
        }
        if Path::new(&name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("usda"))
            && root.is_none()
        {
            if get_u32(bytes, local_offset)? != 0x0403_4b50 {
                return Err(import_error("USDZ root local file header is malformed"));
            }
            let local_flags = get_u16(bytes, local_offset + 6)?;
            let local_method = get_u16(bytes, local_offset + 8)?;
            if local_flags & 1 != 0 {
                return Err(unsupported(
                    "format.usdz_encryption",
                    "encrypted USDZ root layers are unsupported".to_owned(),
                ));
            }
            if local_method != 0 {
                return Err(unsupported(
                    "format.usdz_compression",
                    format!("USDZ compression method {local_method} is unsupported"),
                ));
            }
            let local_name_len = get_u16(bytes, local_offset + 26)? as usize;
            let local_extra_len = get_u16(bytes, local_offset + 28)? as usize;
            let data_offset = local_offset
                .checked_add(30)
                .and_then(|value| value.checked_add(local_name_len))
                .and_then(|value| value.checked_add(local_extra_len))
                .ok_or_else(|| import_error("USDZ root data offset overflows"))?;
            if data_offset % 64 != 0 {
                return Err(import_error("USDZ root layer data is not 64-byte aligned"));
            }
            let data_end = data_offset
                .checked_add(uncompressed_size)
                .ok_or_else(|| import_error("USDZ root layer size overflows"))?;
            let data = bytes
                .get(data_offset..data_end)
                .ok_or_else(|| import_error("USDZ root layer is truncated"))?;
            if crc32(data) != crc {
                return Err(import_error("USDZ root layer checksum does not match"));
            }
            root = Some((name.clone(), data));
        }
        if end <= cursor {
            return Err(import_error("USDZ directory entry length is invalid"));
        }
        cursor = end;
    }
    root.ok_or_else(|| import_error("USDZ archive does not contain a USDA root layer"))
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320_u32 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}
fn put_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn get_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset.saturating_add(2))
        .ok_or_else(|| import_error("USDZ archive is truncated"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}
fn get_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset.saturating_add(4))
        .ok_or_else(|| import_error("USDZ archive is truncated"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}
