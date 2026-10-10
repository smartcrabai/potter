use std::collections::{BTreeMap, BTreeSet};

use glam::DMat4;
use serde_json::{Map, Value, json};

use crate::{
    catalog,
    error::{ErrorCode, Result},
    geom::{Face, Mesh},
    model::{Id, SceneDoc},
};

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub value: Value,
    pub has_warnings: bool,
    pub has_errors: bool,
}

pub fn validate(doc: &SceneDoc, format: Option<&str>) -> Result<ValidationReport> {
    let mut issues = Vec::new();
    let mut checks = vec![
        check("schema", "supported", Value::Null),
        check("id", "supported", Value::Null),
        check("reference", "supported", Value::Null),
        check("parent_cycle", "supported", Value::Null),
        check("collection_cycle", "supported", Value::Null),
        check("mesh_indices", "supported", Value::Null),
        check("non_finite", "supported", Value::Null),
        check("mesh_manifold", "supported", Value::Null),
        check("mesh_winding", "supported", Value::Null),
        check("mesh_area", "supported", Value::Null),
        check("singular_transform", "supported", Value::Null),
        check("unused_data_blocks", "supported", Value::Null),
        check(
            "asset_hash",
            "not_supported",
            json!("External asset URI/hash resolution is not implemented."),
        ),
        check(
            "library_dependencies",
            "not_supported",
            json!("Library dependency and runtime resolution is not implemented."),
        ),
        check(
            "rig_graph",
            "not_supported",
            json!("Rig, skin, and shape-key consistency checks are not implemented."),
        ),
        check(
            "node_graph",
            "not_supported",
            json!("Node graph link and socket checks are not implemented."),
        ),
        check(
            "driver_constraint",
            "not_supported",
            json!("Driver and constraint dependency checks are not implemented."),
        ),
        check(
            "animation",
            "not_supported",
            json!("Full Action/NLA/frame coverage checks are not implemented."),
        ),
        check("rigid_body_simulation", "supported", Value::Null),
        check(
            "simulation_cache",
            "not_supported",
            json!(
                "Cloth, fluid, particle, and other non-rigid-body simulation caches are not implemented."
            ),
        ),
        check(
            "shader_compositor_sequencer",
            "not_supported",
            json!("Shader, color, compositor, and sequencer checks are not implemented."),
        ),
        check(
            "feature_compatibility",
            "not_supported",
            json!("Full Blender-standard feature compatibility checks are not implemented."),
        ),
    ];

    for id in doc
        .scenes
        .keys()
        .chain(doc.collections.keys())
        .chain(doc.nodes.keys())
        .chain(doc.data_blocks.keys())
        .chain(doc.materials.keys())
        .chain(doc.worlds.keys())
        .chain(doc.actions.keys())
        .chain(doc.resources.keys())
        .chain(doc.libraries.keys())
    {
        if !crate::model::is_valid_id(id.as_str()) {
            issue(
                &mut issues,
                "error",
                "INVALID_ID",
                "Registry ID does not match the required ID pattern.",
                Some(id.to_string()),
                None,
                json!({"id":id}),
            );
        }
    }

    for (scene_id, scene) in &doc.scenes {
        if !doc.collections.contains_key(&scene.root_collection) {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Scene root_collection does not exist.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}/root_collection")),
                json!({"target":scene.root_collection}),
            );
        }
        if let Some(camera) = &scene.camera
            && doc
                .nodes
                .get(camera)
                .is_none_or(|node| node.kind != "camera")
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Scene camera does not reference a camera Object.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}/camera")),
                json!({"target":camera}),
            );
        }
        if let Some(world) = &scene.world
            && !doc.worlds.contains_key(world)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Scene world does not exist.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}/world")),
                json!({"target":world}),
            );
        }
        if let Some(clip) = &scene.active_clip
            && !doc.movie_clips.contains_key(clip)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Scene active_clip does not exist.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}/active_clip")),
                json!({"target":clip}),
            );
        }
        if !scene.unit.scale_length.is_finite() || scene.unit.scale_length <= 0.0 {
            issue(
                &mut issues,
                "error",
                "INVALID_UNIT",
                "Scene scale_length must be finite and positive.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}/unit/scale_length")),
                json!({}),
            );
        }
        for (layer_id, layer) in &scene.view_layers {
            for excluded in &layer.excluded_collections {
                if !doc.collections.contains_key(excluded) {
                    issue(
                        &mut issues,
                        "error",
                        "MISSING_REFERENCE",
                        "View Layer exclusion references a missing Collection.",
                        Some(scene_id.to_string()),
                        Some(format!(
                            "/scenes/{scene_id}/view_layers/{layer_id}/excluded_collections"
                        )),
                        json!({"target":excluded}),
                    );
                }
            }
        }
        if !scene.frame_current.is_finite()
            || !scene.fps_base.is_finite()
            || scene.fps == 0
            || scene.fps_base <= 0.0
            || scene.frame_start > scene.frame_end
        {
            issue(
                &mut issues,
                "error",
                "INVALID_SCENE_TIMING",
                "Scene frame range or frame-rate settings are invalid.",
                Some(scene_id.to_string()),
                Some(format!("/scenes/{scene_id}")),
                json!({}),
            );
        }
    }

    for (id, collection) in &doc.collections {
        let mut seen = BTreeSet::new();
        for child in &collection.children {
            if !doc.collections.contains_key(child) {
                issue(
                    &mut issues,
                    "error",
                    "MISSING_REFERENCE",
                    "Collection child does not exist.",
                    Some(id.to_string()),
                    Some(format!("/collections/{id}/children")),
                    json!({"target":child}),
                );
            }
            if !seen.insert(child) {
                issue(
                    &mut issues,
                    "error",
                    "DUPLICATE_REFERENCE",
                    "Collection child is referenced more than once.",
                    Some(id.to_string()),
                    Some(format!("/collections/{id}/children")),
                    json!({"target":child}),
                );
            }
        }
        for object in &collection.objects {
            if !doc.nodes.contains_key(object) {
                issue(
                    &mut issues,
                    "error",
                    "MISSING_REFERENCE",
                    "Collection object does not exist.",
                    Some(id.to_string()),
                    Some(format!("/collections/{id}/objects")),
                    json!({"target":object}),
                );
            }
        }
    }
    for (id, node) in &doc.nodes {
        if let Some(parent) = &node.parent
            && !doc.nodes.contains_key(parent)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Node parent does not exist.",
                Some(id.to_string()),
                Some(format!("/nodes/{id}/parent")),
                json!({"target":parent}),
            );
        }
        if let Some(data) = &node.data
            && !doc.data_blocks.contains_key(data)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Node Data-Block does not exist.",
                Some(id.to_string()),
                Some(format!("/nodes/{id}/data")),
                json!({"target":data}),
            );
        }
        for material in &node.materials {
            if !doc.materials.contains_key(material) {
                issue(
                    &mut issues,
                    "error",
                    "MISSING_REFERENCE",
                    "Node material slot references a missing Material.",
                    Some(id.to_string()),
                    Some(format!("/nodes/{id}/materials")),
                    json!({"target":material}),
                );
            }
        }
        if let Some(action) = &node.action
            && !doc.actions.contains_key(action)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Node Action does not exist.",
                Some(id.to_string()),
                Some(format!("/nodes/{id}/action")),
                json!({"target":action}),
            );
        }
        for (index, tag) in node.tags.iter().enumerate() {
            if !crate::model::is_valid_id(tag) {
                issue(
                    &mut issues,
                    "error",
                    "INVALID_ID",
                    "Node tag does not match the required ID pattern.",
                    Some(id.to_string()),
                    Some(format!("/nodes/{id}/tags/{index}")),
                    json!({"tag":tag}),
                );
            }
        }
        if let Some(data_id) = &node.data
            && let Some(data) = doc.data_blocks.get(data_id)
            && matches!(node.kind.as_str(), "mesh" | "camera" | "light")
            && data.data_type != node.kind
        {
            issue(
                &mut issues,
                "error",
                "REFERENCE_TYPE_MISMATCH",
                "Node kind and Data-Block type do not match.",
                Some(id.to_string()),
                Some(format!("/nodes/{id}/data")),
                json!({"data_id":data_id,"node_kind":node.kind,"data_type":data.data_type}),
            );
        }
        if !node
            .transform
            .translation
            .iter()
            .all(|value| value.is_finite())
            || !node.transform.scale.iter().all(|value| value.is_finite())
            || !node
                .transform
                .rotation
                .iter()
                .all(|value| value.is_finite())
            || node
                .parent_inverse
                .is_some_and(|matrix| matrix.iter().any(|value| !value.is_finite()))
        {
            issue(
                &mut issues,
                "error",
                "NON_FINITE",
                "Node transform contains a non-finite value.",
                Some(id.to_string()),
                Some(format!("/nodes/{id}/transform")),
                json!({}),
            );
        }
    }

    for (data_id, data) in &doc.data_blocks {
        let users = doc
            .nodes
            .values()
            .filter(|node| node.data.as_ref() == Some(data_id))
            .count();
        if users == 0 {
            issue(
                &mut issues,
                "information",
                "UNUSED_DATA_BLOCK",
                "Data-Block is not referenced by any Object and is retained.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}")),
                json!({}),
            );
        }
        if let Some(action) = data
            .shape_keys
            .as_ref()
            .and_then(|shape_keys| shape_keys.action.as_ref())
            && !doc.actions.contains_key(action)
        {
            issue(
                &mut issues,
                "error",
                "MISSING_REFERENCE",
                "Shape-key animation Action does not exist.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}/shape_keys/action")),
                json!({"target":action}),
            );
        }
        if let Some(mesh) = &data.mesh {
            inspect_mesh(mesh, data_id, &mut issues);
            for (node_id, node) in doc
                .nodes
                .iter()
                .filter(|(_, node)| node.data.as_ref() == Some(data_id))
            {
                if !node.materials.is_empty() {
                    for face in &mesh.faces {
                        if usize::try_from(face.material_index)
                            .map_or(true, |index| index >= node.materials.len())
                        {
                            issue(
                                &mut issues,
                                "error",
                                "INVALID_INDEX",
                                "Face material_index is outside the Object material slots.",
                                Some(data_id.to_string()),
                                Some(format!("/data_blocks/{data_id}/mesh/faces/{}", face.id)),
                                json!({"node_id":node_id,"material_index":face.material_index,"slot_count":node.materials.len()}),
                            );
                        }
                    }
                }
            }
        }
    }
    for (id, material) in &doc.materials {
        if !material.base_color.iter().all(|value| value.is_finite())
            || !material.metallic.is_finite()
            || !material.roughness.is_finite()
        {
            issue(
                &mut issues,
                "error",
                "NON_FINITE",
                "Material contains a non-finite value.",
                Some(id.to_string()),
                Some(format!("/materials/{id}")),
                json!({}),
            );
        }
    }
    for (id, world) in &doc.worlds {
        if !world.color.iter().all(|value| value.is_finite()) || !world.strength.is_finite() {
            issue(
                &mut issues,
                "error",
                "NON_FINITE",
                "World contains a non-finite value.",
                Some(id.to_string()),
                Some(format!("/worlds/{id}")),
                json!({}),
            );
        }
    }
    for (id, action) in &doc.actions {
        for curve in &action.fcurves {
            for (index, keyframe) in curve.keyframes.iter().enumerate() {
                if !keyframe.frame.is_finite() || !keyframe.value.is_finite() {
                    issue(
                        &mut issues,
                        "error",
                        "NON_FINITE",
                        "Keyframe contains a non-finite value.",
                        Some(id.to_string()),
                        Some(format!("/actions/{id}/fcurves/keyframes/{index}")),
                        json!({}),
                    );
                }
            }
        }
    }

    let parent_cycles = find_cycles(
        doc.nodes
            .iter()
            .map(|(id, node)| (id, node.parent.as_ref())),
    );
    for cycle in parent_cycles {
        issue(
            &mut issues,
            "error",
            "PARENT_CYCLE",
            "Object parenting contains a cycle.",
            cycle.first().cloned(),
            None,
            json!({"cycle":cycle}),
        );
    }
    for cycle in all_collection_cycles(doc) {
        issue(
            &mut issues,
            "error",
            "COLLECTION_CYCLE",
            "Collection hierarchy contains a cycle.",
            cycle.first().cloned(),
            None,
            json!({"cycle":cycle}),
        );
    }

    let context = crate::eval::EvaluationContext::default();
    match crate::eval::Snapshot::evaluate(doc, &context) {
        Ok(snapshot) => {
            for (id, evaluated) in snapshot.nodes {
                let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                if matrix.determinant().abs() <= f64::EPSILON {
                    issue(
                        &mut issues,
                        "warning",
                        "SINGULAR_TRANSFORM",
                        "Evaluated world transform is singular.",
                        Some(id.to_string()),
                        Some(format!("/nodes/{id}/transform")),
                        json!({}),
                    );
                }
            }
        }
        Err(error) if error.code == ErrorCode::SceneInvalid => {
            // Structural graph errors are already included above; retain an issue if evaluation finds another invalid reference.
            if issues.iter().all(|entry| entry["severity"] != "error") {
                issue(
                    &mut issues,
                    "error",
                    "EVALUATION_INVALID",
                    &error.message,
                    None,
                    None,
                    error.details,
                );
            }
        }
        Err(error) if error.code == ErrorCode::UnsupportedFeature => {
            issue(
                &mut issues,
                "error",
                "UNSUPPORTED_FEATURE",
                &error.message,
                None,
                None,
                error.details,
            );
        }
        Err(error) => issue(
            &mut issues,
            "error",
            "EVALUATION_FAILED",
            &error.message,
            None,
            None,
            error.details,
        ),
    }

    let mut losses = Vec::<Value>::new();
    if let Some(format) = format {
        let entry = catalog::formats_catalog()["formats"]
            .as_array()
            .and_then(|formats| formats.iter().find(|entry| entry["format"] == format))
            .cloned();
        if let Some(entry) = entry {
            let format_losses = if format == "blend" {
                crate::exchange::blend::export_losses(doc)
            } else {
                crate::commands::export::export_losses(doc, format)
            };
            let supported = entry["status"] == "supported";
            if !supported {
                issue(
                    &mut issues,
                    "error",
                    "FORMAT_NOT_SUPPORTED",
                    "Requested output format is not fully supported.",
                    None,
                    None,
                    json!({"format":format,"reason":entry["reason"]}),
                );
            }
            for loss in format_losses {
                issue(
                    &mut issues,
                    "error",
                    "UNREPRESENTABLE_FEATURE",
                    &loss.reason,
                    loss.data_id.clone(),
                    None,
                    json!({"feature_id":loss.feature_id,"suggestion":loss.suggestion}),
                );
                losses.push(json!({"feature_id":loss.feature_id,"data_id":loss.data_id,"reason":loss.reason,"suggestion":loss.suggestion}));
            }
            if !supported && losses.is_empty() {
                losses.push(json!({"feature_id":format!("exchange.format.{format}"),"data_id":Value::Null,"reason":entry["reason"],"suggestion":Value::Null}));
            }
            let loss_status = if supported && losses.is_empty() {
                "supported"
            } else {
                "not_supported"
            };
            checks.push(check("format_loss", loss_status, json!({"format":format,"loss_count":losses.len(),"loss_conditions":entry["loss_conditions"]})));
        } else {
            issue(
                &mut issues,
                "error",
                "FORMAT_UNKNOWN",
                "Requested output format is unknown.",
                None,
                None,
                json!({"format":format}),
            );
            checks.push(check(
                "format_loss",
                "not_supported",
                json!({"format":format,"reason":"unknown format"}),
            ));
        }
    }

    let has_errors = issues.iter().any(|entry| entry["severity"] == "error");
    let has_warnings = issues.iter().any(|entry| entry["severity"] == "warning");
    let compatibility = json!({
        "profile":doc.profile.blender,
        "status":if has_errors {"invalid"} else {"partial"},
        "checks_not_supported":checks.iter().filter(|entry| entry["status"] == "not_supported").count()
    });
    let mut counts = BTreeMap::new();
    for severity in ["error", "warning", "information"] {
        counts.insert(
            severity,
            issues
                .iter()
                .filter(|entry| entry["severity"] == severity)
                .count(),
        );
    }
    let issue_count = issues.len();
    let value = json!({"valid":!has_errors,"issues":issues,"checks":checks,"summary":{"issue_count":issue_count,"by_severity":counts,"scene_id":doc.scene_id,"revision":doc.revision},"compatibility":compatibility,"losses":losses});
    Ok(ValidationReport {
        value,
        has_warnings,
        has_errors,
    })
}

fn check(name: &str, status: &str, reason: Value) -> Value {
    let mut check = Map::new();
    check.insert("check".to_owned(), Value::String(name.to_owned()));
    check.insert("status".to_owned(), Value::String(status.to_owned()));
    check.insert("reason".to_owned(), reason);
    Value::Object(check)
}

fn issue(
    issues: &mut Vec<Value>,
    severity: &str,
    code: &str,
    message: &str,
    data_id: Option<String>,
    pointer: Option<String>,
    details: Value,
) {
    let mut entry = Map::new();
    entry.insert("severity".to_owned(), Value::String(severity.to_owned()));
    entry.insert("code".to_owned(), Value::String(code.to_owned()));
    entry.insert("message".to_owned(), Value::String(message.to_owned()));
    entry.insert(
        "data_id".to_owned(),
        data_id.map_or(Value::Null, Value::String),
    );
    entry.insert(
        "pointer".to_owned(),
        pointer.map_or(Value::Null, Value::String),
    );
    entry.insert("details".to_owned(), details);
    issues.push(Value::Object(entry));
}

fn inspect_mesh(mesh: &Mesh, data_id: &Id, issues: &mut Vec<Value>) {
    if let Err(error) = mesh.validate() {
        issue(
            issues,
            "error",
            "INVALID_MESH",
            &error.to_string(),
            Some(data_id.to_string()),
            Some(format!("/data_blocks/{data_id}/mesh")),
            json!({}),
        );
        return;
    }
    let mut edge_uses = BTreeMap::<(u32, u32), Vec<(u32, bool)>>::new();
    let by_id = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect::<BTreeMap<_, _>>();
    for Face { id, vertices, .. } in &mesh.faces {
        let mut normal = glam::DVec3::ZERO;
        for index in 0..vertices.len() {
            let Some(a) = by_id.get(&vertices[index]).copied() else {
                continue;
            };
            let Some(b) = by_id.get(&vertices[(index + 1) % vertices.len()]).copied() else {
                continue;
            };
            normal += a.cross(b);
            let (left, right) = (vertices[index], vertices[(index + 1) % vertices.len()]);
            let key = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            edge_uses.entry(key).or_default().push((*id, left > right));
        }
        if !normal.is_finite() || normal.length() <= f64::EPSILON {
            issue(
                issues,
                "warning",
                "ZERO_AREA_FACE",
                "Face has zero geometric area.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}/mesh/faces/{id}")),
                json!({"face_id":id}),
            );
        }
    }
    for (edge, uses) in edge_uses {
        match uses.as_slice() {
            [only] => issue(
                issues,
                "warning",
                "OPEN_BOUNDARY",
                "Mesh edge has only one incident face.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}/mesh")),
                json!({"edge":edge,"face_id":only.0}),
            ),
            [first, second] if first.1 == second.1 => issue(
                issues,
                "warning",
                "WINDING_INCONSISTENT",
                "Adjacent faces traverse a shared edge in the same direction.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}/mesh")),
                json!({"edge":edge,"faces":[first.0,second.0]}),
            ),
            uses if uses.len() > 2 => issue(
                issues,
                "warning",
                "NON_MANIFOLD_EDGE",
                "Mesh edge is incident to more than two faces.",
                Some(data_id.to_string()),
                Some(format!("/data_blocks/{data_id}/mesh")),
                json!({"edge":edge,"faces":uses.iter().map(|use_| use_.0).collect::<Vec<_>>()}),
            ),
            _ => {}
        }
    }
}

fn find_cycles<'a>(edges: impl Iterator<Item = (&'a Id, Option<&'a Id>)>) -> Vec<Vec<String>> {
    let graph = edges
        .map(|(id, parent)| (id.to_string(), parent.map(ToString::to_string)))
        .collect::<BTreeMap<_, _>>();
    let mut done = BTreeSet::new();
    let mut cycles = Vec::new();
    for start in graph.keys() {
        let mut path = Vec::new();
        let mut positions = BTreeMap::new();
        let mut current = Some(start.as_str());
        while let Some(node) = current {
            if let Some(index) = positions.get(node).copied() {
                let mut cycle = path[index..].to_vec();
                cycle.push(node.to_owned());
                cycles.push(cycle);
                break;
            }
            if done.contains(node) {
                break;
            }
            positions.insert(node.to_owned(), path.len());
            path.push(node.to_owned());
            current = graph.get(node).and_then(Option::as_deref);
        }
        done.extend(path);
    }
    cycles
}

fn all_collection_cycles(doc: &SceneDoc) -> Vec<Vec<String>> {
    fn visit(
        id: &Id,
        doc: &SceneDoc,
        active: &mut Vec<String>,
        done: &mut BTreeSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        let text = id.to_string();
        if let Some(index) = active.iter().position(|item| item == &text) {
            let mut cycle = active[index..].to_vec();
            cycle.push(text);
            cycles.push(cycle);
            return;
        }
        if done.contains(&text) {
            return;
        }
        active.push(text.clone());
        if let Some(collection) = doc.collections.get(id) {
            for child in &collection.children {
                visit(child, doc, active, done, cycles);
            }
        }
        active.pop();
        done.insert(text);
    }
    let mut done = BTreeSet::new();
    let mut cycles = Vec::new();
    for id in doc.collections.keys() {
        visit(id, doc, &mut Vec::new(), &mut done, &mut cycles);
    }
    cycles
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::DVec3;

    use super::validate;
    use crate::{
        geom::Mesh,
        model::{DataBlock, Id, SceneDoc},
    };

    #[test]
    fn reports_a_zero_area_face_as_a_warning() {
        let mut doc = SceneDoc::new("1e1c61e9-7df9-4ff9-8ed5-39055213a5e5".to_owned());
        let mesh_id = Id::new("mesh_data").unwrap();
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::X * 2.0],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        doc.data_blocks.insert(
            mesh_id,
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        let report = validate(&doc, None).unwrap();
        assert!(report.value["valid"].as_bool().unwrap());
        assert!(report.value["issues"].as_array().is_some_and(|issues| {
            issues
                .iter()
                .any(|item| item["code"] == "ZERO_AREA_FACE" && item["severity"] == "warning")
        }));
    }
}
