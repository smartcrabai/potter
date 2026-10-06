use std::collections::{BTreeMap, HashMap};

use glam::{DMat4, DQuat};
use serde_json::{Value, json};

use crate::{
    cli::InspectArgs,
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot, animation::animated_transform},
    model::{Collection, DataBlock, Id, Material, Node, Scene, SceneDoc, World},
    response::SceneInfo,
    store::Project,
};

use super::util::parse_id;
pub fn run(args: InspectArgs) -> Result<(Option<SceneInfo>, Value)> {
    let project = Project::open(args.scene)?;
    let doc = project.doc();
    let context = evaluation_context(
        args.context.scene_id.as_deref(),
        args.context.view_layer.as_deref(),
        args.context.frame,
    )?;
    let (snapshot, node_errors) =
        Snapshot::evaluate_available_with_cache(doc, &context, Some(project.path()))?;
    let driver_values = crate::eval::rig::evaluate_drivers(doc, snapshot.frame)?;
    let items = if let Some(value) = args.id.as_deref() {
        let id = parse_id(value)?;
        if let Some(error) = node_errors.get(&id) {
            return Err(error.clone());
        }
        vec![inspect_id(doc, &snapshot, &id, &driver_values)?]
    } else if let Some(tag) = args.tag.as_deref() {
        let tag_id = Id::new(tag).map_err(|error| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                error.message,
                json!({ "tag": tag }),
            )
        })?;
        let matching = doc
            .nodes
            .iter()
            .filter(|(_, node)| node.tags.contains(&tag_id.to_string()))
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::TargetNotFound,
                "no objects match tag",
                json!({ "tag": tag }),
            ));
        }
        if let Some(error) = matching.iter().find_map(|(id, _)| node_errors.get(*id)) {
            return Err(error.clone());
        }
        matching
            .into_iter()
            .map(|(id, node)| {
                node_errors.get(id).map_or_else(
                    || node_item(doc, &snapshot, id, node, &driver_values),
                    |error| Ok(node_error_item(id, node, error)),
                )
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        doc.nodes
            .iter()
            .map(|(id, node)| {
                node_errors.get(id).map_or_else(
                    || node_item(doc, &snapshot, id, node, &driver_values),
                    |error| Ok(node_error_item(id, node, error)),
                )
            })
            .chain(
                doc.worlds
                    .iter()
                    .map(|(id, world)| Ok(world_item(doc, id, world))),
            )
            .collect::<Result<Vec<_>>>()?
    };
    let features = if args.features {
        crate::catalog::feature_catalog()["features"].clone()
    } else {
        json!([])
    };
    let summary = summary(doc);
    let scene = project.info()?;
    Ok((
        Some(scene),
        json!({
            "summary": summary,
            "items": items,
            "libraries": doc.libraries,
            "references": {},
            "resources": [],
            "features": features,
            "evaluation": {
                "scene_id": snapshot.scene_id,
                "view_layer": snapshot.view_layer,
                "frame": snapshot.frame,
                "evaluation_hash": snapshot.evaluation_hash,
            },
        }),
    ))
}

fn node_error_item(id: &Id, node: &Node, error: &PotError) -> Value {
    json!({
        "id": id,
        "name": node.name,
        "kind": node.kind,
        "evaluation_error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
    })
}

fn evaluation_context(
    scene_id: Option<&str>,
    view_layer: Option<&str>,
    frame: Option<f64>,
) -> Result<EvaluationContext> {
    let scene_id = scene_id.map(parse_id).transpose()?;
    let view_layer = view_layer.map(parse_id).transpose()?;
    if frame.is_some_and(|value| !value.is_finite()) {
        return Err(PotError::invalid_argument("frame must be finite"));
    }
    Ok(EvaluationContext {
        scene_id,
        view_layer,
        frame,
    })
}

#[derive(Debug)]
struct WorldTransform {
    matrix: [f64; 16],
    decomposable: bool,
    translation: Option<[f64; 3]>,
    rotation: Option<[f64; 4]>,
    scale: Option<[f64; 3]>,
}

impl WorldTransform {
    fn from_matrix(matrix: [f64; 16]) -> Self {
        let mut result = Self {
            matrix,
            decomposable: false,
            translation: None,
            rotation: None,
            scale: None,
        };
        if matrix.iter().any(|component| !component.is_finite()) {
            return result;
        }

        let world = DMat4::from_cols_array(&matrix);
        let determinant = world.determinant();
        if determinant == 0.0 || determinant.is_nan() {
            return result;
        }
        let (scale, rotation, translation) = world.to_scale_rotation_translation();
        let rotation_length = rotation.length();
        if !scale.is_finite()
            || !translation.is_finite()
            || !rotation.is_finite()
            || rotation_length == 0.0
            || !rotation_length.is_finite()
        {
            return result;
        }

        let rotation = rotation / rotation_length;
        let quaternion =
            crate::model::canonicalize_quaternion([rotation.x, rotation.y, rotation.z, rotation.w]);
        let normalized_rotation =
            DQuat::from_xyzw(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        let recomposed =
            DMat4::from_scale_rotation_translation(scale, normalized_rotation, translation)
                .to_cols_array();
        if matrix
            .iter()
            .zip(recomposed)
            .any(|(actual, expected)| (actual - expected).abs() > 1.0e-9 * actual.abs().max(1.0))
        {
            return result;
        }

        result.decomposable = true;
        result.translation = Some(translation.to_array());
        result.rotation = Some(quaternion);
        result.scale = Some(scale.to_array());
        result
    }

    fn to_json(&self) -> Value {
        json!({
            "matrix": self.matrix,
            "decomposable": self.decomposable,
            "translation": self.translation,
            "rotation": self.rotation,
            "scale": self.scale,
        })
    }
}

fn inspect_id(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    id: &Id,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<Value> {
    if let Some(node) = doc.nodes.get(id) {
        return node_item(doc, snapshot, id, node, driver_values);
    }
    if let Some(data) = doc.data_blocks.get(id) {
        return Ok(data_item(doc, id, data));
    }
    if let Some(material) = doc.materials.get(id) {
        return Ok(material_item(doc, id, material));
    }
    if let Some(collection) = doc.collections.get(id) {
        return Ok(collection_item(doc, id, collection));
    }
    if let Some(scene) = doc.scenes.get(id) {
        return Ok(scene_item(doc, id, scene));
    }
    if let Some(world) = doc.worlds.get(id) {
        return Ok(world_item(doc, id, world));
    }
    if let Some(library) = doc.libraries.get(id) {
        return Ok(library_item(id, library));
    }
    Err(PotError::with_details(
        ErrorCode::TargetNotFound,
        "ID not found",
        json!({ "id": id }),
    ))
}
fn library_item(id: &Id, library: &crate::model::Library) -> Value {
    json!({
        "type":"library",
        "id":id,
        "name":library.name,
        "kind":library.kind,
        "uri":library.uri,
        "resolved_path":library.resolved_path,
        "resource":library.resource,
        "hash":library.hash,
        "status":library.status,
        "linked_ids":library.linked_ids,
        "overrides":library.overrides
    })
}

fn linked_item_fields(doc: &SceneDoc, registry: &str, id: &Id) -> Value {
    let marker = doc
        .compatibility
        .get("linked_ids")
        .and_then(Value::as_object)
        .and_then(|linked| linked.get(&format!("{registry}:{id}")));
    let Some(marker) = marker else {
        return json!({"library":null,"library_name":null,"editable":true});
    };
    let library_id = marker.get("library_id").cloned().unwrap_or(Value::Null);
    let library = library_id
        .as_str()
        .and_then(|library_id| Id::new(library_id.to_owned()).ok())
        .and_then(|library_id| doc.libraries.get(&library_id));
    let name = marker.get("library_name").cloned().unwrap_or(Value::Null);
    json!({
        "library":library.map(|library| json!({
            "id":library_id,
            "name":library.name,
            "kind":library.kind,
            "uri":library.uri
        })),
        "library_name":name,
        "editable":false
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "one inspected node is assembled as a stable consumer-facing record"
)]
fn node_item(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    id: &Id,
    node: &Node,
    driver_values: &BTreeMap<(Id, String, u32), f64>,
) -> Result<Value> {
    let evaluated = snapshot.nodes.get(id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "node evaluation is missing",
            json!({ "id": id }),
        )
    })?;
    let users = node
        .data
        .as_ref()
        .map(|data_id| {
            doc.nodes
                .iter()
                .filter(|(_, candidate)| candidate.data.as_ref() == Some(data_id))
                .map(|(user_id, _)| user_id.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let params = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.descriptor.as_ref())
        .map_or(Value::Null, |descriptor| {
            Value::Object(descriptor.params.clone())
        });
    let collections = doc
        .collections
        .iter()
        .filter(|(_, collection)| collection.objects.contains(id))
        .map(|(collection_id, _)| collection_id.to_string())
        .collect::<Vec<_>>();
    let children = doc
        .nodes
        .iter()
        .filter(|(_, candidate)| candidate.parent.as_ref() == Some(id))
        .map(|(child_id, _)| child_id.to_string())
        .collect::<Vec<_>>();
    let animated_transform = animated_transform(node, doc, snapshot.frame)?;
    let evaluated_geometry = snapshot.meshes.get(id).map_or(Value::Null, |mesh| {
        let vertex_indices: HashMap<_, _> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        json!({
            "vertex_count": mesh.vertices.len(),
            "edge_count": mesh.edges.len(),
            "face_count": mesh.faces.len(),
            "triangle_count": mesh.faces.iter().map(|face| face.vertices.len().saturating_sub(2)).sum::<usize>(),
            "positions": mesh.vertices.iter().map(|vertex| vertex.co.to_array()).collect::<Vec<_>>(),
            "faces": mesh.faces.iter().map(|face| face.vertices.iter().map(|id| vertex_indices.get(id).copied()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        })
    });
    let camera_data = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.camera.as_ref());
    let light_data = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.light.as_ref());
    let evaluated_shape_keys = if let Some(shape_keys) = node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id))
        .and_then(|data| data.shape_keys.as_ref())
    {
        crate::eval::rig::evaluate_shape_key_values(
            id,
            node,
            shape_keys,
            doc,
            snapshot.frame,
            driver_values,
        )?
    } else {
        BTreeMap::new()
    };
    let linkage = linked_item_fields(doc, "nodes", id);
    let world_transform = WorldTransform::from_matrix(evaluated.world_matrix);
    let local_matrix = animated_transform.matrix().to_cols_array();

    Ok(json!({
        "type": "node",
        "id": id,
        "name": node.name,
        "kind": node.kind,
        "primitive": node.primitive,
        "params": params,
        "tags": node.tags,
        "parent": node.parent,
        "parent_type": node.parent_type,
        "parent_bone": node.parent_bone,
        "parent_inverse": node.parent_inverse,
        "children": children,
        "collections": collections,
        "data": node.data,
        "camera": camera_data,
        "light": light_data,
        "action": node.action,
        "nla_tracks": node.nla_tracks,
        "pose": node.pose,
        "constraints": node.constraints,
        "drivers": node.drivers,
        "evaluated_bones": snapshot.bone_matrices.get(id),
        "evaluated_shape_keys": evaluated_shape_keys,
        "rigid_body": node.rigid_body,
        "force_field": node.force_field,
        "sharing": { "users": users, "count": users.len(), "shared": users.len() > 1 },
        "materials": node.materials,
        "modifiers": node.modifiers,
        "visible": node.visible,
        "render_visible": node.render_visible,
        "selectable": node.selectable,
        "properties": node.properties,
        "library": linkage["library"],
        "library_name": linkage["library_name"],
        "editable": linkage["editable"],
        "library_override": node.properties.get("library_override"),
        "transform": {
            "evaluated": {
                "translation": animated_transform.translation,
                "rotation": animated_transform.rotation,
                "scale": animated_transform.scale,
                "matrix": local_matrix,
            },
            "world": world_transform.to_json(),
        },


        "bounds": evaluated.bounds,
        "dimensions": evaluated.dimensions,
        "evaluated_geometry": evaluated_geometry,
    }))
}

fn world_item(doc: &SceneDoc, id: &Id, world: &World) -> Value {
    let linkage = linked_item_fields(doc, "worlds", id);
    json!({
        "type":"world",
        "id":id,
        "color":world.color,
        "strength":world.strength,
        "library":linkage["library"],
        "library_name":linkage["library_name"],
        "editable":linkage["editable"]
    })
}

fn data_item(doc: &SceneDoc, id: &Id, data: &DataBlock) -> Value {
    let users = doc
        .nodes
        .iter()
        .filter(|(_, node)| node.data.as_ref() == Some(id))
        .map(|(node_id, _)| node_id.to_string())
        .collect::<Vec<_>>();
    let vertices = data.mesh.as_ref().map_or(0, |mesh| mesh.vertices.len());
    let edges = data.mesh.as_ref().map_or(0, |mesh| mesh.edges.len());
    let faces = data.mesh.as_ref().map_or(0, |mesh| mesh.faces.len());
    let triangles = data.mesh.as_ref().map_or(0, |mesh| {
        mesh.faces
            .iter()
            .map(|face| face.vertices.len().saturating_sub(2))
            .sum::<usize>()
    });
    let linkage = linked_item_fields(doc, "data_blocks", id);
    json!({
        "type": "data_block",
        "id": id,
        "library": linkage["library"],
        "library_name": linkage["library_name"],
        "editable": linkage["editable"],
        "data_type": data.data_type,
        "descriptor": data.descriptor,
        "camera": data.camera,
        "light": data.light,
        "armature": data.armature,
        "shape_keys": data.shape_keys,
        "vertex_groups": data.vertex_groups,
        "vertex_weights": data.vertex_weights,
        "grease_pencil": data.grease_pencil,
        "curve": data.curve.as_ref().map(|curve| json!({
            "dimensions": curve.dimensions,
            "spline_count": curve.splines.len(),
            "spline_types": curve.splines.iter().map(|spline| spline.spline_type).collect::<Vec<_>>(),
            "point_counts": curve.splines.iter().map(|spline| spline.points.len()).collect::<Vec<_>>(),
        })),
        "surface": data.surface.as_ref().map(|surface| json!({
            "grid": [surface.points.len(), surface.points.first().map_or(0, Vec::len)],
            "order": [surface.order_u, surface.order_v],
        })),
        "text": data.text,
        "hair_curves": data.hair_curves.as_ref().map(|hair| json!({"curve_count":hair.curves.len()})),
        "users": users,
        "sharing": { "count": users.len(), "shared": users.len() > 1 },
        "geometry": { "vertices": vertices, "edges": edges, "faces": faces, "triangles": triangles },
    })
}

fn material_item(doc: &SceneDoc, id: &Id, material: &Material) -> Value {
    let linkage = linked_item_fields(doc, "materials", id);
    json!({
        "type":"material",
        "id":id,
        "name":material.name,
        "base_color":material.base_color,
        "metallic":material.metallic,
        "roughness":material.roughness,
        "double_sided":material.double_sided,
        "library":linkage["library"],
        "library_name":linkage["library_name"],
        "editable":linkage["editable"]
    })
}

fn collection_item(doc: &SceneDoc, id: &Id, collection: &Collection) -> Value {
    let linkage = linked_item_fields(doc, "collections", id);
    json!({
        "type":"collection",
        "id":id,
        "name":collection.name,
        "children":collection.children,
        "objects":collection.objects,
        "library":linkage["library"],
        "library_name":linkage["library_name"],
        "editable":linkage["editable"]
    })
}

fn scene_item(doc: &SceneDoc, id: &Id, scene: &Scene) -> Value {
    let world_data = scene
        .world
        .as_ref()
        .and_then(|world_id| doc.worlds.get(world_id));
    json!({ "type": "scene", "id": id, "name": scene.name, "root_collection": scene.root_collection,
        "view_layers": scene.view_layers.keys().collect::<Vec<_>>(), "frame_current": scene.frame_current,
        "frame_start": scene.frame_start, "frame_end": scene.frame_end, "fps": scene.fps, "fps_base": scene.fps_base,
        "camera": scene.camera, "world": scene.world, "world_data": world_data, "unit": scene.unit,
        "render": scene.render, "markers": scene.markers, "rigid_body_world": scene.rigid_body_world })
}

fn summary(doc: &SceneDoc) -> Value {
    json!({
        "scene_id": doc.scene_id,
        "active_scene": doc.active_scene,
        "revision": doc.revision,
        "counts": {
            "scenes": doc.scenes.len(),
            "collections": doc.collections.len(),
            "nodes": doc.nodes.len(),
            "data_blocks": doc.data_blocks.len(),
            "materials": doc.materials.len(),
            "node_groups": doc.node_groups.len(),
            "worlds": doc.worlds.len(),
            "actions": doc.actions.len(),
            "resources": doc.resources.len(),
            "libraries": doc.libraries.len(),
        },
    })
}
#[cfg(test)]
mod tests {
    use glam::{DMat4, DQuat, DVec3, EulerRot};
    use proptest::prelude::*;

    use super::WorldTransform;
    use crate::model::canonicalize_quaternion;

    proptest! {
        #[test]
        fn positive_trs_world_matrices_decompose_back_to_canonical_trs(
            translation in prop::array::uniform3(-1.0e4_f64..1.0e4),
            angles in prop::array::uniform3(-3.0_f64..3.0),
            scale in prop::array::uniform3(0.01_f64..100.0),
        ) {
            let rotation = DQuat::from_euler(EulerRot::XYZ, angles[0], angles[1], angles[2]);
            let matrix = DMat4::from_scale_rotation_translation(
                DVec3::from_array(scale),
                rotation,
                DVec3::from_array(translation),
            ).to_cols_array();
            let decomposed = WorldTransform::from_matrix(matrix);

            prop_assert!(decomposed.decomposable);
            let actual_translation = decomposed.translation.unwrap_or([f64::NAN; 3]);
            let actual_scale = decomposed.scale.unwrap_or([f64::NAN; 3]);
            let actual_rotation = decomposed.rotation.unwrap_or([f64::NAN; 4]);

            let expected_rotation = canonicalize_quaternion([rotation.x, rotation.y, rotation.z, rotation.w]);
            for axis in 0..3 {
                prop_assert!((actual_translation[axis] - translation[axis]).abs() <= 1.0e-9 * translation[axis].abs().max(1.0));
                prop_assert!((actual_scale[axis] - scale[axis]).abs() <= 1.0e-9 * scale[axis].abs().max(1.0));
            }
            for component in 0..4 {
                prop_assert!((actual_rotation[component] - expected_rotation[component]).abs() <= 1.0e-9);
            }
            prop_assert!(actual_rotation[3] >= 0.0);
        }
    }
    #[test]
    fn singular_world_matrix_is_not_decomposable() {
        let matrix = DMat4::from_scale(DVec3::new(1.0, 0.0, 1.0)).to_cols_array();
        let decomposed = WorldTransform::from_matrix(matrix);

        assert!(!decomposed.decomposable);
        assert!(decomposed.translation.is_none());
        assert!(decomposed.rotation.is_none());
        assert!(decomposed.scale.is_none());
        assert_eq!(decomposed.matrix, matrix);
    }
}
