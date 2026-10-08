mod animation;
mod asset;
mod camera;
mod collection;
mod color;
mod compositor;
mod curve;
mod extension;
mod geometry_data;
mod graph;
mod grease_pencil;
mod image;
mod lattice;
mod library;
mod mask;
mod material;
mod mesh;
mod mesh_separate;
mod metaball;
mod modifier;
mod nla;
mod node;
mod node_join;
mod paint;
mod physics;
mod pointcloud;
mod property;
mod render;
mod resource;
mod rig;
mod scene;
mod sculpt;
mod sequencer;
mod shader;
mod text;
mod tracking;
mod uv;
mod volume;

/// Every operation accepted by the batch dispatcher, in stable sorted order.
pub const OP_NAMES: &[&str] = &[
    "action.create",
    "action.delete",
    "action.slot_create",
    "action.update",
    "asset.catalog_create",
    "asset.clear",
    "asset.mark",
    "asset.update",
    "bone.create",
    "bone.delete",
    "bone.update",
    "bone_collection.assign",
    "bone_collection.create",
    "bone_collection.delete",
    "bone_collection.unassign",
    "bone_collection.update",
    "camera.create",
    "camera.update",
    "collection.create",
    "collection.delete",
    "collection.instance_create",
    "collection.link",
    "collection.parent",
    "collection.unlink",
    "collection.update",
    "color.update",
    "compositor.enable",
    "compositor.link",
    "compositor.node_add",
    "compositor.node_remove",
    "compositor.node_update",
    "constraint.create",
    "constraint.delete",
    "constraint.reorder",
    "constraint.update",
    "curve.convert",
    "curve.create",
    "curve.point_add",
    "curve.point_delete",
    "curve.point_update",
    "curve.update",
    "data.make_single_user",
    "driver.create",
    "driver.delete",
    "driver.update",
    "extension.register",
    "fcurve.update",
    "graph.create",
    "graph.delete",
    "graph.interface_update",
    "graph.link",
    "graph.node_add",
    "graph.node_remove",
    "graph.node_update",
    "graph.unlink",
    "grease_pencil.frame_add",
    "grease_pencil.layer_create",
    "grease_pencil.layer_delete",
    "grease_pencil.layer_update",
    "grease_pencil.stroke_add",
    "grease_pencil.stroke_delete",
    "grease_pencil.stroke_update",
    "image.create",
    "image.delete",
    "image.load",
    "image.set_pixels",
    "image.update",
    "keyframe.delete",
    "keyframe.insert",
    "lattice.create",
    "lattice.update",
    "library.append",
    "library.link",
    "library.override",
    "library.register",
    "library.reload",
    "library.relocate",
    "mask.create",
    "mask.delete",
    "mask.update",
    "light.create",
    "light.update",
    "material.create",
    "material.delete",
    "material.update",
    "metaball.create",
    "metaball.update",
    "mesh.attribute_create",
    "mesh.attribute_delete",
    "mesh.attribute_update",
    "mesh.auto_smooth",
    "mesh.bevel",
    "mesh.bisect",
    "mesh.bridge",
    "mesh.delete",
    "mesh.dissolve",
    "mesh.edge_slide",
    "mesh.extrude",
    "mesh.fill",
    "mesh.flip_normals",
    "mesh.inset",
    "mesh.knife",
    "mesh.loop_cut",
    "mesh.mark_sharp_by_angle",
    "mesh.merge",
    "mesh.mirror",
    "mesh.poke",
    "mesh.remesh",
    "mesh.separate",
    "mesh.rip",
    "mesh.screw",
    "mesh.set_attribute",
    "mesh.set_custom_normals",
    "mesh.split",
    "mesh.shade_flat",
    "mesh.shade_smooth",
    "mesh.spin",
    "mesh.subdivide",
    "mesh.symmetrize",
    "mesh.transform_elements",
    "mesh.triangulate",
    "mesh.vertex_slide",
    "mesh.weld",
    "modifier.apply_as_shape_key",
    "modifier.apply",
    "modifier.bind",
    "modifier.create",
    "modifier.delete",
    "modifier.reorder",
    "modifier.unbind",
    "modifier.update",
    "nla.push_down",
    "nla.strip_create",
    "nla.strip_delete",
    "nla.strip_update",
    "nla.track_create",
    "nla.track_delete",
    "nla.track_update",
    "node.create",
    "node.delete",
    "node.duplicate",
    "node.join",
    "node.parent",
    "node.update",
    "paint.texture",
    "paint.vertex",
    "paint.weight",
    "pointcloud.create",
    "pointcloud.update",
    "physics.cloth.create",
    "physics.cloth.delete",
    "physics.cloth.update",
    "physics.collision.create",
    "physics.collision.delete",
    "physics.collision.update",
    "physics.dynamic_paint.create",
    "physics.dynamic_paint.delete",
    "physics.dynamic_paint.update",
    "physics.fluid.create",
    "physics.fluid.delete",
    "physics.fluid.update",
    "physics.force_field.create",
    "physics.force_field.delete",
    "physics.force_field.update",
    "physics.particle_emitter.create",
    "physics.particle_emitter.delete",
    "physics.particle_emitter.update",
    "physics.rigid_body.create",
    "physics.rigid_body.delete",
    "physics.rigid_body.update",
    "physics.soft_body.create",
    "physics.soft_body.delete",
    "physics.soft_body.update",
    "physics.world.update",
    "pose.reset",
    "pose.set",
    "property.delete",
    "property.set",
    "render.passes_update",
    "render.update",
    "resource.pack",
    "resource.unpack",
    "rig.generate_basic_human",
    "scene.create",
    "scene.marker_add",
    "scene.marker_remove",
    "scene.update",
    "sculpt.dyntopo",
    "sculpt.face_set",
    "sculpt.mask",
    "sculpt.multires",
    "sculpt.remesh_voxel",
    "sculpt.stroke",
    "sequencer.strip_create",
    "sequencer.strip_delete",
    "sequencer.strip_update",
    "sequencer.update",
    "shape_key.create",
    "shape_key.delete",
    "shape_key.update",
    "simulation.settings.update",
    "surface.create",
    "text.create",
    "text.delete",
    "text.update",
    "text_object.create",
    "text_object.update",
    "tracking.clip_create",
    "tracking.plane_track",
    "tracking.set_camera_intrinsics",
    "tracking.solve_camera",
    "tracking.solve_object",
    "tracking.track",
    "tracking.track_add",
    "tracking.undistort",
    "uv.pack",
    "uv.pin",
    "uv.transform",
    "uv.unwrap",
    "vertex_group.assign",
    "vertex_group.create",
    "vertex_group.remove",
    "volume.create",
    "volume.update",
    "world.create",
    "world.update",
];
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::{Id, SceneDoc},
};

#[derive(Debug, Clone)]
pub struct ApplyOutcome {
    pub doc: SceneDoc,
    pub changes: Value,
    pub operations: Value,
    pub id_mappings: Value,
    pub asset_blobs: BTreeMap<String, Vec<u8>>,
    pub changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Created,
    Updated,
    Deleted,
}

#[derive(Debug, Default)]
struct ChangeSet {
    registries: BTreeMap<String, BTreeMap<String, ChangeKind>>,
}

impl ChangeSet {
    fn record(&mut self, registry: &str, id: &Id, kind: ChangeKind) {
        let changes = self.registries.entry(registry.to_owned()).or_default();
        let previous = changes.get(id.as_str()).copied();
        let next = match (previous, kind) {
            (Some(ChangeKind::Created), ChangeKind::Updated) => Some(ChangeKind::Created),
            (Some(ChangeKind::Created), ChangeKind::Deleted) => None,
            (Some(ChangeKind::Deleted), ChangeKind::Created) => Some(ChangeKind::Updated),
            (_, value) => Some(value),
        };
        if let Some(next) = next {
            changes.insert(id.to_string(), next);
        } else {
            changes.remove(id.as_str());
        }
    }

    fn reconcile(&mut self, original: &SceneDoc, changed: &SceneDoc) -> Result<()> {
        let entries: Vec<_> = self
            .registries
            .iter()
            .flat_map(|(registry, changes)| {
                changes
                    .keys()
                    .cloned()
                    .map(|id| (registry.clone(), id))
                    .collect::<Vec<_>>()
            })
            .collect();
        for (registry, id_text) in entries {
            let id = Id::new(id_text.clone())
                .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?;
            let before = registry_entry(original, &registry, &id)?;
            let after = registry_entry(changed, &registry, &id)?;
            let kind = match (before, after) {
                (None, Some(_)) => Some(ChangeKind::Created),
                (Some(_), None) => Some(ChangeKind::Deleted),
                (Some(before), Some(after)) if before != after => Some(ChangeKind::Updated),
                _ => None,
            };
            if let Some(changes) = self.registries.get_mut(&registry) {
                if let Some(kind) = kind {
                    changes.insert(id_text.clone(), kind);
                } else {
                    changes.remove(&id_text);
                }
            }
        }
        Ok(())
    }

    fn to_value(&self) -> Value {
        let mut registries = Map::new();
        for registry in [
            "scenes",
            "collections",
            "nodes",
            "data_blocks",
            "images",
            "materials",
            "movie_clips",
            "masks",
            "node_groups",
            "actions",
            "resources",
            "compatibility",
        ] {
            let mut created = Vec::new();
            let mut updated = Vec::new();
            let mut deleted = Vec::new();
            if let Some(changes) = self.registries.get(registry) {
                for (id, kind) in changes {
                    match kind {
                        ChangeKind::Created => created.push(id.as_str()),
                        ChangeKind::Updated => updated.push(id.as_str()),
                        ChangeKind::Deleted => deleted.push(id.as_str()),
                    }
                }
            }
            registries.insert(
                registry.to_owned(),
                json!({"created": created, "updated": updated, "deleted": deleted}),
            );
        }
        Value::Object(registries)
    }

    fn is_empty(&self) -> bool {
        self.registries.values().all(BTreeMap::is_empty)
    }
}
fn registry_entry(doc: &SceneDoc, registry: &str, id: &Id) -> Result<Option<Value>> {
    match registry {
        "scenes" => serialized(doc.scenes.get(id)),
        "collections" => serialized(doc.collections.get(id)),
        "nodes" => serialized(doc.nodes.get(id)),
        "data_blocks" => serialized(doc.data_blocks.get(id)),
        "materials" => serialized(doc.materials.get(id)),
        "images" => serialized(doc.images.get(id)),
        "movie_clips" => serialized(doc.movie_clips.get(id)),
        "actions" => serialized(doc.actions.get(id)),
        "masks" => serialized(doc.masks.get(id)),
        "node_groups" => serialized(doc.node_groups.get(id)),
        "worlds" => serialized(doc.worlds.get(id)),
        "libraries" => serialized(doc.libraries.get(id)),
        "resources" => serialized(doc.resources.get(id)),
        "compatibility" => Ok(doc.compatibility.get(id.as_str()).cloned()),
        _ => Err(PotError::new(
            ErrorCode::InternalError,
            format!("unknown change registry `{registry}`"),
        )),
    }
}

fn serialized<T: serde::Serialize>(value: Option<&T>) -> Result<Option<Value>> {
    value
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
}

struct Engine<'a> {
    doc: &'a mut SceneDoc,
    changes: ChangeSet,
    id_mappings: Map<String, Value>,
    operation_index: usize,
    evaluation_frame: f64,
    asset_root: Option<&'a Path>,
    pending_assets: BTreeMap<String, Vec<u8>>,
}

impl Engine<'_> {
    fn error(&self, code: ErrorCode, message: impl Into<String>, pointer: &str) -> PotError {
        PotError::with_details(
            code,
            message,
            json!({
                "operation_index": self.operation_index,
                "pointer": pointer,
            }),
        )
    }

    fn mark(&mut self, registry: &str, id: &Id, kind: ChangeKind) {
        self.changes.record(registry, id, kind);
    }

    fn map_id(&mut self, registry: &str, old: &Id, new: &Id) {
        let map = self
            .id_mappings
            .entry(registry.to_owned())
            .or_insert_with(|| json!({}));
        if let Some(values) = map.as_object_mut() {
            match values.get(old.as_str()).cloned() {
                None => {
                    values.insert(old.to_string(), Value::String(new.to_string()));
                }
                Some(Value::String(existing)) if existing != new.as_str() => {
                    values.insert(
                        old.to_string(),
                        Value::Array(vec![
                            Value::String(existing),
                            Value::String(new.to_string()),
                        ]),
                    );
                }
                Some(Value::Array(mut existing))
                    if !existing
                        .iter()
                        .any(|value| value.as_str() == Some(new.as_str())) =>
                {
                    existing.push(Value::String(new.to_string()));
                    values.insert(old.to_string(), Value::Array(existing));
                }
                _ => {}
            }
        }
    }
}

pub fn apply_batch(doc: &SceneDoc, batch: &Value) -> Result<ApplyOutcome> {
    apply_batch_internal(doc, batch, None)
}

pub fn apply_batch_with_asset_root(
    doc: &SceneDoc,
    batch: &Value,
    asset_root: &Path,
) -> Result<ApplyOutcome> {
    apply_batch_internal(doc, batch, Some(asset_root))
}

fn apply_batch_internal(
    doc: &SceneDoc,
    batch: &Value,
    asset_root: Option<&Path>,
) -> Result<ApplyOutcome> {
    let batch_object = batch.as_object().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::InvalidOperation,
            "operation batch must be an object",
            json!({"pointer":""}),
        )
    })?;
    for key in batch_object.keys() {
        if ![
            "schema_version",
            "base_revision",
            "operations",
            "evaluation",
        ]
        .contains(&key.as_str())
        {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("unknown batch field `{key}`"),
                json!({"pointer":format!("/{}", pointer_escape(key))}),
            ));
        }
    }
    let schema_version = batch_object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "schema_version must be an integer",
                json!({"pointer":"/schema_version"}),
            )
        })?;
    if schema_version != 1 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedVersion,
            "unsupported operation schema_version",
            json!({"pointer":"/schema_version","schema_version":schema_version}),
        ));
    }
    let base_revision = batch_object
        .get("base_revision")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "base_revision must be a non-negative integer",
                json!({"pointer":"/base_revision"}),
            )
        })?;
    if base_revision != doc.revision {
        return Err(PotError::with_details(
            ErrorCode::RevisionConflict,
            "base_revision does not match the scene revision",
            json!({"pointer":"/base_revision","base_revision":base_revision,"revision":doc.revision}),
        ));
    }
    if let Some(evaluation) = batch_object.get("evaluation") {
        validate_evaluation(evaluation)?;
    }
    let evaluation_frame = resolve_evaluation_frame(doc, batch_object.get("evaluation"))?;
    let operations = batch_object
        .get("operations")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "operations must be an array",
                json!({"pointer":"/operations"}),
            )
        })?;
    let mut working = doc.clone();
    let mut pending_assets = BTreeMap::new();
    let mut changes = ChangeSet::default();
    let mut id_mappings = Map::new();
    let mut operation_results = Vec::with_capacity(operations.len());
    for (index, operation) in operations.iter().enumerate() {
        let object = operation.as_object().ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "operation must be an object",
                json!({"operation_index":index,"pointer":format!("/operations/{index}")}),
            )
        })?;
        let name = object.get("op").and_then(Value::as_str).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidOperation,
                "operation op must be a string",
                json!({"operation_index":index,"pointer":format!("/operations/{index}/op")}),
            )
        })?;
        let mut engine = Engine {
            doc: &mut working,
            changes,
            id_mappings,
            operation_index: index,
            evaluation_frame,
            asset_root,
            pending_assets: std::mem::take(&mut pending_assets),
        };
        if !OP_NAMES.contains(&name) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported operation `{name}`"),
                &operation_pointer(index, "op"),
            ));
        }
        library::ensure_operation_editable(&engine, object)?;
        let result = match name {
            "asset.mark" | "asset.clear" | "asset.update" | "asset.catalog_create" => {
                asset::apply(&mut engine, name, object)
            }
            "scene.create" | "scene.update" => scene::apply(&mut engine, name, object),
            "extension.register" => extension::apply(&mut engine, name, object),
            "collection.create"
            | "collection.instance_create"
            | "collection.update"
            | "collection.delete"
            | "collection.link"
            | "collection.parent"
            | "collection.unlink" => collection::apply(&mut engine, name, object),
            "library.link" | "library.append" | "library.override" | "library.register"
            | "library.reload" | "library.relocate" => library::apply(&mut engine, name, object),
            "mask.create" | "mask.update" | "mask.delete" => mask::apply(&mut engine, name, object),
            "sequencer.strip_create"
            | "sequencer.strip_update"
            | "sequencer.strip_delete"
            | "sequencer.update" => sequencer::apply(&mut engine, name, object),
            "tracking.clip_create"
            | "tracking.track_add"
            | "tracking.track"
            | "tracking.plane_track"
            | "tracking.solve_camera"
            | "tracking.solve_object"
            | "tracking.set_camera_intrinsics"
            | "tracking.undistort" => tracking::apply(&mut engine, name, object),
            "image.create" | "image.delete" | "image.load" | "image.set_pixels"
            | "image.update" => image::apply(&mut engine, name, object),
            "property.set" | "property.delete" => property::apply(&mut engine, name, object),
            "resource.pack" | "resource.unpack" => resource::apply(&mut engine, name, object),
            "text.create" | "text.update" | "text.delete" => text::apply(&mut engine, name, object),
            "curve.create" | "curve.update" | "curve.point_add" | "curve.point_update"
            | "curve.point_delete" | "curve.convert" | "surface.create" | "text_object.create"
            | "text_object.update" => curve::apply(&mut engine, name, object),
            "lattice.create" | "lattice.update" => lattice::apply(&mut engine, name, object),
            "metaball.create" | "metaball.update" => metaball::apply(&mut engine, name, object),
            "pointcloud.create" | "pointcloud.update" => {
                pointcloud::apply(&mut engine, name, object)
            }
            "volume.create" | "volume.update" => volume::apply(&mut engine, name, object),
            "node.join" => node_join::apply(&mut engine, object),
            "node.create" | "node.update" | "node.delete" | "node.duplicate" | "node.parent" => {
                node::apply(&mut engine, name, object)
            }
            "data.make_single_user" => node::make_single_user(&mut engine, object),
            "bone.create"
            | "bone.update"
            | "bone.delete"
            | "bone_collection.create"
            | "bone_collection.update"
            | "bone_collection.delete"
            | "bone_collection.assign"
            | "rig.generate_basic_human"
            | "bone_collection.unassign"
            | "pose.set"
            | "pose.reset"
            | "constraint.create"
            | "constraint.update"
            | "constraint.delete"
            | "constraint.reorder"
            | "shape_key.create"
            | "shape_key.update"
            | "shape_key.delete"
            | "vertex_group.create"
            | "vertex_group.assign"
            | "vertex_group.remove"
            | "driver.create"
            | "driver.update"
            | "driver.delete" => rig::apply(&mut engine, name, object),
            "graph.create"
            | "graph.delete"
            | "graph.node_add"
            | "graph.node_update"
            | "graph.node_remove"
            | "graph.link"
            | "graph.unlink"
            | "graph.interface_update" => graph::apply(&mut engine, name, object),
            "material.create" | "material.update" | "material.delete" => {
                material::apply(&mut engine, name, object)
            }
            "modifier.create"
            | "modifier.update"
            | "modifier.delete"
            | "modifier.reorder"
            | "modifier.bind"
            | "modifier.unbind"
            | "modifier.apply"
            | "modifier.apply_as_shape_key" => modifier::apply(&mut engine, name, object),
            "mesh.separate" => mesh_separate::apply(&mut engine, object),
            "mesh.attribute_create"
            | "mesh.attribute_update"
            | "mesh.attribute_delete"
            | "mesh.auto_smooth"
            | "mesh.edge_slide"
            | "mesh.vertex_slide"
            | "mesh.spin"
            | "mesh.screw"
            | "mesh.rip"
            | "mesh.merge"
            | "mesh.shade_smooth"
            | "mesh.shade_flat"
            | "mesh.mark_sharp_by_angle"
            | "mesh.set_custom_normals"
            | "mesh.poke"
            | "mesh.loop_cut"
            | "mesh.bridge"
            | "mesh.split"
            | "mesh.knife"
            | "mesh.remesh"
            | "mesh.symmetrize"
            | "mesh.transform_elements"
            | "mesh.extrude"
            | "mesh.inset"
            | "mesh.bevel"
            | "mesh.subdivide"
            | "mesh.triangulate"
            | "mesh.delete"
            | "mesh.dissolve"
            | "mesh.weld"
            | "mesh.fill"
            | "mesh.bisect"
            | "mesh.mirror"
            | "mesh.flip_normals"
            | "mesh.set_attribute" => mesh::apply(&mut engine, name, object),
            "sculpt.stroke"
            | "sculpt.dyntopo"
            | "sculpt.mask"
            | "sculpt.face_set"
            | "sculpt.remesh_voxel"
            | "sculpt.multires" => sculpt::apply(&mut engine, name, object),
            "paint.vertex" | "paint.weight" | "paint.texture" => {
                paint::apply(&mut engine, name, object)
            }
            "uv.pin" | "uv.pack" | "uv.unwrap" | "uv.transform" => {
                uv::apply(&mut engine, name, object)
            }
            "action.create" | "action.update" | "action.delete" | "keyframe.insert"
            | "keyframe.delete" | "fcurve.update" => animation::apply(&mut engine, name, object),
            "grease_pencil.layer_create"
            | "grease_pencil.layer_update"
            | "grease_pencil.layer_delete"
            | "grease_pencil.stroke_add"
            | "grease_pencil.stroke_update"
            | "grease_pencil.stroke_delete"
            | "grease_pencil.frame_add" => grease_pencil::apply(&mut engine, name, object),
            "action.slot_create"
            | "nla.track_create"
            | "nla.track_update"
            | "nla.track_delete"
            | "nla.strip_create"
            | "nla.strip_update"
            | "nla.strip_delete"
            | "nla.push_down"
            | "scene.marker_add"
            | "scene.marker_remove" => nla::apply(&mut engine, name, object),
            "camera.create" | "camera.update" | "light.create" | "light.update"
            | "world.create" | "world.update" => camera::apply(&mut engine, name, object),
            "color.update" | "render.passes_update" => color::apply(&mut engine, name, object),
            "compositor.enable"
            | "compositor.node_add"
            | "compositor.node_update"
            | "compositor.node_remove"
            | "compositor.link" => compositor::apply(&mut engine, name, object),
            "render.update" => render::apply(&mut engine, object),
            "physics.cloth.create"
            | "physics.cloth.update"
            | "physics.cloth.delete"
            | "physics.soft_body.create"
            | "physics.soft_body.update"
            | "physics.soft_body.delete"
            | "physics.particle_emitter.create"
            | "physics.particle_emitter.update"
            | "physics.particle_emitter.delete"
            | "physics.fluid.create"
            | "physics.fluid.update"
            | "physics.fluid.delete"
            | "physics.dynamic_paint.create"
            | "physics.dynamic_paint.update"
            | "physics.dynamic_paint.delete"
            | "physics.collision.create"
            | "physics.collision.update"
            | "physics.collision.delete"
            | "physics.force_field.create"
            | "physics.force_field.update"
            | "physics.force_field.delete"
            | "physics.rigid_body.create"
            | "physics.rigid_body.update"
            | "physics.rigid_body.delete"
            | "physics.world.update"
            | "simulation.settings.update" => physics::apply(&mut engine, name, object),
            _ => Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unsupported operation `{name}`"),
                &format!("/operations/{index}/op"),
            )),
        };
        let changed_by_operation = result.map_err(|mut error| {
            let details = error.details.as_object_mut();
            if let Some(details) = details {
                details.entry("operation_index").or_insert(json!(index));
                details
                    .entry("pointer")
                    .or_insert(json!(format!("/operations/{index}")));
            }
            error
        })?;
        let Engine {
            changes: next_changes,
            id_mappings: next_mappings,
            pending_assets: next_pending_assets,
            ..
        } = engine;
        changes = next_changes;
        id_mappings = next_mappings;
        pending_assets = next_pending_assets;
        let batch_changed = !changes.is_empty();
        operation_results.push(json!({"index":index,"op":name,"changed":changed_by_operation,"batch_changed":batch_changed}));
    }
    let mut referenced_assets = BTreeSet::new();
    for image in working.images.values() {
        if let Some(blob) = image.blob.as_deref() {
            referenced_assets.insert(blob);
        }
        referenced_assets.extend(image.tiles.iter().map(|tile| tile.blob.as_str()));
    }
    pending_assets.retain(|digest, _| referenced_assets.contains(digest.as_str()));
    changes.reconcile(doc, &working)?;
    let scene_changed = !changes.is_empty();
    if scene_changed {
        let Some(next_revision) = doc
            .revision
            .checked_add(1)
            .filter(|revision| *revision < (1_u64 << 53))
        else {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "scene revision exceeds the interoperable integer range",
            ));
        };
        working.revision = next_revision;
    }
    Ok(ApplyOutcome {
        doc: working,
        changes: changes.to_value(),
        operations: Value::Array(operation_results),
        id_mappings: Value::Object(id_mappings),
        asset_blobs: pending_assets,
        changed: scene_changed,
    })
}

fn validate_evaluation(value: &Value) -> Result<()> {
    let Some(object) = value.as_object() else {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "evaluation must be an object",
            json!({"pointer":"/evaluation"}),
        ));
    };
    for key in object.keys() {
        if ![
            "scene_id",
            "view_layer",
            "frame",
            "camera",
            "mode",
            "execution_policy",
        ]
        .contains(&key.as_str())
        {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("unknown evaluation field `{key}`"),
                json!({"pointer":format!("/evaluation/{}", pointer_escape(key))}),
            ));
        }
    }
    if let Some(frame) = object.get("frame")
        && frame.as_f64().is_none_or(|number| !number.is_finite())
    {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "evaluation frame must be finite",
            json!({"pointer":"/evaluation/frame"}),
        ));
    }
    for key in [
        "scene_id",
        "view_layer",
        "camera",
        "mode",
        "execution_policy",
    ] {
        if object
            .get(key)
            .is_some_and(|value| !value.is_string() && !value.is_object())
        {
            return Err(PotError::with_details(
                ErrorCode::InvalidOperation,
                format!("evaluation {key} has an invalid type"),
                json!({"pointer":format!("/evaluation/{key}")}),
            ));
        }
    }
    Ok(())
}
fn resolve_evaluation_frame(doc: &SceneDoc, evaluation: Option<&Value>) -> Result<f64> {
    let evaluation = evaluation.and_then(Value::as_object);
    let scene_id = evaluation
        .and_then(|context| context.get("scene_id"))
        .map(|value| {
            let raw = value.as_str().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    "evaluation.scene_id must be a string",
                    json!({"pointer":"/evaluation/scene_id"}),
                )
            })?;
            Id::new(raw).map_err(|error| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    error.message,
                    json!({"pointer":"/evaluation/scene_id"}),
                )
            })
        })
        .transpose()?
        .unwrap_or_else(|| doc.active_scene.clone());
    let scene = doc.scenes.get(&scene_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "evaluation scene context was not found",
            json!({"scene_id":scene_id}),
        )
    })?;
    let frame = evaluation
        .and_then(|context| context.get("frame"))
        .and_then(Value::as_f64)
        .unwrap_or(scene.frame_current);
    if !frame.is_finite() {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "evaluation frame must be finite",
            json!({"pointer":"/evaluation/frame"}),
        ));
    }
    Ok(frame)
}

fn pointer_escape(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn operation_pointer(index: usize, suffix: &str) -> String {
    if suffix.is_empty() {
        format!("/operations/{index}")
    } else {
        format!("/operations/{index}/{suffix}")
    }
}

fn check_fields(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    allowed: &[&str],
    required: &[&str],
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown operation field `{key}`"),
                &operation_pointer(engine.operation_index, &pointer_escape(key)),
            ));
        }
    }
    for key in required {
        if !object.contains_key(*key) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("missing required field `{key}`"),
                &operation_pointer(engine.operation_index, key),
            ));
        }
    }
    Ok(())
}

fn check_set_fields(engine: &Engine<'_>, set: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    check_set_fields_by(engine, set, allowed, |engine, key| {
        let escaped = pointer_escape(key);
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown set field `{key}`"),
            &operation_pointer(engine.operation_index, &format!("set/{escaped}")),
        )
    })
}

fn check_set_fields_by(
    engine: &Engine<'_>,
    set: &Map<String, Value>,
    allowed: &[&str],
    error_for_field: impl FnOnce(&Engine<'_>, &str) -> PotError,
) -> Result<()> {
    for field in set.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(error_for_field(engine, field));
        }
    }
    Ok(())
}

fn data_user_count(engine: &Engine<'_>, data_id: &Id) -> usize {
    engine
        .doc
        .nodes
        .values()
        .filter(|node| node.data.as_ref() == Some(data_id))
        .count()
}

fn clear_primitive_metadata(engine: &mut Engine<'_>, data_id: &Id) {
    let users = engine
        .doc
        .nodes
        .iter()
        .filter(|(_, node)| node.data.as_ref() == Some(data_id) && node.primitive.is_some())
        .map(|(node_id, _)| node_id.clone())
        .collect::<Vec<_>>();
    for node_id in users {
        if let Some(node) = engine.doc.nodes.get_mut(&node_id) {
            node.primitive = None;
        }
        engine.mark("nodes", &node_id, ChangeKind::Updated);
    }
}

#[derive(Clone, Copy)]
enum DataIdCollisionHandling {
    ReserveSuffix,
    TruncateCandidate,
}

fn unique_data_id(
    engine: &Engine<'_>,
    node_id: &Id,
    base_suffix: &str,
    collision_handling: DataIdCollisionHandling,
    exhausted_message: &str,
) -> Result<Id> {
    let base = format!("{node_id}{base_suffix}");
    let mut suffix = 0_u32;
    loop {
        let candidate = if suffix == 0 {
            base.chars().take(64).collect::<String>()
        } else {
            let tail = format!("_{suffix}");
            match collision_handling {
                DataIdCollisionHandling::ReserveSuffix => format!(
                    "{}{tail}",
                    base.chars()
                        .take(64_usize.saturating_sub(tail.len()))
                        .collect::<String>()
                ),
                DataIdCollisionHandling::TruncateCandidate => {
                    format!("{base}{tail}").chars().take(64).collect()
                }
            }
        };
        let candidate_id = Id::new(candidate)?;
        if !engine.doc.data_blocks.contains_key(&candidate_id) {
            return Ok(candidate_id);
        }
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, exhausted_message))?;
    }
}

fn parse_scope<'a>(
    engine: &Engine<'_>,
    value: Option<&'a Value>,
    pointer: &str,
    type_message: &str,
    value_message: &str,
    distinguish_type_error: bool,
) -> Result<Option<&'a str>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(scope) = value.as_str() else {
        let message = if distinguish_type_error {
            type_message
        } else {
            value_message
        };
        return Err(engine.error(ErrorCode::InvalidOperation, message, pointer));
    };
    if ["shared", "single_user"].contains(&scope) {
        Ok(Some(scope))
    } else {
        Err(engine.error(ErrorCode::InvalidOperation, value_message, pointer))
    }
}

#[derive(Clone, Copy)]
enum IdRegistry {
    Nodes,
    DataBlocks,
}

fn unique_id(engine: &Engine<'_>, base: &str, registry: IdRegistry) -> Result<Id> {
    let occupied = |id: &Id| match registry {
        IdRegistry::Nodes => engine.doc.nodes.contains_key(id),
        IdRegistry::DataBlocks => engine.doc.data_blocks.contains_key(id),
    };
    let mut suffix = 0_u32;
    loop {
        let tail = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let prefix_len = 64_usize.saturating_sub(tail.len());
        let prefix = base.chars().take(prefix_len).collect::<String>();
        let candidate = format!("{prefix}{tail}");
        let id = Id::new(candidate)?;
        if !occupied(&id) {
            return Ok(id);
        }
        suffix = suffix.checked_add(1).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "separated ID suffix exhausted")
        })?;
    }
}
fn target_registry(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    unsupported_registry_message: &str,
) -> Result<(String, Id)> {
    let target = operation
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target must be an object",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    if target
        .keys()
        .any(|key| !["registry", "id"].contains(&key.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target has an unknown field",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let registry = match target.get("registry") {
        Some(value) => value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target.registry must be a string",
                &operation_pointer(engine.operation_index, "target/registry"),
            )
        })?,
        None => "nodes",
    }
    .to_owned();
    if ![
        "scenes",
        "collections",
        "nodes",
        "data_blocks",
        "materials",
        "images",
        "masks",
        "worlds",
        "node_groups",
        "actions",
        "resources",
        "libraries",
    ]
    .contains(&registry.as_str())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            unsupported_registry_message,
            &operation_pointer(engine.operation_index, "target/registry"),
        ));
    }
    let text = target.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target.id must be a string",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    let id = parse_id(
        engine,
        text,
        &operation_pointer(engine.operation_index, "target/id"),
    )?;
    if !registry_contains_id(engine, &registry, &id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("{registry} item `{id}` does not exist"),
            &operation_pointer(engine.operation_index, "target/id"),
        ));
    }
    Ok((registry, id))
}

fn registry_contains_id(engine: &Engine<'_>, registry: &str, id: &Id) -> bool {
    match registry {
        "scenes" => engine.doc.scenes.contains_key(id),
        "collections" => engine.doc.collections.contains_key(id),
        "nodes" => engine.doc.nodes.contains_key(id),
        "data_blocks" => engine.doc.data_blocks.contains_key(id),
        "materials" => engine.doc.materials.contains_key(id),
        "images" => engine.doc.images.contains_key(id),
        "masks" => engine.doc.masks.contains_key(id),
        "worlds" => engine.doc.worlds.contains_key(id),
        "node_groups" => engine.doc.node_groups.contains_key(id),
        "actions" => engine.doc.actions.contains_key(id),
        "resources" => engine.doc.resources.contains_key(id),
        "libraries" => engine.doc.libraries.contains_key(id),
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum TargetIdPolicy {
    Strict {
        object_message: &'static str,
        shape_message: &'static str,
        id_message: &'static str,
        require_id: bool,
    },
    NamedField(&'static str),
}

fn target_id_value(
    engine: &Engine<'_>,
    value: Option<&Value>,
    pointer: &str,
    policy: TargetIdPolicy,
) -> Result<Id> {
    let object = value.and_then(Value::as_object).ok_or_else(|| {
        let message = match policy {
            TargetIdPolicy::Strict { object_message, .. } => object_message.to_owned(),
            TargetIdPolicy::NamedField(field) => {
                format!("{field} must be an object containing id")
            }
        };
        engine.error(ErrorCode::InvalidOperation, message, pointer)
    })?;
    match policy {
        TargetIdPolicy::Strict {
            shape_message,
            require_id,
            ..
        } if object.len() != 1 || (require_id && !object.contains_key("id")) => {
            return Err(engine.error(ErrorCode::InvalidOperation, shape_message, pointer));
        }
        TargetIdPolicy::NamedField(field) => {
            for key in object.keys() {
                if key != "id" {
                    return Err(engine.error(
                        ErrorCode::InvalidOperation,
                        format!("unknown {field} field `{key}`"),
                        &format!("{pointer}/{}", pointer_escape(key)),
                    ));
                }
            }
        }
        TargetIdPolicy::Strict { .. } => {}
    }
    let id_message = match policy {
        TargetIdPolicy::Strict { id_message, .. } => id_message.to_owned(),
        TargetIdPolicy::NamedField(field) => format!("{field}.id must be a string"),
    };
    let text = object.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            id_message,
            &format!("{pointer}/id"),
        )
    })?;
    parse_id(engine, text, &format!("{pointer}/id"))
}

#[derive(Clone, Copy)]
enum ObjectSlotError<'a> {
    RegistryName,
    Static(&'a str),
}

fn object_slot<'a>(
    engine: &'a mut Engine<'_>,
    key: &str,
    error: ObjectSlotError<'_>,
) -> Result<&'a mut Map<String, Value>> {
    let slot = engine
        .doc
        .compatibility
        .entry(key.to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    slot.as_object_mut().ok_or_else(|| match error {
        ObjectSlotError::RegistryName => PotError::new(
            ErrorCode::SceneInvalid,
            format!("compatibility `{key}` must be an object"),
        ),
        ObjectSlotError::Static(message) => PotError::new(ErrorCode::SceneInvalid, message),
    })
}
#[derive(Clone, Copy)]
enum FiniteNumberMessage {
    Field,
    FiniteField,
    Value,
}

fn finite_number(
    engine: &Engine<'_>,
    value: Option<&Value>,
    field: &str,
    message: FiniteNumberMessage,
) -> Result<f64> {
    let Some(number) = value.and_then(Value::as_f64) else {
        return Err(finite_number_error(engine, field, message, false));
    };
    if number.is_finite() {
        Ok(number)
    } else {
        Err(finite_number_error(engine, field, message, true))
    }
}

fn finite_number_error(
    engine: &Engine<'_>,
    field: &str,
    message: FiniteNumberMessage,
    is_non_finite: bool,
) -> PotError {
    match message {
        FiniteNumberMessage::Field => {
            let text = if is_non_finite {
                format!("{field} must be finite")
            } else {
                format!("{field} must be a number")
            };
            let pointer = operation_pointer(engine.operation_index, field);
            engine.error(ErrorCode::InvalidOperation, text, &pointer)
        }
        FiniteNumberMessage::FiniteField => {
            let pointer = operation_pointer(engine.operation_index, field);
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a finite number"),
                &pointer,
            )
        }
        FiniteNumberMessage::Value => engine.error(
            ErrorCode::InvalidOperation,
            "value must be a finite number",
            field,
        ),
    }
}

fn read_string(engine: &Engine<'_>, object: &Map<String, Value>, key: &str) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a string"),
                &operation_pointer(engine.operation_index, key),
            )
        })
}

fn read_bool(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
    default: bool,
) -> Result<bool> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a boolean"),
                &operation_pointer(engine.operation_index, key),
            )
        }),
    }
}

fn read_id(engine: &Engine<'_>, object: &Map<String, Value>, key: &str) -> Result<Id> {
    let value = read_string(engine, object, key)?;
    parse_id(
        engine,
        &value,
        &operation_pointer(engine.operation_index, key),
    )
}
fn parse_id(engine: &Engine<'_>, value: &str, pointer: &str) -> Result<Id> {
    Id::new(value).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "ID must match ^[a-z][a-z0-9_-]{0,63}$",
            pointer,
        )
    })
}

fn array_strings(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<Vec<String>> {
    let array = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "value must be an array",
            pointer,
        )
    })?;
    let mut result = Vec::with_capacity(array.len());
    for (index, item) in array.iter().enumerate() {
        let Some(text) = item.as_str() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "array items must be strings",
                &format!("{pointer}/{index}"),
            ));
        };
        parse_id(engine, text, &format!("{pointer}/{index}"))?;
        result.push(text.to_owned());
    }
    result.sort();
    result.dedup();
    Ok(result)
}

fn validate_target(
    engine: &Engine<'_>,
    value: &Value,
    allow_many: bool,
) -> Result<(Option<Id>, Option<String>, bool)> {
    let Some(target) = value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            &operation_pointer(engine.operation_index, "target"),
        ));
    };
    let pointer = operation_pointer(engine.operation_index, "target");
    for key in target.keys() {
        if !["id", "tag", "many"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown target field `{key}`"),
                &format!("{pointer}/{}", pointer_escape(key)),
            ));
        }
    }
    let has_id = target.contains_key("id");
    let has_tag = target.contains_key("tag");
    if has_id == has_tag {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must contain exactly one of id or tag",
            &pointer,
        ));
    }
    let many = target.get("many").map_or(Ok(false), |value| {
        value.as_bool().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target.many must be a boolean",
                &format!("{pointer}/many"),
            )
        })
    })?;
    if has_id && target.contains_key("many") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "many cannot be combined with an ID target",
            &format!("{pointer}/many"),
        ));
    }
    if many && !allow_many {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "many is only valid for operations that support it",
            &format!("{pointer}/many"),
        ));
    }
    if has_id {
        let value = target.get("id").and_then(Value::as_str).ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target.id must be a string",
                &format!("{pointer}/id"),
            )
        })?;
        Ok((
            Some(parse_id(engine, value, &format!("{pointer}/id"))?),
            None,
            false,
        ))
    } else {
        let value = target.get("tag").and_then(Value::as_str).ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target.tag must be a string",
                &format!("{pointer}/tag"),
            )
        })?;
        parse_id(engine, value, &format!("{pointer}/tag"))?;
        Ok((None, Some(value.to_owned()), many))
    }
}

fn resolve_node_targets(engine: &Engine<'_>, target: &Value, allow_many: bool) -> Result<Vec<Id>> {
    let (id, tag, many) = validate_target(engine, target, allow_many)?;
    let mut matches = if let Some(id) = id {
        if engine.doc.nodes.contains_key(&id) {
            vec![id]
        } else {
            Vec::new()
        }
    } else {
        let tag = tag.ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "target tag disappeared",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        engine
            .doc
            .nodes
            .iter()
            .filter(|(_, node)| node.tags.iter().any(|candidate| candidate == &tag))
            .map(|(id, _)| id.clone())
            .collect()
    };
    matches.sort();
    if matches.is_empty() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    if !many && matches.len() > 1 {
        return Err(engine.error(
            ErrorCode::AmbiguousTarget,
            "tag target matched multiple nodes",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(matches)
}

fn resolve_collection_targets(
    engine: &Engine<'_>,
    target: &Value,
    allow_many: bool,
) -> Result<Vec<Id>> {
    let (id, tag, many) = validate_target(engine, target, allow_many)?;
    let matches = if let Some(id) = id {
        if engine.doc.collections.contains_key(&id) {
            vec![id]
        } else {
            Vec::new()
        }
    } else {
        let _ = tag;
        Vec::new()
    };
    if matches.is_empty() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "collection target was not found",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    if !many && matches.len() > 1 {
        return Err(engine.error(
            ErrorCode::AmbiguousTarget,
            "tag target matched multiple collections",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(matches)
}

fn resolve_material_targets(
    engine: &Engine<'_>,
    target: &Value,
    allow_many: bool,
) -> Result<Vec<Id>> {
    let (id, tag, many) = validate_target(engine, target, allow_many)?;
    let matches = if let Some(id) = id {
        if engine.doc.materials.contains_key(&id) {
            vec![id]
        } else {
            Vec::new()
        }
    } else {
        let _ = tag;
        Vec::new()
    };
    if matches.is_empty() {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            "material target was not found",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    if !many && matches.len() > 1 {
        return Err(engine.error(
            ErrorCode::AmbiguousTarget,
            "tag target matched multiple materials",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "unit tests assert operation contracts")]

    use super::apply_batch;
    use crate::{error::ErrorCode, model::SceneDoc};
    use serde_json::json;

    fn batch(operations: serde_json::Value) -> serde_json::Value {
        let mut value = json!({"schema_version":1,"base_revision":0});
        value["operations"] = operations;
        value
    }

    #[test]
    fn empty_batch_is_a_noop_and_preserves_revision() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(&doc, &batch(json!([]))).unwrap();
        assert!(!outcome.changed);
        assert_eq!(outcome.doc.revision, 0);
        assert_eq!(outcome.changes["materials"]["created"], json!([]));
    }

    #[test]
    fn operation_schema_error_has_index_and_pointer() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([{"op":"material.create","id":"clay","unexpected":true}])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["operation_index"], json!(0));
        assert_eq!(error.details["pointer"], json!("/operations/0/unexpected"));
    }

    #[test]
    fn schema_and_revision_are_checked_before_operations() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let mut input = batch(json!([]));
        input["schema_version"] = json!(2);
        assert_eq!(
            apply_batch(&doc, &input).unwrap_err().code,
            ErrorCode::UnsupportedVersion
        );
        input["schema_version"] = json!(1);
        input["base_revision"] = json!(1);
        assert_eq!(
            apply_batch(&doc, &input).unwrap_err().code,
            ErrorCode::RevisionConflict
        );
    }

    #[test]
    fn late_failure_does_not_return_partial_document() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"material.create","id":"clay"},
                {"op":"node.create","id":"bad id","kind":"group"}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert!(
            !doc.materials
                .contains_key(&crate::model::Id::new("clay").unwrap())
        );
    }

    #[test]
    fn material_creation_reports_registry_diff_and_advances_revision() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([{"op":"material.create","id":"clay","name":"Clay"}])),
        )
        .unwrap();
        assert!(outcome.changed);
        assert_eq!(outcome.doc.revision, 1);
        assert_eq!(outcome.changes["materials"]["created"], json!(["clay"]));
        assert_eq!(outcome.operations[0]["index"], json!(0));
    }

    #[test]
    fn create_then_delete_collapses_to_a_noop() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"material.create","id":"clay"},
                {"op":"material.delete","target":{"id":"clay"}}
            ])),
        )
        .unwrap();
        assert!(!outcome.changed);
        assert_eq!(outcome.doc.revision, 0);
        assert!(outcome.doc.materials.is_empty());
    }

    #[test]
    fn invalid_tags_do_not_become_registry_ids() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([{"op":"node.update","target":{"tag":"Bad"},"set":{"name":"x"}}])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["pointer"], json!("/operations/0/target/tag"));
    }
    #[test]
    fn tag_targets_reject_ambiguous_singular_updates() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"left","kind":"group","tags":["part"]},
                {"op":"node.create","id":"right","kind":"group","tags":["part"]},
                {"op":"node.update","target":{"tag":"part"},"set":{"visible":false}}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::AmbiguousTarget);
        assert_eq!(error.details["operation_index"], json!(2));
        assert_eq!(error.details["pointer"], json!("/operations/2/target"));
    }

    #[test]
    fn primitive_node_creation_builds_object_mesh_and_collection_membership() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([{
                "op":"node.create","id":"body","kind":"box","params":{"size":2.0},"transform":{"scale":[0.5,1.0,1.5]}
            }])),
        )
        .unwrap();
        let node = outcome
            .doc
            .nodes
            .get(&crate::model::Id::new("body").unwrap())
            .unwrap();
        assert_eq!(node.kind, "mesh");
        assert_eq!(node.primitive.as_deref(), Some("box"));
        assert_eq!(
            node.data.as_ref().map(ToString::to_string).as_deref(),
            Some("body_mesh")
        );
        let mesh = outcome
            .doc
            .data_blocks
            .get(&crate::model::Id::new("body_mesh").unwrap())
            .unwrap()
            .mesh
            .as_ref()
            .unwrap();
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(
            outcome.doc.collections.values().next().unwrap().objects[0].as_str(),
            "body"
        );
    }

    #[test]
    fn transform_rotation_formats_cannot_be_specified_together() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([{
                "op":"node.create","id":"body","kind":"group",
                "transform":{"rotation":[0.0,0.0,0.0,1.0],"rotation_deg":[0.0,0.0,0.0]}
            }])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(
            error.details["pointer"],
            json!("/operations/0/transform/rotation_deg")
        );
    }

    #[test]
    fn shared_primitive_updates_require_explicit_scope_and_single_user_separates_data() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
                {"op":"node.duplicate","target":{"id":"body"},"id":"linked","mode":"linked"},
                {"op":"node.update","target":{"id":"body"},"set":{"params":{"size":1.0}}}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::SharedDataRequiresScope);
        assert_eq!(error.details["operation_index"], json!(2));

        let outcome = apply_batch(&doc, &batch(json!([
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
            {"op":"node.duplicate","target":{"id":"body"},"id":"linked","mode":"linked"},
            {"op":"node.update","target":{"id":"body"},"scope":"single_user","set":{"params":{"size":1.0}}}
        ]))).unwrap();
        let body_id = crate::model::Id::new("body").unwrap();
        let linked_id = crate::model::Id::new("linked").unwrap();
        assert_eq!(
            outcome.doc.nodes[&body_id].data.as_ref().unwrap().as_str(),
            "body_single"
        );
        assert_eq!(
            outcome.doc.nodes[&linked_id]
                .data
                .as_ref()
                .unwrap()
                .as_str(),
            "body_mesh"
        );
        assert_eq!(
            outcome.changes["data_blocks"]["created"],
            json!(["body_mesh", "body_single"])
        );
        assert_eq!(
            outcome.id_mappings["data_blocks"]["body_mesh"],
            json!("body_single")
        );
    }

    #[test]
    fn material_delete_requires_an_explicit_reassignment_policy() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"material.create","id":"clay"},
                {"op":"node.create","id":"body","kind":"group","material":"clay"},
                {"op":"material.delete","target":{"id":"clay"}}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["operation_index"], json!(2));

        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"material.create","id":"clay"},
                {"op":"node.create","id":"body","kind":"group","material":"clay"},
                {"op":"material.delete","target":{"id":"clay"},"reassign":"default"}
            ])),
        )
        .unwrap();
        assert!(outcome.doc.materials.is_empty());
        let node_materials = &outcome.doc.nodes.values().next().unwrap().materials;
        assert!(node_materials.is_empty(), "{node_materials:?}");
        assert_eq!(outcome.changes["materials"]["created"], json!([]));
        assert_eq!(outcome.changes["materials"]["deleted"], json!([]));
        let reassigned = apply_batch(
            &doc,
            &batch(json!([
                {"op":"material.create","id":"clay"},
                {"op":"material.create","id":"stone"},
                {"op":"node.create","id":"body","kind":"group","material":"clay"},
                {"op":"material.delete","target":{"id":"clay"},"reassign":"stone"}
            ])),
        )
        .unwrap();
        assert!(
            reassigned
                .doc
                .materials
                .contains_key(&crate::model::Id::new("stone").unwrap())
        );
        assert_eq!(
            reassigned.doc.nodes.values().next().unwrap().materials[0].as_str(),
            "stone"
        );
    }

    #[test]
    fn material_update_reports_changes_and_preserves_noop_revision() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let created =
            apply_batch(&doc, &batch(json!([{"op":"material.create","id":"clay"}]))).unwrap();
        let update = json!({
            "schema_version":1,
            "base_revision":created.doc.revision,
            "operations":[{"op":"material.update","target":{"id":"clay"},"set":{"name":"Paint","base_color":[0.2,0.3,0.4,1.0],"roughness":0.25,"double_sided":true}}]
        });
        let updated = apply_batch(&created.doc, &update).unwrap();
        let material_id = crate::model::Id::new("clay").unwrap();
        let material = &updated.doc.materials[&material_id];
        assert_eq!(material.name, "Paint");
        assert_eq!(material.base_color, [0.2, 0.3, 0.4, 1.0]);
        assert_eq!(material.roughness, 0.25);
        assert!(material.double_sided);
        assert_eq!(updated.changes["materials"]["updated"], json!(["clay"]));
        assert_eq!(updated.doc.revision, 2);

        let mut noop_batch = update.clone();
        noop_batch["base_revision"] = json!(updated.doc.revision);
        let noop = apply_batch(&updated.doc, &noop_batch).unwrap();
        assert!(!noop.changed);
        assert_eq!(noop.doc.revision, 2);
        assert_eq!(noop.changes["materials"]["updated"], json!([]));
    }
    #[test]
    fn collection_hierarchy_cycles_are_rejected_with_pointer_details() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"collection.create","id":"child"},
                {"op":"collection.link","collection":"child","child":"collection_root"}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["operation_index"], json!(1));
        assert_eq!(error.details["pointer"], json!("/operations/1/child"));
    }

    #[test]
    fn collection_link_unlink_update_and_delete_change_membership() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let linked = apply_batch(
            &doc,
            &batch(json!([
                {"op":"collection.create","id":"child"},
                {"op":"node.create","id":"object","kind":"group"},
                {"op":"collection.link","collection":"child","object":"object"},
                {"op":"collection.update","target":{"id":"child"},"set":{"name":"Child"}}
            ])),
        )
        .unwrap();
        let child_id = crate::model::Id::new("child").unwrap();
        assert_eq!(linked.doc.collections[&child_id].name, "Child");
        assert_eq!(
            linked.doc.collections[&child_id].objects[0].as_str(),
            "object"
        );

        let unlink_delete = json!({
            "schema_version":1,
            "base_revision":linked.doc.revision,
            "operations":[
                {"op":"collection.unlink","collection":"child","object":"object"},
                {"op":"collection.delete","target":{"id":"child"}}
            ]
        });
        let outcome = apply_batch(&linked.doc, &unlink_delete).unwrap();
        assert!(!outcome.doc.collections.contains_key(&child_id));
        assert!(
            outcome
                .doc
                .nodes
                .contains_key(&crate::model::Id::new("object").unwrap())
        );
        assert_eq!(outcome.changes["collections"]["deleted"], json!(["child"]));
    }

    #[test]
    fn independent_duplicate_gets_an_independent_geometry_block() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{"size":1.0}},
                {"op":"node.duplicate","target":{"id":"body"},"id":"copy"}
            ])),
        )
        .unwrap();
        let body_id = crate::model::Id::new("body").unwrap();
        let copy_id = crate::model::Id::new("copy").unwrap();
        assert_eq!(
            outcome.doc.nodes[&body_id].data.as_ref().unwrap().as_str(),
            "body_mesh"
        );
        assert_eq!(
            outcome.doc.nodes[&copy_id].data.as_ref().unwrap().as_str(),
            "copy_mesh"
        );
        assert_eq!(outcome.doc.data_blocks.len(), 2);
        assert_eq!(
            outcome.id_mappings["data_blocks"]["body_mesh"],
            json!("copy_mesh")
        );
    }
    proptest::proptest! {
        #[test]
        fn material_roughness_values_round_trip(value in 0.0_f64..=1.0) {
            let doc = SceneDoc::new("scene-test".to_owned());
            let created = apply_batch(&doc, &batch(json!([{"op":"material.create","id":"clay"}]))).unwrap();
            let update = json!({"schema_version":1,"base_revision":created.doc.revision,"operations":[{"op":"material.update","target":{"id":"clay"},"set":{"roughness":value}}]});
            let outcome = apply_batch(&created.doc, &update).unwrap();
            let id = crate::model::Id::new("clay").unwrap();
            proptest::prop_assert_eq!(outcome.doc.materials[&id].roughness, value);
        }
    }

    #[test]
    fn scene_create_and_partial_update_track_registry_differences() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(&doc, &batch(json!([
            {"op":"scene.create","id":"scene_edit","name":"Edit"},
            {"op":"scene.update","target":{"id":"scene_edit"},"set":{"frame_current":12.5,"unit":{"scale_length":0.25}}}
        ]))).unwrap();
        let id = crate::model::Id::new("scene_edit").unwrap();
        let scene = &outcome.doc.scenes[&id];
        assert_eq!(scene.frame_current, 12.5);
        assert_eq!(scene.unit.system, "metric");
        assert_eq!(scene.unit.scale_length, 0.25);
        assert_eq!(outcome.changes["scenes"]["created"], json!(["scene_edit"]));
        assert_eq!(outcome.changes["scenes"]["updated"], json!([]));
        assert_eq!(
            outcome.changes["collections"]["created"],
            json!(["scene_edit_root"])
        );
    }

    #[test]
    fn node_parenting_and_recursive_delete_respect_hierarchy() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"root","kind":"group"},
                {"op":"node.create","id":"child","kind":"group","parent":"root"},
                {"op":"node.delete","target":{"id":"root"}}
            ])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["operation_index"], json!(2));

        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"root","kind":"group"},
                {"op":"node.create","id":"child","kind":"group"},
                {"op":"node.parent","target":{"id":"child"},"parent":"root"},
                {"op":"node.delete","target":{"id":"root"},"recursive":true}
            ])),
        )
        .unwrap();
        assert!(outcome.doc.nodes.is_empty());
        assert_eq!(outcome.changes["nodes"]["created"], json!([]));
        assert_eq!(outcome.changes["nodes"]["deleted"], json!([]));
        assert!(!outcome.changed);
        assert_eq!(outcome.doc.revision, 0);
    }

    #[test]
    fn data_make_single_user_creates_an_isolated_block() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{"size":2.0}},
                {"op":"node.duplicate","target":{"id":"body"},"id":"linked","mode":"linked"},
                {"op":"data.make_single_user","target":{"id":"body"}}
            ])),
        )
        .unwrap();
        let body_id = crate::model::Id::new("body").unwrap();
        let linked_id = crate::model::Id::new("linked").unwrap();
        assert_eq!(
            outcome.doc.nodes[&body_id].data.as_ref().unwrap().as_str(),
            "body_single"
        );
        assert_eq!(
            outcome.doc.nodes[&linked_id]
                .data
                .as_ref()
                .unwrap()
                .as_str(),
            "body_mesh"
        );
        assert_eq!(
            outcome.changes["data_blocks"]["created"],
            json!(["body_mesh", "body_single"])
        );
    }
    #[test]
    fn many_single_user_targets_all_receive_their_own_data_block() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let shared = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{"size":2.0},"tags":["part"]},
                {"op":"node.duplicate","target":{"id":"body"},"id":"linked","mode":"linked"}
            ])),
        )
        .unwrap();
        let split = json!({
            "schema_version":1,
            "base_revision":shared.doc.revision,
            "operations":[{"op":"data.make_single_user","target":{"tag":"part","many":true}}]
        });
        let outcome = apply_batch(&shared.doc, &split).unwrap();
        let body_id = crate::model::Id::new("body").unwrap();
        let linked_id = crate::model::Id::new("linked").unwrap();
        assert_eq!(
            outcome.doc.nodes[&body_id].data.as_ref().unwrap().as_str(),
            "body_single"
        );
        assert_eq!(
            outcome.doc.nodes[&linked_id]
                .data
                .as_ref()
                .unwrap()
                .as_str(),
            "linked_single"
        );
        assert_eq!(
            outcome.id_mappings["data_blocks"]["body_mesh"],
            json!(["body_single", "linked_single"])
        );
    }
    #[test]
    fn keep_world_parenting_records_parent_inverse() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(&doc, &batch(json!([
            {"op":"node.create","id":"parent","kind":"group","transform":{"translation":[10.0,0.0,0.0]}},
            {"op":"node.create","id":"child","kind":"group","transform":{"translation":[2.0,0.0,0.0]}},
            {"op":"node.parent","target":{"id":"child"},"parent":"parent"}
        ]))).unwrap();
        let child_id = crate::model::Id::new("child").unwrap();
        let child = &outcome.doc.nodes[&child_id];
        assert_eq!(child.parent.as_ref().unwrap().as_str(), "parent");
        assert_eq!(child.parent_inverse.unwrap()[12], -10.0);
        assert_eq!(child.transform.translation, [2.0, 0.0, 0.0]);
    }
    #[test]
    fn every_geometry_primitive_builds_a_mesh_data_block() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let primitives = [
            "box",
            "uv_sphere",
            "cylinder",
            "plane",
            "cone",
            "torus",
            "icosphere",
            "circle",
            "grid",
        ];
        let operations: Vec<_> = primitives.iter().map(|primitive| {
            json!({"op":"node.create","id":format!("object_{primitive}"),"kind":primitive,"params":{}})
        }).collect();
        let outcome = apply_batch(&doc, &batch(serde_json::Value::Array(operations))).unwrap();
        for primitive in primitives {
            let node_id = crate::model::Id::new(format!("object_{primitive}")).unwrap();
            let node = &outcome.doc.nodes[&node_id];
            assert_eq!(node.kind, "mesh");
            assert_eq!(node.primitive.as_deref(), Some(primitive));
            let data_id = node.data.as_ref().unwrap();
            let mesh = outcome.doc.data_blocks[data_id].mesh.as_ref().unwrap();
            assert!(!mesh.vertices.is_empty(), "primitive mesh has no vertices");
        }
    }

    #[test]
    fn sphere_kind_is_an_alias_for_uv_sphere() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([{"op":"node.create","id":"ball","kind":"sphere","params":{}}])),
        )
        .unwrap();
        let node_id = crate::model::Id::new("ball").unwrap();
        assert_eq!(
            outcome.doc.nodes[&node_id].primitive.as_deref(),
            Some("uv_sphere")
        );
    }

    #[test]
    fn primitive_parameter_errors_report_the_invalid_json_pointer() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let error = apply_batch(
            &doc,
            &batch(json!([{"op":"node.create","id":"box","kind":"box","params":{"size":0}}])),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidOperation);
        assert_eq!(error.details["pointer"], "/operations/0/params/size");
    }

    #[test]
    fn modifier_params_reject_unknown_names_and_enums_with_precise_errors() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let unknown_create = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{}},
                {"op":"modifier.create","target":{"id":"body"},"id":"array","type":"array",
                 "params":{"relative_offset":[1.0,0.0,0.0]}}
            ])),
        )
        .unwrap_err();
        assert_eq!(unknown_create.code, ErrorCode::InvalidArgument);
        assert_eq!(
            unknown_create.details["pointer"],
            "/operations/1/params/relative_offset"
        );

        let invalid_enum = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{}},
                {"op":"modifier.create","target":{"id":"body"},"id":"array","type":"array",
                 "params":{"fit_type":"STRETCH"}}
            ])),
        )
        .unwrap_err();
        assert_eq!(invalid_enum.code, ErrorCode::InvalidArgument);
        assert_eq!(
            invalid_enum.details["pointer"],
            "/operations/1/params/fit_type"
        );
        assert!(invalid_enum.message.contains("FIT_LENGTH"));

        let invalid_update = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"body","kind":"box","params":{}},
                {"op":"modifier.create","target":{"id":"body"},"id":"array","type":"array"},
                {"op":"modifier.update","target":{"id":"body"},"modifier_id":"array",
                 "set":{"params":{"relative_offset":[0.0,0.0,0.0]}}}
            ])),
        )
        .unwrap_err();
        assert_eq!(invalid_update.code, ErrorCode::InvalidArgument);
        assert_eq!(
            invalid_update.details["pointer"],
            "/operations/2/set/params/relative_offset"
        );
    }

    #[test]
    fn camera_and_light_nodes_reference_typed_data_blocks() {
        let doc = SceneDoc::new("scene-test".to_owned());
        let outcome = apply_batch(
            &doc,
            &batch(json!([
                {"op":"node.create","id":"camera","kind":"camera"},
                {"op":"node.create","id":"light","kind":"light"}
            ])),
        )
        .unwrap();
        let camera_node = &outcome.doc.nodes[&crate::model::Id::new("camera").unwrap()];
        let camera_data = &outcome.doc.data_blocks[camera_node.data.as_ref().unwrap()];
        assert_eq!(camera_data.data_type, "camera");
        assert_eq!(camera_data.camera.as_ref().unwrap().lens_mm, 50.0);
        let light_node = &outcome.doc.nodes[&crate::model::Id::new("light").unwrap()];
        let light_data = &outcome.doc.data_blocks[light_node.data.as_ref().unwrap()];
        assert_eq!(light_data.data_type, "light");
        assert_eq!(
            light_data.light.as_ref().unwrap().light_type,
            crate::model::LightType::Point
        );
    }
}
