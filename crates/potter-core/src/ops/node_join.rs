use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, Result},
    geom::{
        Edge, Face, Mesh, Vertex,
        uv_layers::{self, UvLayer},
    },
    model::{
        DataBlock, Id, MaterialLink, Node, ShapeKey, ShapeKeyData, VertexGroup, material_link_at,
    },
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, resolve_node_targets};

struct JoinInput {
    node_id: Id,
    node: Node,
    data_id: Id,
    slot_materials: Vec<Option<Id>>,
    data: DataBlock,
    world: DMat4,
}

struct SlotSources<'a> {
    data: &'a [Option<Id>],
    object: &'a [Option<Id>],
    links: &'a [MaterialLink],
}

impl SlotSources<'_> {
    fn link(&self, slot: usize) -> MaterialLink {
        material_link_at(self.object.len(), self.links, slot)
    }
}

fn slot_sources(input: &JoinInput) -> SlotSources<'_> {
    SlotSources {
        data: &input.data.materials,
        object: &input.node.materials,
        links: &input.node.material_links,
    }
}

#[derive(Default)]
struct ElementMapping {
    vertices: BTreeMap<u32, u32>,
    edges: BTreeMap<u32, u32>,
    faces: BTreeMap<u32, u32>,
}

impl ElementMapping {
    fn to_value(&self) -> Value {
        json!({
            "vertices": self.vertices.iter().map(|(old, new)| (format!("v{old}"), json!(format!("v{new}")))).collect::<Map<_, _>>(),
            "edges": self.edges.iter().map(|(old, new)| (format!("e{old}"), json!(format!("e{new}")))).collect::<Map<_, _>>(),
            "faces": self.faces.iter().map(|(old, new)| (format!("f{old}"), json!(format!("f{new}")))).collect::<Map<_, _>>(),
        })
    }
}

pub(super) fn apply(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "target", "targets"], &[])?;
    let targets = read_targets(engine, operation)?;
    if targets.len() < 2 {
        return Ok(false);
    }

    let frame = engine.evaluation_frame;
    let driver_values = crate::eval::rig::evaluate_drivers(engine.doc, frame)?;
    let world_matrices =
        crate::eval::rig::evaluate_world_matrices_with_drivers(engine.doc, frame, &driver_values)?;
    let destination_id = targets[0].clone();
    let mut inputs = Vec::with_capacity(targets.len());
    for node_id in &targets {
        let node = engine.doc.nodes.get(node_id).cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{node_id}` was not found"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        if node.kind != "mesh" {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("node `{node_id}` is not a mesh object"),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        let data_id = node.data.clone().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("mesh object `{node_id}` has no data block"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        let data = engine
            .doc
            .data_blocks
            .get(&data_id)
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!("data block `{data_id}` was not found"),
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
        if data.data_type != "mesh" || data.mesh.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("data block `{data_id}` has no mesh payload"),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        let world = world_matrices.get(node_id).copied().ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                format!("world transform for node `{node_id}` is missing"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        inputs.push(JoinInput {
            node_id: node_id.clone(),
            node,
            data_id,
            data,
            slot_materials: Vec::new(),
            world,
        });
    }
    let slot_node_ids: BTreeSet<_> = inputs.iter().map(|input| input.node_id.clone()).collect();
    let slot_snapshot = crate::eval::Snapshot::evaluate_nodes_with_cache(
        engine.doc,
        &crate::eval::EvaluationContext {
            frame: Some(engine.evaluation_frame),
            ..crate::eval::EvaluationContext::default()
        },
        engine.asset_root,
        engine.cache_directory,
        &slot_node_ids,
    )?;
    for input in &mut inputs {
        let slot_count = slot_snapshot.slot_count(&input.node_id, &input.node, Some(&input.data));
        input.slot_materials = (0..slot_count)
            .map(|slot| {
                slot_snapshot
                    .slot_material(&input.node_id, &input.node, Some(&input.data), slot)
                    .cloned()
            })
            .collect();
    }

    let source_ids: BTreeSet<_> = targets.iter().skip(1).cloned().collect();
    if engine.doc.nodes.iter().any(|(child_id, node)| {
        node.parent
            .as_ref()
            .is_some_and(|parent| source_ids.contains(parent) && !source_ids.contains(child_id))
    }) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "a joined node has children outside the join target list",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let scene_cameras: Vec<_> = engine
        .doc
        .scenes
        .iter()
        .filter(|(_, scene)| {
            scene
                .camera
                .as_ref()
                .is_some_and(|camera| source_ids.contains(camera))
        })
        .map(|(scene_id, _)| scene_id.clone())
        .collect();

    let destination = &inputs[0];
    let destination_mesh = destination.data.mesh.as_ref().ok_or_else(|| {
        engine.error(
            ErrorCode::InternalError,
            "destination mesh payload disappeared",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut merged_mesh = destination_mesh.clone();
    for input in &inputs {
        input
            .data
            .mesh
            .as_ref()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InternalError,
                    "mesh payload disappeared",
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?
            .validate()
            .map_err(|error| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    format!("mesh `{}` is invalid: {error}", input.data_id),
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?;
    }
    let destination_determinant = destination.world.determinant();
    if !destination_determinant.is_finite() || destination_determinant.abs() <= f64::EPSILON {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "destination mesh object has a singular world transform",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let destination_inverse = destination.world.inverse();
    let destination_slot_count = destination.slot_materials.len();
    let destination_slots = slot_sources(destination);
    let mut merged_materials = destination_slots.object.to_vec();
    merged_materials.resize(destination_slot_count, None);
    let mut merged_material_links = (0..destination_slot_count)
        .map(|slot| destination_slots.link(slot))
        .collect::<Vec<_>>();
    let mut merged_effective_materials = destination.slot_materials.clone();
    validate_material_indices(engine, destination, destination_mesh)?;
    let mut merged_data = destination.data.clone();
    merged_data.materials = destination_slots.data.to_vec();
    merged_data.materials.resize(destination_slot_count, None);
    let mut output_shape_keys = merged_data.shape_keys.take();
    let mut output_groups = merged_data.vertex_groups.clone();
    let mut output_weights = merged_data.vertex_weights.clone();
    let mut mesh_mappings = Map::new();

    for source in inputs.iter().skip(1) {
        let source_mesh = source.data.mesh.as_ref().ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "source mesh payload disappeared",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        validate_material_indices(engine, source, source_mesh)?;
        let source_to_destination = destination_inverse * source.world;
        if !source_to_destination
            .to_cols_array()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("transform for source `{}` is not finite", source.node_id),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }

        let source_slots = slot_sources(source);
        let mapping = append_mesh(
            engine,
            &mut merged_mesh,
            source_mesh,
            source_to_destination,
            &mut merged_materials,
            &mut merged_material_links,
            &mut merged_data.materials,
            &mut merged_effective_materials,
            source,
            &source_slots,
        )?;
        mesh_mappings.insert(source.node_id.to_string(), mapping.to_value());
        let group_mapping = merge_vertex_groups(
            &mut output_groups,
            &source.data.vertex_groups,
            &source.node_id,
        );
        merge_vertex_weights(
            engine,
            &mut output_weights,
            &source.data.vertex_weights,
            &mapping.vertices,
            &group_mapping,
            &source.node_id,
        )?;
        merge_shape_keys(
            engine,
            &mut output_shape_keys,
            source.data.shape_keys.as_ref(),
            &mapping.vertices,
            &group_mapping,
            source_to_destination,
            &source.node_id,
        )?;
    }

    merged_mesh.validate().map_err(|error| {
        engine.error(
            ErrorCode::InternalError,
            format!("joined mesh is invalid: {error}"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    merged_data.mesh = Some(merged_mesh);
    merged_data.shape_keys = output_shape_keys;
    merged_data.vertex_groups = output_groups;
    merged_data.vertex_weights = output_weights;
    if merged_data.mesh.as_ref() != destination.data.mesh.as_ref() {
        merged_data.descriptor = None;
    }

    let mut target_node = destination.node.clone();
    target_node.materials = merged_materials;
    target_node.material_links = merged_material_links;
    let geometry_changed = merged_data.mesh != destination.data.mesh
        || merged_data.shape_keys != destination.data.shape_keys
        || merged_data.vertex_groups != destination.data.vertex_groups
        || merged_data.vertex_weights != destination.data.vertex_weights;
    let data_changed = geometry_changed || merged_data.materials != destination.data.materials;
    if geometry_changed {
        target_node.primitive = None;
    }
    let output_data_id = if data_changed && super::data_user_count(engine, &destination.data_id) > 1
    {
        unique_join_data_id(engine, &destination_id)?
    } else {
        destination.data_id.clone()
    };
    let node_changed = target_node.materials != destination.node.materials
        || target_node.material_links != destination.node.material_links
        || target_node.primitive != destination.node.primitive
        || output_data_id != destination.data_id;
    if data_changed {
        engine
            .doc
            .data_blocks
            .insert(output_data_id.clone(), merged_data);
        engine.mark(
            "data_blocks",
            &output_data_id,
            if output_data_id == destination.data_id {
                ChangeKind::Updated
            } else {
                ChangeKind::Created
            },
        );
    }
    if node_changed {
        target_node.data = Some(output_data_id.clone());
        engine.doc.nodes.insert(destination_id.clone(), target_node);
        engine.mark("nodes", &destination_id, ChangeKind::Updated);
    }

    let mut affected_data: BTreeSet<Id> =
        inputs.iter().map(|input| input.data_id.clone()).collect();
    let changed_collections: BTreeSet<_> = engine
        .doc
        .collections
        .iter()
        .filter(|(_, collection)| collection.objects.iter().any(|id| source_ids.contains(id)))
        .map(|(id, _)| id.clone())
        .collect();
    for source in inputs.iter().skip(1) {
        engine.doc.nodes.remove(&source.node_id);
        engine.mark("nodes", &source.node_id, ChangeKind::Deleted);
        engine.map_id("nodes", &source.node_id, &destination_id);
        if source.data_id != output_data_id {
            engine.map_id("data_blocks", &source.data_id, &output_data_id);
        }
        engine
            .id_mappings
            .entry("mesh_elements".to_owned())
            .or_insert_with(|| json!({}))[source.node_id.as_str()] = mesh_mappings
            .get(source.node_id.as_str())
            .cloned()
            .unwrap_or(Value::Null);
        for collection in engine.doc.collections.values_mut() {
            if collection.objects.contains(&source.node_id) {
                collection
                    .objects
                    .retain(|node_id| node_id != &source.node_id);
            }
        }
    }
    if output_data_id != destination.data_id {
        engine.map_id("data_blocks", &destination.data_id, &output_data_id);
        if let Some(node) = engine.doc.nodes.get_mut(&destination_id) {
            node.data = Some(output_data_id.clone());
        }
    }
    for scene_id in scene_cameras {
        if let Some(scene) = engine.doc.scenes.get_mut(&scene_id) {
            scene.camera = Some(destination_id.clone());
        }
        engine.mark("scenes", &scene_id, ChangeKind::Updated);
    }

    for collection_id in changed_collections {
        engine.mark("collections", &collection_id, ChangeKind::Updated);
    }
    if output_data_id != destination.data_id {
        affected_data.insert(destination.data_id.clone());
    }
    for data_id in affected_data {
        if data_id == output_data_id {
            continue;
        }
        let still_used = engine
            .doc
            .nodes
            .values()
            .any(|node| node.data.as_ref() == Some(&data_id));
        if !still_used && engine.doc.data_blocks.remove(&data_id).is_some() {
            engine.mark("data_blocks", &data_id, ChangeKind::Deleted);
        }
    }
    Ok(true)
}

fn read_targets(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Vec<Id>> {
    if operation.contains_key("target") == operation.contains_key("targets") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "node.join requires exactly one of target or targets",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    if let Some(target) = operation.get("target") {
        return resolve_node_targets(engine, target, true);
    }
    let values = operation
        .get("targets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "targets must be an array of node IDs",
                &operation_pointer(engine.operation_index, "targets"),
            )
        })?;
    let mut seen = BTreeSet::new();
    let mut targets = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let pointer = operation_pointer(engine.operation_index, &format!("targets/{index}"));
        let text = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "target IDs must be strings",
                &pointer,
            )
        })?;
        let id = parse_id(engine, text, &pointer)?;
        if !engine.doc.nodes.contains_key(&id) {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{id}` was not found"),
                &pointer,
            ));
        }
        if seen.insert(id.clone()) {
            targets.push(id);
        }
    }
    if targets.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "targets must contain at least one node ID",
            &operation_pointer(engine.operation_index, "targets"),
        ));
    }
    Ok(targets)
}

fn append_mesh(
    engine: &Engine<'_>,
    destination: &mut Mesh,
    source: &Mesh,
    transform: DMat4,
    destination_materials: &mut Vec<Option<Id>>,
    destination_links: &mut Vec<MaterialLink>,
    destination_data_materials: &mut Vec<Option<Id>>,
    destination_effective_materials: &mut Vec<Option<Id>>,
    source_slots: &JoinInput,
    source_material_state: &SlotSources<'_>,
) -> Result<ElementMapping> {
    let mut mapping = ElementMapping::default();
    // Source face `i` lands at storage index `face_offset + i` once appended.
    let face_offset = destination.faces.len();
    let mut used_vertices: BTreeSet<_> = destination.vertices.iter().map(|item| item.id).collect();
    let mut used_edges: BTreeSet<_> = destination.edges.iter().map(|item| item.id).collect();
    let mut used_faces: BTreeSet<_> = destination.faces.iter().map(|item| item.id).collect();
    for vertex in &source.vertices {
        let id = remap_element_id(
            engine,
            vertex.id,
            &mut used_vertices,
            &mut destination.next_id.vertex,
            "vertex",
        )?;
        mapping.vertices.insert(vertex.id, id);
        let co = transform.transform_point3(vertex.co);
        if !co.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "transformed source vertex is not finite",
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        destination.vertices.push(Vertex { id, co });
    }
    for edge in &source.edges {
        let id = remap_element_id(
            engine,
            edge.id,
            &mut used_edges,
            &mut destination.next_id.edge,
            "edge",
        )?;
        let vertices = [
            *mapping.vertices.get(&edge.vertices[0]).ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "source edge references a missing vertex",
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?,
            *mapping.vertices.get(&edge.vertices[1]).ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    "source edge references a missing vertex",
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?,
        ];
        mapping.edges.insert(edge.id, id);
        destination.edges.push(Edge { id, vertices });
    }
    let material_remap = merge_material_slots(
        destination_materials,
        destination_links,
        destination_data_materials,
        destination_effective_materials,
        source_slots,
        source_material_state,
    );
    for face in &source.faces {
        let id = remap_element_id(
            engine,
            face.id,
            &mut used_faces,
            &mut destination.next_id.face,
            "face",
        )?;
        let vertices = face
            .vertices
            .iter()
            .map(|vertex_id| {
                mapping.vertices.get(vertex_id).copied().ok_or_else(|| {
                    engine.error(
                        ErrorCode::SceneInvalid,
                        "source face references a missing vertex",
                        &operation_pointer(engine.operation_index, "target"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let material_index = match material_remap.get(face.material_index as usize) {
            Some(index) => u32::try_from(*index).map_err(|_| {
                engine.error(
                    ErrorCode::LimitExceeded,
                    "joined material slot index exceeds u32",
                    &operation_pointer(engine.operation_index, "target"),
                )
            })?,
            None if source_slots.slot_materials.is_empty() && face.material_index == 0 => 0,
            None => {
                return Err(engine.error(
                    ErrorCode::SceneInvalid,
                    format!(
                        "source face f{} references an unavailable material slot",
                        face.id
                    ),
                    &operation_pointer(engine.operation_index, "target"),
                ));
            }
        };
        mapping.faces.insert(face.id, id);
        destination.faces.push(Face {
            id,
            vertices,
            material_index,
        });
    }
    merge_uv_layers(destination, source, face_offset)?;
    merge_mesh_attributes(
        engine,
        &mut destination.attributes,
        &source.attributes,
        &mapping,
    )?;
    Ok(mapping)
}

fn remap_element_id(
    engine: &Engine<'_>,
    source_id: u32,
    used: &mut BTreeSet<u32>,
    next_id: &mut u32,
    domain: &str,
) -> Result<u32> {
    let id = if used.insert(source_id) {
        source_id
    } else {
        while used.contains(next_id) {
            *next_id = next_id.checked_add(1).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{domain} IDs are exhausted while joining meshes"),
                    &operation_pointer(engine.operation_index, "targets"),
                )
            })?;
        }
        let id = *next_id;
        used.insert(id);
        id
    };
    *next_id = (*next_id).max(id.checked_add(1).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{domain} IDs are exhausted while joining meshes"),
            &operation_pointer(engine.operation_index, "targets"),
        )
    })?);
    Ok(id)
}

fn merge_material_slots(
    destination_materials: &mut Vec<Option<Id>>,
    destination_links: &mut Vec<MaterialLink>,
    destination_data_materials: &mut Vec<Option<Id>>,
    destination_effective_materials: &mut Vec<Option<Id>>,
    source: &JoinInput,
    source_state: &SlotSources<'_>,
) -> Vec<usize> {
    source
        .slot_materials
        .iter()
        .enumerate()
        .map(|(slot, effective)| {
            let object_material = source_state.object.get(slot).cloned().unwrap_or(None);
            let data_material = source_state.data.get(slot).cloned().unwrap_or(None);
            let link = source_state.link(slot);
            if effective.is_some()
                && let Some(index) = destination_effective_materials.iter().enumerate().find_map(
                    |(index, candidate)| {
                        (candidate == effective
                            && destination_materials.get(index) == Some(&object_material)
                            && destination_links.get(index) == Some(&link)
                            && destination_data_materials.get(index) == Some(&data_material))
                        .then_some(index)
                    },
                )
            {
                return index;
            }
            destination_materials.push(object_material);
            destination_links.push(link);
            destination_data_materials.push(data_material);
            destination_effective_materials.push(effective.clone());
            destination_effective_materials.len() - 1
        })
        .collect()
}

fn validate_material_indices(engine: &Engine<'_>, input: &JoinInput, mesh: &Mesh) -> Result<()> {
    for face in &mesh.faces {
        let valid = if input.slot_materials.is_empty() {
            face.material_index == 0
        } else {
            (face.material_index as usize) < input.slot_materials.len()
        };
        if !valid {
            return Err(engine.error(
                ErrorCode::SceneInvalid,
                format!("face f{} references an unavailable material slot", face.id),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
    }
    Ok(())
}

/// Merges the source mesh's UV layers into `destination` by name. Source face `i` was appended at
/// storage index `face_offset + i`. A layer the destination lacks is appended (up to the layer
/// limit); a layer both share merges per face. The destination keeps its active and render layers,
/// or takes the source's when it had none.
fn merge_uv_layers(destination: &mut Mesh, source: &Mesh, face_offset: usize) -> Result<()> {
    let source_store = uv_layers::read(source);
    if source_store.layers.is_empty() {
        return Ok(());
    }
    let mut store = uv_layers::read(destination);
    let destination_was_empty = store.layers.is_empty();
    for source_layer in &source_store.layers {
        let index = if let Some(index) = store.index_of(&source_layer.name) {
            index
        } else {
            if store.layers.len() >= uv_layers::MAX_LAYERS {
                return Err(super::uv::layer_limit());
            }
            store.layers.push(UvLayer::new(
                source_layer.name.clone(),
                vec![None; destination.faces.len()],
            ));
            store.layers.len() - 1
        };
        let layer = &mut store.layers[index];
        for (source_index, (values, pins)) in source_layer
            .faces
            .iter()
            .zip(&source_layer.pinned)
            .enumerate()
        {
            let target = face_offset + source_index;
            layer.faces[target].clone_from(values);
            layer.pinned[target].clone_from(pins);
        }
    }
    if destination_was_empty {
        store.active = source_store.active;
        store.render = source_store.render;
    }
    uv_layers::write(destination, &store);
    Ok(())
}

fn merge_mesh_attributes(
    engine: &Engine<'_>,
    destination: &mut Map<String, Value>,
    source: &Map<String, Value>,
    mapping: &ElementMapping,
) -> Result<()> {
    for (name, source_attribute) in source {
        // UV layers merge by name in `merge_uv_layers`, which needs the face mapping.
        if name == uv_layers::ATTRIBUTE {
            continue;
        }
        let mut remapped = source_attribute.clone();
        remap_attribute(engine, name, &mut remapped, mapping)?;
        let Some(destination_attribute) = destination.get_mut(name) else {
            destination.insert(name.clone(), remapped);
            continue;
        };
        let (Some(destination_object), Some(source_object)) =
            (destination_attribute.as_object_mut(), remapped.as_object())
        else {
            return Err(attribute_error(
                engine,
                name,
                "mesh attributes must be objects",
            ));
        };
        let mut destination_metadata = destination_object.clone();
        let mut source_metadata = source_object.clone();
        let destination_values = destination_metadata.remove("values");
        let source_values = source_metadata.remove("values");
        if destination_metadata != source_metadata {
            return Err(attribute_error(
                engine,
                name,
                "same-named mesh attributes have incompatible domains or types",
            ));
        }
        let (Some(destination_values), Some(source_values)) = (destination_values, source_values)
        else {
            return Err(attribute_error(
                engine,
                name,
                "mesh attribute values must be present",
            ));
        };
        let (Some(_destination_values), Some(source_values)) =
            (destination_values.as_object(), source_values.as_object())
        else {
            return Err(attribute_error(
                engine,
                name,
                "mesh attribute values must be objects",
            ));
        };
        let destination_values = destination_object
            .get_mut("values")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                attribute_error(engine, name, "mesh attribute values must be objects")
            })?;
        for (key, value) in source_values {
            if let Some(existing) = destination_values.get(key) {
                if existing != value {
                    return Err(attribute_error(
                        engine,
                        name,
                        "mesh attribute value keys collide after joining",
                    ));
                }
            } else {
                destination_values.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(())
}

fn remap_attribute(
    engine: &Engine<'_>,
    name: &str,
    attribute: &mut Value,
    mapping: &ElementMapping,
) -> Result<()> {
    let Some(attribute_object) = attribute.as_object_mut() else {
        return Err(attribute_error(
            engine,
            name,
            "mesh attributes must be objects",
        ));
    };
    let domain = attribute_object
        .get("domain")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let Some(values) = attribute_object.get_mut("values") else {
        return Err(attribute_error(
            engine,
            name,
            "mesh attribute values must be present",
        ));
    };
    let Some(values) = values.as_object_mut() else {
        return Err(attribute_error(
            engine,
            name,
            "mesh attribute values must be objects",
        ));
    };
    let domain_mapping = match domain.as_deref() {
        Some("vertex" | "vertices" | "point" | "points") => Some((&mapping.vertices, 'v')),
        Some("edge" | "edges") => Some((&mapping.edges, 'e')),
        Some("face" | "faces" | "corner") => Some((&mapping.faces, 'f')),
        _ => None,
    };
    if let Some((domain_mapping, prefix)) = domain_mapping {
        let mut remapped_values = Map::new();
        for (key, value) in std::mem::take(values) {
            let remapped_key =
                remap_attribute_key(&key, prefix, domain_mapping).ok_or_else(|| {
                    attribute_error(
                        engine,
                        name,
                        "mesh attribute refers to a missing source element",
                    )
                })?;
            if let Some(existing) = remapped_values.get(&remapped_key) {
                if existing != &value {
                    return Err(attribute_error(
                        engine,
                        name,
                        "mesh attribute value keys collide after joining",
                    ));
                }
            } else {
                remapped_values.insert(remapped_key, value);
            }
        }
        *values = remapped_values;
    }
    Ok(())
}

fn remap_attribute_key(key: &str, prefix: char, mapping: &BTreeMap<u32, u32>) -> Option<String> {
    let Some(id_text) = key.strip_prefix(prefix) else {
        return Some(key.to_owned());
    };
    let Ok(id) = id_text.parse::<u32>() else {
        return Some(key.to_owned());
    };
    mapping.get(&id).map(|mapped| format!("{prefix}{mapped}"))
}

fn attribute_error(engine: &Engine<'_>, name: &str, message: &str) -> crate::error::PotError {
    engine.error(
        ErrorCode::InvalidOperation,
        format!("mesh attribute `{name}`: {message}"),
        &operation_pointer(engine.operation_index, "targets"),
    )
}

fn merge_vertex_groups(
    destination: &mut Vec<VertexGroup>,
    source: &[VertexGroup],
    source_node: &Id,
) -> BTreeMap<Id, Id> {
    let mut mapping = BTreeMap::new();
    for group in source {
        if let Some(existing) = destination
            .iter()
            .find(|candidate| candidate.id == group.id)
            && existing.name == group.name
        {
            mapping.insert(group.id.clone(), existing.id.clone());
            continue;
        }
        let id = if destination.iter().any(|candidate| candidate.id == group.id) {
            unique_group_id(destination, source_node, group)
        } else {
            group.id.clone()
        };
        mapping.insert(group.id.clone(), id.clone());
        destination.push(VertexGroup {
            id,
            name: group.name.clone(),
        });
    }
    mapping
}

fn unique_group_id(destination: &[VertexGroup], source_node: &Id, group: &VertexGroup) -> Id {
    let used: BTreeSet<_> = destination
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect();
    let base = format!("join_{source_node}_{}", group.id);
    for suffix_number in 0_u32.. {
        let suffix = if suffix_number == 0 {
            String::new()
        } else {
            format!("_{suffix_number}")
        };
        let prefix_len = 64_usize.saturating_sub(suffix.len());
        let mut candidate = base.chars().take(prefix_len).collect::<String>();
        candidate.push_str(&suffix);
        if let Ok(id) = Id::new(candidate)
            && !used.contains(&id)
        {
            return id;
        }
    }
    unreachable!("the ID space is finite and a unique group ID must exist")
}

fn merge_vertex_weights(
    engine: &Engine<'_>,
    destination: &mut BTreeMap<u32, BTreeMap<Id, f64>>,
    source: &BTreeMap<u32, BTreeMap<Id, f64>>,
    vertex_mapping: &BTreeMap<u32, u32>,
    group_mapping: &BTreeMap<Id, Id>,
    source_node: &Id,
) -> Result<()> {
    for (vertex_id, weights) in source {
        let mapped_vertex = vertex_mapping.get(vertex_id).copied().ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                format!("vertex weights on `{source_node}` refer to missing vertex v{vertex_id}"),
                &operation_pointer(engine.operation_index, "targets"),
            )
        })?;
        let output = destination.entry(mapped_vertex).or_default();
        for (group_id, weight) in weights {
            let mapped_group = group_mapping.get(group_id).cloned().ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    format!(
                        "vertex weights on `{source_node}` refer to missing group `{group_id}`"
                    ),
                    &operation_pointer(engine.operation_index, "targets"),
                )
            })?;
            output.insert(mapped_group, *weight);
        }
    }
    Ok(())
}

#[expect(
    clippy::float_cmp,
    reason = "shape-key timing and values must match exactly before merging"
)]
fn merge_shape_keys(
    engine: &Engine<'_>,
    destination: &mut Option<ShapeKeyData>,
    source: Option<&ShapeKeyData>,
    vertex_mapping: &BTreeMap<u32, u32>,
    group_mapping: &BTreeMap<Id, Id>,
    transform: DMat4,
    source_node: &Id,
) -> Result<()> {
    let Some(source) = source else {
        return Ok(());
    };
    let output = destination.get_or_insert_with(|| ShapeKeyData {
        absolute: source.absolute,
        evaluation_time: source.evaluation_time,
        action: source.action.clone(),
        action_slot: source.action_slot.clone(),
        muted_action_curves: source.muted_action_curves.clone(),
        ..ShapeKeyData::default()
    });
    if output.absolute != source.absolute || output.evaluation_time != source.evaluation_time {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "joined mesh shape-key timing settings are incompatible",
            &operation_pointer(engine.operation_index, "targets"),
        ));
    }
    if output.action != source.action
        || output.action_slot != source.action_slot
        || output.muted_action_curves != source.muted_action_curves
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "joined mesh shape-key animation actions are incompatible",
            &operation_pointer(engine.operation_index, "targets"),
        ));
    }
    for (vertex_id, position) in &source.basis {
        let mapped = vertex_mapping.get(vertex_id).copied().ok_or_else(|| {
            engine.error(
                ErrorCode::SceneInvalid,
                format!("shape-key basis on `{source_node}` refers to missing vertex v{vertex_id}"),
                &operation_pointer(engine.operation_index, "targets"),
            )
        })?;
        output
            .basis
            .insert(mapped, transform_position(engine, transform, *position)?);
    }

    let mut key_mapping = BTreeMap::new();
    for (key_id, key) in &source.keys {
        let group_id = key
            .vertex_group
            .as_ref()
            .and_then(|id| group_mapping.get(id))
            .cloned()
            .or_else(|| key.vertex_group.clone());
        let mapped_relative = key.relative_key.clone();
        let existing = output.keys.get(key_id);
        let mapped_id = if existing.is_none_or(|candidate| {
            candidate.name == key.name
                && candidate.value == key.value
                && candidate.slider_min == key.slider_min
                && candidate.slider_max == key.slider_max
                && candidate.relative_key == mapped_relative
                && candidate.vertex_group == group_id
                && candidate.frame == key.frame
        }) {
            key_id.clone()
        } else {
            unique_shape_key_id(&output.keys, source_node, key_id)
        };
        key_mapping.insert(key_id.clone(), mapped_id);
    }
    for (key_id, key) in &source.keys {
        let mapped_id = key_mapping.get(key_id).cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::InternalError,
                "shape-key ID mapping disappeared",
                &operation_pointer(engine.operation_index, "targets"),
            )
        })?;
        let mapped_group = key
            .vertex_group
            .as_ref()
            .and_then(|id| group_mapping.get(id))
            .cloned()
            .or_else(|| key.vertex_group.clone());
        let mapped_relative = key
            .relative_key
            .as_ref()
            .and_then(|id| key_mapping.get(id))
            .cloned()
            .or_else(|| key.relative_key.clone());
        let output_key = output
            .keys
            .entry(mapped_id.clone())
            .or_insert_with(|| ShapeKey {
                id: mapped_id,
                name: key.name.clone(),
                value: key.value,
                mute: key.mute,
                slider_min: key.slider_min,
                slider_max: key.slider_max,
                relative_key: mapped_relative.clone(),
                vertex_group: mapped_group.clone(),
                frame: key.frame,
                positions: BTreeMap::new(),
            });
        if output_key.relative_key != mapped_relative || output_key.vertex_group != mapped_group {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "joined mesh shape-key references are incompatible",
                &operation_pointer(engine.operation_index, "targets"),
            ));
        }
        for (vertex_id, position) in &key.positions {
            let mapped_vertex = vertex_mapping.get(vertex_id).copied().ok_or_else(|| {
                engine.error(
                    ErrorCode::SceneInvalid,
                    format!(
                        "shape key `{}` refers to missing vertex v{vertex_id}",
                        key.id
                    ),
                    &operation_pointer(engine.operation_index, "targets"),
                )
            })?;
            output_key.positions.insert(
                mapped_vertex,
                transform_position(engine, transform, *position)?,
            );
        }
    }
    Ok(())
}

fn unique_shape_key_id(keys: &BTreeMap<Id, ShapeKey>, source_node: &Id, key_id: &Id) -> Id {
    let used: BTreeSet<_> = keys.keys().cloned().collect();
    let base = format!("join_{source_node}_{key_id}");
    for suffix in 0_u32.. {
        let suffix = if suffix == 0 {
            String::new()
        } else {
            format!("_{suffix}")
        };
        let prefix_len = 64_usize.saturating_sub(suffix.len());
        let mut candidate = base.chars().take(prefix_len).collect::<String>();
        candidate.push_str(&suffix);
        if let Ok(id) = Id::new(candidate)
            && !used.contains(&id)
        {
            return id;
        }
    }
    unreachable!("the ID space is finite and a unique shape-key ID must exist")
}

fn transform_position(
    engine: &Engine<'_>,
    transform: DMat4,
    position: [f64; 3],
) -> Result<[f64; 3]> {
    let transformed = transform
        .transform_point3(DVec3::from_array(position))
        .to_array();
    if transformed.iter().all(|value| value.is_finite()) {
        Ok(transformed)
    } else {
        Err(engine.error(
            ErrorCode::InvalidOperation,
            "transformed shape-key position is not finite",
            &operation_pointer(engine.operation_index, "targets"),
        ))
    }
}

fn unique_join_data_id(engine: &Engine<'_>, node_id: &Id) -> Result<Id> {
    let suffix = "_joined";
    let prefix_length = 64_usize - suffix.len();
    let base = node_id
        .as_str()
        .chars()
        .take(prefix_length)
        .collect::<String>();
    for suffix_number in 0_u32.. {
        let suffix = if suffix_number == 0 {
            suffix.to_owned()
        } else {
            format!("_joined_{suffix_number}")
        };
        let prefix_length = 64_usize.saturating_sub(suffix.len());
        let mut candidate = node_id
            .as_str()
            .chars()
            .take(prefix_length)
            .collect::<String>();
        candidate.push_str(&suffix);
        if candidate.is_empty() {
            candidate.clone_from(&base);
            candidate.push_str(&suffix);
        }
        let id = parse_id(
            engine,
            &candidate,
            &operation_pointer(engine.operation_index, "target"),
        )?;
        if !engine.doc.data_blocks.contains_key(&id) {
            return Ok(id);
        }
    }
    unreachable!("the ID space is finite and a unique data-block ID must exist")
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "unit tests use fixed valid mesh and operation fixtures"
    )]
    use super::*;
    use crate::geom::IdCounters;
    use crate::model::SceneDoc;
    use crate::ops::ChangeSet;

    fn mesh(vertices: &[[f64; 3]], face_id: u32) -> Mesh {
        let vertices = vertices
            .iter()
            .enumerate()
            .map(|(index, position)| Vertex {
                id: index as u32,
                co: DVec3::from_array(*position),
            })
            .collect::<Vec<_>>();
        Mesh {
            vertices,
            edges: vec![
                Edge {
                    id: 0,
                    vertices: [0, 1],
                },
                Edge {
                    id: 1,
                    vertices: [1, 2],
                },
                Edge {
                    id: 2,
                    vertices: [2, 0],
                },
            ],
            faces: vec![Face {
                id: face_id,
                vertices: vec![0, 1, 2],
                material_index: 0,
            }],
            next_id: IdCounters {
                vertex: 3,
                edge: 3,
                face: face_id + 1,
            },
            ..Mesh::default()
        }
    }

    fn add_mesh(
        doc: &mut SceneDoc,
        node_id: &str,
        data_id: &str,
        mesh: Mesh,
        translation: [f64; 3],
    ) {
        let node_id = Id::new(node_id).unwrap();
        let data_id = Id::new(data_id).unwrap();
        let mut node = Node {
            kind: "mesh".to_owned(),
            data: Some(data_id.clone()),
            ..Node::default()
        };
        node.transform.translation = translation;
        doc.nodes.insert(node_id, node);
        doc.data_blocks.insert(
            data_id,
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
    }
    fn add_material(doc: &mut SceneDoc, id: &str) {
        let id = Id::new(id).unwrap();
        doc.materials.insert(id, crate::model::Material::default());
    }

    #[test]
    fn joins_transformed_meshes_remaps_colliding_ids_and_tracks_deletion() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        add_mesh(
            &mut doc,
            "target",
            "target_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [0.0, 0.0, 0.0],
        );
        add_mesh(
            &mut doc,
            "source",
            "source_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [2.0, 0.0, 0.0],
        );
        doc.data_blocks
            .get_mut(&Id::new("target_data").unwrap())
            .unwrap()
            .mesh
            .as_mut()
            .unwrap()
            .attributes
            .insert(
                "weights".to_owned(),
                json!({"domain":"vertices","type":"float","values":{"v0":1.0}}),
            );
        let source_mesh = doc
            .data_blocks
            .get_mut(&Id::new("source_data").unwrap())
            .unwrap()
            .mesh
            .as_mut()
            .unwrap();
        source_mesh.attributes.insert(
            "weights".to_owned(),
            json!({"domain":"vertices","type":"float","values":{"v0":2.0}}),
        );
        source_mesh.attributes.insert(
            "uv_layers".to_owned(),
            json!({
                "layers": [{"name": "UVMap", "faces": [{"face_id": 0, "uv": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}]}],
                "active": "UVMap",
                "render": "UVMap"
            }),
        );
        add_material(&mut doc, "target_material");
        add_material(&mut doc, "source_material");
        doc.nodes
            .get_mut(&Id::new("target").unwrap())
            .unwrap()
            .materials
            .push(Some(Id::new("target_material").unwrap()));
        doc.nodes
            .get_mut(&Id::new("source").unwrap())
            .unwrap()
            .materials
            .push(Some(Id::new("source_material").unwrap()));
        let root_collection = doc
            .collections
            .get_mut(&Id::new("collection_root").unwrap())
            .unwrap();
        root_collection.objects = vec![Id::new("target").unwrap(), Id::new("source").unwrap()];
        let operation = serde_json::from_value::<Value>(json!({
            "op":"node.join",
            "targets":["target","source"]
        }))
        .unwrap();
        let mut engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            cache_directory: None,
            pending_assets: BTreeMap::default(),
            warnings: Vec::new(),
        };
        assert!(apply(&mut engine, operation.as_object().unwrap()).unwrap());
        let target = Id::new("target").unwrap();
        let source = Id::new("source").unwrap();
        let data_id = engine.doc.nodes[&target].data.as_ref().unwrap();
        let joined = engine.doc.data_blocks[data_id].mesh.as_ref().unwrap();
        assert_eq!(joined.vertices.len(), 6);
        assert_eq!(joined.vertices[3].co, DVec3::new(2.0, 0.0, 0.0));
        assert_eq!(joined.vertices[3].id, 3);
        assert_eq!(joined.faces[1].id, 1);
        assert_eq!(joined.attributes["weights"]["values"]["v3"], json!(2.0));
        assert_eq!(
            joined.attributes["uv_layers"]["layers"][0]["faces"][0]["face_id"],
            json!(1)
        );
        assert_eq!(
            engine.doc.nodes[&target].materials[1],
            Some(Id::new("source_material").unwrap())
        );
        assert_eq!(joined.faces[1].material_index, 1);
        assert!(!engine.doc.nodes.contains_key(&source));
        assert!(
            !engine
                .doc
                .data_blocks
                .contains_key(&Id::new("source_data").unwrap())
        );
        assert!(
            !engine.doc.collections[&Id::new("collection_root").unwrap()]
                .objects
                .contains(&source)
        );
        assert_eq!(engine.id_mappings["nodes"]["source"], json!("target"));
        assert_eq!(
            engine.id_mappings["mesh_elements"]["source"]["vertices"]["v0"],
            json!("v3")
        );
    }

    #[test]
    fn joining_shared_destination_data_makes_only_the_destination_single_user() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        add_mesh(
            &mut doc,
            "target",
            "target_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [0.0; 3],
        );
        add_mesh(
            &mut doc,
            "source",
            "source_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [1.0, 0.0, 0.0],
        );
        doc.nodes.insert(
            Id::new("other").unwrap(),
            Node {
                kind: "mesh".to_owned(),
                data: Some(Id::new("target_data").unwrap()),
                ..Node::default()
            },
        );
        let operation = json!({"op":"node.join","targets":["target","source"]});
        let mut engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            cache_directory: None,
            pending_assets: BTreeMap::default(),
            warnings: Vec::new(),
        };
        assert!(apply(&mut engine, operation.as_object().unwrap()).unwrap());
        let target_data = engine.doc.nodes[&Id::new("target").unwrap()]
            .data
            .as_ref()
            .unwrap();
        assert_eq!(target_data, &Id::new("target_joined").unwrap());
        assert_eq!(
            engine.doc.data_blocks[&Id::new("target_data").unwrap()]
                .mesh
                .as_ref()
                .unwrap()
                .vertices
                .len(),
            3
        );
        assert_eq!(
            engine.doc.data_blocks[target_data]
                .mesh
                .as_ref()
                .unwrap()
                .vertices
                .len(),
            6
        );
        assert_eq!(
            engine.doc.nodes[&Id::new("other").unwrap()].data.as_ref(),
            Some(&Id::new("target_data").unwrap())
        );
    }

    #[test]
    fn join_preserves_data_object_links_and_empty_slot_indices() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        add_mesh(
            &mut doc,
            "target",
            "target_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [0.0; 3],
        );
        add_mesh(
            &mut doc,
            "source",
            "source_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [1.0, 0.0, 0.0],
        );
        let material_ids = [
            "data_blue",
            "data_gold",
            "source_data_gold",
            "source_data_fallback",
            "target_hidden_object",
            "target_red",
            "source_hidden_object",
        ];
        for id in material_ids {
            add_material(&mut doc, id);
        }
        let id = |text| Id::new(text).unwrap();
        let target_data = doc.data_blocks.get_mut(&id("target_data")).unwrap();
        target_data.materials = vec![Some(id("data_blue")), None, Some(id("data_gold"))];
        let source_data = doc.data_blocks.get_mut(&id("source_data")).unwrap();
        source_data.materials = vec![
            Some(id("source_data_gold")),
            Some(id("source_data_fallback")),
        ];
        let target = doc.nodes.get_mut(&id("target")).unwrap();
        target.materials = vec![Some(id("target_hidden_object")), Some(id("target_red"))];
        target.material_links = vec![MaterialLink::Data, MaterialLink::Object];
        let source = doc.nodes.get_mut(&id("source")).unwrap();
        source.materials = vec![Some(id("source_hidden_object")), None];
        source.material_links = vec![MaterialLink::Data, MaterialLink::Object];
        doc.data_blocks
            .get_mut(&id("source_data"))
            .unwrap()
            .mesh
            .as_mut()
            .unwrap()
            .faces[0]
            .material_index = 1;
        doc.collections
            .get_mut(&id("collection_root"))
            .unwrap()
            .objects = vec![id("target"), id("source")];

        let operation = json!({"op":"node.join","targets":["target","source"]});
        let mut engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            cache_directory: None,
            pending_assets: BTreeMap::default(),
            warnings: Vec::new(),
        };
        assert!(apply(&mut engine, operation.as_object().unwrap()).unwrap());

        let target = &engine.doc.nodes[&id("target")];
        let data_id = target.data.as_ref().unwrap();
        let data = &engine.doc.data_blocks[data_id];
        assert_eq!(
            data.materials,
            vec![
                Some(id("data_blue")),
                None,
                Some(id("data_gold")),
                Some(id("source_data_gold")),
                Some(id("source_data_fallback")),
            ]
        );
        assert_eq!(
            target.materials,
            vec![
                Some(id("target_hidden_object")),
                Some(id("target_red")),
                None,
                Some(id("source_hidden_object")),
                None,
            ]
        );
        assert_eq!(
            target.material_links,
            vec![
                MaterialLink::Data,
                MaterialLink::Object,
                MaterialLink::Data,
                MaterialLink::Data,
                MaterialLink::Object,
            ]
        );
        assert_eq!(target.material_slot(Some(data), 0), Some(&id("data_blue")));
        assert_eq!(target.material_slot(Some(data), 1), Some(&id("target_red")));
        assert_eq!(
            target.material_slot(Some(data), 4),
            Some(&id("source_data_fallback"))
        );
        assert_eq!(data.mesh.as_ref().unwrap().faces[1].material_index, 4);
    }

    #[test]
    fn rejects_joining_source_with_surviving_children_without_mutating() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        add_mesh(
            &mut doc,
            "target",
            "target_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [0.0; 3],
        );
        add_mesh(
            &mut doc,
            "source",
            "source_data",
            mesh(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0),
            [0.0; 3],
        );
        doc.nodes.insert(
            Id::new("child").unwrap(),
            Node {
                parent: Some(Id::new("source").unwrap()),
                ..Node::default()
            },
        );
        let operation = json!({"op":"node.join","targets":["target","source"]});
        let mut engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            cache_directory: None,
            pending_assets: BTreeMap::default(),
            warnings: Vec::new(),
        };
        assert!(apply(&mut engine, operation.as_object().unwrap()).is_err());
        assert!(engine.doc.nodes.contains_key(&Id::new("source").unwrap()));
    }
}
