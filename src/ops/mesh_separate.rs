use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Edge, Face, Mesh, Vertex},
    model::{DataBlock, Id, Node},
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, resolve_node_targets};

pub(super) fn apply(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "mode", "elements"],
        &["target", "mode"],
    )?;
    let mode = operation
        .get("mode")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "mode must be a string",
                &operation_pointer(engine.operation_index, "mode"),
            )
        })?;
    if !["selection", "material", "loose_parts"].contains(&mode) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mode must be selection, material, or loose_parts",
            &operation_pointer(engine.operation_index, "mode"),
        ));
    }
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        &operation_pointer(engine.operation_index, "scope"),
        "scope must be a string",
        "scope must be shared or single_user",
        true,
    )?;

    let target_value = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target is required",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let Some(target_fields) = target_value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            &operation_pointer(engine.operation_index, "target"),
        ));
    };
    let mut target = target_fields.clone();
    let nested_elements = target.remove("elements");
    let elements = match (operation.get("elements"), nested_elements) {
        (Some(_), Some(_)) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "elements must be specified either at top level or in target, not both",
                &operation_pointer(engine.operation_index, "elements"),
            ));
        }
        (Some(elements), None) => Some(elements.clone()),
        (None, elements) => elements,
    };
    if mode == "selection" && elements.is_none() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "selection mode requires elements",
            &operation_pointer(engine.operation_index, "elements"),
        ));
    }
    if mode != "selection" && elements.is_some() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements can only be specified in selection mode",
            &operation_pointer(engine.operation_index, "elements"),
        ));
    }
    let targets = resolve_node_targets(engine, &Value::Object(target), true)?;
    let selected_faces = if mode == "selection" {
        Some(parse_face_selection(
            engine,
            elements.as_ref().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "selection mode requires elements",
                    &operation_pointer(engine.operation_index, "elements"),
                )
            })?,
        )?)
    } else {
        None
    };

    let mut sources = Vec::with_capacity(targets.len());
    for node_id in targets {
        let node = engine.doc.nodes.get(&node_id).cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("node `{node_id}` was not found"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        let data_id = node.data.clone().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("node `{node_id}` has no geometry data"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        let data = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                format!("data block `{data_id}` was not found"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        let mesh = data.mesh.as_ref().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("data block `{data_id}` has no mesh payload"),
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
        if data.data_type != "mesh" {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("data block `{data_id}` is not a mesh"),
                &operation_pointer(engine.operation_index, "target"),
            ));
        }
        let partitions = match mode {
            "selection" => selection_partitions(
                engine,
                mesh,
                selected_faces.as_ref().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "face selection disappeared")
                })?,
            )?,
            "material" => material_partitions(mesh),
            "loose_parts" => loose_partitions(mesh),
            _ => unreachable!("mode was validated"),
        };
        sources.push(Source {
            node_id,
            node,
            data_id,
            data: data.clone(),
            partitions,
        });
    }

    let has_shared_data = sources
        .iter()
        .any(|source| super::data_user_count(engine, &source.data_id) > 1);
    if has_shared_data && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry separation requires scope",
            &operation_pointer(engine.operation_index, "scope"),
        ));
    }

    let mut separated = false;
    let mut shared_data = HashSet::new();
    for source in sources {
        if source.partitions.detached_faces.is_empty() {
            continue;
        }
        if scope == Some("shared") && !shared_data.insert(source.data_id.clone()) {
            continue;
        }

        let original_data_id = source.data_id.clone();
        let mut data_id = original_data_id.clone();
        if scope == Some("single_user") && super::data_user_count(engine, &data_id) > 1 {
            let cloned_id = super::unique_id(
                engine,
                &format!("{}_mesh_separate_single", source.node_id),
                super::IdRegistry::DataBlocks,
            )?;
            engine
                .doc
                .data_blocks
                .insert(cloned_id.clone(), source.data.clone());
            if let Some(node) = engine.doc.nodes.get_mut(&source.node_id) {
                node.data = Some(cloned_id.clone());
            }
            engine.mark("data_blocks", &cloned_id, ChangeKind::Created);
            engine.mark("nodes", &source.node_id, ChangeKind::Updated);
            engine.map_id("data_blocks", &original_data_id, &cloned_id);
            data_id = cloned_id;
        }

        let source_mesh =
            source.data.mesh.as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "source mesh disappeared")
            })?;
        let remainder = partition_mesh(source_mesh, &source.partitions.source_faces, true);
        let mut remainder_data = source.data.clone();
        remainder_data.mesh = Some(remainder.clone());
        filter_data_block_attributes(&mut remainder_data, &remainder);
        let had_descriptor = remainder_data.descriptor.take().is_some();
        let missing_data_error = engine.error(
            ErrorCode::TargetNotFound,
            format!("data block `{data_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        );
        let destination = engine
            .doc
            .data_blocks
            .get_mut(&data_id)
            .ok_or(missing_data_error)?;
        *destination = remainder_data;
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
        if had_descriptor {
            engine
                .id_mappings
                .entry("mesh_descriptor_dropped".to_owned())
                .or_insert_with(|| json!({}))[data_id.as_str()] = Value::Bool(true);
        }
        super::clear_primitive_metadata(engine, &data_id);

        let mut changed_collections = BTreeSet::new();
        for (index, face_ids) in source.partitions.detached_faces.iter().enumerate() {
            let node_id = super::unique_id(
                engine,
                &format!("{}_separated_{}", source.node_id, index + 1),
                super::IdRegistry::Nodes,
            )?;
            let data_block_id = super::unique_id(
                engine,
                &format!("{node_id}_mesh"),
                super::IdRegistry::DataBlocks,
            )?;
            let mut data_block = source.data.clone();
            let part_mesh = partition_mesh(source_mesh, face_ids, false);
            data_block.mesh = Some(part_mesh.clone());
            filter_data_block_attributes(&mut data_block, &part_mesh);
            let part_had_descriptor = data_block.descriptor.take().is_some();
            engine
                .doc
                .data_blocks
                .insert(data_block_id.clone(), data_block);
            engine.mark("data_blocks", &data_block_id, ChangeKind::Created);
            engine.map_id("data_blocks", &original_data_id, &data_block_id);
            if part_had_descriptor {
                engine
                    .id_mappings
                    .entry("mesh_descriptor_dropped".to_owned())
                    .or_insert_with(|| json!({}))[data_block_id.as_str()] = Value::Bool(true);
            }

            let mut separated_node = source.node.clone();
            separated_node.name = format!("{} Separated {}", source.node.name, index + 1);
            separated_node.primitive = None;
            separated_node.data = Some(data_block_id);
            engine.doc.nodes.insert(node_id.clone(), separated_node);
            engine.mark("nodes", &node_id, ChangeKind::Created);
            engine.map_id("nodes", &source.node_id, &node_id);
            for (collection_id, collection) in &mut engine.doc.collections {
                if collection.objects.contains(&source.node_id)
                    && !collection.objects.contains(&node_id)
                {
                    collection.objects.push(node_id.clone());
                    collection.objects.sort();
                    changed_collections.insert(collection_id.clone());
                }
            }
        }
        for collection_id in changed_collections {
            engine.mark("collections", &collection_id, ChangeKind::Updated);
        }
        separated = true;
    }
    Ok(separated)
}

struct Source {
    node_id: Id,
    node: Node,
    data_id: Id,
    data: DataBlock,
    partitions: Partitions,
}

struct Partitions {
    source_faces: BTreeSet<u32>,
    detached_faces: Vec<BTreeSet<u32>>,
}

fn selection_partitions(
    engine: &Engine<'_>,
    mesh: &Mesh,
    selected: &BTreeSet<u32>,
) -> Result<Partitions> {
    for id in selected {
        if !mesh.faces.iter().any(|face| face.id == *id) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("face ID f{id} does not exist"),
                &operation_pointer(engine.operation_index, "elements"),
            ));
        }
    }
    let all_faces: BTreeSet<_> = mesh.faces.iter().map(|face| face.id).collect();
    let source_faces = all_faces.difference(selected).copied().collect();
    Ok(Partitions {
        source_faces,
        detached_faces: vec![selected.clone()],
    })
}

fn material_partitions(mesh: &Mesh) -> Partitions {
    let mut by_material = BTreeMap::<u32, BTreeSet<u32>>::new();
    for face in &mesh.faces {
        by_material
            .entry(face.material_index)
            .or_default()
            .insert(face.id);
    }
    grouped_partitions(by_material.into_values().collect())
}

fn loose_partitions(mesh: &Mesh) -> Partitions {
    let mut adjacency = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut edge_faces = BTreeMap::<(u32, u32), Vec<u32>>::new();
    for face in &mesh.faces {
        adjacency.entry(face.id).or_default();
        for pair in face.vertices.iter().copied().zip(
            face.vertices
                .iter()
                .copied()
                .cycle()
                .skip(1)
                .take(face.vertices.len()),
        ) {
            edge_faces
                .entry(crate::geom::edge_key(pair.0, pair.1))
                .or_default()
                .push(face.id);
        }
    }
    for faces in edge_faces.values() {
        for first in faces {
            for second in faces {
                if first != second {
                    adjacency.entry(*first).or_default().insert(*second);
                }
            }
        }
    }

    let mut remaining: BTreeSet<u32> = adjacency.keys().copied().collect();
    let mut groups = Vec::new();
    while let Some(seed) = remaining.first().copied() {
        remaining.remove(&seed);
        let mut pending = vec![seed];
        let mut group = BTreeSet::from([seed]);
        while let Some(face_id) = pending.pop() {
            if let Some(neighbors) = adjacency.get(&face_id) {
                for neighbor in neighbors {
                    if remaining.remove(neighbor) {
                        group.insert(*neighbor);
                        pending.push(*neighbor);
                    }
                }
            }
        }
        groups.push(group);
    }
    grouped_partitions(groups)
}

fn grouped_partitions(mut groups: Vec<BTreeSet<u32>>) -> Partitions {
    groups.retain(|group| !group.is_empty());
    if groups.len() <= 1 {
        return Partitions {
            source_faces: groups.into_iter().next().unwrap_or_default(),
            detached_faces: Vec::new(),
        };
    }
    let source_faces = groups.remove(0);
    Partitions {
        source_faces,
        detached_faces: groups,
    }
}

fn parse_face_selection(engine: &Engine<'_>, value: &Value) -> Result<BTreeSet<u32>> {
    let pointer = operation_pointer(engine.operation_index, "elements");
    let Some(fields) = value.as_object() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements must be an object",
            &pointer,
        ));
    };
    for key in fields.keys() {
        if !["domain", "ids"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown elements field `{key}`"),
                &format!("{pointer}/{}", super::pointer_escape(key)),
            ));
        }
    }
    if fields.get("domain").and_then(Value::as_str) != Some("face") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "selection mode requires elements.domain `face`",
            &format!("{pointer}/domain"),
        ));
    }
    let ids = fields.get("ids").and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "elements.ids must be an array of persistent face IDs",
            &format!("{pointer}/ids"),
        )
    })?;
    if ids.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements.ids must not be empty",
            &format!("{pointer}/ids"),
        ));
    }
    let mut selected = BTreeSet::new();
    for value in ids {
        let value = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "face IDs must be strings such as `f12`",
                &format!("{pointer}/ids"),
            )
        })?;
        let digits = value
            .strip_prefix('f')
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit()))
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("face ID {value:?} must use the f prefix"),
                    &format!("{pointer}/ids"),
                )
            })?;
        let id = digits.parse::<u32>().map_err(|_| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("face ID {value:?} is outside the u32 range"),
                &format!("{pointer}/ids"),
            )
        })?;
        if !selected.insert(id) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("duplicate face ID {value:?}"),
                &format!("{pointer}/ids"),
            ));
        }
    }
    Ok(selected)
}

fn partition_mesh(source: &Mesh, face_ids: &BTreeSet<u32>, include_loose: bool) -> Mesh {
    let selected_faces: Vec<Face> = source
        .faces
        .iter()
        .filter(|face| face_ids.contains(&face.id))
        .cloned()
        .collect();
    let selected_face_ids: BTreeSet<_> = selected_faces.iter().map(|face| face.id).collect();
    let selected_edge_keys = face_edge_keys(&selected_faces);
    let all_face_edge_keys = face_edge_keys(&source.faces);
    let all_face_vertices: BTreeSet<_> = source
        .faces
        .iter()
        .flat_map(|face| face.vertices.iter().copied())
        .collect();
    let mut vertex_ids: BTreeSet<_> = selected_faces
        .iter()
        .flat_map(|face| face.vertices.iter().copied())
        .collect();
    if include_loose {
        vertex_ids.extend(
            source
                .vertices
                .iter()
                .filter(|vertex| !all_face_vertices.contains(&vertex.id))
                .map(|vertex| vertex.id),
        );
        for edge in &source.edges {
            if !all_face_edge_keys
                .contains(&crate::geom::edge_key(edge.vertices[0], edge.vertices[1]))
            {
                vertex_ids.extend(edge.vertices);
            }
        }
    }
    let edges: Vec<Edge> = source
        .edges
        .iter()
        .filter(|edge| {
            let key = crate::geom::edge_key(edge.vertices[0], edge.vertices[1]);
            selected_edge_keys.contains(&key)
                || (include_loose && !all_face_edge_keys.contains(&key))
        })
        .cloned()
        .collect();
    let edge_ids: BTreeSet<_> = edges.iter().map(|edge| edge.id).collect();
    let vertices: Vec<Vertex> = source
        .vertices
        .iter()
        .filter(|vertex| vertex_ids.contains(&vertex.id))
        .cloned()
        .collect();

    let mut result = source.clone();
    result.vertices = vertices;
    result.edges = edges;
    result.faces = selected_faces;
    filter_mesh_attributes(&mut result, &vertex_ids, &edge_ids, &selected_face_ids);
    result
}

fn face_edge_keys(faces: &[Face]) -> BTreeSet<(u32, u32)> {
    faces
        .iter()
        .flat_map(|face| {
            face.vertices
                .iter()
                .copied()
                .zip(
                    face.vertices
                        .iter()
                        .copied()
                        .cycle()
                        .skip(1)
                        .take(face.vertices.len()),
                )
                .map(|(first, second)| crate::geom::edge_key(first, second))
        })
        .collect()
}

fn filter_mesh_attributes(
    mesh: &mut Mesh,
    vertex_ids: &BTreeSet<u32>,
    edge_ids: &BTreeSet<u32>,
    face_ids: &BTreeSet<u32>,
) {
    for attribute in mesh
        .attributes
        .values_mut()
        .filter_map(Value::as_object_mut)
    {
        let Some(domain) = attribute.get("domain").and_then(Value::as_str) else {
            continue;
        };
        let (prefix, ids) = match domain {
            "point" | "vertex" | "vertices" => ('v', vertex_ids),
            "edge" | "edges" => ('e', edge_ids),
            "face" | "faces" | "corner" => ('f', face_ids),
            _ => continue,
        };
        let Some(values) = attribute.get_mut("values").and_then(Value::as_object_mut) else {
            continue;
        };
        values.retain(|key, _| {
            key.strip_prefix(prefix)
                .and_then(|id| id.parse::<u32>().ok())
                .is_some_and(|id| ids.contains(&id))
        });
    }
}

fn filter_data_block_attributes(data: &mut DataBlock, mesh: &Mesh) {
    let vertex_ids: BTreeSet<_> = mesh.vertices.iter().map(|vertex| vertex.id).collect();
    data.vertex_weights
        .retain(|vertex_id, _| vertex_ids.contains(vertex_id));
    if let Some(shape_keys) = &mut data.shape_keys {
        shape_keys
            .basis
            .retain(|vertex_id, _| vertex_ids.contains(vertex_id));
        for key in shape_keys.keys.values_mut() {
            key.positions
                .retain(|vertex_id, _| vertex_ids.contains(vertex_id));
        }
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::super::{IdRegistry, unique_id};
    use super::{apply, loose_partitions, material_partitions, partition_mesh};
    use crate::{
        geom::{Edge, Face, Mesh, Vertex},
        model::{DataBlock, Id, Node, SceneDoc},
        ops::{ChangeSet, Engine},
    };
    use serde_json::{Map, json};
    use std::collections::{BTreeMap, BTreeSet};

    fn two_triangle_mesh() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex { id: 10, co: glam::DVec3::new(0.0, 0.0, 0.0) },
                Vertex { id: 11, co: glam::DVec3::new(1.0, 0.0, 0.0) },
                Vertex { id: 12, co: glam::DVec3::new(0.0, 1.0, 0.0) },
                Vertex { id: 20, co: glam::DVec3::new(3.0, 0.0, 0.0) },
                Vertex { id: 21, co: glam::DVec3::new(4.0, 0.0, 0.0) },
                Vertex { id: 22, co: glam::DVec3::new(3.0, 1.0, 0.0) },
            ],
            edges: vec![
                Edge { id: 5, vertices: [10, 11] },
                Edge { id: 6, vertices: [11, 12] },
                Edge { id: 7, vertices: [10, 12] },
                Edge { id: 15, vertices: [20, 21] },
                Edge { id: 16, vertices: [21, 22] },
                Edge { id: 17, vertices: [20, 22] },
            ],
            faces: vec![
                Face { id: 30, vertices: vec![10, 11, 12], material_index: 2 },
                Face { id: 40, vertices: vec![20, 21, 22], material_index: 0 },
            ],
            attributes: json!({
                "point_attr":{"domain":"point","type":"float","values":{"v10":1,"v11":2,"v12":3,"v20":4,"v21":5,"v22":6}},
                "face_attr":{"domain":"face","type":"int","values":{"f30":8,"f40":9}}
            }).as_object().unwrap().clone(),
            next_id: crate::geom::IdCounters { vertex: 23, edge: 18, face: 41 },
        }
    }

    #[test]
    fn material_and_loose_partitions_are_deterministic_and_keep_persistent_topology() {
        let mesh = two_triangle_mesh();
        let material = material_partitions(&mesh);
        assert_eq!(material.source_faces, BTreeSet::from([40]));
        assert_eq!(material.detached_faces, vec![BTreeSet::from([30])]);
        let loose = loose_partitions(&mesh);
        assert_eq!(loose.source_faces, BTreeSet::from([30]));
        assert_eq!(loose.detached_faces, vec![BTreeSet::from([40])]);

        let part = partition_mesh(&mesh, &BTreeSet::from([30]), false);
        assert_eq!(
            part.vertices
                .iter()
                .map(|vertex| vertex.id)
                .collect::<Vec<_>>(),
            [10, 11, 12]
        );
        assert_eq!(
            part.edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
            [5, 6, 7]
        );
        assert_eq!(
            part.faces.iter().map(|face| face.id).collect::<Vec<_>>(),
            [30]
        );
        assert_eq!(
            part.attributes["point_attr"]["values"]
                .as_object()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(part.attributes["face_attr"]["values"]["f30"], 8);
        assert!(part.attributes["face_attr"]["values"].get("f40").is_none());
        assert_eq!(part.next_id, mesh.next_id);
    }

    #[test]
    fn selection_separates_faces_into_a_mapped_collection_member() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let node_id = Id::new("mesh_source").unwrap();
        let data_id = Id::new("mesh_source_data").unwrap();
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Source".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id.clone()),
                ..Node::default()
            },
        );
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                mesh: Some(two_triangle_mesh()),
                ..DataBlock::default()
            },
        );
        doc.collections
            .get_mut(&Id::new("collection_root").unwrap())
            .unwrap()
            .objects
            .push(node_id.clone());
        let mut engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            pending_assets: BTreeMap::new(),
        };
        let operation = json!({
            "op":"mesh.separate",
            "target":{"id":"mesh_source"},
            "mode":"selection",
            "elements":{"domain":"face","ids":["f30"]}
        })
        .as_object()
        .unwrap()
        .clone();

        assert!(apply(&mut engine, &operation).unwrap());
        let part_node_id =
            Id::new(engine.id_mappings["nodes"]["mesh_source"].as_str().unwrap()).unwrap();
        let part_data_id = Id::new(
            engine.id_mappings["data_blocks"]["mesh_source_data"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            engine.doc.data_blocks[&data_id]
                .mesh
                .as_ref()
                .unwrap()
                .faces
                .iter()
                .map(|face| face.id)
                .collect::<Vec<_>>(),
            [40]
        );
        let part_mesh = engine.doc.data_blocks[&part_data_id].mesh.as_ref().unwrap();
        assert_eq!(
            part_mesh
                .faces
                .iter()
                .map(|face| face.id)
                .collect::<Vec<_>>(),
            [30]
        );
        assert_eq!(part_mesh.attributes["face_attr"]["values"]["f30"], 8);
        assert_eq!(
            engine.doc.nodes[&part_node_id].transform,
            engine.doc.nodes[&node_id].transform
        );
        assert!(
            engine.doc.collections[&Id::new("collection_root").unwrap()]
                .objects
                .contains(&part_node_id)
        );
    }

    #[test]
    fn loose_topology_stays_with_source_remainder() {
        let mut mesh = two_triangle_mesh();
        mesh.vertices.push(Vertex {
            id: 50,
            co: glam::DVec3::ZERO,
        });
        mesh.edges.push(Edge {
            id: 30,
            vertices: [10, 50],
        });
        mesh.next_id.vertex = 51;
        mesh.next_id.edge = 31;
        let source = partition_mesh(&mesh, &BTreeSet::from([30]), true);
        assert_eq!(
            source
                .vertices
                .iter()
                .map(|vertex| vertex.id)
                .collect::<Vec<_>>(),
            [10, 11, 12, 50]
        );
        assert_eq!(
            source.edges.iter().map(|edge| edge.id).collect::<Vec<_>>(),
            [5, 6, 7, 30]
        );
        assert_eq!(
            source.faces.iter().map(|face| face.id).collect::<Vec<_>>(),
            [30]
        );
    }

    #[test]
    fn generated_ids_avoid_collisions_and_fit_id_limits() {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let engine = Engine {
            doc: &mut doc,
            changes: ChangeSet::default(),
            id_mappings: Map::new(),
            operation_index: 0,
            evaluation_frame: 1.0,
            asset_root: None,
            pending_assets: BTreeMap::new(),
        };
        let occupied = Id::new("mesh_separated_1").unwrap();
        engine.doc.nodes.insert(occupied, Node::default());
        let id = unique_id(&engine, "mesh_separated_1", IdRegistry::Nodes).unwrap();
        assert_eq!(id.as_str(), "mesh_separated_1_1");
        assert!(id.as_str().len() <= 64);
    }
}
