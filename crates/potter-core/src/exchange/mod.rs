use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DVec3};
use serde::Serialize;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
    model::{DataBlock, Id, Node, SceneDoc, Transform},
};

#[derive(Debug, Clone, Serialize)]
pub struct Loss {
    pub feature_id: String,
    pub data_id: Option<String>,
    pub reason: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub files: Vec<std::path::PathBuf>,
    pub counts: serde_json::Value,
    pub conversions: Vec<serde_json::Value>,
    pub losses: Vec<Loss>,
}

pub struct ImportedGraph {
    pub doc: SceneDoc,
    pub losses: Vec<Loss>,
    pub id_mappings: serde_json::Value,
    pub compat_blobs: Vec<(String, Vec<u8>)>,
    pub assets: Vec<(String, Vec<u8>)>,
    pub source: serde_json::Value,
}

pub(crate) fn is_default_material_graph(doc: &SceneDoc, graph_id: &Id) -> bool {
    doc.node_groups
        .get(graph_id)
        .is_some_and(crate::graph::is_generated_simple_material_graph)
}

pub struct ExportOptions<'a> {
    pub allow_lossy: bool,
    pub pack: bool,
    pub blender: Option<&'a std::path::Path>,
    pub context: &'a crate::eval::EvaluationContext,
}

/// Geometry for a single exported node in world-space coordinates.
#[derive(Clone, Debug)]
pub(crate) struct ExchangeMesh {
    pub id: String,
    pub name: String,
    pub positions: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
}

pub(crate) fn evaluated_meshes(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
) -> Result<Vec<ExchangeMesh>> {
    let visible_nodes = render_visible_nodes(doc, snapshot)?;
    let mut output = Vec::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible
            || !node.render_visible
            || node.data.is_none()
            || !visible_nodes.contains(node_id)
        {
            continue;
        }
        let Some(source) = snapshot.meshes.get(node_id) else {
            continue;
        };
        let matrix = snapshot
            .nodes
            .get(node_id)
            .map(|state| DMat4::from_cols_array(&state.world_matrix))
            .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "node evaluation missing"))?;
        let positions = source
            .vertices
            .iter()
            .map(|vertex| matrix.transform_point3(vertex.co).to_array())
            .collect::<Vec<_>>();
        let vertex_indices = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let mut faces = Vec::with_capacity(source.faces.len());
        for face in &source.faces {
            let Some(indices) = face
                .vertices
                .iter()
                .map(|id| vertex_indices.get(id).copied())
                .collect::<Option<Vec<_>>>()
            else {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "mesh face references a missing vertex",
                ));
            };
            faces.push(indices);
        }
        output.push(ExchangeMesh {
            id: node_id.to_string(),
            name: node.name.clone(),
            positions,
            faces,
        });
    }
    Ok(output)
}

pub(crate) fn render_visible_nodes(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
) -> Result<BTreeSet<Id>> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "evaluation scene is missing"))?;
    let excluded = snapshot
        .view_layer
        .as_ref()
        .and_then(|id| scene.view_layers.get(id))
        .map(|layer| {
            layer
                .excluded_collections
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut visible = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![scene.root_collection.clone()];
    while let Some(id) = pending.pop() {
        if excluded.contains(&id) || !visited.insert(id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "collection reference is missing")
        })?;
        visible.extend(collection.objects.iter().cloned());
        pending.extend(collection.children.iter().cloned());
    }
    Ok(visible)
}

pub(crate) fn graph_from_meshes(meshes: Vec<ExchangeMesh>, scene_id: String) -> Result<SceneDoc> {
    if meshes.is_empty() {
        return Err(PotError::new(
            ErrorCode::ImportFailed,
            "input contains no mesh geometry",
        ));
    }
    let mut doc = SceneDoc::new(scene_id);
    let mut imported_nodes = Vec::with_capacity(meshes.len());
    for (index, geometry) in meshes.into_iter().enumerate() {
        let stem = valid_stem(&geometry.id, index);
        let node_id = unique_graph_id(
            &doc,
            &stem,
            index,
            GraphIdRegistry::Nodes,
            "imported ID exceeds maximum length",
        )?;
        let data_id = unique_graph_id(
            &doc,
            &format!("{stem}_mesh"),
            index,
            GraphIdRegistry::DataBlocks,
            "imported data ID exceeds maximum length",
        )?;
        let positions = geometry
            .positions
            .iter()
            .map(|position| DVec3::from_array(*position))
            .collect();
        let mesh = Mesh::from_positions_and_faces(positions, geometry.faces).map_err(|error| {
            PotError::with_details(
                ErrorCode::ImportFailed,
                "imported geometry is invalid",
                serde_json::json!({ "reason": error.to_string(), "data_id": data_id }),
            )
        })?;
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: geometry.name,
                kind: "mesh".to_owned(),
                primitive: None,
                tags: Vec::new(),
                parent: None,
                parent_inverse: None,
                transform: Transform::default(),
                data: Some(data_id),
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                ..Node::default()
            },
        );
        imported_nodes.push(node_id);
    }
    let root = Id::new("collection_root")?;
    doc.collections
        .get_mut(&root)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "root collection is missing"))?
        .objects
        .extend(imported_nodes);
    doc.validate()?;
    Ok(doc)
}

pub(crate) fn import_graph_from_meshes(
    meshes: Vec<ExchangeMesh>,
    path: &std::path::Path,
    scene_id: String,
    format: &str,
) -> Result<ImportedGraph> {
    let source = serde_json::json!({ "format": format, "path": path.display().to_string() });
    Ok(ImportedGraph {
        doc: graph_from_meshes(meshes, scene_id)?,
        losses: Vec::new(),
        id_mappings: serde_json::json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source,
    })
}

fn valid_stem(value: &str, index: usize) -> String {
    let mut result = String::with_capacity(value.len() + 16);
    for byte in value.bytes() {
        let character = char::from(byte).to_ascii_lowercase();
        if character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || character == '_'
            || character == '-'
        {
            result.push(character);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() || !result.as_bytes()[0].is_ascii_lowercase() {
        result.insert_str(0, &format!("mesh{index}_"));
    }
    result.truncate(48);
    result
}

#[derive(Clone, Copy)]
enum GraphIdRegistry {
    Nodes,
    DataBlocks,
}

fn unique_graph_id(
    doc: &SceneDoc,
    seed: &str,
    index: usize,
    registry: GraphIdRegistry,
    too_long_message: &str,
) -> Result<Id> {
    let occupied = |id: &Id| match registry {
        GraphIdRegistry::Nodes => doc.nodes.contains_key(id),
        GraphIdRegistry::DataBlocks => doc.data_blocks.contains_key(id),
    };
    let mut candidate = seed.to_owned();
    let mut suffix = index;
    while occupied(&Id::new(candidate.clone())?) {
        suffix = suffix.saturating_add(1);
        candidate = format!("{seed}_{suffix}");
        if candidate.len() > 64 {
            return Err(PotError::new(ErrorCode::LimitExceeded, too_long_message));
        }
    }
    Id::new(candidate)
}

pub(super) fn claim_import_id(id: Id, used: &mut BTreeSet<Id>) -> Option<Id> {
    used.insert(id.clone()).then_some(id)
}

pub(super) fn import_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

#[derive(Clone, Copy)]
pub(super) struct LeReadErrors {
    pub offset_overflow: &'static str,
    pub out_of_bounds: &'static str,
    pub invalid_integer: &'static str,
}

fn read_le_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    errors: LeReadErrors,
) -> Result<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| import_error(errors.offset_overflow))?;
    let bytes = bytes
        .get(offset..end)
        .ok_or_else(|| import_error(errors.out_of_bounds))?;
    bytes
        .try_into()
        .map_err(|_| import_error(errors.invalid_integer))
}

pub(super) fn read_u32(bytes: &[u8], offset: usize, errors: LeReadErrors) -> Result<u32> {
    Ok(u32::from_le_bytes(read_le_array(bytes, offset, errors)?))
}

pub(super) fn read_u64(bytes: &[u8], offset: usize, errors: LeReadErrors) -> Result<u64> {
    Ok(u64::from_le_bytes(read_le_array(bytes, offset, errors)?))
}

pub(super) fn decode_le_array<const N: usize>(
    bytes: &[u8],
    invalid_message: &'static str,
) -> Result<[u8; N]> {
    bytes.try_into().map_err(|_| import_error(invalid_message))
}

pub(super) fn unique_data_id(
    first: Id,
    is_used: impl Fn(&Id) -> bool,
    mut with_suffix: impl FnMut(u32) -> Result<Id>,
    first_suffix: u32,
    exhausted_message: &'static str,
) -> Result<Id> {
    if !is_used(&first) {
        return Ok(first);
    }
    let mut suffix = first_suffix;
    loop {
        let candidate = with_suffix(suffix)?;
        if !is_used(&candidate) {
            return Ok(candidate);
        }
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, exhausted_message))?;
    }
}

pub mod alembic;
pub mod blend;
mod blend_dna;
pub mod bvh;
pub mod fbx;
pub mod fbx_binary;
pub mod gltf;
pub mod obj;
pub mod pdf;
pub mod ply;
pub mod stl;
pub mod svg;
pub mod usd;
pub mod usdc;
mod vector;
