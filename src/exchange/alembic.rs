use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use glam::{DMat4, DVec3};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{ImportedGraph, Loss, evaluated_meshes, import_error};
use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::{Action, DataBlock, FCurve, Id, Interpolation, Keyframe, Node, SceneDoc, Transform},
};

const MAX_FILE_BYTES: usize = 268_435_456;
const MAX_ARCHIVE_RECORDS: usize = 1_000_000;
const MAX_OBJECT_DEPTH: usize = 128;
const MAX_TIME_SAMPLES: usize = 10_000;
const OGAWA_DATA_FLAG: u64 = 1_u64 << 63;

const OGAWA_READ_ERRORS: super::LeReadErrors = super::LeReadErrors {
    offset_overflow: "Ogawa offset overflows",
    out_of_bounds: "Ogawa offset is out of bounds",
    invalid_integer: "Ogawa offset is out of bounds",
};
const OGAWA_OFFSET_MASK: u64 = !OGAWA_DATA_FLAG;
const ALEMBIC_ARCHIVE_VERSION: i32 = 10_505;
const XFORM_MATRIX_OP: u8 = 0x30;

/// Export evaluated scene geometry and transforms as an Ogawa Alembic archive.
pub(crate) fn export(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    allow_lossy: bool,
    _blender: Option<&Path>,
) -> Result<Vec<u8>> {
    if !allow_lossy
        && doc
            .scenes
            .get(&snapshot.scene_id)
            .is_some_and(|scene| scene.rigid_body_world.is_some())
    {
        return Err(unsupported(
            "physics.rigid_body_world",
            "Alembic export does not represent the scene rigid-body world".to_owned(),
        ));
    }
    let evaluated = evaluated_meshes(doc, snapshot)?;
    let evaluated_ids = evaluated
        .iter()
        .map(|mesh| mesh.id.as_str())
        .collect::<BTreeSet<_>>();
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "Alembic export scene is missing"))?;
    if !scene.fps_base.is_finite() || scene.fps_base <= 0.0 || scene.fps == 0 {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic export requires a finite positive frame rate",
        ));
    }
    for (id, node) in &doc.nodes {
        if node
            .parent
            .as_ref()
            .is_some_and(|parent| !doc.nodes.contains_key(parent))
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                format!("Alembic node `{id}` has a missing parent"),
            ));
        }
        if !allow_lossy && (node.rigid_body.is_some() || node.force_field.is_some()) {
            return Err(unsupported(
                "physics.node",
                format!("Alembic export does not represent physics on `{id}`"),
            ));
        }
        if !allow_lossy && !matches!(node.kind.as_str(), "mesh" | "empty") {
            return Err(unsupported(
                "node.kind",
                format!(
                    "Alembic export does not represent node kind `{}`",
                    node.kind
                ),
            ));
        }
        if !allow_lossy && (!node.visible || !node.render_visible || !node.selectable) {
            return Err(unsupported(
                "node.visibility",
                format!("Alembic export does not represent visibility or selectability on `{id}`"),
            ));
        }
        if !allow_lossy && !node.materials.is_empty() {
            return Err(unsupported(
                "material.binding",
                format!("Alembic export does not represent material bindings on `{id}`"),
            ));
        }
        if !allow_lossy && (!node.tags.is_empty() || !node.properties.is_empty()) {
            return Err(unsupported(
                "node.metadata",
                format!("Alembic export does not represent custom node metadata on `{id}`"),
            ));
        }
        if let Some(data_id) = &node.data {
            let data = doc.data_blocks.get(data_id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "Alembic node data block is missing",
                )
            })?;
            if !allow_lossy && (data.camera.is_some() || data.light.is_some()) {
                return Err(unsupported(
                    "node.data",
                    format!("Alembic export does not represent camera or light data on `{id}`"),
                ));
            }
            if node.kind == "mesh" && (data.data_type != "mesh" || data.mesh.is_none()) {
                return Err(unsupported(
                    "node.data",
                    format!("Alembic mesh node `{id}` has unsupported data"),
                ));
            }
            if !allow_lossy && node.kind == "empty" && data.mesh.is_some() {
                return Err(unsupported(
                    "node.data",
                    format!("Alembic empty node `{id}` contains mesh data"),
                ));
            }
        }
        if node.action.is_some()
            && !doc
                .actions
                .contains_key(node.action.as_ref().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "Alembic action ID disappeared")
                })?)
        {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Alembic node references a missing action",
            ));
        }
    }

    let frames = sample_frames(doc, snapshot)?;
    let rate = f64::from(scene.fps) / scene.fps_base;
    let mut sampled = Vec::with_capacity(frames.len());
    for frame in &frames {
        let owned;
        let value = if frame.to_bits() == snapshot.frame.to_bits() {
            snapshot
        } else {
            owned = Snapshot::evaluate(
                doc,
                &EvaluationContext {
                    scene_id: Some(snapshot.scene_id.clone()),
                    view_layer: snapshot.view_layer.clone(),
                    frame: Some(*frame),
                },
            )?;
            &owned
        };
        let mut matrices = BTreeMap::new();
        for id in doc.nodes.keys() {
            let node_state = value.nodes.get(id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic node evaluation is missing",
                )
            })?;
            let world = DMat4::from_cols_array(&node_state.world_matrix);
            let local =
                if let Some(parent_id) = doc.nodes.get(id).and_then(|node| node.parent.as_ref()) {
                    let parent_state = value.nodes.get(parent_id).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "Alembic parent evaluation is missing",
                        )
                    })?;
                    let parent_world = DMat4::from_cols_array(&parent_state.world_matrix);
                    let determinant = parent_world.determinant();
                    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
                        return Err(PotError::new(
                            ErrorCode::EvaluationFailed,
                            "Alembic parent transform is singular",
                        ));
                    }
                    parent_world.inverse() * world
                } else {
                    world
                };
            if !local.is_finite() {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic node transform is not finite",
                ));
            }
            matrices.insert(id.clone(), local.to_cols_array());
        }
        let mut meshes = BTreeMap::new();
        for id in doc
            .nodes
            .keys()
            .filter(|id| evaluated_ids.contains(id.as_str()))
        {
            let source = value.meshes.get(id).ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic evaluated mesh is missing",
                )
            })?;
            meshes.insert(id.clone(), mesh_sample(source, allow_lossy)?);
        }
        sampled.push(FrameSample {
            seconds: *frame / rate,
            matrices,
            meshes,
        });
    }
    let time_samples = if sampled.len() > 1 {
        sampled
            .iter()
            .map(|sample| sample.seconds)
            .collect::<Vec<_>>()
    } else {
        vec![0.0]
    };
    let mut children = BTreeMap::<Option<Id>, Vec<Id>>::new();
    for (id, node) in &doc.nodes {
        children
            .entry(node.parent.clone())
            .or_default()
            .push(id.clone());
    }
    let mut objects = Vec::new();
    let mut active = BTreeSet::new();
    if let Some(roots) = children.get(&None) {
        for id in roots {
            objects.push(write_node(id, doc, &children, &sampled, &mut active, 0)?);
        }
    }
    if objects.is_empty() && !doc.nodes.is_empty() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic scene node hierarchy has no root",
        ));
    }
    let root_metadata = metadata_bytes(&BTreeMap::from([
        ("abc_version".to_owned(), "1.5.5".to_owned()),
        ("application".to_owned(), "Potter".to_owned()),
        ("potter_fps".to_owned(), scene.fps.to_string()),
        ("potter_fps_base".to_owned(), scene.fps_base.to_string()),
        ("potter_frame".to_owned(), snapshot.frame.to_string()),
    ]))?;
    let time_table = encode_time_sampling_table(&time_samples, sampled.len())?;
    let root = Chunk::Group(vec![
        Chunk::Data(0_i32.to_le_bytes().to_vec()),
        Chunk::Data(ALEMBIC_ARCHIVE_VERSION.to_le_bytes().to_vec()),
        object_data_root(objects)?,
        Chunk::Data(root_metadata),
        Chunk::Data(time_table),
        Chunk::Data(Vec::new()),
    ]);
    encode_archive(root)
}

/// Read a native Ogawa Alembic archive containing Xform and `PolyMesh` objects.
pub(crate) fn import(
    file: &Path,
    scene_id: String,
    _blender: Option<&Path>,
) -> Result<ImportedGraph> {
    if !file.is_file() {
        return Err(PotError::new(
            ErrorCode::FileNotFound,
            "Alembic file does not exist",
        ));
    }
    let file_size = file.metadata().map_err(|error| PotError::io(&error))?.len();
    if file_size
        > u64::try_from(MAX_FILE_BYTES).map_err(|_| {
            PotError::new(
                ErrorCode::InternalError,
                "Alembic size limit conversion failed",
            )
        })?
    {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic archive exceeds the size limit",
        ));
    }
    let bytes = fs::read(file).map_err(|error| PotError::io(&error))?;
    let archive = parse_archive(&bytes)?;
    let fps = metadata_f64(&archive.metadata, "potter_fps").unwrap_or(24.0);
    let fps_base = metadata_f64(&archive.metadata, "potter_fps_base").unwrap_or(1.0);
    if !fps.is_finite() || fps <= 0.0 || !fps_base.is_finite() || fps_base <= 0.0 {
        return Err(import_error("Alembic frame rate metadata is invalid"));
    }
    let mut doc = SceneDoc::new(scene_id);
    if let Some(scene) = doc.scenes.get_mut(&doc.active_scene) {
        scene.fps = float_to_u32(fps, "Alembic fps")?;
        scene.fps_base = fps_base;
    }
    let mut used_ids = BTreeSet::new();
    let mut used_data_ids = BTreeSet::new();
    let mut used_action_ids = BTreeSet::new();
    let mut losses = Vec::new();
    let mut roots = Vec::new();
    let first_frame = metadata_f64(&archive.metadata, "potter_frame").unwrap_or_else(|| {
        archive
            .time_samplings
            .get(1)
            .or_else(|| archive.time_samplings.first())
            .and_then(|sampling| sampling.times.first())
            .map_or(1.0, |time| time * fps / fps_base)
    });
    if !first_frame.is_finite() {
        return Err(import_error("Alembic scene frame metadata is not finite"));
    }
    if let Some(scene) = doc.scenes.get_mut(&doc.active_scene) {
        scene.frame_current = first_frame;
    }
    for object in &archive.objects {
        let id = import_object(
            object,
            archive.bytes,
            None,
            &archive.time_samplings,
            fps,
            fps_base,
            &mut doc,
            &mut used_ids,
            &mut used_data_ids,
            &mut used_action_ids,
            &mut losses,
            0,
        )?;
        if let Some(id) = id {
            roots.push(id);
        }
    }
    doc.collections
        .get_mut(&Id::from_static("collection_root"))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "Alembic root collection is missing",
            )
        })?
        .objects = roots;
    doc.validate()?;
    Ok(ImportedGraph {
        doc,
        losses,
        id_mappings: json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source: json!({"format":"alembic","path":file.display().to_string()}),
    })
}

#[derive(Clone)]
struct FrameSample {
    seconds: f64,
    matrices: BTreeMap<Id, [f64; 16]>,
    meshes: BTreeMap<Id, MeshSample>,
}

#[derive(Clone)]
struct MeshSample {
    positions: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    uvs: Option<Vec<Vec<[f64; 2]>>>,
    normals: Option<Vec<[f64; 3]>>,
}

fn sample_frames(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<f64>> {
    let mut frames = vec![snapshot.frame];
    for node in doc.nodes.values() {
        let Some(action_id) = &node.action else {
            continue;
        };
        let action = doc.actions.get(action_id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "Alembic node action is missing")
        })?;
        for curve in &action.fcurves {
            for key in &curve.keyframes {
                if !key.frame.is_finite() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic animation sample time is not finite",
                    ));
                }
                frames.push(key.frame);
            }
        }
    }
    frames.sort_by(f64::total_cmp);
    frames.dedup_by(|first, second| first.to_bits() == second.to_bits());
    if frames.len() > MAX_TIME_SAMPLES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic export exceeds the time-sample limit",
        ));
    }
    Ok(frames)
}

fn mesh_sample(mesh: &Mesh, allow_lossy: bool) -> Result<MeshSample> {
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect::<Vec<_>>();
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<BTreeMap<_, _>>();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let indices = face
            .vertices
            .iter()
            .map(|id| vertex_indices.get(id).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "Alembic mesh face references a missing vertex",
                )
            })?;
        faces.push(indices);
    }
    for attribute in mesh.attributes.keys() {
        let supported_attribute = matches!(attribute.as_str(), "uv_map" | "alembic_normals")
            || (allow_lossy && attribute == "vertex_groups");
        if !supported_attribute {
            return Err(unsupported(
                "mesh.attribute",
                format!("Alembic export does not represent mesh attribute `{attribute}`"),
            ));
        }
    }
    let uvs = mesh_uvs(mesh)?;
    let normals = mesh_normals(mesh, &faces)?;
    Ok(MeshSample {
        positions,
        faces,
        uvs,
        normals,
    })
}

fn mesh_normals(mesh: &Mesh, faces: &[Vec<usize>]) -> Result<Option<Vec<[f64; 3]>>> {
    let Some(attribute) = mesh.attributes.get("alembic_normals") else {
        return Ok(None);
    };
    let scope = attribute
        .get("scope")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "Alembic normals have no geometry scope",
            )
        })?;
    let source = attribute
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "Alembic normals have no values"))?;
    let values = source
        .iter()
        .map(|value| {
            let vector = value
                .as_array()
                .filter(|components| components.len() == 3)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic normal must have three values",
                    )
                })?;
            let result = [
                vector[0].as_f64().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "Alembic normal is not numeric")
                })?,
                vector[1].as_f64().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "Alembic normal is not numeric")
                })?,
                vector[2].as_f64().ok_or_else(|| {
                    PotError::new(ErrorCode::SceneInvalid, "Alembic normal is not numeric")
                })?,
            ];
            if result.iter().any(|component| !component.is_finite()) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "Alembic normal is not finite",
                ));
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output = Vec::new();
    let mut corner = 0;
    for (face_index, face) in faces.iter().enumerate() {
        for vertex in face {
            let value_index = match scope {
                "vertex" | "varying" => *vertex,
                "facevarying" => corner,
                "uniform" => face_index,
                "constant" => 0,
                _ => {
                    return Err(unsupported(
                        "alembic.geom_scope",
                        format!("Alembic normal scope `{scope}` is unsupported"),
                    ));
                }
            };
            output.push(*values.get(value_index).ok_or_else(|| {
                PotError::new(
                    ErrorCode::SceneInvalid,
                    "Alembic normal value count does not match its scope",
                )
            })?);
            corner += 1;
        }
    }
    let expected = match scope {
        "vertex" | "varying" => mesh.vertices.len(),
        "facevarying" => corner,
        "uniform" => faces.len(),
        "constant" => 1,
        _ => 0,
    };
    if values.len() != expected {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic normal value count does not match its scope",
        ));
    }
    Ok(Some(output))
}

fn mesh_uvs(mesh: &Mesh) -> Result<Option<Vec<Vec<[f64; 2]>>>> {
    let Some(value) = mesh.attributes.get("uv_map") else {
        return Ok(None);
    };
    let entries = value.as_array().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic UV map attribute is not an array",
        )
    })?;
    let mut values = BTreeMap::<u32, Vec<[f64; 2]>>::new();
    for entry in entries {
        let face_id = entry
            .get("face_id")
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                PotError::new(ErrorCode::SceneInvalid, "Alembic UV map face ID is invalid")
            })?;
        let coordinates = entry.get("uv").and_then(Value::as_array).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "Alembic UV map has no corner coordinates",
            )
        })?;
        let coordinates = coordinates
            .iter()
            .map(|coordinate| {
                let pair = coordinate
                    .as_array()
                    .filter(|values| values.len() == 2)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "Alembic UV coordinate must have two values",
                        )
                    })?;
                let u = pair[0].as_f64().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic U coordinate is not numeric",
                    )
                })?;
                let v = pair[1].as_f64().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic V coordinate is not numeric",
                    )
                })?;
                if !u.is_finite() || !v.is_finite() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic UV coordinate is not finite",
                    ));
                }
                Ok([u, v])
            })
            .collect::<Result<Vec<_>>>()?;
        if values.insert(face_id, coordinates).is_some() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Alembic UV map has duplicate face IDs",
            ));
        }
    }
    let mut output = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let coordinates = values.remove(&face.id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "Alembic UV map is missing a face")
        })?;
        if coordinates.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "Alembic UV corner count does not match topology",
            ));
        }
        output.push(coordinates);
    }
    if !values.is_empty() {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic UV map contains an unknown face",
        ));
    }
    Ok(Some(output))
}

fn write_node(
    id: &Id,
    doc: &SceneDoc,
    children: &BTreeMap<Option<Id>, Vec<Id>>,
    samples: &[FrameSample],
    active: &mut BTreeSet<Id>,
    depth: usize,
) -> Result<AbcObject> {
    if depth > MAX_OBJECT_DEPTH || !active.insert(id.clone()) {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "Alembic node hierarchy is cyclic or too deep",
        ));
    }
    let node = doc.nodes.get(id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "Alembic node disappeared during export",
        )
    })?;
    let matrix_samples = samples
        .iter()
        .map(|sample| {
            sample.matrices.get(id).copied().ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Alembic transform sample is missing",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let properties = vec![write_xform_properties(&matrix_samples, samples.len())?];
    let mut child_objects = Vec::new();
    if node.kind == "mesh" {
        let mesh_samples = samples
            .iter()
            .map(|sample| sample.meshes.get(id))
            .collect::<Option<Vec<_>>>();
        if let Some(mesh_samples) = mesh_samples {
            child_objects.push(write_polymesh(&mesh_samples, samples.len())?);
        }
    }
    if let Some(nested) = children.get(&Some(id.clone())) {
        for child in nested {
            child_objects.push(write_node(
                child,
                doc,
                children,
                samples,
                active,
                depth + 1,
            )?);
        }
    }
    active.remove(id);
    Ok(AbcObject {
        name: id.as_str().to_owned(),
        metadata: BTreeMap::from([
            ("potter_id".to_owned(), id.as_str().to_owned()),
            (
                "potter_name_hex".to_owned(),
                hex::encode(node.name.as_bytes()),
            ),
            ("schema".to_owned(), "AbcGeom_Xform_v3".to_owned()),
            (
                "schemaObjTitle".to_owned(),
                "AbcGeom_Xform_v3:.xform".to_owned(),
            ),
        ]),
        properties,
        children: child_objects,
    })
}

fn write_xform_properties(matrices: &[[f64; 16]], sample_count: usize) -> Result<AbcProperty> {
    let values = matrices
        .iter()
        .map(|matrix| encode_f64s(&matrix_channels(*matrix)))
        .collect::<Result<Vec<_>>>()?;
    compound_property(
        ".xform",
        metadata_map(&[
            ("schema", "AbcGeom_Xform_v3"),
            ("schemaObjTitle", "AbcGeom_Xform_v3:.xform"),
        ]),
        vec![
            scalar_property(".inherits", 0, 1, BTreeMap::new(), 0, vec![vec![1]])?,
            scalar_property(
                ".ops",
                1,
                1,
                BTreeMap::new(),
                0,
                vec![vec![XFORM_MATRIX_OP]],
            )?,
            scalar_property(
                ".vals",
                11,
                16,
                BTreeMap::new(),
                u32::from(sample_count > 1),
                values,
            )?,
        ],
    )
}

fn write_polymesh(samples: &[&MeshSample], sample_count: usize) -> Result<AbcObject> {
    let sampling_index = u32::from(sample_count > 1);
    let mut positions = Vec::with_capacity(samples.len());
    let mut counts = Vec::with_capacity(samples.len());
    let mut indices = Vec::with_capacity(samples.len());
    let mut bounds = Vec::with_capacity(samples.len());
    let mut normals = Vec::with_capacity(samples.len());
    let mut uvs = Vec::with_capacity(samples.len());
    let mut uv_indices = Vec::with_capacity(samples.len());
    for sample in samples {
        positions.push(encode_vec3_f32(&sample.positions)?);
        let mut face_counts = Vec::with_capacity(sample.faces.len());
        let mut face_indices = Vec::new();
        for face in &sample.faces {
            face_counts.push(i32::try_from(face.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic face vertex count exceeds int32",
                )
            })?);
            for index in face {
                face_indices.push(i32::try_from(*index).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "Alembic vertex index exceeds int32",
                    )
                })?);
            }
        }
        counts.push(encode_i32s(&face_counts));
        indices.push(encode_i32s(&face_indices));
        bounds.push(encode_bounds(&sample.positions)?);
        let generated_normals;
        let normal_values = if let Some(normals) = &sample.normals {
            normals.as_slice()
        } else {
            generated_normals = corner_normals(sample)?;
            &generated_normals
        };
        normals.push(encode_vec3_f32(normal_values)?);
        let face_uvs = if let Some(uvs) = &sample.uvs {
            if uvs.len() != sample.faces.len() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "Alembic UV face count does not match topology",
                ));
            }
            for (coordinates, face) in uvs.iter().zip(&sample.faces) {
                if coordinates.len() != face.len() {
                    return Err(PotError::new(
                        ErrorCode::SceneInvalid,
                        "Alembic UV corner count does not match topology",
                    ));
                }
            }
            uvs.iter().flatten().copied().collect()
        } else {
            let vertex_uv = generated_uvs(&sample.positions);
            sample
                .faces
                .iter()
                .flatten()
                .map(|index| {
                    vertex_uv.get(*index).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "Alembic UV references a missing vertex",
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        uvs.push(encode_vec2_f32(&face_uvs)?);
        let indices = (0..face_uvs.len())
            .map(|index| {
                u32::try_from(index).map_err(|_| {
                    PotError::new(ErrorCode::LimitExceeded, "Alembic UV index exceeds uint32")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        uv_indices.push(encode_u32s(&indices));
    }
    let p_meta = metadata_map(&[("geoScope", "vtx"), ("interpretation", "point")]);
    let n_meta = geomparam_metadata("facevarying", "normal", 3);
    let uv_meta = geomparam_metadata("facevarying", "vector", 2);
    let geometry = compound_property(
        ".geom",
        metadata_map(&[
            ("schema", "AbcGeom_PolyMesh_v1"),
            ("schemaBaseType", "AbcGeom_GeomBase_v1"),
            ("schemaObjTitle", "AbcGeom_PolyMesh_v1:.geom"),
        ]),
        vec![
            array_property("P", 10, 3, p_meta, sampling_index, positions)?,
            array_property(
                ".faceIndices",
                6,
                1,
                BTreeMap::new(),
                sampling_index,
                indices,
            )?,
            array_property(".faceCounts", 6, 1, BTreeMap::new(), sampling_index, counts)?,
            scalar_property(
                ".selfBnds",
                11,
                6,
                metadata_map(&[("interpretation", "box")]),
                sampling_index,
                bounds,
            )?,
            array_property("N", 10, 3, n_meta, sampling_index, normals)?,
            geomparam_property("uv", uv_meta, 10, 2, sampling_index, uvs, uv_indices)?,
        ],
    )?;
    Ok(AbcObject {
        name: "geometry".to_owned(),
        metadata: BTreeMap::from([
            ("schema".to_owned(), "AbcGeom_PolyMesh_v1".to_owned()),
            (
                "schemaBaseType".to_owned(),
                "AbcGeom_GeomBase_v1".to_owned(),
            ),
            (
                "schemaObjTitle".to_owned(),
                "AbcGeom_PolyMesh_v1:.geom".to_owned(),
            ),
        ]),
        properties: vec![geometry],
        children: Vec::new(),
    })
}

fn corner_normals(sample: &MeshSample) -> Result<Vec<[f64; 3]>> {
    let mut normals = Vec::new();
    for face in &sample.faces {
        let first = *face
            .first()
            .ok_or_else(|| import_error("Alembic mesh contains an empty face"))?;
        let second = *face
            .get(1)
            .ok_or_else(|| import_error("Alembic mesh face has fewer than three vertices"))?;
        let third = *face
            .get(2)
            .ok_or_else(|| import_error("Alembic mesh face has fewer than three vertices"))?;
        let a = DVec3::from_array(
            *sample
                .positions
                .get(first)
                .ok_or_else(|| import_error("Alembic mesh index is out of range"))?,
        );
        let b = DVec3::from_array(
            *sample
                .positions
                .get(second)
                .ok_or_else(|| import_error("Alembic mesh index is out of range"))?,
        );
        let c = DVec3::from_array(
            *sample
                .positions
                .get(third)
                .ok_or_else(|| import_error("Alembic mesh index is out of range"))?,
        );
        let normal = (b - a).cross(c - a).normalize_or_zero().to_array();
        normals.extend(std::iter::repeat_n(normal, face.len()));
    }
    Ok(normals)
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

fn encode_vec3_f32(values: &[[f64; 3]]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(values.len().saturating_mul(12));
    for value in values {
        for component in value {
            output.extend_from_slice(&float_to_f32(*component)?.to_le_bytes());
        }
    }
    Ok(output)
}

fn encode_vec2_f32(values: &[[f64; 2]]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        for component in value {
            output.extend_from_slice(&float_to_f32(*component)?.to_le_bytes());
        }
    }
    Ok(output)
}

fn encode_u32s(values: &[u32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn encode_i32s(values: &[i32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output
}

fn encode_f64s(values: &[f64]) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        if !value.is_finite() {
            return Err(PotError::new(
                ErrorCode::ExportFailed,
                "Alembic sample contains a non-finite value",
            ));
        }
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output)
}

fn encode_bounds(positions: &[[f64; 3]]) -> Result<Vec<u8>> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for position in positions {
        for axis in 0..3 {
            if !position[axis].is_finite() {
                return Err(PotError::new(
                    ErrorCode::ExportFailed,
                    "Alembic mesh position is not finite",
                ));
            }
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    if positions.is_empty() {
        min = [0.0; 3];
        max = [0.0; 3];
    }
    encode_f64s(&[min[0], min[1], min[2], max[0], max[1], max[2]])
}

fn float_to_f32(value: f64) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "Alembic float32 sample is out of range",
        ));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Alembic P3f and geom-param samples use float32"
    )]
    let converted = value as f32;
    Ok(converted)
}

fn matrix_channels(matrix: [f64; 16]) -> [f64; 16] {
    // Alembic stores Imath row-major matrices; this byte order is the transpose of glam's
    // column-major representation and therefore preserves the same column-vector transform.
    matrix
}

fn geomparam_metadata(scope: &str, interpretation: &str, extent: u8) -> BTreeMap<String, String> {
    let scope = match scope {
        "vertex" => "vtx",
        "varying" => "vry",
        "uniform" => "uni",
        "facevarying" => "fvr",
        "constant" => "con",
        scope => scope,
    };
    metadata_map(&[
        ("arrayExtent", "1"),
        ("geoScope", scope),
        ("interpretation", interpretation),
        ("isGeomParam", "true"),
        ("podExtent", if extent == 2 { "2" } else { "3" }),
        ("podName", "float32_t"),
    ])
}

fn metadata_map(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

#[derive(Clone)]
struct AbcObject {
    name: String,
    metadata: BTreeMap<String, String>,
    properties: Vec<AbcProperty>,
    children: Vec<AbcObject>,
}

#[derive(Clone)]
struct AbcProperty {
    descriptor: PropertyDescriptor,
    body: PropertyBody,
}

#[derive(Clone)]
enum PropertyBody {
    Compound(Vec<AbcProperty>),
    Scalar(Vec<Vec<u8>>),
    Array(Vec<Vec<u8>>),
}

#[derive(Clone)]
enum Chunk {
    Data(Vec<u8>),
    Group(Vec<Chunk>),
}

#[derive(Clone)]
struct PropertyDescriptor {
    name: String,
    property_type: u32,
    pod: u8,
    extent: u8,
    sample_count: usize,
    time_sampling_index: u32,
    metadata: BTreeMap<String, String>,
}

fn object_data_root(objects: Vec<AbcObject>) -> Result<Chunk> {
    let headers = objects
        .iter()
        .map(|object| encode_object_header(&object.name, &object.metadata))
        .collect::<Result<Vec<_>>>()?;
    let mut children = vec![compound_tree(&[])?];
    for object in objects {
        children.push(object_chunk(object)?);
    }
    children.push(Chunk::Data(encode_object_headers(headers)));
    Ok(Chunk::Group(children))
}

fn object_chunk(object: AbcObject) -> Result<Chunk> {
    if object.name.is_empty() {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "Alembic object name is empty",
        ));
    }
    let headers = object
        .children
        .iter()
        .map(|child| encode_object_header(&child.name, &child.metadata))
        .collect::<Result<Vec<_>>>()?;
    let mut children = vec![compound_tree(&object.properties)?];
    for child in object.children {
        children.push(object_chunk(child)?);
    }
    children.push(Chunk::Data(encode_object_headers(headers)));
    Ok(Chunk::Group(children))
}
fn encode_object_headers(headers: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = concat_bytes(headers);
    bytes.extend_from_slice(&[0; 32]);
    bytes
}

fn compound_property(
    name: &str,
    metadata: BTreeMap<String, String>,
    properties: Vec<AbcProperty>,
) -> Result<AbcProperty> {
    let descriptor = PropertyDescriptor {
        name: name.to_owned(),
        property_type: 0,
        pod: 0,
        extent: 0,
        sample_count: 0,
        time_sampling_index: 0,
        metadata,
    };
    encode_property_header(&descriptor)?;
    Ok(AbcProperty {
        descriptor,
        body: PropertyBody::Compound(properties),
    })
}

fn scalar_property(
    name: &str,
    pod: u8,
    extent: u8,
    metadata: BTreeMap<String, String>,
    time_sampling_index: u32,
    samples: Vec<Vec<u8>>,
) -> Result<AbcProperty> {
    property_with_samples(
        name,
        1,
        pod,
        extent,
        metadata,
        time_sampling_index,
        samples,
        false,
    )
}

fn array_property(
    name: &str,
    pod: u8,
    extent: u8,
    metadata: BTreeMap<String, String>,
    time_sampling_index: u32,
    samples: Vec<Vec<u8>>,
) -> Result<AbcProperty> {
    property_with_samples(
        name,
        2,
        pod,
        extent,
        metadata,
        time_sampling_index,
        samples,
        true,
    )
}

fn property_with_samples(
    name: &str,
    property_type: u32,
    pod: u8,
    extent: u8,
    metadata: BTreeMap<String, String>,
    time_sampling_index: u32,
    samples: Vec<Vec<u8>>,
    is_array: bool,
) -> Result<AbcProperty> {
    if samples.is_empty() || extent == 0 {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "Alembic property has no samples or extent",
        ));
    }
    let descriptor = PropertyDescriptor {
        name: name.to_owned(),
        property_type,
        pod,
        extent,
        sample_count: samples.len(),
        time_sampling_index,
        metadata,
    };
    encode_property_header(&descriptor)?;
    let body = if is_array {
        PropertyBody::Array(samples)
    } else {
        PropertyBody::Scalar(samples)
    };
    Ok(AbcProperty { descriptor, body })
}

fn geomparam_property(
    name: &str,
    metadata: BTreeMap<String, String>,
    pod: u8,
    extent: u8,
    time_sampling_index: u32,
    samples: Vec<Vec<u8>>,
    index_samples: Vec<Vec<u8>>,
) -> Result<AbcProperty> {
    let values = array_property(
        ".vals",
        pod,
        extent,
        metadata.clone(),
        time_sampling_index,
        samples,
    )?;
    let indices = array_property(
        ".indices",
        5,
        1,
        BTreeMap::new(),
        time_sampling_index,
        index_samples,
    )?;
    compound_property(name, metadata, vec![values, indices])
}

fn compound_tree(properties: &[AbcProperty]) -> Result<Chunk> {
    let mut children = properties
        .iter()
        .map(property_chunk)
        .collect::<Result<Vec<_>>>()?;
    let headers = properties
        .iter()
        .map(|property| encode_property_header(&property.descriptor))
        .collect::<Result<Vec<_>>>()?;
    children.push(Chunk::Data(concat_bytes(headers)));
    Ok(Chunk::Group(children))
}

fn property_chunk(property: &AbcProperty) -> Result<Chunk> {
    let body = match &property.body {
        PropertyBody::Compound(properties) => compound_tree(properties)?,
        PropertyBody::Scalar(samples) => Chunk::Group(
            samples
                .iter()
                .map(|sample| Chunk::Data(sample_data(sample)))
                .collect(),
        ),
        PropertyBody::Array(samples) => Chunk::Group(
            samples
                .iter()
                .flat_map(|sample| [Chunk::Data(sample_data(sample)), Chunk::Data(Vec::new())])
                .collect(),
        ),
    };
    Ok(body)
}

fn encode_property_header(descriptor: &PropertyDescriptor) -> Result<Vec<u8>> {
    let metadata = metadata_bytes(&descriptor.metadata)?;
    let name_len = u32::try_from(descriptor.name.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic property name is too long",
        )
    })?;
    let metadata_len = u32::try_from(metadata.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic property metadata is too long",
        )
    })?;
    let sample_count = u32::try_from(descriptor.sample_count).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic property sample count exceeds uint32",
        )
    })?;
    let size_hint = size_hint([
        name_len,
        metadata_len,
        sample_count,
        descriptor.time_sampling_index,
    ]);
    let mut info = (size_hint << 2) | (255_u32 << 20);
    if descriptor.property_type != 0 {
        info |= descriptor.property_type;
        info |= u32::from(descriptor.pod) << 4;
        info |= u32::from(descriptor.extent) << 12;
        if descriptor.time_sampling_index != 0 {
            info |= 0x100;
        }
        if descriptor.sample_count == 1 {
            info |= 0x800;
        }
    }
    let mut out = info.to_le_bytes().to_vec();
    if descriptor.property_type != 0 {
        put_hint(&mut out, sample_count, size_hint);
        if descriptor.time_sampling_index != 0 {
            put_hint(&mut out, descriptor.time_sampling_index, size_hint);
        }
    }
    put_hint(&mut out, name_len, size_hint);
    out.extend_from_slice(descriptor.name.as_bytes());
    put_hint(&mut out, metadata_len, size_hint);
    out.extend_from_slice(&metadata);
    Ok(out)
}

fn encode_object_header(name: &str, metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let name_len = u32::try_from(name.len())
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "Alembic object name is too long"))?;
    let metadata = metadata_bytes(metadata)?;
    let metadata_len = u32::try_from(metadata.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic object metadata is too long",
        )
    })?;
    let mut out = name_len.to_le_bytes().to_vec();
    out.extend_from_slice(name.as_bytes());
    out.push(255);
    out.extend_from_slice(&metadata_len.to_le_bytes());
    out.extend_from_slice(&metadata);
    Ok(out)
}
fn metadata_bytes(metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let mut out = String::new();
    for (key, value) in metadata {
        if key.contains([';', '=']) || value.contains([';', '=']) {
            return Err(PotError::new(
                ErrorCode::ExportFailed,
                "Alembic metadata contains an unescaped delimiter",
            ));
        }
        out.push_str(key);
        out.push('=');
        out.push_str(value);
        out.push(';');
    }
    Ok(out.into_bytes())
}

fn concat_bytes(chunks: Vec<Vec<u8>>) -> Vec<u8> {
    let capacity = chunks.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(capacity);
    for chunk in chunks {
        out.extend_from_slice(&chunk);
    }
    out
}

fn sample_data(raw: &[u8]) -> Vec<u8> {
    let digest = Sha256::digest(raw);
    let mut out = Vec::with_capacity(16_usize.saturating_add(raw.len()));
    out.extend_from_slice(&digest[..16]);
    out.extend_from_slice(raw);
    out
}

fn encode_time_sampling_table(times: &[f64], sample_count: usize) -> Result<Vec<u8>> {
    let mut all = vec![(vec![0.0], 1.0, 1_usize)];
    if sample_count > 1 {
        let mut cycle = times.to_vec();
        for time in &cycle {
            if !time.is_finite() {
                return Err(PotError::new(
                    ErrorCode::ExportFailed,
                    "Alembic sample time is not finite",
                ));
            }
        }
        let last = *cycle.last().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "Alembic time sample list is empty",
            )
        })?;
        let previous = cycle
            .get(cycle.len().saturating_sub(2))
            .copied()
            .unwrap_or(0.0);
        let delta = (last - previous).abs().max(1.0e-9);
        let first = cycle.first().copied().unwrap_or(0.0);
        let time_per_cycle = (last - first + delta).max(delta);
        all.push((std::mem::take(&mut cycle), time_per_cycle, sample_count));
    }
    let mut out = Vec::new();
    for (sample_times, time_per_cycle, maximum_samples) in all {
        put_hint(
            &mut out,
            u32::try_from(maximum_samples).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic max sample count exceeds uint32",
                )
            })?,
            2,
        );
        out.extend_from_slice(&time_per_cycle.to_le_bytes());
        put_hint(
            &mut out,
            u32::try_from(sample_times.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic time sampling exceeds uint32",
                )
            })?,
            2,
        );
        for time in sample_times {
            out.extend_from_slice(&time.to_le_bytes());
        }
    }
    Ok(out)
}

fn size_hint(values: [u32; 4]) -> u32 {
    let max = values.into_iter().max().unwrap_or(0);
    if max > u32::from(u16::MAX) {
        2
    } else {
        u32::from(max > u32::from(u8::MAX))
    }
}

fn put_hint(out: &mut Vec<u8>, value: u32, hint: u32) {
    let bytes = value.to_le_bytes();
    let size = match hint {
        0 => 1,
        1 => 2,
        _ => 4,
    };
    out.extend_from_slice(&bytes[..size]);
}

fn encode_archive(root: Chunk) -> Result<Vec<u8>> {
    let mut bytes = vec![0_u8; 16];
    bytes[..5].copy_from_slice(b"Ogawa");
    bytes[5] = 0xff;
    bytes[6] = 0;
    bytes[7] = 1;
    let root_offset = emit_chunk(root, &mut bytes)?;
    bytes[8..16].copy_from_slice(&root_offset.to_le_bytes());
    if bytes.len() > MAX_FILE_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic archive exceeds the size limit",
        ));
    }
    Ok(bytes)
}

fn emit_chunk(chunk: Chunk, out: &mut Vec<u8>) -> Result<u64> {
    match chunk {
        Chunk::Data(data) => {
            if data.is_empty() {
                return Ok(OGAWA_DATA_FLAG);
            }
            let offset = u64::try_from(out.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic file offset exceeds uint64",
                )
            })?;
            if offset & OGAWA_DATA_FLAG != 0 {
                return Err(PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic file offset exceeds Ogawa range",
                ));
            }
            let length = u64::try_from(data.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic data block exceeds uint64",
                )
            })?;
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&data);
            Ok(OGAWA_DATA_FLAG | offset)
        }
        Chunk::Group(children) => {
            if children.is_empty() {
                return Ok(0);
            }
            if children.len() > MAX_ARCHIVE_RECORDS {
                return Err(PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic group exceeds the child limit",
                ));
            }
            let refs = children
                .into_iter()
                .map(|child| emit_chunk(child, out))
                .collect::<Result<Vec<_>>>()?;
            let offset = u64::try_from(out.len()).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic group offset exceeds uint64",
                )
            })?;
            if offset & OGAWA_DATA_FLAG != 0 {
                return Err(PotError::new(
                    ErrorCode::LimitExceeded,
                    "Alembic group offset exceeds Ogawa range",
                ));
            }
            out.extend_from_slice(
                &u64::try_from(refs.len())
                    .map_err(|_| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "Alembic group count exceeds uint64",
                        )
                    })?
                    .to_le_bytes(),
            );
            for reference in refs {
                out.extend_from_slice(&reference.to_le_bytes());
            }
            Ok(offset)
        }
    }
}

#[derive(Clone, Copy)]
struct OgawaRef {
    offset: u64,
    is_data: bool,
}

impl OgawaRef {
    fn from_word(word: u64) -> Self {
        Self {
            offset: word & OGAWA_OFFSET_MASK,
            is_data: word & OGAWA_DATA_FLAG != 0,
        }
    }
}

struct OgawaReader<'a> {
    bytes: &'a [u8],
    records: usize,
}

impl<'a> OgawaReader<'a> {
    fn group(&mut self, reference: OgawaRef) -> Result<Vec<OgawaRef>> {
        if reference.is_data {
            return Err(import_error("Ogawa group reference points to data"));
        }
        if reference.offset == 0 {
            return Ok(Vec::new());
        }
        let offset = usize_from_u64(reference.offset, "Ogawa group offset")?;
        let count = usize_from_u64(
            super::read_u64(self.bytes, offset, OGAWA_READ_ERRORS)?,
            "Ogawa group child count",
        )?;
        if count > MAX_ARCHIVE_RECORDS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "Ogawa group exceeds the child limit",
            ));
        }
        let start = offset
            .checked_add(8)
            .ok_or_else(|| import_error("Ogawa group offset overflows"))?;
        let byte_len = count
            .checked_mul(8)
            .ok_or_else(|| import_error("Ogawa group size overflows"))?;
        let end = start
            .checked_add(byte_len)
            .ok_or_else(|| import_error("Ogawa group size overflows"))?;
        let raw = self
            .bytes
            .get(start..end)
            .ok_or_else(|| import_error("Ogawa group is truncated"))?;
        self.records = self
            .records
            .checked_add(count)
            .ok_or_else(|| import_error("Ogawa record count overflows"))?;
        if self.records > MAX_ARCHIVE_RECORDS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "Ogawa archive has too many group references",
            ));
        }
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                OgawaRef::from_word(u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]))
            })
            .collect())
    }

    fn data(&self, reference: OgawaRef) -> Result<&'a [u8]> {
        if !reference.is_data {
            return Err(import_error("Ogawa data reference points to a group"));
        }
        if reference.offset == 0 {
            return Ok(&[]);
        }
        let offset = usize_from_u64(reference.offset, "Ogawa data offset")?;
        let length = usize_from_u64(
            super::read_u64(self.bytes, offset, OGAWA_READ_ERRORS)?,
            "Ogawa data size",
        )?;
        let start = offset
            .checked_add(8)
            .ok_or_else(|| import_error("Ogawa data offset overflows"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| import_error("Ogawa data size overflows"))?;
        self.bytes
            .get(start..end)
            .ok_or_else(|| import_error("Ogawa data block is truncated"))
    }
}

struct ParsedProperty {
    name: String,
    property_type: u32,
    pod: u8,
    extent: u8,
    sample_count: usize,
    first_changed: usize,
    last_changed: usize,
    constant: bool,
    time_sampling_index: usize,
    metadata: BTreeMap<String, String>,
    children: Vec<ParsedProperty>,
    samples: Vec<OgawaRef>,
}

struct ParsedObject {
    name: String,
    metadata: BTreeMap<String, String>,
    properties: Vec<ParsedProperty>,
    children: Vec<ParsedObject>,
}

struct TimeSampling {
    time_per_cycle: f64,
    times: Vec<f64>,
}

struct ParsedArchive<'a> {
    bytes: &'a [u8],
    metadata: BTreeMap<String, String>,
    time_samplings: Vec<TimeSampling>,
    objects: Vec<ParsedObject>,
}

fn parse_archive(bytes: &[u8]) -> Result<ParsedArchive<'_>> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic archive exceeds the size limit",
        ));
    }
    if bytes.len() < 16 || bytes.get(..5) != Some(&b"Ogawa"[..]) {
        return Err(import_error("Alembic file does not have Ogawa magic"));
    }
    if bytes[5] != 0xff {
        return Err(import_error("Ogawa archive is not closed"));
    }
    if bytes[6..8] != [0, 1] {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedVersion,
            "Ogawa archive version is unsupported",
            json!({"version": [bytes[6], bytes[7]]}),
        ));
    }
    let root_offset = super::read_u64(bytes, 8, OGAWA_READ_ERRORS)?;
    if root_offset == 0 || root_offset & OGAWA_DATA_FLAG != 0 {
        return Err(import_error("Ogawa root group offset is invalid"));
    }
    let mut reader = OgawaReader { bytes, records: 0 };
    let root = reader.group(OgawaRef {
        offset: root_offset,
        is_data: false,
    })?;
    if root.len() < 6 {
        return Err(import_error("Alembic archive root group is incomplete"));
    }
    let archive_version_data = reader.data(
        *root
            .first()
            .ok_or_else(|| import_error("Alembic archive version is missing"))?,
    )?;
    if archive_version_data.len() != 4 || archive_version_data != 0_i32.to_le_bytes() {
        return Err(PotError::new(
            ErrorCode::UnsupportedVersion,
            "Alembic archive version is unsupported",
        ));
    }
    let file_version_data = reader.data(
        *root
            .get(1)
            .ok_or_else(|| import_error("Alembic archive version is missing"))?,
    )?;
    if file_version_data.len() != 4 {
        return Err(import_error("Alembic archive version data is malformed"));
    }
    let file_version = i32::from_le_bytes([
        file_version_data[0],
        file_version_data[1],
        file_version_data[2],
        file_version_data[3],
    ]);
    if file_version < 10_000 {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedVersion,
            "Alembic archive version is unsupported",
            json!({"version": file_version}),
        ));
    }
    let metadata = parse_metadata(
        reader.data(
            *root
                .get(3)
                .ok_or_else(|| import_error("Alembic archive metadata is missing"))?,
        )?,
    )?;
    let time_samplings = parse_time_sampling_table(
        reader.data(
            *root
                .get(4)
                .ok_or_else(|| import_error("Alembic time sampling data is missing"))?,
        )?,
    )?;
    let indexed_bytes = reader.data(
        *root
            .get(5)
            .ok_or_else(|| import_error("Alembic indexed metadata is missing"))?,
    )?;
    let indexed_metadata = parse_indexed_metadata(indexed_bytes)?;
    let mut active = BTreeSet::new();
    let root_object = parse_object_data(
        &mut reader,
        *root
            .get(2)
            .ok_or_else(|| import_error("Alembic root objects are missing"))?,
        "root".to_owned(),
        BTreeMap::new(),
        &indexed_metadata,
        &mut active,
        0,
    )?;
    Ok(ParsedArchive {
        bytes,
        metadata,
        time_samplings,
        objects: root_object.children,
    })
}

fn parse_object_data(
    reader: &mut OgawaReader<'_>,
    reference: OgawaRef,
    name: String,
    metadata: BTreeMap<String, String>,
    indexed_metadata: &[BTreeMap<String, String>],
    active: &mut BTreeSet<u64>,
    depth: usize,
) -> Result<ParsedObject> {
    if depth > MAX_OBJECT_DEPTH || !active.insert(reference.offset) {
        return Err(import_error(
            "Alembic object hierarchy is cyclic or too deep",
        ));
    }
    let children = reader.group(reference)?;
    if children.len() < 2 {
        return Err(import_error("Alembic object data group is incomplete"));
    }
    let properties = parse_compound(
        reader,
        *children
            .first()
            .ok_or_else(|| import_error("Alembic object properties are missing"))?,
        indexed_metadata,
        &mut BTreeSet::new(),
        0,
    )?;
    let headers_ref = *children
        .last()
        .ok_or_else(|| import_error("Alembic object headers are missing"))?;
    let object_refs = &children[1..children.len() - 1];
    let headers = parse_object_headers(
        reader.data(headers_ref)?,
        object_refs.len(),
        indexed_metadata,
    )?;
    let mut nested = Vec::with_capacity(object_refs.len());
    for (child, (child_name, child_metadata)) in object_refs.iter().copied().zip(headers) {
        if child.is_data {
            return Err(import_error("Alembic child object is not a group"));
        }
        nested.push(parse_object_data(
            reader,
            child,
            child_name,
            child_metadata,
            indexed_metadata,
            active,
            depth + 1,
        )?);
    }
    active.remove(&reference.offset);
    Ok(ParsedObject {
        name,
        metadata,
        properties,
        children: nested,
    })
}

fn parse_compound(
    reader: &mut OgawaReader<'_>,
    reference: OgawaRef,
    indexed_metadata: &[BTreeMap<String, String>],
    active: &mut BTreeSet<u64>,
    depth: usize,
) -> Result<Vec<ParsedProperty>> {
    if depth > MAX_OBJECT_DEPTH || !active.insert(reference.offset) {
        return Err(import_error(
            "Alembic property hierarchy is cyclic or too deep",
        ));
    }
    let children = reader.group(reference)?;
    if children.is_empty() {
        active.remove(&reference.offset);
        return Ok(Vec::new());
    }
    let header_ref = *children
        .last()
        .ok_or_else(|| import_error("Alembic compound headers are missing"))?;
    let headers = parse_property_headers(
        reader.data(header_ref)?,
        children.len() - 1,
        indexed_metadata,
    )?;
    let mut properties = Vec::with_capacity(headers.len());
    for (child, header) in children[..children.len() - 1].iter().copied().zip(headers) {
        if !child.is_data && header.property_type == 0 {
            let nested = parse_compound(reader, child, indexed_metadata, active, depth + 1)?;
            properties.push(ParsedProperty {
                children: nested,
                samples: Vec::new(),
                ..header
            });
        } else {
            let values = reader.group(child)?;
            let expected = match header.property_type {
                1 => sample_storage_count(&header),
                2 => sample_storage_count(&header)
                    .checked_mul(2)
                    .ok_or_else(|| import_error("Alembic array sample count overflows"))?,
                _ => return Err(import_error("Alembic property type is invalid")),
            };
            if values.len() != expected || values.iter().any(|value| !value.is_data) {
                return Err(import_error("Alembic property sample data is malformed"));
            }
            if header.property_type == 2 {
                for dimensions in values.as_chunks::<2>().0.iter().map(|pair| pair[1]) {
                    let _ = reader.data(dimensions)?;
                }
            }
            let samples = if header.property_type == 2 {
                values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| pair[0])
                    .collect()
            } else {
                values
            };
            properties.push(ParsedProperty {
                children: Vec::new(),
                samples,
                ..header
            });
        }
    }
    active.remove(&reference.offset);
    Ok(properties)
}

fn sample_storage_count(header: &ParsedProperty) -> usize {
    if header.constant {
        1
    } else if header.first_changed != 1
        || header.last_changed.saturating_add(1) != header.sample_count
    {
        header
            .last_changed
            .saturating_sub(header.first_changed)
            .saturating_add(2)
    } else {
        header.sample_count
    }
}

fn parse_property_headers(
    bytes: &[u8],
    count: usize,
    indexed_metadata: &[BTreeMap<String, String>],
) -> Result<Vec<ParsedProperty>> {
    let mut cursor = 0;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        let info = cursor_u32(bytes, &mut cursor)?;
        let property_type = match info & 0x3 {
            3 => 2,
            property_type => property_type,
        };
        let size_hint = (info >> 2) & 0x3;
        let pod = u8::try_from((info >> 4) & 0xf)
            .map_err(|_| import_error("Alembic property POD is invalid"))?;
        let has_time_sampling = info & 0x100 != 0;
        let has_changed_range = info & 0x200 != 0;
        let constant = info & 0x800 != 0;
        let extent = u8::try_from((info >> 12) & 0xff)
            .map_err(|_| import_error("Alembic property extent is invalid"))?;
        let metadata_index = usize::from(
            u8::try_from((info >> 20) & 0xff)
                .map_err(|_| import_error("Alembic metadata index is invalid"))?,
        );
        let (sample_count, first_changed, last_changed, time_sampling_index) = if property_type == 0
        {
            (0, 0, 0, 0)
        } else {
            let sample_count = usize_from_u32(
                cursor_hint(bytes, &mut cursor, size_hint)?,
                "Alembic sample count",
            )?;
            let (first_changed, last_changed) = if has_changed_range {
                (
                    usize_from_u32(
                        cursor_hint(bytes, &mut cursor, size_hint)?,
                        "Alembic first changed sample",
                    )?,
                    usize_from_u32(
                        cursor_hint(bytes, &mut cursor, size_hint)?,
                        "Alembic last changed sample",
                    )?,
                )
            } else if constant {
                (0, 0)
            } else {
                (1, sample_count.saturating_sub(1))
            };
            let time_sampling_index = if has_time_sampling {
                usize_from_u32(
                    cursor_hint(bytes, &mut cursor, size_hint)?,
                    "Alembic time sampling index",
                )?
            } else {
                0
            };
            if sample_count == 0
                || extent == 0
                || first_changed > last_changed
                || last_changed >= sample_count
            {
                return Err(import_error("Alembic property sampling header is invalid"));
            }
            (
                sample_count,
                first_changed,
                last_changed,
                time_sampling_index,
            )
        };
        let name_length = usize_from_u32(
            cursor_hint(bytes, &mut cursor, size_hint)?,
            "Alembic property name length",
        )?;
        let name = cursor_string(bytes, &mut cursor, name_length)?;
        let metadata = if metadata_index == 255 {
            let length = usize_from_u32(
                cursor_hint(bytes, &mut cursor, size_hint)?,
                "Alembic property metadata length",
            )?;
            parse_metadata(cursor_slice(bytes, &mut cursor, length)?)?
        } else {
            indexed_metadata
                .get(metadata_index)
                .cloned()
                .ok_or_else(|| import_error("Alembic property metadata index is out of range"))?
        };
        if property_type > 2 {
            return Err(import_error(format!(
                "Alembic property `{name}` has unsupported property type {property_type}"
            )));
        }
        output.push(ParsedProperty {
            name,
            property_type,
            pod,
            extent,
            sample_count,
            first_changed,
            last_changed,
            constant,
            time_sampling_index,
            metadata,
            children: Vec::new(),
            samples: Vec::new(),
        });
    }
    if cursor != bytes.len() {
        return Err(import_error(
            "Alembic property header stream has trailing bytes",
        ));
    }
    Ok(output)
}

fn parse_object_headers(
    bytes: &[u8],
    count: usize,
    indexed_metadata: &[BTreeMap<String, String>],
) -> Result<Vec<(String, BTreeMap<String, String>)>> {
    let header_end = bytes
        .len()
        .checked_sub(32)
        .ok_or_else(|| import_error("Alembic object header hashes are missing"))?;
    let header_bytes = &bytes[..header_end];
    let mut cursor = 0;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = usize_from_u32(
            cursor_u32(header_bytes, &mut cursor)?,
            "Alembic object name length",
        )?;
        let name = cursor_string(header_bytes, &mut cursor, name_len)?;
        let index = usize::from(
            *header_bytes
                .get(cursor)
                .ok_or_else(|| import_error("Alembic object metadata index is truncated"))?,
        );
        cursor += 1;
        let metadata = if index == 255 {
            let length = usize_from_u32(
                cursor_u32(header_bytes, &mut cursor)?,
                "Alembic object metadata length",
            )?;
            parse_metadata(cursor_slice(header_bytes, &mut cursor, length)?)?
        } else {
            indexed_metadata
                .get(index)
                .cloned()
                .ok_or_else(|| import_error("Alembic object metadata index is out of range"))?
        };
        output.push((name, metadata));
    }
    if cursor != header_bytes.len() {
        return Err(import_error(
            "Alembic object header stream has trailing bytes",
        ));
    }
    Ok(output)
}

fn parse_time_sampling_table(bytes: &[u8]) -> Result<Vec<TimeSampling>> {
    if bytes.is_empty() {
        return Err(import_error("Alembic time sampling table is empty"));
    }
    let mut cursor = 0;
    let mut output = Vec::new();
    while cursor < bytes.len() {
        if output.len() > 254 {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "Alembic archive has too many time samplings",
            ));
        }
        let _max_sample = cursor_hint(bytes, &mut cursor, 2)?;
        let time_per_cycle = cursor_f64(bytes, &mut cursor)?;
        let count = usize_from_u32(
            cursor_hint(bytes, &mut cursor, 2)?,
            "Alembic time sample count",
        )?;
        let remaining = bytes.len().saturating_sub(cursor);
        if count == 0 || !time_per_cycle.is_finite() || count > MAX_TIME_SAMPLES {
            return Err(import_error("Alembic time sampling entry is invalid"));
        }
        if count > remaining / 8 {
            return Err(import_error("Alembic time samples are truncated"));
        }
        let mut times = Vec::with_capacity(count);
        for _ in 0..count {
            let time = cursor_f64(bytes, &mut cursor)?;
            if !time.is_finite() {
                return Err(import_error("Alembic time sample is not finite"));
            }
            times.push(time);
        }
        output.push(TimeSampling {
            time_per_cycle,
            times,
        });
    }
    Ok(output)
}

fn parse_indexed_metadata(bytes: &[u8]) -> Result<Vec<BTreeMap<String, String>>> {
    let mut cursor = 0;
    let mut output = vec![BTreeMap::new()];
    while cursor < bytes.len() {
        if output.len() >= 255 {
            return Err(import_error(
                "Alembic indexed metadata count exceeds its limit",
            ));
        }
        let length = usize::from(
            *bytes
                .get(cursor)
                .ok_or_else(|| import_error("Alembic indexed metadata length is truncated"))?,
        );
        cursor += 1;
        output.push(parse_metadata(cursor_slice(bytes, &mut cursor, length)?)?);
    }
    Ok(output)
}

fn parse_metadata(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| import_error("Alembic metadata is not UTF-8"))?;
    let mut output = BTreeMap::new();
    for entry in text.split(';').filter(|entry| !entry.is_empty()) {
        let (key, value) = entry
            .split_once('=')
            .ok_or_else(|| import_error("Alembic metadata entry is malformed"))?;
        if key.is_empty() || output.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(import_error(
                "Alembic metadata contains a duplicate or empty key",
            ));
        }
    }
    Ok(output)
}

fn import_object(
    object: &ParsedObject,
    bytes: &[u8],
    parent: Option<&Id>,
    time_samplings: &[TimeSampling],
    fps: f64,
    fps_base: f64,
    doc: &mut SceneDoc,
    used_ids: &mut BTreeSet<Id>,
    used_data_ids: &mut BTreeSet<Id>,
    used_action_ids: &mut BTreeSet<Id>,
    losses: &mut Vec<Loss>,
    depth: usize,
) -> Result<Option<Id>> {
    if depth > MAX_OBJECT_DEPTH {
        return Err(import_error("Alembic object hierarchy is too deep"));
    }
    let schema = object.metadata.get("schema").map_or("", String::as_str);
    match schema {
        "AbcGeom_Xform_v3" | "AbcGeom_Xform_v2" | "AbcGeom_Xform_v1" => import_xform_object(
            object,
            bytes,
            parent,
            time_samplings,
            fps,
            fps_base,
            doc,
            used_ids,
            used_data_ids,
            used_action_ids,
            losses,
            depth,
        ),
        "AbcGeom_PolyMesh_v1" | "AbcGeom_PolyMesh_v2" => {
            let id = unique_import_id(
                object.metadata.get("potter_id").map(String::as_str),
                &object.name,
                used_ids,
            )?;
            let mesh = import_polymesh(object, bytes, time_samplings, fps, fps_base, losses)?;
            let data_id = unique_data_id(&id, used_data_ids)?;
            doc.data_blocks
                .insert(data_id.clone(), mesh_data_block(mesh));
            doc.nodes.insert(
                id.clone(),
                imported_node(
                    object.name.clone(),
                    "mesh",
                    parent.cloned(),
                    Some(data_id),
                    Transform::default(),
                    None,
                ),
            );
            for child in &object.children {
                if child
                    .metadata
                    .get("schema")
                    .is_some_and(|schema| schema.starts_with("AbcGeom_Xform"))
                {
                    import_object(
                        child,
                        bytes,
                        Some(&id),
                        time_samplings,
                        fps,
                        fps_base,
                        doc,
                        used_ids,
                        used_data_ids,
                        used_action_ids,
                        losses,
                        depth + 1,
                    )?;
                } else {
                    return Err(unsupported(
                        "alembic.object_schema",
                        format!(
                            "Alembic child schema `{}` is unsupported",
                            child
                                .metadata
                                .get("schema")
                                .map_or("unknown", String::as_str)
                        ),
                    ));
                }
            }
            Ok(Some(id))
        }
        _ => Err(unsupported(
            "alembic.object_schema",
            format!("Alembic object schema `{schema}` is unsupported"),
        )),
    }
}

fn import_xform_object(
    object: &ParsedObject,
    bytes: &[u8],
    parent: Option<&Id>,
    time_samplings: &[TimeSampling],
    fps: f64,
    fps_base: f64,
    doc: &mut SceneDoc,
    used_ids: &mut BTreeSet<Id>,
    used_data_ids: &mut BTreeSet<Id>,
    used_action_ids: &mut BTreeSet<Id>,
    losses: &mut Vec<Loss>,
    depth: usize,
) -> Result<Option<Id>> {
    let id = unique_import_id(
        object.metadata.get("potter_id").map(String::as_str),
        &object.name,
        used_ids,
    )?;
    let name = object_name(object)?;
    let xform = property(&object.properties, ".xform")
        .ok_or_else(|| import_error("Alembic Xform schema has no .xform compound"))?;
    if xform.property_type != 0 {
        return Err(import_error("Alembic .xform property is not compound"));
    }
    let inherits = property(&xform.children, ".inherits")
        .map(|property| decode_numeric_sample(bytes, property, 0))
        .transpose()?
        .and_then(|values| values.first().copied())
        .unwrap_or(1.0);
    if inherits == 0.0 && parent.is_some() {
        return Err(unsupported(
            "transform.inherits",
            "Alembic non-inheriting transforms are unsupported".to_owned(),
        ));
    }
    let ops = property(&xform.children, ".ops")
        .ok_or_else(|| import_error("Alembic Xform .ops property is missing"))?;
    let op_values = decode_numeric_sample(bytes, ops, 0)?;
    let op_bytes = op_values
        .iter()
        .map(|value| exact_u8(*value, "Alembic Xform op"))
        .collect::<Result<Vec<_>>>()?;
    if op_bytes.is_empty() {
        return Err(import_error("Alembic Xform has no operations"));
    }
    let vals = property(&xform.children, ".vals")
        .ok_or_else(|| import_error("Alembic Xform .vals property is missing"))?;
    let sample_count = vals.sample_count;
    if sample_count == 0 || sample_count > MAX_TIME_SAMPLES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic Xform has too many samples",
        ));
    }
    let mut matrices = Vec::with_capacity(sample_count);
    let mut frames = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let values = decode_numeric_sample(bytes, vals, sample_index)?;
        matrices.push(xform_matrix(&op_bytes, &values)?);
        frames.push(sample_frame(
            vals,
            sample_index,
            time_samplings,
            fps,
            fps_base,
        )?);
    }
    let transform = matrix_to_transform(
        *matrices
            .first()
            .ok_or_else(|| import_error("Alembic Xform has no samples"))?,
    )?;
    let has_mesh_child = object.children.iter().any(|child| {
        child
            .metadata
            .get("schema")
            .is_some_and(|schema| schema.starts_with("AbcGeom_PolyMesh"))
    });
    let data_id = if has_mesh_child {
        let mesh_object = object
            .children
            .iter()
            .find(|child| {
                child
                    .metadata
                    .get("schema")
                    .is_some_and(|schema| schema.starts_with("AbcGeom_PolyMesh"))
            })
            .ok_or_else(|| import_error("Alembic PolyMesh child is missing"))?;
        let mesh = import_polymesh(mesh_object, bytes, time_samplings, fps, fps_base, losses)?;
        let mesh_id = unique_data_id(&id, used_data_ids)?;
        doc.data_blocks
            .insert(mesh_id.clone(), mesh_data_block(mesh));
        Some(mesh_id)
    } else {
        None
    };
    let action = if matrices.len() > 1 {
        let (action_id, action) =
            create_transform_action(&id, &matrices, &frames, used_action_ids)?;
        doc.actions.insert(action_id.clone(), action);
        Some(action_id)
    } else {
        None
    };
    let node_kind = if data_id.is_some() { "mesh" } else { "empty" };
    doc.nodes.insert(
        id.clone(),
        imported_node(name, node_kind, parent.cloned(), data_id, transform, action),
    );
    for child in &object.children {
        let child_schema = child.metadata.get("schema").map_or("", String::as_str);
        if child_schema.starts_with("AbcGeom_PolyMesh") {
            continue;
        }
        if !child_schema.starts_with("AbcGeom_Xform") {
            return Err(unsupported(
                "alembic.object_schema",
                format!("Alembic child schema `{child_schema}` is unsupported"),
            ));
        }
        import_object(
            child,
            bytes,
            Some(&id),
            time_samplings,
            fps,
            fps_base,
            doc,
            used_ids,
            used_data_ids,
            used_action_ids,
            losses,
            depth + 1,
        )?;
    }
    Ok(Some(id))
}

fn import_polymesh(
    object: &ParsedObject,
    bytes: &[u8],
    time_samplings: &[TimeSampling],
    fps: f64,
    fps_base: f64,
    losses: &mut Vec<Loss>,
) -> Result<Mesh> {
    let geom = property(&object.properties, ".geom")
        .ok_or_else(|| import_error("Alembic PolyMesh schema has no .geom compound"))?;
    if geom.property_type != 0 {
        return Err(import_error("Alembic .geom property is not compound"));
    }
    let positions_prop = property(&geom.children, "P")
        .ok_or_else(|| import_error("Alembic PolyMesh P array is missing"))?;
    let counts_prop = property(&geom.children, ".faceCounts")
        .ok_or_else(|| import_error("Alembic PolyMesh .faceCounts array is missing"))?;
    let indices_prop = property(&geom.children, ".faceIndices")
        .ok_or_else(|| import_error("Alembic PolyMesh .faceIndices array is missing"))?;
    let normals = property(&geom.children, "N");
    let uv = property(&geom.children, "uv");
    let mut sample_count = positions_prop
        .sample_count
        .max(counts_prop.sample_count)
        .max(indices_prop.sample_count);
    for parameter in [normals, uv].into_iter().flatten() {
        let values = geomparam_values_property(parameter)?;
        sample_count = sample_count.max(values.sample_count);
    }
    if sample_count == 0 || sample_count > MAX_TIME_SAMPLES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic PolyMesh has too many samples",
        ));
    }
    for property in [positions_prop, counts_prop, indices_prop] {
        if property.sample_count != 1 && property.sample_count != sample_count {
            return Err(import_error(
                "Alembic PolyMesh sample counts are inconsistent",
            ));
        }
    }
    let mut mesh_samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let positions = decode_vec3_sample(
            bytes,
            positions_prop,
            sample_slot(positions_prop, sample_index),
        )?;
        let counts = decode_i32_sample(bytes, counts_prop, sample_slot(counts_prop, sample_index))?;
        let indices =
            decode_i32_sample(bytes, indices_prop, sample_slot(indices_prop, sample_index))?;
        let faces = decode_faces(&positions, &counts, &indices)?;
        mesh_samples.push(MeshSample {
            positions,
            faces,
            uvs: None,
            normals: None,
        });
    }
    let first = mesh_samples
        .first()
        .ok_or_else(|| import_error("Alembic PolyMesh has no samples"))?;
    let mut mesh = Mesh::from_positions_and_faces(
        first
            .positions
            .iter()
            .copied()
            .map(DVec3::from_array)
            .collect(),
        first.faces.clone(),
    )
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "Alembic PolyMesh topology is invalid",
            json!({"reason":error.to_string()}),
        )
    })?;
    let mut time_property = positions_prop;
    if counts_prop.sample_count > time_property.sample_count {
        time_property = counts_prop;
    }
    if indices_prop.sample_count > time_property.sample_count {
        time_property = indices_prop;
    }
    for parameter in [normals, uv].into_iter().flatten() {
        let values = geomparam_values_property(parameter)?;
        if values.sample_count != 1 && values.sample_count != sample_count {
            return Err(import_error(
                "Alembic geom parameter sample counts are inconsistent",
            ));
        }
        if values.sample_count > time_property.sample_count {
            time_property = values;
        }
        if let Some(indices) = property(&parameter.children, ".indices")
            && indices.sample_count != 1
            && indices.sample_count != sample_count
        {
            return Err(import_error(
                "Alembic geom parameter index sample counts are inconsistent",
            ));
        }
    }
    let frames = (0..sample_count)
        .map(|index| {
            sample_frame(
                time_property,
                sample_slot(time_property, index),
                time_samplings,
                fps,
                fps_base,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let normal_samples = normals
        .map(|parameter| {
            (0..sample_count)
                .map(|sample_index| {
                    let values = geomparam_values_property(parameter)?;
                    geomparam_values(
                        bytes,
                        parameter,
                        &mesh_samples[sample_index],
                        3,
                        sample_slot(values, sample_index),
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let uv_samples = uv
        .map(|parameter| {
            (0..sample_count)
                .map(|sample_index| {
                    let values = geomparam_values_property(parameter)?;
                    geomparam_values(
                        bytes,
                        parameter,
                        &mesh_samples[sample_index],
                        2,
                        sample_slot(values, sample_index),
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    if let Some(normal_values) = normal_samples.as_ref().and_then(|values| values.first()) {
        mesh.attributes.insert(
            "alembic_normals".to_owned(),
            json!({"scope":normal_values.0,"values":normal_values.1}),
        );
    }
    if let Some(uv_values) = uv_samples.as_ref().and_then(|values| values.first()) {
        let mut corner_offset = 0;
        let mut faces_uv = Vec::with_capacity(mesh.faces.len());
        for (face_index, (face, indices)) in mesh.faces.iter().zip(&first.faces).enumerate() {
            let mut coordinates = Vec::with_capacity(indices.len());
            for (corner, vertex) in indices.iter().enumerate() {
                let value_index = match uv_values.0.as_str() {
                    "vertex" | "varying" => *vertex,
                    "facevarying" => corner_offset + corner,
                    "uniform" => face_index,
                    "constant" => 0,
                    scope => {
                        return Err(unsupported(
                            "alembic.geom_scope",
                            format!("Alembic UV scope `{scope}` is unsupported"),
                        ));
                    }
                };
                coordinates.push(
                    uv_values
                        .1
                        .get(value_index)
                        .cloned()
                        .ok_or_else(|| import_error("Alembic UV value index is out of range"))?,
                );
            }
            corner_offset += indices.len();
            faces_uv.push(json!({"face_id":face.id,"uv":coordinates}));
        }
        mesh.attributes
            .insert("uv_map".to_owned(), Value::Array(faces_uv));
    }
    if sample_count > 1 {
        mesh.attributes.insert(
            "alembic_time_samples".to_owned(),
            json!({
                "frames": frames,
                "positions": mesh_samples.iter().map(|sample| &sample.positions).collect::<Vec<_>>(),
                "faces": mesh_samples.iter().map(|sample| &sample.faces).collect::<Vec<_>>(),
                "normals": normal_samples,
                "uv": uv_samples,
            }),
        );
        losses.push(Loss {
            feature_id: "geometry.point_time_samples".to_owned(),
            data_id: Some(object.name.clone()),
            reason: "Alembic polymesh samples are preserved as mesh metadata but are not evaluated as animated scene geometry".to_owned(),
            suggestion: Some("Use an Alembic-capable application for time-varying point geometry".to_owned()),
        });
    }
    Ok(mesh)
}

/// Decoded time sample used by Blender's Alembic mesh cache modifier.
#[derive(Clone, Debug)]
pub(crate) struct MeshCacheSample {
    pub time_seconds: f64,
    pub positions: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    pub uv_faces: Option<Vec<Vec<[f64; 2]>>>,
    pub colors: Option<(String, Vec<Vec<f64>>)>,
    pub velocities: Option<(String, Vec<Vec<f64>>)>,
}

/// Read a `PolyMesh`'s evaluated attributes and time samples from an Ogawa archive.
///
/// The object path is an Alembic path, including the archive's virtual `root` segment.
/// `read_uv` and `read_color` avoid decoding unrequested geometric parameters.
///
/// # Errors
///
/// Returns import errors for malformed archives or paths without a `PolyMesh`.
#[expect(
    clippy::too_many_lines,
    reason = "decoding one cache sample set needs coordinated topology, geom-parameter, and time-sampling validation"
)]
pub(crate) fn read_mesh_cache(
    bytes: &[u8],
    object_path: &str,
    read_uv: bool,
    read_color: bool,
    read_velocity: bool,
) -> Result<Vec<MeshCacheSample>> {
    let archive = parse_archive(bytes)?;
    let blender_coordinates = archive
        .metadata
        .get("application")
        .is_some_and(|application| application.starts_with("Blender"))
        || archive
            .metadata
            .get("_ai_Application")
            .is_some_and(|application| application == "Blender");
    let object = alembic_object_at_path(&archive.objects, object_path)?;
    let object = if is_polymesh(object) {
        object
    } else {
        let mut meshes = Vec::new();
        collect_polymeshes(object, &mut meshes);
        match meshes.as_slice() {
            [mesh] => *mesh,
            [] => {
                return Err(PotError::with_details(
                    ErrorCode::ImportFailed,
                    "Alembic object path does not contain a PolyMesh",
                    json!({"object_path":object_path}),
                ));
            }
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::ImportFailed,
                    "Alembic object path contains multiple PolyMeshes",
                    json!({"object_path":object_path}),
                ));
            }
        }
    };
    let geom = property(&object.properties, ".geom")
        .ok_or_else(|| import_error("Alembic PolyMesh schema has no .geom compound"))?;
    if geom.property_type != 0 {
        return Err(import_error("Alembic .geom property is not compound"));
    }
    let positions = property(&geom.children, "P")
        .ok_or_else(|| import_error("Alembic PolyMesh P array is missing"))?;
    let counts = property(&geom.children, ".faceCounts")
        .ok_or_else(|| import_error("Alembic PolyMesh .faceCounts array is missing"))?;
    let indices = property(&geom.children, ".faceIndices")
        .ok_or_else(|| import_error("Alembic PolyMesh .faceIndices array is missing"))?;
    let uv = read_uv.then(|| property(&geom.children, "uv")).flatten();
    let color = read_color
        .then(|| property(&geom.children, "color").or_else(|| property(&geom.children, "Cs")))
        .flatten();
    let velocity = read_velocity
        .then(|| property(&geom.children, "velocities").or_else(|| property(&geom.children, "v")))
        .flatten()
        .filter(|parameter| matches!(parameter.property_type, 0 | 2));
    let mut sample_count = positions
        .sample_count
        .max(counts.sample_count)
        .max(indices.sample_count);
    let mut time_property = [positions, counts, indices]
        .into_iter()
        .max_by_key(|parameter| parameter.sample_count)
        .ok_or_else(|| import_error("Alembic PolyMesh has no time-sampled properties"))?;
    for parameter in [uv, color, velocity].into_iter().flatten() {
        let values = geomparam_values_property(parameter)?;
        sample_count = sample_count.max(values.sample_count);
        if values.sample_count != 1 && values.sample_count != sample_count {
            return Err(import_error(
                "Alembic geom parameter sample counts are inconsistent",
            ));
        }
        if values.sample_count > time_property.sample_count {
            time_property = values;
        }
    }
    if sample_count == 0 || sample_count > MAX_TIME_SAMPLES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "Alembic PolyMesh has too many samples",
        ));
    }
    for parameter in [positions, counts, indices] {
        if parameter.sample_count != 1 && parameter.sample_count != sample_count {
            return Err(import_error(
                "Alembic PolyMesh sample counts are inconsistent",
            ));
        }
    }
    let mut result = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let point_sample = sample_slot(positions, sample_index);
        let face_sample = sample_slot(counts, sample_index);
        let index_sample = sample_slot(indices, sample_index);
        let points = decode_vec3_sample(bytes, positions, point_sample)?
            .into_iter()
            .map(|[x, y, z]| {
                if blender_coordinates {
                    [x, -z, y]
                } else {
                    [x, y, z]
                }
            })
            .collect::<Vec<_>>();
        let face_counts = decode_i32_sample(bytes, counts, face_sample)?;
        let face_indices = decode_i32_sample(bytes, indices, index_sample)?;
        let faces = decode_faces(&points, &face_counts, &face_indices)?;
        let geometry_sample = MeshSample {
            positions: points.clone(),
            faces: faces.clone(),
            uvs: None,
            normals: None,
        };
        let uv_faces = uv
            .map(|parameter| {
                let values = geomparam_values_property(parameter)?;
                validate_cache_parameter_samples(values, sample_count)?;
                let (scope, tuples) = geomparam_values(
                    bytes,
                    parameter,
                    &geometry_sample,
                    2,
                    sample_slot(values, sample_index),
                )?;
                cache_uv_faces(&faces, &scope, &tuples)
            })
            .transpose()?;
        let colors = color
            .map(|parameter| {
                let values = geomparam_values_property(parameter)?;
                validate_cache_parameter_samples(values, sample_count)?;
                let extent = usize::from(values.extent);
                if !matches!(extent, 3 | 4) {
                    return Err(unsupported(
                        "alembic.color_extent",
                        format!("Alembic color extent {extent} is unsupported"),
                    ));
                }
                geomparam_values(
                    bytes,
                    parameter,
                    &geometry_sample,
                    extent,
                    sample_slot(values, sample_index),
                )
            })
            .transpose()?;
        let velocities = velocity
            .map(|parameter| {
                let values = geomparam_values_property(parameter)?;
                validate_cache_parameter_samples(values, sample_count)?;
                if values.extent != 3 {
                    return Err(unsupported(
                        "alembic.velocity_extent",
                        format!("Alembic velocity extent {} is unsupported", values.extent),
                    ));
                }
                geomparam_values(
                    bytes,
                    parameter,
                    &geometry_sample,
                    3,
                    sample_slot(values, sample_index),
                )
            })
            .transpose()?;
        let velocities = velocities.map(|(scope, values)| {
            let values = if blender_coordinates {
                values
                    .into_iter()
                    .map(|value| vec![value[0], -value[2], value[1]])
                    .collect()
            } else {
                values
            };
            (scope, values)
        });
        let sample_time = sample_seconds(
            time_property,
            sample_slot(time_property, sample_index),
            &archive.time_samplings,
        )?;
        result.push(MeshCacheSample {
            time_seconds: sample_time,
            positions: points,
            faces,
            uv_faces,
            colors,
            velocities,
        });
    }
    Ok(result)
}

/// Sample the cumulative Alembic Xform world matrix at the requested cache time and spatial scale.
///
/// # Errors
///
/// Returns an import error for malformed archives, missing paths, invalid values, or missing Xform
/// schemas.
pub(crate) fn read_xform_matrix(
    bytes: &[u8],
    object_path: &str,
    time_seconds: f64,
    scale: f64,
) -> Result<[f64; 16]> {
    if !time_seconds.is_finite() || !scale.is_finite() {
        return Err(import_error("Alembic Xform time and scale must be finite"));
    }
    let archive = parse_archive(bytes)?;
    let mut segments = object_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.first() == Some(&"root") {
        segments.remove(0);
    }
    if segments.is_empty() {
        return Err(import_error("Alembic Xform object path is empty"));
    }
    let mut children = archive.objects.as_slice();
    let mut world = DMat4::IDENTITY;
    let scale_matrix = DMat4::from_scale(DVec3::splat(scale));
    let mut has_xform = false;
    for (index, segment) in segments.into_iter().enumerate() {
        let object = children
            .iter()
            .find(|object| object.name == segment)
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    format!("Alembic object path segment `{segment}` was not found"),
                    json!({"object_path":object_path,"segment":segment}),
                )
            })?;
        if let Some((local, inherits)) = sample_xform_object(
            object,
            bytes,
            &archive.time_samplings,
            time_seconds,
            index == 0,
        )? {
            world = if inherits {
                world * local
            } else {
                scale_matrix * local
            };
            has_xform = true;
        }
        children = &object.children;
    }
    if !has_xform {
        return Err(PotError::with_details(
            ErrorCode::ImportFailed,
            "Alembic object path has no Xform schema",
            json!({"object_path":object_path}),
        ));
    }
    let blender_coordinates = archive
        .metadata
        .get("application")
        .is_some_and(|application| application.starts_with("Blender"))
        || archive
            .metadata
            .get("_ai_Application")
            .is_some_and(|application| application == "Blender");
    let matrix = if blender_coordinates {
        let basis = DMat4::from_rotation_x(std::f64::consts::FRAC_PI_2);
        (basis * world * basis.inverse()).to_cols_array()
    } else {
        world.to_cols_array()
    };
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(import_error("Alembic Xform matrix is not finite"));
    }
    Ok(matrix)
}

fn sample_xform_object(
    object: &ParsedObject,
    bytes: &[u8],
    samplings: &[TimeSampling],
    time_seconds: f64,
    top_level_object: bool,
) -> Result<Option<(DMat4, bool)>> {
    if !object
        .metadata
        .get("schema")
        .is_some_and(|schema| schema.starts_with("AbcGeom_Xform"))
    {
        return Ok(None);
    }
    let xform = property(&object.properties, ".xform")
        .ok_or_else(|| import_error("Alembic Xform schema has no .xform compound"))?;
    if xform.property_type != 0 {
        return Err(import_error("Alembic .xform property is not compound"));
    }
    // Blender treats an Alembic Xform directly under the archive root as not inheriting,
    // regardless of its schema flag, so CacheFile scale is applied to that root transform.
    let inherits = !top_level_object
        && property(&xform.children, ".inherits")
            .map(|value| decode_numeric_sample(bytes, value, 0))
            .transpose()?
            .and_then(|values| values.first().copied())
            .unwrap_or(1.0)
            != 0.0;
    let ops = property(&xform.children, ".ops")
        .ok_or_else(|| import_error("Alembic Xform .ops property is missing"))?;
    let ops = decode_numeric_sample(bytes, ops, 0)?
        .iter()
        .map(|value| exact_u8(*value, "Alembic Xform op"))
        .collect::<Result<Vec<_>>>()?;
    let values = property(&xform.children, ".vals")
        .ok_or_else(|| import_error("Alembic Xform .vals property is missing"))?;
    if values.sample_count == 0 || values.sample_count > MAX_TIME_SAMPLES {
        return Err(import_error("Alembic Xform sample count is invalid"));
    }
    let mut samples = Vec::with_capacity(values.sample_count);
    for index in 0..values.sample_count {
        let channels = decode_numeric_sample(bytes, values, sample_slot(values, index))?;
        let matrix = xform_matrix(&ops, &channels)?;
        let time = sample_seconds(values, sample_slot(values, index), samplings)?;
        if samples
            .last()
            .is_some_and(|(previous, _): &(f64, [f64; 16])| *previous > time)
        {
            return Err(import_error("Alembic Xform sample times are not ordered"));
        }
        samples.push((time, matrix));
    }
    let matrix = interpolate_xform_samples(&samples, time_seconds)?;
    Ok(Some((DMat4::from_cols_array(&matrix), inherits)))
}

fn interpolate_xform_samples(samples: &[(f64, [f64; 16])], time_seconds: f64) -> Result<[f64; 16]> {
    let Some((first_time, first_matrix)) = samples.first() else {
        return Err(import_error("Alembic Xform has no samples"));
    };
    if time_seconds <= *first_time {
        return Ok(*first_matrix);
    }
    for pair in samples.windows(2) {
        let [(left_time, left), (right_time, right)] = pair else {
            continue;
        };
        if time_seconds > *right_time {
            continue;
        }
        let exact_sample_tolerance = f64::EPSILON * right_time.abs().max(1.0);
        if (time_seconds - right_time).abs() <= exact_sample_tolerance {
            return Ok(*right);
        }
        let weight = (time_seconds - left_time) / (right_time - left_time);
        let left = DMat4::from_cols_array(left);
        let right = DMat4::from_cols_array(right);
        let (left_scale, left_rotation, left_translation) = left.to_scale_rotation_translation();
        let (right_scale, right_rotation, right_translation) =
            right.to_scale_rotation_translation();
        let matrix = DMat4::from_scale_rotation_translation(
            left_scale.lerp(right_scale, weight),
            left_rotation.slerp(right_rotation, weight).normalize(),
            left_translation.lerp(right_translation, weight),
        );
        return Ok(matrix.to_cols_array());
    }
    samples
        .last()
        .map(|(_, matrix)| *matrix)
        .ok_or_else(|| import_error("Alembic Xform has no samples"))
}

fn alembic_object_at_path<'a>(
    objects: &'a [ParsedObject],
    object_path: &str,
) -> Result<&'a ParsedObject> {
    let mut segments = object_path.split('/').filter(|segment| !segment.is_empty());
    if segments.clone().next() == Some("root") {
        segments.next();
    }
    let mut children = objects;
    let mut found = None;
    for segment in segments {
        found = Some(
            children
                .iter()
                .find(|object| object.name == segment)
                .ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::ImportFailed,
                        format!("Alembic object path segment `{segment}` was not found"),
                        json!({"object_path":object_path,"segment":segment}),
                    )
                })?,
        );
        children = &found
            .ok_or_else(|| import_error("Alembic object path is empty"))?
            .children;
    }
    found.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "Alembic object path is empty",
            json!({"object_path":object_path}),
        )
    })
}

fn is_polymesh(object: &ParsedObject) -> bool {
    object
        .metadata
        .get("schema")
        .is_some_and(|schema| schema.starts_with("AbcGeom_PolyMesh"))
}

fn collect_polymeshes<'a>(object: &'a ParsedObject, output: &mut Vec<&'a ParsedObject>) {
    if is_polymesh(object) {
        output.push(object);
    }
    for child in &object.children {
        collect_polymeshes(child, output);
    }
}

fn validate_cache_parameter_samples(property: &ParsedProperty, sample_count: usize) -> Result<()> {
    if property.sample_count != 1 && property.sample_count != sample_count {
        return Err(import_error(
            "Alembic geom parameter sample counts are inconsistent",
        ));
    }
    Ok(())
}

fn cache_uv_faces(
    faces: &[Vec<usize>],
    scope: &str,
    values: &[Vec<f64>],
) -> Result<Vec<Vec<[f64; 2]>>> {
    let mut corner_offset = 0;
    faces
        .iter()
        .enumerate()
        .map(|(face_index, face)| {
            face.iter()
                .enumerate()
                .map(|(corner, vertex)| {
                    let value_index = match scope {
                        "vertex" | "varying" => *vertex,
                        "facevarying" => corner_offset + corner,
                        "uniform" => face_index,
                        "constant" => 0,
                        _ => {
                            return Err(unsupported(
                                "alembic.geom_scope",
                                format!("Alembic UV scope `{scope}` is unsupported"),
                            ));
                        }
                    };
                    let value = values
                        .get(value_index)
                        .ok_or_else(|| import_error("Alembic UV value index is out of range"))?;
                    let coordinates = value
                        .as_slice()
                        .try_into()
                        .map_err(|_| import_error("Alembic UV tuple extent is invalid"))?;
                    Ok(coordinates)
                })
                .collect::<Result<Vec<[f64; 2]>>>()
                .inspect(|_| {
                    corner_offset += face.len();
                })
        })
        .collect()
}

fn sample_seconds(
    property: &ParsedProperty,
    sample_index: usize,
    samplings: &[TimeSampling],
) -> Result<f64> {
    let sampling = samplings
        .get(property.time_sampling_index)
        .ok_or_else(|| import_error("Alembic property time sampling index is out of range"))?;
    if sample_index >= property.sample_count || sampling.times.is_empty() {
        return Err(import_error("Alembic time sample index is out of range"));
    }
    let time_index = sample_index % sampling.times.len();
    let cycles = f64::from(
        u32::try_from(sample_index / sampling.times.len())
            .map_err(|_| import_error("Alembic time sample cycle exceeds uint32"))?,
    );
    let time = sampling
        .times
        .get(time_index)
        .copied()
        .ok_or_else(|| import_error("Alembic time sample is missing"))?
        + cycles * sampling.time_per_cycle;
    if !time.is_finite() {
        return Err(import_error("Alembic time sample is not finite"));
    }
    Ok(time)
}

fn geomparam_values_property(parameter: &ParsedProperty) -> Result<&ParsedProperty> {
    let values = match parameter.property_type {
        0 => property(&parameter.children, ".vals")
            .ok_or_else(|| import_error("Alembic geom parameter .vals is missing"))?,
        2 => parameter,
        _ => {
            return Err(import_error(
                "Alembic geom parameter has an invalid property type",
            ));
        }
    };
    if values.property_type != 2 {
        return Err(import_error(
            "Alembic geom parameter values are not an array",
        ));
    }
    Ok(values)
}

fn geomparam_values(
    bytes: &[u8],
    geomparam: &ParsedProperty,
    sample: &MeshSample,
    extent: usize,
    sample_index: usize,
) -> Result<(String, Vec<Vec<f64>>)> {
    let values_prop = geomparam_values_property(geomparam)?;
    let values = decode_float_sample(bytes, values_prop, sample_index)?;
    if values.len() % extent != 0 {
        return Err(import_error("Alembic geom parameter extent is invalid"));
    }
    let tuples = values
        .chunks_exact(extent)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    let corner_count = sample.faces.iter().map(Vec::len).sum::<usize>();
    let stored_scope = geomparam
        .metadata
        .get("geoScope")
        .map_or("vertex", String::as_str);
    let scope = match stored_scope {
        "vertex" | "vtx" => "vertex",
        "varying" | "vry" => "varying",
        "facevarying" | "fvr" => "facevarying",
        "uniform" | "uni" => "uniform",
        "constant" | "con" => "constant",
        scope => {
            return Err(unsupported(
                "alembic.geom_scope",
                format!("Alembic geometry scope `{scope}` is unsupported"),
            ));
        }
    }
    .to_owned();
    let output_count = match scope.as_str() {
        "vertex" | "varying" => sample.positions.len(),
        "facevarying" => corner_count,
        "uniform" => sample.faces.len(),
        "constant" => 1,
        _ => unreachable!(),
    };
    if let Some(indices_prop) = (geomparam.property_type == 0)
        .then(|| property(&geomparam.children, ".indices"))
        .flatten()
    {
        let indices = decode_indices(bytes, indices_prop, sample_slot(indices_prop, sample_index))?;
        if indices.len() != output_count {
            return Err(import_error(
                "Alembic geom parameter index count is invalid",
            ));
        }
        let expanded = indices
            .into_iter()
            .map(|index| {
                tuples
                    .get(index)
                    .cloned()
                    .ok_or_else(|| import_error("Alembic geom parameter index is out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((scope, expanded))
    } else {
        if tuples.len() != output_count {
            return Err(import_error(
                "Alembic geom parameter value count does not match its scope",
            ));
        }
        Ok((scope, tuples))
    }
}

fn sample_slot(property: &ParsedProperty, sample_index: usize) -> usize {
    if property.sample_count == 1 {
        0
    } else {
        sample_index
    }
}

fn decode_faces(
    positions: &[[f64; 3]],
    counts: &[i32],
    indices: &[i32],
) -> Result<Vec<Vec<usize>>> {
    let mut offset = 0_usize;
    let mut faces = Vec::with_capacity(counts.len());
    for count in counts {
        let count =
            usize::try_from(*count).map_err(|_| import_error("Alembic face count is negative"))?;
        if count < 3 {
            return Err(import_error(
                "Alembic mesh face has fewer than three vertices",
            ));
        }
        let end = offset
            .checked_add(count)
            .ok_or_else(|| import_error("Alembic mesh face count overflows"))?;
        let values = indices
            .get(offset..end)
            .ok_or_else(|| import_error("Alembic mesh face indices are truncated"))?;
        let face = values
            .iter()
            .map(|index| {
                let index = usize::try_from(*index)
                    .map_err(|_| import_error("Alembic vertex index is negative"))?;
                if index >= positions.len() {
                    return Err(import_error("Alembic vertex index is out of range"));
                }
                Ok(index)
            })
            .collect::<Result<Vec<_>>>()?;
        faces.push(face);
        offset = end;
    }
    if offset != indices.len() {
        return Err(import_error(
            "Alembic face counts do not match face indices",
        ));
    }
    Ok(faces)
}

fn decode_vec3_sample(
    bytes: &[u8],
    property: &ParsedProperty,
    index: usize,
) -> Result<Vec<[f64; 3]>> {
    if property.extent != 3 {
        return Err(import_error("Alembic P array extent is not three"));
    }
    let values = decode_float_sample(bytes, property, index)?;
    if values.len() % 3 != 0 {
        return Err(import_error("Alembic P array size is invalid"));
    }
    Ok(values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|value| [value[0], value[1], value[2]])
        .collect())
}

fn decode_i32_sample(bytes: &[u8], property: &ParsedProperty, index: usize) -> Result<Vec<i32>> {
    if property.pod != 6 || property.extent != 1 {
        return Err(unsupported(
            "alembic.array_type",
            format!("Alembic property `{}` is not int32 data", property.name),
        ));
    }
    let raw = raw_sample(bytes, property, index)?;
    if raw.len() % 4 != 0 {
        return Err(import_error("Alembic int32 sample is malformed"));
    }
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn decode_indices(bytes: &[u8], property: &ParsedProperty, index: usize) -> Result<Vec<usize>> {
    if property.extent != 1 {
        return Err(import_error(
            "Alembic geom parameter index extent is invalid",
        ));
    }
    let raw = raw_sample(bytes, property, index)?;
    match property.pod {
        5 => {
            if raw.len() % 4 != 0 {
                return Err(import_error("Alembic uint32 indices are malformed"));
            }
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| {
                    usize::try_from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .map_err(|_| import_error("Alembic geom parameter index is out of range"))
                })
                .collect()
        }
        6 => decode_i32_sample(bytes, property, index)?
            .into_iter()
            .map(|value| {
                usize::try_from(value)
                    .map_err(|_| import_error("Alembic geom parameter index is negative"))
            })
            .collect(),
        _ => Err(unsupported(
            "alembic.array_type",
            format!(
                "Alembic property `{}` is not an integer index array",
                property.name
            ),
        )),
    }
}

fn decode_float_sample(bytes: &[u8], property: &ParsedProperty, index: usize) -> Result<Vec<f64>> {
    if !matches!(property.pod, 10 | 11) {
        return Err(unsupported(
            "alembic.array_type",
            format!("Alembic property `{}` is not float data", property.name),
        ));
    }
    let raw = raw_sample(bytes, property, index)?;
    let width = if property.pod == 10 { 4 } else { 8 };
    if raw.len() % width != 0 || raw.len() % (width * usize::from(property.extent)) != 0 {
        return Err(import_error("Alembic floating-point sample is malformed"));
    }
    if property.pod == 10 {
        Ok(raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect())
    } else {
        Ok(raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect())
    }
}

fn decode_numeric_sample(
    bytes: &[u8],
    property: &ParsedProperty,
    index: usize,
) -> Result<Vec<f64>> {
    match property.pod {
        0 | 1 => Ok(raw_sample(bytes, property, index)?
            .iter()
            .map(|byte| f64::from(*byte))
            .collect()),
        5 => {
            let raw = raw_sample(bytes, property, index)?;
            if raw.len() % 4 != 0 {
                return Err(import_error("Alembic uint32 sample is malformed"));
            }
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| {
                    f64::from(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                })
                .collect())
        }
        6 => Ok(decode_i32_sample(bytes, property, index)?
            .into_iter()
            .map(f64::from)
            .collect()),
        10 | 11 => decode_float_sample(bytes, property, index),
        _ => Err(unsupported(
            "alembic.array_type",
            format!(
                "Alembic property `{}` uses unsupported POD type {}",
                property.name, property.pod
            ),
        )),
    }
}

fn raw_sample<'a>(
    bytes: &'a [u8],
    property: &ParsedProperty,
    logical_index: usize,
) -> Result<&'a [u8]> {
    if logical_index >= property.sample_count {
        return Err(import_error("Alembic sample index is out of range"));
    }
    let stored_index = if property.constant || logical_index < property.first_changed {
        0
    } else if logical_index >= property.last_changed {
        property
            .last_changed
            .saturating_sub(property.first_changed)
            .saturating_add(1)
    } else {
        logical_index
            .saturating_sub(property.first_changed)
            .saturating_add(1)
    };
    let reference = property
        .samples
        .get(stored_index)
        .copied()
        .ok_or_else(|| import_error("Alembic sample data is missing"))?;
    let data = OgawaReader { bytes, records: 0 }.data(reference)?;
    if data.len() < 16 {
        return Err(import_error("Alembic sample hash is truncated"));
    }
    Ok(&data[16..])
}

fn sample_frame(
    property: &ParsedProperty,
    sample_index: usize,
    samplings: &[TimeSampling],
    fps: f64,
    fps_base: f64,
) -> Result<f64> {
    let sampling = samplings
        .get(property.time_sampling_index)
        .ok_or_else(|| import_error("Alembic property time sampling index is out of range"))?;
    if sample_index >= property.sample_count {
        return Err(import_error("Alembic time sample index is out of range"));
    }
    let time_index = sample_index % sampling.times.len();
    let cycles = f64::from(
        u32::try_from(sample_index / sampling.times.len())
            .map_err(|_| import_error("Alembic time sample cycle exceeds uint32"))?,
    );
    let time = sampling
        .times
        .get(time_index)
        .copied()
        .ok_or_else(|| import_error("Alembic time sample is missing"))?
        + cycles * sampling.time_per_cycle;
    let frame = time * fps / fps_base;
    if !frame.is_finite() {
        return Err(import_error("Alembic frame sample is not finite"));
    }
    Ok(frame)
}

fn xform_matrix(ops: &[u8], values: &[f64]) -> Result<[f64; 16]> {
    let mut cursor = 0_usize;
    let mut matrix = DMat4::IDENTITY;
    for encoded in ops {
        let operation = encoded >> 4;
        let channel_count = match operation {
            0 | 1 => 3,
            2 => 4,
            3 => 16,
            4..=6 => 1,
            _ => {
                return Err(unsupported(
                    "alembic.xform_op",
                    format!("Alembic Xform op type `{operation}` is unsupported"),
                ));
            }
        };
        let end = cursor
            .checked_add(channel_count)
            .ok_or_else(|| import_error("Alembic Xform channel count overflows"))?;
        let channels = values
            .get(cursor..end)
            .ok_or_else(|| import_error("Alembic Xform channels are truncated"))?;
        let operation_matrix = match operation {
            0 => DMat4::from_scale(DVec3::new(channels[0], channels[1], channels[2])),
            1 => DMat4::from_translation(DVec3::new(channels[0], channels[1], channels[2])),
            2 => {
                let axis = DVec3::new(channels[0], channels[1], channels[2]);
                if axis.length_squared() <= f64::EPSILON {
                    return Err(import_error("Alembic Xform rotation axis is zero"));
                }
                DMat4::from_quat(glam::DQuat::from_axis_angle(
                    axis.normalize(),
                    channels[3].to_radians(),
                ))
            }
            3 => DMat4::from_cols_array(
                channels
                    .try_into()
                    .map_err(|_| import_error("Alembic matrix op is malformed"))?,
            ),
            4 => DMat4::from_rotation_x(channels[0].to_radians()),
            5 => DMat4::from_rotation_y(channels[0].to_radians()),
            6 => DMat4::from_rotation_z(channels[0].to_radians()),
            _ => return Err(import_error("Alembic Xform op type is invalid")),
        };
        matrix *= operation_matrix;
        cursor = end;
    }
    if cursor != values.len() || !matrix.is_finite() {
        return Err(import_error(
            "Alembic Xform has invalid or non-finite channels",
        ));
    }
    Ok(matrix.to_cols_array())
}

fn matrix_to_transform(matrix_values: [f64; 16]) -> Result<Transform> {
    let matrix = DMat4::from_cols_array(&matrix_values);
    let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
    if !scale.is_finite() || !rotation.is_finite() || !translation.is_finite() {
        return Err(import_error("Alembic Xform matrix is not finite"));
    }
    let transform = Transform::from_rotation_quat(
        translation.to_array(),
        [rotation.x, rotation.y, rotation.z, rotation.w],
        scale.to_array(),
    )
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "Alembic Xform matrix is invalid",
            json!({"reason":error.message}),
        )
    })?;
    let reconstructed = transform.matrix().to_cols_array();
    if matrix_values
        .iter()
        .zip(reconstructed)
        .any(|(original, value)| (*original - value).abs() > 1.0e-8 * (1.0 + original.abs()))
    {
        return Err(unsupported(
            "transform.matrix_shear",
            "Alembic Xform matrix contains shear not representable as TRS".to_owned(),
        ));
    }
    Ok(transform)
}

fn create_transform_action(
    id: &Id,
    matrices: &[[f64; 16]],
    frames: &[f64],
    used: &mut BTreeSet<Id>,
) -> Result<(Id, Action)> {
    if matrices.len() != frames.len() {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "Alembic transform samples and times differ",
        ));
    }
    let mut transforms = matrices
        .iter()
        .copied()
        .map(matrix_to_transform)
        .collect::<Result<Vec<_>>>()?;
    for index in 1..transforms.len() {
        let previous = transforms
            .get(index - 1)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "Alembic previous rotation sample is missing",
                )
            })?
            .rotation;
        let current = transforms
            .get(index)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "Alembic rotation sample is missing",
                )
            })?
            .rotation;
        let dot = previous[0] * current[0]
            + previous[1] * current[1]
            + previous[2] * current[2]
            + previous[3] * current[3];
        if dot < 0.0 {
            for component in &mut transforms
                .get_mut(index)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "Alembic rotation sample is missing",
                    )
                })?
                .rotation
            {
                *component = -*component;
            }
        }
    }
    let mut curves = Vec::new();
    for (path, width, component) in [
        ("transform.translation", 3_usize, 0_usize),
        ("transform.rotation_quaternion", 4, 1),
        ("transform.scale", 3, 2),
    ] {
        for index in 0..width {
            let keyframes = transforms
                .iter()
                .zip(frames)
                .map(|(transform, frame)| {
                    let value = match component {
                        0 => transform.translation[index],
                        1 => transform.rotation[index],
                        _ => transform.scale[index],
                    };
                    Keyframe {
                        frame: *frame,
                        value,
                        interpolation: Interpolation::Linear,
                        ..Keyframe::default()
                    }
                })
                .collect();
            curves.push(FCurve {
                path: path.to_owned(),
                index: u32::try_from(index).map_err(|_| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "Alembic channel index exceeds uint32",
                    )
                })?,
                keyframes,
                extrapolation: crate::model::Extrapolation::default(),
            });
        }
    }
    let base = id.as_str().chars().take(48).collect::<String>();
    let mut stem = format!("{base}_action");
    let mut suffix = 0_u32;
    loop {
        let candidate = Id::new(stem.clone())?;
        if used.insert(candidate.clone()) {
            return Ok((
                candidate,
                Action {
                    name: "Alembic Transform".to_owned(),
                    fcurves: curves,
                    ..Action::default()
                },
            ));
        }
        suffix = suffix.checked_add(1).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Alembic action ID suffix overflow",
            )
        })?;
        stem = format!("{base}_action_{suffix}");
    }
}

fn imported_node(
    name: String,
    kind: &str,
    parent: Option<Id>,
    data: Option<Id>,
    transform: Transform,
    action: Option<Id>,
) -> Node {
    Node {
        name,
        kind: kind.to_owned(),
        parent,
        transform,
        data,
        action,
        visible: true,
        render_visible: true,
        selectable: true,
        ..Node::default()
    }
}

fn mesh_data_block(mesh: Mesh) -> DataBlock {
    DataBlock {
        data_type: "mesh".to_owned(),
        mesh: Some(mesh),
        ..DataBlock::default()
    }
}

fn unique_import_id(raw: Option<&str>, fallback: &str, used: &mut BTreeSet<Id>) -> Result<Id> {
    if let Some(raw) = raw
        && let Ok(id) = Id::new(raw.to_owned())
        && let Some(id) = super::claim_import_id(id, used)
    {
        return Ok(id);
    }
    let mut seed = String::with_capacity(fallback.len());
    for byte in fallback.bytes() {
        let character = char::from(byte).to_ascii_lowercase();
        seed.push(
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
    if seed.is_empty() || !seed.as_bytes()[0].is_ascii_lowercase() {
        seed.insert_str(0, "node_");
    }
    seed.truncate(56);
    for index in 0_u32.. {
        let candidate = if index == 0 {
            seed.clone()
        } else {
            format!("{seed}_{index}")
        };
        let id = Id::new(candidate)?;
        if let Some(id) = super::claim_import_id(id, used) {
            return Ok(id);
        }
    }
    Err(PotError::new(
        ErrorCode::LimitExceeded,
        "Alembic import exhausted node IDs",
    ))
}

fn unique_data_id(id: &Id, used: &mut BTreeSet<Id>) -> Result<Id> {
    let mut stem = format!("{}_mesh", id.as_str());
    if stem.len() > 55 {
        stem.truncate(55);
    }
    let first = Id::new(stem.clone())?;
    let result = super::unique_data_id(
        first,
        |candidate| used.contains(candidate),
        |suffix| Id::new(format!("{stem}_{suffix}")),
        1,
        "Alembic import exhausted data IDs",
    )?;
    used.insert(result.clone());
    Ok(result)
}

fn object_name(object: &ParsedObject) -> Result<String> {
    let Some(encoded) = object.metadata.get("potter_name_hex") else {
        return Ok(object.name.clone());
    };
    let bytes = hex::decode(encoded)
        .map_err(|_| import_error("Alembic Potter name metadata is malformed"))?;
    String::from_utf8(bytes).map_err(|_| import_error("Alembic Potter name metadata is not UTF-8"))
}

fn property<'a>(properties: &'a [ParsedProperty], name: &str) -> Option<&'a ParsedProperty> {
    properties.iter().find(|property| property.name == name)
}

fn metadata_f64(metadata: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    metadata.get(key).and_then(|value| value.parse().ok())
}

fn float_to_u32(value: f64, field: &str) -> Result<u32> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 || value > f64::from(u32::MAX) {
        return Err(import_error(format!(
            "{field} metadata is not a positive uint32"
        )));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated uint32 frame rate"
    )]
    let result = value as u32;
    Ok(result)
}

fn exact_u8(value: f64, field: &str) -> Result<u8> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=f64::from(u8::MAX)).contains(&value) {
        return Err(import_error(format!("{field} is not a uint8")));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated uint8 Alembic operation code"
    )]
    let result = value as u8;
    Ok(result)
}

fn cursor_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32> {
    let raw = cursor_slice(bytes, cursor, 4)?;
    Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
}

fn cursor_f64(bytes: &[u8], cursor: &mut usize) -> Result<f64> {
    let raw = cursor_slice(bytes, cursor, 8)?;
    Ok(f64::from_le_bytes([
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
    ]))
}

fn cursor_hint(bytes: &[u8], cursor: &mut usize, hint: u32) -> Result<u32> {
    let width = match hint {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => return Err(import_error("Alembic integer size hint is invalid")),
    };
    let raw = cursor_slice(bytes, cursor, width)?;
    let mut full = [0_u8; 4];
    full[..width].copy_from_slice(raw);
    Ok(u32::from_le_bytes(full))
}

fn cursor_string(bytes: &[u8], cursor: &mut usize, length: usize) -> Result<String> {
    let raw = cursor_slice(bytes, cursor, length)?;
    std::str::from_utf8(raw)
        .map(str::to_owned)
        .map_err(|_| import_error("Alembic string is not UTF-8"))
}

fn cursor_slice<'a>(bytes: &'a [u8], cursor: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = cursor
        .checked_add(length)
        .ok_or_else(|| import_error("Alembic header offset overflows"))?;
    let raw = bytes
        .get(*cursor..end)
        .ok_or_else(|| import_error("Alembic header stream is truncated"))?;
    *cursor = end;
    Ok(raw)
}

fn usize_from_u64(value: u64, description: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| import_error(format!("{description} is out of bounds")))
}

fn usize_from_u32(value: u32, description: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| import_error(format!("{description} is out of bounds")))
}

fn unsupported(feature_id: &str, message: String) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id":feature_id}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::BoxParams;

    const SOURCE_SCENE_ID: &str = "00000000-0000-4000-8000-000000000001";
    const IMPORT_SCENE_ID: &str = "00000000-0000-4000-8000-000000000002";

    #[test]
    fn round_trip_preserves_mesh_topology_and_transform_samples() -> Result<()> {
        let mut doc = SceneDoc::new(SOURCE_SCENE_ID.to_owned());
        let mesh_id = Id::new("box")?;
        let data_id = Id::new("box_mesh")?;
        let action_id = Id::new("box_move")?;
        let mesh = Mesh::box_mesh(BoxParams::default())
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.to_string()))?;
        let expected_vertices = mesh.vertices.len();
        let expected_faces = mesh.faces.len();
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            mesh_id.clone(),
            Node {
                name: "Round-trip box".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                action: Some(action_id.clone()),
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "test root collection is missing")
            })?
            .objects
            .push(mesh_id.clone());
        doc.actions.insert(
            action_id,
            Action {
                name: "Box move".to_owned(),
                fcurves: vec![FCurve {
                    path: "transform.translation".to_owned(),
                    index: 0,
                    keyframes: vec![
                        Keyframe {
                            frame: 1.0,
                            value: 0.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                        Keyframe {
                            frame: 2.0,
                            value: 3.0,
                            interpolation: Interpolation::Linear,
                            ..Keyframe::default()
                        },
                    ],
                    extrapolation: crate::model::Extrapolation::default(),
                }],
                ..Action::default()
            },
        );
        let snapshot = Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                frame: Some(1.0),
                ..EvaluationContext::default()
            },
        )?;
        let archive = export(&doc, &snapshot, false, None)?;
        assert_eq!(archive.get(..5), Some(b"Ogawa".as_slice()));
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), archive).map_err(|error| PotError::io(&error))?;
        let imported = import(file.path(), IMPORT_SCENE_ID.to_owned(), None)?;
        let imported_node = imported.doc.nodes.get(&mesh_id).ok_or_else(|| {
            PotError::new(ErrorCode::ImportFailed, "round-trip mesh node is missing")
        })?;
        let imported_data_id = imported_node.data.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::ImportFailed,
                "round-trip mesh data ID is missing",
            )
        })?;
        let imported_mesh = imported
            .doc
            .data_blocks
            .get(imported_data_id)
            .and_then(|block| block.mesh.as_ref())
            .ok_or_else(|| {
                PotError::new(ErrorCode::ImportFailed, "round-trip mesh data is missing")
            })?;
        assert_eq!(imported_mesh.vertices.len(), expected_vertices);
        assert_eq!(imported_mesh.faces.len(), expected_faces);
        let imported_action_id = imported_node.action.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::ImportFailed,
                "round-trip transform action is missing",
            )
        })?;
        let imported_action = imported
            .doc
            .actions
            .get(imported_action_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::ImportFailed, "round-trip action data is missing")
            })?;
        let x_curve = imported_action
            .fcurves
            .iter()
            .find(|curve| curve.path == "transform.translation" && curve.index == 0)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::ImportFailed,
                    "round-trip X translation curve is missing",
                )
            })?;
        let final_key = x_curve
            .keyframes
            .iter()
            .find(|key| key.frame == 2.0)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::ImportFailed,
                    "round-trip frame-two key is missing",
                )
            })?;
        assert!((final_key.value - 3.0).abs() < 1.0e-6);
        assert_eq!(
            imported.doc.scenes[&imported.doc.active_scene].frame_current,
            1.0
        );
        Ok(())
    }

    #[test]
    fn rejects_non_ogawa_input_without_a_converter() -> Result<()> {
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), b"not an Alembic archive").map_err(|error| PotError::io(&error))?;
        let error = import(file.path(), IMPORT_SCENE_ID.to_owned(), None)
            .err()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "invalid Alembic input was accepted",
                )
            })?;
        assert_eq!(error.code, ErrorCode::ImportFailed);
        Ok(())
    }
}
