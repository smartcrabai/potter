use std::collections::BTreeSet;

use glam::{DMat4, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, sculpt as sculpt_geom},
    model::{Id, Modifier, SceneDoc},
};

use super::{ChangeKind, Engine, check_fields, resolve_node_targets};

/// Blender's subdivision/multires level range (`rna_modifier.cc` `levels` hard maximum).
const MAX_MULTIRES_LEVELS: u64 = 11;

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "sculpt.stroke" => apply_stroke(engine, operation),
        "sculpt.dyntopo" => apply_dyntopo(engine, operation),
        "sculpt.mask" => apply_mask(engine, operation),
        "sculpt.face_set" => apply_face_set(engine, operation),
        "sculpt.remesh_voxel" => apply_remesh(engine, operation),
        "sculpt.multires" => apply_multires(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported sculpt operation `{name}`"),
            "/op",
        )),
    }
}

fn apply_stroke(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "target", "scope", "brush", "samples", "falloff", "symmetry", "delta", "view",
            "dyntopo",
        ],
        &["target", "brush", "samples"],
    )?;
    let stroke = parse_stroke(engine, operation)?;
    let dyntopo_edge_length = parse_dyntopo_edge_length(engine, operation)?;
    let targets = prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (node_id, data_id) in targets {
        let world = node_world_matrix(engine.doc, &node_id)?;
        let inverse = world.inverse();
        if !inverse.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "sculpt target has a singular world transform",
                "/target",
            ));
        }
        let before = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|data| data.mesh.as_ref())
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target data has no mesh",
                    "/target",
                )
            })?;
        if dyntopo_edge_length.is_some() && has_multires_detail(engine, &node_id, &before) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "dynamic topology is unavailable on a mesh with Multires detail",
                "/dyntopo",
            ));
        }
        let multires_modifier = engine
            .doc
            .nodes
            .get(&node_id)
            .and_then(|node| active_multires_modifier(&node.modifiers, dyntopo_edge_length))
            .map(|(index, modifier)| (index, modifier.clone()));
        let (sculpted_mesh, mesh_changed) = if let Some((modifier_index, modifier)) =
            multires_modifier
        {
            let deformed_base = multires_deformed_base(engine, &node_id, &before, modifier_index)?;
            crate::geom::modifiers::sculpt_multires(
                &before,
                deformed_base.as_ref().unwrap_or(&before),
                &modifier,
                &stroke,
                world,
            )?
        } else {
            let mut world_mesh = before;
            for vertex in &mut world_mesh.vertices {
                vertex.co = world.transform_point3(vertex.co);
            }
            let mesh_changed = sculpt_geom::apply_stroke_with_dyntopo(
                &mut world_mesh,
                &stroke,
                dyntopo_edge_length,
            )?;
            if mesh_changed {
                for vertex in &mut world_mesh.vertices {
                    vertex.co = inverse.transform_point3(vertex.co);
                }
            }
            (world_mesh, mesh_changed)
        };

        if mesh_changed {
            let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
                PotError::new(ErrorCode::TargetNotFound, "mesh data block disappeared")
            })?;
            block.mesh = Some(sculpted_mesh);
            block.descriptor = None;
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            changed = true;
        }
    }
    Ok(changed)
}

/// Whether the node carries Multires detail: a Multires modifier or stored `multires_displacement`.
/// Dynamic topology would change the corner count those rows are keyed to, so it is refused.
fn has_multires_detail(engine: &Engine<'_>, node_id: &Id, mesh: &Mesh) -> bool {
    mesh.attributes.contains_key("multires_displacement")
        || engine.doc.nodes.get(node_id).is_some_and(|node| {
            node.modifiers
                .iter()
                .any(|modifier| modifier.modifier_type == "multires")
        })
}

fn active_multires_modifier(
    modifiers: &[Modifier],
    dyntopo_edge_length: Option<f64>,
) -> Option<(usize, &Modifier)> {
    if dyntopo_edge_length.is_some() {
        return None;
    }
    let (index, modifier) = modifiers.iter().enumerate().find(|(_, modifier)| {
        modifier.is_active(crate::eval::EvaluationPurpose::Preview)
            && modifier.modifier_type == "multires"
    })?;
    (modifier
        .params
        .get("sculpt_levels")
        .and_then(Value::as_u64)
        .is_some_and(|levels| levels > 0)
        && !modifier
            .params
            .get("use_sculpt_base_mesh")
            .and_then(Value::as_bool)
            .unwrap_or(false))
    .then_some((index, modifier))
}

fn same_mesh_topology(left: &Mesh, right: &Mesh) -> bool {
    left.vertices
        .iter()
        .map(|vertex| vertex.id)
        .eq(right.vertices.iter().map(|vertex| vertex.id))
        && left
            .edges
            .iter()
            .map(|edge| (edge.id, edge.vertices))
            .eq(right.edges.iter().map(|edge| (edge.id, edge.vertices)))
        && left
            .faces
            .iter()
            .map(|face| (face.id, &face.vertices))
            .eq(right.faces.iter().map(|face| (face.id, &face.vertices)))
}

fn multires_deformed_base(
    engine: &Engine<'_>,
    node_id: &Id,
    source: &Mesh,
    multires_index: usize,
) -> Result<Option<Mesh>> {
    let node = engine
        .doc
        .nodes
        .get(node_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "sculpt node was not found"))?;
    let preceding = node.modifiers.get(..multires_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "multires modifier index is out of range",
        )
    })?;
    let keep = preceding
        .iter()
        .enumerate()
        .find(|(_, modifier)| {
            modifier.is_active(crate::eval::EvaluationPurpose::Preview)
                && !super::modifier::only_deform(&modifier.modifier_type)
        })
        .map_or(preceding.len(), |(index, _)| index);
    if !preceding[..keep]
        .iter()
        .any(|modifier| modifier.is_active(crate::eval::EvaluationPurpose::Preview))
    {
        return Ok(None);
    }

    let frame = engine.evaluation_frame;
    let driver_values = crate::eval::rig::evaluate_drivers(engine.doc, frame)?;
    let node_ids = BTreeSet::from([node_id.clone()]);
    let mut snapshot = crate::eval::Snapshot::evaluate_nodes_with_modifier_prefix_with_cache(
        engine.doc,
        &crate::eval::EvaluationContext {
            frame: Some(frame),
            purpose: crate::eval::EvaluationPurpose::Preview,
            ..crate::eval::EvaluationContext::default()
        },
        engine.asset_root,
        engine.cache_directory,
        &node_ids,
        node_id,
        keep,
        &driver_values,
    )?;
    let deformed = snapshot.meshes.remove(node_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "bounded modifier prefix snapshot omitted the sculpt target mesh",
            json!({"node_id":node_id}),
        )
    })?;
    if !same_mesh_topology(source, &deformed) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "OnlyDeform modifier changed multires base topology",
        ));
    }
    Ok(Some(deformed))
}

/// Whether `mesh` stores per-face-corner data (a UV layer, or a corner-domain attribute such as a
/// corner colour), which dynamic topology cannot keep.
fn has_face_corner_data(mesh: &Mesh) -> bool {
    !crate::geom::uv_layers::read(mesh).layers.is_empty()
        || mesh.attributes.iter().any(|(name, attribute)| {
            name != crate::geom::uv_layers::ATTRIBUTE
                && attribute.get("domain").and_then(Value::as_str) == Some("corner")
        })
}

fn apply_dyntopo(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "scope",
            "samples",
            "falloff",
            "symmetry",
            "edge_length",
        ],
        &["target", "samples", "edge_length"],
    )?;
    let edge_length = super::finite_number(
        engine,
        Some(operation.get("edge_length").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "edge_length is required",
                "/edge_length",
            )
        })?),
        "/edge_length",
        super::FiniteNumberMessage::Value,
    )?;
    if edge_length <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "edge_length must be positive",
            "/edge_length",
        ));
    }
    let mut stroke_operation = operation.clone();
    stroke_operation.insert("brush".to_owned(), Value::String("draw".to_owned()));
    let stroke = parse_stroke(engine, &stroke_operation)?;
    let targets = prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (node_id, data_id) in targets {
        let world = node_world_matrix(engine.doc, &node_id)?;
        let inverse = world.inverse();
        if !inverse.is_finite() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "sculpt target has a singular world transform",
                "/target",
            ));
        }
        let before = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|data| data.mesh.as_ref())
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target data has no mesh",
                    "/target",
                )
            })?;
        if has_multires_detail(engine, &node_id, &before) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "dynamic topology is unavailable on a mesh with Multires detail",
                "/dyntopo",
            ));
        }
        if has_face_corner_data(&before) {
            // Blender's dynamic topology warns this at enable time (`check_attribute_warning`,
            // `sculpt_dyntopo.cc`): its rebuilt faces take default loop data.
            engine.warnings.push(PotError::with_details(
                ErrorCode::UnrepresentableFeature,
                "Dyntopo will not preserve face sets, colors, UVs, or other attributes",
                json!({
                    "feature_id": "sculpt.dyntopo.face_data_lost",
                    "operation_index": engine.operation_index,
                    "pointer": "/dyntopo",
                    "data_id": data_id.as_str(),
                }),
            ));
        }
        let mut world_mesh = before;
        for vertex in &mut world_mesh.vertices {
            vertex.co = world.transform_point3(vertex.co);
        }
        if sculpt_geom::dyntopo_under_brush(&mut world_mesh, &stroke, edge_length)? {
            for vertex in &mut world_mesh.vertices {
                vertex.co = inverse.transform_point3(vertex.co);
            }
            let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
                PotError::new(ErrorCode::TargetNotFound, "mesh data block disappeared")
            })?;
            block.mesh = Some(world_mesh);
            block.descriptor = None;
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            changed = true;
        }
    }
    Ok(changed)
}

fn apply_mask(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "action", "value", "elements"],
        &["target", "action"],
    )?;
    let action = string_field(engine, operation, "action")?;
    if !["set", "clear", "invert"].contains(&action.as_str()) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "sculpt.mask action must be set, clear, or invert",
            "/action",
        ));
    }
    let value = match operation.get("value") {
        Some(value) => super::finite_number(
            engine,
            Some(value),
            "/value",
            super::FiniteNumberMessage::Value,
        )?,
        None if action == "set" => 1.0,
        None => 0.0,
    };
    if !(0.0..=1.0).contains(&value) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mask value must be between zero and one",
            "/value",
        ));
    }
    let selection = parse_vertex_selection(engine, operation.get("elements"))?;
    let targets = prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (_, data_id) in targets {
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "mesh data block was not found")
        })?;
        let mesh = block
            .mesh
            .as_mut()
            .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "target data has no mesh"))?;
        let selected = selected_vertices(mesh, selection.as_deref())?;
        let values = attribute_values(mesh, "sculpt.mask", "vertices")?;
        let values = values
            .as_object_mut()
            .ok_or_else(|| PotError::invalid_argument("mask values must be an object"))?;
        let mut data_changed = false;
        for id in selected {
            let key = format!("v{id}");
            let previous = values.get(&key).and_then(Value::as_f64).unwrap_or(0.0);
            let next = match action.as_str() {
                "set" => value,
                "clear" => 0.0,
                "invert" => 1.0 - previous,
                _ => previous,
            };
            if (previous - next).abs() > f64::EPSILON {
                values.insert(key, json!(next));
                data_changed = true;
            }
        }
        if data_changed {
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn apply_face_set(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "elements", "set_id"],
        &["target", "elements", "set_id"],
    )?;
    let set_id = operation
        .get("set_id")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set_id must be a positive integer",
                "/set_id",
            )
        })?;
    let selection = operation.get("elements").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "face_set requires face elements",
            "/elements",
        )
    })?;
    let face_ids = parse_element_ids(engine, selection, "face")?;
    let targets = prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (_, data_id) in targets {
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "mesh data block was not found")
        })?;
        let mesh = block
            .mesh
            .as_mut()
            .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "target data has no mesh"))?;
        if face_ids
            .iter()
            .any(|face_id| !mesh.faces.iter().any(|face| face.id == *face_id))
        {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "face selection includes a missing element ID",
            ));
        }
        let values = attribute_values(mesh, "sculpt.face_set", "faces")?;
        let values = values
            .as_object_mut()
            .ok_or_else(|| PotError::invalid_argument("face-set values must be an object"))?;
        let mut data_changed = false;
        for face_id in &face_ids {
            let key = format!("f{face_id}");
            if values.get(&key) != Some(&json!(set_id)) {
                values.insert(key, json!(set_id));
                data_changed = true;
            }
        }
        if data_changed {
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn apply_remesh(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "scope", "voxel_size"],
        &["target", "voxel_size"],
    )?;
    let voxel_size = super::finite_number(
        engine,
        Some(operation.get("voxel_size").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "voxel_size is required",
                "/voxel_size",
            )
        })?),
        "/voxel_size",
        super::FiniteNumberMessage::Value,
    )?;
    if voxel_size <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "voxel_size must be positive",
            "/voxel_size",
        ));
    }
    let targets = prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (_, data_id) in targets {
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "mesh data block was not found")
        })?;
        let mesh = block
            .mesh
            .as_ref()
            .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "target data has no mesh"))?;
        let remeshed = crate::geom::remesh::remesh(mesh, voxel_size)?;
        if block.mesh.as_ref() != Some(&remeshed) {
            block.mesh = Some(remeshed);
            block.descriptor = None;
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            changed = true;
        }
    }
    Ok(changed)
}

fn apply_multires(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "id", "levels", "render_levels"],
        &["target", "id", "levels"],
    )?;
    let targets = resolve_node_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(ErrorCode::InvalidOperation, "target is required", "/target")
        })?,
        true,
    )?;
    let modifier_id = super::read_id(engine, operation, "id")?;
    let levels = operation
        .get("levels")
        .and_then(Value::as_u64)
        .filter(|value| (1..=MAX_MULTIRES_LEVELS).contains(value))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "levels must be between 1 and 11",
                "/levels",
            )
        })?;
    let render_levels = match operation.get("render_levels") {
        None => levels,
        Some(value) => value.as_u64().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "render_levels must be an integer",
                "/render_levels",
            )
        })?,
    };
    if render_levels > MAX_MULTIRES_LEVELS {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "render_levels must not exceed 11",
            "/render_levels",
        ));
    }
    let mut changed = false;
    for node_id in targets {
        if engine.doc.nodes.get(&node_id).is_some_and(|node| {
            node.modifiers
                .iter()
                .any(|modifier| modifier.id == modifier_id)
        }) {
            return Err(engine.error(
                ErrorCode::IdExists,
                format!("modifier `{modifier_id}` already exists"),
                "/id",
            ));
        }
        let data_id = engine
            .doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone());
        let mut total_levels = levels;
        let mut data_changed = false;
        if let Some(data_id) = &data_id
            && let Some(data) = engine.doc.data_blocks.get_mut(data_id)
            && let Some(mesh) = data.mesh.as_mut()
        {
            if let Some(record) = mesh.attributes.get("multires_displacement") {
                total_levels = record
                    .get("total_levels")
                    .and_then(Value::as_u64)
                    .filter(|value| (1..=13).contains(value))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InvalidOperation,
                            "existing multires_displacement has an invalid total_levels",
                        )
                    })?;
            } else {
                mesh.attributes.insert(
                    "multires_displacement".to_owned(),
                    json!({"total_levels": levels}),
                );
                data_changed = true;
            }
        }
        if data_changed && let Some(data_id) = &data_id {
            engine.mark("data_blocks", data_id, ChangeKind::Updated);
        }
        let levels = levels.min(total_levels);
        let render_levels = render_levels.min(total_levels);
        let node = engine
            .doc
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "node was not found"))?;
        let index = super::modifier::insertion_index(&node.modifiers, "multires");
        node.modifiers.insert(
            index,
            Modifier {
                id: modifier_id.clone(),
                modifier_type: "multires".to_owned(),
                name: "Multires".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: Map::from_iter([
                    ("levels".to_owned(), json!(levels)),
                    ("sculpt_levels".to_owned(), json!(levels)),
                    ("render_levels".to_owned(), json!(render_levels)),
                    ("total_levels".to_owned(), json!(total_levels)),
                ]),
                binding_data: None,
                runtime: crate::model::ModifierRuntime::default(),
            },
        );
        engine.mark("nodes", &node_id, ChangeKind::Updated);
        changed = true;
    }
    Ok(changed)
}

pub(super) fn parse_stroke(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<sculpt_geom::Stroke> {
    let brush = match string_field(engine, operation, "brush")?.as_str() {
        "draw" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Draw),
        "clay_strips" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::ClayStrips),
        "inflate" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Inflate),
        "grab" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Grab),
        "smooth" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Smooth),
        "flatten" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Flatten),
        "pinch" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Pinch),
        "crease" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Crease),
        "layer" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Layer),
        "snake_hook" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::SnakeHook),
        "thumb" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Thumb),
        "rotate" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Rotate),
        "nudge" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Nudge),
        "blob" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Blob),
        "scrape" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Scrape),
        "fill" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Fill),
        "draw_sharp" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::DrawSharp),
        "elastic_deform" => sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Elastic),
        "pose_lite" => sculpt_geom::Brush::Vertex(sculpt_geom::VertexBrush::PoseLite),
        "boundary_lite" => sculpt_geom::Brush::Vertex(sculpt_geom::VertexBrush::BoundaryLite),
        other => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown sculpt brush `{other}`"),
                "/brush",
            ));
        }
    };
    if brush == sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::Elastic)
        && operation.contains_key("falloff")
    {
        // Blender's Elastic Grab has no falloff: its Kelvinlet fade is the only spatial factor.
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elastic_deform has no falloff; Blender's Elastic Grab uses its Kelvinlet fade",
            "/falloff",
        ));
    }
    if operation
        .get("falloff")
        .is_some_and(|value| !value.is_string())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "falloff must be a string",
            "/falloff",
        ));
    }
    // Blender 5.2.2's essentials presets: Clay Strips uses SMOOTHER, Draw Sharp uses POW4, and the
    // other dab brushes use SMOOTH. Potter's stroke carries no brush curve, so the default follows
    // the brush.
    let default_falloff = match brush {
        sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::ClayStrips) => "smoother",
        sculpt_geom::Brush::Dab(sculpt_geom::DabBrush::DrawSharp) => "pow4",
        _ => "smooth",
    };
    let falloff = match operation
        .get("falloff")
        .and_then(Value::as_str)
        .unwrap_or(default_falloff)
    {
        "smooth" => sculpt_geom::Falloff::Smooth,
        "smoother" => sculpt_geom::Falloff::Smoother,
        "pow4" => sculpt_geom::Falloff::Pow4,
        "sphere" => sculpt_geom::Falloff::Sphere,
        "root" => sculpt_geom::Falloff::Root,
        "sharp" => sculpt_geom::Falloff::Sharp,
        "linear" => sculpt_geom::Falloff::Linear,
        "constant" => sculpt_geom::Falloff::Constant,
        other => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown falloff `{other}`"),
                "/falloff",
            ));
        }
    };
    let values = operation
        .get("samples")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "samples must be an array",
                "/samples",
            )
        })?;
    let mut samples = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let sample = value.as_object().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "stroke sample must be an object",
                &format!("/samples/{index}"),
            )
        })?;
        if sample.keys().any(|key| {
            !["position", "pressure", "radius", "strength", "time"].contains(&key.as_str())
        }) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "stroke sample contains an unknown field",
                &format!("/samples/{index}"),
            ));
        }
        let position = vector3(
            engine,
            sample.get("position"),
            &format!("/samples/{index}/position"),
        )?;
        let number = |name| {
            sample
                .get(name)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        format!("sample `{name}` is required"),
                        &format!("/samples/{index}/{name}"),
                    )
                })
                .and_then(|value| {
                    super::finite_number(
                        engine,
                        Some(value),
                        &format!("/samples/{index}/{name}"),
                        super::FiniteNumberMessage::Value,
                    )
                })
        };
        samples.push(sculpt_geom::StrokeSample {
            position,
            pressure: number("pressure")?,
            radius: number("radius")?,
            strength: number("strength")?,
            time: number("time")?,
        });
    }
    let symmetry = match operation.get("symmetry") {
        None => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map(str::to_owned).ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "symmetry axes must be strings",
                        "/symmetry",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "symmetry must be an array",
                "/symmetry",
            ));
        }
    };
    let delta = operation
        .get("delta")
        .map(|value| vector3(engine, Some(value), "/delta"))
        .transpose()?;
    let view = operation
        .get("view")
        .map(|value| vector3(engine, Some(value), "/view"))
        .transpose()?;
    Ok(sculpt_geom::Stroke {
        brush,
        samples,
        falloff,
        symmetry,
        delta,
        view,
    })
}

fn parse_dyntopo_edge_length(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<Option<f64>> {
    let Some(value) = operation.get("dyntopo") else {
        return Ok(None);
    };
    let options = value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "dyntopo must be an object",
            "/dyntopo",
        )
    })?;
    if options.keys().any(|key| key != "edge_length") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "dyntopo contains an unknown field",
            "/dyntopo",
        ));
    }
    let edge_length = super::finite_number(
        engine,
        Some(options.get("edge_length").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "dyntopo edge_length is required",
                "/dyntopo/edge_length",
            )
        })?),
        "/dyntopo/edge_length",
        super::FiniteNumberMessage::Value,
    )?;
    if edge_length <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "dyntopo edge_length must be positive",
            "/dyntopo/edge_length",
        ));
    }
    Ok(Some(edge_length))
}

pub(super) fn prepare_mesh_targets(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<Vec<(Id, Id)>> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(ErrorCode::InvalidOperation, "target is required", "/target")
    })?;
    let nodes = resolve_node_targets(engine, target, true)?;
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        false,
    )?;
    let mut pairs = Vec::with_capacity(nodes.len());
    for node_id in &nodes {
        let data_id = engine
            .doc
            .nodes
            .get(node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{node_id}` has no mesh data"),
                    "/target",
                )
            })?;
        if super::data_user_count(engine, &data_id) > 1 && scope.is_none() {
            return Err(engine.error(
                ErrorCode::SharedDataRequiresScope,
                "shared geometry edits require scope",
                "/scope",
            ));
        }
        pairs.push((node_id.clone(), data_id));
    }
    if scope == Some("single_user") {
        for (node_id, data_id) in &mut pairs {
            if super::data_user_count(engine, data_id) <= 1 {
                continue;
            }
            let block = engine
                .doc
                .data_blocks
                .get(data_id)
                .cloned()
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` not found"),
                        "/target",
                    )
                })?;
            let clone_id = super::unique_data_id(
                engine,
                node_id,
                "_sculpt_single",
                super::DataIdCollisionHandling::TruncateCandidate,
                "data-block ID suffix exhausted",
            )?;
            engine.doc.data_blocks.insert(clone_id.clone(), block);
            if let Some(node) = engine.doc.nodes.get_mut(node_id) {
                node.data = Some(clone_id.clone());
            }
            engine.mark("data_blocks", &clone_id, ChangeKind::Created);
            engine.mark("nodes", node_id, ChangeKind::Updated);
            engine.map_id("data_blocks", data_id, &clone_id);
            *data_id = clone_id;
        }
    }
    if scope == Some("shared") {
        let mut seen = BTreeSet::new();
        pairs.retain(|(_, data_id)| seen.insert(data_id.clone()));
    }
    Ok(pairs)
}

pub(super) fn node_world_matrix(doc: &SceneDoc, node_id: &Id) -> Result<DMat4> {
    let mut chain = Vec::new();
    let mut cursor = Some(node_id.clone());
    let mut seen = BTreeSet::new();
    while let Some(id) = cursor.take() {
        if !seen.insert(id.clone()) {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "node parent cycle prevents world sculpt transform",
            ));
        }
        let node = doc.nodes.get(&id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, format!("node `{id}` not found"))
        })?;
        chain.push((node.transform.matrix(), node.parent_inverse));
        cursor.clone_from(&node.parent);
    }
    let mut world = DMat4::IDENTITY;
    for (local, parent_inverse) in chain.into_iter().rev() {
        let inverse =
            parent_inverse.map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix));
        world = world * inverse * local;
    }
    Ok(world)
}

fn parse_vertex_selection(
    engine: &Engine<'_>,
    selection: Option<&Value>,
) -> Result<Option<Vec<u32>>> {
    selection
        .map(|value| parse_element_ids(engine, value, "vertex"))
        .transpose()
}

fn parse_element_ids(
    engine: &Engine<'_>,
    value: &Value,
    expected_domain: &str,
) -> Result<Vec<u32>> {
    let object = value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "elements must be an object",
            "/elements",
        )
    })?;
    if object
        .keys()
        .any(|key| !["domain", "ids"].contains(&key.as_str()))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements contains an unknown field",
            "/elements",
        ));
    }
    if object.get("domain").and_then(Value::as_str) != Some(expected_domain) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("elements.domain must be {expected_domain}"),
            "/elements/domain",
        ));
    }
    let ids = object.get("ids").and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "elements.ids must be an array",
            "/elements/ids",
        )
    })?;
    if ids.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements.ids must not be empty",
            "/elements/ids",
        ));
    }
    let parsed = ids
        .iter()
        .map(|value| {
            let text = value.as_str().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "element IDs must be strings",
                    "/elements/ids",
                )
            })?;
            text.strip_prefix(if expected_domain == "face" { 'f' } else { 'v' })
                .and_then(|value| value.parse::<u32>().ok())
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "element ID does not match its domain",
                        "/elements/ids",
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut seen = BTreeSet::new();
    if parsed.iter().any(|id| !seen.insert(*id)) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "elements.ids must not contain duplicates",
            "/elements/ids",
        ));
    }
    Ok(parsed)
}

fn selected_vertices(mesh: &Mesh, selection: Option<&[u32]>) -> Result<Vec<u32>> {
    let selected = selection.map_or_else(
        || mesh.vertices.iter().map(|vertex| vertex.id).collect(),
        ToOwned::to_owned,
    );
    if selected
        .iter()
        .any(|id| !mesh.vertices.iter().any(|vertex| vertex.id == *id))
    {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "vertex selection includes a missing element ID",
        ));
    }
    Ok(selected)
}

pub(super) fn attribute_values<'a>(
    mesh: &'a mut Mesh,
    name: &str,
    domain: &str,
) -> Result<&'a mut Value> {
    let attribute = mesh
        .attributes
        .entry(name.to_owned())
        .or_insert_with(|| json!({"domain":domain,"values":{}}));
    if attribute.get("domain").and_then(Value::as_str) != Some(domain) {
        return Err(PotError::invalid_argument(format!(
            "attribute `{name}` has a different domain"
        )));
    }
    let object = attribute.as_object_mut().ok_or_else(|| {
        PotError::invalid_argument(format!("attribute `{name}` must be an object"))
    })?;
    let values = object.entry("values").or_insert_with(|| json!({}));
    if !values.is_object() {
        return Err(PotError::invalid_argument(format!(
            "attribute `{name}` values must be an object"
        )));
    }
    Ok(values)
}

fn string_field(engine: &Engine<'_>, object: &Map<String, Value>, field: &str) -> Result<String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a string"),
                &format!("/{field}"),
            )
        })
}

fn vector3(engine: &Engine<'_>, value: Option<&Value>, pointer: &str) -> Result<DVec3> {
    let values = value
        .and_then(Value::as_array)
        .filter(|values| values.len() == 3)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "value must be an array of three numbers",
                pointer,
            )
        })?;
    let [x, y, z] = [
        super::finite_number(
            engine,
            Some(&values[0]),
            pointer,
            super::FiniteNumberMessage::Value,
        )?,
        super::finite_number(
            engine,
            Some(&values[1]),
            pointer,
            super::FiniteNumberMessage::Value,
        )?,
        super::finite_number(
            engine,
            Some(&values[2]),
            pointer,
            super::FiniteNumberMessage::Value,
        )?,
    ];
    Ok(DVec3::new(x, y, z))
}

#[cfg(test)]
mod multires_selection_tests {
    #![expect(clippy::unwrap_used, reason = "tests")]
    use super::*;

    fn multires_modifier(id: &str, show_viewport: bool) -> Modifier {
        Modifier {
            id: Id::new(id).unwrap(),
            modifier_type: "multires".to_owned(),
            name: "Multires".to_owned(),
            enabled: true,
            show_viewport,
            show_render: true,
            params: Map::from_iter([("sculpt_levels".to_owned(), serde_json::json!(1))]),
            binding_data: None,
            runtime: crate::model::ModifierRuntime::default(),
        }
    }

    #[test]
    fn multires_sculpt_uses_preview_active_slot() {
        let modifiers = [
            multires_modifier("hidden_multires", false),
            multires_modifier("active_multires", true),
        ];
        assert_eq!(
            active_multires_modifier(&modifiers, None).map(|(index, _)| index),
            Some(1)
        );
        let hidden_only = [multires_modifier("hidden_multires", false)];
        assert!(active_multires_modifier(&hidden_only, None).is_none());
    }
}
