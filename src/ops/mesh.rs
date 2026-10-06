use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::edit,
    model::Id,
};

use super::{ChangeKind, Engine, check_fields, resolve_node_targets};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    if ![
        "mesh.transform_elements",
        "mesh.extrude",
        "mesh.inset",
        "mesh.bevel",
        "mesh.subdivide",
        "mesh.triangulate",
        "mesh.poke",
        "mesh.delete",
        "mesh.dissolve",
        "mesh.weld",
        "mesh.fill",
        "mesh.bisect",
        "mesh.mirror",
        "mesh.flip_normals",
        "mesh.set_attribute",
        "mesh.attribute_create",
        "mesh.attribute_update",
        "mesh.attribute_delete",
        "mesh.auto_smooth",
        "mesh.edge_slide",
        "mesh.vertex_slide",
        "mesh.spin",
        "mesh.screw",
        "mesh.rip",
        "mesh.merge",
        "mesh.shade_smooth",
        "mesh.shade_flat",
        "mesh.mark_sharp_by_angle",
        "mesh.set_custom_normals",
        "mesh.bridge",
        "mesh.split",
        "mesh.knife",
        "mesh.remesh",
        "mesh.symmetrize",
    ]
    .contains(&name)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported mesh operation `{name}`"),
            "/op",
        ));
    }
    check_fields(engine, operation, allowed_fields(name), &["target"])?;
    if name == "mesh.bevel"
        && operation
            .get("segments")
            .is_some_and(|segments| segments.as_u64() != Some(1))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "mesh.bevel supports exactly one segment",
            "/segments",
        ));
    }
    let (target, target_elements) = split_target(engine, operation.get("target"))?;
    let targets = resolve_node_targets(engine, &target, true)?;
    let elements = match (operation.get("elements"), target_elements.as_ref()) {
        (Some(_), Some(_)) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "elements must be specified either at top level or in target, not both",
                "/elements",
            ));
        }
        (Some(elements), None) | (None, Some(elements)) => Some(elements.clone()),
        (None, None) => None,
    };
    let scope = super::parse_scope(
        engine,
        operation.get("scope"),
        "/scope",
        "scope must be a string",
        "scope must be shared or single_user",
        true,
    )?;
    let mut node_data = Vec::with_capacity(targets.len());
    for target_id in &targets {
        let data_id = engine
            .doc
            .nodes
            .get(target_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("node `{target_id}` has no geometry data"),
                    "/target",
                )
            })?;
        node_data.push((target_id.clone(), data_id));
    }
    let has_shared_data = node_data
        .iter()
        .any(|(_, data_id)| super::data_user_count(engine, data_id) > 1);
    if has_shared_data && scope.is_none() {
        return Err(engine.error(
            ErrorCode::SharedDataRequiresScope,
            "shared geometry edits require scope",
            "/scope",
        ));
    }
    if scope == Some("single_user") {
        for (target_id, data_id) in &node_data {
            if super::data_user_count(engine, data_id) <= 1 {
                continue;
            }
            let cloned = engine
                .doc
                .data_blocks
                .get(data_id)
                .cloned()
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        format!("data block `{data_id}` was not found"),
                        "/target",
                    )
                })?;
            let new_id = super::unique_data_id(
                engine,
                target_id,
                "_mesh_single",
                super::DataIdCollisionHandling::ReserveSuffix,
                "data-block ID suffix exhausted",
            )?;
            engine.doc.data_blocks.insert(new_id.clone(), cloned);
            if let Some(node) = engine.doc.nodes.get_mut(target_id) {
                node.data = Some(new_id.clone());
            }
            engine.mark("data_blocks", &new_id, ChangeKind::Created);
            engine.mark("nodes", target_id, ChangeKind::Updated);
            engine.map_id("data_blocks", data_id, &new_id);
        }
    }
    let mut args = operation.clone();
    args.remove("op");
    args.remove("target");
    args.remove("scope");
    if let Some(threshold) = args.remove("merge_threshold") {
        if args.contains_key("threshold") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "provide either threshold or merge_threshold, not both",
                "/threshold",
            ));
        }
        args.insert("threshold".to_owned(), threshold);
    }
    if let Some(attribute) = args.remove("attribute") {
        if args.contains_key("name") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "provide either name or attribute, not both",
                "/name",
            ));
        }
        args.insert("name".to_owned(), attribute);
    }
    if let Some(material_index) = args.remove("material_index") {
        args.entry("name".to_owned())
            .or_insert_with(|| Value::String("material_index".to_owned()));
        args.entry("value".to_owned()).or_insert(material_index);
    }
    if name == "mesh.inset" {
        let width = args.remove("width");
        let distance = args.remove("distance");
        if width.is_some() && distance.is_some() || args.contains_key("amount") && width.is_some() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "provide only one inset amount, width, or distance",
                "/amount",
            ));
        }
        if let Some(amount) = width.or(distance) {
            args.insert("amount".to_owned(), amount);
        }
    } else if name == "mesh.weld" {
        if let Some(distance) = args.remove("distance") {
            if args.contains_key("threshold") {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "provide either threshold or distance, not both",
                    "/threshold",
                ));
            }
            args.insert("threshold".to_owned(), distance);
        }
    } else if name == "mesh.bevel" {
        drop(args.remove("segments"));
    }
    if let Some(plane) = args.remove("plane") {
        let Some(plane) = plane.as_object() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "plane must be an object",
                "/plane",
            ));
        };
        for field in ["point", "normal"] {
            if let Some(value) = plane.get(field) {
                args.entry(field.to_owned())
                    .or_insert_with(|| value.clone());
            }
        }
    }
    if let Some(elements) = elements {
        args.insert("elements".to_owned(), elements);
    }
    let mut changed = false;
    let mut edited_data = Vec::<Id>::new();
    for target_id in targets {
        let data_id = engine
            .doc
            .nodes
            .get(&target_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                PotError::new(ErrorCode::InvalidOperation, "node has no geometry data")
            })?;
        if scope == Some("shared") && edited_data.contains(&data_id) {
            continue;
        }
        let (data_changed, had_descriptor, mapping) = {
            let data = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::TargetNotFound,
                    format!("data block `{data_id}` was not found"),
                    json!({"data_id":data_id}),
                )
            })?;
            let mesh = data.mesh.as_mut().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::InvalidOperation,
                    format!("data block `{data_id}` has no mesh payload"),
                    json!({"data_id":data_id}),
                )
            })?;
            let had_descriptor = data.descriptor.is_some();
            let before = mesh.clone();
            let mapping = edit::apply(mesh, name, &Value::Object(args.clone()))?;
            let changed = *mesh != before;
            if changed {
                data.descriptor = None;
            }
            (changed, had_descriptor, mapping)
        };
        if data_changed {
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            super::clear_primitive_metadata(engine, &data_id);
            if had_descriptor {
                engine
                    .id_mappings
                    .entry("mesh_descriptor_dropped".to_owned())
                    .or_insert_with(|| json!({}))[data_id.as_str()] = Value::Bool(true);
            }
            if !mapping.is_null() {
                engine
                    .id_mappings
                    .entry("mesh_elements".to_owned())
                    .or_insert_with(|| json!({}))[data_id.as_str()] = mapping;
            }
            changed = true;
        }
        edited_data.push(data_id);
    }
    Ok(changed)
}

fn split_target(engine: &Engine<'_>, value: Option<&Value>) -> Result<(Value, Option<Value>)> {
    let Some(Value::Object(target)) = value else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "target must be an object",
            "/target",
        ));
    };
    let mut target = target.clone();
    let elements = target.remove("elements");
    Ok((Value::Object(target), elements))
}

fn allowed_fields(name: &str) -> &'static [&'static str] {
    match name {
        "mesh.transform_elements" => &[
            "op",
            "target",
            "scope",
            "elements",
            "translation",
            "rotation",
            "scale",
            "pivot",
            "proportional",
        ],
        "mesh.extrude" => &[
            "op",
            "target",
            "scope",
            "elements",
            "distance",
            "offset",
            "individual",
        ],
        "mesh.inset" => &[
            "op", "target", "scope", "elements", "amount", "width", "distance",
        ],
        "mesh.edge_slide" | "mesh.vertex_slide" => &["op", "target", "scope", "elements", "factor"],
        "mesh.spin" => &[
            "op", "target", "scope", "elements", "axis", "angle", "center", "steps",
        ],
        "mesh.screw" => &[
            "op", "target", "scope", "elements", "axis", "angle", "center", "distance", "steps",
        ],
        "mesh.merge" => &["op", "target", "scope", "elements", "mode", "cursor"],
        "mesh.rip" | "mesh.shade_smooth" | "mesh.shade_flat" | "mesh.triangulate" | "mesh.poke"
        | "mesh.delete" | "mesh.dissolve" | "mesh.fill" | "mesh.flip_normals" | "mesh.bridge"
        | "mesh.split" => &["op", "target", "scope", "elements"],
        "mesh.auto_smooth" | "mesh.mark_sharp_by_angle" => {
            &["op", "target", "scope", "elements", "angle"]
        }
        "mesh.set_custom_normals" => &["op", "target", "scope", "elements", "normals"],
        "mesh.bevel" => &[
            "op", "target", "scope", "elements", "width", "amount", "segments",
        ],
        "mesh.subdivide" | "mesh.loop_cut" => &["op", "target", "scope", "elements", "cuts"],
        "mesh.weld" => &["op", "target", "scope", "elements", "threshold", "distance"],
        "mesh.bisect" => &[
            "op",
            "target",
            "scope",
            "elements",
            "plane",
            "point",
            "normal",
            "threshold",
            "clear_side",
        ],
        "mesh.mirror" => &[
            "op",
            "target",
            "scope",
            "elements",
            "axis",
            "merge",
            "merge_threshold",
            "threshold",
            "origin",
        ],
        "mesh.set_attribute" => &[
            "op",
            "target",
            "scope",
            "elements",
            "attribute",
            "name",
            "value",
            "material_index",
        ],
        "mesh.attribute_create" => &["op", "target", "scope", "name", "domain", "type", "default"],
        "mesh.attribute_update" => &["op", "target", "scope", "elements", "name", "value"],
        "mesh.attribute_delete" => &["op", "target", "scope", "elements", "name"],
        "mesh.knife" => &[
            "op",
            "target",
            "scope",
            "elements",
            "plane",
            "point",
            "normal",
            "threshold",
        ],
        "mesh.remesh" => &["op", "target", "scope", "voxel_size"],
        "mesh.symmetrize" => &[
            "op",
            "target",
            "scope",
            "elements",
            "axis",
            "direction",
            "origin",
            "threshold",
        ],
        _ => &[],
    }
}
