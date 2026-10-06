use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::pointcloud::PointCloudData,
    model::DataBlock,
};

use super::{
    ChangeKind, Engine, check_fields, check_set_fields,
    geometry_data::{create_geometry_node, merge_set, read_set, selected, typed_node_data_id},
    operation_pointer,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "pointcloud.create" => create(engine, operation),
        "pointcloud.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid point-cloud operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "points",
            "attributes",
            "next_id",
            "collection",
            "transform",
            "parent",
            "parent_inverse",
            "materials",
            "visible",
            "render_visible",
            "selectable",
        ],
        &["id"],
    )?;
    let mut data =
        selected::<PointCloudData>(engine, operation, &["points", "attributes", "next_id"], "")?;
    if !operation.contains_key("next_id") {
        normalize_next_id(engine, &mut data, 0, "")?;
    }
    validate(engine, &data, "")?;
    create_geometry_node(
        engine,
        operation,
        "pointcloud",
        "_pointcloud",
        DataBlock {
            data_type: "pointcloud".to_owned(),
            mesh: None,
            pointcloud: Some(data),
            ..DataBlock::default()
        },
    )
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = read_set(engine, operation)?;
    check_set_fields(engine, set, &["points", "attributes", "next_id"])?;
    let (_, data_id) = typed_node_data_id(engine, operation, "pointcloud", "pointcloud")?;
    let current = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.pointcloud.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "point-cloud", "target"))?;
    let mut updated: PointCloudData = merge_set(engine, &current, set, "set")?;
    if !set.contains_key("next_id") {
        normalize_next_id(engine, &mut updated, current.next_id, "set")?;
    }
    validate(engine, &updated, "set")?;
    if updated == current {
        return Ok(false);
    }
    let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "point-cloud data block disappeared",
        )
    })?;
    block.pointcloud = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn normalize_next_id(
    engine: &Engine<'_>,
    data: &mut PointCloudData,
    minimum: u32,
    pointer: &str,
) -> Result<()> {
    let Some(maximum) = data.points.iter().map(|point| point.id).max() else {
        data.next_id = minimum;
        return Ok(());
    };
    let next = maximum.checked_add(1).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "point IDs leave no valid next_id",
            &operation_pointer(engine.operation_index, pointer),
        )
    })?;
    data.next_id = minimum.max(next);
    Ok(())
}

fn validate(engine: &Engine<'_>, data: &PointCloudData, field: &str) -> Result<()> {
    if data.attributes.contains_key("point_radius") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "`point_radius` is reserved for point-cloud radii",
            &operation_pointer(engine.operation_index, field),
        ));
    }
    data.validate().map_err(|error| {
        engine.error(
            error.code,
            format!("point-cloud geometry is invalid: {}", error.message),
            &operation_pointer(engine.operation_index, field),
        )
    })
}
