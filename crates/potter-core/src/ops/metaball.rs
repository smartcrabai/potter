use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::metaball::MetaballData,
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
        "metaball.create" => create(engine, operation),
        "metaball.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid metaball operation dispatch",
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
            "elements",
            "resolution",
            "render_resolution",
            "threshold",
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
    let data = selected::<MetaballData>(
        engine,
        operation,
        &["elements", "resolution", "render_resolution", "threshold"],
        "",
    )?;
    validate(engine, &data, "")?;
    create_geometry_node(
        engine,
        operation,
        "metaball",
        "_metaball",
        DataBlock {
            data_type: "metaball".to_owned(),
            mesh: None,
            metaball: Some(data),
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
    let fields = ["elements", "resolution", "render_resolution", "threshold"];
    check_set_fields(engine, set, &fields)?;
    let (_, data_id) = typed_node_data_id(engine, operation, "metaball", "metaball")?;
    let current = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.metaball.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "metaball", "target"))?;
    let updated: MetaballData = merge_set(engine, &current, set, "set")?;
    validate(engine, &updated, "set")?;
    if updated == current {
        return Ok(false);
    }
    let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "metaball data block disappeared")
    })?;
    block.metaball = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn validate(engine: &Engine<'_>, data: &MetaballData, field: &str) -> Result<()> {
    crate::geom::metaball::to_mesh(data)
        .map(|_| ())
        .map_err(|error| {
            engine.error(
                error.code,
                format!("metaball geometry is invalid: {}", error.message),
                &operation_pointer(engine.operation_index, field),
            )
        })
}
