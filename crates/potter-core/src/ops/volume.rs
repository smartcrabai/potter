use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::volume::VolumeData,
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
        "volume.create" => create(engine, operation),
        "volume.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid volume operation dispatch",
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
            "grids",
            "source",
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
    let data = selected::<VolumeData>(engine, operation, &["grids", "source"], "")?;
    validate(engine, &data, "")?;
    create_geometry_node(
        engine,
        operation,
        "volume",
        "_volume",
        DataBlock {
            data_type: "volume".to_owned(),
            mesh: None,
            volume: Some(data),
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
    check_set_fields(engine, set, &["grids", "source"])?;
    let (_, data_id) = typed_node_data_id(engine, operation, "volume", "volume")?;
    let current = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.volume.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "volume", "target"))?;
    let updated: VolumeData = merge_set(engine, &current, set, "set")?;
    validate(engine, &updated, "set")?;
    if updated == current {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "volume data block disappeared")
        })?;
    block.volume = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn validate(engine: &Engine<'_>, data: &VolumeData, field: &str) -> Result<()> {
    crate::geom::volume::validate_data(data).map_err(|error| {
        let mut operation_error = engine.error(
            error.code,
            format!("volume data is invalid: {}", error.message),
            &operation_pointer(engine.operation_index, field),
        );
        if let Some(feature_id) = error.details.get("feature_id")
            && let Some(details) = operation_error.details.as_object_mut()
        {
            details.insert("feature_id".to_owned(), feature_id.clone());
        }
        operation_error
    })
}
