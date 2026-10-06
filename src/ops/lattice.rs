use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{self, Mesh, lattice::LatticeData},
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
        "lattice.create" => create(engine, operation),
        "lattice.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid lattice operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    let fields = [
        "op",
        "id",
        "name",
        "data_id",
        "points_u",
        "points_v",
        "points_w",
        "interpolation",
        "points",
        "domain",
        "collection",
        "transform",
        "parent",
        "parent_inverse",
        "materials",
        "visible",
        "render_visible",
        "selectable",
    ];
    check_fields(engine, operation, &fields, &["id"])?;
    let data = selected::<LatticeData>(
        engine,
        operation,
        &[
            "points_u",
            "points_v",
            "points_w",
            "interpolation",
            "points",
            "domain",
        ],
        "",
    )?;
    validate(engine, &data, "")?;
    create_geometry_node(
        engine,
        operation,
        "lattice",
        "_lattice",
        DataBlock {
            data_type: "lattice".to_owned(),
            mesh: None,
            lattice: Some(data),
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
    check_set_fields(
        engine,
        set,
        &[
            "points_u",
            "points_v",
            "points_w",
            "interpolation",
            "points",
            "domain",
        ],
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "lattice", "lattice")?;
    let current = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.lattice.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "lattice", "target"))?;
    let updated: LatticeData = merge_set(engine, &current, set, "set")?;
    validate(engine, &updated, "set")?;
    if updated == current {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "lattice data block disappeared")
        })?;
    block.lattice = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn validate(engine: &Engine<'_>, data: &LatticeData, field: &str) -> Result<()> {
    geom::lattice::deform_mesh(&Mesh::default(), data, None)
        .map(|_| ())
        .map_err(|error| {
            engine.error(
                error.code,
                format!("lattice geometry is invalid: {}", error.message),
                &operation_pointer(engine.operation_index, field),
            )
        })
}
