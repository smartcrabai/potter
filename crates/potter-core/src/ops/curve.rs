use std::collections::BTreeSet;

use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom,
    model::{
        CurveData, CurveHandleType, CurvePoint, CurveSplineType, DataBlock, HairCurve,
        HairCurveAttribute, HairCurvesData, Id, SurfaceData, TextObjectData,
    },
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_id, read_string,
    resolve_node_targets,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "curve.create" => create_curve(engine, operation),
        "curve.update" => update_curve(engine, operation),
        "curve.point_add" => add_curve_point(engine, operation),
        "curve.point_update" => update_curve_point(engine, operation),
        "curve.point_delete" => delete_curve_point(engine, operation),
        "curve.convert" => convert_curve(engine, operation),
        "surface.create" => create_surface(engine, operation),
        "text_object.create" => create_text_object(engine, operation),
        "text_object.update" => update_text_object(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid curve/surface/text operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "splines",
            "resolution_u",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "taper_radius_mode",
            "use_map_taper",
            "fill_mode",
            "fill_solver",
            "fill_rule",
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
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_curve")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let curve = decode_selected::<CurveData>(
        engine,
        operation,
        &[
            "splines",
            "resolution_u",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "taper_radius_mode",
            "use_map_taper",
            "fill_mode",
            "fill_solver",
            "fill_rule",
        ],
        "",
    )?;
    validate_curve(engine, &curve, "")?;
    validate_curve_taper(engine, &curve, &data_id, "taper")?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "curve".to_owned(),
            mesh: None,
            curve: Some(curve),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "curve", &id, &data_id)
}

fn update_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "splines",
            "resolution_u",
            "dimensions",
            "bevel_depth",
            "bevel_resolution",
            "extrude",
            "taper",
            "taper_radius_mode",
            "use_map_taper",
            "fill_mode",
            "fill_solver",
            "fill_rule",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated: Value = serialize(&curve)?;
    merge_values(&mut updated, set);
    let updated: CurveData = decode_value(engine, updated, "set")?;
    validate_curve(engine, &updated, "set")?;
    validate_curve_taper(engine, &updated, &data_id, "set/taper")?;
    if updated == curve {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    block.curve = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn add_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "index", "point"],
        &["target", "spline_index", "point"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point = decode_value::<CurvePoint>(
        engine,
        operation.get("point").cloned().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing required field `point`",
                &operation_pointer(engine.operation_index, "point"),
            )
        })?,
        "point",
    )?;
    let index = operation
        .get("index")
        .map(|_| read_index(engine, operation, "index"))
        .transpose()?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let spline = updated.splines.get_mut(spline_index).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "spline_index is out of range",
            &operation_pointer(engine.operation_index, "spline_index"),
        )
    })?;
    let insertion_index = index.unwrap_or(spline.points.len());
    if insertion_index > spline.points.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "index is out of range",
            &operation_pointer(engine.operation_index, "index"),
        ));
    }
    spline.points.insert(insertion_index, point);
    validate_curve(engine, &updated, "point")?;
    store_curve(engine, &data_id, updated)
}

fn update_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "point_index", "set"],
        &["target", "spline_index", "point_index", "set"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point_index = read_index(engine, operation, "point_index")?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "co",
            "handle_left",
            "handle_right",
            "handle_type",
            "handle_left_type",
            "handle_right_type",
            "weight",
            "radius",
            "tilt",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let point = updated
        .splines
        .get_mut(spline_index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "spline_index is out of range",
                &operation_pointer(engine.operation_index, "spline_index"),
            )
        })?
        .points
        .get_mut(point_index)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "point_index is out of range",
                &operation_pointer(engine.operation_index, "point_index"),
            )
        })?;
    let mut point_value = serialize(point)?;
    merge_values(&mut point_value, set);
    if set.contains_key("handle_type") {
        let fields = point_value.as_object_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "serialized curve point is not an object",
            )
        })?;
        for side in ["handle_left_type", "handle_right_type"] {
            if !set.contains_key(side) {
                fields.remove(side);
            }
        }
    }
    let updated_point: CurvePoint = decode_value(engine, point_value, "set")?;
    let changed = *point != updated_point;
    if !changed {
        return Ok(false);
    }
    *point = updated_point;
    validate_curve(engine, &updated, "set")?;
    store_curve(engine, &data_id, updated)
}

fn delete_curve_point(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "spline_index", "point_index"],
        &["target", "spline_index", "point_index"],
    )?;
    let spline_index = read_index(engine, operation, "spline_index")?;
    let point_index = read_index(engine, operation, "point_index")?;
    let (_, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let mut updated = curve;
    let spline = updated.splines.get_mut(spline_index).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "spline_index is out of range",
            &operation_pointer(engine.operation_index, "spline_index"),
        )
    })?;
    if point_index >= spline.points.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "point_index is out of range",
            &operation_pointer(engine.operation_index, "point_index"),
        ));
    }
    spline.points.remove(point_index);
    validate_curve(engine, &updated, "target")?;
    store_curve(engine, &data_id, updated)
}

fn convert_curve(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "to"],
        &["target", "to"],
    )?;
    let destination = read_string(engine, operation, "to")?;
    if destination == "curves" {
        return convert_curve_to_curves(engine, operation);
    }
    if destination != "mesh" {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("curve conversion to `{destination}` is not supported"),
            json!({
                "operation_index": engine.operation_index,
                "pointer": operation_pointer(engine.operation_index, "to"),
                "feature_id": "curve.convert.destination.other",
                "destination": destination,
            }),
        ));
    }
    let (node_id, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let curve = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.curve.as_ref())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
    let taper_profiles =
        crate::eval::curve_taper_profiles(engine.doc, std::slice::from_ref(&node_id))?;
    let taper_profile = curve
        .taper
        .as_ref()
        .and_then(|taper| taper_profiles.get(taper))
        .map(Vec::as_slice);
    let mesh = geom::curve::evaluate_curve_with_taper(curve, taper_profile).map_err(|error| {
        geom::curve::curve_error(
            &error,
            ErrorCode::InvalidOperation,
            "curve geometry is invalid",
            json!({
                "operation_index": engine.operation_index,
                "pointer": operation_pointer(engine.operation_index, "target"),
            }),
        )
    })?;
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    *block = DataBlock {
        data_type: "mesh".to_owned(),
        mesh: Some(mesh),
        ..DataBlock::default()
    };
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "curve node disappeared"))?;
    "mesh".clone_into(&mut node.kind);
    node.primitive = None;
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    record_curve_loss(engine, &data_id)?;
    Ok(true)
}
fn convert_curve_to_curves(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<bool> {
    let (node_id, data_id) = typed_node_data_id(engine, operation, "curve", "curve")?;
    let node = engine
        .doc
        .nodes
        .get(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "curve object disappeared"))?;
    if node.modifiers.iter().any(|modifier| modifier.enabled) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "curve conversion to Curves requires evaluated modifiers, which are not implemented",
            json!({
                "operation_index": engine.operation_index,
                "pointer": operation_pointer(engine.operation_index, "target"),
                "feature_id": "curve.convert.evaluated_modifiers",
                "node_id": node_id,
            }),
        ));
    }
    let converted_data_id = if super::data_user_count(engine, &data_id) > 1 {
        super::unique_id(
            engine,
            &format!("{node_id}_curves"),
            super::IdRegistry::DataBlocks,
        )?
    } else {
        data_id.clone()
    };
    let (hair_curves, materials, has_production_settings) = {
        let block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "curve data block disappeared")
        })?;
        if block.shape_keys.is_some() {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "curve conversion to Curves requires evaluated shape keys, which are not implemented",
                json!({
                    "operation_index": engine.operation_index,
                    "pointer": operation_pointer(engine.operation_index, "target"),
                    "feature_id": "curve.convert.shape_keys",
                    "data_id": data_id,
                }),
            ));
        }
        let curve = block
            .curve
            .as_ref()
            .ok_or_else(|| super::geometry_data::missing_payload(engine, "curve", "target"))?;
        if curve.splines.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Blender leaves Curve objects without splines unchanged during conversion to Curves",
                json!({
                    "operation_index": engine.operation_index,
                    "pointer": operation_pointer(engine.operation_index, "target"),
                    "feature_id": "curve.convert.empty",
                    "data_id": data_id,
                }),
            ));
        }
        if curve.has_bevel_object
            || curve.bevel_depth != 0.0
            || curve.extrude != 0.0
            || (curve.dimensions == crate::model::CurveDimensions::TwoD
                && curve.fill_mode != crate::model::CurveFillMode::None)
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Blender leaves Curve objects unchanged when bevel objects, bevel depth, extrusion, or 2D fill requires implicit mesh conversion",
                json!({
                    "operation_index": engine.operation_index,
                    "pointer": operation_pointer(engine.operation_index, "target"),
                    "feature_id": "curve.convert.production_settings",
                    "data_id": data_id,
                }),
            ));
        }
        if curve
            .splines
            .iter()
            .any(|spline| spline.spline_type == CurveSplineType::Nurbs && spline.custom_knots)
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Curve conversion cannot preserve custom NURBS knot vectors",
                json!({
                    "operation_index": engine.operation_index,
                    "pointer": operation_pointer(engine.operation_index, "target"),
                    "feature_id": "curve.nurbs.knot_mode",
                    "data_id": data_id,
                }),
            ));
        }
        (
            legacy_curve_to_hair_curves(curve)?,
            block.materials.clone(),
            curve_has_unrepresented_production_settings(curve),
        )
    };
    let converted_block = DataBlock {
        data_type: "hair_curves".to_owned(),
        mesh: None,
        materials,
        hair_curves: Some(hair_curves),
        ..DataBlock::default()
    };
    if converted_data_id == data_id {
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
        *block = converted_block;
        engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    } else {
        engine
            .doc
            .data_blocks
            .insert(converted_data_id.clone(), converted_block);
        engine.mark("data_blocks", &converted_data_id, ChangeKind::Created);
        engine.map_id("data_blocks", &data_id, &converted_data_id);
    }
    let node = engine
        .doc
        .nodes
        .get_mut(&node_id)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "curve node disappeared"))?;
    node.data = Some(converted_data_id.clone());
    "hair_curves".clone_into(&mut node.kind);
    node.primitive = None;
    node.modifiers.clear();
    engine.mark("nodes", &node_id, ChangeKind::Updated);
    if has_production_settings {
        record_curve_curves_loss(engine, &converted_data_id)?;
    }
    Ok(true)
}

fn legacy_curve_to_hair_curves(curve: &CurveData) -> Result<HairCurvesData> {
    let point_count = curve
        .splines
        .iter()
        .try_fold(0_usize, |count, spline| {
            count.checked_add(spline.points.len())
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidArgument,
                "curve point count exceeds the supported range",
            )
        })?;
    let normal_mode = match curve.twist_mode.as_str() {
        "MINIMUM" => 0,
        "Z_UP" | "TANGENT" => 1,
        _ => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!(
                    "curve twist mode `{}` has no Curves normal-mode mapping",
                    curve.twist_mode
                ),
                json!({
                    "feature_id":"curve.convert.twist_mode",
                    "twist_mode":curve.twist_mode,
                }),
            ));
        }
    };
    let mut data = HairCurvesData {
        curves: Vec::with_capacity(curve.splines.len()),
        ..HairCurvesData::default()
    };
    let has_bezier = curve
        .splines
        .iter()
        .any(|spline| spline.spline_type == CurveSplineType::Bezier);
    let has_nurbs = curve
        .splines
        .iter()
        .any(|spline| spline.spline_type == CurveSplineType::Nurbs);
    let has_resolution = has_bezier || has_nurbs;
    let mut curve_types = Vec::with_capacity(curve.splines.len());
    let mut cyclic = Vec::with_capacity(curve.splines.len());
    let mut resolutions = has_resolution.then(|| Vec::with_capacity(curve.splines.len()));
    let mut nurbs_orders = has_nurbs.then(|| Vec::with_capacity(curve.splines.len()));
    let mut knots_modes = has_nurbs.then(|| Vec::with_capacity(curve.splines.len()));
    let mut normal_modes = Vec::with_capacity(curve.splines.len());
    let mut radii = Vec::with_capacity(point_count);
    let mut tilts = Vec::with_capacity(point_count);
    let mut left_handles = has_bezier.then(|| Vec::with_capacity(point_count));
    let mut right_handles = has_bezier.then(|| Vec::with_capacity(point_count));
    let mut left_handle_types = has_bezier.then(|| Vec::with_capacity(point_count));
    let mut right_handle_types = has_bezier.then(|| Vec::with_capacity(point_count));
    let mut nurbs_weights = has_nurbs.then(|| vec![json!(1.0); point_count]);
    let mut point_offset = 0;
    for spline in &curve.splines {
        let (curve_type, resolution, nurbs_order, knots_mode) = match spline.spline_type {
            CurveSplineType::Poly => (1, 12, 4, 0),
            CurveSplineType::Bezier => (2, spline.resolution, 4, 0),
            CurveSplineType::Nurbs => {
                let knots_mode = match (spline.use_endpoint, spline.use_bezier) {
                    (false, false) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (true, true) => 3,
                };
                (3, spline.resolution, spline.order, knots_mode)
            }
        };
        curve_types.push(json!(curve_type));
        cyclic.push(json!(spline.cyclic));
        if let Some(resolutions) = &mut resolutions {
            resolutions.push(json!(i32::try_from(resolution).map_err(|_| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "curve spline resolution exceeds Blender's integer range",
                )
            })?));
        }
        if let Some(nurbs_orders) = &mut nurbs_orders {
            nurbs_orders.push(json!(i8::try_from(nurbs_order).map_err(|_| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "NURBS order exceeds Blender's integer range",
                )
            })?));
        }
        if let Some(knots_modes) = &mut knots_modes {
            knots_modes.push(json!(knots_mode));
        }
        normal_modes.push(json!(normal_mode));
        let mut points = Vec::with_capacity(spline.points.len());
        for (point_index, point) in spline.points.iter().enumerate() {
            points.push(point.co);
            radii.push(json!(point.radius));
            tilts.push(json!(point.tilt));
            if spline.spline_type == CurveSplineType::Nurbs
                && let Some(nurbs_weights) = &mut nurbs_weights
            {
                nurbs_weights[point_offset + point_index] = json!(point.weight);
            }
            if let Some(left_handles) = &mut left_handles {
                if spline.spline_type == CurveSplineType::Bezier {
                    left_handles.push(json!(point.handle_left));
                } else {
                    left_handles.push(json!([0.0, 0.0, 0.0]));
                }
            }
            if let Some(right_handles) = &mut right_handles {
                if spline.spline_type == CurveSplineType::Bezier {
                    right_handles.push(json!(point.handle_right));
                } else {
                    right_handles.push(json!([0.0, 0.0, 0.0]));
                }
            }
            if let Some(left_handle_types) = &mut left_handle_types {
                left_handle_types.push(json!(if spline.spline_type == CurveSplineType::Bezier {
                    handle_type_attribute(point.handle_left_type.unwrap_or(point.handle_type))
                } else {
                    0
                }));
            }
            if let Some(right_handle_types) = &mut right_handle_types {
                right_handle_types.push(json!(if spline.spline_type == CurveSplineType::Bezier {
                    handle_type_attribute(point.handle_right_type.unwrap_or(point.handle_type))
                } else {
                    0
                }));
            }
        }
        point_offset += spline.points.len();
        data.curves.push(HairCurve {
            radius: spline.points.first().map_or(0.0, |point| point.radius),
            points,
        });
    }
    data.attributes.insert(
        "curve_type".to_owned(),
        hair_curve_attribute("CURVE", "INT8", curve_types),
    );
    data.attributes.insert(
        "cyclic".to_owned(),
        hair_curve_attribute("CURVE", "BOOLEAN", cyclic),
    );
    if let Some(resolutions) = resolutions {
        data.attributes.insert(
            "resolution".to_owned(),
            hair_curve_attribute("CURVE", "INT", resolutions),
        );
    }
    if let Some(nurbs_orders) = nurbs_orders {
        data.attributes.insert(
            "nurbs_order".to_owned(),
            hair_curve_attribute("CURVE", "INT8", nurbs_orders),
        );
    }
    if let Some(knots_modes) = knots_modes {
        data.attributes.insert(
            "knots_mode".to_owned(),
            hair_curve_attribute("CURVE", "INT8", knots_modes),
        );
    }
    data.attributes.insert(
        "normal_mode".to_owned(),
        hair_curve_attribute("CURVE", "INT8", normal_modes),
    );
    data.attributes.insert(
        "radius".to_owned(),
        hair_curve_attribute("POINT", "FLOAT", radii),
    );
    data.attributes.insert(
        "tilt".to_owned(),
        hair_curve_attribute("POINT", "FLOAT", tilts),
    );
    if let Some(left_handles) = left_handles {
        data.attributes.insert(
            "handle_left".to_owned(),
            hair_curve_attribute("POINT", "FLOAT_VECTOR", left_handles),
        );
    }
    if let Some(right_handles) = right_handles {
        data.attributes.insert(
            "handle_right".to_owned(),
            hair_curve_attribute("POINT", "FLOAT_VECTOR", right_handles),
        );
    }
    if let Some(left_handle_types) = left_handle_types {
        data.attributes.insert(
            "handle_type_left".to_owned(),
            hair_curve_attribute("POINT", "INT8", left_handle_types),
        );
    }
    if let Some(right_handle_types) = right_handle_types {
        data.attributes.insert(
            "handle_type_right".to_owned(),
            hair_curve_attribute("POINT", "INT8", right_handle_types),
        );
    }
    if let Some(nurbs_weights) = nurbs_weights {
        data.attributes.insert(
            "nurbs_weight".to_owned(),
            hair_curve_attribute("POINT", "FLOAT", nurbs_weights),
        );
    }
    Ok(data)
}

fn handle_type_attribute(handle_type: CurveHandleType) -> i8 {
    match handle_type {
        CurveHandleType::Free => 0,
        CurveHandleType::Auto => 1,
        CurveHandleType::Vector => 2,
        CurveHandleType::Aligned => 3,
    }
}

fn hair_curve_attribute(domain: &str, data_type: &str, values: Vec<Value>) -> HairCurveAttribute {
    HairCurveAttribute {
        domain: domain.to_owned(),
        data_type: data_type.to_owned(),
        values,
    }
}

fn curve_has_unrepresented_production_settings(curve: &CurveData) -> bool {
    curve.taper.is_some()
        || curve.taper_radius_mode != crate::model::CurveTaperRadiusMode::Override
        || curve.use_map_taper
        || curve.use_path
        || curve.path_duration != 100
        || curve.eval_time != 0.0
        || !curve.eval_time_fcurves.is_empty()
}

fn record_curve_curves_loss(engine: &mut Engine<'_>, data_id: &Id) -> Result<()> {
    let existed = engine.doc.compatibility.contains_key("losses");
    let slot = engine
        .doc
        .compatibility
        .entry("losses".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let losses = slot.as_array_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "compatibility losses must be an array",
        )
    })?;
    losses.push(json!({
        "feature_id": "curve.convert.production_settings",
        "data_id": data_id,
        "reason": "Blender converts legacy Curve data to Curves only when its evaluated geometry retains a Curves component; bevel, extrusion, and 2D fill trigger implicit mesh conversion, while taper, path, and evaluation-time settings are recorded as lossy conversions.",
        "suggestion": "Keep the legacy Curve object when its production settings must remain editable.",
    }));
    let compatibility_id = Id::new("losses")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?;
    engine.mark(
        "compatibility",
        &compatibility_id,
        if existed {
            ChangeKind::Updated
        } else {
            ChangeKind::Created
        },
    );
    Ok(())
}

fn create_surface(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "points",
            "order_u",
            "order_v",
            "resolution",
            "cyclic_u",
            "cyclic_v",
            "use_endpoint_u",
            "use_endpoint_v",
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
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_surface")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let surface = decode_selected::<SurfaceData>(
        engine,
        operation,
        &[
            "points",
            "order_u",
            "order_v",
            "resolution",
            "cyclic_u",
            "cyclic_v",
            "use_endpoint_u",
            "use_endpoint_v",
        ],
        "",
    )?;
    geom::curve::evaluate_surface(&surface).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("surface geometry is invalid: {error}"),
            &operation_pointer(engine.operation_index, "points"),
        )
    })?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "surface".to_owned(),
            mesh: None,
            surface: Some(surface),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "surface", &id, &data_id)
}

fn create_text_object(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "data_id",
            "body",
            "font",
            "font_name",
            "font_bold",
            "font_bold_italic",
            "font_italic",
            "font_bold_name",
            "font_bold_italic_name",
            "font_italic_name",
            "character_formats",
            "resolution_u",
            "size",
            "align_x",
            "align_y",
            "fill_mode",
            "fill_solver",
            "fill_rule",
            "extrude",
            "bevel_depth",
            "character_spacing",
            "word_spacing",
            "line_spacing",
            "shear",
            "offset_x",
            "offset_y",
            "small_caps_scale",
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
    let id = read_id(engine, operation, "id")?;
    let data_id = create_data_id(engine, operation, &id, "_text")?;
    ensure_data_id_available(engine, &data_id, "data_id")?;
    let text = decode_selected::<TextObjectData>(
        engine,
        operation,
        &[
            "body",
            "font",
            "font_name",
            "font_bold",
            "font_bold_italic",
            "font_italic",
            "font_bold_name",
            "font_bold_italic_name",
            "font_italic_name",
            "character_formats",
            "resolution_u",
            "size",
            "align_x",
            "align_y",
            "fill_mode",
            "fill_solver",
            "fill_rule",
            "extrude",
            "bevel_depth",
            "character_spacing",
            "word_spacing",
            "line_spacing",
            "shear",
            "offset_x",
            "offset_y",
            "small_caps_scale",
        ],
        "",
    )?;
    geom::text::validate_text_data(&text).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("text data is invalid: {error}"),
            &operation_pointer(engine.operation_index, "body"),
        )
    })?;
    insert_data_block(
        engine,
        &data_id,
        DataBlock {
            data_type: "text".to_owned(),
            mesh: None,
            text: Some(text),
            ..DataBlock::default()
        },
    );
    create_node(engine, operation, "text", &id, &data_id)
}

fn update_text_object(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let set = read_set(engine, operation)?;
    check_curve_set_fields(
        engine,
        set,
        &[
            "body",
            "font",
            "font_name",
            "font_bold",
            "font_bold_italic",
            "font_italic",
            "font_bold_name",
            "font_bold_italic_name",
            "font_italic_name",
            "character_formats",
            "resolution_u",
            "size",
            "align_x",
            "align_y",
            "fill_mode",
            "fill_solver",
            "fill_rule",
            "extrude",
            "bevel_depth",
            "character_spacing",
            "word_spacing",
            "line_spacing",
            "shear",
            "offset_x",
            "offset_y",
            "small_caps_scale",
        ],
        "set",
    )?;
    let (_, data_id) = typed_node_data_id(engine, operation, "text", "text")?;
    let text = engine
        .doc
        .data_blocks
        .get(&data_id)
        .and_then(|block| block.text.clone())
        .ok_or_else(|| super::geometry_data::missing_payload(engine, "text", "target"))?;
    let mut updated: Value = serialize(&text)?;
    merge_values(&mut updated, set);
    let mut updated: TextObjectData = decode_value(engine, updated, "set")?;
    if updated.body != text.body && !set.contains_key("character_formats") {
        updated
            .character_formats
            .truncate(updated.body.chars().count());
    }
    geom::text::validate_text_data(&updated).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("text data is invalid: {error}"),
            &operation_pointer(engine.operation_index, "set"),
        )
    })?;
    if updated == text {
        return Ok(false);
    }
    let block =
        engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "text data block disappeared")
        })?;
    block.text = Some(updated);
    engine.mark("data_blocks", &data_id, ChangeKind::Updated);
    Ok(true)
}

fn create_node(
    engine: &mut Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    id: &Id,
    data_id: &Id,
) -> Result<bool> {
    let mut node_operation = Map::new();
    node_operation.insert("op".to_owned(), Value::String("node.create".to_owned()));
    node_operation.insert("kind".to_owned(), Value::String(kind.to_owned()));
    node_operation.insert("id".to_owned(), Value::String(id.to_string()));
    node_operation.insert("data".to_owned(), Value::String(data_id.to_string()));
    for field in [
        "name",
        "collection",
        "transform",
        "parent",
        "parent_inverse",
        "materials",
        "visible",
        "render_visible",
        "selectable",
    ] {
        if let Some(value) = operation.get(field) {
            node_operation.insert(field.to_owned(), value.clone());
        }
    }
    super::node::apply(engine, "node.create", &node_operation)
}

fn create_data_id(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    node_id: &Id,
    suffix: &str,
) -> Result<Id> {
    if operation.contains_key("data_id") {
        let value = operation
            .get("data_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "data_id must be a string",
                    &operation_pointer(engine.operation_index, "data_id"),
                )
            })?;
        parse_id(
            engine,
            value,
            &operation_pointer(engine.operation_index, "data_id"),
        )
    } else {
        let generated = format!("{node_id}{suffix}");
        parse_id(
            engine,
            &generated,
            &operation_pointer(engine.operation_index, "id"),
        )
    }
}

fn ensure_data_id_available(engine: &Engine<'_>, data_id: &Id, field: &str) -> Result<()> {
    if engine.doc.data_blocks.contains_key(data_id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("data-block ID `{data_id}` already exists"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(())
}

fn insert_data_block(engine: &mut Engine<'_>, id: &Id, data: DataBlock) {
    engine.doc.data_blocks.insert(id.clone(), data);
    engine.mark("data_blocks", id, ChangeKind::Created);
}

fn typed_node_data_id(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    kind: &str,
    data_type: &str,
) -> Result<(Id, Id)> {
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let mut targets = resolve_node_targets(engine, target, false)?;
    let node_id = targets.pop().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "node target was not found",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let node = engine.doc.nodes.get(&node_id).ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("node `{node_id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if node.kind != kind {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("target node must be a {kind}"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let data_id = node.data.clone().ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node has no data block"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let block = engine.doc.data_blocks.get(&data_id).ok_or_else(|| {
        engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node data block `{data_id}` is missing"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    if block.data_type != data_type {
        return Err(engine.error(
            ErrorCode::SceneInvalid,
            format!("{kind} node must reference a {data_type} Data-Block"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    Ok((node_id, data_id))
}

fn store_curve(engine: &mut Engine<'_>, data_id: &Id, curve: CurveData) -> Result<bool> {
    let block =
        engine.doc.data_blocks.get_mut(data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "curve data block disappeared")
        })?;
    if block.curve.as_ref() == Some(&curve) {
        return Ok(false);
    }
    block.curve = Some(curve);
    engine.mark("data_blocks", data_id, ChangeKind::Updated);
    Ok(true)
}

fn record_curve_loss(engine: &mut Engine<'_>, data_id: &Id) -> Result<()> {
    let existed = engine.doc.compatibility.contains_key("losses");
    let slot = engine
        .doc
        .compatibility
        .entry("losses".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()));
    let losses = slot.as_array_mut().ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "compatibility losses must be an array",
        )
    })?;
    losses.push(json!({
        "feature_id": "curve.production_data",
        "data_id": data_id,
        "reason": "Converting a curve to a mesh discards its editable spline and curve settings.",
        "suggestion": "Keep the curve object when its parametric production data must remain editable.",
    }));
    let compatibility_id = Id::new("losses")
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.message))?;
    engine.mark(
        "compatibility",
        &compatibility_id,
        if existed {
            ChangeKind::Updated
        } else {
            ChangeKind::Created
        },
    );
    Ok(())
}

fn read_set<'a>(
    engine: &Engine<'_>,
    operation: &'a Map<String, Value>,
) -> Result<&'a Map<String, Value>> {
    operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })
}

fn check_curve_set_fields(
    engine: &Engine<'_>,
    values: &Map<String, Value>,
    allowed: &[&str],
    prefix: &str,
) -> Result<()> {
    super::check_set_fields_by(engine, values, allowed, |engine, key| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("unknown field `{key}`"),
            &operation_pointer(engine.operation_index, &format!("{prefix}/{key}")),
        )
    })
}

fn read_index(engine: &Engine<'_>, operation: &Map<String, Value>, field: &str) -> Result<usize> {
    let pointer = operation_pointer(engine.operation_index, field);
    operation
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a non-negative integer"),
                &pointer,
            )
        })
}

fn decode_selected<T: DeserializeOwned>(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    fields: &[&str],
    pointer: &str,
) -> Result<T> {
    let mut values = Map::new();
    for field in fields {
        if let Some(value) = operation.get(*field) {
            values.insert((*field).to_owned(), value.clone());
        }
    }
    decode_value(engine, Value::Object(values), pointer)
}

fn decode_value<T: DeserializeOwned>(
    engine: &Engine<'_>,
    value: Value,
    pointer: &str,
) -> Result<T> {
    serde_json::from_value(value).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid curve, surface, or text data: {error}"),
            &operation_pointer(engine.operation_index, pointer),
        )
    })
}

fn serialize<T: serde::Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
}

fn merge_values(destination: &mut Value, source: &Map<String, Value>) {
    if let Some(destination) = destination.as_object_mut() {
        for (key, value) in source {
            destination.insert(key.clone(), value.clone());
        }
    }
}

fn validate_curve(engine: &Engine<'_>, curve: &CurveData, pointer: &str) -> Result<()> {
    geom::curve::evaluate_curve_with_taper(curve, Some(&[]))
        .map(|_| ())
        .map_err(|error| {
            geom::curve::curve_error(
                &error,
                ErrorCode::InvalidOperation,
                "curve geometry is invalid",
                json!({
                    "operation_index": engine.operation_index,
                    "pointer": operation_pointer(engine.operation_index, pointer),
                }),
            )
        })
}

// Validate only the edited curve's dependency chain; unrelated data must not fail this operation.

fn validate_curve_taper(
    engine: &Engine<'_>,
    root_curve: &CurveData,
    root_id: &Id,
    pointer: &str,
) -> Result<()> {
    fn visit(
        engine: &Engine<'_>,
        id: &Id,
        root_curve: &CurveData,
        root_id: &Id,
        pointer: &str,
        active: &mut Vec<Id>,
        complete: &mut BTreeSet<Id>,
    ) -> Result<()> {
        if let Some(start) = active.iter().position(|active_id| active_id == id) {
            let mut cycle: Vec<String> = active[start..].iter().map(ToString::to_string).collect();
            cycle.push(id.to_string());
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("curve taper dependency cycle: {}", cycle.join(" -> ")),
                &operation_pointer(engine.operation_index, pointer),
            ));
        }
        if complete.contains(id) {
            return Ok(());
        }
        let curve = if id == root_id {
            root_curve
        } else {
            let block = engine.doc.data_blocks.get(id).ok_or_else(|| {
                engine.error(
                    ErrorCode::DependencyMissing,
                    "curve taper references a missing Data-Block",
                    &operation_pointer(engine.operation_index, pointer),
                )
            })?;
            if block.data_type != "curve" {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    "curve taper reference is not a curve Data-Block",
                    &operation_pointer(engine.operation_index, pointer),
                ));
            }
            block.curve.as_ref().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "curve taper reference is not a curve Data-Block",
                    &operation_pointer(engine.operation_index, pointer),
                )
            })?
        };

        active.push(id.clone());
        if let Some(taper) = &curve.taper {
            visit(
                engine, taper, root_curve, root_id, pointer, active, complete,
            )?;
        }
        active.pop();
        complete.insert(id.clone());
        Ok(())
    }

    visit(
        engine,
        root_id,
        root_curve,
        root_id,
        pointer,
        &mut Vec::new(),
        &mut BTreeSet::new(),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "unit tests assert operation contracts")]

    use super::super::apply_batch;
    use crate::{
        error::ErrorCode,
        model::{Id, SceneDoc},
    };
    use serde_json::{Value, json};

    fn batch(doc: &SceneDoc, operations: Value) -> Value {
        let mut batch = json!({
            "schema_version": 1,
            "base_revision": doc.revision,
        });
        batch["operations"] = operations;
        batch
    }

    fn curve_create(id: &str, data_id: &str, taper: Option<&str>) -> Value {
        let mut operation = json!({
            "op": "curve.create",
            "id": id,
            "data_id": data_id,
            "splines": [{
                "type": "poly",
                "points": [{"co": [0.0, 0.0, 0.0]}, {"co": [1.0, 0.0, 0.0]}],
            }],
        });
        if let Some(taper) = taper {
            operation["taper"] = json!(taper);
        }
        operation
    }

    #[test]
    fn curve_create_and_update_reject_missing_and_non_curve_tapers() {
        let base = SceneDoc::new("scene-test".to_owned());
        let missing = apply_batch(
            &base,
            &batch(
                &base,
                json!([curve_create("owner", "owner_data", Some("missing"))]),
            ),
        )
        .unwrap_err();
        assert_eq!(missing.code, ErrorCode::DependencyMissing);
        assert_eq!(missing.details["operation_index"], json!(0));
        assert_eq!(missing.details["pointer"], json!("/operations/0/taper"));

        // Blender 5.2.2's displist_calc_taper returns 1 for non-curves; reject references before evaluation.

        let created = apply_batch(
            &base,
            &batch(
                &base,
                json!([
                    {"op":"node.create","id":"box","kind":"box","params":{"size":1.0}},
                    curve_create("owner", "owner_data", Some("box_mesh")),
                ]),
            ),
        )
        .unwrap_err();
        assert_eq!(created.code, ErrorCode::InvalidOperation);
        assert_eq!(created.details["operation_index"], json!(1));

        let valid = apply_batch(
            &base,
            &batch(
                &base,
                json!([
                    {"op":"node.create","id":"box","kind":"box","params":{"size":1.0}},
                    curve_create("owner", "owner_data", None),
                ]),
            ),
        )
        .unwrap();
        let missing_update = apply_batch(
            &valid.doc,
            &batch(
                &valid.doc,
                json!([{"op":"curve.update","target":{"id":"owner"},"set":{"taper":"missing"}}]),
            ),
        )
        .unwrap_err();
        assert_eq!(missing_update.code, ErrorCode::DependencyMissing);
        assert_eq!(
            missing_update.details["pointer"],
            json!("/operations/0/set/taper")
        );

        let non_curve_update = apply_batch(
            &valid.doc,
            &batch(
                &valid.doc,
                json!([{"op":"curve.update","target":{"id":"owner"},"set":{"taper":"box_mesh"}}]),
            ),
        )
        .unwrap_err();
        assert_eq!(non_curve_update.code, ErrorCode::InvalidOperation);
    }

    #[test]
    fn curve_create_and_update_reject_taper_cycles_at_operation_time() {
        let base = SceneDoc::new("scene-test".to_owned());
        let self_cycle = apply_batch(
            &base,
            &batch(
                &base,
                json!([curve_create("self", "self_data", Some("self_data"))]),
            ),
        )
        .unwrap_err();
        assert_eq!(self_cycle.code, ErrorCode::InvalidOperation);
        assert!(self_cycle.message.contains("curve taper dependency cycle"));

        let created = apply_batch(
            &base,
            &batch(
                &base,
                json!([
                    curve_create("first", "first_data", None),
                    curve_create("second", "second_data", None),
                ]),
            ),
        )
        .unwrap();
        let cycle = apply_batch(
            &created.doc,
            &batch(
                &created.doc,
                json!([
                    {"op":"curve.update","target":{"id":"first"},"set":{"taper":"second_data"}},
                    {"op":"curve.update","target":{"id":"second"},"set":{"taper":"first_data"}},
                ]),
            ),
        )
        .unwrap_err();
        assert_eq!(cycle.code, ErrorCode::InvalidOperation);
        assert_eq!(cycle.details["operation_index"], json!(1));
        assert!(cycle.message.contains("curve taper dependency cycle"));
    }

    #[test]
    fn deleting_taper_data_with_cascade_clears_surviving_curve_reference() {
        let base = SceneDoc::new("scene-test".to_owned());
        let created = apply_batch(
            &base,
            &batch(
                &base,
                json!([
                    curve_create("profile", "profile_data", None),
                    curve_create("owner", "owner_data", Some("profile_data")),
                ]),
            ),
        )
        .unwrap();
        let deleted = apply_batch(
            &created.doc,
            &batch(
                &created.doc,
                json!([{"op":"node.delete","target":{"id":"profile"},"cascade_data":true}]),
            ),
        )
        .unwrap();
        let owner_data = Id::new("owner_data").unwrap();
        assert!(
            !deleted
                .doc
                .data_blocks
                .contains_key(&Id::new("profile_data").unwrap())
        );
        assert_eq!(
            deleted.doc.data_blocks[&owner_data]
                .curve
                .as_ref()
                .unwrap()
                .taper,
            None
        );
        assert_eq!(
            deleted.changes["data_blocks"]["updated"],
            json!(["owner_data"])
        );
    }

    #[test]
    fn curve_update_does_not_validate_unrelated_taper_dependencies() {
        let base = SceneDoc::new("scene-test".to_owned());
        let created = apply_batch(
            &base,
            &batch(
                &base,
                json!([
                    curve_create("unrelated", "unrelated_data", None),
                    curve_create("target", "target_data", None),
                ]),
            ),
        )
        .unwrap();
        let mut doc = created.doc;
        doc.data_blocks
            .get_mut(&Id::new("unrelated_data").unwrap())
            .unwrap()
            .curve
            .as_mut()
            .unwrap()
            .taper = Some(Id::new("missing").unwrap());

        let updated = apply_batch(
            &doc,
            &batch(
                &doc,
                json!([{"op":"curve.update","target":{"id":"target"},"set":{"bevel_depth":0.25}}]),
            ),
        )
        .unwrap();
        assert_eq!(
            updated.doc.data_blocks[&Id::new("target_data").unwrap()]
                .curve
                .as_ref()
                .unwrap()
                .bevel_depth,
            0.25
        );
    }
}
