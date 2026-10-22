//! Data transfer modifier: per-domain mapping, layer selection, and mixing.

use super::{
    AttributeLayerSelection, Edge, Face, Mesh, Modifier, Result, SceneOperand, boolean,
    get_group_weights, invalid, number, optional_string, string, string_set,
};
use crate::blender_rng::BlenderRng;
use crate::geom::bvh::{Bounds, KdopTree, NearestHit, RayHit, TREE_EPSILON, kdop_bounds};
use crate::geom::color_layers::{self, ColorDomain};
use crate::geom::uv_layers::{UvLayer, UvStore, read as read_uv_store, read_layers, write_layers};
use glam::{DVec2, DVec3};
use serde_json::{Map, Value, json};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// `BLI_bvhtree_find_nearest` (depth-first, `BLI_kdopbvh.cc:1696`) over a leaf tree, for transfer's
/// nearest-point mappings. `test(leaf, co)` returns the leaf's squared distance and a payload, or
/// Blender's `(a + b) * 0.5`: the same two operations in the same order, written so the midpoint lint does not match.
fn halved(sum: f64) -> f64 {
    sum * 0.5
}

/// `None` when the leaf is rejected; the nearest accepted leaf within `max_distance_squared` wins.
trait NearestQuery {
    fn nearest<T>(
        &self,
        co: DVec3,
        max_distance_squared: f64,
        test: impl FnMut(usize, DVec3) -> Option<(f64, T)>,
    ) -> Option<(usize, f64, T)>;
}

impl NearestQuery for KdopTree {
    fn nearest<T>(
        &self,
        co: DVec3,
        max_distance_squared: f64,
        mut test: impl FnMut(usize, DVec3) -> Option<(f64, T)>,
    ) -> Option<(usize, f64, T)> {
        let co_f32 = co.as_vec3().to_array();
        let mut nearest = NearestHit {
            index: -1,
            co: co_f32,
            dist_sq: max_distance_squared as f32,
        };
        let mut found = None;
        self.find_nearest(
            co_f32,
            &mut nearest,
            &mut |leaf: usize, _: [f32; 3], hit: &mut NearestHit| {
                let Some((distance, payload)) = test(leaf, co) else {
                    return;
                };
                if distance < f64::from(hit.dist_sq) {
                    hit.index = leaf as isize;
                    hit.co = co_f32;
                    hit.dist_sq = distance as f32;
                    found = Some((leaf, distance, payload));
                }
            },
            false,
        );
        found
    }
}

pub(super) fn data_transfer(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    subject_layer_selection: &AttributeLayerSelection,
) -> Result<Mesh> {
    // The resolver keys each reference by its parameter name (`eval.rs`), so the source operand
    // is `object`; the parameter's value is the object id and is not a key.
    let Some(_) = optional_string(modifier, "object")? else {
        return Ok(mesh.clone());
    };
    let Some(source) = operands.get("object").and_then(|items| items.first()) else {
        return Ok(mesh.clone());
    };
    let use_transform = boolean(modifier, "use_object_transform")?;
    let (source_to_destination_f32, destination_to_source_f32) = if use_transform {
        let matrices = source
            .blender_matrices_f32
            .as_ref()
            .ok_or_else(|| invalid(modifier, "object", "source Blender f32 transform matrices"))?;
        let local_to_subject = matrices
            .local_to_subject
            .ok_or_else(|| invalid(modifier, "object", "source local-to-subject f32 transform"))?;
        let subject_to_local = matrices
            .subject_to_local
            .ok_or_else(|| invalid(modifier, "object", "source subject-to-local f32 transform"))?;
        (local_to_subject, subject_to_local)
    } else {
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        (identity, identity)
    };
    let use_vertices = boolean(modifier, "use_vert_data")?;
    let use_edges = boolean(modifier, "use_edge_data")?;
    let use_loops = boolean(modifier, "use_loop_data")?;
    let use_polygons = boolean(modifier, "use_poly_data")?;
    let vertex_mapping = string(modifier, "vert_mapping")?;
    let edge_mapping = string(modifier, "edge_mapping")?;
    let loop_mapping = string(modifier, "loop_mapping")?;
    let polygon_mapping = string(modifier, "poly_mapping")?;
    validate_transfer_mapping(modifier, "vert_mapping", vertex_mapping)?;
    validate_transfer_mapping(modifier, "edge_mapping", edge_mapping)?;
    validate_transfer_mapping(modifier, "loop_mapping", loop_mapping)?;
    validate_transfer_mapping(modifier, "poly_mapping", polygon_mapping)?;
    let factor = number(modifier, "mix_factor")?.clamp(0.0, 1.0);
    let mix_mode = string(modifier, "mix_mode")?;
    if !matches!(
        mix_mode,
        "REPLACE" | "ABOVE_THRESHOLD" | "BELOW_THRESHOLD" | "MIX" | "ADD" | "SUB" | "MUL"
    ) {
        return Err(invalid(
            modifier,
            "mix_mode",
            "REPLACE, ABOVE_THRESHOLD, BELOW_THRESHOLD, MIX, ADD, SUB, or MUL",
        ));
    }
    for field in [
        "layers_vgroup_select_src",
        "layers_vcol_vert_select_src",
        "layers_vcol_loop_select_src",
        "layers_uv_select_src",
        "layers_vgroup_select_dst",
        "layers_vcol_vert_select_dst",
        "layers_vcol_loop_select_dst",
        "layers_uv_select_dst",
    ] {
        if string(modifier, field)?.is_empty() {
            return Err(invalid(modifier, field, "a non-empty selector"));
        }
    }
    let use_max_distance = boolean(modifier, "use_max_distance")?;
    let max_distance = if use_max_distance {
        number(modifier, "max_distance")?
    } else {
        f64::INFINITY
    };
    let ray_radius = number(modifier, "ray_radius")?;
    let islands_precision = number(modifier, "islands_precision")?;
    if (use_max_distance && !max_distance.is_finite())
        || max_distance < 0.0
        || !ray_radius.is_finite()
        || ray_radius < 0.0
        || !islands_precision.is_finite()
        || !(0.0..=1.0).contains(&islands_precision)
    {
        return Err(invalid(
            modifier,
            "max_distance, ray_radius, or islands_precision",
            "a non-negative distance and ray radius, and island precision from zero through one",
        ));
    }
    let data_types_verts = string_set(modifier, "data_types_verts")?;
    let data_types_edges = string_set(modifier, "data_types_edges")?;
    let data_types_loops = string_set(modifier, "data_types_loops")?;
    let data_types_polys = string_set(modifier, "data_types_polys")?;
    let needs_loop_normals = use_loops
        && (data_types_loops.iter().any(|name| name == "CUSTOM_NORMAL")
            || matches!(loop_mapping, "NEAREST_NORMAL" | "POLYINTERP_LNORPROJ"));
    let needs_vertex_normals = (use_vertices && vertex_mapping == "POLYINTERP_VNORPROJ")
        || (use_edges && edge_mapping == "EDGEINTERP_VNORPROJ");
    let needs_source_face_graph = (use_vertices
        && !data_types_verts.is_empty()
        && matches!(
            vertex_mapping,
            "POLY_NEAREST" | "POLYINTERP_NEAREST" | "POLYINTERP_VNORPROJ"
        ))
        || (use_edges && !data_types_edges.is_empty() && edge_mapping == "POLY_NEAREST")
        || (use_loops && !data_types_loops.is_empty() && loop_mapping != "TOPOLOGY")
        || (use_polygons && !data_types_polys.is_empty() && polygon_mapping != "TOPOLOGY");
    let mapping_context = TransferMapContext::new(
        &source.mesh,
        mesh,
        source_to_destination_f32,
        destination_to_source_f32,
        max_distance,
        ray_radius,
        islands_precision,
        &source.layer_selection,
        subject_layer_selection,
        &source.selected_bones,
        &source.deforming_bones,
        needs_loop_normals,
        needs_vertex_normals,
        needs_source_face_graph,
    )?;
    let mask_group = optional_string(modifier, "vertex_group")?.filter(|name| !name.is_empty());
    let mask_weights = if let Some(name) = mask_group {
        let exists = mesh
            .attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .is_some_and(|groups| groups.contains_key(name));
        exists.then(|| get_group_weights(mesh, name)).transpose()?
    } else {
        None
    };
    let invert_mask = boolean(modifier, "invert_vertex_group")? && mask_weights.is_some();
    let mut result = mesh.clone();
    if use_vertices {
        for name in &data_types_verts {
            match name.as_str() {
                "VGROUP_WEIGHTS" => transfer_vertex_groups(
                    &mut result,
                    &source.mesh,
                    modifier,
                    vertex_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "BEVEL_WEIGHT_VERT" => transfer_vertex_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "bevel_weight_vert",
                    vertex_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "COLOR_VERTEX" => transfer_vertex_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "color",
                    vertex_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_verts",
                        "VGROUP_WEIGHTS, BEVEL_WEIGHT_VERT, or COLOR_VERTEX",
                    ));
                }
            }
        }
    }
    if use_edges {
        for name in &data_types_edges {
            match name.as_str() {
                "SHARP_EDGE" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "sharp",
                    edge_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "SEAM" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "seam",
                    edge_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "CREASE" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "crease_edge",
                    edge_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "BEVEL_WEIGHT_EDGE" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "bevel_weight_edge",
                    edge_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "FREESTYLE_EDGE" => transfer_edge_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "freestyle_edge",
                    edge_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_edges",
                        "SHARP_EDGE, SEAM, CREASE, BEVEL_WEIGHT_EDGE, or FREESTYLE_EDGE",
                    ));
                }
            }
        }
    }
    if use_loops {
        for name in &data_types_loops {
            match name.as_str() {
                "UV" => transfer_uv(
                    &mut result,
                    &source.mesh,
                    modifier,
                    loop_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "CUSTOM_NORMAL" => transfer_custom_normals(
                    &mut result,
                    modifier,
                    loop_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                )?,
                "COLOR_CORNER" => transfer_color_layers(
                    &mut result,
                    &source.mesh,
                    modifier,
                    loop_mapping,
                    factor,
                    mix_mode,
                    mask_weights.as_ref(),
                    invert_mask,
                    &mapping_context,
                    "corner",
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_loops",
                        "CUSTOM_NORMAL, UV, or COLOR_CORNER",
                    ));
                }
            }
        }
    }
    if use_polygons {
        for name in &data_types_polys {
            match name.as_str() {
                "SMOOTH" => transfer_typed_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "shade_smooth",
                    "shade_smooth",
                    polygon_mapping,
                    factor,
                    mix_mode,
                    &mapping_context,
                )?,
                "FREESTYLE_FACE" => transfer_typed_attribute(
                    &mut result,
                    &source.mesh,
                    modifier,
                    "freestyle_face",
                    "freestyle_face",
                    polygon_mapping,
                    factor,
                    mix_mode,
                    &mapping_context,
                )?,
                _ => {
                    return Err(invalid(
                        modifier,
                        "data_types_polys",
                        "SMOOTH or FREESTYLE_FACE",
                    ));
                }
            }
        }
    }
    Ok(result)
}

fn validate_transfer_mapping(modifier: &Modifier, field: &str, value: &str) -> Result<()> {
    let valid = match field {
        "vert_mapping" => matches!(
            value,
            "TOPOLOGY"
                | "NEAREST"
                | "EDGE_NEAREST"
                | "EDGEINTERP_NEAREST"
                | "POLY_NEAREST"
                | "POLYINTERP_NEAREST"
                | "POLYINTERP_VNORPROJ"
        ),
        "edge_mapping" => matches!(
            value,
            "TOPOLOGY" | "VERT_NEAREST" | "NEAREST" | "POLY_NEAREST" | "EDGEINTERP_VNORPROJ"
        ),
        "loop_mapping" => matches!(
            value,
            "TOPOLOGY"
                | "NEAREST_NORMAL"
                | "NEAREST_POLYNOR"
                | "NEAREST_POLY"
                | "POLYINTERP_NEAREST"
                | "POLYINTERP_LNORPROJ"
        ),
        "poly_mapping" => matches!(
            value,
            "TOPOLOGY" | "NEAREST" | "NORMAL" | "POLYINTERP_PNORPROJ"
        ),
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid(
            modifier,
            field,
            "a supported Blender data mapping identifier",
        ))
    }
}

fn transfer_vertex_groups(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    let source_groups = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid(
                modifier,
                "data_types_verts",
                "a source mesh with vertex groups",
            )
        })?;
    let mut groups = destination
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut source_order = mapping_context
        .source_layer_selection
        .vertex_group_names
        .clone();
    if source_order.is_empty() {
        source_order = ordered_group_names(source, source_groups);
    }
    let mut destination_order = mapping_context
        .destination_layer_selection
        .vertex_group_names
        .clone();
    if destination_order.is_empty() {
        destination_order = ordered_group_names(destination, &groups);
    }
    let source_selection = string(modifier, "layers_vgroup_select_src")?;
    let destination_selection = string(modifier, "layers_vgroup_select_dst")?;
    let selected_source = match source_selection {
        "ACTIVE" => mapping_context
            .source_layer_selection
            .active_vertex_group_index
            .and_then(|index| source_order.get(index).cloned())
            .into_iter()
            .collect(),
        "ALL" => source_order.clone(),
        "BONE_SELECT" => source_order
            .iter()
            .filter(|name| mapping_context.selected_bones.contains(*name))
            .cloned()
            .collect(),
        "BONE_DEFORM" => source_order
            .iter()
            .filter(|name| mapping_context.deforming_bones.contains(*name))
            .cloned()
            .collect(),
        _ => {
            let Some(name) = source_order
                .iter()
                .find(|name| name.as_str() == source_selection)
            else {
                return Err(invalid(
                    modifier,
                    "layers_vgroup_select_src",
                    "ALL or a valid source group",
                ));
            };
            vec![name.clone()]
        }
    };
    for source_name in selected_source {
        let Some(source_index) = source_order.iter().position(|name| name == &source_name) else {
            continue;
        };
        let destination_name = match destination_selection {
            "NAME" => Some(source_name.clone()),
            "ACTIVE" => mapping_context
                .destination_layer_selection
                .active_vertex_group_index
                .and_then(|index| destination_order.get(index).cloned()),
            "INDEX" => destination_order.get(source_index).cloned(),
            _ => {
                if matches!(source_selection, "ALL" | "BONE_SELECT" | "BONE_DEFORM") {
                    return Err(invalid(
                        modifier,
                        "layers_vgroup_select_dst",
                        "dynamic destination groups require a single source group",
                    ));
                }
                let Some(name) = destination_order
                    .iter()
                    .find(|name| name.as_str() == destination_selection)
                else {
                    return Err(invalid(
                        modifier,
                        "layers_vgroup_select_dst",
                        "NAME, INDEX, or a valid destination group",
                    ));
                };
                Some(name.clone())
            }
        };
        let Some(destination_name) = destination_name else {
            continue;
        };
        if !groups.contains_key(&destination_name) {
            continue;
        }
        let source_weights = source_groups
            .get(&source_name)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                invalid(
                    modifier,
                    "data_types_verts",
                    "valid source vertex-group weights",
                )
            })?;
        let mut destination_weights = groups
            .get(&destination_name)
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (index, vertex) in destination.vertices.iter().enumerate() {
            let Some(mapped) = mapped_vertex_sources(mapping_context, destination, index, mapping)
            else {
                continue;
            };
            let source_value = mapped
                .0
                .iter()
                .map(|(source_index, weight)| {
                    let source_id = source.vertices[*source_index].id;
                    source_weights
                        .get(&format!("v{source_id}"))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        * weight
                })
                .sum::<f64>();
            let key = format!("v{}", vertex.id);
            let prior = destination_weights
                .get(&key)
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let mut mask = mask_weights.map_or(1.0, |weights| {
                weights.get(&vertex.id).copied().unwrap_or(0.0)
            });
            if invert_mask {
                mask = 1.0 - mask;
            }
            let mixed = blend_attribute(prior, source_value, factor * mask, mix_mode, modifier)?;
            destination_weights.insert(key, json!(mixed.clamp(0.0, 1.0)));
        }
        groups.insert(destination_name, Value::Object(destination_weights));
    }
    destination
        .attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));
    Ok(())
}

fn transfer_vertex_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    name: &str,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    if name == "color" {
        return transfer_color_layers(
            destination,
            source,
            modifier,
            mapping,
            factor,
            mix_mode,
            mask_weights,
            invert_mask,
            mapping_context,
            "point",
        );
    }
    if let Some(source_attribute) = source.attributes.get(name) {
        let source_values = source_attribute
            .get("values")
            .and_then(Value::as_object)
            .or_else(|| source_attribute.as_object())
            .ok_or_else(|| invalid(modifier, name, "a vertex-ID keyed source attribute"))?;
        let Some(destination_attribute) = destination.attributes.get(name) else {
            return Ok(());
        };
        let mut output = destination_attribute
            .get("values")
            .and_then(Value::as_object)
            .or_else(|| destination_attribute.as_object())
            .cloned()
            .unwrap_or_default();
        for (index, vertex) in destination.vertices.iter().enumerate() {
            let Some(mapped) = mapped_vertex_sources(mapping_context, destination, index, mapping)
            else {
                continue;
            };
            let Some(source_value) = weighted_vertex_value(source, source_values, &mapped) else {
                continue;
            };
            let key = format!("v{}", vertex.id);
            if name == "bevel_weight_vert"
                && let Some(source_value) = source_value.as_f64()
            {
                let prior = output.get(&key).and_then(Value::as_f64).unwrap_or(0.0);
                let mask = transfer_mask_weight(mask_weights, vertex.id, invert_mask);
                output.insert(
                    key,
                    json!(blend_attribute(
                        prior,
                        source_value,
                        factor * mask,
                        mix_mode,
                        modifier
                    )?),
                );
            } else {
                output.insert(key, source_value);
            }
        }
        if let Some(source_fields) = source_attribute.as_object()
            && source_fields.contains_key("values")
        {
            let mut fields = destination_attribute
                .as_object()
                .cloned()
                .unwrap_or_default();
            fields.insert("values".to_owned(), Value::Object(output));
            destination
                .attributes
                .insert(name.to_owned(), Value::Object(fields));
        } else {
            destination
                .attributes
                .insert(name.to_owned(), Value::Object(output));
        }
        return Ok(());
    }
    let Some(source_layers) = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    let Some(source_index) = source_layers.iter().position(|attribute| {
        attribute.get("name").and_then(Value::as_str) == Some(name)
            && attribute.get("domain").and_then(Value::as_str) == Some("POINT")
    }) else {
        return Ok(());
    };
    let Some(destination_index) = destination
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .and_then(|layers| {
            layers.iter().position(|attribute| {
                attribute.get("name").and_then(Value::as_str) == Some(name)
                    && attribute.get("domain").and_then(Value::as_str) == Some("POINT")
            })
        })
    else {
        return Ok(());
    };
    let source_values = source_layers[source_index]
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(modifier, name, "source point-attribute values"))?;
    let destination_values = destination.attributes["blender_attributes"][destination_index]
        .get("values")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| invalid(modifier, name, "destination point-attribute values"))?;
    let mut output = destination_values;
    for index in 0..destination.vertices.len() {
        let Some(mapped) = mapped_vertex_sources(mapping_context, destination, index, mapping)
        else {
            continue;
        };
        let samples = mapped
            .0
            .iter()
            .filter_map(|(source_index, weight)| {
                Some((source_values.get(*source_index)?.clone(), *weight))
            })
            .collect::<Vec<_>>();
        if let Some(value) = weighted_values(&samples) {
            if name == "bevel_weight_vert"
                && let Some(source_value) = value.as_f64()
            {
                let prior = output.get(index).and_then(Value::as_f64).unwrap_or(0.0);
                let mask =
                    transfer_mask_weight(mask_weights, destination.vertices[index].id, invert_mask);
                output[index] = json!(blend_attribute(
                    prior,
                    source_value,
                    factor * mask,
                    mix_mode,
                    modifier,
                )?);
            } else {
                output[index] = value;
            }
        }
    }
    if let Some(layers) = destination
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
        && let Some(layer) = layers.get_mut(destination_index)
    {
        layer["values"] = Value::Array(output);
    }
    Ok(())
}

fn transfer_edge_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    name: &str,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    let intrinsic_flag = matches!(name, "sharp" | "seam");
    // The modifier never creates a layer (`use_create = false`, MOD_datatransfer.cc), and the
    // sharp/seam mappings exist only for a destination layer that is already present
    // (blenkernel/intern/data_transfer.cc, `lookup_for_write_span` before `add_item`).
    if intrinsic_flag && !destination.attributes.contains_key(name) {
        return Ok(());
    }
    let float_attribute = matches!(name, "crease_edge" | "bevel_weight_edge");
    let blender_index = (!intrinsic_flag)
        .then(|| {
            destination
                .attributes
                .get("blender_attributes")
                .and_then(Value::as_array)
                .and_then(|layers| {
                    layers.iter().position(|attribute| {
                        attribute.get("name").and_then(Value::as_str) == Some(name)
                            && attribute.get("domain").and_then(Value::as_str) == Some("EDGE")
                    })
                })
        })
        .flatten();
    let mut output = destination
        .attributes
        .get(name)
        .and_then(|value| value.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut array_output = blender_index.and_then(|index| {
        destination
            .attributes
            .get("blender_attributes")?
            .as_array()?
            .get(index)?
            .get("values")?
            .as_array()
            .cloned()
    });
    if let Some(array) = &array_output {
        for (index, edge) in destination.edges.iter().enumerate() {
            if let Some(value) = array.get(index) {
                output.insert(format!("e{}", edge.id), value.clone());
            }
        }
    }
    if intrinsic_flag && output.is_empty() {
        for (index, edge) in destination.edges.iter().enumerate() {
            if let Some(value) = source_edge_value(destination, name, index, edge.id) {
                output.insert(format!("e{}", edge.id), value);
            }
        }
    }
    if !intrinsic_flag && blender_index.is_none() && !destination.attributes.contains_key(name) {
        return Ok(());
    }
    for (destination_index, edge) in destination.edges.iter().enumerate() {
        let Some(mapped) =
            mapped_edge_sources(mapping_context, destination, destination_index, mapping)
        else {
            continue;
        };
        let key = format!("e{}", edge.id);
        let mut has_source = false;
        let mut scalar_source = 0.0;
        let mut bool_source = false;
        for &(source_index, weight) in &mapped.0 {
            let Some(source_edge) = source.edges.get(source_index) else {
                continue;
            };
            let Some(value) = source_edge_value(source, name, source_index, source_edge.id) else {
                continue;
            };
            if float_attribute {
                let Some(value) = value.as_f64() else {
                    continue;
                };
                scalar_source += value * weight;
            } else if weight > 0.0 && value.as_bool().unwrap_or(false) {
                bool_source = true;
            }
            has_source = true;
        }
        if !has_source {
            continue;
        }
        let value = if float_attribute {
            let prior = array_output
                .as_ref()
                .and_then(|values| values.get(destination_index))
                .and_then(Value::as_f64)
                .or_else(|| output.get(&key).and_then(Value::as_f64))
                .unwrap_or(0.0);
            let mask = transfer_edge_mask_weight(edge, mask_weights, invert_mask);
            json!(blend_attribute(
                prior,
                scalar_source,
                factor * mask,
                mix_mode,
                modifier
            )?)
        } else {
            Value::Bool(bool_source)
        };
        output.insert(key, value.clone());
        if let Some(array) = array_output.as_mut()
            && let Some(slot) = array.get_mut(destination_index)
        {
            *slot = value;
        }
    }
    if intrinsic_flag {
        destination.attributes.insert(
            name.to_owned(),
            json!({"domain":"edge","type":"bool","values":output}),
        );
    } else if let Some(index) = blender_index
        && let Some(array) = array_output
        && let Some(layers) = destination
            .attributes
            .get_mut("blender_attributes")
            .and_then(Value::as_array_mut)
        && let Some(layer) = layers.get_mut(index)
    {
        layer["values"] = Value::Array(array);
    } else if destination.attributes.contains_key(name) {
        let kind = if float_attribute { "float" } else { "bool" };
        destination.attributes.insert(
            name.to_owned(),
            json!({"domain":"edge","type":kind,"values":output}),
        );
    }
    Ok(())
}

fn transfer_edge_mask_weight(
    edge: &Edge,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert: bool,
) -> f64 {
    let Some(weights) = mask_weights else {
        return 1.0;
    };
    let [first, second] = edge.vertices;
    let weight = halved(
        weights.get(&first).copied().unwrap_or(0.0) + weights.get(&second).copied().unwrap_or(0.0),
    );
    if invert { 1.0 - weight } else { weight }
}
fn transfer_uv(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    factor: f64,
    _mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    let source_store = read_uv_store(source);
    let source_layers = &source_store.layers;
    let mut destination_layers = read_layers(destination);
    let destination_active = read_uv_store(destination).active;
    let source_selection = string(modifier, "layers_uv_select_src")?;
    let destination_selection = string(modifier, "layers_uv_select_dst")?;
    if matches!(source_selection, "BONE_SELECT" | "BONE_DEFORM") {
        return Ok(());
    }
    if !matches!(source_selection, "ACTIVE" | "ALL")
        && !source_layers
            .iter()
            .any(|layer| layer.name == source_selection)
    {
        return Err(invalid(
            modifier,
            "layers_uv_select_src",
            "ALL or an existing source UV layer",
        ));
    }
    let source_names = select_uv_layers(&source_store, source_selection);
    for source_name in source_names {
        let Some(source_index) = source_layers
            .iter()
            .position(|layer| layer.name == source_name)
        else {
            continue;
        };
        let destination_index = match destination_selection {
            "NAME" => ensure_uv_layer(
                &mut destination_layers,
                destination.faces.len(),
                &source_name,
                modifier,
            )?,
            "ACTIVE" => match destination_active
                .as_deref()
                .and_then(|name| {
                    destination_layers
                        .iter()
                        .position(|layer| layer.name == name)
                })
                .or((!destination_layers.is_empty()).then_some(0))
            {
                Some(index) => index,
                None => ensure_uv_layer(
                    &mut destination_layers,
                    destination.faces.len(),
                    &source_name,
                    modifier,
                )?,
            },
            "INDEX" => match source_index.cmp(&destination_layers.len()) {
                std::cmp::Ordering::Less => source_index,
                std::cmp::Ordering::Equal => push_uv_layer(
                    &mut destination_layers,
                    destination.faces.len(),
                    &source_name,
                    modifier,
                )?,
                std::cmp::Ordering::Greater => {
                    return Err(invalid(
                        modifier,
                        "layers_uv_select_dst",
                        "a destination layer at each source layer index",
                    ));
                }
            },
            _ => {
                if source_selection == "ALL" {
                    return Err(invalid(
                        modifier,
                        "layers_uv_select_dst",
                        "dynamic destination UV layers require a single source layer",
                    ));
                }
                match destination_layers
                    .iter()
                    .position(|layer| layer.name == destination_selection)
                {
                    Some(index) => index,
                    None => {
                        return Err(invalid(
                            modifier,
                            "layers_uv_select_dst",
                            "NAME, INDEX, or an existing destination UV layer",
                        ));
                    }
                }
            }
        };
        let source_layer = &source_layers[source_index];
        let target_layer = &mut destination_layers[destination_index];
        for (face_index, face) in destination.faces.iter().enumerate() {
            // Blender's zeroed UV is V-up 0, which Potter stores V-down as 1.
            let mut row = target_layer.faces[face_index]
                .clone()
                .unwrap_or_else(|| vec![[0.0, 1.0]; face.vertices.len()]);
            let mappings =
                uv_loop_mapping_for_face(mapping_context, destination, face_index, mapping);
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                let Some(mapped) = mappings[corner].as_ref() else {
                    continue;
                };
                let samples = mapped
                    .0
                    .iter()
                    .filter_map(|(loop_index, weight)| {
                        Some((
                            layer_uv_at_loop(source, source_layer, *loop_index)?,
                            *weight,
                        ))
                    })
                    .collect::<Vec<_>>();
                if samples.is_empty() {
                    continue;
                }
                let target = samples.iter().fold(DVec2::ZERO, |sum, (uv, weight)| {
                    sum + DVec2::from_array(*uv) * *weight
                });
                let mut mask = mask_weights.map_or(1.0, |weights| {
                    weights.get(vertex_id).copied().unwrap_or(0.0)
                });
                if invert_mask {
                    mask = 1.0 - mask;
                }
                let old = DVec2::from_array(row[corner]);
                row[corner] = old.lerp(target, factor * mask).to_array();
            }
            target_layer.faces[face_index] = Some(row);
        }
    }
    write_layers(destination, &destination_layers);
    Ok(())
}

fn transfer_custom_normals(
    destination: &mut Mesh,
    modifier: &Modifier,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    let source_normals = mapping_context
        .source_corner_normals
        .as_deref()
        .ok_or_else(|| invalid(modifier, "data_types_loops", "source corner normals"))?;
    let mut destination_normals = mapping_context
        .destination_corner_normals
        .clone()
        .ok_or_else(|| invalid(modifier, "data_types_loops", "destination corner normals"))?;
    let mut loop_index = 0;
    for (face_index, face) in destination.faces.iter().enumerate() {
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            let Some(mapped) =
                mapped_loop_sources(mapping_context, destination, face_index, corner, mapping)
            else {
                loop_index += 1;
                continue;
            };
            let mut source_normal = DVec3::ZERO;
            for &(source_loop, weight) in &mapped.0 {
                let Some(normal) = source_normals.get(source_loop) else {
                    continue;
                };
                source_normal += *normal * weight;
            }
            source_normal =
                mapping_context.normal_to_destination(source_normal.normalize_or_zero());
            let mask = transfer_mask_weight(mask_weights, *vertex_id, invert_mask);
            destination_normals[loop_index] = blend_normal(
                destination_normals[loop_index],
                source_normal,
                factor * mask,
                mix_mode,
            );
            loop_index += 1;
        }
    }
    let custom_vectors = destination_normals
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    crate::geom::clnor::encode_vectors(
        destination,
        Some(&custom_vectors),
        crate::geom::clnor::NormalCodeOptions::default(),
    )
}

fn transfer_typed_attribute(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    source_name: &str,
    destination_name: &str,
    mapping: &str,
    _factor: f64,
    _mix_mode: &str,
    mapping_context: &TransferMapContext<'_>,
) -> Result<()> {
    let domain = if source_name == "shade_smooth" {
        "face"
    } else {
        source
            .attributes
            .get(source_name)
            .and_then(|attribute| attribute.get("domain"))
            .and_then(Value::as_str)
            .unwrap_or("face")
    };
    if domain != "face" {
        return Err(invalid(
            modifier,
            source_name,
            "a face-domain source attribute",
        ));
    }
    let blender_index = destination
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .and_then(|layers| {
            layers.iter().position(|attribute| {
                attribute.get("name").and_then(Value::as_str) == Some(destination_name)
                    && attribute.get("domain").and_then(Value::as_str) == Some("FACE")
            })
        });
    let destination_exists = destination_name == "shade_smooth"
        || destination.attributes.contains_key(destination_name)
        || blender_index.is_some();
    if !destination_exists {
        return Ok(());
    }
    let mut output = destination
        .attributes
        .get(destination_name)
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut array_output = blender_index.and_then(|index| {
        destination
            .attributes
            .get("blender_attributes")?
            .as_array()?
            .get(index)?
            .get("values")?
            .as_array()
            .cloned()
    });
    for (face_index, face) in destination.faces.iter().enumerate() {
        let Some(mapped) = mapped_face_sources(mapping_context, destination, face_index, mapping)
        else {
            continue;
        };
        let samples = mapped
            .0
            .iter()
            .filter_map(|(source_index, weight)| {
                let source_face = source.faces.get(*source_index)?;
                Some((
                    source_face_value(source, source_name, *source_index, source_face.id)?,
                    *weight,
                ))
            })
            .collect::<Vec<_>>();
        let Some(value) = weighted_values(&samples) else {
            continue;
        };
        output.insert(format!("f{}", face.id), value.clone());
        if let Some(array) = array_output.as_mut()
            && let Some(slot) = array.get_mut(face_index)
        {
            *slot = value;
        }
    }
    if source_name == "shade_smooth" {
        destination.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"face","type":"bool","values":output}),
        );
    } else if let Some(index) = blender_index
        && let Some(array) = array_output
        && let Some(layers) = destination
            .attributes
            .get_mut("blender_attributes")
            .and_then(Value::as_array_mut)
        && let Some(layer) = layers.get_mut(index)
    {
        layer["values"] = Value::Array(array);
    } else {
        let kind = source
            .attributes
            .get(source_name)
            .and_then(|attribute| attribute.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("bool");
        destination.attributes.insert(
            destination_name.to_owned(),
            json!({"domain":"face","type":kind,"values":output}),
        );
    }
    Ok(())
}

fn face_center(mesh: &Mesh, face: &Face, vertex_indices: &BTreeMap<u32, usize>) -> DVec3 {
    let mut center = DVec3::ZERO;
    let mut count = 0_u32;
    for id in &face.vertices {
        if let Some(&index) = vertex_indices.get(id)
            && let Some(vertex) = mesh.vertices.get(index)
        {
            center += vertex.co;
            count += 1;
        }
    }
    if count > 0 {
        center / f64::from(count)
    } else {
        DVec3::ZERO
    }
}

fn blend_attribute(a: f64, b: f64, factor: f64, mode: &str, modifier: &Modifier) -> Result<f64> {
    match mode {
        "REPLACE" => Ok(a + (b - a) * factor),
        "ABOVE_THRESHOLD" => Ok(if a < factor { a } else { b }),
        "BELOW_THRESHOLD" => Ok(if a > factor { a } else { b }),
        "MIX" => {
            let target = halved(a + b);
            Ok(a + (target - a) * factor)
        }
        "ADD" => Ok(a + b * factor),
        "SUB" => Ok(a - b * factor),
        "MUL" => Ok(a * (1.0 + (b - 1.0) * factor)),
        _ => Err(invalid(
            modifier,
            "mix_mode",
            "REPLACE, ABOVE_THRESHOLD, BELOW_THRESHOLD, MIX, ADD, SUB, or MUL",
        )),
    }
}

#[derive(Clone, Debug, Default)]
struct MappedSources(Vec<(usize, f64)>);
#[derive(Clone, Debug)]
struct LoopCandidate {
    sources: MappedSources,
    source_face: usize,
    factor: f64,
    from_vertex: bool,
}
#[derive(Clone, Debug)]
struct SourceFaceLink {
    face: usize,
    is_cut: bool,
    cost: f32,
}

#[derive(Clone, Debug, Default)]
struct SourceIsland {
    faces: Vec<usize>,
    vertices: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct SourceTriangle {
    face: usize,
    vertices: [usize; 3],
}

struct SourceFaceGraph {
    face_islands: Vec<usize>,
    islands: Vec<SourceIsland>,
    face_centers: Vec<glam::Vec3>,
    face_normals: Vec<DVec3>,
    face_local_indices: Vec<usize>,
    vertex_faces: Vec<Vec<usize>>,
    links: Vec<Vec<SourceFaceLink>>,
    triangles: Vec<SourceTriangle>,
    triangle_bvh: KdopTree,
}

#[derive(Clone, Copy)]
struct AStarHeapNode {
    cost: f32,
    face: usize,
}

#[derive(Default)]
struct AStarHeap {
    nodes: Vec<AStarHeapNode>,
}

impl AStarHeap {
    fn insert(&mut self, cost: f32, face: usize) {
        let active = AStarHeapNode { cost, face };
        let mut index = self.nodes.len();
        self.nodes.push(active);
        while index > 0 {
            let parent = (index - 1) / 2;
            if cost >= self.nodes[parent].cost {
                break;
            }
            self.nodes[index] = self.nodes[parent];
            index = parent;
        }
        self.nodes[index] = active;
    }

    fn pop_min(&mut self) -> Option<usize> {
        let first = *self.nodes.first()?;
        let last = self.nodes.pop()?;
        if self.nodes.is_empty() {
            return Some(first.face);
        }
        let mut index = 0;
        self.nodes[index] = last;
        loop {
            let left = index * 2 + 1;
            let right = left + 1;
            let mut smallest = index;
            if left < self.nodes.len() && self.nodes[left].cost < last.cost {
                smallest = left;
            }
            if right < self.nodes.len() && self.nodes[right].cost < self.nodes[smallest].cost {
                smallest = right;
            }
            if smallest == index {
                break;
            }
            self.nodes[index] = self.nodes[smallest];
            index = smallest;
            self.nodes[index] = last;
        }
        Some(first.face)
    }
}

#[derive(Default)]
struct AStarWorkspace {
    queue: AStarHeap,
    costs: Vec<f32>,
    steps: Vec<usize>,
    previous: Vec<Option<usize>>,
    previous_cut: Vec<bool>,
    done: Vec<bool>,
    path: Vec<usize>,
    cuts: Vec<bool>,
}

impl AStarWorkspace {
    fn prepare(&mut self, node_count: usize) {
        self.queue.nodes.clear();
        self.costs.resize(node_count, f32::INFINITY);
        self.costs.fill(f32::INFINITY);
        self.steps.resize(node_count, 0);
        self.steps.fill(0);
        self.previous.resize(node_count, None);
        self.previous.fill(None);
        self.previous_cut.resize(node_count, false);
        self.previous_cut.fill(false);
        self.done.resize(node_count, false);
        self.done.fill(false);
        self.path.clear();
        self.cuts.clear();
    }
}

#[derive(Default)]
struct PolyProjectionWorkspace {
    weights: Vec<f32>,
    touched: Vec<usize>,
}

#[derive(Default)]
struct EdgeProjectionWorkspace {
    weights: Vec<f32>,
    touched: Vec<usize>,
}
struct TransferMapContext<'a> {
    source: &'a Mesh,
    source_to_destination_f32: [f32; 16],
    destination_to_source_f32: [f32; 16],
    source_positions: Vec<DVec3>,
    source_vertex_indices: BTreeMap<u32, usize>,
    destination_vertex_indices: BTreeMap<u32, usize>,
    source_edge_indices: BTreeMap<(u32, u32), usize>,
    source_vertex_edges: Vec<Vec<usize>>,
    source_loop_starts: Vec<usize>,
    destination_loop_starts: Vec<usize>,
    source_corner_normals: Option<Vec<DVec3>>,
    destination_corner_normals: Option<Vec<DVec3>>,
    destination_vertex_normals: Option<Vec<DVec3>>,
    source_vertex_bvh: KdopTree,
    source_edge_bvh: KdopTree,
    source_face_graph: Option<SourceFaceGraph>,
    astar_workspace: RefCell<AStarWorkspace>,
    max_distance: f64,
    ray_radius: f64,
    islands_precision: f64,
    source_layer_selection: &'a AttributeLayerSelection,
    destination_layer_selection: &'a AttributeLayerSelection,
    selected_bones: &'a std::collections::BTreeSet<String>,
    deforming_bones: &'a std::collections::BTreeSet<String>,
    poly_rng: RefCell<BlenderRng>,
    poly_face_map_cache: RefCell<BTreeMap<usize, Option<MappedSources>>>,
    poly_projection: RefCell<PolyProjectionWorkspace>,
    edge_projection: RefCell<EdgeProjectionWorkspace>,
}

impl<'a> TransferMapContext<'a> {
    fn new(
        source: &'a Mesh,
        destination: &Mesh,
        source_to_destination_f32: [f32; 16],
        destination_to_source_f32: [f32; 16],
        max_distance: f64,
        ray_radius: f64,
        islands_precision: f64,
        source_layer_selection: &'a AttributeLayerSelection,
        destination_layer_selection: &'a AttributeLayerSelection,
        selected_bones: &'a std::collections::BTreeSet<String>,
        deforming_bones: &'a std::collections::BTreeSet<String>,
        needs_loop_normals: bool,
        needs_vertex_normals: bool,
        needs_source_face_graph: bool,
    ) -> Result<Self> {
        let source_positions = source
            .vertices
            .iter()
            .map(|vertex| vertex.co.as_vec3().as_dvec3())
            .collect::<Vec<_>>();
        let source_vertex_indices = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let destination_vertex_indices = destination
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let source_edge_indices = source
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                let [first, second] = edge.vertices;
                ((first.min(second), first.max(second)), index)
            })
            .collect::<BTreeMap<_, _>>();
        let mut source_vertex_edges = vec![Vec::new(); source.vertices.len()];
        for (edge_index, edge) in source.edges.iter().enumerate() {
            for vertex_id in edge.vertices {
                if let Some(&vertex_index) = source_vertex_indices.get(&vertex_id)
                    && source_vertex_edges[vertex_index].last() != Some(&edge_index)
                {
                    source_vertex_edges[vertex_index].push(edge_index);
                }
            }
        }
        let source_loop_starts = face_loop_starts(source);
        let destination_loop_starts = face_loop_starts(destination);
        let ray_radius_f32 = ray_radius as f32;
        let max_ray_radius = f64::from(ray_radius_f32 * 25.0);
        if !ray_radius_f32.is_finite() || !max_ray_radius.is_finite() {
            return Err(crate::error::PotError::invalid_argument(
                "ray_radius is outside Blender's finite f32 range",
            ));
        }
        let vertex_bounds = source_positions
            .iter()
            .map(|&point| padded_leaf_bounds([point, point, point], 0.0))
            .collect::<Result<Vec<_>>>()?;
        let source_vertex_bvh = KdopTree::new(vertex_bounds, 2, 6);
        let edge_bounds = source
            .edges
            .iter()
            .map(|edge| {
                let [first, second] = edge
                    .vertices
                    .map(|id| source_vertex_indices.get(&id).copied());
                let [Some(first), Some(second)] = [first, second] else {
                    return Err(crate::error::PotError::invalid_argument(
                        "source edge references a missing vertex",
                    ));
                };
                let first = source_positions[first];
                let second = source_positions[second];
                // The segment's box is the box of the degenerate triangle (first, second, second).
                padded_leaf_bounds([first, second, second], max_ray_radius)
            })
            .collect::<Result<Vec<_>>>()?;
        let source_edge_bvh = KdopTree::new(edge_bounds, 2, 6);
        let source_corner_normals = if needs_loop_normals {
            Some(crate::geom::normals::corner_normals(
                source,
                &source_positions,
            )?)
        } else {
            None
        };
        let destination_positions = destination
            .vertices
            .iter()
            .map(|vertex| vertex.co.as_vec3().as_dvec3())
            .collect::<Vec<_>>();
        let destination_corner_normals = if needs_loop_normals {
            Some(crate::geom::normals::corner_normals(
                destination,
                &destination_positions,
            )?)
        } else {
            None
        };
        let destination_vertex_normals = if needs_vertex_normals {
            Some(vertex_normals(destination, &destination_vertex_indices))
        } else {
            None
        };
        let source_face_graph = if needs_source_face_graph {
            Some(build_source_face_graph(
                source,
                &source_positions,
                &source_vertex_indices,
                &source_edge_indices,
                max_ray_radius,
            )?)
        } else {
            None
        };
        Ok(Self {
            source,
            source_to_destination_f32,
            destination_to_source_f32,
            source_positions,
            source_vertex_indices,
            destination_vertex_indices,
            source_edge_indices,
            source_vertex_edges,
            source_loop_starts,
            destination_loop_starts,
            source_corner_normals,
            destination_corner_normals,
            destination_vertex_normals,
            source_vertex_bvh,
            source_edge_bvh,
            source_face_graph,
            astar_workspace: RefCell::new(AStarWorkspace::default()),
            max_distance,
            ray_radius,
            islands_precision,
            source_layer_selection,
            destination_layer_selection,
            selected_bones,
            deforming_bones,
            poly_rng: RefCell::new(BlenderRng::seed_random(0)),
            poly_face_map_cache: RefCell::new(BTreeMap::new()),
            poly_projection: RefCell::new(PolyProjectionWorkspace::default()),
            edge_projection: RefCell::new(EdgeProjectionWorkspace::default()),
        })
    }

    fn point_to_source(&self, point: DVec3) -> DVec3 {
        crate::geom::blender_matrix::blender_transform_point_f32(
            &self.destination_to_source_f32,
            point.as_vec3(),
        )
        .as_dvec3()
    }

    fn normal_to_source(&self, normal: DVec3) -> DVec3 {
        Self::blender_apply_normal_f32(&self.source_to_destination_f32, normal)
    }

    fn normal_to_destination(&self, normal: DVec3) -> DVec3 {
        Self::blender_invert_normal_f32(&self.source_to_destination_f32, normal)
    }

    fn source_corner_normal(&self, face_index: usize, corner: usize) -> Option<DVec3> {
        let index = self
            .source_loop_starts
            .get(face_index)?
            .checked_add(corner)?;
        self.source_corner_normals.as_ref()?.get(index).copied()
    }

    fn destination_corner_normal(&self, face_index: usize, corner: usize) -> Option<DVec3> {
        let index = self
            .destination_loop_starts
            .get(face_index)?
            .checked_add(corner)?;
        self.destination_corner_normals
            .as_ref()?
            .get(index)
            .copied()
    }

    fn source_face_graph(&self) -> Option<&SourceFaceGraph> {
        self.source_face_graph.as_ref()
    }

    fn blender_apply_normal_f32(matrix: &[f32; 16], normal: DVec3) -> DVec3 {
        let normal = normal.as_vec3();
        glam::Vec3::new(
            normal.x * matrix[0] + normal.y * matrix[4] + normal.z * matrix[8],
            normal.x * matrix[1] + normal.y * matrix[5] + normal.z * matrix[9],
            normal.x * matrix[2] + normal.y * matrix[6] + normal.z * matrix[10],
        )
        .normalize_or_zero()
        .as_dvec3()
    }

    fn blender_invert_normal_f32(matrix: &[f32; 16], normal: DVec3) -> DVec3 {
        let normal = normal.as_vec3();
        glam::Vec3::new(
            normal.x * matrix[0] + normal.y * matrix[1] + normal.z * matrix[2],
            normal.x * matrix[4] + normal.y * matrix[5] + normal.z * matrix[6],
            normal.x * matrix[8] + normal.y * matrix[9] + normal.z * matrix[10],
        )
        .normalize_or_zero()
        .as_dvec3()
    }
    fn destination_vertex_normal(&self, index: usize) -> Option<DVec3> {
        self.destination_vertex_normals
            .as_ref()?
            .get(index)
            .copied()
    }
}
const POLY_UNSET: u8 = 0;
const POLY_CENTER_INIT: u8 = 1;
const POLY_COMPLETE: u8 = 2;

fn add_astar_face_links(
    edge_index: usize,
    island_index: usize,
    is_cut: bool,
    edge_faces: &[Vec<usize>],
    face_islands: &[usize],
    face_local_indices: &[usize],
    face_centers: &[glam::Vec3],
    face_status: &mut [u8],
    done_edge_generations: &mut [usize],
    scratch: &mut Vec<Option<usize>>,
    links: &mut [Vec<SourceFaceLink>],
) {
    let incident = &edge_faces[edge_index];
    scratch.clear();
    scratch.resize(incident.len(), None);
    for (position, &face_index) in incident.iter().enumerate() {
        if face_islands[face_index] != island_index {
            continue;
        }
        let local_index = face_local_indices[face_index];
        if face_status[local_index] == POLY_COMPLETE {
            scratch[position] = Some(face_index);
            continue;
        }
        if face_status[local_index] == POLY_UNSET {
            face_status[local_index] = POLY_CENTER_INIT;
        }
        for previous in (0..position).rev() {
            let Some(other_face) = scratch[previous] else {
                continue;
            };
            if face_status[face_local_indices[other_face]] == POLY_COMPLETE {
                continue;
            }
            let cost = face_centers[other_face].distance(face_centers[face_index]);
            links[other_face].push(SourceFaceLink {
                face: face_index,
                is_cut,
                cost,
            });
            links[face_index].push(SourceFaceLink {
                face: other_face,
                is_cut,
                cost,
            });
        }
        scratch[position] = Some(face_index);
    }
    done_edge_generations[edge_index] = island_index;
}

fn build_source_face_graph(
    source: &Mesh,
    positions: &[DVec3],
    vertex_indices: &BTreeMap<u32, usize>,
    edge_indices: &BTreeMap<(u32, u32), usize>,
    ray_padding: f64,
) -> Result<SourceFaceGraph> {
    let mut edge_faces = vec![Vec::new(); source.edges.len()];
    let mut vertex_faces = vec![Vec::new(); source.vertices.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            if let Some(&vertex_index) = vertex_indices.get(vertex_id)
                && vertex_faces[vertex_index].last() != Some(&face_index)
            {
                vertex_faces[vertex_index].push(face_index);
            }
            if let Some(&next_id) = face.vertices.get((corner + 1) % face.vertices.len()) {
                let key = ((*vertex_id).min(next_id), (*vertex_id).max(next_id));
                if let Some(&edge_index) = edge_indices.get(&key)
                    && edge_faces[edge_index].last() != Some(&face_index)
                {
                    edge_faces[edge_index].push(face_index);
                }
            }
        }
    }
    let seam_edges = source
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| {
            source_edge_value(source, "seam", index, edge.id)
                .and_then(|value| value.as_bool().or_else(|| value.as_i64().map(|n| n != 0)))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    let mut parents = (0..source.faces.len()).collect::<Vec<_>>();
    for (edge_index, incident) in edge_faces.iter().enumerate() {
        if seam_edges[edge_index] {
            continue;
        }
        if let Some((&first, rest)) = incident.split_first() {
            for &other in rest {
                union_face_islands(&mut parents, first, other);
            }
        }
    }
    let mut root_to_island = BTreeMap::new();
    let mut islands = Vec::<SourceIsland>::new();
    let mut face_islands = Vec::with_capacity(source.faces.len());
    for face_index in 0..source.faces.len() {
        let root = find_face_island(&mut parents, face_index);
        let island_index = if let Some(&island) = root_to_island.get(&root) {
            island
        } else {
            let island = islands.len();
            islands.push(SourceIsland::default());
            root_to_island.insert(root, island);
            island
        };
        islands[island_index].faces.push(face_index);
        face_islands.push(island_index);
    }
    let mut island_vertices = vec![std::collections::BTreeSet::default(); islands.len()];
    for (face_index, face) in source.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            if let Some(&vertex_index) = vertex_indices.get(vertex_id) {
                island_vertices[face_islands[face_index]].insert(vertex_index);
            }
        }
    }
    for (island, vertices) in islands.iter_mut().zip(island_vertices) {
        island.vertices = vertices.into_iter().collect();
    }
    let face_centers = source
        .faces
        .iter()
        .map(|face| {
            let mut center = glam::Vec3::ZERO;
            let mut count = 0;
            for vertex_id in &face.vertices {
                if let Some(&index) = vertex_indices.get(vertex_id) {
                    let point = positions[index];
                    center += glam::Vec3::new(point.x as f32, point.y as f32, point.z as f32);
                    count += 1;
                }
            }
            if count == 0 {
                glam::Vec3::ZERO
            } else {
                center / count as f32
            }
        })
        .collect::<Vec<_>>();
    let face_normals = source
        .faces
        .iter()
        .map(|face| face_normal_at(face, positions, vertex_indices))
        .collect::<Vec<_>>();
    let mut links = vec![Vec::new(); source.faces.len()];
    let mut innercuts = vec![Vec::new(); islands.len()];
    let mut boundary_counts = BTreeMap::<(usize, usize), u8>::new();
    for (face_index, face) in source.faces.iter().enumerate() {
        let island_index = face_islands[face_index];
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            let Some(&next_id) = face.vertices.get((corner + 1) % face.vertices.len()) else {
                continue;
            };
            let key = ((*vertex_id).min(next_id), (*vertex_id).max(next_id));
            let Some(&edge_index) = edge_indices.get(&key) else {
                continue;
            };
            if !seam_edges[edge_index] {
                continue;
            }
            let count = boundary_counts
                .entry((island_index, edge_index))
                .or_default();
            if *count < 2 {
                *count += 1;
                if *count == 2 {
                    innercuts[island_index].push(edge_index);
                }
            }
        }
    }
    let mut face_local_indices = vec![usize::MAX; source.faces.len()];
    let mut max_island_faces = 0;
    for island in &islands {
        max_island_faces = max_island_faces.max(island.faces.len());
        for (local_index, &face_index) in island.faces.iter().enumerate() {
            face_local_indices[face_index] = local_index;
        }
    }
    let mut face_status = vec![POLY_UNSET; max_island_faces];
    let mut done_edge_generations = vec![usize::MAX; source.edges.len()];
    let max_edge_faces = edge_faces.iter().map(Vec::len).max().unwrap_or_default();
    let mut edge_face_scratch = Vec::with_capacity(max_edge_faces);
    for island_index in 0..islands.len() {
        face_status.resize(islands[island_index].faces.len(), POLY_UNSET);
        face_status.fill(POLY_UNSET);
        for edge_index in innercuts[island_index].iter().copied() {
            add_astar_face_links(
                edge_index,
                island_index,
                true,
                &edge_faces,
                &face_islands,
                &face_local_indices,
                &face_centers,
                &mut face_status,
                &mut done_edge_generations,
                &mut edge_face_scratch,
                &mut links,
            );
        }
        for face_index in islands[island_index].faces.iter().copied().rev() {
            let local_index = face_local_indices[face_index];
            if face_status[local_index] == POLY_COMPLETE {
                continue;
            }
            let face = &source.faces[face_index];
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                let Some(&next_id) = face.vertices.get((corner + 1) % face.vertices.len()) else {
                    continue;
                };
                let key = ((*vertex_id).min(next_id), (*vertex_id).max(next_id));
                let Some(&edge_index) = edge_indices.get(&key) else {
                    continue;
                };
                if done_edge_generations[edge_index] != island_index {
                    add_astar_face_links(
                        edge_index,
                        island_index,
                        false,
                        &edge_faces,
                        &face_islands,
                        &face_local_indices,
                        &face_centers,
                        &mut face_status,
                        &mut done_edge_generations,
                        &mut edge_face_scratch,
                        &mut links,
                    );
                }
            }
            face_status[local_index] = POLY_COMPLETE;
        }
    }
    let face_positions = source
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .filter_map(|vertex_id| vertex_indices.get(vertex_id))
                .map(|&index| {
                    let point = positions[index];
                    [point.x as f32, point.y as f32, point.z as f32]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let triangles: Vec<SourceTriangle> = crate::geom::triangulate::triangulate_mesh(
        &face_positions,
        crate::geom::triangulate::QuadMethod::Fixed,
        crate::geom::triangulate::NgonMethod::Clip,
        3,
    )
    .into_iter()
    .filter_map(|triangle| {
        let face = source.faces.get(triangle.face)?;
        if triangle.corners.len() != 3 {
            return None;
        }
        let vertex = |corner: usize| {
            face.vertices
                .get(corner)
                .and_then(|vertex_id| vertex_indices.get(vertex_id))
                .copied()
        };
        Some(SourceTriangle {
            face: triangle.face,
            vertices: [
                vertex(triangle.corners[0])?,
                vertex(triangle.corners[1])?,
                vertex(triangle.corners[2])?,
            ],
        })
    })
    .collect();
    let triangle_bounds = triangles
        .iter()
        .map(|triangle| {
            let [first, second, third] = triangle.vertices.map(|index| positions[index]);
            padded_leaf_bounds([first, second, third], ray_padding)
        })
        .collect::<Result<Vec<_>>>()?;
    let triangle_bvh = KdopTree::new(triangle_bounds, 2, 6);
    Ok(SourceFaceGraph {
        face_islands,
        islands,
        face_centers,
        face_normals,
        face_local_indices,
        vertex_faces,
        links,
        triangles,
        triangle_bvh,
    })
}

fn astar_face_path(
    graph: &SourceFaceGraph,
    island_index: usize,
    start: usize,
    destination: usize,
    max_steps: usize,
    workspace: &mut AStarWorkspace,
) -> bool {
    if start >= graph.links.len() || destination >= graph.links.len() {
        return false;
    }
    let Some(island) = graph.islands.get(island_index) else {
        return false;
    };
    let start_local = graph.face_local_indices[start];
    let destination_local = graph.face_local_indices[destination];
    if start_local >= island.faces.len() || destination_local >= island.faces.len() {
        return false;
    }
    workspace.prepare(island.faces.len());
    workspace.costs[start_local] = 0.0;
    workspace.queue.insert(
        graph.face_centers[start].distance(graph.face_centers[destination]),
        start_local,
    );
    while let Some(local_face) = workspace.queue.pop_min() {
        if workspace.done[local_face] {
            continue;
        }
        if max_steps > 0 && workspace.steps[local_face] > max_steps {
            continue;
        }
        if local_face == destination_local {
            workspace.path.clear();
            let mut current = destination_local;
            workspace.path.push(island.faces[current]);
            while current != start_local {
                let Some(previous) = workspace.previous[current] else {
                    return false;
                };
                current = previous;
                workspace.path.push(island.faces[current]);
            }
            workspace.path.reverse();
            workspace.cuts.resize(workspace.path.len(), false);
            workspace.cuts.fill(false);
            for index in 1..workspace.path.len() {
                let local_index = graph.face_local_indices[workspace.path[index]];
                workspace.cuts[index] = workspace.previous_cut[local_index];
            }
            return true;
        }
        workspace.done[local_face] = true;
        let face = island.faces[local_face];
        for link in &graph.links[face] {
            let next_local = graph.face_local_indices[link.face];
            if workspace.done[next_local] {
                continue;
            }
            let candidate_cost = workspace.costs[local_face] + link.cost;
            if candidate_cost < workspace.costs[next_local] {
                workspace.costs[next_local] = candidate_cost;
                workspace.steps[next_local] = workspace.steps[local_face] + 1;
                workspace.previous[next_local] = Some(local_face);
                workspace.previous_cut[next_local] = link.is_cut;
                workspace.queue.insert(
                    candidate_cost
                        + graph.face_centers[link.face].distance(graph.face_centers[destination]),
                    next_local,
                );
            }
        }
    }
    false
}

fn mapped_vertex_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_index: usize,
    mapping: &str,
) -> Option<MappedSources> {
    let vertex = destination.vertices.get(destination_index)?;
    let point = context.point_to_source(vertex.co);
    match mapping {
        "TOPOLOGY" => (context.source.vertices.len() == destination.vertices.len())
            .then(|| MappedSources(vec![(destination_index, 1.0)])),
        "NEAREST" => nearest_vertex(context, point).map(|index| MappedSources(vec![(index, 1.0)])),
        "EDGE_NEAREST" | "EDGEINTERP_NEAREST" => {
            let (edge_index, closest, _) = nearest_edge(context, point)?;
            let edge = context.source.edges.get(edge_index)?;
            let [Some(first), Some(second)] = edge
                .vertices
                .map(|id| context.source_vertex_indices.get(&id).copied())
            else {
                return None;
            };
            if mapping == "EDGE_NEAREST" {
                let source_index = if point.distance_squared(context.source_positions[first])
                    <= point.distance_squared(context.source_positions[second])
                {
                    first
                } else {
                    second
                };
                Some(MappedSources(vec![(source_index, 1.0)]))
            } else {
                let factor = segment_factor(
                    closest,
                    context.source_positions[first],
                    context.source_positions[second],
                );
                Some(MappedSources(vec![(first, 1.0 - factor), (second, factor)]))
            }
        }
        "POLY_NEAREST" | "POLYINTERP_NEAREST" => {
            let (face_index, hit, _) = nearest_surface(context, point)?;
            let face = context.source.faces.get(face_index)?;
            if mapping == "POLY_NEAREST" {
                let index = face
                    .vertices
                    .iter()
                    .filter_map(|id| context.source_vertex_indices.get(id).copied())
                    .min_by(|&first, &second| {
                        context.source_positions[first]
                            .distance_squared(hit)
                            .total_cmp(&context.source_positions[second].distance_squared(hit))
                    })?;
                Some(MappedSources(vec![(index, 1.0)]))
            } else {
                Some(polygon_vertex_weights(context, face, hit)?)
            }
        }
        "POLYINTERP_VNORPROJ" => {
            let normal = context
                .destination_vertex_normal(destination_index)
                .map_or(DVec3::Z, |normal| context.normal_to_source(normal));
            let (face_index, hit, _) = raycast_surface(context, point, normal)?;
            let face = context.source.faces.get(face_index)?;
            Some(polygon_vertex_weights(context, face, hit)?)
        }
        _ => None,
    }
}

fn mapped_edge_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_edge_index: usize,
    mapping: &str,
) -> Option<MappedSources> {
    let edge = destination.edges.get(destination_edge_index)?;
    let [first_id, second_id] = edge.vertices;
    let first_destination = context
        .destination_vertex_indices
        .get(&first_id)
        .and_then(|&index| destination.vertices.get(index))?
        .co;
    let second_destination = context
        .destination_vertex_indices
        .get(&second_id)
        .and_then(|&index| destination.vertices.get(index))?
        .co;
    let first = context.point_to_source(first_destination);
    let second = context.point_to_source(second_destination);
    let midpoint = (first + second) * 0.5;
    match mapping {
        "TOPOLOGY" => (context.source.edges.len() == destination.edges.len())
            .then(|| MappedSources(vec![(destination_edge_index, 1.0)])),
        "NEAREST" => {
            nearest_edge(context, midpoint).map(|(index, _, _)| MappedSources(vec![(index, 1.0)]))
        }
        "VERT_NEAREST" => {
            let source_vertex = [
                nearest_vertex(context, first)?,
                nearest_vertex(context, second)?,
            ];
            let first_distance = context.source_positions[source_vertex[0]].distance(first);
            let second_distance = context.source_positions[source_vertex[1]].distance(second);
            let mut best: Option<(usize, f64)> = None;
            for (source_index, hit_distance, other_destination) in [
                (source_vertex[0], first_distance, second_destination),
                (source_vertex[1], second_distance, first_destination),
            ] {
                for &edge_index in &context.source_vertex_edges[source_index] {
                    let Some(candidate) = context.source.edges.get(edge_index) else {
                        continue;
                    };
                    let [Some(first_source), Some(second_source)] = candidate
                        .vertices
                        .map(|id| context.source_vertex_indices.get(&id).copied())
                    else {
                        continue;
                    };
                    let other_source = if first_source == source_index {
                        second_source
                    } else if second_source == source_index {
                        first_source
                    } else {
                        continue;
                    };
                    let score = hit_distance
                        + context.source_positions[other_source].distance(other_destination);
                    if best.is_none_or(|(_, prior)| score < prior) {
                        best = Some((edge_index, score));
                    }
                }
            }
            best.map(|(index, _)| MappedSources(vec![(index, 1.0)]))
        }
        "POLY_NEAREST" => {
            let (face_index, _, _) = nearest_surface(context, midpoint)?;
            let face = context.source.faces.get(face_index)?;
            let mut best = None;
            for (corner, &first_id) in face.vertices.iter().enumerate() {
                let Some(&second_id) = face.vertices.get((corner + 1) % face.vertices.len()) else {
                    continue;
                };
                let key = (first_id.min(second_id), first_id.max(second_id));
                let Some(&edge_index) = context.source_edge_indices.get(&key) else {
                    continue;
                };
                let candidate = &context.source.edges[edge_index];
                let [Some(a), Some(b)] = candidate
                    .vertices
                    .map(|id| context.source_vertex_indices.get(&id).copied())
                else {
                    continue;
                };
                let center = (context.source_positions[a] + context.source_positions[b]) * 0.5;
                let distance = center.distance_squared(midpoint);
                if best.is_none_or(|(_, prior)| distance < prior) {
                    best = Some((edge_index, distance));
                }
            }
            best.map(|(index, _)| MappedSources(vec![(index, 1.0)]))
        }
        "EDGEINTERP_VNORPROJ" => projected_edge_sources(context, destination, edge),
        _ => None,
    }
}

fn mapped_loop_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_face_index: usize,
    destination_corner: usize,
    mapping: &str,
) -> Option<MappedSources> {
    if mapping == "TOPOLOGY" {
        let destination_total = destination
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        let source_total = context
            .source
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>();
        if destination_total != source_total {
            return None;
        }
        return Some(MappedSources(vec![(
            context
                .destination_loop_starts
                .get(destination_face_index)
                .copied()?
                .checked_add(destination_corner)?,
            1.0,
        )]));
    }
    let face = destination.faces.get(destination_face_index)?;
    let vertex_id = *face.vertices.get(destination_corner)?;
    let vertex_index = *context.destination_vertex_indices.get(&vertex_id)?;
    let vertex = destination.vertices.get(vertex_index)?;
    let point = context.point_to_source(vertex.co);
    let destination_normal = context
        .destination_corner_normal(destination_face_index, destination_corner)
        .unwrap_or_else(|| {
            destination_face_normal(destination, face, &context.destination_vertex_indices)
        });
    if matches!(mapping, "NEAREST_NORMAL" | "NEAREST_POLYNOR") {
        let nearest = nearest_vertex(context, point)?;
        let use_loop_normal = mapping == "NEAREST_NORMAL";
        let destination_center = context.point_to_source(face_center(
            destination,
            face,
            &context.destination_vertex_indices,
        ));
        let destination_normal = context.normal_to_source(destination_normal);
        let mut best: Option<(usize, f64, f64)> = None;
        let graph = context.source_face_graph()?;
        for &source_face_index in graph.vertex_faces.get(nearest)? {
            let source_face = context.source.faces.get(source_face_index)?;
            let loop_start = *context.source_loop_starts.get(source_face_index)?;
            let source_normal = graph.face_normals[source_face_index];
            let center_distance = f64::from(
                graph.face_centers[source_face_index]
                    .distance_squared(destination_center.as_vec3()),
            );
            for (corner, source_id) in source_face.vertices.iter().enumerate() {
                if context.source_vertex_indices.get(source_id).copied() != Some(nearest) {
                    continue;
                }
                let normal = if use_loop_normal {
                    context
                        .source_corner_normal(source_face_index, corner)
                        .unwrap_or(source_normal)
                } else {
                    source_normal
                };
                let score = normal.dot(destination_normal);
                let index = loop_start + corner;
                if best.is_none_or(|(_, prior_score, prior_distance)| {
                    score > prior_score + 1.0e-6
                        || ((score - prior_score).abs() <= 1.0e-6
                            && center_distance < prior_distance)
                }) {
                    best = Some((index, score, center_distance));
                }
            }
        }
        return best.map(|(index, _, _)| MappedSources(vec![(index, 1.0)]));
    }

    let (face_index, hit, _) = if mapping == "POLYINTERP_LNORPROJ" {
        let normal = context.normal_to_source(destination_normal);
        let attempts = if context.ray_radius > 0.0 { 3 } else { 1 };
        let mut ray_hit = None;
        let mut weight = 1.0;
        for _ in 0..attempts {
            let radius = if context.ray_radius > 0.0 {
                context.ray_radius / weight
            } else {
                0.0
            };
            if let Some(hit) = raycast_surface_with_radius(context, point, normal, radius) {
                ray_hit = Some(hit);
                break;
            }
            weight /= 5.0;
        }
        ray_hit.or_else(|| nearest_surface(context, point))?
    } else {
        nearest_surface(context, point)?
    };
    let source_face = context.source.faces.get(face_index)?;
    let loop_start = *context.source_loop_starts.get(face_index)?;
    if mapping == "NEAREST_POLY" {
        let corner = source_face
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(corner, id)| {
                context
                    .source_vertex_indices
                    .get(id)
                    .copied()
                    .map(|index| (corner, index))
            })
            .min_by(|(_, first), (_, second)| {
                context.source_positions[*first]
                    .distance_squared(hit)
                    .total_cmp(&context.source_positions[*second].distance_squared(hit))
            })?
            .0;
        Some(MappedSources(vec![(loop_start + corner, 1.0)]))
    } else {
        Some(polygon_loop_weights(context, source_face, hit, loop_start)?)
    }
}

fn uv_loop_mapping_for_face(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_face_index: usize,
    mapping: &str,
) -> Vec<Option<MappedSources>> {
    let Some(destination_face) = destination.faces.get(destination_face_index) else {
        return Vec::new();
    };
    if mapping == "TOPOLOGY" {
        return (0..destination_face.vertices.len())
            .map(|corner| {
                mapped_loop_sources(
                    context,
                    destination,
                    destination_face_index,
                    corner,
                    mapping,
                )
            })
            .collect();
    }
    let Some(graph) = context.source_face_graph() else {
        return vec![None; destination_face.vertices.len()];
    };
    let mut best_factor = 0.0;
    let mut best_candidates = None;
    let mut best_island_index = None;
    for island_index in 0..graph.islands.len() {
        let candidates = (0..destination_face.vertices.len())
            .map(|corner| {
                loop_candidate_for_island(
                    context,
                    destination,
                    destination_face_index,
                    corner,
                    mapping,
                    island_index,
                    graph,
                )
            })
            .collect::<Vec<_>>();
        let factor = candidates
            .iter()
            .flatten()
            .map(|candidate| candidate.factor)
            .sum::<f64>()
            / destination_face.vertices.len().max(1) as f64;
        if factor > best_factor {
            best_factor = factor;
            best_candidates = Some(candidates);
            best_island_index = Some(island_index);
        }
    }
    let Some(mut candidates) = best_candidates else {
        return vec![None; destination_face.vertices.len()];
    };
    let Some(best_island_index) = best_island_index else {
        return vec![None; destination_face.vertices.len()];
    };
    let max_steps = if context.islands_precision > 0.0 {
        ((64.0_f32 * context.islands_precision as f32 + 0.499_f32) as usize).max(1)
    } else {
        0
    };
    let mut previous_face = None;
    let mut output = Vec::with_capacity(candidates.len());
    for (corner, candidate) in candidates.iter_mut().enumerate() {
        let Some(candidate) = candidate else {
            output.push(None);
            continue;
        };
        if max_steps > 0
            && let Some(previous) = previous_face
            && previous != candidate.source_face
        {
            let mut workspace = context.astar_workspace.borrow_mut();
            if astar_face_path(
                graph,
                best_island_index,
                previous,
                candidate.source_face,
                max_steps,
                &mut workspace,
            ) {
                let mut corrected_face = None;
                for index in (1..workspace.path.len()).rev() {
                    if workspace.cuts[index] {
                        corrected_face = Some(workspace.path[index - 1]);
                    }
                }
                if let Some(corrected_face) = corrected_face {
                    let vertex_id = destination_face.vertices[corner];
                    if let Some(&vertex_index) = context.destination_vertex_indices.get(&vertex_id)
                        && let Some(vertex) = destination.vertices.get(vertex_index)
                    {
                        let point = context.point_to_source(vertex.co);
                        if candidate.from_vertex {
                            if let Some(source_loop) =
                                nearest_source_loop_on_face(context, corrected_face, point)
                            {
                                candidate.sources = MappedSources(vec![(source_loop, 1.0)]);
                                candidate.source_face = corrected_face;
                            }
                        } else if let Some(hit) =
                            nearest_point_on_source_face(context, graph, corrected_face, point)
                            && let Some(face) = context.source.faces.get(corrected_face)
                            && let Some(loop_start) =
                                context.source_loop_starts.get(corrected_face).copied()
                            && let Some(sources) =
                                polygon_loop_weights(context, face, hit, loop_start)
                        {
                            candidate.sources = sources;
                            candidate.source_face = corrected_face;
                        }
                    }
                }
            }
        }
        previous_face = Some(candidate.source_face);
        output.push(Some(candidate.sources.clone()));
    }
    output
}

fn loop_candidate_for_island(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_face_index: usize,
    destination_corner: usize,
    mapping: &str,
    island_index: usize,
    graph: &SourceFaceGraph,
) -> Option<LoopCandidate> {
    let destination_face = destination.faces.get(destination_face_index)?;
    let vertex_id = *destination_face.vertices.get(destination_corner)?;
    let vertex_index = *context.destination_vertex_indices.get(&vertex_id)?;
    let destination_vertex = destination.vertices.get(vertex_index)?;
    let point = context.point_to_source(destination_vertex.co);
    let destination_normal = context
        .destination_corner_normal(destination_face_index, destination_corner)
        .unwrap_or_else(|| {
            destination_face_normal(
                destination,
                destination_face,
                &context.destination_vertex_indices,
            )
        });
    if matches!(mapping, "NEAREST_NORMAL" | "NEAREST_POLYNOR") {
        let island = graph.islands.get(island_index)?;
        let nearest_vertex = island
            .vertices
            .iter()
            .copied()
            .filter(|&index| {
                context.source_positions[index].distance_squared(point)
                    <= context.max_distance * context.max_distance
            })
            .min_by(|&first, &second| {
                context.source_positions[first]
                    .distance_squared(point)
                    .total_cmp(&context.source_positions[second].distance_squared(point))
            })?;
        let destination_normal = context.normal_to_source(destination_normal);
        let destination_center = context
            .point_to_source(face_center(
                destination,
                destination_face,
                &context.destination_vertex_indices,
            ))
            .as_vec3();
        let use_loop_normal = mapping == "NEAREST_NORMAL";
        let mut best: Option<(usize, usize, f64, f64)> = None;
        for &source_face_index in &graph.vertex_faces[nearest_vertex] {
            if graph.face_islands[source_face_index] != island_index {
                continue;
            }
            let source_face = context.source.faces.get(source_face_index)?;
            let face_normal = graph.face_normals[source_face_index];
            for (corner, source_id) in source_face.vertices.iter().enumerate() {
                if context.source_vertex_indices.get(source_id).copied() != Some(nearest_vertex) {
                    continue;
                }
                let normal = if use_loop_normal {
                    context
                        .source_corner_normal(source_face_index, corner)
                        .unwrap_or(face_normal)
                } else {
                    face_normal
                };
                let score = normal.dot(destination_normal);
                let center_distance = f64::from(
                    graph.face_centers[source_face_index].distance_squared(destination_center),
                );
                if best.is_none_or(|(_, _, prior_score, prior_distance)| {
                    score > prior_score - 1e-6
                        && (score > prior_score + 1e-6 || center_distance < prior_distance)
                }) {
                    best = Some((source_face_index, corner, score, center_distance));
                }
            }
        }
        let (source_face, corner, score, _) = best?;
        let distance = context.source_positions[nearest_vertex].distance(point);
        let factor = if distance > 0.0 {
            halved(score + 1.0) / distance
        } else {
            1e18
        };
        return Some(LoopCandidate {
            sources: MappedSources(vec![(
                context
                    .source_loop_starts
                    .get(source_face)
                    .copied()?
                    .checked_add(corner)?,
                1.0,
            )]),
            source_face,
            factor,
            from_vertex: true,
        });
    }
    let (source_face, hit, _distance, factor) = if mapping == "POLYINTERP_LNORPROJ" {
        let normal = context.normal_to_source(destination_normal);
        let attempts = if context.ray_radius > 0.0 { 3 } else { 1 };
        let mut ray_hit = None;
        let mut weight = 1.0;
        for _ in 0..attempts {
            let radius = if context.ray_radius > 0.0 {
                context.ray_radius / weight
            } else {
                0.0
            };
            if let Some((face, hit, distance)) =
                raycast_surface_in_island(context, point, normal, radius, Some(island_index))
            {
                ray_hit = Some((face, hit, distance, weight));
                break;
            }
            weight /= 5.0;
        }
        if let Some((face, hit, distance, weight)) = ray_hit {
            (
                face,
                hit,
                distance,
                if distance > 0.0 {
                    weight / distance
                } else {
                    1e18 * weight
                },
            )
        } else {
            let (face, hit, distance_squared) =
                nearest_surface_in_island(context, point, Some(island_index))?;
            (face, hit, distance_squared.sqrt(), 0.0)
        }
    } else {
        let (face, hit, distance_squared) =
            nearest_surface_in_island(context, point, Some(island_index))?;
        let distance = distance_squared.sqrt();
        (
            face,
            hit,
            distance,
            if distance > 0.0 { 1.0 / distance } else { 1e18 },
        )
    };
    let face = context.source.faces.get(source_face)?;
    let sources = if mapping == "NEAREST_POLY" {
        let corner = face
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(corner, id)| {
                let vertex_index = context.source_vertex_indices.get(id).copied()?;
                Some((corner, vertex_index))
            })
            .min_by(|(_, first), (_, second)| {
                context.source_positions[*first]
                    .distance_squared(hit)
                    .total_cmp(&context.source_positions[*second].distance_squared(hit))
            })?
            .0;
        MappedSources(vec![(
            context
                .source_loop_starts
                .get(source_face)
                .copied()?
                .checked_add(corner)?,
            1.0,
        )])
    } else {
        polygon_loop_weights(
            context,
            face,
            hit,
            context.source_loop_starts.get(source_face).copied()?,
        )?
    };
    Some(LoopCandidate {
        sources,
        source_face,
        factor,
        from_vertex: false,
    })
}

fn nearest_source_loop_on_face(
    context: &TransferMapContext<'_>,
    face_index: usize,
    point: DVec3,
) -> Option<usize> {
    let face = context.source.faces.get(face_index)?;
    face.vertices
        .iter()
        .enumerate()
        .filter_map(|(corner, vertex_id)| {
            let vertex_index = context.source_vertex_indices.get(vertex_id).copied()?;
            Some((
                context
                    .source_loop_starts
                    .get(face_index)
                    .copied()?
                    .checked_add(corner)?,
                vertex_index,
            ))
        })
        .min_by(|(_, first), (_, second)| {
            context.source_positions[*first]
                .distance_squared(point)
                .total_cmp(&context.source_positions[*second].distance_squared(point))
        })
        .map(|(loop_index, _)| loop_index)
}

fn nearest_point_on_source_face(
    context: &TransferMapContext<'_>,
    graph: &SourceFaceGraph,
    face_index: usize,
    point: DVec3,
) -> Option<DVec3> {
    let point = point.as_vec3();
    graph
        .triangles
        .iter()
        .filter(|triangle| triangle.face == face_index)
        .map(|triangle| {
            let [a, b, c] = triangle
                .vertices
                .map(|index| context.source_positions[index].as_vec3());
            closest_point_triangle_blender(point, a, b, c).as_dvec3()
        })
        .min_by(|first, second| {
            first
                .as_vec3()
                .distance_squared(point)
                .total_cmp(&second.as_vec3().distance_squared(point))
        })
}
fn mapped_face_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    destination_face_index: usize,
    mapping: &str,
) -> Option<MappedSources> {
    if mapping == "POLYINTERP_PNORPROJ"
        && let Some(cached) = context
            .poly_face_map_cache
            .borrow()
            .get(&destination_face_index)
    {
        return cached.clone();
    }
    let face = destination.faces.get(destination_face_index)?;
    let point = context.point_to_source(face_center(
        destination,
        face,
        &context.destination_vertex_indices,
    ));
    let result = match mapping {
        "TOPOLOGY" => (context.source.faces.len() == destination.faces.len())
            .then(|| MappedSources(vec![(destination_face_index, 1.0)])),
        "NEAREST" => {
            nearest_surface(context, point).map(|(index, _, _)| MappedSources(vec![(index, 1.0)]))
        }
        "NORMAL" => {
            let normal = context.normal_to_source(destination_face_normal(
                destination,
                face,
                &context.destination_vertex_indices,
            ));
            raycast_surface(context, point, normal)
                .map(|(index, _, _)| MappedSources(vec![(index, 1.0)]))
        }
        "POLYINTERP_PNORPROJ" => projected_polygon_sources(context, destination, face),
        _ => None,
    };
    if mapping == "POLYINTERP_PNORPROJ" {
        context
            .poly_face_map_cache
            .borrow_mut()
            .insert(destination_face_index, result.clone());
    }
    result
}

fn projected_polygon_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    face: &Face,
) -> Option<MappedSources> {
    let positions = face
        .vertices
        .iter()
        .map(|id| {
            let index = *context.destination_vertex_indices.get(id)?;
            let vertex = destination.vertices.get(index)?;
            let point = context.point_to_source(vertex.co);
            Some(glam::Vec3::new(
                point.x as f32,
                point.y as f32,
                point.z as f32,
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    if positions.len() < 3 {
        return None;
    }
    let transformed_normal = context.normal_to_source(destination_face_normal(
        destination,
        face,
        &context.destination_vertex_indices,
    ));
    let normal = glam::Vec3::new(
        transformed_normal.x as f32,
        transformed_normal.y as f32,
        transformed_normal.z as f32,
    )
    .normalize_or_zero();
    if normal == glam::Vec3::ZERO {
        return None;
    }
    let horizontal_norm = normal.x * normal.x + normal.y * normal.y;
    let basis_x = if horizontal_norm > f32::EPSILON {
        glam::Vec3::new(normal.y, -normal.x, 0.0) / horizontal_norm.sqrt()
    } else {
        glam::Vec3::new(if normal.z < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0)
    };
    let basis_y = if horizontal_norm > f32::EPSILON {
        glam::Vec3::new(
            -normal.z * basis_x.y,
            normal.z * basis_x.x,
            normal.x * basis_x.y - normal.y * basis_x.x,
        )
    } else {
        glam::Vec3::Y
    };
    let projected = positions
        .iter()
        .map(|point| glam::Vec2::new(basis_x.dot(*point), basis_y.dot(*point)))
        .collect::<Vec<_>>();
    let center = positions.iter().copied().sum::<glam::Vec3>() / positions.len() as f32;
    let center_z = normal.dot(center);
    let (minimum, maximum) = projected.iter().fold(
        (
            glam::Vec2::splat(f32::INFINITY),
            glam::Vec2::splat(f32::NEG_INFINITY),
        ),
        |(minimum, maximum), point| (minimum.min(*point), maximum.max(*point)),
    );
    let size = maximum - minimum;
    let ray_radius = context.ray_radius as f32;
    let side = if ray_radius > 0.0 {
        ((size.max_element() / ray_radius + 0.5) as i32).clamp(4, 20)
    } else {
        4
    };
    let total_samples = (side * side) as usize;
    let face_positions = positions
        .iter()
        .map(|&point| point.to_array())
        .collect::<Vec<_>>();
    let triangles = crate::geom::triangulate::triangulate_mesh(
        &[face_positions],
        crate::geom::triangulate::QuadMethod::Fixed,
        crate::geom::triangulate::NgonMethod::Clip,
        3,
    );
    let triangle_corners = triangles
        .iter()
        .filter(|triangle| triangle.face == 0 && triangle.corners.len() == 3)
        .map(|triangle| {
            [
                triangle.corners[0],
                triangle.corners[1],
                triangle.corners[2],
            ]
        })
        .collect::<Vec<_>>();
    if triangle_corners.is_empty() {
        return None;
    }
    let triangle_area = |triangle: [usize; 3]| {
        let first = projected[triangle[0]];
        let second = projected[triangle[1]];
        let third = projected[triangle[2]];
        ((first.x - second.x) * (second.y - third.y) + (first.y - second.y) * (third.x - second.x))
            .abs()
            * 0.5
    };
    let mut polygon_cross_area = 0.0_f32;
    let mut previous = projected[projected.len() - 1];
    for current in &projected {
        polygon_cross_area += (previous.x - current.x) * (current.y + previous.y);
        previous = *current;
    }
    let polygon_area = (0.5 * polygon_cross_area).abs();
    let polygon_area_inverse = 1.0 / polygon_area.max(1e-9);
    let mut rng = context.poly_rng.borrow_mut();
    let mut weight_workspace = context.poly_projection.borrow_mut();
    weight_workspace
        .weights
        .resize(context.source.faces.len(), 0.0);
    weight_workspace.touched.clear();
    let mut done_area = 0.0_f32;
    let mut done_samples = 0_usize;
    for triangle in triangle_corners {
        done_area += triangle_area(triangle);
        let target_samples =
            (total_samples as f32 * done_area * polygon_area_inverse + 0.5) as usize;
        let sample_count = target_samples.saturating_sub(done_samples);
        done_samples = target_samples;
        let [first, second, third] = triangle;
        let [first_2d, second_2d, third_2d] =
            [projected[first], projected[second], projected[third]];
        for _ in 0..sample_count {
            let mut u = rng.get_float();
            let mut v = rng.get_float();
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let sample_2d = first_2d + (second_2d - first_2d) * u + (third_2d - first_2d) * v;
            let sample =
                (basis_x * sample_2d.x + basis_y * sample_2d.y + normal * center_z).as_dvec3();
            let attempts = if ray_radius > 0.0 { 3 } else { 1 };
            let mut approximate_weight = 1.0_f32;
            for _ in 0..attempts {
                let radius = if ray_radius > 0.0 {
                    ray_radius / approximate_weight
                } else {
                    0.0
                };
                if let Some((source_face, _, _)) = raycast_surface_with_radius(
                    context,
                    sample,
                    normal.as_dvec3(),
                    f64::from(radius),
                ) {
                    if weight_workspace.weights[source_face] == 0.0 {
                        weight_workspace.touched.push(source_face);
                    }
                    weight_workspace.weights[source_face] += approximate_weight;
                    break;
                }
                approximate_weight /= 5.0;
            }
        }
    }
    let total_weight = weight_workspace
        .touched
        .iter()
        .map(|&source_face| weight_workspace.weights[source_face])
        .sum::<f32>();
    if total_weight <= 0.0 {
        return None;
    }
    weight_workspace.touched.sort_unstable();
    let mut sources = Vec::with_capacity(weight_workspace.touched.len());
    for &source_face in &weight_workspace.touched {
        sources.push((
            source_face,
            f64::from(weight_workspace.weights[source_face] / total_weight),
        ));
    }
    while let Some(source_face) = weight_workspace.touched.pop() {
        weight_workspace.weights[source_face] = 0.0;
    }
    Some(MappedSources(sources))
}

fn blender_max_distance_squared(max_distance: f64) -> f64 {
    let max_distance = max_distance.min(f64::from(f32::MAX)) as f32;
    f64::from(max_distance * max_distance)
}

fn closest_point_triangle_blender(
    point: glam::Vec3,
    first: glam::Vec3,
    second: glam::Vec3,
    third: glam::Vec3,
) -> glam::Vec3 {
    let edge_ab = second - first;
    let edge_ac = third - first;
    let offset_a = point - first;
    let d1 = edge_ab.dot(offset_a);
    let d2 = edge_ac.dot(offset_a);
    if d1 <= 0.0 && d2 <= 0.0 {
        return first;
    }
    let offset_b = point - second;
    let d3 = edge_ab.dot(offset_b);
    let d4 = edge_ac.dot(offset_b);
    if d3 >= 0.0 && d4 <= d3 {
        return second;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let length_squared = d1 - d3;
        return if length_squared == 0.0 {
            first
        } else {
            first + edge_ab * (d1 / length_squared)
        };
    }
    let offset_c = point - third;
    let d5 = edge_ab.dot(offset_c);
    let d6 = edge_ac.dot(offset_c);
    if d6 >= 0.0 && d5 <= d6 {
        return third;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let length_squared = d2 - d6;
        return if length_squared == 0.0 {
            first
        } else {
            first + edge_ac * (d2 / length_squared)
        };
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let length_squared = (d4 - d3) + (d5 - d6);
        return if length_squared == 0.0 {
            second
        } else {
            second + (third - second) * ((d4 - d3) / length_squared)
        };
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    first + edge_ab * v + edge_ac * w
}
fn nearest_vertex(context: &TransferMapContext<'_>, point: DVec3) -> Option<usize> {
    let point = point.as_vec3();
    context
        .source_vertex_bvh
        .nearest(
            point.as_dvec3(),
            blender_max_distance_squared(context.max_distance),
            |index, _| {
                let position = context.source_positions.get(index)?.as_vec3();
                Some((f64::from(position.distance_squared(point)), ()))
            },
        )
        .map(|(index, _, ())| index)
}

fn nearest_edge(context: &TransferMapContext<'_>, point: DVec3) -> Option<(usize, DVec3, f64)> {
    let point_f32 = point.as_vec3();
    let max_distance_squared = blender_max_distance_squared(context.max_distance);
    let (index, distance, closest) = context.source_edge_bvh.nearest(
        point_f32.as_dvec3(),
        max_distance_squared,
        |index, _| {
            let edge = context.source.edges.get(index)?;
            let [a, b] = edge
                .vertices
                .map(|id| context.source_vertex_indices.get(&id).copied());
            let [Some(a), Some(b)] = [a, b] else {
                return None;
            };
            let first = context.source_positions.get(a)?.as_vec3();
            let second = context.source_positions.get(b)?.as_vec3();
            let edge_direction = second - first;
            let length_squared = edge_direction.length_squared();
            let factor = if length_squared == 0.0 {
                0.0
            } else {
                ((point_f32 - first).dot(edge_direction) / length_squared).clamp(0.0, 1.0)
            };
            let closest = first + edge_direction * factor;
            Some((
                f64::from(closest.distance_squared(point_f32)),
                closest.as_dvec3(),
            ))
        },
    )?;
    Some((index, closest, distance))
}

fn nearest_surface(context: &TransferMapContext<'_>, point: DVec3) -> Option<(usize, DVec3, f64)> {
    nearest_surface_in_island(context, point, None)
}

fn nearest_surface_in_island(
    context: &TransferMapContext<'_>,
    point: DVec3,
    island: Option<usize>,
) -> Option<(usize, DVec3, f64)> {
    let graph = context.source_face_graph()?;
    let point_f32 = point.as_vec3();
    let (triangle_index, distance, closest) = graph.triangle_bvh.nearest(
        point_f32.as_dvec3(),
        blender_max_distance_squared(context.max_distance),
        |triangle_index, _| {
            let triangle = graph.triangles.get(triangle_index)?;
            if island.is_some_and(|index| graph.face_islands[triangle.face] != index) {
                return None;
            }
            let [first, second, third] = triangle
                .vertices
                .map(|index| context.source_positions[index].as_vec3());
            let closest = closest_point_triangle_blender(point_f32, first, second, third);
            Some((
                f64::from(closest.distance_squared(point_f32)),
                closest.as_dvec3(),
            ))
        },
    )?;
    let face = graph.triangles.get(triangle_index)?.face;
    Some((face, closest, distance))
}

/// The leaf box of the triangle `points`, grown by `padding` on every side. The box carries the
/// tree epsilon from `KdopTree::leaf_bounds_for_triangle`; a point or segment is a degenerate
/// triangle of its corners.
fn padded_leaf_bounds(points: [DVec3; 3], padding: f64) -> Result<Bounds> {
    let mut bounds = kdop_bounds(
        &points.map(|point| point.as_vec3().to_array()),
        TREE_EPSILON,
        6,
    );
    let padding = padding as f32;
    for axis in 0..3 {
        bounds[2 * axis] -= padding;
        bounds[2 * axis + 1] += padding;
    }
    if !bounds.iter().all(|value| value.is_finite()) {
        return Err(crate::error::PotError::invalid_argument(
            "spatial primitive bounds must be finite",
        ));
    }
    Ok(bounds)
}

/// Nearest leaf hit of `test` along `origin + t * direction` for `t` in `[0, max_distance]`,
/// traversed with `KdopTree::ray_cast`. `test(leaf, t_max)` returns the leaf's hit distance and
/// payload, where `t_max` is the nearest distance found so far, clamped to `max_distance`. A later
/// leaf replaces the hit only when strictly nearer, as Blender's ray-cast callbacks do.
fn ray_cast_nearest<T>(
    tree: &KdopTree,
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    mut test: impl FnMut(usize, f64) -> Option<(f64, T)>,
) -> Option<(usize, f64, T)> {
    if !origin.is_finite()
        || !direction.is_finite()
        || direction == DVec3::ZERO
        || max_distance.is_nan()
        || max_distance < 0.0
    {
        return None;
    }
    let max_distance = max_distance.min(f64::from(f32::MAX));
    let mut nearest = None;
    tree.ray_cast(
        origin.as_vec3().to_array(),
        direction.as_vec3().to_array(),
        &mut RayHit::open(),
        &mut |leaf, _ray, running| {
            let t_max = f64::from(running.dist).min(max_distance);
            let Some((distance, payload)) = test(leaf, t_max) else {
                return;
            };
            let nearer = running.index < 0 || distance < f64::from(running.dist);
            if !distance.is_finite() || distance < 0.0 || distance > t_max || !nearer {
                return;
            }
            running.index = leaf as isize;
            running.dist = distance as f32;
            nearest = Some((leaf, distance, payload));
        },
    );
    nearest
}

fn raycast_surface(
    context: &TransferMapContext<'_>,
    origin: DVec3,
    normal: DVec3,
) -> Option<(usize, DVec3, f64)> {
    raycast_surface_with_radius(context, origin, normal, context.ray_radius)
}

fn raycast_surface_with_radius(
    context: &TransferMapContext<'_>,
    origin: DVec3,
    normal: DVec3,
    radius: f64,
) -> Option<(usize, DVec3, f64)> {
    raycast_surface_in_island(context, origin, normal, radius, None)
}

fn raycast_surface_in_island(
    context: &TransferMapContext<'_>,
    origin: DVec3,
    normal: DVec3,
    radius: f64,
    island: Option<usize>,
) -> Option<(usize, DVec3, f64)> {
    let normal = normal.as_vec3().as_dvec3();
    if normal == DVec3::ZERO {
        return None;
    }
    let graph = context.source_face_graph()?;
    let max_distance = context.max_distance.min(f64::from(f32::MAX));
    let mut best: Option<(usize, DVec3, f64)> = None;
    for direction in [normal, -normal] {
        let hit = ray_cast_nearest(
            &graph.triangle_bvh,
            origin,
            direction,
            max_distance,
            |index, max| {
                let triangle = graph.triangles.get(index)?;
                if island.is_some_and(|value| graph.face_islands[triangle.face] != value) {
                    return None;
                }
                let [a, b, c] = triangle
                    .vertices
                    .map(|vertex| context.source_positions[vertex].as_vec3());
                let origin_f32 = origin.as_vec3();
                let direction_f32 = direction.as_vec3();
                let hit = if radius > 0.0 {
                    sphere_ray_triangle(
                        origin_f32,
                        direction_f32,
                        max as f32,
                        radius as f32,
                        a,
                        b,
                        c,
                    )
                    .map(|distance| {
                        (
                            (origin_f32 + direction_f32 * distance).as_dvec3(),
                            f64::from(distance),
                        )
                    })
                } else {
                    ray_triangle(origin_f32, direction_f32, a, b, c)
                        .map(|(distance, point)| (point.as_dvec3(), f64::from(distance)))
                }?;
                (hit.1 >= 0.0 && hit.1 < max).then_some((hit.1, hit.0))
            },
        );
        if let Some((triangle_index, distance, point)) = hit {
            let face = graph.triangles.get(triangle_index)?.face;
            if best.is_none_or(|(_, _, prior)| distance < prior) {
                best = Some((face, point, distance));
            }
        }
    }
    best
}

fn sphere_ray_triangle(
    origin: glam::Vec3,
    direction: glam::Vec3,
    max_distance: f32,
    radius: f32,
    v0: glam::Vec3,
    v1: glam::Vec3,
    v2: glam::Vec3,
) -> Option<f32> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let velocity = direction * max_distance;
    let mut normal = edge1.cross(edge2).normalize_or_zero();
    if normal.dot(velocity) > 0.0 {
        normal = -normal;
    }
    let plane_distance = origin.dot(normal) - v0.dot(normal);
    let normal_velocity = normal.dot(velocity);
    let radius_squared = radius * radius;
    if normal_velocity.abs() < 0.000_001 {
        if plane_distance.abs() >= radius {
            return None;
        }
    } else {
        let mut first = (-plane_distance + radius) / normal_velocity;
        let mut second = (-plane_distance - radius) / normal_velocity;
        if first > second {
            std::mem::swap(&mut first, &mut second);
        }
        if first > 1.0 || second < 0.0 {
            return None;
        }
        first = first.clamp(0.0, 1.0);
        let point = origin + velocity * first - normal * radius;
        let a = edge1.dot(edge1);
        let b = edge1.dot(edge2);
        let c = edge2.dot(edge2);
        let offset = point - v0;
        let d = offset.dot(edge1);
        let e = offset.dot(edge2);
        let x = d * c - e * b;
        let y = e * a - d * b;
        let z = x + y - (a * c - b * b);
        if z <= 0.0 && x >= 0.0 && y >= 0.0 {
            return Some(first * max_distance);
        }
    }

    let mut lambda = 1.0;
    let mut found = false;
    let velocity_squared = velocity.dot(velocity);
    for vertex in [v0, v1, v2] {
        let offset = origin - vertex;
        let a = velocity_squared;
        let b = 2.0 * velocity.dot(offset);
        let c = offset.dot(offset) - radius_squared;
        if let Some(root) = lowest_sphere_root(a, b, c, lambda) {
            lambda = root;
            found = true;
        }
    }
    for (start, end, base) in [(v0, v1, v0), (v0, v2, v0), (v1, v2, v1)] {
        let edge = end - start;
        let edge_length_squared = edge.dot(edge);
        let edge_velocity = edge.dot(velocity);
        let base_offset = base - origin;
        let edge_base = edge.dot(base_offset);
        let a = edge_length_squared * -velocity_squared + edge_velocity * edge_velocity;
        let b = 2.0 * (edge_length_squared * velocity.dot(base_offset) - edge_velocity * edge_base);
        let c = edge_length_squared * (radius_squared - base_offset.dot(base_offset))
            + edge_base * edge_base;
        if let Some(root) = lowest_sphere_root(a, b, c, lambda) {
            let edge_factor = (edge_velocity * root - edge_base) / edge_length_squared;
            if (0.0..=1.0).contains(&edge_factor) {
                lambda = root;
                found = true;
            }
        }
    }
    found.then_some(lambda * max_distance)
}

fn lowest_sphere_root(a: f32, b: f32, c: f32, maximum: f32) -> Option<f32> {
    let determinant = b * b - 4.0 * a * c;
    if determinant < 0.0 {
        return None;
    }
    let root = determinant.sqrt();
    let mut first = (-b - root) / (2.0 * a);
    let mut second = (-b + root) / (2.0 * a);
    if first > second {
        std::mem::swap(&mut first, &mut second);
    }
    if first > 0.0 && first < maximum {
        Some(first)
    } else if second > 0.0 && second < maximum {
        Some(second)
    } else {
        None
    }
}

fn ray_triangle(
    origin: glam::Vec3,
    direction: glam::Vec3,
    first: glam::Vec3,
    second: glam::Vec3,
    third: glam::Vec3,
) -> Option<(f32, glam::Vec3)> {
    let edge1 = second - first;
    let edge2 = third - first;
    let p = direction.cross(edge2);
    let determinant = edge1.dot(p);
    if determinant == 0.0 {
        return None;
    }
    let inverse = 1.0 / determinant;
    let offset = origin - first;
    let u = inverse * offset.dot(p);
    let epsilon = f32::EPSILON;
    if u < -epsilon || u > 1.0 + epsilon {
        return None;
    }
    let q = offset.cross(edge1);
    let v = inverse * direction.dot(q);
    if v < -epsilon || u + v > 1.0 + epsilon {
        return None;
    }
    let distance = inverse * edge2.dot(q);
    if distance < 0.0 {
        return None;
    }
    Some((distance, origin + direction * distance))
}

fn polygon_vertex_weights(
    context: &TransferMapContext<'_>,
    face: &Face,
    point: DVec3,
) -> Option<MappedSources> {
    let weights = polygon_weights(context, face, point)?;
    let mut sources = Vec::with_capacity(weights.len());
    for (corner, weight) in weights.into_iter().enumerate() {
        let vertex_id = face.vertices.get(corner)?;
        let vertex_index = context.source_vertex_indices.get(vertex_id).copied()?;
        sources.push((vertex_index, weight));
    }
    Some(MappedSources(sources))
}

fn polygon_loop_weights(
    context: &TransferMapContext<'_>,
    face: &Face,
    point: DVec3,
    start: usize,
) -> Option<MappedSources> {
    let weights = polygon_weights(context, face, point)?;
    Some(MappedSources(
        weights
            .into_iter()
            .enumerate()
            .map(|(corner, weight)| (start + corner, weight))
            .collect(),
    ))
}

fn polygon_weights(
    context: &TransferMapContext<'_>,
    face: &Face,
    point: DVec3,
) -> Option<Vec<f64>> {
    let positions = face
        .vertices
        .iter()
        .map(|id| {
            context
                .source_vertex_indices
                .get(id)
                .map(|&index| context.source_positions[index])
        })
        .collect::<Option<Vec<_>>>()?;
    let count = positions.len();
    if count < 3 {
        return None;
    }
    let max_value = positions
        .iter()
        .map(|position| (*position - point).abs().max_element())
        .fold(0.0, f64::max);
    let epsilon = 16.0 * f64::from(f32::EPSILON) * max_value;
    let epsilon_squared = epsilon * epsilon;
    let directions = positions
        .iter()
        .map(|position| *position - point)
        .collect::<Vec<_>>();
    let lengths = directions
        .iter()
        .map(|direction| direction.length())
        .collect::<Vec<_>>();
    let mut weights = vec![0.0; count];
    if let Some(index) = lengths.iter().position(|length| *length < epsilon) {
        weights[index] = 1.0;
        return Some(weights);
    }
    for index in 0..count {
        let next = (index + 1) % count;
        let factor = segment_factor(point, positions[index], positions[next]);
        let closest = positions[index].lerp(positions[next], factor);
        if point.distance_squared(closest) < epsilon_squared {
            weights[index] = 1.0 - factor;
            weights[next] = factor;
            return Some(weights);
        }
    }
    let mut half_tangents = vec![0.0; count];
    for index in 0..count {
        let current = directions[index];
        let next = directions[(index + 1) % count];
        let area = current.cross(next).length();
        if area != 0.0 {
            let tangent =
                (lengths[index] * lengths[(index + 1) % count] - current.dot(next)) / area;
            if tangent.is_finite() {
                half_tangents[index] = tangent;
            }
        }
    }
    for index in 0..count {
        let previous = (index + count - 1) % count;
        weights[index] = (half_tangents[previous] + half_tangents[index]) / lengths[index];
    }
    let total = weights.iter().sum::<f64>();
    if total != 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    Some(weights)
}

fn projected_edge_sources(
    context: &TransferMapContext<'_>,
    destination: &Mesh,
    edge: &Edge,
) -> Option<MappedSources> {
    let [Some(first_index), Some(second_index)] = edge
        .vertices
        .map(|id| context.destination_vertex_indices.get(&id).copied())
    else {
        return None;
    };
    let first = context
        .point_to_source(destination.vertices.get(first_index)?.co)
        .as_vec3();
    let second = context
        .point_to_source(destination.vertices.get(second_index)?.co)
        .as_vec3();
    let first_normal = context
        .normal_to_source(context.destination_vertex_normal(first_index)?)
        .as_vec3();
    let second_normal = context
        .normal_to_source(context.destination_vertex_normal(second_index)?)
        .as_vec3();
    let ray_radius = context.ray_radius as f32;
    let grid_size = if ray_radius > 0.0 {
        ((first.distance(second) / ray_radius + 0.5) as i32).clamp(5, 100) as usize
    } else {
        5
    };
    let grid_step = 1.0_f32 / grid_size as f32;
    let attempts = if ray_radius > 0.0 { 3 } else { 1 };
    let mut workspace = context.edge_projection.borrow_mut();
    workspace.weights.resize(context.source.edges.len(), 0.0);
    workspace.touched.clear();
    let mut total_weight = 0.0_f32;
    for sample_index in 0..grid_size {
        let factor = grid_step * sample_index as f32;
        let origin = (first * (1.0 - factor) + second * factor).as_dvec3();
        let normal = blender_slerp_safe_f32(first_normal, second_normal, factor);
        if normal == glam::Vec3::ZERO {
            continue;
        }
        let mut approximate_weight = 1.0_f32;
        for _ in 0..attempts {
            let radius = if ray_radius > 0.0 {
                ray_radius / approximate_weight
            } else {
                0.0
            };
            let forward = raycast_source_edge(context, origin, normal.as_dvec3(), radius);
            let backward = raycast_source_edge(context, origin, (-normal).as_dvec3(), radius);
            let hit = match (forward, backward) {
                (Some(forward), Some(backward)) if backward.1 < forward.1 => Some(backward),
                (Some(forward), _) => Some(forward),
                (None, backward) => backward,
            };
            if let Some((source_edge, _)) = hit {
                if workspace.weights[source_edge] == 0.0 {
                    workspace.touched.push(source_edge);
                }
                workspace.weights[source_edge] += approximate_weight;
                total_weight += approximate_weight;
                break;
            }
            approximate_weight /= 5.0;
        }
    }
    if total_weight <= grid_size as f32 / 2.0 {
        while let Some(source_edge) = workspace.touched.pop() {
            workspace.weights[source_edge] = 0.0;
        }
        return None;
    }
    workspace.touched.sort_unstable();
    let mut sources = Vec::with_capacity(workspace.touched.len());
    for &source_edge in &workspace.touched {
        sources.push((
            source_edge,
            f64::from(workspace.weights[source_edge] / total_weight),
        ));
    }
    while let Some(source_edge) = workspace.touched.pop() {
        workspace.weights[source_edge] = 0.0;
    }
    Some(MappedSources(sources))
}

fn raycast_source_edge(
    context: &TransferMapContext<'_>,
    origin: DVec3,
    direction: DVec3,
    radius: f32,
) -> Option<(usize, f32)> {
    let max_distance = context.max_distance.min(f64::from(f32::MAX));
    ray_cast_nearest(
        &context.source_edge_bvh,
        origin,
        direction,
        max_distance,
        |index, max| {
            let edge = context.source.edges.get(index)?;
            let [Some(first), Some(second)] = edge
                .vertices
                .map(|id| context.source_vertex_indices.get(&id).copied())
            else {
                return None;
            };
            let first = context.source_positions.get(first)?.as_vec3();
            let second = context.source_positions.get(second)?.as_vec3();
            let (distance, _) = mesh_edge_spherecast(
                origin.as_vec3(),
                direction.as_vec3(),
                max as f32,
                radius,
                first,
                second,
            )?;
            Some((f64::from(distance), ()))
        },
    )
    .map(|(edge, distance, ())| (edge, distance as f32))
}

fn mesh_edge_spherecast(
    origin: glam::Vec3,
    direction: glam::Vec3,
    max_distance: f32,
    radius: f32,
    first: glam::Vec3,
    second: glam::Vec3,
) -> Option<(f32, glam::Vec3)> {
    let ray_end = origin + direction;
    if first == second {
        let ray_segment = ray_end - origin;
        let length_squared = ray_segment.length_squared();
        let factor = if length_squared == 0.0 {
            0.0
        } else {
            ((first - origin).dot(ray_segment) / length_squared).clamp(0.0, 1.0)
        };
        let hit = origin + ray_segment * factor;
        if (hit - origin).dot(ray_segment) >= 0.0 {
            let distance = hit.distance(origin);
            if distance < max_distance {
                return Some((distance, hit));
            }
        }
        return None;
    }

    let edge = second - first;
    let ray = ray_end - origin;
    let offset = origin - first;
    let normal = edge.cross(ray);
    let denominator = normal.length_squared();
    if denominator == 0.0 {
        return None;
    }
    let (mut edge_point, ray_point) = if offset.dot(normal).abs() <= 1.0e-6 {
        let cross = offset.cross(ray);
        let point = first + edge * (cross.dot(normal) / denominator);
        (point, point)
    } else {
        let shift = first - origin;
        let shift = normal * (shift.dot(normal) / denominator);
        let shifted_ray_start = origin + shift;
        let shifted_ray_end = ray_end + shift;
        let line_offset = shifted_ray_start - first;
        let shifted_ray = shifted_ray_end - shifted_ray_start;
        let cross = edge.cross(shifted_ray);
        let cross_offset = line_offset.cross(shifted_ray);
        let factor = cross_offset.dot(cross) / cross.length_squared();
        let first_point = first + edge * factor;
        (first_point, first_point - shift)
    };
    if (ray_point - origin).dot(ray) < 0.0 {
        return None;
    }
    let distance = ray_point.distance(origin);
    if distance >= max_distance {
        return None;
    }
    let edge_length_squared = edge.length_squared();
    let edge_factor = if edge_length_squared > 0.0 {
        (edge_point - first).dot(edge) / edge_length_squared
    } else {
        0.0
    };
    if edge_factor < 0.0 {
        edge_point = first;
    } else if edge_factor > 1.0 {
        edge_point = second;
    }
    ((edge_point - ray_point).length_squared() <= radius * radius).then_some((distance, ray_point))
}

fn blender_slerp_safe_f32(first: glam::Vec3, second: glam::Vec3, factor: f32) -> glam::Vec3 {
    if let Some(value) = blender_slerp_f32(first, second, factor) {
        return value;
    }
    let absolute = first.abs();
    let orthogonal = if absolute.x > absolute.y && absolute.x > absolute.z {
        glam::Vec3::new(-first.y - first.z, first.x, first.x)
    } else if absolute.y > absolute.z {
        glam::Vec3::new(first.y, -first.x - first.z, first.y)
    } else {
        glam::Vec3::new(first.z, first.z, -first.x - first.y)
    }
    .normalize_or_zero();
    if factor < 0.5 {
        blender_slerp_f32(first, orthogonal, factor * 2.0).unwrap_or(first)
    } else {
        blender_slerp_f32(orthogonal, second, (factor - 0.5) * 2.0).unwrap_or(second)
    }
}

fn blender_slerp_f32(first: glam::Vec3, second: glam::Vec3, factor: f32) -> Option<glam::Vec3> {
    let cosine = first.dot(second);
    if cosine < -1.0 + f32::EPSILON {
        return None;
    }
    let (first_weight, second_weight) = if cosine.abs() < 1.0 - 1.0e-4 {
        let angle = cosine.acos();
        let sine = angle.sin();
        (
            ((1.0 - factor) * angle).sin() / sine,
            (factor * angle).sin() / sine,
        )
    } else {
        (1.0 - factor, factor)
    };
    Some(first * first_weight + second * second_weight)
}

fn segment_factor(point: DVec3, first: DVec3, second: DVec3) -> f64 {
    let direction = second - first;
    let length = direction.length_squared();
    if length <= f64::EPSILON {
        0.0
    } else {
        ((point - first).dot(direction) / length).clamp(0.0, 1.0)
    }
}

fn polygon_normal_from_face(
    face: &Face,
    mut position: impl FnMut(u32) -> Option<glam::Vec3>,
) -> DVec3 {
    if face.vertices.len() < 3 {
        return DVec3::Z;
    }
    let mut normal = glam::Vec3::ZERO;
    let mut first: Option<glam::Vec3> = None;
    let mut previous: Option<glam::Vec3> = None;
    let mut count = 0;
    for &vertex_id in &face.vertices {
        let Some(current) = position(vertex_id) else {
            continue;
        };
        if let Some(previous) = previous {
            normal.x += (previous.y - current.y) * (previous.z + current.z);
            normal.y += (previous.z - current.z) * (previous.x + current.x);
            normal.z += (previous.x - current.x) * (previous.y + current.y);
        } else {
            first = Some(current);
        }
        previous = Some(current);
        count += 1;
    }
    if count < 3 {
        return DVec3::Z;
    }
    if let (Some(first), Some(previous)) = (first, previous) {
        normal.x += (previous.y - first.y) * (previous.z + first.z);
        normal.y += (previous.z - first.z) * (previous.x + first.x);
        normal.z += (previous.x - first.x) * (previous.y + first.y);
    }
    normal.normalize_or_zero().as_dvec3()
}

fn destination_face_normal(
    mesh: &Mesh,
    face: &Face,
    vertex_indices: &BTreeMap<u32, usize>,
) -> DVec3 {
    polygon_normal_from_face(face, |id| {
        vertex_indices
            .get(&id)
            .and_then(|&index| mesh.vertices.get(index))
            .map(|vertex| vertex.co.as_vec3())
    })
}

fn face_normal_at(
    face: &Face,
    positions: &[DVec3],
    vertex_indices: &BTreeMap<u32, usize>,
) -> DVec3 {
    polygon_normal_from_face(face, |id| {
        vertex_indices
            .get(&id)
            .and_then(|&index| positions.get(index))
            .map(|position| position.as_vec3())
    })
}

fn vertex_normals(mesh: &Mesh, vertex_indices: &BTreeMap<u32, usize>) -> Vec<DVec3> {
    let mut incident = vec![Vec::<(usize, usize)>::new(); mesh.vertices.len()];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            let Some(&vertex_index) = vertex_indices.get(vertex_id) else {
                continue;
            };
            if incident[vertex_index]
                .last()
                .is_none_or(|(prior_face, _)| *prior_face != face_index)
            {
                incident[vertex_index].push((face_index, corner));
            }
        }
    }
    let face_normals = mesh
        .faces
        .iter()
        .map(|face| destination_face_normal(mesh, face, vertex_indices).as_vec3())
        .collect::<Vec<_>>();
    let mut normals = vec![glam::Vec3::ZERO; mesh.vertices.len()];
    for (vertex_index, vertex) in mesh.vertices.iter().enumerate() {
        let position = vertex.co.as_vec3();
        if incident[vertex_index].is_empty() {
            normals[vertex_index] = position.normalize_or_zero();
            continue;
        }
        let mut normal = glam::Vec3::ZERO;
        for &(face_index, corner) in &incident[vertex_index] {
            let face = &mesh.faces[face_index];
            let count = face.vertices.len();
            let previous_id = face.vertices[(corner + count - 1) % count];
            let next_id = face.vertices[(corner + 1) % count];
            let Some(previous_index) = vertex_indices.get(&previous_id).copied() else {
                continue;
            };
            let Some(next_index) = vertex_indices.get(&next_id).copied() else {
                continue;
            };
            let Some(previous) = mesh
                .vertices
                .get(previous_index)
                .map(|vertex| vertex.co.as_vec3())
            else {
                continue;
            };
            let Some(next) = mesh
                .vertices
                .get(next_index)
                .map(|vertex| vertex.co.as_vec3())
            else {
                continue;
            };
            let direction_previous = (previous - position).normalize_or_zero();
            let direction_next = (next - position).normalize_or_zero();
            let angle = direction_previous
                .dot(direction_next)
                .clamp(-1.0, 1.0)
                .acos();
            normal += face_normals[face_index] * angle;
        }
        normals[vertex_index] = normal.normalize_or_zero();
    }
    normals.into_iter().map(glam::Vec3::as_dvec3).collect()
}

fn face_loop_starts(mesh: &Mesh) -> Vec<usize> {
    let mut starts = Vec::with_capacity(mesh.faces.len());
    let mut next = 0;
    for face in &mesh.faces {
        starts.push(next);
        next += face.vertices.len();
    }
    starts
}
fn loop_start(mesh: &Mesh, face_index: usize) -> usize {
    mesh.faces
        .iter()
        .take(face_index)
        .map(|face| face.vertices.len())
        .sum()
}

fn loop_face_index(mesh: &Mesh, loop_index: usize) -> usize {
    let mut start = 0;
    for (index, face) in mesh.faces.iter().enumerate() {
        start += face.vertices.len();
        if loop_index < start {
            return index;
        }
    }
    mesh.faces.len().saturating_sub(1)
}
fn ordered_group_names(mesh: &Mesh, groups: &Map<String, Value>) -> Vec<String> {
    if let Some(order) = mesh
        .attributes
        .get("vertex_group_order")
        .and_then(Value::as_array)
    {
        let ordered = order
            .iter()
            .filter_map(Value::as_str)
            .filter(|name| groups.contains_key(*name))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if !ordered.is_empty() {
            return ordered;
        }
    }
    groups.keys().cloned().collect()
}

fn weighted_vertex_value(
    source: &Mesh,
    values: &Map<String, Value>,
    mapped: &MappedSources,
) -> Option<Value> {
    let samples = mapped
        .0
        .iter()
        .filter_map(|(index, weight)| {
            let id = source.vertices.get(*index)?.id;
            Some((values.get(&format!("v{id}"))?.clone(), *weight))
        })
        .collect::<Vec<_>>();
    weighted_values(&samples)
}

fn weighted_values(samples: &[(Value, f64)]) -> Option<Value> {
    let first = &samples.first()?.0;
    if first.is_boolean() {
        return Some(Value::Bool(samples.iter().any(|(value, weight)| {
            value.as_bool() == Some(true) && *weight > 0.0
        })));
    }
    if first.as_f64().is_some() {
        return Some(json!(
            samples
                .iter()
                .map(|(value, weight)| value.as_f64().unwrap_or(0.0) * weight)
                .sum::<f64>()
        ));
    }
    if let Some(first_array) = first.as_array() {
        let values = (0..first_array.len())
            .map(|index| {
                let component_samples = samples
                    .iter()
                    .filter_map(|(value, weight)| {
                        Some((value.as_array()?.get(index)?.clone(), *weight))
                    })
                    .collect::<Vec<_>>();
                weighted_values(&component_samples)
            })
            .collect::<Option<Vec<_>>>()?;
        return Some(Value::Array(values));
    }
    samples
        .iter()
        .max_by(|(_, first), (_, second)| first.total_cmp(second))
        .map(|(value, _)| value.clone())
}

fn source_edge_value(source: &Mesh, name: &str, index: usize, edge_id: u32) -> Option<Value> {
    if matches!(name, "sharp" | "seam") {
        let value = source
            .attributes
            .get(name)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
            .and_then(|values| values.get(&format!("e{edge_id}")))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return Some(Value::Bool(value));
    }
    if let Some(value) = source
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values").or(Some(attribute)))
        .and_then(Value::as_object)
        .and_then(|values| values.get(&format!("e{edge_id}")))
    {
        return Some(value.clone());
    }
    source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|attribute| {
            attribute.get("name").and_then(Value::as_str) == Some(name)
                && attribute.get("domain").and_then(Value::as_str) == Some("EDGE")
        })
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_array)
        .and_then(|values| values.get(index))
        .cloned()
}

fn mix_color(
    destination: [f64; 4],
    source: [f64; 4],
    factor: f64,
    mode: &str,
    byte_color: bool,
) -> [f64; 4] {
    let factor = factor.clamp(0.0, 1.0);
    if byte_color {
        return mix_byte_color(destination, source, factor, mode);
    }
    if matches!(mode, "ABOVE_THRESHOLD" | "BELOW_THRESHOLD") {
        let mean = (destination[0] + destination[1] + destination[2]) / 3.0;
        let replace = if mode == "ABOVE_THRESHOLD" {
            mean >= factor
        } else {
            mean <= factor
        };
        return if replace { source } else { destination };
    }
    let mut target = destination;
    if source[3] != 0.0 {
        for channel in 0..3 {
            target[channel] = match mode {
                "MIX" => (1.0 - source[3]) * destination[channel] + source[channel],
                "ADD" => destination[channel] + source[channel] * destination[3],
                "SUB" => (destination[channel] - source[channel] * destination[3]).max(0.0),
                "MUL" => {
                    (1.0 - source[3]) * destination[channel]
                        + destination[channel] * source[channel] * destination[3]
                }
                _ => source[channel],
            };
        }
        target[3] = match mode {
            "MIX" => (1.0 - source[3]) * destination[3] + source[3],
            "ADD" | "SUB" | "MUL" => destination[3],
            _ => source[3],
        };
    } else if !matches!(mode, "MIX" | "ADD" | "SUB" | "MUL") {
        target = source;
    }
    std::array::from_fn(|index| destination[index] + (target[index] - destination[index]) * factor)
}

fn mix_byte_color(destination: [f64; 4], source: [f64; 4], factor: f64, mode: &str) -> [f64; 4] {
    let destination = linear_color_to_bytes(destination);
    let source = linear_color_to_bytes(source);
    if matches!(mode, "ABOVE_THRESHOLD" | "BELOW_THRESHOLD") {
        let mean = (u16::from(destination[0])
            + u16::from(destination[1])
            + u16::from(destination[2])) as f64
            / 3.0;
        let replace = if mode == "ABOVE_THRESHOLD" {
            mean >= factor
        } else {
            mean <= factor
        };
        return bytes_to_linear_color(if replace { source } else { destination });
    }
    let mut target = destination;
    let alpha = i64::from(source[3]);
    if alpha != 0 {
        match mode {
            "MIX" => {
                let mix = alpha;
                let inverse = 255 - mix;
                let total_alpha = inverse * i64::from(destination[3]) + mix * 255;
                for channel in 0..3 {
                    let numerator =
                        inverse * i64::from(destination[channel]) * i64::from(destination[3])
                            + mix * 255 * i64::from(source[channel]);
                    target[channel] = divide_round_i(numerator, total_alpha).clamp(0, 255) as u8;
                }
                target[3] = divide_round_i(total_alpha, 255).clamp(0, 255) as u8;
            }
            "ADD" | "SUB" => {
                for channel in 0..3 {
                    let numerator = i64::from(destination[channel]) * 255
                        + if mode == "ADD" {
                            i64::from(source[channel]) * alpha
                        } else {
                            -i64::from(source[channel]) * alpha
                        };
                    let blended = divide_round_i(numerator, 255);
                    target[channel] = if mode == "ADD" {
                        blended.min(255) as u8
                    } else {
                        blended.max(0) as u8
                    };
                }
            }
            "MUL" => {
                let mix = alpha;
                let inverse = 255 - mix;
                for channel in 0..3 {
                    let dst = i64::from(destination[channel]);
                    let numerator = inverse * dst * 255 + mix * dst * i64::from(source[channel]);
                    target[channel] = divide_round_i(numerator, 255 * 255).clamp(0, 255) as u8;
                }
            }
            _ => target = source,
        }
    } else if !matches!(mode, "MIX" | "ADD" | "SUB" | "MUL") {
        target = source;
    }
    let t = (255.0 * factor) as i64;
    let mt = 255 - t;
    let blended_alpha = mt * i64::from(destination[3]) + t * i64::from(target[3]);
    if blended_alpha <= 0 {
        target = destination;
        target[3] = 0;
    } else {
        for channel in 0..3 {
            let numerator = mt * i64::from(destination[channel]) * i64::from(destination[3])
                + t * i64::from(target[channel]) * i64::from(target[3]);
            target[channel] = divide_round_i(numerator, blended_alpha).clamp(0, 255) as u8;
        }
        target[3] = divide_round_i(blended_alpha, 255).clamp(0, 255) as u8;
    }
    bytes_to_linear_color(target)
}

fn linear_color_to_bytes(color: [f64; 4]) -> [u8; 4] {
    let srgb = |value: f64| {
        if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    };
    std::array::from_fn(|index| {
        let value = if index == 3 {
            color[index]
        } else {
            srgb(color[index])
        };
        (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
    })
}

fn bytes_to_linear_color(color: [u8; 4]) -> [f64; 4] {
    let linear = |value: f64| {
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    std::array::from_fn(|index| {
        let value = f64::from(color[index]) / 255.0;
        if index == 3 { value } else { linear(value) }
    })
}

fn divide_round_i(numerator: i64, denominator: i64) -> i64 {
    if numerator < 0 {
        -((-numerator + denominator / 2) / denominator)
    } else {
        (numerator + denominator / 2) / denominator
    }
}

fn transfer_color_layers(
    destination: &mut Mesh,
    source: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    factor: f64,
    mix_mode: &str,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mapping_context: &TransferMapContext<'_>,
    domain: &str,
) -> Result<()> {
    let (source_selector, destination_selector, color_domain) = if domain == "point" {
        (
            "layers_vcol_vert_select_src",
            "layers_vcol_vert_select_dst",
            ColorDomain::Point,
        )
    } else {
        (
            "layers_vcol_loop_select_src",
            "layers_vcol_loop_select_dst",
            ColorDomain::Corner,
        )
    };
    let source_selection = string(modifier, source_selector)?;
    let destination_selection = string(modifier, destination_selector)?;
    if matches!(source_selection, "BONE_SELECT" | "BONE_DEFORM") {
        return Ok(());
    }
    if !matches!(source_selection, "ACTIVE" | "ALL") {
        return Err(invalid(modifier, source_selector, "ALL or ACTIVE"));
    }
    if !matches!(destination_selection, "ACTIVE" | "NAME" | "INDEX") {
        return Err(invalid(
            modifier,
            destination_selector,
            "NAME, INDEX, or ACTIVE",
        ));
    }
    let source_store = color_layers::read(source);
    let mut destination_store = color_layers::read(destination);
    let source_indices = source_store
        .layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| (layer.domain == color_domain).then_some(index))
        .collect::<Vec<_>>();
    let destination_indices = destination_store
        .layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| (layer.domain == color_domain).then_some(index))
        .collect::<Vec<_>>();
    let active_name = |store: &color_layers::ColorStore| {
        store
            .active
            .as_deref()
            .filter(|name| {
                store
                    .layers
                    .iter()
                    .any(|layer| layer.domain == color_domain && layer.name.as_str() == *name)
            })
            .map(str::to_owned)
    };
    let source_active = active_name(&source_store);
    let destination_active = active_name(&destination_store);
    for (domain_index, &source_index) in source_indices.iter().enumerate() {
        let source_layer = &source_store.layers[source_index];
        let selected = match source_selection {
            "ACTIVE" => source_active.as_deref() == Some(source_layer.name.as_str()),
            "ALL" => true,
            _ => false,
        };
        if !selected {
            continue;
        }
        let destination_index = match destination_selection {
            "NAME" => destination_indices
                .iter()
                .copied()
                .find(|&index| destination_store.layers[index].name == source_layer.name),
            "ACTIVE" => destination_active.as_deref().and_then(|name| {
                destination_indices
                    .iter()
                    .copied()
                    .find(|&index| destination_store.layers[index].name == name)
            }),
            "INDEX" => destination_indices.get(domain_index).copied(),
            _ => None,
        };
        let Some(destination_index) = destination_index else {
            continue;
        };
        let byte_color = color_layer_is_byte(
            destination,
            &destination_store.layers[destination_index].name,
        );
        let destination_layer = &mut destination_store.layers[destination_index];
        if color_domain == ColorDomain::Point {
            for (index, vertex) in destination.vertices.iter().enumerate() {
                let Some(mapped) =
                    mapped_vertex_sources(mapping_context, destination, index, mapping)
                else {
                    continue;
                };
                let Some(target) = weighted_color(&source_layer.values, &mapped) else {
                    continue;
                };
                let Some(old) = destination_layer.values.get(index).copied() else {
                    continue;
                };
                let mask = transfer_mask_weight(mask_weights, vertex.id, invert_mask);
                destination_layer.values[index] =
                    mix_color(old, target, factor * mask, mix_mode, byte_color);
            }
        } else {
            let mut destination_loop = 0;
            for (face_index, face) in destination.faces.iter().enumerate() {
                for corner in 0..face.vertices.len() {
                    let mapped = mapped_loop_sources(
                        mapping_context,
                        destination,
                        face_index,
                        corner,
                        mapping,
                    );
                    if let Some(mapped) = mapped
                        && let Some(target) = weighted_color(&source_layer.values, &mapped)
                        && let Some(old) = destination_layer.values.get(destination_loop).copied()
                    {
                        let mask =
                            transfer_mask_weight(mask_weights, face.vertices[corner], invert_mask);
                        destination_layer.values[destination_loop] =
                            mix_color(old, target, factor * mask, mix_mode, byte_color);
                    }
                    destination_loop += 1;
                }
            }
        }
    }
    color_layers::write(destination, &destination_store);
    Ok(())
}

fn color_layer_is_byte(mesh: &Mesh, name: &str) -> bool {
    mesh.attributes
        .get("blender_metadata")
        .and_then(|metadata| metadata.get("color_data_types"))
        .and_then(|data_types| data_types.get(name))
        .and_then(Value::as_str)
        == Some("BYTE_COLOR")
}

fn weighted_color(values: &[[f64; 4]], mapped: &MappedSources) -> Option<[f64; 4]> {
    let mut result = [0.0; 4];
    for &(index, weight) in &mapped.0 {
        let color = values.get(index)?;
        for channel in 0..4 {
            result[channel] += color[channel] * weight;
        }
    }
    Some(result)
}

fn transfer_mask_weight(
    mask_weights: Option<&BTreeMap<u32, f64>>,
    vertex_id: u32,
    invert: bool,
) -> f64 {
    let Some(weights) = mask_weights else {
        return 1.0;
    };
    let weight = weights.get(&vertex_id).copied().unwrap_or(0.0);
    if invert { 1.0 - weight } else { weight }
}
fn layer_uv_at_loop(mesh: &Mesh, layer: &UvLayer, loop_index: usize) -> Option<[f64; 2]> {
    let face_index = loop_face_index(mesh, loop_index);
    let corner = loop_index.checked_sub(loop_start(mesh, face_index))?;
    layer.faces.get(face_index)?.as_ref()?.get(corner).copied()
}

/// Index of the UV layer named `name`, appended when no layer has that name.
fn ensure_uv_layer(
    layers: &mut Vec<UvLayer>,
    face_count: usize,
    name: &str,
    modifier: &Modifier,
) -> Result<usize> {
    match layers.iter().position(|layer| layer.name == name) {
        Some(index) => Ok(index),
        None => push_uv_layer(layers, face_count, name, modifier),
    }
}

/// Appends an empty UV layer named `name`, within the layer limit and under a name no layer has.
fn push_uv_layer(
    layers: &mut Vec<UvLayer>,
    face_count: usize,
    name: &str,
    modifier: &Modifier,
) -> Result<usize> {
    if layers.len() >= crate::geom::uv_layers::MAX_LAYERS {
        return Err(invalid(
            modifier,
            "layers_uv_select_dst",
            "at most 8 UV layers per mesh",
        ));
    }
    if layers.iter().any(|layer| layer.name == name) {
        return Err(invalid(
            modifier,
            "layers_uv_select_dst",
            "a destination layer name that no layer already has",
        ));
    }
    layers.push(UvLayer::new(name, vec![None; face_count]));
    Ok(layers.len() - 1)
}

fn select_uv_layers(store: &UvStore, selection: &str) -> Vec<String> {
    match selection {
        "ACTIVE" => store
            .layers
            .iter()
            .filter(|layer| Some(layer.name.as_str()) == store.active.as_deref())
            .map(|layer| layer.name.clone())
            .collect(),
        "ALL" => store
            .layers
            .iter()
            .map(|layer| layer.name.clone())
            .collect(),
        _ => store
            .layers
            .iter()
            .filter(|layer| layer.name == selection)
            .map(|layer| layer.name.clone())
            .collect(),
    }
}

fn blend_normal(first: DVec3, second: DVec3, factor: f64, mode: &str) -> DVec3 {
    if matches!(mode, "ABOVE_THRESHOLD" | "BELOW_THRESHOLD") {
        return second.normalize_or_zero();
    }
    let target = match mode {
        "MIX" | "ADD" => first + second,
        "SUB" => first - second,
        "MUL" => first * second,
        _ => second,
    }
    .normalize_or_zero();
    slerp_normal(first.normalize_or_zero(), target, factor.clamp(0.0, 1.0))
}

fn slerp_normal(first: DVec3, second: DVec3, factor: f64) -> DVec3 {
    let dot = first.dot(second).clamp(-1.0, 1.0);
    if dot < -1.0 + f64::from(f32::EPSILON) {
        let absolute = first.abs();
        let orthogonal = if absolute.x > absolute.y && absolute.x > absolute.z {
            DVec3::new(-first.y - first.z, first.x, first.x)
        } else if absolute.y > absolute.z {
            DVec3::new(first.y, -first.x - first.z, first.y)
        } else {
            DVec3::new(first.z, first.z, -first.x - first.y)
        }
        .normalize_or_zero();
        return if factor < 0.5 {
            slerp_normal(first, orthogonal, factor * 2.0)
        } else {
            slerp_normal(orthogonal, second, (factor - 0.5) * 2.0)
        };
    }
    let angle = dot.acos();
    if !angle.is_finite() || angle.abs() < 1.0e-8 {
        return first.lerp(second, factor).normalize_or_zero();
    }
    let sin_angle = angle.sin();
    (first * (((1.0 - factor) * angle).sin() / sin_angle)
        + second * ((factor * angle).sin() / sin_angle))
        .normalize_or_zero()
}

fn find_face_island(parents: &mut [usize], mut index: usize) -> usize {
    while parents[index] != index {
        parents[index] = parents[parents[index]];
        index = parents[index];
    }
    index
}

fn union_face_islands(parents: &mut [usize], first: usize, second: usize) {
    let first = find_face_island(parents, first);
    let second = find_face_island(parents, second);
    if first != second {
        let root = first.min(second);
        parents[first.max(second)] = root;
    }
}
fn source_face_value(source: &Mesh, name: &str, index: usize, face_id: u32) -> Option<Value> {
    if name == "shade_smooth" {
        let value = source
            .attributes
            .get("shade_smooth")
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
            .and_then(|values| values.get(&format!("f{face_id}")))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return Some(Value::Bool(value));
    }
    if let Some(value) = source
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values").or(Some(attribute)))
        .and_then(Value::as_object)
        .and_then(|values| values.get(&format!("f{face_id}")))
    {
        return Some(value.clone());
    }
    source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|attribute| {
            attribute.get("name").and_then(Value::as_str) == Some(name)
                && attribute.get("domain").and_then(Value::as_str) == Some("FACE")
        })
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_array)
        .and_then(|values| values.get(index))
        .cloned()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use glam::DVec3;
    use serde_json::json;

    use super::data_transfer;
    use crate::{
        geom::{
            Mesh,
            modifiers::attributes::{
                AttributeLayerSelection, SceneOperand, SceneOperandMatricesF32,
            },
        },
        model::{Id, Modifier, ModifierRuntime},
    };

    #[test]
    fn transferred_sharp_edge_splits_smooth_corner_normals()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut destination = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(0.0, -1.0, 1.0),
            ],
            vec![vec![0, 1, 2], vec![1, 0, 3]],
        )?;
        let shared_edge = destination
            .edges
            .iter()
            .find(|edge| edge.vertices.contains(&0) && edge.vertices.contains(&1))
            .ok_or("shared edge missing")?
            .id;
        let mut smooth_faces = serde_json::Map::new();
        smooth_faces.insert("f0".to_owned(), json!(true));
        smooth_faces.insert("f1".to_owned(), json!(true));
        destination.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"face","type":"bool","values":smooth_faces}),
        );
        let mut destination_sharp = serde_json::Map::new();
        for edge in &destination.edges {
            destination_sharp.insert(format!("e{}", edge.id), json!(false));
        }
        destination.attributes.insert(
            "sharp".to_owned(),
            json!({"domain":"edge","type":"bool","values":destination_sharp}),
        );
        let mut source = destination.clone();
        source.attributes["sharp"]["values"]
            .as_object_mut()
            .ok_or("sharp edge values must be an object")?
            .insert(format!("e{shared_edge}"), json!(true));

        let modifier = Modifier {
            baked_ocean_cache: None,
            id: Id::new("transfer_sharp")?,
            modifier_type: "data_transfer".to_owned(),
            name: "Data Transfer".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: json!({
                "object":"source",
                "use_edge_data":true,
                "data_types_edges":["SHARP_EDGE"],
                "edge_mapping":"TOPOLOGY"
            })
            .as_object()
            .ok_or("modifier params must be an object")?
            .clone(),
            binding_data: None,
            runtime: ModifierRuntime::default(),
        };
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        let source_operand = SceneOperand {
            mesh: source,
            local_to_subject: None,
            blender_matrices_f32: Some(SceneOperandMatricesF32 {
                world: None,
                local_to_subject: Some(identity),
                subject_to_local: Some(identity),
            }),
            camera: None,
            armature: false,
            object_scale: DVec3::ONE,
            selected_bones: std::collections::BTreeSet::default(),
            deforming_bones: std::collections::BTreeSet::default(),
            layer_selection: AttributeLayerSelection::default(),
        };
        let operands = BTreeMap::from([("object".to_owned(), vec![source_operand])]);
        let transferred = data_transfer(
            &destination,
            &modifier,
            &operands,
            &AttributeLayerSelection::default(),
        )?;
        let sharp = transferred.attributes["sharp"]["values"]
            .get(format!("e{shared_edge}"))
            .and_then(serde_json::Value::as_bool)
            .ok_or("transferred sharp edge is missing")?;
        assert!(sharp);

        let positions = transferred
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let normals = crate::geom::normals::corner_normals(&transferred, &positions)?;
        assert!(normals[0].dot(normals[4]) < 0.99);
        assert!(normals[1].dot(normals[3]) < 0.99);
        Ok(())
    }
}
