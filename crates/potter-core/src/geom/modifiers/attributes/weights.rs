//! Vertex-weight edit/mix/proximity and their texture-mask weights.

use super::{
    BTreeMap, DMat4, DVec2, DVec3, ErrorCode, Map, Mesh, Modifier, PotError, Result, SceneOperand,
    Value, boolean, closest_point_triangle, curve_points, falloff_type, get_group_weights, invalid,
    json, number, optional_string, string, string_set,
};
use crate::blender_rng::BlenderRng;

pub(super) fn texture_mask_weights(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture: Option<&crate::geom::ResolvedTexture>,
    texture_mapping: Option<&crate::model::TextureCoordinateMapping>,
    rng: &mut crate::blender_rng::BlenderRng,
) -> Result<Option<BTreeMap<u32, f64>>> {
    if modifier.modifier_type == "vertex_weight_proximity"
        && (optional_string(modifier, "target")?.is_none()
            || operands
                .get("target")
                .and_then(|objects| objects.first())
                .is_none())
    {
        return Ok(None);
    }
    let Some(texture_id) = modifier.params.get("mask_texture") else {
        return Ok(None);
    };
    if texture_id.is_null() || texture_id.as_str().is_some_and(str::is_empty) {
        return Ok(None);
    }
    let texture = texture.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "weight modifier texture was not resolved",
            json!({"modifier_id":modifier.id,"texture_id":texture_id}),
        )
    })?;
    // Only image textures are supported as vertex-weight masks; a Blender legacy texture is a typed
    // unsupported feature rather than being sampled without a Blender oracle.
    if !matches!(
        texture.source(),
        crate::geom::legacy_texture::TextureSource::Image(_)
            | crate::geom::legacy_texture::TextureSource::ImageWithSettings(..)
    ) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "weight modifier texture is not a resolved image texture",
            json!({
                "feature_id": format!("modifier.{}.texture_type", modifier.modifier_type),
                "modifier_id": modifier.id,
            }),
        ));
    }
    let mapping = string(modifier, "mask_tex_mapping")?;
    let channel = string(modifier, "mask_tex_use_channel")?;
    let uv_layer = string(modifier, "mask_tex_uv_layer")?;
    if !matches!(mapping, "LOCAL" | "GLOBAL" | "OBJECT" | "UV") {
        return Err(invalid(
            modifier,
            "mask_tex_mapping",
            "LOCAL, GLOBAL, OBJECT, or UV",
        ));
    }
    if !matches!(
        channel,
        "INT" | "RED" | "GREEN" | "BLUE" | "HUE" | "SAT" | "VAL" | "ALPHA"
    ) {
        return Err(invalid(
            modifier,
            "mask_tex_use_channel",
            "a Blender image channel",
        ));
    }
    let coordinates = texture_coordinates(mesh, modifier, mapping, uv_layer, texture_mapping)?;
    let source = texture.source();
    let mut weights = BTreeMap::new();
    for (vertex, coordinate) in mesh.vertices.iter().zip(coordinates) {
        let sample = crate::geom::legacy_texture::sample(source, coordinate, rng)?;
        let value = match channel {
            "INT" => sample.intensity,
            "ALPHA" => sample.alpha,
            _ => texture_channel(
                [sample.rgb[0], sample.rgb[1], sample.rgb[2], sample.alpha],
                channel,
            ),
        };
        weights.insert(vertex.id, value);
    }
    Ok(Some(weights))
}

/// Per-vertex mask-texture coordinates, aligned with `mesh.vertices`. `mapping` is one of
/// LOCAL, GLOBAL, OBJECT, or UV, validated by the caller. Non-UV coordinates come from the
/// evaluation context's `TextureCoordinateMapping`, which already resolves the OBJECT and BONE
/// reference (a missing bone uses the object world, as Blender's `MOD_util.cc` does).
fn texture_coordinates(
    mesh: &Mesh,
    modifier: &Modifier,
    mapping: &str,
    uv_layer: &str,
    texture_mapping: Option<&crate::model::TextureCoordinateMapping>,
) -> Result<Vec<DVec3>> {
    if mapping == "UV" {
        let uv_coordinates = vertex_uv_coordinates(mesh, modifier, uv_layer)?;
        return Ok(mesh
            .vertices
            .iter()
            .map(|vertex| {
                uv_coordinates
                    .get(&vertex.id)
                    .copied()
                    .map_or(vertex.co, |uv| {
                        DVec3::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0)
                    })
            })
            .collect());
    }
    let texture_mapping = texture_mapping.ok_or_else(|| {
        invalid(
            modifier,
            "mask_tex_mapping",
            "a texture coordinate mapping resolved by the evaluation context",
        )
    })?;
    let global = mapping == "GLOBAL";
    Ok(mesh
        .vertices
        .iter()
        .map(|vertex| {
            if global {
                texture_mapping.global_coordinate(vertex.co)
            } else {
                texture_mapping.object_coordinate(vertex.co)
            }
        })
        .collect())
}

fn vertex_uv_coordinates(
    mesh: &Mesh,
    modifier: &Modifier,
    requested_layer: &str,
) -> Result<BTreeMap<u32, DVec2>> {
    let store = crate::geom::uv_layers::read(mesh);
    let layers = store.layers;
    if layers.is_empty() {
        return Ok(BTreeMap::new());
    }
    let active_layer = store
        .active
        .as_deref()
        .filter(|name| layers.iter().any(|layer| layer.name == *name))
        .unwrap_or(layers[0].name.as_str());
    let layer_name = if layers.iter().any(|layer| layer.name == requested_layer) {
        requested_layer
    } else {
        active_layer
    };
    let Some(layer) = layers.iter().find(|layer| layer.name == layer_name) else {
        return Ok(BTreeMap::new());
    };
    let mut coordinates = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, DVec2::ZERO))
        .collect::<BTreeMap<_, _>>();
    let mut assigned = std::collections::BTreeSet::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            if !coordinates.contains_key(vertex_id) {
                return Err(invalid(
                    modifier,
                    "mask_tex_uv_layer",
                    "a mesh with valid face vertices",
                ));
            }
            if !assigned.insert(*vertex_id) {
                continue;
            }
            let uv = layer
                .faces
                .get(face_index)
                .and_then(Option::as_ref)
                .and_then(|values| values.get(corner))
                .copied()
                .unwrap_or([0.0, 0.0]);
            coordinates.insert(*vertex_id, DVec2::from_array(uv));
        }
    }
    Ok(coordinates)
}
fn texture_channel(color: [f64; 4], channel: &str) -> f64 {
    let [red, green, blue, alpha] = color;
    let maximum = red.max(green).max(blue);
    let minimum = red.min(green).min(blue);
    let delta = maximum - minimum;
    let hue = if delta <= f64::EPSILON {
        0.0
    } else if red >= green && red >= blue {
        ((green - blue) / delta).rem_euclid(6.0) / 6.0
    } else if green >= blue {
        ((blue - red) / delta + 2.0) / 6.0
    } else {
        ((red - green) / delta + 4.0) / 6.0
    };
    match channel {
        "RED" => red,
        "GREEN" => green,
        "BLUE" => blue,
        "HUE" => hue,
        "SAT" => {
            if maximum > f64::EPSILON {
                delta / maximum
            } else {
                0.0
            }
        }
        "VAL" => maximum,
        "ALPHA" => alpha,
        _ => 0.2126_f64.mul_add(red, 0.7152_f64.mul_add(green, 0.0722 * blue)),
    }
}

#[expect(
    clippy::map_entry,
    reason = "the presence check selects the remove branch, so it cannot become an entry call"
)]
pub(super) fn vertex_weight_edit(
    mesh: &Mesh,
    modifier: &Modifier,
    texture_weights: Option<&BTreeMap<u32, f64>>,
    subject_name: &str,
) -> Result<Mesh> {
    let group = string(modifier, "vertex_group")?;
    let Some(existing) = existing_group_weights(mesh, group)? else {
        return Ok(mesh.clone());
    };
    let falloff = falloff_type(modifier)?;
    let invert_falloff = boolean(modifier, "invert_falloff")?;
    let use_add = boolean(modifier, "use_add")?;
    let use_remove = boolean(modifier, "use_remove")?;
    let add_threshold = number(modifier, "add_threshold")?;
    let remove_threshold = number(modifier, "remove_threshold")?;
    let default = number(modifier, "default_weight")?;
    let normalize = boolean(modifier, "normalize")?;
    let mask_constant = number(modifier, "mask_constant")?;
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group")?;
    let mask_weights = mask_group
        .map(|name| existing_group_weights(mesh, name))
        .transpose()?
        .flatten();
    let curve = curve_points(modifier)?;
    let mut random = BlenderRng::seed_random(object_name_hash(subject_name));
    let mut values = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let original = existing.get(&vertex.id).copied().unwrap_or(default);
        let mut mapped = apply_falloff(original, falloff, curve.as_deref(), &mut random);
        if invert_falloff {
            mapped = 1.0 - mapped;
        }
        values.push(mask_weight(
            original,
            mapped,
            vertex.id,
            texture_weights,
            mask_weights.as_ref(),
            mask_invert,
            mask_constant,
        ));
    }
    if normalize {
        normalize_weights(&mut values);
    } else {
        clamp_weights(&mut values);
    }
    let mut updated = existing;
    for (vertex, weight) in mesh.vertices.iter().zip(values) {
        if updated.contains_key(&vertex.id) {
            if use_remove && weight <= remove_threshold {
                updated.remove(&vertex.id);
            } else {
                updated.insert(vertex.id, weight);
            }
        } else if use_add && weight >= add_threshold {
            updated.insert(vertex.id, weight);
        }
    }
    let mut result = mesh.clone();
    write_group_weights_preserving_zero(&mut result, group, &updated)?;
    Ok(result)
}

pub(super) fn vertex_weight_mix(
    mesh: &Mesh,
    modifier: &Modifier,
    texture_weights: Option<&BTreeMap<u32, f64>>,
) -> Result<Mesh> {
    let group_a = string(modifier, "vertex_group_a")?;
    let group_b = string(modifier, "vertex_group_b")?;
    let Some(weights_a) = existing_group_weights(mesh, group_a)? else {
        return Ok(mesh.clone());
    };
    let weights_b = if group_b.is_empty() {
        None
    } else {
        let Some(weights) = existing_group_weights(mesh, group_b)? else {
            return Ok(mesh.clone());
        };
        Some(weights)
    };
    let mode = string(modifier, "mix_mode")?;
    let mix_set = string(modifier, "mix_set")?;
    let default_a = number(modifier, "default_weight_a")?;
    let default_b = number(modifier, "default_weight_b")?;
    let invert_a = boolean(modifier, "invert_vertex_group_a")?;
    let invert_b = boolean(modifier, "invert_vertex_group_b")?;
    let normalize = boolean(modifier, "normalize")?;
    let mask_constant = number(modifier, "mask_constant")?;
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group")?;
    let mask_weights = mask_group
        .map(|name| existing_group_weights(mesh, name))
        .transpose()?
        .flatten();
    let mut updated = weights_a.clone();
    let mut selected_values = Vec::new();
    let mut selected_ids = Vec::new();
    for vertex in &mesh.vertices {
        let a_assigned = weights_a.contains_key(&vertex.id);
        let b_assigned = weights_b
            .as_ref()
            .is_some_and(|weights| weights.contains_key(&vertex.id));
        let selected = match mix_set {
            "ALL" => true,
            "A" => a_assigned,
            "B" => b_assigned,
            "OR" => a_assigned || b_assigned,
            "AND" => a_assigned && b_assigned,
            _ => return Err(invalid(modifier, "mix_set", "ALL, A, B, OR, or AND")),
        };
        if !selected {
            continue;
        }
        let raw_a = weights_a.get(&vertex.id).copied().unwrap_or(default_a);
        let raw_b = weights_b
            .as_ref()
            .and_then(|weights| weights.get(&vertex.id).copied())
            .unwrap_or(default_b);
        let a = if invert_a { 1.0 - raw_a } else { raw_a };
        let b = if invert_b { 1.0 - raw_b } else { raw_b };
        let mixed = mix_weights(a, b, mode, modifier)?;
        let blended = mask_weight(
            a,
            mixed,
            vertex.id,
            texture_weights,
            mask_weights.as_ref(),
            mask_invert,
            mask_constant,
        );
        selected_ids.push(vertex.id);
        selected_values.push(blended);
    }
    if normalize {
        normalize_weights(&mut selected_values);
    } else {
        clamp_weights(&mut selected_values);
    }
    for (id, weight) in selected_ids.into_iter().zip(selected_values) {
        updated.insert(id, weight);
    }
    let mut result = mesh.clone();
    write_group_weights_preserving_zero(&mut result, group_a, &updated)?;
    Ok(result)
}

fn mix_weights(a: f64, mut b: f64, mode: &str, modifier: &Modifier) -> Result<f64> {
    if mode == "DIV" {
        const ZERO_FLOOR: f64 = 1.0e-32;
        if b < 0.0 && b > -ZERO_FLOOR {
            b = -ZERO_FLOOR;
        } else if (0.0..ZERO_FLOOR).contains(&b) {
            b = ZERO_FLOOR;
        }
    }
    match mode {
        "SET" => Ok(b),
        "ADD" => Ok(a + b),
        "SUB" => Ok(a - b),
        "MUL" => Ok(a * b),
        "DIV" => Ok(a / b),
        "DIF" => Ok((a - b).abs()),
        "AVG" => Ok(f64::midpoint(a, b)),
        "MIN" => Ok(a.min(b)),
        "MAX" => Ok(a.max(b)),
        _ => Err(invalid(
            modifier,
            "mix_mode",
            "SET, ADD, SUB, MUL, DIV, DIF, AVG, MIN, or MAX",
        )),
    }
}

pub(super) fn vertex_weight_proximity(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture_weights: Option<&BTreeMap<u32, f64>>,
    subject_world: DMat4,
    subject_name: &str,
) -> Result<Mesh> {
    if optional_string(modifier, "target")?.is_none() {
        return Ok(mesh.clone());
    }
    let Some(target) = operands.get("target").and_then(|objects| objects.first()) else {
        return Ok(mesh.clone());
    };
    let group = string(modifier, "vertex_group")?;
    let Some(existing) = existing_group_weights(mesh, group)? else {
        return Ok(mesh.clone());
    };
    let mode = string(modifier, "proximity_mode")?;
    let geometry = string_set(modifier, "proximity_geometry")?;
    if !matches!(mode, "OBJECT" | "GEOMETRY") {
        return Err(invalid(modifier, "proximity_mode", "OBJECT or GEOMETRY"));
    }
    if geometry
        .iter()
        .any(|kind| !matches!(kind.as_str(), "VERTEX" | "EDGE" | "FACE"))
    {
        return Err(invalid(
            modifier,
            "proximity_geometry",
            "a set containing VERTEX, EDGE, or FACE",
        ));
    }
    let minimum = number(modifier, "min_dist")?;
    let maximum = number(modifier, "max_dist")?;
    let falloff = falloff_type(modifier)?;
    let invert_falloff = boolean(modifier, "invert_falloff")?;
    let curve = curve_points(modifier)?;
    let mask_constant = number(modifier, "mask_constant")?;
    let mask_group = optional_string(modifier, "mask_vertex_group")?;
    let mask_invert = boolean(modifier, "invert_mask_vertex_group")?;
    let mask_weights = mask_group
        .map(|name| existing_group_weights(mesh, name))
        .transpose()?
        .flatten();
    let matrices = target.blender_matrices_f32.ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "proximity target Blender matrices are missing",
        )
    })?;
    let use_origin_distance = mode == "OBJECT" || (mode == "GEOMETRY" && geometry.is_empty());
    let origin_distance = if use_origin_distance {
        let target_world = matrices.world.as_ref().ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "proximity target Blender world matrix is missing",
            )
        })?;
        let subject_world_f32 = weight_matrix_f32(subject_world);
        let subject_origin = glam::Vec3::new(
            subject_world_f32[12],
            subject_world_f32[13],
            subject_world_f32[14],
        );
        let target_origin = glam::Vec3::new(target_world[12], target_world[13], target_world[14]);
        Some((
            subject_world_f32,
            target_origin,
            f64::from((subject_origin - target_origin).length()),
        ))
    } else {
        None
    };
    let target_inverse = if mode == "GEOMETRY" && !geometry.is_empty() {
        Some(matrices.subject_to_local.ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "proximity target subject-to-local Blender matrix is missing",
            )
        })?)
    } else {
        None
    };
    let triangles = if mode == "GEOMETRY" && geometry.iter().any(|kind| kind == "FACE") {
        let positions = target
            .mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co.as_vec3().as_dvec3()))
            .collect::<BTreeMap<_, _>>();
        target
            .mesh
            .tessellate()
            .map_err(|error| {
                PotError::invalid_argument(format!(
                    "invalid proximity target triangulation: {error}"
                ))
            })?
            .into_iter()
            .filter_map(|triangle| {
                Some([
                    *positions.get(&triangle[0])?,
                    *positions.get(&triangle[1])?,
                    *positions.get(&triangle[2])?,
                ])
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let target_vertices = target
        .mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3().as_dvec3())
        .collect::<Vec<_>>();
    let target_edges = target
        .mesh
        .edges
        .iter()
        .filter_map(|edge| {
            Some([
                target
                    .mesh
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == edge.vertices[0])?
                    .co
                    .as_vec3()
                    .as_dvec3(),
                target
                    .mesh
                    .vertices
                    .iter()
                    .find(|vertex| vertex.id == edge.vertices[1])?
                    .co
                    .as_vec3()
                    .as_dvec3(),
            ])
        })
        .collect::<Vec<_>>();
    let mut random = BlenderRng::seed_random(object_name_hash(subject_name));
    let mut updated = existing.clone();
    let mut selected_ids = Vec::with_capacity(existing.len());
    let mut selected_values = Vec::with_capacity(existing.len());
    for vertex in &mesh.vertices {
        let Some(original) = existing.get(&vertex.id).copied() else {
            continue;
        };
        let distance = match mode {
            "OBJECT" => origin_distance
                .as_ref()
                .map(|(_, _, distance)| *distance)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "proximity object world matrices are missing",
                    )
                })?,
            "GEOMETRY" if geometry.is_empty() => {
                let (subject_world_f32, target_origin, _) =
                    origin_distance.as_ref().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "proximity origin world matrices are missing",
                        )
                    })?;
                f64::from(
                    crate::geom::blender_matrix::blender_transform_point_f32(
                        subject_world_f32,
                        vertex.co.as_vec3(),
                    )
                    .distance(*target_origin),
                )
            }
            "GEOMETRY" => {
                let subject_to_local = target_inverse.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "proximity target subject-to-local Blender matrix is missing",
                    )
                })?;
                let point = crate::geom::blender_matrix::blender_transform_point_f32(
                    &subject_to_local,
                    vertex.co.as_vec3(),
                )
                .as_dvec3();
                distance_to_target(
                    point,
                    &geometry,
                    &target_vertices,
                    &target_edges,
                    &triangles,
                )
            }
            _ => unreachable!("proximity mode was validated"),
        };
        let mut mapped = map_proximity_distance(distance, minimum, maximum);
        mapped = apply_falloff(mapped, falloff, curve.as_deref(), &mut random);
        if invert_falloff {
            mapped = 1.0 - mapped;
        }
        selected_ids.push(vertex.id);
        selected_values.push(mask_weight(
            original,
            mapped,
            vertex.id,
            texture_weights,
            mask_weights.as_ref(),
            mask_invert,
            mask_constant,
        ));
    }
    if boolean(modifier, "normalize")? {
        normalize_weights(&mut selected_values);
    } else {
        clamp_weights(&mut selected_values);
    }
    for (id, weight) in selected_ids.into_iter().zip(selected_values) {
        updated.insert(id, weight);
    }
    let mut result = mesh.clone();
    write_group_weights_preserving_zero(&mut result, group, &updated)?;
    Ok(result)
}

#[expect(
    clippy::float_cmp,
    reason = "Blender tests an exactly degenerate distance range (MOD_weightvg_proximity)"
)]
fn map_proximity_distance(distance: f64, minimum: f64, maximum: f64) -> f64 {
    if maximum == minimum {
        f64::from(distance >= maximum)
    } else if maximum > minimum {
        if distance >= maximum {
            1.0
        } else if distance <= minimum {
            0.0
        } else {
            (distance - minimum) / (maximum - minimum)
        }
    } else if distance <= maximum {
        1.0
    } else if distance >= minimum {
        0.0
    } else {
        (distance - minimum) / (maximum - minimum)
    }
}

fn distance_to_target(
    point: DVec3,
    geometry: &[String],
    vertices: &[DVec3],
    edges: &[[DVec3; 2]],
    triangles: &[[DVec3; 3]],
) -> f64 {
    let mut distance = f64::INFINITY;
    if geometry.iter().any(|kind| kind == "VERTEX") {
        distance = distance.min(
            vertices
                .iter()
                .map(|vertex| point.distance(*vertex))
                .fold(f64::INFINITY, f64::min),
        );
    }
    if geometry.iter().any(|kind| kind == "EDGE") {
        distance = distance.min(
            edges
                .iter()
                .map(|edge| point_segment_distance(point, edge[0], edge[1]))
                .fold(f64::INFINITY, f64::min),
        );
    }
    if geometry.iter().any(|kind| kind == "FACE") {
        distance = distance.min(
            triangles
                .iter()
                .map(|triangle| {
                    closest_point_triangle(point, triangle[0], triangle[1], triangle[2])
                        .distance(point)
                })
                .fold(f64::INFINITY, f64::min),
        );
    }
    distance
}

fn point_segment_distance(point: DVec3, first: DVec3, second: DVec3) -> f64 {
    let segment = second - first;
    let length = segment.length_squared();
    let factor = if length > f64::EPSILON {
        ((point - first).dot(segment) / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(first + segment * factor)
}

fn existing_group_weights(mesh: &Mesh, group: &str) -> Result<Option<BTreeMap<u32, f64>>> {
    let Some(groups) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    if !groups.contains_key(group) {
        return Ok(None);
    }
    get_group_weights(mesh, group).map(Some)
}

fn write_group_weights_preserving_zero(
    mesh: &mut Mesh,
    group: &str,
    weights: &BTreeMap<u32, f64>,
) -> Result<()> {
    let groups = mesh
        .attributes
        .entry("vertex_groups".to_owned())
        .or_insert_with(|| Value::Object(Map::new()));
    let groups = groups.as_object_mut().ok_or_else(|| {
        PotError::invalid_argument("vertex_groups mesh attribute must be an object")
    })?;
    let mut values = Map::new();
    for (id, weight) in weights {
        values.insert(format!("v{id}"), json!(weight));
    }
    groups.insert(group.to_owned(), Value::Object(values));
    Ok(())
}

fn mask_weight(
    original: f64,
    mapped: f64,
    vertex_id: u32,
    texture_weights: Option<&BTreeMap<u32, f64>>,
    mask_weights: Option<&BTreeMap<u32, f64>>,
    invert_mask: bool,
    mask_constant: f64,
) -> f64 {
    let Some(factor) = texture_weights
        .and_then(|weights| weights.get(&vertex_id).copied())
        .map(|weight| weight * mask_constant)
        .or_else(|| {
            mask_weights.map(|weights| {
                let value = weights.get(&vertex_id).copied().unwrap_or(0.0);
                (if invert_mask { 1.0 - value } else { value }) * mask_constant
            })
        })
    else {
        return mapped * mask_constant + original * (1.0 - mask_constant);
    };
    mapped * factor + original * (1.0 - factor)
}

fn normalize_weights(weights: &mut [f64]) {
    let Some((&first, rest)) = weights.split_first() else {
        return;
    };
    let mut minimum = first;
    let mut maximum = first;
    for value in rest {
        minimum = minimum.min(*value);
        maximum = maximum.max(*value);
    }
    let range = maximum - minimum;
    let (minimum, factor) = if range.abs() > f64::from(f32::EPSILON) {
        (minimum, 1.0 / range)
    } else {
        (0.0, 1.0)
    };
    for value in weights {
        *value = ((*value - minimum) * factor).clamp(0.0, 1.0);
    }
}

fn clamp_weights(weights: &mut [f64]) {
    for weight in weights {
        *weight = weight.clamp(0.0, 1.0);
    }
}

/// Blender's `BLI_ghashutil_strhash_p` (seed 5381, `h = h * 33 + (signed char)c`) over the object
/// name without its two-character ID-type prefix (Blender's `ob->id.name + 2`); Potter subject
/// names are already unprefixed. RANDOM falloff seeds its per-object stream from this hash.
fn object_name_hash(name: &str) -> u32 {
    name.bytes().fold(5381_u32, |hash, byte| {
        hash.wrapping_mul(33).wrapping_add(byte as i8 as i32 as u32)
    })
}

fn apply_falloff(
    value: f64,
    falloff: &str,
    curve: Option<&[DVec2]>,
    random: &mut BlenderRng,
) -> f64 {
    let value = value.clamp(0.0, 1.0);
    match falloff {
        "CURVE" => curve.map_or(value, |points| curve_mapping_value(points, value)),
        "SHARP" => value * value,
        "SMOOTH" => value * value * (3.0 - 2.0 * value),
        "ROOT" => value.sqrt(),
        "ICON_SPHERECURVE" => (2.0 * value - value * value).sqrt(),
        "RANDOM" => f64::from(random.get_float() * value as f32),
        "STEP" => f64::from(value >= 0.5),
        _ => value,
    }
}

fn curve_mapping_value(points: &[DVec2], value: f64) -> f64 {
    const RESOLUTION: usize = 32;
    const TABLE_SIZE: usize = 256;
    let mut handles = points
        .iter()
        .copied()
        .map(|point| [point, point, point])
        .collect::<Vec<_>>();
    for index in 0..handles.len() {
        let point = handles[index][1];
        let previous = if index == 0 {
            point * 2.0 - handles[index + 1][1]
        } else {
            handles[index - 1][1]
        };
        let next = if index + 1 == handles.len() {
            point * 2.0 - handles[index - 1][1]
        } else {
            handles[index + 1][1]
        };
        let delta_a = point - previous;
        let delta_b = next - point;
        let mut length_a = delta_a.length();
        let mut length_b = delta_b.length();
        if length_a == 0.0 {
            length_a = 1.0;
        }
        if length_b == 0.0 {
            length_b = 1.0;
        }
        let tangent = delta_a / length_a + delta_b / length_b;
        let tangent_length = tangent.length() * 2.5614;
        if tangent_length != 0.0 {
            handles[index][0] = point - tangent * (length_a / tangent_length);
            handles[index][2] = point + tangent * (length_b / tangent_length);
        }
    }
    if handles.len() > 2 {
        let handle_len = handles[0][1].distance(handles[0][2]);
        let mut vector = handles[1][0];
        vector.x = vector.x.max(handles[0][1].x);
        vector -= handles[0][1];
        let length = vector.length();
        if length > f64::EPSILON {
            vector *= handle_len / length;
            handles[0][2] = handles[0][1] + vector;
            handles[0][0] = handles[0][1] - vector;
        }
        let last = handles.len() - 1;
        let handle_len = handles[last][1].distance(handles[last][0]);
        let mut vector = handles[last - 1][2];
        vector.x = vector.x.min(handles[last][1].x);
        vector -= handles[last][1];
        let length = vector.length();
        if length > f64::EPSILON {
            vector *= handle_len / length;
            handles[last][0] = handles[last][1] + vector;
            handles[last][2] = handles[last][1] - vector;
        }
    }
    let mut curve = Vec::with_capacity((handles.len() - 1) * RESOLUTION);
    for segment in 0..handles.len() - 1 {
        let mut start = handles[segment];
        let mut end = handles[segment + 1];
        let delta_a = start[1] - start[2];
        let delta_b = end[1] - end[0];
        let length_a = delta_a.x.abs();
        let length_b = delta_b.x.abs();
        let span = end[1].x - start[1].x;
        if length_a + length_b > span && length_a + length_b > 0.0 {
            let factor = span / (length_a + length_b);
            start[2] = start[1] - delta_a * factor;
            end[0] = end[1] - delta_b * factor;
        }
        for step in 0..RESOLUTION {
            let t = step as f64 / (RESOLUTION - 1) as f64;
            let inverse = 1.0 - t;
            curve.push(
                start[1] * inverse.powi(3)
                    + start[2] * (3.0 * inverse.powi(2) * t)
                    + end[0] * (3.0 * inverse * t.powi(2))
                    + end[1] * t.powi(3),
            );
        }
    }
    let minimum_x = 0.0;
    let maximum_x = 1.0;
    let mut table = [0.0; TABLE_SIZE + 1];
    for (index, output) in table.iter_mut().enumerate() {
        let x = minimum_x + (maximum_x - minimum_x) * index as f64 / TABLE_SIZE as f64;
        let mut upper = curve.partition_point(|point| point.x <= x);
        if upper == 0 {
            let first = curve[0];
            let handle = handles[0][0];
            let tangent = first - handle;
            *output = first.y + tangent.y * (x - first.x) / tangent.x;
        } else if upper >= curve.len() {
            *output = curve[curve.len() - 1].y;
        } else {
            upper = upper.max(1);
            let first = curve[upper - 1];
            let second = curve[upper];
            let span = second.x - first.x;
            let factor = if span > f64::EPSILON {
                (x - first.x) / span
            } else {
                0.0
            };
            *output = first.y + (second.y - first.y) * factor;
        }
    }
    let position = (value - minimum_x) * TABLE_SIZE as f64 / (maximum_x - minimum_x);
    if position >= TABLE_SIZE as f64 {
        return table[TABLE_SIZE].clamp(0.0, 1.0);
    }
    let lower = position.max(0.0) as usize;
    let fraction = position - lower as f64;
    (table[lower] + (table[lower + 1] - table[lower]) * fraction).clamp(0.0, 1.0)
}
#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores object world matrices and vertex positions in f32"
)]
fn weight_matrix_f32(matrix: DMat4) -> [f32; 16] {
    matrix.to_cols_array().map(|value| value as f32)
}
