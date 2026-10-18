//! Vertex-weight edit/mix/proximity and their image/bone mask coordinates.

use super::*;

pub(super) fn texture_mask_weights(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    texture: Option<crate::geom::legacy_texture::TextureSource<'_>>,
    subject_world: DMat4,
    rng: &mut crate::blender_rng::BlenderRng,
) -> Result<Option<BTreeMap<u32, f64>>> {
    if modifier.modifier_type == "vertex_weight_proximity"
        && (optional_string(modifier, "target")?.is_none()
            || operands.get("target").and_then(|objects| objects.first()).is_none())
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
    let uv_coordinates = if mapping == "UV" {
        Some(vertex_uv_coordinates(mesh, modifier, uv_layer)?)
    } else {
        None
    };
    let object_matrix = texture_object_matrix(modifier, operands, mapping)?;
    let coordinates = texture_coordinates(
        mesh,
        mapping,
        uv_coordinates.as_ref(),
        weight_matrix_f32(subject_world),
        object_matrix.as_ref(),
    );
    let mut weights = BTreeMap::new();
    for (vertex, coordinate) in mesh.vertices.iter().zip(coordinates) {
        let sample = crate::geom::legacy_texture::sample(texture, coordinate, rng)?;
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

/// The mask object's Blender-order f32 subject-to-object matrix. Only OBJECT mapping with a
/// mapping object uses it; other mappings and an unset object yield `None` (LOCAL coordinates).
fn texture_object_matrix(
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    mapping: &str,
) -> Result<Option<[f32; 16]>> {
    if mapping != "OBJECT" || optional_string(modifier, "mask_tex_map_object")?.is_none() {
        return Ok(None);
    }
    operands
        .get("mask_tex_map_object")
        .and_then(|objects| objects.first())
        .and_then(|object| object.blender_matrices_f32)
        .and_then(|matrices| matrices.subject_to_local)
        .map(Some)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::DependencyMissing,
                "weight texture object coordinates were not resolved",
                json!({"modifier_id":modifier.id,"parameter":"mask_tex_map_object"}),
            )
        })
}

/// Per-vertex mask-texture coordinates, aligned with `mesh.vertices`. Matrix transforms run in
/// f32 as in Blender's `MOD_get_texture_coords`.
fn texture_coordinates(
    mesh: &Mesh,
    mapping: &str,
    uv_coordinates: Option<&BTreeMap<u32, DVec2>>,
    subject_world: [f32; 16],
    object_matrix: Option<&[f32; 16]>,
) -> Vec<DVec3> {
    mesh.vertices
        .iter()
        .map(|vertex| match mapping {
            "LOCAL" => vertex.co,
            "GLOBAL" => transform_point_f32(&subject_world, vertex.co),
            "OBJECT" => object_matrix
                .map_or(vertex.co, |matrix| transform_point_f32(matrix, vertex.co)),
            "UV" => uv_coordinates
                .and_then(|coordinates| coordinates.get(&vertex.id).copied())
                .map_or(vertex.co, |uv| {
                    DVec3::new(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0)
                }),
            _ => unreachable!("texture mapping was validated"),
        })
        .collect()
}

fn transform_point_f32(matrix: &[f32; 16], point: DVec3) -> DVec3 {
    crate::geom::blender_matrix::blender_transform_point_f32(matrix, point.as_vec3()).as_dvec3()
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
                return Err(invalid(modifier, "mask_tex_uv_layer", "a mesh with valid face vertices"));
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
    let mut random = BlenderRandom::for_object(subject_name);
    let mut values = Vec::with_capacity(mesh.vertices.len());
    for vertex in &mesh.vertices {
        let original = existing.get(&vertex.id).copied().unwrap_or(default);
        let mut mapped = apply_falloff(
            original,
            falloff,
            curve.as_deref(),
            &mut random,
        );
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
        } else if b >= 0.0 && b < ZERO_FLOOR {
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
        Some((subject_world_f32, target_origin, f64::from((subject_origin - target_origin).length())))
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
            .triangulate()
            .map_err(|error| {
                PotError::invalid_argument(format!("invalid proximity target triangulation: {error}"))
            })?
            .into_iter()
            .filter_map(|triangle| {
                Some([
                    *positions.get(&triangle[0])?,
                    *positions.get(&triangle[1])?,
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
                target.mesh.vertices.iter().find(|vertex| vertex.id == edge.vertices[0])?.co.as_vec3().as_dvec3(),
                target.mesh.vertices.iter().find(|vertex| vertex.id == edge.vertices[1])?.co.as_vec3().as_dvec3(),
            ])
        })
        .collect::<Vec<_>>();
    let mut random = BlenderRandom::for_object(subject_name);
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
                .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "proximity object world matrices are missing"))?,
            "GEOMETRY" if geometry.is_empty() => {
                let (subject_world_f32, target_origin, _) = origin_distance
                    .as_ref()
                    .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "proximity origin world matrices are missing"))?;
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
    let groups = groups
        .as_object_mut()
        .ok_or_else(|| PotError::invalid_argument("vertex_groups mesh attribute must be an object"))?;
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

// Weight falloff RANDOM's per-object stream (distinct from the shared texture `BlenderRng`).
struct BlenderRandom {
    state: u64,
}

impl BlenderRandom {
    const NOISE_HASH: [u8; 256] = [
        0xA2, 0xA0, 0x19, 0x3B, 0xF8, 0xEB, 0xAA, 0xEE, 0xF3, 0x1C, 0x67, 0x28, 0x1D, 0xED, 0x00,
        0xDE, 0x95, 0x2E, 0xDC, 0x3F, 0x3A, 0x82, 0x35, 0x4D, 0x6C, 0xBA, 0x36, 0xD0, 0xF6, 0x0C,
        0x79, 0x32, 0xD1, 0x59, 0xF4, 0x08, 0x8B, 0x63, 0x89, 0x2F, 0xB8, 0xB4, 0x97, 0x83, 0xF2,
        0x8F, 0x18, 0xC7, 0x51, 0x14, 0x65, 0x87, 0x48, 0x20, 0x42, 0xA8, 0x80, 0xB5, 0x40, 0x13,
        0xB2, 0x22, 0x7E, 0x57, 0xBC, 0x7F, 0x6B, 0x9D, 0x86, 0x4C, 0xC8, 0xDB, 0x7C, 0xD5, 0x25,
        0x4E, 0x5A, 0x55, 0x74, 0x50, 0xCD, 0xB3, 0x7A, 0xBB, 0xC3, 0xCB, 0xB6, 0xE2, 0xE4, 0xEC,
        0xFD, 0x98, 0x0B, 0x96, 0xD3, 0x9E, 0x5C, 0xA1, 0x64, 0xF1, 0x81, 0x61, 0xE1, 0xC4, 0x24,
        0x72, 0x49, 0x8C, 0x90, 0x4B, 0x84, 0x34, 0x38, 0xAB, 0x78, 0xCA, 0x1F, 0x01, 0xD7, 0x93,
        0x11, 0xC1, 0x58, 0xA9, 0x31, 0xF9, 0x44, 0x6D, 0xBF, 0x33, 0x9C, 0x5F, 0x09, 0x94, 0xA3,
        0x85, 0x06, 0xC6, 0x9A, 0x1E, 0x7B, 0x46, 0x15, 0x30, 0x27, 0x2B, 0x1B, 0x71, 0x3C, 0x5B,
        0xD6, 0x6F, 0x62, 0xAC, 0x4F, 0xC2, 0xC0, 0x0E, 0xB1, 0x23, 0xA7, 0xDF, 0x47, 0xB0, 0x77,
        0x69, 0x05, 0xE9, 0xE6, 0xE7, 0x76, 0x73, 0x0F, 0xFE, 0x6E, 0x9B, 0x56, 0xEF, 0x12, 0xA5,
        0x37, 0xFC, 0xAE, 0xD9, 0x03, 0x8E, 0xDD, 0x10, 0xB9, 0xCE, 0xC9, 0x8D, 0xDA, 0x2A, 0xBD,
        0x68, 0x17, 0x9F, 0xBE, 0xD4, 0x0A, 0xCC, 0xD2, 0xE8, 0x43, 0x3D, 0x70, 0xB7, 0x02, 0x7D,
        0x99, 0xD8, 0x0D, 0x60, 0x8A, 0x04, 0x2C, 0x3E, 0x92, 0xE5, 0xAF, 0x53, 0x07, 0xE0, 0x29,
        0xA6, 0xC5, 0xE3, 0xF5, 0xF7, 0x4A, 0x41, 0x26, 0x6A, 0x16, 0x5E, 0x52, 0x2D, 0x21, 0xAD,
        0xF0, 0x91, 0xFF, 0xEA, 0x54, 0xFA, 0x66, 0x1A, 0x45, 0x39, 0xCF, 0x75, 0xA4, 0x88, 0xFB,
        0x5D,
    ];

    fn for_object(name: &str) -> Self {
        let hash = name.bytes().fold(5381_u32, |hash, byte| {
            let byte = (byte as i8 as i32) as u32;
            hash.wrapping_mul(33).wrapping_add(byte)
        });
        let seed = |seed: u32| (u64::from(seed) << 16) | 0x330e;
        let mut state = seed(hash.wrapping_add(u32::from(Self::NOISE_HASH[(hash & 255) as usize])));
        let mut value = Self::next_uint(state);
        state = seed(value.wrapping_add(u32::from(Self::NOISE_HASH[(value & 255) as usize])));
        value = Self::next_uint(state);
        state = seed(value.wrapping_add(u32::from(Self::NOISE_HASH[(value & 255) as usize])));
        Self { state }
    }

    fn next_uint(state: u64) -> u32 {
        let state = state.wrapping_mul(0x5deece66d).wrapping_add(0xb) & 0x0000_ffff_ffff_ffff;
        (state >> 17) as u32
    }

    fn next_float(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(0x5deece66d)
            .wrapping_add(0xb)
            & 0x0000_ffff_ffff_ffff;
        ((self.state >> 17) as f32 / 2_147_483_648.0_f32) as f64
    }
}

fn apply_falloff(value: f64, falloff: &str, curve: Option<&[DVec2]>, random: &mut BlenderRandom) -> f64 {
    let value = value.clamp(0.0, 1.0);
    match falloff {
        "CURVE" => curve.map_or(value, |points| curve_mapping_value(points, value)),
        "SHARP" => value * value,
        "SMOOTH" => value * value * (3.0 - 2.0 * value),
        "ROOT" => value.sqrt(),
        "ICON_SPHERECURVE" => (2.0 * value - value * value).sqrt(),
        "RANDOM" => value * random.next_float(),
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


