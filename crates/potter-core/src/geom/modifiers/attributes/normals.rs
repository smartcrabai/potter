//! Weighted-normal and normal-edit modifiers.

use super::*;
use std::collections::BTreeSet;
use std::{ffi::c_void, mem::size_of, os::raw::c_int};

use crate::geom::normals::corner_normals;
use glam::Vec3;

type WeightedModeCompare = unsafe extern "C" fn(*const c_void, *const c_void) -> c_int;

unsafe extern "C" {
    fn qsort(
        base: *mut c_void,
        count: usize,
        element_size: usize,
        compare: Option<WeightedModeCompare>,
    );
}

pub(super) fn weighted_normal(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let mode = string(modifier, "mode")?;
    if !matches!(mode, "FACE_AREA" | "CORNER_ANGLE" | "FACE_AREA_WITH_ANGLE") {
        return Err(invalid(
            modifier,
            "mode",
            "FACE_AREA, CORNER_ANGLE, or FACE_AREA_WITH_ANGLE",
        ));
    }
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }

    let weight = blender_weight(number(modifier, "weight")?);
    let threshold = number(modifier, "thresh")?.max(0.0) as f32;
    let group = optional_string(modifier, "vertex_group")?;
    let group_weights = group
        .filter(|name| vertex_group_exists(mesh, name))
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group")?;
    let keep_sharp = boolean(modifier, "keep_sharp")?;
    let use_face_influence = boolean(modifier, "use_face_influence")?;
    let mut smooth_faces = crate::geom::normals::smooth_face_flags(mesh)?;
    if let Some(sharp_faces) = mesh
        .attributes
        .get("sharp_face")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
    {
        for (index, face) in mesh.faces.iter().enumerate() {
            if sharp_faces
                .get(&format!("f{}", face.id))
                .and_then(Value::as_bool)
                == Some(true)
            {
                smooth_faces[index] = false;
            }
        }
    }
    let sharp_edges = crate::geom::normals::sharp_edge_flags(mesh)?;
    let face_strengths = mesh
        .attributes
        .get("__mod_weightednormals_faceweight")
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    let face_influence = use_face_influence && face_strengths.is_some();
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let total_corners = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    let mut face_corner_starts = Vec::with_capacity(mesh.faces.len());
    let mut corner_offset = 0;
    let mut face_normals = Vec::with_capacity(mesh.faces.len());
    let mut face_areas = Vec::with_capacity(mesh.faces.len());
    let mut corner_angles = if mode == "FACE_AREA" {
        Vec::new()
    } else {
        Vec::with_capacity(total_corners)
    };
    for face in &mesh.faces {
        face_corner_starts.push(corner_offset);
        let points = face_points(mesh, &face.vertices)?
            .into_iter()
            .map(|point| Vec3::new(point.x as f32, point.y as f32, point.z as f32))
            .collect::<Vec<_>>();
        face_normals.push(blender_face_normal(&points));
        face_areas.push(blender_face_area(&points));
        if mode != "FACE_AREA" {
            corner_angles.extend(
                (0..face.vertices.len()).map(|corner| blender_corner_angle(&points, corner)),
            );
        }
        corner_offset += face.vertices.len();
    }

    let mut incident = HashMap::<u32, Vec<usize>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            incident.entry(*vertex_id).or_default().push(face_index);
        }
    }
    let mut edge_uses = HashMap::<(u32, u32), Vec<(usize, bool)>>::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for corner in 0..face.vertices.len() {
            let first = face.vertices[corner];
            let second = face.vertices[(corner + 1) % face.vertices.len()];
            let (key, direction) = if first < second {
                ((first, second), true)
            } else {
                ((second, first), false)
            };
            edge_uses
                .entry(key)
                .or_default()
                .push((face_index, direction));
        }
    }
    let sharp_edge_keys = mesh
        .edges
        .iter()
        .zip(&sharp_edges)
        .filter_map(|(edge, sharp)| {
            (*sharp).then(|| {
                if edge.vertices[0] < edge.vertices[1] {
                    (edge.vertices[0], edge.vertices[1])
                } else {
                    (edge.vertices[1], edge.vertices[0])
                }
            })
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut fan_by_corner = vec![0; total_corners];
    if keep_sharp {
        let mut next_fan = 0;
        for (vertex_id, adjacent_faces) in &incident {
            let mut assigned = std::collections::BTreeSet::new();
            for &start in adjacent_faces {
                if assigned.contains(&start) {
                    continue;
                }
                let fan = smooth_fan_faces(
                    mesh,
                    *vertex_id,
                    start,
                    adjacent_faces,
                    &edge_uses,
                    &sharp_edge_keys,
                    &smooth_faces,
                );
                for face_index in fan {
                    assigned.insert(face_index);
                    let corner_in_face = mesh.faces[face_index]
                        .vertices
                        .iter()
                        .position(|id| id == vertex_id)
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "weighted-normal fan contains a non-incident face",
                            )
                        })?;
                    fan_by_corner[face_corner_starts[face_index] + corner_in_face] = next_fan;
                }
                next_fan += 1;
            }
        }
    }

    let mut events = Vec::<WeightedEvent>::with_capacity(total_corners);
    let mut corner = 0;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for vertex_id in &face.vertices {
            let value = match mode {
                "FACE_AREA" => face_areas[face_index],
                "CORNER_ANGLE" => corner_angles[corner],
                _ => face_areas[face_index] * corner_angles[corner],
            };
            events.push(WeightedEvent {
                face: face_index,
                vertex: *vertex_id,
                corner,
                value,
            });
            corner += 1;
        }
    }
    let mut mode_pairs = if mode == "FACE_AREA" {
        face_areas
            .iter()
            .enumerate()
            .map(|(index, &value)| WeightedModePair { value, index })
            .collect::<Vec<_>>()
    } else {
        events
            .iter()
            .map(|event| WeightedModePair {
                value: event.value,
                index: event.corner,
            })
            .collect::<Vec<_>>()
    };
    blender_sort_mode_pairs(&mut mode_pairs);
    let ordered_events = if mode == "FACE_AREA" {
        let mut ordered = Vec::with_capacity(total_corners);
        for pair in mode_pairs {
            let face = pair.index;
            let start = face_corner_starts[face];
            let end = start + mesh.faces[face].vertices.len();
            ordered.extend_from_slice(&events[start..end]);
        }
        ordered
    } else {
        mode_pairs
            .iter()
            .map(|pair| events[pair.index])
            .collect::<Vec<_>>()
    };
    let mut items = BTreeMap::<WeightedItemKey, WeightedItem>::new();
    let mut cached_inverse_powers = [0.0_f32; 128];
    for event in ordered_events {
        let group_member = group_weights
            .as_ref()
            .is_some_and(|weights| weights.contains_key(&event.vertex));
        if group_weights.is_some() && (group_member == invert_group) {
            continue;
        }
        let key = if keep_sharp {
            WeightedItemKey::Fan(event.vertex, fan_by_corner[event.corner])
        } else {
            WeightedItemKey::Vertex(event.vertex)
        };
        let strength = if face_influence {
            face_strengths
                .and_then(|values| values.get(&format!("f{}", mesh.faces[event.face].id)))
                .and_then(Value::as_i64)
                .unwrap_or(0) as i32
        } else {
            0
        };
        let item = items.entry(key).or_insert_with(|| WeightedItem {
            normal: Vec3::ZERO,
            current_value: 0.0,
            loops_num: 0,
            current_strength: 0,
        });
        if face_influence && strength < item.current_strength {
            continue;
        }
        if face_influence && strength > item.current_strength {
            item.current_strength = strength;
            item.current_value = 0.0;
            item.loops_num = 0;
            item.normal = Vec3::ZERO;
        }
        if item.current_value == 0.0 {
            item.current_value = event.value;
        }
        if !compare_float(item.current_value, event.value, threshold) {
            item.loops_num += 1;
            item.current_value = event.value;
        }
        let loops_num = usize::try_from(item.loops_num).unwrap_or(usize::MAX);
        let inverse_weight = if loops_num < cached_inverse_powers.len() {
            if cached_inverse_powers[loops_num] == 0.0 {
                cached_inverse_powers[loops_num] = 1.0 / weight.powf(item.loops_num as f32);
            }
            cached_inverse_powers[loops_num]
        } else {
            1.0 / weight.powf(item.loops_num as f32)
        };
        item.normal += face_normals[event.face] * (event.value * inverse_weight);
    }
    for item in items.values_mut() {
        let length = item.normal.length();
        item.normal = if length >= 1.0e-6 {
            item.normal / length
        } else {
            Vec3::ZERO
        };
    }

    let corner_normals = if keep_sharp || group_weights.is_some() {
        Some(evaluated_corner_normals(mesh, &positions)?)
    } else {
        None
    };
    let vertex_fallback = if !keep_sharp
        && group_weights.is_none()
        && items.values().any(|item| item.normal == Vec3::ZERO)
    {
        Some(crate::geom::normals::export_normals(mesh, &positions)?.vertices)
    } else {
        None
    };
    let vertex_indices = vertex_fallback.as_ref().map(|_| {
        mesh.vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>()
    });
    let mut result = mesh.clone();
    let mut output = corner_normals.unwrap_or_else(|| vec![DVec3::ZERO; total_corners]);
    let mut output_corner = 0;
    for face in &mesh.faces {
        for vertex_id in &face.vertices {
            let key = if keep_sharp {
                WeightedItemKey::Fan(*vertex_id, fan_by_corner[output_corner])
            } else {
                WeightedItemKey::Vertex(*vertex_id)
            };
            if let Some(item) = items.get(&key) {
                if item.normal != Vec3::ZERO {
                    output[output_corner] = DVec3::new(
                        f64::from(item.normal.x),
                        f64::from(item.normal.y),
                        f64::from(item.normal.z),
                    );
                } else if let (Some(vertices), Some(indices)) =
                    (vertex_fallback.as_ref(), vertex_indices.as_ref())
                    && let Some(index) = indices.get(vertex_id)
                {
                    output[output_corner] = vertices[*index];
                }
            }
            output_corner += 1;
        }
    }
    write_custom_normals(&mut result, &output)?;
    Ok(result)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum WeightedItemKey {
    Vertex(u32),
    Fan(u32, usize),
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct WeightedModePair {
    value: f32,
    index: usize,
}

struct WeightedItem {
    normal: Vec3,
    current_value: f32,
    loops_num: i32,
    current_strength: i32,
}

#[derive(Clone, Copy)]
struct WeightedEvent {
    face: usize,
    vertex: u32,
    corner: usize,
    value: f32,
}

unsafe extern "C" fn compare_weighted_mode_pairs(
    first: *const c_void,
    second: *const c_void,
) -> c_int {
    // SAFETY: `qsort` calls this only with pointers to elements of the `WeightedModePair` slice.
    let first = unsafe { &*first.cast::<WeightedModePair>() };
    // SAFETY: `qsort` calls this only with pointers to elements of the `WeightedModePair` slice.
    let second = unsafe { &*second.cast::<WeightedModePair>() };
    if first.value < second.value {
        1
    } else if first.value > second.value {
        -1
    } else {
        0
    }
}

/// Blender sorts `ModePair` with C `qsort`; its tie permutation changes f32 accumulation order.
fn blender_sort_mode_pairs(pairs: &mut [WeightedModePair]) {
    if pairs.len() < 2 {
        return;
    }
    // SAFETY: `pairs` is a contiguous, correctly aligned slice. The comparator reads only
    // `WeightedModePair::value` and the C library swaps complete elements in place.
    unsafe {
        qsort(
            pairs.as_mut_ptr().cast(),
            pairs.len(),
            size_of::<WeightedModePair>(),
            Some(compare_weighted_mode_pairs),
        );
    }
}

fn blender_weight(weight: f64) -> f32 {
    let parameter = (weight.clamp(1.0, 100.0) as f32) / 50.0;
    if weight == 100.0 {
        i16::MAX as f32
    } else if weight == 1.0 {
        1.0 / i16::MAX as f32
    } else if (parameter - 1.0) * 25.0 > 1.0 {
        (parameter - 1.0) * 25.0
    } else {
        parameter
    }
}

fn compare_float(first: f32, second: f32, tolerance: f32) -> bool {
    (first - second).abs() <= tolerance
}

fn blender_dot(first: Vec3, second: Vec3) -> f32 {
    first.x * second.x + first.y * second.y + first.z * second.z
}

fn blender_cross(first: Vec3, second: Vec3) -> Vec3 {
    Vec3::new(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )
}

fn blender_normalize(vector: Vec3) -> Vec3 {
    let length_squared = blender_dot(vector, vector);
    if length_squared > 1.0e-35_f32 {
        vector * (1.0 / length_squared.sqrt())
    } else {
        Vec3::ZERO
    }
}

fn blender_face_normal(points: &[Vec3]) -> Vec3 {
    let normal = match points {
        [first, second, third] => {
            blender_normalize(blender_cross(*first - *second, *second - *third))
        }
        [first, second, third, fourth] => {
            blender_normalize(blender_cross(*first - *third, *second - *fourth))
        }
        _ => {
            let mut normal = Vec3::ZERO;
            let mut previous = points[points.len() - 1];
            for &current in points {
                normal.x += (previous.y - current.y) * (previous.z + current.z);
                normal.y += (previous.z - current.z) * (previous.x + current.x);
                normal.z += (previous.x - current.x) * (previous.y + current.y);
                previous = current;
            }
            blender_normalize(normal)
        }
    };
    if normal == Vec3::ZERO {
        Vec3::Z
    } else {
        normal
    }
}

fn blender_face_area(points: &[Vec3]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let normal = if points.len() == 3 {
        blender_cross(points[1] - points[0], points[2] - points[0])
    } else {
        let mut normal = Vec3::ZERO;
        let mut previous = points[points.len() - 1];
        for &current in points {
            normal.x += (previous.y - current.y) * (previous.z + current.z);
            normal.y += (previous.z - current.z) * (previous.x + current.x);
            normal.z += (previous.x - current.x) * (previous.y + current.y);
            previous = current;
        }
        normal
    };
    normal.length() * 0.5
}

fn blender_corner_angle(points: &[Vec3], corner: usize) -> f32 {
    let center = points[corner];
    let previous = blender_normalize(points[(corner + points.len() - 1) % points.len()] - center);
    let next = blender_normalize(center - points[(corner + 1) % points.len()]);
    let edge_angle = blender_dot(previous, next).clamp(-1.0, 1.0).acos();
    std::f32::consts::PI - edge_angle
}

fn vertex_group_exists(mesh: &Mesh, group: &str) -> bool {
    mesh.attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .is_some_and(|groups| groups.contains_key(group))
}

fn write_custom_normals(mesh: &mut Mesh, normals: &[DVec3]) -> Result<()> {
    let corners = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len())
        .sum::<usize>();
    if normals.len() != corners {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "normal modifier output does not describe every mesh corner",
        ));
    }
    let vectors = normals
        .iter()
        .map(|&normal| (normal != DVec3::ZERO).then_some(normal))
        .collect::<Vec<_>>();
    crate::geom::clnor::encode_vectors(
        mesh,
        Some(&vectors),
        crate::geom::clnor::NormalCodeOptions {
            corner_order: None,
            setter_input: crate::geom::clnor::SetterInput::AlreadyNormalized,
        },
    )
}

fn evaluated_corner_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    let computed = crate::geom::normals::setter_corner_normals(mesh, positions)?;
    let Some(custom) = crate::geom::clnor::corner_custom_normals(
        mesh,
        positions,
        crate::geom::clnor::NormalCodeOptions::default(),
    )?
    else {
        return Ok(computed);
    };
    Ok(custom
        .into_iter()
        .zip(computed)
        .map(|(normal, computed)| normal.unwrap_or(computed))
        .collect())
}

fn faces_share_smooth_edge_at_vertex(
    mesh: &Mesh,
    first_index: usize,
    second_index: usize,
    vertex_id: u32,
    edge_uses: &HashMap<(u32, u32), Vec<(usize, bool)>>,
    sharp_edge_keys: &std::collections::BTreeSet<(u32, u32)>,
    smooth_faces: &[bool],
) -> bool {
    if !smooth_faces.get(first_index).copied().unwrap_or(false)
        || !smooth_faces.get(second_index).copied().unwrap_or(false)
    {
        return false;
    }
    let first = &mesh.faces[first_index];
    let Some(index) = first.vertices.iter().position(|id| *id == vertex_id) else {
        return false;
    };
    let neighbors = [
        first.vertices[(index + first.vertices.len() - 1) % first.vertices.len()],
        first.vertices[(index + 1) % first.vertices.len()],
    ];
    neighbors.into_iter().any(|neighbor| {
        let key = if vertex_id < neighbor {
            (vertex_id, neighbor)
        } else {
            (neighbor, vertex_id)
        };
        if sharp_edge_keys.contains(&key) {
            return false;
        }
        let Some(uses) = edge_uses.get(&key) else {
            return false;
        };
        uses.len() == 2
            && uses
                .iter()
                .any(|(face_index, _)| *face_index == first_index)
            && uses
                .iter()
                .any(|(face_index, _)| *face_index == second_index)
            && uses[0].1 != uses[1].1
    })
}

fn smooth_fan_faces(
    mesh: &Mesh,
    vertex_id: u32,
    start: usize,
    incident: &[usize],
    edge_uses: &HashMap<(u32, u32), Vec<(usize, bool)>>,
    sharp_edge_keys: &std::collections::BTreeSet<(u32, u32)>,
    smooth_faces: &[bool],
) -> Vec<usize> {
    let mut fan = vec![start];
    let mut cursor = 0;
    while cursor < fan.len() {
        let current = fan[cursor];
        for candidate in incident {
            if !fan.contains(candidate)
                && faces_share_smooth_edge_at_vertex(
                    mesh,
                    current,
                    *candidate,
                    vertex_id,
                    edge_uses,
                    sharp_edge_keys,
                    smooth_faces,
                )
            {
                fan.push(*candidate);
            }
        }
        cursor += 1;
    }
    fan.sort_unstable();
    fan
}

pub(super) fn normal_edit(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
) -> Result<Mesh> {
    let mode = string(modifier, "mode")?;
    if !matches!(mode, "RADIAL" | "DIRECTIONAL") {
        return Err(invalid(modifier, "mode", "RADIAL or DIRECTIONAL"));
    }
    if mesh.faces.is_empty() {
        return Ok(mesh.clone());
    }
    let target = operands.get("target").and_then(|targets| targets.first());
    if mode == "DIRECTIONAL" && target.is_none() {
        return Ok(mesh.clone());
    }
    let offset = vector3(modifier, "offset")?;
    let mix_mode = string(modifier, "mix_mode")?;
    if !matches!(mix_mode, "COPY" | "ADD" | "SUB" | "MUL") {
        return Err(invalid(modifier, "mix_mode", "COPY, ADD, SUB, or MUL"));
    }
    let factor = number(modifier, "mix_factor")?.clamp(0.0, 1.0);
    let limit = number(modifier, "mix_limit")?;
    let parallel = boolean(modifier, "use_direction_parallel")?;
    let group = optional_string(modifier, "vertex_group")?;
    let group_weights = group
        .filter(|name| vertex_group_exists(mesh, name))
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group")?;
    let no_polynors_fix = boolean(modifier, "no_polynors_fix")?;
    let target_origin = if let Some(target) = target {
        let local_to_subject = target
            .blender_matrices_f32
            .as_ref()
            .and_then(|matrices| matrices.local_to_subject.as_ref())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Normal Edit target Blender transform is missing",
                )
            })?;
        crate::geom::blender_matrix::blender_transform_point_f32(local_to_subject, Vec3::ZERO)
            .as_dvec3()
    } else {
        DVec3::ZERO
    };
    let use_current_normals = !(mix_mode == "COPY"
        && factor == 1.0
        && group.is_none_or(str::is_empty)
        && limit as f32 == std::f32::consts::PI);
    let positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let current_normals = if use_current_normals {
        Some(evaluated_corner_normals(mesh, &positions)?)
    } else {
        None
    };
    let size = if mode == "RADIAL" {
        radial_size(mesh, target)
    } else {
        DVec3::ONE
    };
    let target_direction = (target_origin - offset).normalize_or_zero();
    let radial_center = if target.is_none() {
        offset
    } else {
        target_origin
    };
    let mut face_corner_start = 0;
    let mut normals = Vec::with_capacity(mesh.faces.iter().map(|face| face.vertices.len()).sum());
    for face in &mesh.faces {
        for (corner_index, vertex_id) in face.vertices.iter().enumerate() {
            let vertex = mesh
                .vertex(*vertex_id)
                .ok_or_else(|| invalid(modifier, "target", "a mesh with valid face vertices"))?;
            let destination = if mode == "RADIAL" {
                radial_normal(vertex.co, radial_center, size)
            } else if parallel {
                target_direction
            } else {
                (target_origin - vertex.co).normalize_or_zero()
            };
            let Some(current_normals) = current_normals.as_ref() else {
                normals.push(destination);
                continue;
            };
            let current = current_normals[face_corner_start + corner_index];
            let mixed = match mix_mode {
                "COPY" => destination,
                "ADD" => (destination + current).normalize_or_zero(),
                "SUB" => (destination - current).normalize_or_zero(),
                "MUL" => (destination * current).normalize_or_zero(),
                _ => unreachable!(),
            };
            let vertex_factor = group_weights.as_ref().map_or(1.0, |weights| {
                let weight = weights.get(vertex_id).copied().unwrap_or(0.0);
                if invert_group { 1.0 - weight } else { weight }
            });
            let mut blend_factor = factor * vertex_factor;
            if (limit as f32) < std::f32::consts::PI {
                let angle = current.angle_between(mixed);
                if angle > f64::EPSILON {
                    blend_factor = blend_factor.min(limit / angle);
                }
            }
            normals.push(slerp_safe(current, mixed, blend_factor));
        }
        face_corner_start += face.vertices.len();
    }

    let mut flips = vec![false; mesh.faces.len()];
    if !no_polynors_fix {
        let mut corner = 0;
        for (face_index, face) in mesh.faces.iter().enumerate() {
            let end = corner + face.vertices.len();
            let sum = normals[corner..end]
                .iter()
                .copied()
                .fold(DVec3::ZERO, |sum, normal| sum + normal);
            let average = sum.normalize_or_zero();
            if average.length_squared() > 0.0 && face_normal(mesh, face)?.dot(average) < 0.0 {
                flips[face_index] = true;
                normals[corner + 1..end].reverse();
            }
            corner = end;
        }
    }

    let mut result = mesh.clone();
    if flips.iter().any(|flip| *flip) {
        flip_corner_attributes(&mut result, &flips);
        for (face, flip) in result.faces.iter_mut().zip(&flips) {
            if *flip {
                face.vertices[1..].reverse();
            }
        }
    }
    if normals.contains(&DVec3::ZERO) {
        let fallback = evaluated_corner_normals(&result, &positions)?;
        for (normal, fallback) in normals.iter_mut().zip(fallback) {
            if *normal == DVec3::ZERO {
                *normal = fallback;
            }
        }
    }
    write_custom_normals(&mut result, &normals)?;
    Ok(result)
}

fn radial_size(mesh: &Mesh, target: Option<&SceneOperand>) -> DVec3 {
    let size = target.map_or_else(
        || {
            let Some(first) = mesh.vertices.first() else {
                return DVec3::ONE;
            };
            let (minimum, maximum) = mesh
                .vertices
                .iter()
                .fold((first.co, first.co), |(minimum, maximum), vertex| {
                    (minimum.min(vertex.co), maximum.max(vertex.co))
                });
            maximum - minimum
        },
        |target| target.object_scale.abs(),
    );
    if size == DVec3::ZERO {
        DVec3::ONE
    } else {
        size.max(DVec3::splat(f32::EPSILON as f64))
    }
}

fn radial_normal(position: DVec3, target_origin: DVec3, size: DVec3) -> DVec3 {
    let co = position - target_origin;
    let [a, b, c] = size.to_array();
    let [x, y, z] = co.to_array();
    let m2 = (b * b) / (a * a);
    let n2 = (c * c) / (a * a);
    let a2 = x * x + y * y / m2 + z * z / n2;
    let b2 = m2 * x * x + y * y + m2 * z * z / n2;
    let c2 = n2 * x * x + n2 * y * y / m2 + z * z;
    DVec3::new(x / a2, y / b2, z / c2).normalize_or_zero()
}

fn slerp_safe(from: DVec3, to: DVec3, factor: f64) -> DVec3 {
    let cosine = from.dot(to).clamp(-1.0, 1.0);
    if cosine < -1.0 + f32::EPSILON as f64 {
        let axis = from.abs().max_element();
        let orthogonal = if from.x.abs() == axis {
            DVec3::new(-from.y - from.z, from.x, from.x)
        } else if from.y.abs() == axis {
            DVec3::new(from.y, -from.x - from.z, from.y)
        } else {
            DVec3::new(from.z, from.z, -from.x - from.y)
        }
        .normalize_or_zero();
        if factor < 0.5 {
            slerp(from, orthogonal, factor * 2.0)
        } else {
            slerp(orthogonal, to, (factor - 0.5) * 2.0)
        }
    } else {
        slerp(from, to, factor)
    }
}

fn slerp(from: DVec3, to: DVec3, factor: f64) -> DVec3 {
    let cosine = from.dot(to).clamp(-1.0, 1.0);
    if cosine.abs() < 1.0 - 1.0e-4 {
        let omega = cosine.acos();
        let sine = omega.sin();
        from * (((1.0 - factor) * omega).sin() / sine) + to * ((factor * omega).sin() / sine)
    } else {
        from * (1.0 - factor) + to * factor
    }
}

fn flip_corner_attributes(mesh: &mut Mesh, flips: &[bool]) {
    let mut uv_layers = crate::geom::uv_layers::read_layers(mesh);
    for (face_index, flip) in flips.iter().enumerate() {
        if *flip {
            for layer in &mut uv_layers {
                if let Some(Some(values)) = layer.faces.get_mut(face_index) {
                    values[1..].reverse();
                }
                if let Some(Some(pinned)) = layer.pinned.get_mut(face_index) {
                    pinned[1..].reverse();
                }
            }
        }
    }

    let mut corner_start = 0;
    let mut permutation = Vec::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let count = face.vertices.len();
        let mut row = (0..count).collect::<Vec<_>>();
        if flips.get(face_index).copied().unwrap_or(false) {
            row[1..].reverse();
        }
        permutation.extend(row.into_iter().map(|corner| corner_start + corner));
        corner_start += count;
    }
    for (name, attribute) in &mut mesh.attributes {
        if name == "uv_layers" {
            continue;
        }
        if name == "blender_attributes" {
            if let Some(entries) = attribute.as_array_mut() {
                for entry in entries {
                    if entry.get("domain").and_then(Value::as_str) == Some("CORNER")
                        && let Some(values) = entry.get_mut("values").and_then(Value::as_array_mut)
                    {
                        reorder_values(values, &permutation);
                    }
                }
            }
            continue;
        }
        let is_corner = attribute
            .get("domain")
            .and_then(Value::as_str)
            .is_some_and(|domain| domain.eq_ignore_ascii_case("corner"));
        if !is_corner {
            continue;
        }
        if let Some(values) = attribute.get_mut("values").and_then(Value::as_object_mut) {
            for (face_index, flip) in flips.iter().enumerate() {
                if *flip
                    && let Some(row) = mesh
                        .faces
                        .get(face_index)
                        .and_then(|face| values.get_mut(&format!("f{}", face.id)))
                        .and_then(Value::as_array_mut)
                {
                    row[1..].reverse();
                }
            }
        } else if let Some(values) = attribute.get_mut("values").and_then(Value::as_array_mut) {
            reorder_values(values, &permutation);
        }
        if name == "custom_normal"
            && let Some(order) = attribute
                .get_mut("corner_order")
                .and_then(Value::as_object_mut)
        {
            for (face_index, flip) in flips.iter().enumerate() {
                if *flip
                    && let Some(row) = mesh
                        .faces
                        .get(face_index)
                        .and_then(|face| order.get_mut(&format!("f{}", face.id)))
                        .and_then(Value::as_array_mut)
                {
                    row[1..].reverse();
                }
            }
        }
    }
    crate::geom::uv_layers::write_layers(mesh, &uv_layers);
    let mut colors = crate::geom::color_layers::read(mesh);
    for layer in &mut colors.layers {
        if layer.domain == crate::geom::color_layers::ColorDomain::Corner
            && layer.values.len() == permutation.len()
        {
            let reordered = permutation
                .iter()
                .map(|&source| layer.values[source])
                .collect::<Vec<_>>();
            layer.values = reordered;
        }
    }
    crate::geom::color_layers::write(mesh, &colors);
}

fn reorder_values(values: &mut [Value], permutation: &[usize]) {
    if values.len() != permutation.len() {
        return;
    }
    let old = values.to_vec();
    for (destination, &source) in permutation.iter().enumerate() {
        values[destination] = old[source].clone();
    }
}
