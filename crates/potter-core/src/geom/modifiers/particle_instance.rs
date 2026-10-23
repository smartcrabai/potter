// SPDX-License-Identifier: GPL-3.0-or-later

//! Blender's Particle Instance modifier (`MOD_particleinstance.cc`) over native particle states.
//!
//! Particle order follows Blender's `psys->particles` then `psys->child`: parents are the states
//! with `is_child == false`, children follow in list order. Selection uses the exact
//! particle-system RNG (`psys_frand`), so `particle_amount`/`particle_offset` choose the same
//! instances as Blender for the same particle count and system seed.

use std::collections::HashMap;

use glam::{DMat4, DQuat, DVec3};
use serde_json::{Map, Value};

use crate::{
    blender_f32,
    blender_rng::{hash_frand, psys_frand},
    error::{ErrorCode, PotError, Result},
    geom::{
        Edge, Face, Mesh, Vertex,
        color_layers::{self, ColorDomain, ColorLayer},
        provenance::{Origin, remap_attributes},
    },
    sim::{ParticleLifeState, ParticlePathInterpolation, ParticlePathKey, ParticleState},
};

use super::simulation::{bool_param, number_param, string_param, value_or_default};

const KIND: &str = "particle_instance";

/// Blender's `unit_float_to_uchar_clamp`, used by `store_float_in_vcol`.
fn unit_float_to_byte(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value > 1.0 - 0.5 / 255.0 {
        255
    } else {
        (255.0 * value + 0.5) as u8
    }
}

/// Blender's byte color stores grey RGB and raw alpha `1/255`; the canonical store is linear RGBA.
fn byte_color(value: f32) -> [f64; 4] {
    // Blender stores the grey byte and raw alpha byte 1; its `Color Attribute` readback decodes them
    // in f32 (`srgb_to_linearrgb_uchar4`, `math_color.cc`). `color_layers` keeps that op order, so
    // this matches Blender's bits where f64 decoding differs by about 3 ulp.
    let byte = f32::from(unit_float_to_byte(value));
    color_layers::byte_color_decoded_components([byte, byte, byte, 1.0])
}

/// Instances the source `mesh` at each selected particle of `particles`.
///
/// `system_seed` is Blender's `ParticleSystem.seed`, `source_world` is the particle object's
/// world matrix (used by `space = LOCAL`), and `track_flag` is the instancing object's
/// `Object.track_axis` (Blender `trackflag`, 0..=5). Blender leaves the mesh unchanged when the
/// particle system has no parents or when neither parents nor children are selected.
pub fn particle_instance(
    mesh: &Mesh,
    particles: &[ParticleState],
    system_seed: u32,
    source_world: DMat4,
    track_flag: u32,
    params: &Map<String, Value>,
) -> Result<Mesh> {
    let use_parents = bool_param(KIND, params, "use_normal")?;
    let use_children = bool_param(KIND, params, "use_children")?;
    let use_size = bool_param(KIND, params, "use_size")?;
    let show_alive = bool_param(KIND, params, "show_alive")?;
    let show_dead = bool_param(KIND, params, "show_dead")?;
    let show_unborn = bool_param(KIND, params, "show_unborn")?;
    let use_path = bool_param(KIND, params, "use_path")?;
    let preserve_shape = bool_param(KIND, params, "use_preserve_shape")?;
    let position = number_param(KIND, params, "position")? as f32;
    let random_position = number_param(KIND, params, "random_position")? as f32;
    let rotation = number_param(KIND, params, "rotation")? as f32;
    let random_rotation = number_param(KIND, params, "random_rotation")? as f32;
    let axis = match string_param(KIND, params, "axis")? {
        "X" => 0,
        "Y" => 1,
        "Z" => 2,
        _ => return Err(invalid("axis must be X, Y, or Z")),
    };
    let local_space = match string_param(KIND, params, "space")? {
        "WORLD" => false,
        "LOCAL" => true,
        _ => return Err(invalid("space must be WORLD or LOCAL")),
    };
    let amount = number_param(KIND, params, "particle_amount")? as f32;
    let offset = number_param(KIND, params, "particle_offset")? as f32;
    let index_layer = layer_name(params, "index_layer_name")?;
    let value_layer = layer_name(params, "value_layer_name")?;

    let total_parents = particles.iter().take_while(|state| !state.is_child).count();
    let total_children = particles.len() - total_parents;
    if total_parents == 0 {
        return Ok(mesh.clone());
    }
    // Blender's `part_start`/`part_end` particle-number range.
    let part_start = if use_parents { 0 } else { total_parents };
    let part_end =
        if use_parents { total_parents } else { 0 } + if use_children { total_children } else { 0 };
    if part_end == 0 {
        return Ok(mesh.clone());
    }
    let source_inverse = if local_space {
        let inverse = source_world.inverse();
        if !inverse
            .to_cols_array()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(invalid("particle source transform must be invertible"));
        }
        Some(inverse)
    } else {
        None
    };

    let total = particles.len();
    let state_at = |number: usize| &particles[number];
    let skipped = |number: usize| -> bool {
        let state = state_at(number);
        let parent = if state.parent_count > 0 {
            usize::try_from(state.parent_indices[0])
                .ok()
                .and_then(|index| particles.get(index))
                .unwrap_or(state)
        } else {
            state
                .parent_index
                .and_then(|index| usize::try_from(index).ok())
                .and_then(|index| particles.get(index))
                .unwrap_or(state)
        };
        match parent.life_state {
            ParticleLifeState::Unborn if !show_unborn => return true,
            ParticleLifeState::Alive if !show_alive => return true,
            ParticleLifeState::Dead if !show_dead => return true,
            _ => {}
        }
        if amount.total_cmp(&1.0).is_eq() {
            return false;
        }
        let count = total as f32;
        let random_index = truncate(psys_frand(system_seed, 3578 + number as u32) * count) % total;
        let minimum = truncate(count * offset) % (total + 1);
        let maximum = truncate(count * (offset + amount)) % (total + 1);
        match maximum.cmp(&minimum) {
            std::cmp::Ordering::Greater => random_index < minimum || random_index >= maximum,
            std::cmp::Ordering::Less => random_index < minimum && random_index >= maximum,
            std::cmp::Ordering::Equal => true,
        }
    };

    let vertex_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let edge_pairs = mesh
        .edges
        .iter()
        .map(|edge| {
            Ok([
                lookup_vertex(&vertex_index, edge.vertices[0])?,
                lookup_vertex(&vertex_index, edge.vertices[1])?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let face_pairs = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| lookup_vertex(&vertex_index, *id))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let corners_per_instance = if index_layer.is_empty() && value_layer.is_empty() {
        0
    } else {
        face_pairs.iter().try_fold(0_usize, |total, face| {
            total
                .checked_add(face.len())
                .ok_or_else(|| limit("particle instance corner count overflows"))
        })?
    };
    let track = (track_flag % 3) as usize;
    let bounds = mesh.bounds();
    let (min_co, max_co) = bounds.map_or((0.0, 0.0), |bounds| {
        (bounds.min[track] as f32, bounds.max[track] as f32)
    });
    let track_negative = track_flag > 2;
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(part_end.saturating_sub(part_start))
        .map_err(|_| limit("particle instance selection exceeds memory limits"))?;
    for number in part_start..part_end {
        if !skipped(number) {
            selected.push(number);
        }
    }
    let vertex_count = mesh
        .vertices
        .len()
        .checked_mul(selected.len())
        .ok_or_else(|| limit("particle instance vertex count overflows"))?;
    let edge_count = mesh
        .edges
        .len()
        .checked_mul(selected.len())
        .ok_or_else(|| limit("particle instance edge count overflows"))?;
    let face_count = mesh
        .faces
        .len()
        .checked_mul(selected.len())
        .ok_or_else(|| limit("particle instance face count overflows"))?;
    let corner_count = corners_per_instance
        .checked_mul(selected.len())
        .ok_or_else(|| limit("particle instance corner count overflows"))?;
    let mut output = Mesh::default();
    output
        .vertices
        .try_reserve_exact(vertex_count)
        .map_err(|_| limit("particle instance vertex allocation exceeds memory limits"))?;
    output
        .edges
        .try_reserve_exact(edge_count)
        .map_err(|_| limit("particle instance edge allocation exceeds memory limits"))?;
    output
        .faces
        .try_reserve_exact(face_count)
        .map_err(|_| limit("particle instance face allocation exceeds memory limits"))?;
    let mut vertex_sources = Vec::new();
    vertex_sources
        .try_reserve_exact(vertex_count)
        .map_err(|_| limit("particle instance provenance allocation exceeds memory limits"))?;
    let mut face_sources = Vec::new();
    face_sources
        .try_reserve_exact(face_count)
        .map_err(|_| limit("particle instance provenance allocation exceeds memory limits"))?;
    let mut index_corners = Vec::new();
    index_corners
        .try_reserve_exact(corner_count)
        .map_err(|_| limit("particle instance color allocation exceeds memory limits"))?;
    let mut value_corners = Vec::new();
    value_corners
        .try_reserve_exact(corner_count)
        .map_err(|_| limit("particle instance color allocation exceeds memory limits"))?;
    for number in selected {
        let state = state_at(number);
        let has_path = use_path && !state.path.is_empty();
        let particle_rotation = if has_path {
            [1.0, 0.0, 0.0, 0.0]
        } else {
            let [x, y, z, w] = state.rotation.map(|value| value as f32);
            let rotation = [w, x, y, z];
            let length_squared = rotation.iter().map(|value| value * value).sum::<f32>();
            if !rotation.iter().all(|value| value.is_finite()) || length_squared <= 1.0e-20 {
                return Err(invalid(
                    "particle instance rotation must be finite and nonzero",
                ));
            }
            rotation
        };
        let path_frame_seed = if has_path {
            let rotation =
                DQuat::from_array(state.path_root_rotation.unwrap_or(state.path[0].rotation));
            if !rotation.is_finite() || rotation.length_squared() <= 1.0e-20 {
                return Err(invalid(
                    "particle path root rotation must be finite and nonzero",
                ));
            }
            rotation.normalize()
        } else {
            DQuat::IDENTITY
        };
        // Blender scales by `size[p]`, one array filled with parent sizes (if used) then child sizes
        // (if used). Without parents that array starts at child 0, so child number p reads the
        // size of child p, while the instanced state is child p - totpart.
        let scale = if !use_size {
            1.0
        } else if use_parents {
            state.size
        } else {
            particles[total_parents + number].size
        };
        let random_position_offset = if has_path && random_position != 0.0 {
            random_position * hash_frand(system_seed.wrapping_add(number as u32))
        } else {
            0.0
        };
        let spin = if has_path && (rotation > 0.0 || random_rotation > 0.0) {
            let random_angle = if random_rotation == 0.0 {
                0.0
            } else {
                random_rotation * (psys_frand(system_seed, 19_957_323 + number as u32) - 0.5)
            };
            let angle = (std::f64::consts::TAU * f64::from(rotation + random_angle)) as f32;
            DQuat::from_rotation_z(f64::from(angle))
        } else {
            DQuat::IDENTITY
        };
        let mut path_frame = None;
        let mut previous_direction: Option<DVec3> = None;
        let base = output.vertices.len();
        for (source, vertex) in mesh.vertices.iter().enumerate() {
            let mut oriented = orient(vertex.co, axis, track);
            let transform = if has_path {
                let mut path_time = if preserve_shape {
                    position * (1.0 - random_position_offset)
                } else if !max_co.total_cmp(&min_co).is_eq() {
                    ((oriented.to_array()[axis] as f32 - min_co) / (max_co - min_co))
                        * position
                        * (1.0 - random_position_offset)
                } else {
                    0.0
                };
                if !preserve_shape {
                    if track_negative {
                        path_time = 1.0 - path_time;
                    }
                    let mut coordinates = oriented.to_array();
                    coordinates[axis] = 0.0;
                    oriented = DVec3::from_array(coordinates);
                }
                let (position, velocity) =
                    sample_path(&state.path, path_time, state.path_interpolation);
                let direction = velocity.normalize_or_zero();
                let frame = if source == 0 {
                    path_frame_seed * spin
                } else {
                    let frame = path_frame.unwrap_or(path_frame_seed * spin);
                    match previous_direction {
                        Some(previous)
                            if previous.length_squared() > f64::EPSILON
                                && direction.length_squared() > f64::EPSILON =>
                        {
                            DQuat::from_rotation_arc(previous, direction) * frame
                        }
                        _ => frame,
                    }
                };
                path_frame = Some(frame);
                previous_direction = Some(direction);
                Some((frame, position))
            } else {
                None
            };
            let mut point = if let Some((orientation, translation)) = transform {
                orientation * oriented * scale + translation
            } else {
                let source_position = blender_f32::vec3(oriented.to_array());
                let mut point = blender_f32::mul_qt_v3(particle_rotation, source_position);
                if use_size {
                    point = blender_f32::scale3(point, scale as f32);
                }
                let translation = state.position.map(|value| value as f32);
                DVec3::from_array([
                    f64::from(point[0] + translation[0]),
                    f64::from(point[1] + translation[1]),
                    f64::from(point[2] + translation[2]),
                ])
            };
            if let Some(inverse) = source_inverse {
                point = inverse.transform_point3(point);
            }
            if !point.is_finite() {
                return Err(invalid("particle instance produced non-finite coordinates"));
            }
            let id = allocate(&mut output.next_id.vertex)?;
            output.vertices.push(Vertex { id, co: point });
            vertex_sources.push(Some(source));
        }
        for [first, second] in &edge_pairs {
            let id = allocate(&mut output.next_id.edge)?;
            output.edges.push(Edge {
                id,
                vertices: [
                    output.vertices[base + first].id,
                    output.vertices[base + second].id,
                ],
            });
        }
        for (face_index, (face, source_vertices)) in mesh.faces.iter().zip(&face_pairs).enumerate()
        {
            let id = allocate(&mut output.next_id.face)?;
            output.faces.push(Face {
                id,
                vertices: source_vertices
                    .iter()
                    .map(|vertex| output.vertices[base + vertex].id)
                    .collect(),
                material_index: face.material_index,
            });
            face_sources.push(Some(face_index));
        }
        if !index_layer.is_empty() {
            let color = byte_color((number as f32) / (total_parents as f32 - 1.0));
            index_corners.extend(std::iter::repeat_n(color, corners_per_instance));
        }
        if !value_layer.is_empty() {
            let color = byte_color(psys_frand(system_seed, 77091 + 283 * number as u32));
            value_corners.extend(std::iter::repeat_n(color, corners_per_instance));
        }
    }
    remap_attributes(
        mesh,
        &mut output,
        &Origin::copies(vertex_sources, face_sources),
    )?;
    store_corner_layer(&mut output, &index_layer, index_corners)?;
    store_corner_layer(&mut output, &value_layer, value_corners)?;
    output.validate().map_err(|error| {
        invalid(&format!(
            "particle instancing produced invalid geometry: {error}"
        ))
    })?;
    Ok(output)
}

/// Writes a named CORNER color layer (`lookup_or_add` in Blender); an empty name adds nothing.
fn store_corner_layer(mesh: &mut Mesh, name: &str, values: Vec<[f64; 4]>) -> Result<()> {
    if name.is_empty() {
        return Ok(());
    }
    let mut store = color_layers::read(mesh);
    if let Some(index) = store.layers.iter().position(|layer| layer.name == name) {
        let is_byte_corner = store.layers[index].domain == ColorDomain::Corner
            && mesh
                .attributes
                .get("blender_metadata")
                .and_then(|metadata| metadata.get("color_data_types"))
                .and_then(|types| types.get(name))
                .and_then(Value::as_str)
                == Some("BYTE_COLOR");
        if !is_byte_corner {
            return Ok(());
        }
        store.layers[index] = ColorLayer {
            name: name.to_owned(),
            domain: ColorDomain::Corner,
            values,
        };
    } else {
        store.layers.push(ColorLayer {
            name: name.to_owned(),
            domain: ColorDomain::Corner,
            values,
        });
        let metadata = mesh
            .attributes
            .entry("blender_metadata".to_owned())
            .or_insert_with(|| serde_json::json!({}));
        let metadata = metadata
            .as_object_mut()
            .ok_or_else(|| invalid("blender_metadata must be an object"))?;
        let data_types = metadata
            .entry("color_data_types".to_owned())
            .or_insert_with(|| serde_json::json!({}));
        let data_types = data_types
            .as_object_mut()
            .ok_or_else(|| invalid("blender_metadata.color_data_types must be an object"))?;
        data_types.insert(name.to_owned(), Value::String("BYTE_COLOR".to_owned()));
    }
    color_layers::write(mesh, &store);
    Ok(())
}

fn layer_name(params: &Map<String, Value>, name: &str) -> Result<String> {
    value_or_default(KIND, params, name)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(&format!("{name} must be a string")))
}

/// Interpolates Blender's particle path position and normalized-time derivative.
fn sample_path(
    keys: &[ParticlePathKey],
    time: f32,
    interpolation: ParticlePathInterpolation,
) -> (DVec3, DVec3) {
    let time = f64::from(time.clamp(0.0, 1.0));
    if keys.len() == 1 {
        let key = keys[0];
        return (
            DVec3::from_array(key.position),
            DVec3::from_array(key.velocity),
        );
    }
    let upper = keys.partition_point(|key| key.time < time);
    if interpolation != ParticlePathInterpolation::HairBspline {
        if upper == 0 {
            let key = keys[0];
            return (
                DVec3::from_array(key.position),
                DVec3::from_array(key.velocity),
            );
        }
        if upper == keys.len() {
            let key = keys[keys.len() - 1];
            return (
                DVec3::from_array(key.position),
                DVec3::from_array(key.velocity),
            );
        }
    }
    let (segment, factor) = if upper == 0 {
        (0, 0.0)
    } else if upper == keys.len() {
        (keys.len() - 2, 1.0)
    } else {
        let segment = upper - 1;
        let duration = keys[segment + 1].time - keys[segment].time;
        if duration <= 0.0 {
            let key = keys[segment];
            return (
                DVec3::from_array(key.position),
                DVec3::from_array(key.velocity),
            );
        }
        (
            segment,
            ((time - keys[segment].time) / duration).clamp(0.0, 1.0),
        )
    };
    let first = keys[segment];
    let second = keys[segment + 1];
    let duration = second.time - first.time;
    if duration <= 0.0 {
        return (
            DVec3::from_array(first.position),
            DVec3::from_array(first.velocity),
        );
    }
    let (position, velocity) = if interpolation == ParticlePathInterpolation::HairBspline {
        let t2 = factor * factor;
        let t3 = t2 * factor;
        let one_minus_t = 1.0 - factor;
        let controls = [
            DVec3::from_array(keys[segment.saturating_sub(1)].position),
            DVec3::from_array(first.position),
            DVec3::from_array(second.position),
            DVec3::from_array(keys[(segment + 2).min(keys.len() - 1)].position),
        ];
        let weights = [
            one_minus_t.powi(3) / 6.0,
            (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
            (-3.0 * t3 + 3.0 * t2 + 3.0 * factor + 1.0) / 6.0,
            t3 / 6.0,
        ];
        let derivatives = [
            -0.5 * one_minus_t * one_minus_t,
            (9.0 * t2 - 12.0 * factor) / 6.0,
            (-9.0 * t2 + 6.0 * factor + 3.0) / 6.0,
            0.5 * t2,
        ];
        let position = controls
            .iter()
            .zip(weights)
            .fold(DVec3::ZERO, |sum, (control, weight)| {
                sum + *control * weight
            });
        let velocity = controls
            .iter()
            .zip(derivatives)
            .fold(DVec3::ZERO, |sum, (control, derivative)| {
                sum + *control * derivative
            })
            / duration;
        (position, velocity)
    } else if matches!(
        interpolation,
        ParticlePathInterpolation::Linear | ParticlePathInterpolation::HairCache
    ) {
        (
            DVec3::from_array(first.position).lerp(DVec3::from_array(second.position), factor),
            DVec3::from_array(first.velocity).lerp(DVec3::from_array(second.velocity), factor),
        )
    } else {
        let first_position = DVec3::from_array(first.position);
        let second_position = DVec3::from_array(second.position);
        let first_velocity = DVec3::from_array(first.velocity);
        let second_velocity = DVec3::from_array(second.velocity);
        let t2 = factor * factor;
        let t3 = t2 * factor;
        let position = first_position * (2.0 * t3 - 3.0 * t2 + 1.0)
            + first_velocity * (duration * (t3 - 2.0 * t2 + factor))
            + second_position * (-2.0 * t3 + 3.0 * t2)
            + second_velocity * (duration * (t3 - t2));
        let velocity = first_position * ((6.0 * t2 - 6.0 * factor) / duration)
            + first_velocity * (3.0 * t2 - 4.0 * factor + 1.0)
            + second_position * ((-6.0 * t2 + 6.0 * factor) / duration)
            + second_velocity * (3.0 * t2 - 2.0 * factor);
        (position, velocity)
    };
    (position, velocity)
}

/// C `int(value)` for the non-negative products used by the selection; negatives clamp to zero.
fn truncate(value: f32) -> usize {
    if value.is_finite() && value > 0.0 {
        value as usize
    } else {
        0
    }
}

/// `MOD_particleinstance.cc`: `positions[axis] = co[track]`, then the cyclic remainder.
fn orient(local: DVec3, axis: usize, track: usize) -> DVec3 {
    let source = local.to_array();
    let mut destination = [0.0; 3];
    destination[axis] = source[track];
    destination[(axis + 1) % 3] = source[(track + 1) % 3];
    destination[(axis + 2) % 3] = source[(track + 2) % 3];
    DVec3::from_array(destination)
}

fn lookup_vertex(index: &HashMap<u32, usize>, id: u32) -> Result<usize> {
    index
        .get(&id)
        .copied()
        .ok_or_else(|| invalid("particle instance face references a missing vertex"))
}

fn allocate(next: &mut u32) -> Result<u32> {
    let id = *next;
    *next = next.checked_add(1).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "particle instance element IDs are exhausted",
        )
    })?;
    Ok(id)
}

fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DQuat, DVec3};
    use serde_json::{Map, json};

    use crate::{
        geom::{Mesh, color_layers},
        sim::{ParticleLifeState, ParticlePathKey, ParticleState},
    };

    use super::particle_instance;

    #[test]
    fn matches_blender_522_particle_amount_size_track_and_byte_color_oracle()
    -> crate::error::Result<()> {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(-1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
        let particles = [
            ParticleState {
                position: [-1.0, -1.0, 0.0],
                life_state: ParticleLifeState::Alive,
                size: 0.05,
                ..ParticleState::default()
            },
            ParticleState {
                position: [1.0, -1.0, 0.0],
                life_state: ParticleLifeState::Unborn,
                size: 0.05,
                ..ParticleState::default()
            },
            ParticleState {
                position: [-1.0, -1.0, 0.0],
                life_state: ParticleLifeState::Unborn,
                size: 0.05,
                ..ParticleState::default()
            },
            ParticleState {
                position: [1.0, 1.0, 0.0],
                life_state: ParticleLifeState::Unborn,
                size: 0.05,
                ..ParticleState::default()
            },
        ];
        let params = Map::from_iter([
            ("particle_amount".to_owned(), json!(0.5)),
            ("particle_offset".to_owned(), json!(0.25)),
            ("use_size".to_owned(), json!(true)),
            ("index_layer_name".to_owned(), json!("idx")),
            ("value_layer_name".to_owned(), json!("val")),
        ]);
        let result = particle_instance(&mesh, &particles, 7, DMat4::IDENTITY, 1, &params)?;
        let expected = [
            [1.0, -1.05, -0.05],
            [1.0, -0.95, -0.05],
            [1.0, -0.95, 0.05],
            [1.0, -1.05, 0.05],
            [-1.0, -1.05, -0.05],
            [-1.0, -0.95, -0.05],
            [-1.0, -0.95, 0.05],
            [-1.0, -1.05, 0.05],
        ];
        assert_eq!(result.vertices.len(), expected.len());
        assert_eq!(result.faces.len(), 2);
        assert_eq!(result.faces[0].vertices, vec![0, 1, 2, 3]);
        assert_eq!(result.faces[1].vertices, vec![4, 5, 6, 7]);
        for (vertex, expected) in result.vertices.iter().zip(expected) {
            assert!(
                vertex.co.distance(DVec3::from_array(expected)) < 1.0e-7,
                "vertex {}: actual {}, expected {:?}",
                vertex.id,
                vertex.co,
                expected,
            );
        }
        let colors = color_layers::read(&result);
        let index = colors
            .index_of("idx")
            .and_then(|index| colors.layers.get(index))
            .ok_or_else(|| crate::error::PotError::invalid_argument("missing index color"))?;
        let value = colors
            .index_of("val")
            .and_then(|index| colors.layers.get(index))
            .ok_or_else(|| crate::error::PotError::invalid_argument("missing value color"))?;
        assert_eq!(index.domain, color_layers::ColorDomain::Corner);
        assert_eq!(value.domain, color_layers::ColorDomain::Corner);
        assert_eq!(index.values.len(), 8);
        assert_eq!(value.values.len(), 8);
        for color in &index.values[..4] {
            assert!((color[0] - 0.090_841_71).abs() < 1.0e-7);
            assert!((color[3] - 1.0 / 255.0).abs() < 1.0e-8);
        }
        for color in &index.values[4..] {
            assert!((color[0] - 0.401_977_87).abs() < 1.0e-7);
        }
        for color in &value.values[..4] {
            assert!(
                (color[0] - 0.610_495_75).abs() < 1.0e-7,
                "value colour {} differs from 0.61049575",
                color[0]
            );
        }
        for color in &value.values[4..] {
            assert!((color[0] - 0.258_182_91).abs() < 1.0e-7);
        }
        let encodings = result
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .ok_or_else(|| crate::error::PotError::invalid_argument("missing color encoding"))?;
        assert_eq!(
            encodings.get("idx").and_then(serde_json::Value::as_str),
            Some("BYTE_COLOR")
        );
        assert_eq!(
            encodings.get("val").and_then(serde_json::Value::as_str),
            Some("BYTE_COLOR")
        );
        Ok(())
    }

    #[test]
    fn samples_blender_hair_path_keys_at_particle_instance_position() -> crate::error::Result<()> {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
            vec![vec![0, 1, 2, 3]],
        )
        .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
        let path =
            [(0.0, 0.0), (0.25, 0.6), (0.5, 1.2), (0.75, 1.8), (1.0, 2.4)].map(|(time, z)| {
                ParticlePathKey {
                    time,
                    position: [-1.166_753_053_665_161_1, -0.512_642_920_017_242_4, z],
                    velocity: [0.0, 0.0, 1.0],
                    rotation: DQuat::IDENTITY.to_array(),
                }
            });
        let state = ParticleState {
            path: path.to_vec(),
            ..ParticleState::default()
        };
        let params = Map::from_iter([
            ("use_path".to_owned(), json!(true)),
            ("use_preserve_shape".to_owned(), json!(true)),
            ("position".to_owned(), json!(0.5)),
        ]);
        let result = particle_instance(&mesh, &[state], 31, DMat4::IDENTITY, 1, &params)?;
        assert_eq!(result.vertices.len(), 4);
        assert!(
            result.vertices[0].co.distance(DVec3::new(
                -1.166_753_053_665_161_1,
                -0.512_642_920_017_242_4,
                1.2,
            )) < 1.0e-7
        );
        Ok(())
    }

    #[test]
    fn child_only_range_with_fewer_children_is_empty() -> crate::error::Result<()> {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
        let particles = [
            ParticleState::default(),
            ParticleState::default(),
            ParticleState::default(),
            ParticleState {
                is_child: true,
                parent_index: Some(0),
                ..ParticleState::default()
            },
        ];
        let params = Map::from_iter([
            ("use_normal".to_owned(), json!(false)),
            ("use_children".to_owned(), json!(true)),
        ]);

        let output = particle_instance(&mesh, &particles, 0, DMat4::IDENTITY, 1, &params)?;

        assert_eq!(output.vertices.len(), 0);
        assert_eq!(output.edges.len(), 0);
        assert_eq!(output.faces.len(), 0);
        Ok(())
    }
}
