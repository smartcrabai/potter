use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat3, DMat4, DQuat, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Edge, Face, Mesh, Vertex, edge_key,
        volume::{VolumeData, VolumeGeneratedSource, VolumeGrid, VolumeSource},
    },
    image::ImageData,
    sim::ParticleState,
};

/// Instantiates the current mesh at each visible particle position in the target object's space.
pub fn particle_instance(
    mesh: &Mesh,
    particles: &[ParticleState],
    source_world: DMat4,
    _target_world: DMat4,
    params: &Map<String, Value>,
    _frame: f64,
    fps: f64,
) -> Result<Mesh> {
    if !fps.is_finite() || fps <= 0.0 {
        return Err(invalid(
            "particle instance frame rate must be finite and positive",
        ));
    }
    let space = string_param(params, "space", "WORLD")?;
    if !matches!(space, "WORLD" | "LOCAL") {
        return Err(unsupported(
            "modifier.particle_instance.space",
            "unsupported particle instance space",
        ));
    }
    let inverse_source = if space == "LOCAL" {
        Some(invert(source_world)?)
    } else {
        None
    };
    let show_alive = bool_param(params, "show_alive", true)?;
    let show_dead = bool_param(params, "show_dead", false)?;
    let show_unborn = bool_param(params, "show_unborn", false)?;
    let use_children = bool_param(params, "use_children", false)?;
    let use_normal = bool_param(params, "use_normal", true)?;
    let use_size = bool_param(params, "use_size", false)?;
    if bool_param(params, "use_path", false)? {
        return Err(unsupported(
            "modifier.particle_instance.use_path",
            "Blender sampled paths require particle cache trajectories, which are not represented by the current particle states",
        ));
    }
    let axis = string_param(params, "axis", "X")?;
    let axis_index = match axis {
        "X" => 0,
        "Y" => 1,
        "Z" => 2,
        _ => return Err(invalid("axis must be X, Y, or Z")),
    };
    let track_axis = string_param(params, "track_axis", axis)?;
    let track_index = match track_axis {
        "X" | "POS_X" | "NEG_X" => 0,
        "Y" | "POS_Y" | "NEG_Y" => 1,
        "Z" | "POS_Z" | "NEG_Z" => 2,
        _ => return Err(invalid("track_axis must be X, Y, or Z")),
    };
    let orient_point = |point: DVec3| {
        let source = point.to_array();
        let mut destination = [0.0; 3];
        for offset in 0..3 {
            destination[(axis_index + offset) % 3] = source[(track_index + offset) % 3];
        }
        DVec3::from_array(destination)
    };
    let axis_matrix = DMat4::from_mat3(DMat3::from_cols(
        orient_point(DVec3::X),
        orient_point(DVec3::Y),
        orient_point(DVec3::Z),
    ));
    let mut transforms = Vec::new();
    for particle in particles {
        if particle.is_child {
            if !use_children {
                continue;
            }
        } else if !use_normal {
            continue;
        }
        let visible = match particle.life_state {
            crate::sim::ParticleLifeState::Alive => show_alive,
            crate::sim::ParticleLifeState::Dead => show_dead,
            crate::sim::ParticleLifeState::Unborn => show_unborn,
        };
        if !visible {
            continue;
        }
        let (world_translation, particle_rotation) =
            if particle.life_state == crate::sim::ParticleLifeState::Unborn {
                (
                    DVec3::from_array(particle.birth_position),
                    DQuat::from_array(particle.birth_rotation),
                )
            } else {
                (
                    DVec3::from_array(particle.position),
                    DQuat::from_array(particle.rotation),
                )
            };
        if !particle_rotation.is_finite() || particle_rotation.length_squared() <= 1.0e-20 {
            return Err(invalid(
                "particle instance rotation must be finite and nonzero",
            ));
        }
        let orientation = particle_rotation.normalize();
        let rotation = orientation;
        let scale = if use_size { particle.size } else { 1.0 };
        let instance_matrix =
            DMat4::from_scale_rotation_translation(DVec3::splat(scale), rotation, DVec3::ZERO)
                * axis_matrix;
        let (matrix, translation) = if space == "LOCAL" {
            let inverse_source = inverse_source
                .ok_or_else(|| invalid("local particle-instance transform is unavailable"))?;
            let transformed = inverse_source * instance_matrix;
            (
                DMat4::from_mat3(DMat3::from_cols(
                    transformed.x_axis.truncate(),
                    transformed.y_axis.truncate(),
                    transformed.z_axis.truncate(),
                )),
                inverse_source.transform_point3(world_translation),
            )
        } else {
            (instance_matrix, world_translation)
        };
        transforms.push((matrix, translation));
    }
    append_instances(mesh, &transforms)
}

/// Splits source faces into disconnected particle-driven fragments.
pub fn explode(
    mesh: &Mesh,
    particles: &[ParticleState],
    target_world: DMat4,
    params: &Map<String, Value>,
    _frame: f64,
    _lifetime: f64,
) -> Result<Mesh> {
    let show_alive = bool_param(params, "show_alive", true)?;
    let show_dead = bool_param(params, "show_dead", false)?;
    let show_unborn = bool_param(params, "show_unborn", false)?;
    let use_edge_cut = bool_param(params, "use_edge_cut", false)?;
    let use_size = bool_param(params, "use_size", false)?;
    let protect = number_param(params, "protect", 0.0)?;
    if !(0.0..=1.0).contains(&protect) {
        return Err(invalid("protect must be in [0, 1]"));
    }
    let alive = particles
        .iter()
        .filter(|particle| match particle.life_state {
            crate::sim::ParticleLifeState::Alive => show_alive,
            crate::sim::ParticleLifeState::Dead => show_dead,
            crate::sim::ParticleLifeState::Unborn => show_unborn,
        })
        .collect::<Vec<_>>();
    if alive.is_empty() {
        return Ok(Mesh::default());
    }
    let inverse_world = invert(target_world)?;
    let positions = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, (index, vertex.co)))
        .collect::<BTreeMap<_, _>>();
    let local_particles = alive
        .iter()
        .map(|particle| {
            (
                inverse_world.transform_point3(DVec3::from_array(particle.birth_position)),
                particle.size,
            )
        })
        .collect::<Vec<_>>();
    let face_particles = mesh
        .faces
        .iter()
        .map(|face| {
            let points = face
                .vertices
                .iter()
                .map(|id| {
                    positions
                        .get(id)
                        .map(|(_, point)| *point)
                        .ok_or_else(|| invalid("explode face references a missing vertex"))
                })
                .collect::<Result<Vec<_>>>()?;
            let center = points.iter().copied().sum::<DVec3>() / points.len() as f64;
            Ok(nearest_particle(center, &local_particles))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut vertex_particles = vec![alive.len(); mesh.vertices.len()];
    for (face, particle_index) in mesh.faces.iter().zip(&face_particles) {
        for vertex_id in &face.vertices {
            let vertex_index = positions
                .get(vertex_id)
                .map(|(index, _)| *index)
                .ok_or_else(|| invalid("explode face references a missing vertex"))?;
            vertex_particles[vertex_index] = *particle_index;
        }
    }
    let mut geometry_points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co)
        .collect::<Vec<_>>();
    let mut split_edges = BTreeMap::<(usize, usize), usize>::new();
    if use_edge_cut {
        for face in &mesh.faces {
            let vertex_indices = face
                .vertices
                .iter()
                .map(|vertex_id| {
                    positions
                        .get(vertex_id)
                        .map(|(index, _)| *index)
                        .ok_or_else(|| invalid("explode face references a missing vertex"))
                })
                .collect::<Result<Vec<_>>>()?;
            for index in 0..vertex_indices.len() {
                let first = vertex_indices[index];
                let second = vertex_indices[(index + 1) % vertex_indices.len()];
                if vertex_particles[first] != vertex_particles[second] {
                    add_split_edge(&mut split_edges, &mut geometry_points, first, second);
                }
            }
            if vertex_indices.len() == 4
                && (0..4).all(|index| {
                    vertex_particles[vertex_indices[index]]
                        != vertex_particles[vertex_indices[(index + 1) % 4]]
                })
            {
                add_split_edge(
                    &mut split_edges,
                    &mut geometry_points,
                    vertex_indices[0],
                    vertex_indices[2],
                );
            }
        }
    }

    let mut output = Mesh::default();
    let mut output_vertices = BTreeMap::<(usize, usize), u32>::new();
    let mut output_edges = BTreeSet::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let source_indices = face
            .vertices
            .iter()
            .map(|vertex_id| {
                positions
                    .get(vertex_id)
                    .map(|(index, _)| *index)
                    .ok_or_else(|| invalid("explode face references a missing vertex"))
            })
            .collect::<Result<Vec<_>>>()?;
        let face_particle = face_particles[face_index];
        let fragments = if use_edge_cut {
            edge_cut_fragments(
                &source_indices,
                &vertex_particles,
                &split_edges,
                face_particle,
            )?
        } else {
            vec![(source_indices, face_particle)]
        };
        let protection = (1.0 - protect) * vertex_group_weight(mesh, face, params);
        for (fragment, fragment_particle) in fragments {
            let particle = alive[fragment_particle];
            let current_position = DVec3::from_array(particle.position);
            let birth_position = DVec3::from_array(particle.birth_position);
            let current_rotation = DQuat::from_array(particle.rotation);
            let birth_rotation = DQuat::from_array(particle.birth_rotation);
            if !current_rotation.is_finite()
                || !birth_rotation.is_finite()
                || current_rotation.length_squared() <= 1.0e-20
                || birth_rotation.length_squared() <= 1.0e-20
            {
                return Err(invalid(
                    "explode particle rotations must be finite and nonzero",
                ));
            }
            let rotation_delta =
                current_rotation.normalize() * birth_rotation.normalize().inverse();
            let size = if use_size { particle.size } else { 1.0 };
            let mut expanded = Vec::with_capacity(fragment.len() * 2);
            for (first, second) in fragment
                .iter()
                .copied()
                .zip(fragment.iter().copied().cycle().skip(1))
                .take(fragment.len())
            {
                expanded.push(first);
                if let Some(midpoint) = split_edges.get(&edge_key(first, second)) {
                    expanded.push(*midpoint);
                }
            }
            let mut vertex_ids = Vec::with_capacity(expanded.len());
            for point_index in expanded {
                let key = (point_index, fragment_particle);
                let id = if let Some(id) = output_vertices.get(&key) {
                    *id
                } else {
                    let id = take_id(&mut output.next_id.vertex)?;
                    let source_world = target_world.transform_point3(geometry_points[point_index]);
                    let relative = source_world - birth_position;
                    let world_point =
                        current_position + rotation_delta * relative * (size * protection);
                    let position = inverse_world.transform_point3(world_point);
                    output.vertices.push(Vertex { id, co: position });
                    output_vertices.insert(key, id);
                    id
                };
                vertex_ids.push(id);
            }
            for (first, second) in super::cyclic_pairs(&vertex_ids) {
                let key = edge_key(*first, *second);
                if output_edges.insert(key) {
                    let id = take_id(&mut output.next_id.edge)?;
                    output.edges.push(Edge {
                        id,
                        vertices: [*first, *second],
                    });
                }
            }
            let id = take_id(&mut output.next_id.face)?;
            output.faces.push(Face {
                id,
                vertices: vertex_ids,
                material_index: face.material_index,
            });
        }
    }

    output
        .validate()
        .map_err(|error| invalid(&format!("explode produced invalid geometry: {error}")))?;
    Ok(output)
}
fn add_split_edge(
    split_edges: &mut BTreeMap<(usize, usize), usize>,
    points: &mut Vec<DVec3>,
    first: usize,
    second: usize,
) {
    split_edges
        .entry(edge_key(first, second))
        .or_insert_with(|| {
            let midpoint = (points[first] + points[second]) * 0.5;
            let index = points.len();
            points.push(midpoint);
            index
        });
}

fn edge_cut_fragments(
    vertices: &[usize],
    vertex_particles: &[usize],
    split_edges: &BTreeMap<(usize, usize), usize>,
    face_particle: usize,
) -> Result<Vec<(Vec<usize>, usize)>> {
    match vertices.len() {
        3 => triangle_cut_fragments(vertices, vertex_particles, split_edges),
        4 => quad_cut_fragments(vertices, vertex_particles, split_edges, face_particle),
        count if count > 4 => {
            let has_cut = (1..count - 1).any(|index| {
                let triangle = [vertices[0], vertices[index], vertices[index + 1]];
                [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ]
                .into_iter()
                .any(|(first, second)| split_edges.contains_key(&edge_key(first, second)))
            });
            if !has_cut {
                return Ok(vec![(vertices.to_vec(), face_particle)]);
            }
            let mut fragments = Vec::new();
            for index in 1..count - 1 {
                fragments.extend(triangle_cut_fragments(
                    &[vertices[0], vertices[index], vertices[index + 1]],
                    vertex_particles,
                    split_edges,
                )?);
            }
            Ok(fragments)
        }
        _ => Err(invalid(
            "explode faces must contain at least three vertices",
        )),
    }
}

fn triangle_cut_fragments(
    vertices: &[usize],
    vertex_particles: &[usize],
    split_edges: &BTreeMap<(usize, usize), usize>,
) -> Result<Vec<(Vec<usize>, usize)>> {
    let [first, second, third] = vertices else {
        return Err(invalid(
            "explode triangle must contain exactly three vertices",
        ));
    };
    let edge = |a: usize, b: usize| {
        split_edges
            .get(&edge_key(a, b))
            .copied()
            .ok_or_else(|| invalid("explode triangle cut edge is missing its midpoint"))
    };
    let mask = usize::from(split_edges.contains_key(&edge_key(*first, *second)))
        | (usize::from(split_edges.contains_key(&edge_key(*second, *third))) << 1)
        | (usize::from(split_edges.contains_key(&edge_key(*third, *first))) << 2);
    let ordered = match mask {
        0 | 5 | 7 => [*first, *second, *third],
        3 => [*second, *third, *first],
        6 => [*third, *first, *second],
        _ => {
            return Err(invalid(
                "explode triangle has an impossible edge-cut pattern",
            ));
        }
    };
    let [a, b, c] = ordered;
    let owner_a = vertex_particles[a];
    let owner_b = vertex_particles[b];
    let owner_c = vertex_particles[c];
    match mask {
        0 => Ok(vec![(vec![a, b, c], owner_a)]),
        3 | 5 | 6 => Ok(vec![
            (vec![a, edge(a, b)?, edge(a, c)?], owner_a),
            (vec![edge(a, b)?, b, c, edge(a, c)?], owner_b),
        ]),
        7 => Ok(vec![
            (vec![a, edge(a, b)?, edge(b, c)?, edge(a, c)?], owner_a),
            (vec![edge(a, b)?, b, edge(b, c)?], owner_b),
            (vec![edge(a, c)?, edge(b, c)?, c], owner_c),
        ]),
        _ => Err(invalid(
            "explode triangle has an unsupported edge-cut pattern",
        )),
    }
}

fn quad_cut_fragments(
    vertices: &[usize],
    vertex_particles: &[usize],
    split_edges: &BTreeMap<(usize, usize), usize>,
    face_particle: usize,
) -> Result<Vec<(Vec<usize>, usize)>> {
    let [first, second, third, fourth] = vertices else {
        return Err(invalid("explode quad must contain exactly four vertices"));
    };
    let original = [*first, *second, *third, *fourth];
    let mut mask = 0;
    for index in 0..4 {
        if split_edges.contains_key(&edge_key(original[index], original[(index + 1) % 4])) {
            mask |= 1 << index;
        }
    }
    if mask == 0 {
        return Ok(vec![(vertices.to_vec(), face_particle)]);
    }
    let order = match mask {
        3 | 10 | 11 | 15 => [0, 1, 2, 3],
        5..=7 => [1, 2, 3, 0],
        9 | 13 => [3, 0, 1, 2],
        12 | 14 => [2, 3, 0, 1],
        _ => return Err(invalid("explode quad has an impossible edge-cut pattern")),
    };
    let [a, b, c, d] = order.map(|index| original[index]);
    let owner = |vertex: usize| vertex_particles[vertex];
    let edge = |first: usize, second: usize| {
        split_edges
            .get(&edge_key(first, second))
            .copied()
            .ok_or_else(|| invalid("explode quad cut edge is missing its midpoint"))
    };
    match mask {
        3 | 6 | 9 | 12 => Ok(vec![
            (vec![a, edge(a, b)?, edge(b, c)?, c], owner(a)),
            (vec![edge(a, b)?, b, edge(b, c)?], owner(b)),
            (vec![a, c, d], owner(a)),
        ]),
        5 | 10 => Ok(vec![
            (vec![a, b, edge(b, c)?, edge(a, d)?], owner(a)),
            (vec![edge(a, d)?, edge(b, c)?, c, d], owner(c)),
        ]),
        15 => Ok(vec![
            (vec![a, edge(a, b)?, edge(a, c)?, edge(a, d)?], owner(a)),
            (vec![edge(a, b)?, b, edge(b, c)?, edge(a, c)?], owner(b)),
            (vec![edge(a, c)?, edge(b, c)?, c, edge(c, d)?], owner(c)),
            (vec![edge(a, d)?, edge(a, c)?, edge(c, d)?, d], owner(d)),
        ]),
        7 | 11 | 13 | 14 => Ok(vec![
            (vec![a, edge(a, b)?, edge(b, c)?, edge(a, d)?], owner(a)),
            (vec![edge(a, b)?, b, edge(b, c)?], owner(b)),
            (vec![edge(a, d)?, edge(b, c)?, c, d], owner(d)),
        ]),
        _ => Err(invalid("explode quad has an unsupported edge-cut pattern")),
    }
}

/// Resamples inline or VDB scalar volume grids using mapped texture color as a displacement vector.
pub fn displace_volume(
    volume: &VolumeData,
    params: &Map<String, Value>,
    texture_type: &str,
    image: Option<&ImageData>,
    volume_world: DMat4,
    texture_map_object_world: Option<DMat4>,
) -> Result<VolumeData> {
    let strength = number_param(params, "strength", 0.5)?;
    if !matches!(texture_type, "NONE" | "NOISE" | "IMAGE") {
        return Err(unsupported(
            "modifier.volume_displace.texture_type",
            "only deterministic noise and image textures are supported",
        ));
    }
    if texture_type == "NONE" || strength == 0.0 {
        return Ok(volume.clone());
    }
    let texture = VolumeDisplacement::new(
        params,
        texture_type,
        image,
        strength,
        volume_world,
        texture_map_object_world,
    )?;
    match &volume.source {
        VolumeSource::Generated(_) => {
            let mut displaced = volume.clone();
            if displaced.grids.is_empty() {
                return Err(invalid(
                    "volume displacement requires at least one scalar grid",
                ));
            }
            for grid in &mut displaced.grids {
                displace_inline_grid(grid, &texture)?;
            }
            Ok(displaced)
        }
        VolumeSource::File(source) if source.format == "vdb" => {
            displace_vdb_volume(volume, &texture)
        }
        VolumeSource::File(source) => Err(unsupported(
            "modifier.volume_displace.file_format",
            &format!(
                "file-backed volume format `{}` is not supported",
                source.format
            ),
        )),
    }
}

struct VolumeDisplacement<'a> {
    texture_type: &'a str,
    image: Option<&'a ImageData>,
    mapping: DMat4,
    strength: f64,
    mid_level: [f64; 3],
    sample_radius: f64,
}

impl<'a> VolumeDisplacement<'a> {
    fn new(
        params: &Map<String, Value>,
        texture_type: &'a str,
        image: Option<&'a ImageData>,
        strength: f64,
        volume_world: DMat4,
        texture_map_object_world: Option<DMat4>,
    ) -> Result<Self> {
        if texture_type == "IMAGE" && image.is_none() {
            return Err(unsupported(
                "modifier.volume_displace.image_texture",
                "image texture source is unavailable",
            ));
        }
        let sample_radius = number_param(params, "texture_sample_radius", 1.0)?;
        if sample_radius < 0.0 {
            return Err(invalid("texture_sample_radius must be non-negative"));
        }
        if sample_radius > 16.0 {
            return Err(limit(
                "texture_sample_radius exceeds the 16-voxel output padding limit",
            ));
        }
        let mode = string_param(params, "texture_map_mode", "LOCAL")?;
        let mapping = match mode {
            "LOCAL" => DMat4::IDENTITY,
            "GLOBAL" => volume_world,
            "OBJECT" => {
                let object_world = texture_map_object_world.ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::InvalidOperation,
                        "OBJECT texture mapping requires texture_map_object",
                        json!({"feature_id":"modifier.volume_displace.texture_map_object"}),
                    )
                })?;
                invert_transform(object_world)? * volume_world
            }
            _ => {
                return Err(invalid("texture_map_mode must be LOCAL, GLOBAL, or OBJECT"));
            }
        };
        if !mapping.is_finite() || mapping.determinant().abs() <= 1.0e-15 {
            return Err(invalid(
                "volume texture coordinate transform must be finite and invertible",
            ));
        }
        Ok(Self {
            texture_type,
            image,
            mapping,
            strength,
            mid_level: vector_param(params, "texture_mid_level", [0.5; 3])?,
            sample_radius,
        })
    }

    fn displacement_at(&self, local_position: DVec3) -> Result<DVec3> {
        let texture_position = self.mapping.transform_point3(local_position);
        if !texture_position.is_finite() {
            return Err(invalid("volume texture coordinates are not finite"));
        }
        let color = match self.texture_type {
            "NOISE" => value_noise_color(texture_position),
            "IMAGE" => {
                let image = self
                    .image
                    .ok_or_else(|| invalid("image texture source disappeared"))?;
                // Blender's flat Image Texture projection maps object coordinates from
                // [-1, 1] to UVs and repeats outside the unit square.
                let uv = [
                    (texture_position.x * 0.5 + 0.5).rem_euclid(1.0),
                    (texture_position.y * 0.5 + 0.5).rem_euclid(1.0),
                ];
                let rgba = sample_repeated_image(image, uv)?;
                [rgba[0], rgba[1], rgba[2]]
            }
            _ => {
                return Err(unsupported(
                    "modifier.volume_displace.texture_type",
                    "unsupported volume displacement texture source",
                ));
            }
        };
        Ok(DVec3::new(
            color[0] - self.mid_level[0],
            color[1] - self.mid_level[1],
            color[2] - self.mid_level[2],
        ) * self.strength)
    }
}
fn sample_repeated_image(image: &ImageData, uv: [f64; 2]) -> Result<[f64; 4]> {
    let width = usize::try_from(image.width)
        .map_err(|_| limit("image texture width exceeds addressable memory"))?;
    let height = usize::try_from(image.height)
        .map_err(|_| limit("image texture height exceeds addressable memory"))?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| limit("image texture pixel count overflows"))?;
    if width == 0 || height == 0 || image.pixels.len() != pixel_count {
        return Err(invalid("image texture dimensions do not match its pixels"));
    }
    if uv.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(invalid("image texture coordinates must be finite"));
    }
    let width_f64 = f64::from(image.width);
    let height_f64 = f64::from(image.height);
    let width_i64 = i64::from(image.width);
    let height_i64 = i64::from(image.height);
    if image.interpolation == crate::image::ImageInterpolation::Closest {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "wrapped coordinates are bounded by the u32 image dimensions"
        )]
        let x = (uv[0].rem_euclid(1.0) * width_f64).floor() as i64;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "wrapped coordinates are bounded by the u32 image dimensions"
        )]
        let y = (uv[1].rem_euclid(1.0) * height_f64).floor() as i64;
        let x = usize::try_from(x.rem_euclid(width_i64))
            .map_err(|_| invalid("image texture x coordinate is invalid"))?;
        let y = usize::try_from(y.rem_euclid(height_i64))
            .map_err(|_| invalid("image texture y coordinate is invalid"))?;
        return Ok(image.pixels[y * width + x]);
    }
    let x = uv[0].rem_euclid(1.0) * width_f64 - 0.5;
    let y = uv[1].rem_euclid(1.0) * height_f64 - 0.5;
    let x_floor = x.floor();
    let y_floor = y.floor();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "wrapped coordinates are bounded by the u32 image dimensions"
    )]
    let x0 = x_floor as i64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "wrapped coordinates are bounded by the u32 image dimensions"
    )]
    let y0 = y_floor as i64;
    let x1 = (x0 + 1).rem_euclid(width_i64);
    let y1 = (y0 + 1).rem_euclid(height_i64);
    let x0 = usize::try_from(x0.rem_euclid(width_i64))
        .map_err(|_| invalid("image texture x coordinate is invalid"))?;
    let y0 = usize::try_from(y0.rem_euclid(height_i64))
        .map_err(|_| invalid("image texture y coordinate is invalid"))?;
    let x1 = usize::try_from(x1).map_err(|_| invalid("image texture x coordinate is invalid"))?;
    let y1 = usize::try_from(y1).map_err(|_| invalid("image texture y coordinate is invalid"))?;
    let fraction_x = x - x_floor;
    let fraction_y = y - y_floor;
    let top: [f64; 4] = std::array::from_fn(|channel| {
        let first = image.pixels[y0 * width + x0][channel];
        let second = image.pixels[y0 * width + x1][channel];
        first + (second - first) * fraction_x
    });
    let bottom: [f64; 4] = std::array::from_fn(|channel| {
        let first = image.pixels[y1 * width + x0][channel];
        let second = image.pixels[y1 * width + x1][channel];
        first + (second - first) * fraction_x
    });
    Ok(std::array::from_fn(|channel| {
        top[channel] + (bottom[channel] - top[channel]) * fraction_y
    }))
}

fn displace_inline_grid(grid: &mut VolumeGrid, texture: &VolumeDisplacement<'_>) -> Result<()> {
    if let Some(blob) = &grid.blob_f32 {
        if grid.values.is_some() {
            return Err(invalid(
                "volume grid must not contain both values and blob_f32",
            ));
        }
        if blob.len() % 4 != 0 {
            return Err(invalid("volume blob_f32 length must be a multiple of four"));
        }
        grid.values = Some(
            blob.as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect(),
        );
        grid.blob_f32 = None;
    }
    let original = grid
        .values
        .as_ref()
        .ok_or_else(|| invalid("volume grid has no inline samples"))?
        .clone();
    let source_dims = grid.dims.map(|dimension| dimension as usize);
    let source_count = source_dims
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| invalid("volume grid sample count overflows"))?;
    if source_count == 0 || original.len() != source_count {
        return Err(invalid(
            "volume grid sample count does not match its dimensions",
        ));
    }
    if !grid.voxel_size.is_finite() || grid.voxel_size <= 0.0 || !grid.origin.is_finite() {
        return Err(invalid(
            "volume grid transform must be finite with positive voxel_size",
        ));
    }
    let dilation_radius = texture.sample_radius * texture.strength.abs() / grid.voxel_size / 2.0;
    if !dilation_radius.is_finite() || dilation_radius > 16.0 {
        return Err(limit("volume displacement dilation exceeds 16 voxels"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated dilation radius is bounded to 16"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "validated dilation radius is non-negative"
    )]
    let padding = dilation_radius.ceil() as usize;
    let padding_total = padding
        .checked_mul(2)
        .ok_or_else(|| limit("volume displacement padding overflows"))?;
    let mut output_dims = [0_usize; 3];
    for axis in 0..3 {
        output_dims[axis] = source_dims[axis]
            .checked_add(padding_total)
            .ok_or_else(|| limit("volume displacement dimensions overflow"))?;
    }
    if output_dims.contains(&0) || output_dims.iter().any(|dimension| *dimension > 512) {
        return Err(limit(
            "displaced volume grid exceeds the 512-voxel axis limit",
        ));
    }
    let output_count = output_dims
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| limit("displaced volume grid sample count overflows"))?;
    if output_count > 1_000_000 {
        return Err(limit("displaced volume grid exceeds one million samples"));
    }
    let original_origin = grid.origin;
    let voxel_size = DVec3::splat(grid.voxel_size);
    let output_origin = original_origin - voxel_size * padding as f64;
    let mut output = Vec::with_capacity(output_count);
    for z in 0..output_dims[2] {
        for y in 0..output_dims[1] {
            for x in 0..output_dims[0] {
                let local_position =
                    output_origin + voxel_size * DVec3::new(x as f64, y as f64, z as f64);
                let displacement = texture.displacement_at(local_position)?;
                let source_position = local_position - displacement;
                let source_coordinate = (source_position - original_origin)
                    .to_array()
                    .map(|value| value / grid.voxel_size);
                let value = sample_volume_grid(&original, source_dims, source_coordinate);
                if !value.is_finite() || value > f64::from(f32::MAX) {
                    return Err(invalid("volume displacement produced a non-finite sample"));
                }
                output.push(value.max(0.0) as f32);
            }
        }
    }
    grid.dims = [
        u32::try_from(output_dims[0]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[1]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
        u32::try_from(output_dims[2]).map_err(|_| limit("volume grid dimension exceeds limits"))?,
    ];
    grid.origin = output_origin;
    grid.values = Some(output);
    grid.blob_f32 = None;
    Ok(())
}

fn displace_vdb_volume(
    volume: &VolumeData,
    texture: &VolumeDisplacement<'_>,
) -> Result<VolumeData> {
    let decoded = volume.decoded_vdb.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "file-backed VDB volume is not decoded for evaluation",
            json!({"feature_id":"volume.openvdb_evaluation.resource"}),
        )
    })?;
    if decoded.grids.len() > 1 {
        return Err(unsupported(
            "modifier.volume_displace.vdb_multiple_grids",
            "Volume Displace supports one scalar VDB density grid at a time",
        ));
    }
    let Some(grid) = decoded.density_grid() else {
        return Err(unsupported(
            "modifier.volume_displace.vdb_density_grid",
            "VDB volume has no scalar density grid to displace",
        ));
    };
    if grid.value_type.split('_').nth(1) != Some("float") {
        return Err(unsupported(
            "modifier.volume_displace.vdb_grid_type",
            "only scalar float VDB density grids can be displaced",
        ));
    }
    let Some((active_minimum, active_maximum)) = grid.active_bbox else {
        return Ok(volume.clone());
    };
    let voxel_size = grid.voxel_size().max_element();
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        return Err(invalid(
            "VDB density grid transform must be finite with positive voxel_size",
        ));
    }
    let dilation_radius =
        (texture.sample_radius * texture.strength.abs() / voxel_size / 2.0).ceil();
    if !dilation_radius.is_finite() || dilation_radius > 16.0 {
        return Err(limit("VDB displacement dilation exceeds 16 voxels"));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated ceil radius is integral and bounded to 16"
    )]
    let padding = dilation_radius as i32;
    let mut output_minimum = [0_i32; 3];
    let mut output_maximum = [0_i32; 3];
    let mut dimensions = [0_usize; 3];
    for axis in 0..3 {
        output_minimum[axis] = active_minimum[axis]
            .checked_sub(padding)
            .ok_or_else(|| limit("VDB displacement minimum index overflows"))?;
        output_maximum[axis] = active_maximum[axis]
            .checked_add(padding)
            .ok_or_else(|| limit("VDB displacement maximum index overflows"))?;
        let length = i64::from(output_maximum[axis]) - i64::from(output_minimum[axis]) + 1;
        dimensions[axis] = usize::try_from(length)
            .ok()
            .filter(|dimension| *dimension > 0 && *dimension <= 512)
            .ok_or_else(|| limit("displaced VDB grid exceeds the 512-voxel axis limit"))?;
    }
    let sample_count = dimensions
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| limit("displaced VDB grid sample count overflows"))?;
    if sample_count > 1_000_000 {
        return Err(limit("displaced VDB grid exceeds one million samples"));
    }
    let index_to_world = grid.transform.index_to_world;
    let output_origin = index_to_world.transform_point3(DVec3::new(
        f64::from(output_minimum[0]),
        f64::from(output_minimum[1]),
        f64::from(output_minimum[2]),
    ));
    let mut values = Vec::with_capacity(sample_count);
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let index = [
                    output_minimum[0]
                        + i32::try_from(x)
                            .map_err(|_| limit("VDB output x index exceeds signed range"))?,
                    output_minimum[1]
                        + i32::try_from(y)
                            .map_err(|_| limit("VDB output y index exceeds signed range"))?,
                    output_minimum[2]
                        + i32::try_from(z)
                            .map_err(|_| limit("VDB output z index exceeds signed range"))?,
                ];
                let index_position = DVec3::new(
                    f64::from(index[0]),
                    f64::from(index[1]),
                    f64::from(index[2]),
                );
                let object_position = index_to_world.transform_point3(index_position);
                let displacement = texture.displacement_at(object_position)? / voxel_size;
                let source_position =
                    index_to_world.transform_point3(index_position - displacement);
                let value = grid.sample_world(source_position).max(0.0);
                if !value.is_finite() || value > f64::from(f32::MAX) {
                    return Err(invalid("VDB displacement produced a non-finite density"));
                }
                values.push(value as f32);
            }
        }
    }
    let mut displaced = volume.clone();
    displaced.source = VolumeSource::Generated(VolumeGeneratedSource {
        algorithm: "volume_displace_vdb".to_owned(),
    });
    displaced.decoded_vdb = None;
    displaced.grids = vec![VolumeGrid {
        dims: [
            u32::try_from(dimensions[0]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
            u32::try_from(dimensions[1]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
            u32::try_from(dimensions[2]).map_err(|_| limit("VDB grid dimension exceeds limits"))?,
        ],
        voxel_size,
        origin: output_origin,
        values: Some(values),
        ..VolumeGrid::default()
    }];
    Ok(displaced)
}

fn append_instances(mesh: &Mesh, transforms: &[(DMat4, DVec3)]) -> Result<Mesh> {
    if transforms.is_empty() {
        return Ok(Mesh::default());
    }
    let vertex_count = mesh
        .vertices
        .len()
        .checked_mul(transforms.len())
        .ok_or_else(|| invalid("particle instance vertex count overflows"))?;
    let edge_count = mesh
        .edges
        .len()
        .checked_mul(transforms.len())
        .ok_or_else(|| invalid("particle instance edge count overflows"))?;
    let face_count = mesh
        .faces
        .len()
        .checked_mul(transforms.len())
        .ok_or_else(|| invalid("particle instance face count overflows"))?;
    let mut output = Mesh::default();
    output.vertices.reserve(vertex_count);
    output.edges.reserve(edge_count);
    output.faces.reserve(face_count);
    for (matrix, translation) in transforms {
        let mut remap = BTreeMap::new();
        for vertex in &mesh.vertices {
            let id = take_id(&mut output.next_id.vertex)?;
            let position = matrix.transform_point3(vertex.co) + *translation;
            output.vertices.push(Vertex { id, co: position });
            remap.insert(vertex.id, id);
        }
        for edge in &mesh.edges {
            let id = take_id(&mut output.next_id.edge)?;
            let vertices = [
                *remap
                    .get(&edge.vertices[0])
                    .ok_or_else(|| invalid("particle instance edge references a missing vertex"))?,
                *remap
                    .get(&edge.vertices[1])
                    .ok_or_else(|| invalid("particle instance edge references a missing vertex"))?,
            ];
            output.edges.push(Edge { id, vertices });
        }
        for face in &mesh.faces {
            let id = take_id(&mut output.next_id.face)?;
            let vertices = face
                .vertices
                .iter()
                .map(|vertex_id| {
                    remap.get(vertex_id).copied().ok_or_else(|| {
                        invalid("particle instance face references a missing vertex")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            output.faces.push(Face {
                id,
                vertices,
                material_index: face.material_index,
            });
        }
    }
    output.validate().map_err(|error| {
        invalid(&format!(
            "particle instancing produced invalid geometry: {error}"
        ))
    })?;
    Ok(output)
}

fn value_noise_color(point: DVec3) -> [f64; 3] {
    if !point.is_finite() || point.abs().max_element() > 9.0e18 {
        return [0.5; 3];
    }
    let floor = point.floor();
    let base = floor.to_array().map(|value| value as i64);
    let mut fade = (point - floor).to_array();
    for component in &mut fade {
        *component = *component * *component * (3.0 - 2.0 * *component);
    }
    std::array::from_fn(|channel| {
        let mut value = 0.0;
        for z in 0..=1 {
            for y in 0..=1 {
                for x in 0..=1 {
                    let cell = [base[0] + x, base[1] + y, base[2] + z];
                    let weight = (if x == 0 { 1.0 - fade[0] } else { fade[0] })
                        * (if y == 0 { 1.0 - fade[1] } else { fade[1] })
                        * (if z == 0 { 1.0 - fade[2] } else { fade[2] });
                    value += hash_value_noise(cell, channel) * weight;
                }
            }
        }
        value
    })
}

fn sample_volume_grid(samples: &[f32], dims: [usize; 3], coordinate: [f64; 3]) -> f64 {
    if coordinate.iter().any(|value| !value.is_finite())
        || (0..3).any(|axis| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "volume grid dimensions are bounded to 512 samples per axis"
            )]
            let maximum = dims[axis] as f64;
            coordinate[axis] < -1.0 || coordinate[axis] > maximum
        })
    {
        return 0.0;
    }
    let lower: [i64; 3] = coordinate.map(|value| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bounded finite coordinates are floored to integer voxel indices"
        )]
        {
            value.floor() as i64
        }
    });
    let fraction = std::array::from_fn::<_, 3, _>(|axis| {
        #[expect(
            clippy::cast_precision_loss,
            reason = "voxel indices are bounded to the 512-sample grid limit"
        )]
        let lower = lower[axis] as f64;
        coordinate[axis] - lower
    });
    let sample = |point: [i64; 3]| {
        if (0..3).any(|axis| {
            #[expect(
                clippy::cast_possible_wrap,
                reason = "volume dimensions originate as u32 and fit in i64"
            )]
            let dimension = dims[axis] as i64;
            point[axis] < 0 || point[axis] >= dimension
        }) {
            return 0.0;
        }
        let [Ok(x), Ok(y), Ok(z)] = point.map(usize::try_from) else {
            return 0.0;
        };
        f64::from(samples[(z * dims[1] + y) * dims[0] + x])
    };
    let lerp = |first: f64, second: f64, amount: f64| first + (second - first) * amount;
    let x00 = lerp(
        sample(lower),
        sample([lower[0] + 1, lower[1], lower[2]]),
        fraction[0],
    );
    let x10 = lerp(
        sample([lower[0], lower[1] + 1, lower[2]]),
        sample([lower[0] + 1, lower[1] + 1, lower[2]]),
        fraction[0],
    );
    let x01 = lerp(
        sample([lower[0], lower[1], lower[2] + 1]),
        sample([lower[0] + 1, lower[1], lower[2] + 1]),
        fraction[0],
    );
    let x11 = lerp(
        sample([lower[0], lower[1] + 1, lower[2] + 1]),
        sample([lower[0] + 1, lower[1] + 1, lower[2] + 1]),
        fraction[0],
    );
    lerp(
        lerp(x00, x10, fraction[1]),
        lerp(x01, x11, fraction[1]),
        fraction[2],
    )
}

fn hash_value_noise(point: [i64; 3], channel: usize) -> f64 {
    let mut value = (point[0] as u64).wrapping_mul(0x9e37_79b9)
        ^ (point[1] as u64).wrapping_mul(0x85eb_ca6b)
        ^ (point[2] as u64).wrapping_mul(0xc2b2_ae35)
        ^ (channel as u64).wrapping_mul(0x27d4_eb2d);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    (value as f64) / (u64::MAX as f64)
}

fn vertex_group_weight(mesh: &Mesh, face: &Face, params: &Map<String, Value>) -> f64 {
    let Some(group_name) = params.get("vertex_group").and_then(Value::as_str) else {
        return 1.0;
    };
    let Some(group) = mesh
        .attributes
        .get("vertex_groups")
        .and_then(|groups| groups.get(group_name))
        .and_then(Value::as_object)
    else {
        return 0.0;
    };
    let mut total = 0.0;
    let mut count = 0.0;
    for id in &face.vertices {
        total += group
            .get(&format!("v{id}"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        count += 1.0;
    }
    if count == 0.0 { 0.0 } else { total / count }
}

fn take_id(next: &mut u32) -> Result<u32> {
    let id = *next;
    *next = next.checked_add(1).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "modifier output exhausted mesh element IDs",
        )
    })?;
    Ok(id)
}
fn nearest_particle(point: DVec3, particles: &[(DVec3, f64)]) -> usize {
    let point = point.as_vec3();
    let mut nearest = 0;
    let mut nearest_distance = f32::INFINITY;
    for (index, (position, _)) in particles.iter().enumerate() {
        let distance = position.as_vec3().distance_squared(point);
        if index == 0 || distance < nearest_distance {
            nearest = index;
            nearest_distance = distance;
        }
    }
    nearest
}

fn invert_transform(matrix: DMat4) -> Result<DMat4> {
    if !matrix.is_finite() || matrix.determinant().abs() <= 1.0e-15 {
        return Err(invalid(
            "texture mapping object transform must be finite and invertible",
        ));
    }
    Ok(matrix.inverse())
}

fn invert(matrix: DMat4) -> Result<DMat4> {
    if !matrix.is_finite() || matrix.determinant().abs() <= 1.0e-15 {
        return Err(invalid(
            "particle object transform must be finite and invertible",
        ));
    }
    Ok(matrix.inverse())
}

fn bool_param(params: &Map<String, Value>, name: &str, default: bool) -> Result<bool> {
    match params.get(name) {
        Some(Value::Bool(value)) => Ok(*value),
        None => Ok(default),
        _ => Err(invalid(&format!("{name} must be a boolean"))),
    }
}

fn number_param(params: &Map<String, Value>, name: &str, default: f64) -> Result<f64> {
    let value = params.get(name).and_then(Value::as_f64).unwrap_or(default);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(&format!("{name} must be finite")))
    }
}

fn vector_param(params: &Map<String, Value>, name: &str, default: [f64; 3]) -> Result<[f64; 3]> {
    let Some(values) = params.get(name) else {
        return Ok(default);
    };
    let Some(values) = values.as_array().filter(|values| values.len() == 3) else {
        return Err(invalid(&format!(
            "{name} must contain three finite numbers"
        )));
    };
    let mut result = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        result[index] = value
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid(&format!("{name} must contain three finite numbers")))?;
    }
    Ok(result)
}

fn string_param<'a>(
    params: &'a Map<String, Value>,
    name: &str,
    default: &'a str,
) -> Result<&'a str> {
    let value = params.get(name).and_then(Value::as_str).unwrap_or(default);
    if value.is_empty() {
        Err(invalid(&format!("{name} must not be empty")))
    } else {
        Ok(value)
    }
}

fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

fn unsupported(feature_id: &str, reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":feature_id,"reason":reason,"status":"not_supported"}),
    )
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DVec3};
    use serde_json::{Map, json};

    use super::{displace_volume, particle_instance};
    use crate::{
        geom::{
            primitive,
            volume::{VolumeData, VolumeGrid},
        },
        sim::ParticleState,
    };

    #[test]
    fn instances_duplicate_mesh_apply_normal_orientation_and_translate_to_particle_coordinates()
    -> crate::error::Result<()> {
        let mesh = primitive("plane", &json!({}))
            .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
        let particles = vec![ParticleState {
            position: [2.0, 0.0, 0.0],
            rotation: [
                0.0,
                -std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            velocity: [0.0, 0.0, 0.0],
            birth_frame: 1.0,
            ..ParticleState::default()
        }];
        let result = particle_instance(
            &mesh,
            &particles,
            DMat4::IDENTITY,
            DMat4::IDENTITY,
            &Map::new(),
            1.0,
            20.0,
        )?;
        assert_eq!(result.vertices.len(), mesh.vertices.len());
        assert_eq!(result.faces.len(), mesh.faces.len());
        let expected = [
            DVec3::new(2.0, -1.0, -1.0),
            DVec3::new(2.0, -1.0, 1.0),
            DVec3::new(2.0, 1.0, 1.0),
            DVec3::new(2.0, 1.0, -1.0),
        ];
        for (vertex, expected) in result.vertices.iter().zip(expected) {
            assert!(
                vertex.co.distance(expected) <= 1.0e-12,
                "instance vertex {}: actual={}, expected={expected}",
                vertex.id,
                vertex.co,
            );
        }
        Ok(())
    }

    #[test]
    fn particle_instance_selects_parent_particles_and_orients_their_axis()
    -> crate::error::Result<()> {
        let mesh = primitive("plane", &json!({}))
            .map_err(|error| crate::error::PotError::invalid_argument(error.to_string()))?;
        let particle = ParticleState {
            position: [0.0; 3],
            rotation: [
                0.0,
                -std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            velocity: [0.0, 0.0, 0.0],
            birth_frame: 1.0,
            ..ParticleState::default()
        };
        let result = particle_instance(
            &mesh,
            std::slice::from_ref(&particle),
            DMat4::IDENTITY,
            DMat4::IDENTITY,
            &Map::from_iter([
                ("use_normal".to_owned(), json!(true)),
                ("axis".to_owned(), json!("X")),
            ]),
            1.0,
            24.0,
        )?;
        let bounds = result.bounds().ok_or_else(|| {
            crate::error::PotError::new(
                crate::error::ErrorCode::EvaluationFailed,
                "oriented particle instance has no bounds",
            )
        })?;
        assert!(bounds.min.x.abs() <= 1.0e-12);
        assert!(bounds.max.x.abs() <= 1.0e-12);
        assert!((bounds.min.y + 1.0).abs() <= 1.0e-12);
        assert!((bounds.max.y - 1.0).abs() <= 1.0e-12);
        assert!((bounds.min.z + 1.0).abs() <= 1.0e-12);
        assert!((bounds.max.z - 1.0).abs() <= 1.0e-12);
        let no_parents = particle_instance(
            &mesh,
            std::slice::from_ref(&particle),
            DMat4::IDENTITY,
            DMat4::IDENTITY,
            &Map::from_iter([("use_normal".to_owned(), json!(false))]),
            1.0,
            24.0,
        )?;
        assert!(no_parents.vertices.is_empty());
        Ok(())
    }

    #[test]
    fn volume_displace_resamples_and_rejects_unknown_textures() -> crate::error::Result<()> {
        let volume = VolumeData {
            grids: vec![VolumeGrid {
                dims: [3, 1, 1],
                values: Some(vec![0.0, 1.0, 0.0]),
                ..VolumeGrid::default()
            }],
            ..VolumeData::default()
        };
        let unchanged = displace_volume(&volume, &Map::new(), "NONE", None, DMat4::IDENTITY, None)?;
        assert_eq!(
            unchanged.grids[0].values.as_deref(),
            Some(&[0.0_f32, 1.0, 0.0][..])
        );

        let image = crate::image::ImageData {
            width: 1,
            height: 1,
            pixels: vec![[1.0, 0.5, 0.5, 1.0]],
            tiles: std::collections::BTreeMap::new(),
            interpolation: crate::image::ImageInterpolation::Linear,
        };
        let displaced = displace_volume(
            &volume,
            &Map::from_iter([
                ("texture_map_mode".into(), json!("LOCAL")),
                ("strength".into(), json!(1.0)),
                ("texture_mid_level".into(), json!([0.5, 0.5, 0.5])),
                ("texture_sample_radius".into(), json!(0.0)),
            ]),
            "IMAGE",
            Some(&image),
            DMat4::IDENTITY,
            None,
        )?;
        assert_eq!(
            displaced.grids[0].values.as_deref(),
            Some(&[0.0_f32, 0.5, 0.5][..])
        );
        let Err(error) =
            displace_volume(&volume, &Map::new(), "VORONOI", None, DMat4::IDENTITY, None)
        else {
            return Err(crate::error::PotError::new(
                crate::error::ErrorCode::InternalError,
                "unsupported texture type unexpectedly succeeded",
            ));
        };
        assert_eq!(error.code, crate::error::ErrorCode::UnsupportedFeature);
        Ok(())
    }
}
