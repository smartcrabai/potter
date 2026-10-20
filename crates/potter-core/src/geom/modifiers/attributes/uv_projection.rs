//! UV project and UV warp modifiers, including projector cameras and bone references.

use super::*;
use glam::{Vec2, Vec3};

fn active_uv_layer_index(store: &crate::geom::uv_layers::UvStore) -> Option<usize> {
    store
        .active
        .as_deref()
        .and_then(|name| store.index_of(name))
        .or((!store.layers.is_empty()).then_some(0))
}

pub(super) fn uv_project(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
    subject_world: DMat4,
) -> Result<Mesh> {
    let Some(projectors) = operands.get("projectors").filter(|items| !items.is_empty()) else {
        return Ok(mesh.clone());
    };
    let aspect_x = (number(modifier, "aspect_x")? as f32).max(1.0);
    let aspect_y = (number(modifier, "aspect_y")? as f32).max(1.0);
    // Blender clamps these to zero, then treats zero as the default scale of one.
    let scale_x = (number(modifier, "scale_x")? as f32).max(0.0);
    let scale_y = (number(modifier, "scale_y")? as f32).max(0.0);
    let scale_x = if scale_x == 0.0 { 1.0 } else { scale_x };
    let scale_y = if scale_y == 0.0 { 1.0 } else { scale_y };
    // `projectors_num` (MOD_uvproject.cc) limits which projector slots are read; later slots are ignored.
    let projector_count = number(modifier, "projector_count")?.max(1.0) as usize;

    let requested_layer = string(modifier, "uv_layer")?;
    let mut store = crate::geom::uv_layers::read(mesh);
    let existing_index = if requested_layer.is_empty() {
        active_uv_layer_index(&store)
    } else {
        store.index_of(requested_layer)
    };
    let layer_index = if let Some(index) = existing_index {
        index
    } else {
        if store.layers.len() >= crate::geom::uv_layers::MAX_LAYERS {
            return Err(invalid(modifier, "uv_layer", "an available UV layer slot"));
        }
        let name = if requested_layer.is_empty() {
            "Float2"
        } else {
            requested_layer
        };
        store.layers.push(crate::geom::uv_layers::UvLayer::new(
            name,
            vec![None; mesh.faces.len()],
        ));
        store.layers.len() - 1
    };
    let subject_world_f32 = subject_world.to_cols_array().map(|value| value as f32);
    let world_positions = mesh
        .vertices
        .iter()
        .map(|vertex| {
            crate::geom::blender_matrix::blender_transform_point_f32(
                &subject_world_f32,
                Vec3::new(vertex.co.x as f32, vertex.co.y as f32, vertex.co.z as f32),
            )
        })
        .collect::<Vec<_>>();
    let projector_data = projectors
        .iter()
        .take(projector_count)
        .map(|projector| {
            make_projector_f32(
                projector,
                aspect_x as f32,
                aspect_y as f32,
                scale_x as f32,
                scale_y as f32,
                modifier,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let vertex_indices = super::super::id_indices(mesh);
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let face_vertex_indices = face
            .vertices
            .iter()
            .map(|vertex_id| {
                vertex_indices
                    .get(vertex_id)
                    .copied()
                    .ok_or_else(|| invalid(modifier, "projectors", "valid face vertices"))
            })
            .collect::<Result<Vec<_>>>()?;
        let normal = face_normal_f32(mesh, &face_vertex_indices);
        // Blender keeps the first projector when two camera directions tie. Normals and dots
        // stay f32 and the projector normal retains its object scale, as in MOD_uvproject.cc.
        let mut best_index = 0;
        let mut best_dot = projector_data[0].normal.x * normal.x
            + projector_data[0].normal.y * normal.y
            + projector_data[0].normal.z * normal.z;
        for (index, projector) in projector_data.iter().enumerate().skip(1) {
            let dot = projector.normal.x * normal.x
                + projector.normal.y * normal.y
                + projector.normal.z * normal.z;
            if dot > best_dot {
                best_index = index;
                best_dot = dot;
            }
        }
        let projection = &projector_data[best_index].projection;
        let mut coords = Vec::with_capacity(face.vertices.len());
        for vertex_index in face_vertex_indices {
            let uv_up = project_world_point_f32(world_positions[vertex_index], projection);
            coords.push([f64::from(uv_up.x), f64::from(1.0_f32 - uv_up.y)]);
        }
        store.layers[layer_index].faces[face_index] = Some(coords);
    }
    let mut result = mesh.clone();
    crate::geom::uv_layers::write_layers(&mut result, &store.layers);
    Ok(result)
}
fn normalize_vector_blender_f32(vector: Vec3) -> (Vec3, f32) {
    let length_squared = vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
    if length_squared > 1.0e-35_f32 {
        let length = length_squared.sqrt();
        (vector * (1.0 / length), length)
    } else {
        (Vec3::ZERO, 0.0)
    }
}

fn face_normal_f32(mesh: &Mesh, indices: &[usize]) -> Vec3 {
    let point = |index: usize| {
        let value = mesh.vertices[index].co;
        Vec3::new(value.x as f32, value.y as f32, value.z as f32)
    };
    let normal = match indices {
        [a, b, c, d] => (point(*a) - point(*c)).cross(point(*b) - point(*d)),
        [a, b, c] => (point(*a) - point(*b)).cross(point(*b) - point(*c)),
        values if values.len() > 4 => {
            let mut normal = Vec3::ZERO;
            let mut previous = values.last().map_or(Vec3::ZERO, |&index| point(index));
            for &index in values {
                let current = point(index);
                normal.x += (previous.y - current.y) * (previous.z + current.z);
                normal.y += (previous.z - current.z) * (previous.x + current.x);
                normal.z += (previous.x - current.x) * (previous.y + current.y);
                previous = current;
            }
            normal
        }
        _ => Vec3::ZERO,
    };
    let (normal, length) = normalize_vector_blender_f32(normal);
    if length == 0.0 { Vec3::Z } else { normal }
}

enum ProjectorProjection {
    Matrix([f32; 16]),
    Panorama {
        world_to_camera: [f32; 16],
        aspect_x: f32,
        aspect_y: f32,
        shift_x: f32,
        shift_y: f32,
        camera_size: f32,
    },
}

struct ProjectorF32 {
    normal: Vec3,
    projection: ProjectorProjection,
}

fn make_projector_f32(
    projector: &SceneOperand,
    aspect_x: f32,
    aspect_y: f32,
    scale_x: f32,
    scale_y: f32,
    modifier: &Modifier,
) -> Result<ProjectorF32> {
    let world = projector
        .blender_matrices_f32
        .as_ref()
        .and_then(|matrices| matrices.world)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::EvaluationFailed,
                "UV Project projector world matrix is missing from the evaluated scene context",
                json!({"modifier_id":modifier.id}),
            )
        })?;
    let normal = Vec3::new(world[8], world[9], world[10]);
    let Some(camera) = &projector.camera else {
        let world_to_projector = crate::geom::blender_matrix::blender_inverse_m4(world);
        if matrix_is_zero(&world_to_projector) {
            return Err(invalid(
                modifier,
                "projectors",
                "invertible projector transforms",
            ));
        }
        let offset = uv_projection_offset_matrix();
        return Ok(ProjectorF32 {
            normal,
            projection: ProjectorProjection::Matrix(
                crate::geom::blender_matrix::blender_multiply_m4(offset, world_to_projector),
            ),
        });
    };

    if matches!(camera.projection, crate::model::CameraProjection::Panorama) {
        let normalized_world = normalize_camera_world_f32(world);
        let world_to_camera = crate::geom::blender_matrix::blender_inverse_m4(normalized_world);
        if matrix_is_zero(&world_to_camera) {
            return Err(invalid(
                modifier,
                "projectors",
                "invertible projector transforms",
            ));
        }
        let (base_aspect_x, base_aspect_y) = if aspect_x > aspect_y {
            (1.0, aspect_x / aspect_y)
        } else {
            (aspect_y / aspect_x, 1.0)
        };
        return Ok(ProjectorF32 {
            normal,
            projection: ProjectorProjection::Panorama {
                world_to_camera,
                aspect_x: base_aspect_x * scale_x,
                aspect_y: base_aspect_y * scale_y,
                shift_x: 0.5 - camera.shift[0] as f32 * base_aspect_x,
                shift_y: 0.5 - camera.shift[1] as f32 * base_aspect_y,
                camera_size: camera.ortho_scale as f32,
            },
        });
    }

    let world_to_projector = crate::geom::blender_matrix::blender_inverse_m4(world);
    if matrix_is_zero(&world_to_projector) {
        return Err(invalid(
            modifier,
            "projectors",
            "invertible projector transforms",
        ));
    }
    let win_matrix = camera_projection_matrix_f32(camera, aspect_x, aspect_y, scale_x, scale_y);
    let temporary =
        crate::geom::blender_matrix::blender_multiply_m4(win_matrix, world_to_projector);
    let projection =
        crate::geom::blender_matrix::blender_multiply_m4(uv_projection_offset_matrix(), temporary);
    Ok(ProjectorF32 {
        normal,
        projection: ProjectorProjection::Matrix(projection),
    })
}

fn camera_projection_matrix_f32(
    camera: &crate::model::CameraData,
    aspect_x: f32,
    aspect_y: f32,
    scale_x: f32,
    scale_y: f32,
) -> [f32; 16] {
    let ycor = aspect_y / aspect_x;
    let sensor_fit = match camera.sensor_fit.as_str() {
        "HORIZONTAL" => true,
        "VERTICAL" => false,
        _ => aspect_x >= aspect_y,
    };
    let sensor_size = if camera.sensor_fit == "VERTICAL" {
        camera.sensor_height_mm as f32
    } else {
        camera.sensor_width_mm as f32
    };
    let lens = (camera.lens_mm as f32).max(1.0e-9);
    let ortho_scale = (camera.ortho_scale as f32).max(1.0e-9);
    let clip_start = camera.clip_start as f32;
    let clip_end = (camera.clip_end as f32).max(clip_start + 1.0e-3);
    let is_ortho = camera.projection == crate::model::CameraProjection::Orthographic;
    let mut pixel_size = if is_ortho {
        ortho_scale
    } else {
        (sensor_size * clip_start) / lens
    };
    let view_factor = if sensor_fit { 1.0 } else { ycor };
    pixel_size /= view_factor;
    pixel_size *= 1.0;

    let shift_x = camera.shift[0] as f32;
    let shift_y = camera.shift[1] as f32;
    let dx = shift_x * view_factor;
    let dy = shift_y * view_factor;
    let mut left = (-0.5_f32 + dx) * pixel_size;
    let mut right = (0.5_f32 + dx) * pixel_size;
    let mut bottom = (-0.5_f32 * ycor + dy) * pixel_size;
    let mut top = (0.5_f32 * ycor + dy) * pixel_size;
    left *= scale_x;
    right *= scale_x;
    bottom *= scale_y;
    top *= scale_y;

    let x_delta = right - left;
    let y_delta = top - bottom;
    let z_delta = clip_end - clip_start;
    if is_ortho {
        let mut matrix = [0.0_f32; 16];
        matrix[0] = 1.0;
        matrix[5] = 1.0;
        matrix[10] = 1.0;
        matrix[15] = 1.0;
        matrix[0] = 2.0 / x_delta;
        matrix[12] = -(right + left) / x_delta;
        matrix[5] = 2.0 / y_delta;
        matrix[13] = -(top + bottom) / y_delta;
        matrix[10] = -2.0 / z_delta;
        matrix[14] = -(clip_end + clip_start) / z_delta;
        matrix
    } else {
        let mut matrix = [0.0_f32; 16];
        matrix[0] = clip_start * 2.0 / x_delta;
        matrix[5] = clip_start * 2.0 / y_delta;
        matrix[8] = (right + left) / x_delta;
        matrix[9] = (top + bottom) / y_delta;
        matrix[10] = -(clip_end + clip_start) / z_delta;
        matrix[11] = -1.0;
        matrix[14] = (-2.0 * clip_start * clip_end) / z_delta;
        matrix
    }
}

fn uv_projection_offset_matrix() -> [f32; 16] {
    [
        0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0,
    ]
}

fn normalize_camera_world_f32(mut matrix: [f32; 16]) -> [f32; 16] {
    for column in 0..3 {
        let index = column * 4;
        let (axis, length) = normalize_vector_blender_f32(Vec3::new(
            matrix[index],
            matrix[index + 1],
            matrix[index + 2],
        ));
        matrix[index] = axis.x;
        matrix[index + 1] = axis.y;
        matrix[index + 2] = axis.z;
        if length != 0.0 {
            matrix[index + 3] /= length;
        }
    }
    matrix
}

fn matrix_is_zero(matrix: &[f32; 16]) -> bool {
    matrix.iter().all(|value| *value == 0.0)
}

fn project_world_point_f32(point: Vec3, projection: &ProjectorProjection) -> Vec2 {
    match projection {
        ProjectorProjection::Matrix(matrix) => {
            let w = (point.x * matrix[3] + point.y * matrix[7] + matrix[11] * point.z + matrix[15])
                .abs();
            if w == 0.0 {
                return Vec2::ZERO;
            }
            Vec2::new(
                (point.x * matrix[0] + point.y * matrix[4] + matrix[8] * point.z + matrix[12]) / w,
                (point.x * matrix[1] + point.y * matrix[5] + matrix[9] * point.z + matrix[13]) / w,
            )
        }
        ProjectorProjection::Panorama {
            world_to_camera,
            aspect_x,
            aspect_y,
            shift_x,
            shift_y,
            camera_size,
        } => {
            let camera_point =
                crate::geom::blender_matrix::blender_transform_point_f32(world_to_camera, point);
            let angle = camera_point.x.atan2(-camera_point.z) / (std::f32::consts::PI * 2.0);
            Vec2::new(
                angle * *aspect_x + *shift_x,
                camera_point.y / *camera_size * *aspect_y + *shift_y,
            )
        }
    }
}

/// Build Blender's complete UV Warp matrix from original object or pose-bone world matrices.
pub(crate) fn build_uv_warp_matrix_f32(
    source_world: Option<[f32; 16]>,
    destination_world: Option<[f32; 16]>,
    modifier: &Modifier,
) -> Result<[f32; 16]> {
    let axis_u = warp_axis_index(string(modifier, "axis_u")?, modifier, "axis_u")?;
    let axis_v = warp_axis_index(string(modifier, "axis_v")?, modifier, "axis_v")?;
    let center = vector2(modifier, "center")?;
    let center = [center.x as f32, center.y as f32];
    let offset = vector2(modifier, "offset")?;
    let offset = [offset.x as f32, offset.y as f32];
    let scale = vector2(modifier, "scale")?;
    let scale = [scale.x as f32, scale.y as f32, 1.0_f32];
    let rotation = number(modifier, "rotation")? as f32;

    let mut warp =
        if let (Some(source_world), Some(destination_world)) = (source_world, destination_world) {
            let inverse_destination =
                crate::geom::blender_matrix::blender_inverse_m4(destination_world);
            let mut object_warp =
                crate::geom::blender_matrix::blender_multiply_m4(inverse_destination, source_world);

            if center[0] != 0.0 || center[1] != 0.0 {
                let mut center_matrix = identity_matrix_f32();
                center_matrix[12 + axis_u] = center[0];
                center_matrix[12 + axis_v] = center[1];
                let inverse_center = crate::geom::blender_matrix::blender_inverse_m4(center_matrix);
                object_warp =
                    crate::geom::blender_matrix::blender_multiply_m4(object_warp, inverse_center);
                object_warp =
                    crate::geom::blender_matrix::blender_multiply_m4(center_matrix, object_warp);
            }

            let mut shuffle = [0.0_f32; 16];
            shuffle[axis_u * 4] = 1.0;
            shuffle[axis_v * 4 + 1] = 1.0;
            shuffle[15] = 1.0;
            let mut shuffle_transposed = [0.0_f32; 16];
            for column in 0..4 {
                for row in 0..4 {
                    shuffle_transposed[column * 4 + row] = shuffle[row * 4 + column];
                }
            }
            object_warp = crate::geom::blender_matrix::blender_multiply_m4(shuffle, object_warp);
            crate::geom::blender_matrix::blender_multiply_m4(object_warp, shuffle_transposed)
        } else {
            identity_matrix_f32()
        };
    blender_translate_m4(&mut warp, [center[0], center[1], 0.0]);
    blender_rescale_m4(&mut warp, scale);
    blender_rotate_z_m4(&mut warp, rotation);
    blender_translate_m4(&mut warp, [offset[0], offset[1], 0.0]);
    blender_translate_m4(&mut warp, [-center[0], -center[1], 0.0]);
    Ok(warp)
}

fn warp_axis_index(axis: &str, modifier: &Modifier, parameter: &str) -> Result<usize> {
    match axis {
        "X" => Ok(0),
        "Y" => Ok(1),
        "Z" => Ok(2),
        _ => Err(invalid(modifier, parameter, "X, Y, or Z")),
    }
}

fn identity_matrix_f32() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn blender_translate_m4(matrix: &mut [f32; 16], translation: [f32; 3]) {
    matrix[12] +=
        translation[0] * matrix[0] + translation[1] * matrix[4] + translation[2] * matrix[8];
    matrix[13] +=
        translation[0] * matrix[1] + translation[1] * matrix[5] + translation[2] * matrix[9];
    matrix[14] +=
        translation[0] * matrix[2] + translation[1] * matrix[6] + translation[2] * matrix[10];
}

fn blender_rescale_m4(matrix: &mut [f32; 16], scale: [f32; 3]) {
    for column in 0..3 {
        for row in 0..3 {
            matrix[column * 4 + row] *= scale[column];
        }
    }
}

fn blender_rotate_z_m4(matrix: &mut [f32; 16], angle: f32) {
    let cosine = angle.cos();
    let sine = angle.sin();
    for row in 0..4 {
        let x = matrix[row];
        let y = matrix[4 + row];
        matrix[row] = cosine * x + sine * y;
        matrix[4 + row] = -sine * x + cosine * y;
    }
}

pub(super) fn uv_warp(
    mesh: &Mesh,
    modifier: &Modifier,
    _operands: &BTreeMap<String, Vec<SceneOperand>>,
) -> Result<Mesh> {
    let mut store = crate::geom::uv_layers::read(mesh);
    if store.layers.is_empty() {
        return Ok(mesh.clone());
    }
    let requested_layer = string(modifier, "uv_layer")?;
    let layer_index = if requested_layer.is_empty() {
        active_uv_layer_index(&store)
    } else {
        store
            .index_of(requested_layer)
            .or_else(|| active_uv_layer_index(&store))
    };
    let Some(layer_index) = layer_index else {
        return Ok(mesh.clone());
    };
    let matrix = if let Some(columns) = modifier.runtime.uv_warp_matrix {
        flatten_runtime_matrix(columns)
    } else if ["object_from", "object_to"].iter().all(|parameter| {
        modifier
            .params
            .get(*parameter)
            .is_none_or(|value| value.is_null() || value.as_str().is_some_and(str::is_empty))
    }) {
        build_uv_warp_matrix_f32(None, None, modifier)?
    } else {
        return Err(PotError::with_details(
            ErrorCode::EvaluationFailed,
            "UV Warp matrix is missing from the evaluated scene context",
            json!({"modifier_id":modifier.id}),
        ));
    };
    let group = optional_string(modifier, "vertex_group")?;
    let has_deform_weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .is_some_and(|groups| {
            groups.values().any(|weights| {
                weights
                    .as_object()
                    .is_some_and(|weights| !weights.is_empty())
            })
        });
    let weights = group
        .filter(|name| {
            has_deform_weights
                && mesh
                    .attributes
                    .get("vertex_groups")
                    .and_then(Value::as_object)
                    .is_some_and(|groups| groups.contains_key(*name))
        })
        .map(|name| get_group_weights(mesh, name))
        .transpose()?;
    let invert_group = boolean(modifier, "invert_vertex_group")?;
    let source_faces = &store.layers[layer_index].faces;
    let mut output_faces = Vec::with_capacity(mesh.faces.len());
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let mut uv_down = Vec::with_capacity(face.vertices.len());
        let base_values = source_faces.get(face_index).and_then(Option::as_ref);
        for (corner, vertex_id) in face.vertices.iter().enumerate() {
            let base_down = base_values
                .and_then(|values| values.get(corner))
                .map(|value| DVec2::from_array(*value))
                .unwrap_or(DVec2::ZERO);
            // Canonical mesh UVs are V-down; Blender UV Warp works in V-up coordinates.
            let base = Vec3::new(base_down.x as f32, (1.0 - base_down.y) as f32, 0.0);
            let mut weight = weights.as_ref().map_or(1.0_f32, |values| {
                values.get(vertex_id).copied().unwrap_or(0.0) as f32
            });
            if invert_group && weights.is_some() {
                weight = 1.0 - weight;
            }
            let transformed =
                crate::geom::blender_matrix::blender_transform_point_f32(&matrix, base);
            let output = Vec2::new(
                base.x + (transformed.x - base.x) * weight,
                base.y + (transformed.y - base.y) * weight,
            );
            uv_down.push([f64::from(output.x), f64::from(1.0_f32 - output.y)]);
        }
        output_faces.push(Some(uv_down));
    }
    store.layers[layer_index].faces = output_faces;
    let mut result = mesh.clone();
    crate::geom::uv_layers::write_layers(&mut result, &store.layers);
    Ok(result)
}

fn flatten_runtime_matrix(columns: [[f32; 4]; 4]) -> [f32; 16] {
    std::array::from_fn(|index| columns[index / 4][index % 4])
}
