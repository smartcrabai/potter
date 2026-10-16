//! UV project and UV warp modifiers, including projector cameras and bone references.

use super::*;

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
    let aspect_x = number(modifier, "aspect_x")?.max(1.0);
    let aspect_y = number(modifier, "aspect_y")?.max(1.0);
    // Blender clamps these to zero, then treats zero as the default scale of one.
    let scale_x = number(modifier, "scale_x")?.max(0.0);
    let scale_y = number(modifier, "scale_y")?.max(0.0);
    let scale_x = if scale_x == 0.0 { 1.0 } else { scale_x };
    let scale_y = if scale_y == 0.0 { 1.0 } else { scale_y };

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
    let vertex_indices = super::super::id_indices(mesh);
    let mut projector_normals = Vec::with_capacity(projectors.len());
    let mut world_to_projectors = Vec::with_capacity(projectors.len());
    for projector in projectors {
        let projector_world = subject_world * projector.local_to_subject;
        projector_normals.push(projector_world.transform_vector3(DVec3::Z));
        let camera_world = if projector.camera.as_ref().is_some_and(|camera| {
            matches!(
                camera.projection,
                crate::model::CameraProjection::Panorama | crate::model::CameraProjection::Fisheye
            )
        }) {
            normalize_camera_transform(projector_world)
        } else {
            projector_world
        };
        let world_to_projector = camera_world.inverse();
        if !world_to_projector.is_finite() {
            return Err(invalid(
                modifier,
                "projectors",
                "invertible projector transforms",
            ));
        }
        world_to_projectors.push(world_to_projector);
    }
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
        let normal = polygon_normal_from_indices(mesh, &face_vertex_indices);
        // Blender keeps the first projector when two camera directions tie.
        let mut best_index = 0;
        let mut best_dot = projector_normals[0].dot(normal);
        for (index, projector_normal) in projector_normals.iter().enumerate().skip(1) {
            let dot = projector_normal.dot(normal);
            if dot > best_dot {
                best_index = index;
                best_dot = dot;
            }
        }
        let projector = &projectors[best_index];
        let world_to_projector = world_to_projectors[best_index];
        let mut coords = Vec::with_capacity(face.vertices.len());
        for vertex_index in face_vertex_indices {
            let uv_up = project_uv_point(
                mesh.vertices[vertex_index].co,
                projector,
                world_to_projector,
                subject_world,
                aspect_x,
                aspect_y,
                scale_x,
                scale_y,
            )?
            .unwrap_or(DVec2::ZERO);
            // Canonical UV storage uses V-down; Blender's projection matrix produces V-up.
            coords.push([uv_up.x, 1.0 - uv_up.y]);
        }
        store.layers[layer_index].faces[face_index] = Some(coords);
    }
    let mut result = mesh.clone();
    crate::geom::uv_layers::write_layers(&mut result, &store.layers);
    Ok(result)
}
fn polygon_normal_from_indices(mesh: &Mesh, indices: &[usize]) -> DVec3 {
    if indices.len() < 3 {
        return DVec3::Z;
    }
    let mut normal = DVec3::ZERO;
    for (index, &current_index) in indices.iter().enumerate() {
        let current = mesh.vertices[current_index].co;
        let next = mesh.vertices[indices[(index + 1) % indices.len()]].co;
        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }
    normal.normalize_or_zero()
}

fn normalize_camera_transform(transform: DMat4) -> DMat4 {
    DMat4::from_cols(
        transform.x_axis.truncate().normalize_or_zero().extend(transform.x_axis.w),
        transform.y_axis.truncate().normalize_or_zero().extend(transform.y_axis.w),
        transform.z_axis.truncate().normalize_or_zero().extend(transform.z_axis.w),
        transform.w_axis,
    )
}
#[expect(
    clippy::manual_midpoint,
    reason = "matches Blender's float evaluation order for parity"
)]
fn project_uv_point(
    point: DVec3,
    projector: &SceneOperand,
    world_to_projector: DMat4,
    subject_world: DMat4,
    aspect_x: f64,
    aspect_y: f64,
    scale_x: f64,
    scale_y: f64,
) -> Result<Option<DVec2>> {
    let world = subject_world.transform_point3(point);
    let local = world_to_projector.transform_point3(world);
    let Some(camera) = &projector.camera else {
        return Ok(Some(DVec2::new(
            (local.x + 1.0) * 0.5,
            (local.y + 1.0) * 0.5,
        )));
    };
    if matches!(
        camera.projection,
        crate::model::CameraProjection::Panorama | crate::model::CameraProjection::Fisheye
    ) {
        let (aspect_u, aspect_v) = if aspect_x > aspect_y {
            (1.0, aspect_x / aspect_y)
        } else {
            (aspect_y / aspect_x, 1.0)
        };
        return Ok(Some(DVec2::new(
            local.x.atan2(-local.z) / (2.0 * std::f64::consts::PI) * aspect_u * scale_x + 0.5
                - camera.shift[0] * aspect_u,
            local.y / camera.ortho_scale * aspect_v * scale_y + 0.5
                - camera.shift[1] * aspect_v,
        )));
    }

    let ycor = aspect_y / aspect_x;
    let sensor_fit = match camera.sensor_fit.as_str() {
        "HORIZONTAL" => "HORIZONTAL",
        "VERTICAL" => "VERTICAL",
        _ if ycor <= 1.0 => "HORIZONTAL",
        _ => "VERTICAL",
    };
    let view_factor = if sensor_fit == "HORIZONTAL" {
        1.0
    } else {
        ycor
    };
    let shift_x = camera.shift[0];
    let shift_y = camera.shift[1];
    let ndc = match camera.projection {
        crate::model::CameraProjection::Orthographic => DVec2::new(
            2.0 * local.x * view_factor / (camera.ortho_scale * scale_x)
                - 2.0 * shift_x * view_factor,
            2.0 * local.y * view_factor / (ycor * camera.ortho_scale * scale_y)
                - 2.0 * shift_y * view_factor / ycor,
        ),
        crate::model::CameraProjection::Perspective => {
            let depth = -local.z;
            if depth.abs() <= f64::EPSILON {
                return Ok(None);
            }
            // CAMERA_SENSOR_FIT_AUTO selects sensor X in camera_params_from_object.
            let sensor_size = if camera.sensor_fit == "VERTICAL" {
                camera.sensor_height_mm
            } else {
                camera.sensor_width_mm
            };
            DVec2::new(
                2.0 * local.x * view_factor * camera.lens_mm / (depth * sensor_size * scale_x)
                    - 2.0 * shift_x * view_factor,
                2.0 * local.y * view_factor * camera.lens_mm
                    / (depth * ycor * sensor_size * scale_y)
                    - 2.0 * shift_y * view_factor / ycor,
            )
        }
        crate::model::CameraProjection::Panorama | crate::model::CameraProjection::Fisheye => {
            unreachable!("panoramic cameras were handled above")
        }
    };
    Ok(Some((ndc + DVec2::ONE) * 0.5))
}

pub(super) fn uv_warp(
    mesh: &Mesh,
    modifier: &Modifier,
    operands: &BTreeMap<String, Vec<SceneOperand>>,
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
    let center = vector2(modifier, "center")?;
    let axis_u = selected_axis(string(modifier, "axis_u")?, modifier, "axis_u")?;
    let axis_v = selected_axis(string(modifier, "axis_v")?, modifier, "axis_v")?;
    let offset = vector2(modifier, "offset")?;
    let scale = vector2(modifier, "scale")?;
    let rotation = number(modifier, "rotation")?;
    let group = optional_string(modifier, "vertex_group")?;
    let has_deform_weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .is_some_and(|groups| {
            groups
                .values()
                .any(|weights| weights.as_object().is_some_and(|weights| !weights.is_empty()))
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
    let from = operands.get("object_from").and_then(|items| items.first());
    let to = operands.get("object_to").and_then(|items| items.first());
    let from_bone = optional_string(modifier, "bone_from")?.is_some();
    let to_bone = optional_string(modifier, "bone_to")?.is_some();
    let object_transform = match (from, to) {
        (Some(from), Some(to)) => {
            let from_matrix = if from_bone {
                warp_bone_matrix(
                    modifier,
                    "bone_from",
                    modifier.runtime.warp_from_matrix,
                )?
            } else {
                from.local_to_subject
            };
            let to_matrix = if to_bone {
                warp_bone_matrix(
                    modifier,
                    "bone_to",
                    modifier.runtime.warp_to_matrix,
                )?
            } else {
                to.local_to_subject
            };
            let inverse = to_matrix.inverse();
            if !inverse.is_finite() {
                return Err(invalid(
                    modifier,
                    "object_to",
                    "an invertible object transform",
                ));
            }
            Some(inverse * from_matrix)
        }
        _ => None,
    };
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
            let base = DVec2::new(base_down.x, 1.0 - base_down.y);
            let mut mask = weights
                .as_ref()
                .map_or(1.0, |values| values.get(vertex_id).copied().unwrap_or(0.0))
                .clamp(0.0, 1.0);
            if invert_group && weights.is_some() {
                mask = 1.0 - mask;
            }
            let relative = base - center + offset;
            let rotated = DMat3::from_rotation_z(rotation)
                .mul_vec3(DVec3::new(relative.x, relative.y, 0.0))
                .truncate();
            let direct = center + DVec2::new(rotated.x * scale.x, rotated.y * scale.y);
            let transformed = object_transform.map_or(direct, |transform| {
                let center_3d = axis_u * center.x + axis_v * center.y;
                let direct_3d = axis_u * direct.x + axis_v * direct.y;
                let mapped = center_3d + transform.transform_point3(direct_3d - center_3d);
                DVec2::new(mapped.dot(axis_u), mapped.dot(axis_v))
            });
            let output = base.lerp(transformed, mask);
            uv_down.push([output.x, 1.0 - output.y]);
        }
        output_faces.push(Some(uv_down));
    }
    store.layers[layer_index].faces = output_faces;
    let mut result = mesh.clone();
    crate::geom::uv_layers::write_layers(&mut result, &store.layers);
    Ok(result)
}
fn warp_bone_matrix(modifier: &Modifier, parameter: &str, matrix: Option<[f64; 16]>) -> Result<DMat4> {
    let matrix = matrix.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            format!("UV Warp `{parameter}` transform is missing from the evaluated scene context"),
            json!({"modifier_id":modifier.id,"parameter":parameter}),
        )
    })?;
    Ok(DMat4::from_cols_array(&matrix))
}

