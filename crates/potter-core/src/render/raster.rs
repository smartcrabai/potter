use glam::DVec3;

use crate::error::{ErrorCode, PotError, Result};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Projection {
    Orthographic { height: f64 },
    Perspective { lens_mm: f64, sensor_width_mm: f64 },
    Panorama { fisheye: bool },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DepthOfField {
    pub(crate) focus_distance: f64,
    pub(crate) aperture_radius: f64,
    pub(crate) aperture_blades: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum StereoMode {
    #[default]
    None,
    SideBySide,
    Anaglyph,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Camera {
    pub(crate) position: DVec3,
    pub(crate) target: DVec3,
    pub(crate) up: DVec3,
    pub(crate) near: f64,
    pub(crate) far: f64,
    pub(crate) projection: Projection,
    pub(crate) shift: [f64; 2],
    pub(crate) depth_of_field: Option<DepthOfField>,
    pub(crate) stereo_mode: StereoMode,
    pub(crate) interocular_distance: f64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Mode {
    Solid,
    Beauty,
    Wire,
    Normal,
    Depth,
    Id,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AlphaMode {
    Opaque,
    Blend,
    Clip,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Triangle {
    pub(crate) positions: [DVec3; 3],
    pub(crate) object_index: u32,
    pub(crate) element_index: u32,
    pub(crate) color: [f64; 4],
    pub(crate) alpha_mode: AlphaMode,
    pub(crate) alpha_threshold: f64,
    pub(crate) double_sided: bool,
    pub(crate) material_index: usize,
    pub(crate) uv: [[f64; 2]; 3],
    pub(crate) uv_available: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Line {
    pub(crate) start: DVec3,
    pub(crate) end: DVec3,
    pub(crate) radius_start: f64,
    pub(crate) radius_end: f64,
    pub(crate) color: [f64; 4],
    pub(crate) object_index: u32,
    pub(crate) element_index: u32,
}

#[derive(Debug)]
pub(crate) struct RasterOutput {
    pub(crate) rgba: Vec<u8>,
    pub(crate) linear_rgba: Vec<[f32; 4]>,
    pub(crate) ids: Vec<u32>,
    pub(crate) depths: Vec<f32>,
    pub(crate) elements: Vec<u32>,
    pub(crate) normals: Vec<[f32; 3]>,
    pub(crate) albedo: Vec<[f32; 4]>,
    pub(crate) emission: Vec<[f32; 4]>,
    pub(crate) ambient_occlusion: Vec<f32>,
}

#[derive(Clone, Copy)]
struct CameraFrame {
    forward: DVec3,
    right: DVec3,
    up: DVec3,
    near: f64,
    far: f64,
    shift: [f64; 2],
    depth_of_field: Option<DepthOfField>,
    stereo_mode: StereoMode,
    interocular_distance: f64,
    projection: ProjectionFrame,
}

#[derive(Clone, Copy)]
enum ProjectionFrame {
    Orthographic {
        half_width: f64,
        half_height: f64,
    },
    Perspective {
        horizontal_scale: f64,
        vertical_scale: f64,
    },
    Panorama {
        fisheye: bool,
    },
}

struct PreparedTriangle {
    edge1: DVec3,
    edge2: DVec3,
    normal: DVec3,
    edge_altitudes: [f64; 3],
    source: Triangle,
}

#[derive(Clone, Copy)]
struct Ray {
    origin: DVec3,
    direction: DVec3,
}

#[derive(Clone, Copy)]
struct Hit {
    depth: f64,
    barycentric: [f64; 3],
    direction: DVec3,
}

pub(crate) fn ray_for_pixel(
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Result<(DVec3, DVec3)> {
    if !pixel_in_bounds(width, height, x, y) {
        return Err(PotError::invalid_argument(
            "pixel coordinates must lie inside a non-empty image",
        ));
    }

    let frame = camera_frame(camera, width, height)?;
    let ray = ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?;
    Ok((ray.origin, ray.direction))
}

pub(crate) fn ray_for_sample(
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<(DVec3, DVec3)> {
    let frame = camera_frame(camera, width, height)?;
    let ray = ray_for_subpixel_in_frame(
        &frame,
        camera.position,
        width,
        height,
        x,
        y,
        sample_x,
        sample_y,
    )?;
    Ok((ray.origin, ray.direction))
}

pub(crate) fn camera_eye(camera: &Camera, eye: f64) -> Camera {
    let forward = (camera.target - camera.position).normalize_or_zero();
    let right = forward.cross(camera.up).normalize_or_zero();
    let offset = right * (camera.interocular_distance * eye * 0.5);
    let mut result = *camera;
    result.position += offset;
    result.target += offset;
    result.stereo_mode = StereoMode::None;
    result
}

#[expect(
    clippy::too_many_arguments,
    reason = "raster configuration is explicit at this low-level boundary"
)]
pub(crate) fn rasterize_with_samples(
    triangles: &[Triangle],
    camera: &Camera,
    width: u32,
    height: u32,
    mode: Mode,
    background: [u8; 4],
    samples: u32,
    seed: u32,
) -> Result<RasterOutput> {
    if samples == 0 {
        return Err(PotError::invalid_argument(
            "sample count must be greater than zero",
        ));
    }
    if camera.stereo_mode == StereoMode::Anaglyph {
        let left_camera = camera_eye(camera, -1.0);
        let right_camera = camera_eye(camera, 1.0);
        let mut left_output = rasterize_with_samples(
            triangles,
            &left_camera,
            width,
            height,
            mode,
            background,
            samples,
            seed,
        )?;
        let right_output = rasterize_with_samples(
            triangles,
            &right_camera,
            width,
            height,
            mode,
            background,
            samples,
            seed,
        )?;
        let (left_pixels, _) = left_output.rgba.as_chunks_mut::<4>();
        let (right_pixels, _) = right_output.rgba.as_chunks::<4>();
        for (left, right) in left_pixels.iter_mut().zip(right_pixels) {
            left[1] = right[1];
            left[2] = right[2];
            left[3] = u8::try_from(u16::midpoint(u16::from(left[3]), u16::from(right[3])))
                .unwrap_or(u8::MAX);
        }
        for (left, right) in left_output
            .linear_rgba
            .iter_mut()
            .zip(right_output.linear_rgba.iter())
        {
            left[1] = right[1];
            left[2] = right[2];
            left[3] = f32::midpoint(left[3], right[3]);
        }
        for (left, right) in left_output
            .ambient_occlusion
            .iter_mut()
            .zip(right_output.ambient_occlusion.iter())
        {
            *left = f32::midpoint(*left, *right);
        }
        return Ok(left_output);
    }
    let pixel_count = checked_pixel_count(width, height)?;
    let rgba_len = pixel_count.checked_mul(4).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image dimensions exceed the supported raster size",
        )
    })?;
    let frame = camera_frame(camera, width, height)?;
    let prepared = prepare_triangles(triangles)?;
    let ao_bvh = if matches!(mode, Mode::Solid | Mode::Beauty | Mode::Wire) && !triangles.is_empty()
    {
        Some(super::bvh::Bvh::build(triangles)?)
    } else {
        None
    };

    let background_linear = background_to_linear_f64(background);
    let mut rgba = allocate_filled(rgba_len, 0_u8, "RGBA image")?;
    let (pixels, _) = rgba.as_chunks_mut::<4>();
    for pixel in pixels {
        pixel.copy_from_slice(&background);
    }
    let mut linear_rgba = allocate_filled(
        pixel_count,
        linear_color_to_f32(background_linear),
        "linear RGBA image",
    )?;
    let mut ids = allocate_filled(pixel_count, 0_u32, "object ID buffer")?;
    let mut depths = allocate_filled(pixel_count, 0_f32, "depth buffer")?;
    let mut elements = allocate_filled(pixel_count, 0_u32, "element ID buffer")?;
    let mut normals = allocate_filled(pixel_count, [0.0_f32; 3], "normal pass")?;
    let mut albedo = allocate_filled(pixel_count, [0.0_f32; 4], "albedo pass")?;
    let emission = allocate_filled(pixel_count, [0.0_f32; 4], "emission pass")?;
    let mut ambient_occlusion = allocate_filled(pixel_count, 1.0_f32, "ambient occlusion pass")?;
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;

    for y in 0..height {
        for x in 0..width {
            let center_ray = ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?;
            let center_hit = nearest_visible_hit(&prepared, center_ray, &frame);
            let pixel_index = usize::try_from(y)
                .ok()
                .and_then(|row| row.checked_mul(width_usize))
                .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "pixel index exceeds platform limits",
                    )
                })?;
            if let Some((triangle_index, hit)) = center_hit {
                let triangle = &prepared[triangle_index];
                ids[pixel_index] = triangle.source.object_index;
                depths[pixel_index] = depth_to_f32(hit.depth);
                elements[pixel_index] = triangle.source.element_index;
                normals[pixel_index] = triangle.normal.to_array().map(|value| value as f32);
                albedo[pixel_index] = linear_color_to_f32(triangle.source.color);
                if let Some(bvh) = &ao_bvh {
                    let mut normal = triangle.normal;
                    if normal.dot(hit.direction) > 0.0 {
                        normal = -normal;
                    }
                    let point = center_ray.origin + center_ray.direction * hit.depth;
                    let visibility = ambient_visibility(bvh, triangles, point, normal, 1.0);
                    ambient_occlusion[pixel_index] =
                        linear_color_to_f32([visibility, visibility, visibility, 1.0])[0];
                }
            }

            if matches!(mode, Mode::Solid | Mode::Beauty | Mode::Wire) {
                let mut average = [0.0_f64; 4];
                for sample_index in 0..samples {
                    let (sample_x, sample_y) = sample_offset(seed, x, y, sample_index);
                    let (sample_ray, sample_hit) = if sample_index == 0 {
                        (center_ray, center_hit)
                    } else {
                        let ray = ray_for_subpixel_in_frame(
                            &frame,
                            camera.position,
                            width,
                            height,
                            x,
                            y,
                            sample_x,
                            sample_y,
                        )?;
                        (ray, nearest_hit(&prepared, ray, &frame))
                    };
                    let sample_color = if matches!(mode, Mode::Wire) {
                        sample_hit.map_or(background_linear, |(triangle_index, hit)| {
                            let triangle = &prepared[triangle_index];
                            if is_wire_edge(triangle, hit, &frame, width, height) {
                                triangle.source.color
                            } else {
                                background_linear
                            }
                        })
                    } else if let Some((triangle_index, _)) = sample_hit {
                        let triangle = &prepared[triangle_index].source;
                        if triangle_opacity(triangle) >= 1.0 {
                            [triangle.color[0], triangle.color[1], triangle.color[2], 1.0]
                        } else {
                            composite_sample(&prepared, sample_ray, &frame, background_linear)
                        }
                    } else {
                        background_linear
                    };
                    let count = f64::from(sample_index + 1);
                    let previous_weight = 1.0 - 1.0 / count;
                    for channel in 0..4 {
                        average[channel] =
                            average[channel] * previous_weight + sample_color[channel] / count;
                    }
                }
                if let Some(ao) = ambient_occlusion.get(pixel_index).copied() {
                    let ao = f64::from(ao);
                    for channel in &mut average[..3] {
                        *channel *= ao;
                    }
                }
                let color = color_to_srgba(average);
                rgba[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&color);
                linear_rgba[pixel_index] = linear_color_to_f32(average);
            } else if let Some((triangle_index, hit)) = center_hit {
                let triangle = &prepared[triangle_index];
                let (color, linear_color) = match mode {
                    Mode::Normal => {
                        let color = normal_to_rgba(triangle.normal);
                        (color, display_color_to_float(color))
                    }
                    Mode::Depth => {
                        let color = depth_to_rgba(hit.depth, frame.near, frame.far);
                        (color, display_color_to_float(color))
                    }
                    Mode::Id => {
                        let color = id_to_rgba(triangle.source.object_index);
                        (color, display_color_to_float(color))
                    }
                    Mode::Solid | Mode::Beauty | Mode::Wire => continue,
                };
                let rgba_index = pixel_index * 4;
                rgba[rgba_index..rgba_index + 4].copy_from_slice(&color);
                linear_rgba[pixel_index] = linear_color;
            }
        }
    }

    Ok(RasterOutput {
        rgba,
        linear_rgba,
        ids,
        depths,
        elements,
        normals,
        albedo,
        emission,
        ambient_occlusion,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "line rasterization uses the same explicit configuration as triangle rasterization"
)]
pub(crate) fn rasterize_with_lines(
    triangles: &[Triangle],
    lines: &[Line],
    camera: &Camera,
    width: u32,
    height: u32,
    mode: Mode,
    background: [u8; 4],
    samples: u32,
    seed: u32,
) -> Result<RasterOutput> {
    let mut output = rasterize_with_samples(
        triangles, camera, width, height, mode, background, samples, seed,
    )?;
    if lines.is_empty() {
        return Ok(output);
    }
    let frame = camera_frame(camera, width, height)?;
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    for line in lines {
        if !is_finite_vec3(line.start)
            || !is_finite_vec3(line.end)
            || !line.radius_start.is_finite()
            || !line.radius_end.is_finite()
            || line.radius_start < 0.0
            || line.radius_end < 0.0
            || !line.color.iter().all(|channel| channel.is_finite())
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "evaluated stroke contains invalid geometry or color",
            ));
        }
        let Some(start) = project_world(camera, &frame, width, height, line.start) else {
            continue;
        };
        let Some(end) = project_world(camera, &frame, width, height, line.end) else {
            continue;
        };
        let radius_start = line.radius_start * start.pixel_scale;
        let radius_end = line.radius_end * end.pixel_scale;
        let max_radius = radius_start.max(radius_end).max(0.5);
        let min_x = screen_bound(start.x.min(end.x) - max_radius, width);
        let max_x = screen_bound(start.x.max(end.x) + max_radius, width);
        let min_y = screen_bound(start.y.min(end.y) - max_radius, height);
        let max_y = screen_bound(start.y.max(end.y) + max_radius, height);
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length_squared = dx * dx + dy * dy;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pixel_x = f64::from(x) + 0.5;
                let pixel_y = f64::from(y) + 0.5;
                let fraction = if length_squared <= f64::EPSILON {
                    0.0
                } else {
                    (((pixel_x - start.x) * dx + (pixel_y - start.y) * dy) / length_squared)
                        .clamp(0.0, 1.0)
                };
                let nearest_x = start.x + dx * fraction;
                let nearest_y = start.y + dy * fraction;
                let radius = (radius_start + (radius_end - radius_start) * fraction).max(0.5);
                let distance_x = pixel_x - nearest_x;
                let distance_y = pixel_y - nearest_y;
                if distance_x * distance_x + distance_y * distance_y > radius * radius {
                    continue;
                }
                let world_position = line.start.lerp(line.end, fraction);
                let depth = (world_position - camera.position).dot(frame.forward);
                if !depth.is_finite() || depth < frame.near || depth > frame.far {
                    continue;
                }
                let pixel_index = usize::try_from(y)
                    .ok()
                    .and_then(|row| row.checked_mul(width_usize))
                    .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "stroke pixel index exceeds platform limits",
                        )
                    })?;
                let old_depth = output.depths[pixel_index];
                let old_object = output.ids[pixel_index];
                let old_element = output.elements[pixel_index];
                let tolerance = 1.0e-6_f64.max(depth.abs() * 1.0e-6);
                let visible = old_object == 0
                    || depth < f64::from(old_depth) - tolerance
                    || ((depth - f64::from(old_depth)).abs() <= tolerance
                        && (line.object_index < old_object
                            || (line.object_index == old_object
                                && line.element_index < old_element)));
                if !visible {
                    continue;
                }
                let color = line.color;
                let encoded = match mode {
                    Mode::Id => id_to_rgba(line.object_index),
                    Mode::Depth => depth_to_rgba(depth, frame.near, frame.far),
                    Mode::Normal => normal_to_rgba(frame.forward),
                    Mode::Solid | Mode::Beauty | Mode::Wire => color_to_srgba(color),
                };
                let rgba_index = pixel_index * 4;
                output.rgba[rgba_index..rgba_index + 4].copy_from_slice(&encoded);
                output.linear_rgba[pixel_index] =
                    if matches!(mode, Mode::Solid | Mode::Beauty | Mode::Wire) {
                        linear_color_to_f32(color)
                    } else {
                        display_color_to_float(encoded)
                    };
                output.ids[pixel_index] = line.object_index;
                output.depths[pixel_index] = depth_to_f32(depth);
                output.elements[pixel_index] = line.element_index;
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy)]
struct ProjectedPoint {
    x: f64,
    y: f64,
    pixel_scale: f64,
}

fn project_world(
    camera: &Camera,
    frame: &CameraFrame,
    width: u32,
    height: u32,
    point: DVec3,
) -> Option<ProjectedPoint> {
    let relative = point - camera.position;
    let depth = relative.dot(frame.forward);
    if !depth.is_finite() || depth < frame.near || depth > frame.far {
        return None;
    }
    let right = relative.dot(frame.right);
    let up = relative.dot(frame.up);
    let (mut ndc_x, mut ndc_y, scale_x, scale_y) = match frame.projection {
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => (
            right / half_width,
            up / half_height,
            f64::from(width) / (2.0 * half_width),
            f64::from(height) / (2.0 * half_height),
        ),
        ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } => (
            right / (depth * horizontal_scale),
            up / (depth * vertical_scale),
            f64::from(width) / (2.0 * depth * horizontal_scale),
            f64::from(height) / (2.0 * depth * vertical_scale),
        ),
        ProjectionFrame::Panorama { fisheye: false } => {
            let longitude = right.atan2(relative.dot(frame.forward));
            let latitude = (up / relative.length()).clamp(-1.0, 1.0).asin();
            (
                longitude / std::f64::consts::PI,
                latitude / std::f64::consts::FRAC_PI_2,
                f64::from(width) / (std::f64::consts::TAU * depth),
                f64::from(height) / (std::f64::consts::PI * depth),
            )
        }
        ProjectionFrame::Panorama { fisheye: true } => {
            let direction = relative.normalize_or_zero();
            let angle = direction.dot(frame.forward).clamp(-1.0, 1.0).acos();
            let radial = DVec3::new(right, up, 0.0).normalize_or_zero();
            (
                radial.x * angle / std::f64::consts::PI,
                radial.y * angle / std::f64::consts::PI,
                f64::from(width) / (2.0 * depth),
                f64::from(height) / (2.0 * depth),
            )
        }
    };
    ndc_x -= 2.0 * frame.shift[0];
    ndc_y -= 2.0 * frame.shift[1];
    let x = (ndc_x + 1.0) * f64::from(width) * 0.5 - 0.5;
    let y = (1.0 - ndc_y) * f64::from(height) * 0.5 - 0.5;
    let pixel_scale = scale_x.max(scale_y);
    (x.is_finite() && y.is_finite() && pixel_scale.is_finite()).then_some(ProjectedPoint {
        x,
        y,
        pixel_scale,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "screen bounds are clamped to a valid non-negative image coordinate"
)]
fn screen_bound(value: f64, extent: u32) -> u32 {
    value
        .floor()
        .clamp(0.0, f64::from(extent.saturating_sub(1))) as u32
}
fn checked_pixel_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "image width and height must both be non-zero",
        ));
    }
    let width = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let height = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image height exceeds platform limits",
        )
    })?;
    width.checked_mul(height).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image dimensions exceed the supported raster size",
        )
    })
}

fn pixel_in_bounds(width: u32, height: u32, x: u32, y: u32) -> bool {
    width != 0 && height != 0 && x < width && y < height
}

fn camera_frame(camera: &Camera, width: u32, height: u32) -> Result<CameraFrame> {
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "image width and height must both be non-zero",
        ));
    }
    let view_width = if camera.stereo_mode == StereoMode::SideBySide {
        width / 2
    } else {
        width
    };
    if view_width == 0
        || !is_finite_vec3(camera.position)
        || !is_finite_vec3(camera.target)
        || !is_finite_vec3(camera.up)
        || !camera.near.is_finite()
        || !camera.far.is_finite()
        || camera.near <= 0.0
        || camera.far <= camera.near
        || camera.shift.iter().any(|value| !value.is_finite())
        || !camera.interocular_distance.is_finite()
        || camera.interocular_distance < 0.0
        || camera.depth_of_field.is_some_and(|dof| {
            !dof.focus_distance.is_finite()
                || dof.focus_distance <= 0.0
                || !dof.aperture_radius.is_finite()
                || dof.aperture_radius < 0.0
        })
    {
        return Err(PotError::invalid_argument("camera parameters are invalid"));
    }

    let forward = safe_normalize(camera.target - camera.position)
        .ok_or_else(|| PotError::invalid_argument("camera position and target must differ"))?;
    let up_hint = safe_normalize(camera.up)
        .ok_or_else(|| PotError::invalid_argument("camera up vector must be non-zero"))?;
    let right = safe_normalize(forward.cross(up_hint)).ok_or_else(|| {
        PotError::invalid_argument("camera up vector must not be parallel to its view direction")
    })?;
    let up = safe_normalize(right.cross(forward))
        .ok_or_else(|| PotError::invalid_argument("camera orientation is invalid"))?;

    let projection = match camera.projection {
        Projection::Orthographic {
            height: view_height,
        } if view_height.is_finite() && view_height > 0.0 => {
            let half_height = view_height * 0.5;
            let half_width = half_height * (f64::from(view_width) / f64::from(height));
            if !half_height.is_finite() || !half_width.is_finite() || half_width <= 0.0 {
                return Err(PotError::invalid_argument(
                    "orthographic projection dimensions are out of range",
                ));
            }
            ProjectionFrame::Orthographic {
                half_width,
                half_height,
            }
        }
        Projection::Perspective {
            lens_mm,
            sensor_width_mm,
        } if lens_mm.is_finite()
            && sensor_width_mm.is_finite()
            && lens_mm > 0.0
            && sensor_width_mm > 0.0 =>
        {
            let horizontal_scale = sensor_width_mm / (2.0 * lens_mm);
            let vertical_scale = horizontal_scale * (f64::from(height) / f64::from(view_width));
            if !horizontal_scale.is_finite()
                || !vertical_scale.is_finite()
                || horizontal_scale <= 0.0
                || vertical_scale <= 0.0
            {
                return Err(PotError::invalid_argument(
                    "perspective projection dimensions are out of range",
                ));
            }
            ProjectionFrame::Perspective {
                horizontal_scale,
                vertical_scale,
            }
        }
        Projection::Panorama { fisheye } => ProjectionFrame::Panorama { fisheye },
        _ => {
            return Err(PotError::invalid_argument(
                "projection parameters are invalid",
            ));
        }
    };

    Ok(CameraFrame {
        forward,
        right,
        up,
        near: camera.near,
        far: camera.far,
        shift: camera.shift,
        depth_of_field: camera.depth_of_field,
        stereo_mode: camera.stereo_mode,
        interocular_distance: camera.interocular_distance,
        projection,
    })
}

pub(crate) fn ambient_visibility(
    bvh: &super::bvh::Bvh,
    triangles: &[Triangle],
    point: DVec3,
    normal: DVec3,
    distance: f64,
) -> f64 {
    const SAMPLE_COUNT: u32 = 12;
    let normal = normal.normalize_or_zero();
    if normal == DVec3::ZERO || !distance.is_finite() || distance <= 0.0 {
        return 1.0;
    }
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent).normalize_or_zero();
    let origin = point + normal * 1.0e-5;
    let mut visible = 0_u32;
    for index in 0..SAMPLE_COUNT {
        let fraction = (f64::from(index) + 0.5) / f64::from(SAMPLE_COUNT);
        let radius = (1.0 - fraction * fraction).sqrt();
        let angle = f64::from(index) * 2.399_963_229_728_653;
        let direction = (normal * fraction
            + tangent * (radius * angle.cos())
            + bitangent * (radius * angle.sin()))
        .normalize_or_zero();
        if bvh
            .closest_hit_two_sided(triangles, origin, direction, 1.0e-5, distance)
            .is_none()
        {
            visible += 1;
        }
    }
    f64::from(visible) / f64::from(SAMPLE_COUNT)
}

fn ray_for_pixel_in_frame(
    frame: &CameraFrame,
    camera_position: DVec3,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Result<Ray> {
    ray_for_subpixel_in_frame(frame, camera_position, width, height, x, y, 0.5, 0.5)
}

#[expect(
    clippy::too_many_arguments,
    reason = "subpixel ray generation keeps pixel and sample coordinates explicit"
)]
fn ray_for_subpixel_in_frame(
    frame: &CameraFrame,
    camera_position: DVec3,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<Ray> {
    if !pixel_in_bounds(width, height, x, y)
        || !sample_x.is_finite()
        || !sample_y.is_finite()
        || !(0.0..1.0).contains(&sample_x)
        || !(0.0..1.0).contains(&sample_y)
    {
        return Err(PotError::invalid_argument(
            "pixel sample coordinates must lie inside a non-empty image",
        ));
    }

    let (view_x, view_width, eye) = if frame.stereo_mode == StereoMode::SideBySide {
        let split = width / 2;
        if x < split {
            (x, split, -1.0)
        } else {
            (x - split, width - split, 1.0)
        }
    } else {
        (x, width, 0.0)
    };
    let ndc_x =
        2.0 * ((f64::from(view_x) + sample_x) / f64::from(view_width)) - 1.0 + 2.0 * frame.shift[0];
    let ndc_y = 1.0 - 2.0 * ((f64::from(y) + sample_y) / f64::from(height)) + 2.0 * frame.shift[1];
    let origin = camera_position + frame.right * (eye * frame.interocular_distance * 0.5);
    let mut ray = match frame.projection {
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => Ray {
            origin: origin + frame.right * (ndc_x * half_width) + frame.up * (ndc_y * half_height),
            direction: frame.forward,
        },
        ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } => {
            let direction = safe_normalize(
                frame.forward
                    + frame.right * (ndc_x * horizontal_scale)
                    + frame.up * (ndc_y * vertical_scale),
            )
            .ok_or_else(|| PotError::invalid_argument("pixel ray is out of range"))?;
            Ray { origin, direction }
        }
        ProjectionFrame::Panorama { fisheye: false } => {
            let longitude = ndc_x * std::f64::consts::PI;
            let latitude = ndc_y * std::f64::consts::FRAC_PI_2;
            let direction = safe_normalize(
                frame.forward * (longitude.cos() * latitude.cos())
                    + frame.right * (longitude.sin() * latitude.cos())
                    + frame.up * latitude.sin(),
            )
            .ok_or_else(|| PotError::invalid_argument("panorama ray is out of range"))?;
            Ray { origin, direction }
        }
        ProjectionFrame::Panorama { fisheye: true } => {
            let radius = ndc_x.hypot(ndc_y);
            if radius > 1.0 {
                return Err(PotError::invalid_argument(
                    "fisheye pixel falls outside the projection circle",
                ));
            }
            let angle = radius * std::f64::consts::PI;
            let direction = if radius <= f64::EPSILON {
                frame.forward
            } else {
                let radial = (frame.right * ndc_x + frame.up * ndc_y) / radius;
                frame.forward * angle.cos() + radial * angle.sin()
            };
            Ray { origin, direction }
        }
    };
    if let Some(dof) = frame.depth_of_field
        && dof.aperture_radius > 0.0
    {
        let plane_distance = dof.focus_distance / ray.direction.dot(frame.forward).max(1.0e-6);
        let focus_point = ray.origin + ray.direction * plane_distance;
        let aperture =
            concentric_disk(sample_x, sample_y, dof.aperture_blades) * dof.aperture_radius;
        ray.origin += frame.right * aperture.x + frame.up * aperture.y;
        ray.direction = safe_normalize(focus_point - ray.origin)
            .ok_or_else(|| PotError::invalid_argument("depth-of-field ray is out of range"))?;
    }
    if !is_finite_vec3(ray.origin) || !is_finite_vec3(ray.direction) {
        return Err(PotError::invalid_argument("pixel ray is out of range"));
    }
    Ok(ray)
}

fn concentric_disk(sample_x: f64, sample_y: f64, blades: u32) -> glam::DVec2 {
    let offset = glam::DVec2::new(2.0 * sample_x - 1.0, 2.0 * sample_y - 1.0);
    if offset == glam::DVec2::ZERO {
        return offset;
    }
    let (radius, angle) = if offset.x.abs() > offset.y.abs() {
        (
            offset.x,
            std::f64::consts::FRAC_PI_4 * (offset.y / offset.x),
        )
    } else {
        (
            offset.y,
            std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (offset.x / offset.y),
        )
    };
    let mut point = glam::DVec2::new(radius * angle.cos(), radius * angle.sin());
    if blades >= 3 {
        let sector = std::f64::consts::TAU / f64::from(blades);
        let local_angle = angle.rem_euclid(sector) - sector * 0.5;
        point *= (sector * 0.5).cos() / local_angle.cos().max(f64::MIN_POSITIVE);
    }
    point
}

fn sample_offset(seed: u32, x: u32, y: u32, sample_index: u32) -> (f64, f64) {
    if sample_index == 0 {
        return (0.5, 0.5);
    }
    let key = seed
        ^ x.wrapping_mul(0x9e37_79b9)
        ^ y.wrapping_mul(0x85eb_ca6b)
        ^ sample_index.wrapping_mul(0xc2b2_ae35);
    let x_offset = f64::from(mix_sample_hash(key)) / 4_294_967_296.0;
    let y_offset = f64::from(mix_sample_hash(key ^ 0xa511_e9b3)) / 4_294_967_296.0;
    (x_offset, y_offset)
}

fn mix_sample_hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn nearest_hit(
    triangles: &[PreparedTriangle],
    ray: Ray,
    frame: &CameraFrame,
) -> Option<(usize, Hit)> {
    let mut nearest: Option<(usize, Hit)> = None;
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        if let Some(hit) = intersect_triangle(triangle, ray, frame)
            && nearest.is_none_or(|(_, closest)| hit.depth < closest.depth)
        {
            nearest = Some((triangle_index, hit));
        }
    }
    nearest
}

fn nearest_visible_hit(
    triangles: &[PreparedTriangle],
    ray: Ray,
    frame: &CameraFrame,
) -> Option<(usize, Hit)> {
    let mut nearest: Option<(usize, Hit)> = None;
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        if triangle_opacity(&triangle.source) > 0.0
            && let Some(hit) = intersect_triangle(triangle, ray, frame)
            && nearest.is_none_or(|(_, closest)| hit.depth < closest.depth)
        {
            nearest = Some((triangle_index, hit));
        }
    }
    nearest
}

fn triangle_opacity(triangle: &Triangle) -> f64 {
    match triangle.alpha_mode {
        AlphaMode::Opaque => 1.0,
        AlphaMode::Blend => triangle.color[3].clamp(0.0, 1.0),
        AlphaMode::Clip => {
            if triangle.color[3] >= triangle.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn composite_sample(
    triangles: &[PreparedTriangle],
    ray: Ray,
    frame: &CameraFrame,
    background: [f64; 4],
) -> [f64; 4] {
    const MAX_ALPHA_LAYERS: usize = 16;
    let mut hits: [Option<(usize, Hit)>; MAX_ALPHA_LAYERS] = [None; MAX_ALPHA_LAYERS];
    let mut hit_count = 0;
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let Some(hit) = intersect_triangle(triangle, ray, frame) else {
            continue;
        };
        if triangle_opacity(&triangle.source) <= 0.0 {
            continue;
        }
        let mut insert_at = hit_count;
        while insert_at > 0
            && hits[insert_at - 1].is_some_and(|(_, previous)| previous.depth > hit.depth)
        {
            insert_at -= 1;
        }
        if insert_at >= MAX_ALPHA_LAYERS {
            continue;
        }
        let next_count = (hit_count + 1).min(MAX_ALPHA_LAYERS);
        for index in (insert_at + 1..next_count).rev() {
            hits[index] = hits[index - 1];
        }
        hits[insert_at] = Some((triangle_index, hit));
        hit_count = next_count;
    }
    let mut transmittance = 1.0;
    let mut alpha = 0.0;
    let mut premultiplied = DVec3::ZERO;
    for (triangle_index, _) in hits.iter().take(hit_count).flatten().copied() {
        let triangle = &triangles[triangle_index].source;
        let color = triangle.color;
        let opacity = triangle_opacity(triangle);
        let contribution = transmittance * opacity;
        premultiplied += DVec3::new(color[0], color[1], color[2]) * contribution;
        alpha += contribution;
        transmittance *= 1.0 - opacity;
        if transmittance <= 1.0e-6 {
            break;
        }
    }
    let background_alpha = background[3].clamp(0.0, 1.0);
    premultiplied += DVec3::new(background[0], background[1], background[2])
        * (transmittance * background_alpha);
    alpha += transmittance * background_alpha;
    if alpha <= f64::MIN_POSITIVE {
        return background;
    }
    let color = premultiplied / alpha;
    [color.x, color.y, color.z, alpha]
}

fn prepare_triangles(triangles: &[Triangle]) -> Result<Vec<PreparedTriangle>> {
    let mut prepared = Vec::new();
    prepared.try_reserve_exact(triangles.len()).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "triangle data exceeds available memory",
        )
    })?;

    for triangle in triangles {
        if triangle
            .positions
            .iter()
            .any(|point| !is_finite_vec3(*point))
            || triangle
                .color
                .iter()
                .any(|component| !component.is_finite())
        {
            return Err(PotError::invalid_argument(
                "triangle positions and colors must be finite",
            ));
        }
        let edge1 = triangle.positions[1] - triangle.positions[0];
        let edge2 = triangle.positions[2] - triangle.positions[0];
        if !is_finite_vec3(edge1) || !is_finite_vec3(edge2) {
            return Err(PotError::invalid_argument(
                "triangle coordinates exceed the supported numeric range",
            ));
        }
        let Some(normal) = safe_normalize(edge1.cross(edge2)) else {
            continue;
        };

        let edge0_altitude = altitude_from_opposite_edge(
            normal,
            triangle.positions[2] - triangle.positions[1],
            triangle.positions[0] - triangle.positions[1],
        );
        let edge1_altitude = altitude_from_opposite_edge(
            normal,
            triangle.positions[0] - triangle.positions[2],
            triangle.positions[1] - triangle.positions[2],
        );
        let edge2_altitude = altitude_from_opposite_edge(
            normal,
            triangle.positions[1] - triangle.positions[0],
            triangle.positions[2] - triangle.positions[0],
        );
        let (Some(edge0_altitude), Some(edge1_altitude), Some(edge2_altitude)) =
            (edge0_altitude, edge1_altitude, edge2_altitude)
        else {
            continue;
        };
        if edge0_altitude <= f64::MIN_POSITIVE
            || edge1_altitude <= f64::MIN_POSITIVE
            || edge2_altitude <= f64::MIN_POSITIVE
        {
            continue;
        }
        prepared.push(PreparedTriangle {
            edge1,
            edge2,
            normal,
            edge_altitudes: [edge0_altitude, edge1_altitude, edge2_altitude],
            source: *triangle,
        });
    }
    Ok(prepared)
}

fn altitude_from_opposite_edge(
    normal: DVec3,
    opposite_edge: DVec3,
    to_vertex: DVec3,
) -> Option<f64> {
    if !is_finite_vec3(opposite_edge) || !is_finite_vec3(to_vertex) {
        return None;
    }
    let edge_direction = safe_normalize(opposite_edge)?;
    let perpendicular = normal.cross(edge_direction);
    let altitude = perpendicular.dot(to_vertex).abs();
    altitude.is_finite().then_some(altitude)
}

fn intersect_triangle(triangle: &PreparedTriangle, ray: Ray, frame: &CameraFrame) -> Option<Hit> {
    const BARYCENTRIC_EPSILON: f64 = 1.0e-12;
    let p_vector = ray.direction.cross(triangle.edge2);
    let determinant = triangle.edge1.dot(p_vector);
    if !determinant.is_finite()
        || determinant.abs() <= f64::MIN_POSITIVE
        || (!triangle.source.double_sided && determinant <= 0.0)
    {
        return None;
    }

    let inverse_determinant = 1.0 / determinant;
    if !inverse_determinant.is_finite() {
        return None;
    }
    let from_vertex = ray.origin - triangle.source.positions[0];
    let barycentric_1 = from_vertex.dot(p_vector) * inverse_determinant;
    let q_vector = from_vertex.cross(triangle.edge1);
    let barycentric_2 = ray.direction.dot(q_vector) * inverse_determinant;
    let distance = triangle.edge2.dot(q_vector) * inverse_determinant;
    let barycentric_0 = 1.0 - barycentric_1 - barycentric_2;
    if !barycentric_0.is_finite()
        || !barycentric_1.is_finite()
        || !barycentric_2.is_finite()
        || !distance.is_finite()
        || distance <= 0.0
        || barycentric_0 < -BARYCENTRIC_EPSILON
        || barycentric_1 < -BARYCENTRIC_EPSILON
        || barycentric_2 < -BARYCENTRIC_EPSILON
        || barycentric_0 > 1.0 + BARYCENTRIC_EPSILON
        || barycentric_1 > 1.0 + BARYCENTRIC_EPSILON
        || barycentric_2 > 1.0 + BARYCENTRIC_EPSILON
    {
        return None;
    }

    let depth = distance * ray.direction.dot(frame.forward);
    if !depth.is_finite() || depth < frame.near || depth > frame.far {
        return None;
    }
    Some(Hit {
        depth,
        barycentric: [
            barycentric_0.clamp(0.0, 1.0),
            barycentric_1.clamp(0.0, 1.0),
            barycentric_2.clamp(0.0, 1.0),
        ],
        direction: ray.direction,
    })
}

fn is_wire_edge(
    triangle: &PreparedTriangle,
    hit: Hit,
    frame: &CameraFrame,
    width: u32,
    height: u32,
) -> bool {
    let pixel_footprint = match frame.projection {
        ProjectionFrame::Orthographic { half_height, .. } => {
            (2.0 * half_height) / f64::from(height)
        }
        ProjectionFrame::Perspective {
            horizontal_scale, ..
        } => (2.0 * hit.depth * horizontal_scale) / f64::from(width),
        ProjectionFrame::Panorama { .. } => (std::f64::consts::PI * hit.depth) / f64::from(height),
    };
    let incidence = hit.direction.dot(triangle.normal).abs().max(1.0e-6);
    let edge_radius = (pixel_footprint / incidence) * 0.65;
    if !edge_radius.is_finite() {
        return true;
    }
    triangle
        .edge_altitudes
        .iter()
        .zip(hit.barycentric)
        .any(|(altitude, barycentric)| barycentric * altitude <= edge_radius)
}

fn color_to_srgba(color: [f64; 4]) -> [u8; 4] {
    [
        linear_to_srgb_byte(color[0]),
        linear_to_srgb_byte(color[1]),
        linear_to_srgb_byte(color[2]),
        unit_to_byte(color[3]),
    ]
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "depth is finite and non-negative before clamping to the f32 maximum"
)]
fn depth_to_f32(depth: f64) -> f32 {
    depth.min(f64::from(f32::MAX)) as f32
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "linear color channels are clamped to the f32 range"
)]
fn linear_color_to_f32(color: [f64; 4]) -> [f32; 4] {
    let maximum = f64::from(f32::MAX);
    color.map(|channel| channel.clamp(-maximum, maximum) as f32)
}
fn background_to_linear_f64(background: [u8; 4]) -> [f64; 4] {
    [
        srgb_byte_to_linear(background[0]),
        srgb_byte_to_linear(background[1]),
        srgb_byte_to_linear(background[2]),
        f64::from(background[3]) / 255.0,
    ]
}

fn srgb_byte_to_linear(channel: u8) -> f64 {
    let encoded = f64::from(channel) / 255.0;
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn display_color_to_float(color: [u8; 4]) -> [f32; 4] {
    color.map(|channel| f32::from(channel) / 255.0)
}

fn linear_to_srgb_byte(channel: f64) -> u8 {
    let linear = channel.clamp(0.0, 1.0);
    unit_to_byte(crate::color::linear_to_srgb(linear))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the input is clamped to the normalized 8-bit channel range"
)]
fn unit_to_byte(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn normal_to_rgba(normal: DVec3) -> [u8; 4] {
    [
        unit_to_byte(normal.x * 0.5 + 0.5),
        unit_to_byte(normal.y * 0.5 + 0.5),
        unit_to_byte(normal.z * 0.5 + 0.5),
        255,
    ]
}

fn depth_to_rgba(depth: f64, near: f64, far: f64) -> [u8; 4] {
    let normalized = ((depth - near) / (far - near)).clamp(0.0, 1.0);
    let grayscale = unit_to_byte(normalized);
    [grayscale, grayscale, grayscale, 255]
}

fn id_to_rgba(object_index: u32) -> [u8; 4] {
    let mut value = object_index.wrapping_add(0x9e37_79b9);
    value = (value ^ (value >> 16)).wrapping_mul(0x7feb_352d);
    value = (value ^ (value >> 15)).wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    let [red, green, blue, _] = value.to_le_bytes();
    [
        64 + (red & 0xbf),
        64 + (green & 0xbf),
        64 + (blue & 0xbf),
        255,
    ]
}

fn allocate_filled<T: Clone>(length: usize, value: T, name: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            format!("{name} exceeds available memory"),
        )
    })?;
    values.resize(length, value);
    Ok(values)
}

fn safe_normalize(vector: DVec3) -> Option<DVec3> {
    if !is_finite_vec3(vector) {
        return None;
    }
    let scale = vector.x.abs().max(vector.y.abs()).max(vector.z.abs());
    if scale <= f64::MIN_POSITIVE {
        return None;
    }
    let scaled = vector / scale;
    let length = scaled.length_squared().sqrt();
    (length.is_finite() && length > 0.0).then_some(scaled / length)
}

fn is_finite_vec3(vector: DVec3) -> bool {
    vector.x.is_finite() && vector.y.is_finite() && vector.z.is_finite()
}

#[cfg(test)]
mod tests {
    use super::{Camera, Projection, StereoMode, camera_frame, pixel_in_bounds, ray_for_pixel};
    use glam::DVec3;

    fn camera(projection: Projection) -> Camera {
        Camera {
            position: DVec3::new(1.0, 2.0, 3.0),
            target: DVec3::new(1.0, 2.0, 2.0),
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection,
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        }
    }

    #[test]
    fn perspective_shift_moves_center_ray_along_camera_axes() {
        let width = 8;
        let height = 8;
        let mut camera = camera(Projection::Perspective {
            lens_mm: 50.0,
            sensor_width_mm: 36.0,
        });
        let baseline = ray_for_pixel(&camera, width, height, 4, 4);
        let frame = camera_frame(&camera, width, height);
        assert!(baseline.is_ok() && frame.is_ok());
        let (Ok((_, baseline)), Ok(frame)) = (baseline, frame) else {
            return;
        };
        camera.shift = [0.25, -0.25];
        let shifted = ray_for_pixel(&camera, width, height, 4, 4);
        assert!(shifted.is_ok());
        let Ok((_, shifted)) = shifted else {
            return;
        };
        let delta = shifted - baseline;
        assert!(delta.dot(frame.right) > 0.0);
        assert!(delta.dot(frame.up) < 0.0);
    }

    #[test]
    fn pixel_rays_round_trip_to_their_pixel_centers() {
        let width = 7;
        let height = 5;
        for projection in [
            Projection::Orthographic { height: 4.0 },
            Projection::Perspective {
                lens_mm: 50.0,
                sensor_width_mm: 36.0,
            },
        ] {
            let camera = camera(projection);
            let frame = camera_frame(&camera, width, height);
            assert!(frame.is_ok());
            let Ok(frame) = frame else {
                return;
            };
            let x = 2;
            let y = 3;
            let ray = ray_for_pixel(&camera, width, height, x, y);
            assert!(ray.is_ok());
            let Ok((origin, direction)) = ray else {
                return;
            };
            let point = origin + direction * 5.0;
            let relative = point - camera.position;
            let camera_x = relative.dot(frame.right);
            let camera_y = relative.dot(frame.up);
            let camera_depth = relative.dot(frame.forward);
            let (ndc_x, ndc_y) = match frame.projection {
                super::ProjectionFrame::Orthographic {
                    half_width,
                    half_height,
                } => (camera_x / half_width, camera_y / half_height),
                super::ProjectionFrame::Perspective {
                    horizontal_scale,
                    vertical_scale,
                    ..
                } => (
                    camera_x / camera_depth / horizontal_scale,
                    camera_y / camera_depth / vertical_scale,
                ),
                super::ProjectionFrame::Panorama { fisheye: false } => (
                    camera_x.atan2(camera_depth) / std::f64::consts::PI,
                    (camera_y / relative.length()).asin() / std::f64::consts::FRAC_PI_2,
                ),
                super::ProjectionFrame::Panorama { fisheye: true } => {
                    let angle = (camera_depth / relative.length()).clamp(-1.0, 1.0).acos();
                    let radial = glam::DVec2::new(camera_x, camera_y).normalize_or_zero();
                    (
                        radial.x * angle / std::f64::consts::PI,
                        radial.y * angle / std::f64::consts::PI,
                    )
                }
            };
            let screen_x = f64::midpoint(ndc_x, 1.0) * f64::from(width);
            let screen_y = f64::midpoint(-ndc_y, 1.0) * f64::from(height);
            assert!((screen_x - (f64::from(x) + 0.5)).abs() < 1.0e-10);
            assert!((screen_y - (f64::from(y) + 0.5)).abs() < 1.0e-10);
        }
    }

    #[test]
    fn pixel_rays_reject_empty_images_and_out_of_bounds_coordinates() {
        let camera = camera(Projection::Orthographic { height: 2.0 });
        assert!(ray_for_pixel(&camera, 0, 1, 0, 0).is_err());
        assert!(ray_for_pixel(&camera, 1, 0, 0, 0).is_err());
        assert!(ray_for_pixel(&camera, 2, 2, 2, 0).is_err());
        assert!(ray_for_pixel(&camera, 2, 2, 0, 2).is_err());
        assert!(pixel_in_bounds(2, 2, 1, 1));
        assert!(!pixel_in_bounds(2, 2, 2, 1));
        assert!(!pixel_in_bounds(2, 2, 1, 2));
    }
}

#[cfg(test)]
mod line_tests {
    use super::{Camera, Line, Mode, Projection, StereoMode, rasterize_with_lines};
    use glam::DVec3;

    #[test]
    fn center_sample_stroke_writes_color_object_element_and_depth() {
        let camera = Camera {
            position: DVec3::new(1.0, 2.0, 3.0),
            target: DVec3::new(1.0, 2.0, 2.0),
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 4.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        };
        let line = Line {
            start: DVec3::new(0.0, 2.0, 2.0),
            end: DVec3::new(2.0, 2.0, 2.0),
            radius_start: 0.1,
            radius_end: 0.1,
            color: [1.0, 0.0, 0.0, 1.0],
            object_index: 1,
            element_index: 2,
        };
        let output = rasterize_with_lines(
            &[],
            &[line],
            &camera,
            7,
            5,
            Mode::Beauty,
            [242, 242, 242, 255],
            1,
            0,
        );
        assert!(output.is_ok());
        let Ok(output) = output else {
            return;
        };
        let index = 2 * 7 + 3;
        assert_eq!(output.ids[index], 1);
        assert_eq!(output.elements[index], 2);
        assert!((f64::from(output.depths[index]) - 1.0).abs() < 1.0e-6);
        assert_eq!(&output.rgba[index * 4..index * 4 + 4], &[255, 0, 0, 255]);
    }
}

#[cfg(kani)]
mod kani_verification {
    use super::pixel_in_bounds;

    #[kani::proof]
    fn pixel_bounds_accept_only_valid_coordinates() {
        let width: u32 = kani::any();
        let height: u32 = kani::any();
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        if pixel_in_bounds(width, height, x, y) {
            kani::assert(width > 0, "an in-bounds pixel has positive width");
            kani::assert(height > 0, "an in-bounds pixel has positive height");
            kani::assert(x < width, "an in-bounds x coordinate is below width");
            kani::assert(y < height, "an in-bounds y coordinate is below height");
        } else {
            kani::assert(
                width == 0 || height == 0 || x >= width || y >= height,
                "a rejected pixel is outside a non-empty image",
            );
        }
    }
}
