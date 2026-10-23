use std::f64::consts::TAU;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::wireframe::{self, PointSquare, ScreenPoint, WireEdge, WirePoint, WireSegment};
use crate::error::{ErrorCode, PotError, Result};
use crate::float::equal_f64;
use crate::geom::GeneratedCorners;
use crate::model::{CameraData, CameraProjection, RayKind};

/// A camera projection in Blender's units: angles in radians, lengths in millimetres.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum Projection {
    Orthographic {
        height: f64,
    },
    Perspective {
        lens_mm: f64,
        sensor_width_mm: f64,
    },
    /// Cycles' equirectangular mapping over the longitude and latitude limits `[min, max]`.
    Equirectangular {
        longitude: [f64; 2],
        latitude: [f64; 2],
    },
    FisheyeEquidistant {
        fov: f64,
    },
    /// Cycles' equisolid fisheye: `fisheye_lens` is Blender's focal length for this mapping, not the perspective `lens`.
    FisheyeEquisolid {
        fov: f64,
        fisheye_lens: f64,
        sensor_width_mm: f64,
        sensor_height_mm: f64,
        sensor_fit: SensorFit,
    },
}

/// Blender's sensor fit. The fit's size is the width (AUTO and HORIZONTAL) or the height (VERTICAL), and the other side
/// follows the render's aspect (`blender_camera_viewplane` in `intern/cycles/blender/camera.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub(crate) enum SensorFit {
    Auto,
    Horizontal,
    Vertical,
}

/// The sensor of an equisolid fisheye for a `width` x `height` view, in millimetres, as Cycles sets `sensorwidth` and
/// `sensorheight` for a fisheye camera.
fn equisolid_sensor(
    sensor_width_mm: f64,
    sensor_height_mm: f64,
    sensor_fit: SensorFit,
    width: f64,
    height: f64,
) -> (f64, f64) {
    let (horizontal_fit, size) = match sensor_fit {
        SensorFit::Auto => (width > height, sensor_width_mm),
        SensorFit::Horizontal => (true, sensor_width_mm),
        SensorFit::Vertical => (false, sensor_height_mm),
    };
    if horizontal_fit {
        (size, size * height / width)
    } else {
        (size * width / height, size)
    }
}

/// Which renderer a camera is prepared for. Cycles draws a panorama camera with its panorama projection. Blender's EEVEE
/// and Workbench draw it as a perspective camera with `lens`: EEVEE's panorama branch is disabled (`eevee_camera.cc`), and
/// Workbench's camera matrix is the same as for a perspective camera (both verified by `probe_engine_panorama_camera.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CameraEngine {
    /// The path tracer, which is Cycles' model: a panorama camera uses its panorama projection.
    Path,
    /// The rasterizer (solid, wire, normal, depth, and id modes, and the realtime engine): a panorama camera is perspective.
    Raster,
}

/// The render projection of a stored camera. Panorama types and fisheye settings come from the camera data. Every render,
/// preview, pick, and adaptive dicing reads the camera through this one function, with the engine that renders it.
pub(crate) fn projection_for_camera(
    camera_data: &CameraData,
    engine: CameraEngine,
) -> Result<Projection> {
    Ok(match camera_data.projection {
        CameraProjection::Orthographic => Projection::Orthographic {
            height: camera_data.ortho_scale,
        },
        CameraProjection::Perspective => Projection::Perspective {
            lens_mm: camera_data.lens_mm,
            sensor_width_mm: camera_data.sensor_width_mm,
        },
        CameraProjection::Panorama => {
            let panorama = match camera_data.panorama_type.as_str() {
                "equirectangular" => Projection::Equirectangular {
                    longitude: [camera_data.longitude_min, camera_data.longitude_max],
                    latitude: [camera_data.latitude_min, camera_data.latitude_max],
                },
                "fisheye_equidistant" => Projection::FisheyeEquidistant {
                    fov: camera_data.fisheye_fov,
                },
                "fisheye_equisolid" => Projection::FisheyeEquisolid {
                    fov: camera_data.fisheye_fov,
                    fisheye_lens: camera_data.fisheye_lens,
                    sensor_width_mm: camera_data.sensor_width_mm,
                    sensor_height_mm: camera_data.sensor_height_mm,
                    sensor_fit: sensor_fit(&camera_data.sensor_fit)?,
                },
                _ => {
                    return Err(PotError::invalid_argument(
                        "camera panorama_type must be equirectangular, fisheye_equidistant, or fisheye_equisolid",
                    ));
                }
            };
            match engine {
                CameraEngine::Path => panorama,
                // The rasterizer draws the panorama camera as the perspective camera Blender's EEVEE and Workbench draw.
                CameraEngine::Raster => Projection::Perspective {
                    lens_mm: camera_data.lens_mm,
                    sensor_width_mm: camera_data.sensor_width_mm,
                },
            }
        }
    })
}

fn sensor_fit(name: &str) -> Result<SensorFit> {
    match name {
        "AUTO" => Ok(SensorFit::Auto),
        "HORIZONTAL" => Ok(SensorFit::Horizontal),
        "VERTICAL" => Ok(SensorFit::Vertical),
        _ => Err(PotError::invalid_argument(
            "camera sensor_fit must be AUTO, HORIZONTAL, or VERTICAL",
        )),
    }
}

impl Projection {
    /// The angle one pixel spans at the image centre, in radians, for the panorama projections.
    pub(crate) fn panorama_angle_per_pixel(self, width: u32, height: u32) -> Option<f64> {
        let height = f64::from(height);
        match self {
            Self::Equirectangular { latitude, .. } => Some((latitude[1] - latitude[0]) / height),
            Self::FisheyeEquidistant { fov } => Some(fov / height),
            Self::FisheyeEquisolid {
                fisheye_lens,
                sensor_width_mm,
                sensor_height_mm,
                sensor_fit,
                ..
            } => {
                let (_, sensor_height) = equisolid_sensor(
                    sensor_width_mm,
                    sensor_height_mm,
                    sensor_fit,
                    f64::from(width),
                    height,
                );
                Some(sensor_height / (fisheye_lens * height))
            }
            Self::Orthographic { .. } | Self::Perspective { .. } => None,
        }
    }

    /// Whether this projection covers every direction. Its depth is then the distance from the camera: a view-axis depth
    /// is negative behind the camera, where a panorama still has pixels, so it cannot order or clip them.
    pub(crate) fn measures_ray_distance(self) -> bool {
        !matches!(self, Self::Orthographic { .. } | Self::Perspective { .. })
    }
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

impl Camera {
    /// The depth a world point records: its distance along the view axis for a planar camera, and its distance from the
    /// camera for a panorama, where `Projection::measures_ray_distance` holds. Depth buffers, pick, and clipping use this.
    pub(crate) fn depth_of(&self, point: DVec3) -> f64 {
        let relative = point - self.position;
        if self.projection.measures_ray_distance() {
            relative.length()
        } else {
            relative.dot((self.target - self.position).normalize_or_zero())
        }
    }

    /// The distance along a ray with `direction` (a unit vector) to a point whose recorded depth is `depth`.
    pub(crate) fn ray_distance_for_depth(&self, direction: DVec3, depth: f64) -> f64 {
        if self.projection.measures_ray_distance() {
            depth
        } else {
            let forward = (self.target - self.position).normalize_or_zero();
            depth / direction.dot(forward).max(f64::MIN_POSITIVE)
        }
    }
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
#[derive(Debug, Clone)]
pub(crate) struct Triangle {
    pub(crate) positions: [DVec3; 3],
    /// The Generated coordinate of each corner and the Jacobian the corners share. A hit interpolates the corners over
    /// its barycentric weights, as Cycles stores Generated per vertex and interpolates it
    /// (`intern/cycles/blender/mesh.cpp`, `ATTR_STD_GENERATED`), so a corner value of zero stays zero.
    pub(crate) generated: GeneratedCorners,
    pub(crate) object_index: u32,
    /// Source element index; face records carry per-slot UV bindings through subdivision copies.
    pub(crate) element_index: u32,
    pub(crate) color: [f64; 4],
    pub(crate) alpha_mode: AlphaMode,
    pub(crate) alpha_threshold: f64,
    pub(crate) double_sided: bool,
    pub(crate) material_index: usize,
    pub(crate) uv: [[f64; 2]; 3],
    /// Corner UVs of every UV layer of the source mesh, in layer order; empty without layers.
    pub(crate) layer_uvs: Vec<[[f64; 2]; 3]>,
    /// World-space corner shading normals; `None` for flat faces, which keep their geometric normal.
    pub(crate) normals: Option<[DVec3; 3]>,
    /// Corner colors of every color layer the object carries, in `ObjectRecord::color_layers`
    /// order; empty without color layers.
    pub(crate) layer_colors: Vec<[[f64; 4]; 3]>,
    /// Evaluates the material graph at each hit of a realtime Beauty triangle whose graph reads color layers or needs
    /// hit data. Every realtime Beauty hit is shaded per pixel regardless, flat triangles included.
    pub(crate) lit_per_pixel: bool,
    /// The owning object's ray mask (one `RayKind` bit each); realtime Beauty filters primary rays.
    pub(crate) ray_mask: u8,
}

impl Triangle {
    /// This triangle's shading attributes over new corner geometry, whose Generated coordinates are `generated`.
    pub(crate) fn with_corners(
        &self,
        positions: [DVec3; 3],
        generated: GeneratedCorners,
        uv: [[f64; 2]; 3],
        layer_uvs: Vec<[[f64; 2]; 3]>,
        layer_colors: Vec<[[f64; 4]; 3]>,
    ) -> Self {
        Self {
            positions,
            generated,
            object_index: self.object_index,
            element_index: self.element_index,
            color: self.color,
            alpha_mode: self.alpha_mode,
            alpha_threshold: self.alpha_threshold,
            double_sided: self.double_sided,
            material_index: self.material_index,
            uv,
            layer_uvs,
            normals: self.normals,
            layer_colors,
            lit_per_pixel: self.lit_per_pixel,
            ray_mask: self.ray_mask,
        }
    }

    /// A copy without the per-layer data, for passes that only shade the triangle. Per-pixel
    /// shading evaluates the material graph, which reads that data, so such copies keep it.
    pub(crate) fn shading_copy(&self) -> Self {
        if self.lit_per_pixel {
            self.clone()
        } else {
            self.with_corners(
                self.positions,
                self.generated,
                self.uv,
                Vec::new(),
                Vec::new(),
            )
        }
    }
}

/// Where a visible hit lies on its triangle, for per-pixel shading.
pub(crate) struct SurfaceHit {
    /// Hit point in world space.
    pub(crate) point: DVec3,
    /// Shadow-ray origin (see [`smooth_shadow_origin`]).
    pub(crate) shadow_point: DVec3,
    /// Interpolated shading normal.
    pub(crate) normal: DVec3,
    /// Unit triangle normal.
    pub(crate) geometric: DVec3,
    /// Barycentric weights of the hit over the triangle's corners.
    pub(crate) barycentric: [f64; 3],
    /// Unit direction from the hit back toward the camera.
    pub(crate) view: DVec3,
}

/// What the raster hit evaluator returns: its linear RGBA, whether it is a camera surface,
/// and the tint of the transparent share it passes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SurfaceSample {
    pub(crate) color: [f64; 4],
    /// Whether this sample is an actual surface rather than a volume-only boundary.
    pub(crate) has_surface: bool,
    /// The Transparent closure's colour per channel, as `(1 - alpha) * tint` is its transmittance. White for a
    /// surface that passes nothing, and a Blender DITHERED material's channel mean (`scene.rs` `beauty_tint`).
    pub(crate) transparent_tint: [f64; 3],
}

impl SurfaceSample {
    /// A sample with no transparent share: its colour, and a white tint.
    pub(crate) fn opaque(color: [f64; 4]) -> Self {
        Self {
            color,
            has_surface: true,
            transparent_tint: [1.0; 3],
        }
    }
}

/// Linear RGBA and the transparent tint at `hit` on `triangle`.
pub(crate) trait SurfaceShader {
    fn shade(&self, triangle: &Triangle, hit: &SurfaceHit) -> Result<SurfaceSample>;
}

/// The camera's view axes in world space, in Blender's view-space order: `right`, `up`, and
/// `backward` (toward the viewer). Workbench's studio lights are fixed in this space.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewBasis {
    right: DVec3,
    up: DVec3,
    backward: DVec3,
}

impl ViewBasis {
    pub(crate) fn from_camera(camera: &Camera) -> Result<Self> {
        let (forward, right, up) = view_axes(camera)?;
        Ok(Self {
            right,
            up,
            backward: -forward,
        })
    }

    /// `direction`, given in world space, expressed in view space.
    pub(crate) fn to_view(self, direction: DVec3) -> DVec3 {
        DVec3::new(
            direction.dot(self.right),
            direction.dot(self.up),
            direction.dot(self.backward),
        )
    }
}

/// Shadow-ray origin for a hit on a smooth triangle, after Hanika's shadow-terminator offset
/// (Cycles' `shadow_ray_smooth_surface_offset`). It lifts the flat hit point onto the surface
/// that the corner normals describe, so shadow rays near the terminator do not re-enter the
/// curved surface. Flat triangles return `point` unchanged.
pub(crate) fn smooth_shadow_origin(
    triangle: &Triangle,
    point: DVec3,
    barycentric: [f64; 3],
) -> DVec3 {
    let Some(normals) = triangle.normals else {
        return point;
    };
    let mut offset = DVec3::ZERO;
    for (corner, normal) in normals.iter().enumerate() {
        let tangent = point - triangle.positions[corner];
        let below = tangent.dot(*normal).min(0.0);
        offset += (tangent - *normal * below) * barycentric[corner];
    }
    point + offset
}

/// Barycentric blend of corner normals, normalised; `fallback` covers blends that cancel.
pub(crate) fn blend_corner_normals(
    corners: [DVec3; 3],
    weights: [f64; 3],
    fallback: DVec3,
) -> DVec3 {
    let normal = (corners[0] * weights[0] + corners[1] * weights[1] + corners[2] * weights[2])
        .normalize_or_zero();
    if normal == DVec3::ZERO {
        fallback
    } else {
        normal
    }
}

/// Barycentric blend of one UV layer's corner values.
pub(crate) fn blend_uv(corners: &[[f64; 2]; 3], weights: [f64; 3]) -> [f64; 2] {
    [
        corners[0][0] * weights[0] + corners[1][0] * weights[1] + corners[2][0] * weights[2],
        corners[0][1] * weights[0] + corners[1][1] * weights[1] + corners[2][1] * weights[2],
    ]
}

/// Barycentric blend of one color layer's corner values.
pub(crate) fn blend_color(corners: &[[f64; 4]; 3], weights: [f64; 3]) -> [f64; 4] {
    std::array::from_fn(|channel| {
        corners[0][channel] * weights[0]
            + corners[1][channel] * weights[1]
            + corners[2][channel] * weights[2]
    })
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
    /// Object ray mask; realtime Beauty leaves out lines hidden from camera rays.
    pub(crate) ray_mask: u8,
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
pub(crate) struct CameraFrame {
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
    /// The angle one pixel spans at the centre of a panorama image; `None` for the planar projections.
    panorama_pixel_angle: Option<f64>,
    /// Whether a point's depth is its distance from the camera (a panorama) rather than its view-axis distance.
    depth_is_distance: bool,
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
    Equirectangular {
        longitude: [f64; 2],
        latitude: [f64; 2],
    },
    FisheyeEquidistant {
        fov: f64,
    },
    /// The sensor is the fitted size of the view, in millimetres.
    FisheyeEquisolid {
        fov: f64,
        fisheye_lens: f64,
        sensor_width_mm: f64,
        sensor_height_mm: f64,
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
    distance: f64,
    barycentric: [f64; 3],
    direction: DVec3,
}

#[derive(Clone, Copy)]
struct VisibleHit {
    triangle_index: usize,
    hit: Hit,
    color: [f64; 4],
    transparent_tint: [f64; 3],
}

/// The world ray through a pixel centre, or `None` when the projection has no ray there (outside a fisheye's circle).
pub(crate) fn ray_for_pixel(
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Result<Option<(DVec3, DVec3)>> {
    if !pixel_in_bounds(width, height, x, y) {
        return Err(PotError::invalid_argument(
            "pixel coordinates must lie inside a non-empty image",
        ));
    }

    let frame = camera_frame(camera, width, height)?;
    let ray = ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?;
    Ok(ray.map(|ray| (ray.origin, ray.direction)))
}

/// The world ray through a sample of a pixel, or `None` when the projection has no ray there.
pub(crate) fn ray_for_sample(
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<Option<(DVec3, DVec3)>> {
    let frame = camera_frame(camera, width, height)?;
    ray_for_sample_in_frame(&frame, camera, width, height, x, y, sample_x, sample_y)
}

/// [`ray_for_sample`] with the frame already built by [`camera_frame`], so a render builds each eye's frame once.
pub(crate) fn ray_for_sample_in_frame(
    frame: &CameraFrame,
    camera: &Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<Option<(DVec3, DVec3)>> {
    let ray = ray_for_subpixel_in_frame(
        frame,
        camera.position,
        width,
        height,
        x,
        y,
        sample_x,
        sample_y,
    )?;
    Ok(ray.map(|ray| (ray.origin, ray.direction)))
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

/// The scene world's radiance along a view ray. Realtime Beauty shows it where no surface covers a sample, as EEVEE
/// shows its world, and modes without one show their constant background colour.
pub(crate) type WorldRadiance<'a> = &'a dyn Fn(DVec3) -> Result<DVec3>;

/// The background a sample shows: the world's radiance along its ray when a world is given and the ray exists, else the
/// constant background. The constant's alpha is the film's alpha either way.
fn sample_background(
    constant: [f64; 4],
    world: Option<WorldRadiance<'_>>,
    direction: Option<DVec3>,
) -> Result<[f64; 4]> {
    match (world, direction) {
        (Some(world), Some(direction)) => {
            let radiance = world(direction)?;
            Ok([radiance.x, radiance.y, radiance.z, constant[3]])
        }
        _ => Ok(constant),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "raster configuration is explicit at this low-level boundary"
)]
pub(super) fn rasterize_with_samples(
    triangles: &[Triangle],
    camera: &Camera,
    width: u32,
    height: u32,
    mode: Mode,
    background: [u8; 4],
    world: Option<WorldRadiance<'_>>,
    samples: u32,
    seed: u32,
    filter_size: f64,
    shader: Option<&dyn SurfaceShader>,
    prepared_materials: &[crate::shader::PreparedMaterial],
    camera_lights: Option<&super::path::RealtimeCameraLightContext<'_>>,
    volume_pass: Option<&mut super::path::RealtimeVolumePass<'_, '_>>,
) -> Result<RasterOutput> {
    if samples == 0 {
        return Err(PotError::invalid_argument(
            "sample count must be greater than zero",
        ));
    }
    let mut volume_pass = volume_pass;
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
            world,
            samples,
            seed,
            filter_size,
            shader,
            prepared_materials,
            camera_lights,
            volume_pass.as_deref_mut(),
        )?;
        let right_output = rasterize_with_samples(
            triangles,
            &right_camera,
            width,
            height,
            mode,
            background,
            world,
            samples,
            seed,
            filter_size,
            shader,
            prepared_materials,
            camera_lights,
            volume_pass.as_deref_mut(),
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
    let prepared = prepare_triangles(triangles, mode)?;
    let visibility = super::bvh::Bvh::build_from_positions(
        prepared.iter().map(|triangle| triangle.source.positions),
    )?;
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

    // Realtime Beauty reconstructs its samples with EEVEE's film filter; the other modes keep each pixel's own samples.
    let mut film = film_radius(mode, samples, filter_size)
        .map(|radius| {
            // Side-by-side stereo puts the right eye at `width / 2`, as `ray_for_subpixel_in_frame` does.
            let eye_split =
                (camera.stereo_mode == StereoMode::SideBySide).then_some((width / 2) as usize);
            FilmFilter::new(radius, width, height, eye_split)
        })
        .transpose()?;
    let mut candidates = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let center_ray = ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?;
            let center_hit = match center_ray {
                Some(ray) => nearest_visible_hit(
                    &prepared,
                    &visibility,
                    &mut candidates,
                    ray,
                    &frame,
                    shader,
                )?,
                None => None,
            };
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
            if let Some(visible_hit) = center_hit {
                let triangle = &prepared[visible_hit.triangle_index];
                let hit = visible_hit.hit;
                ids[pixel_index] = triangle.source.object_index;
                depths[pixel_index] = depth_to_f32(hit.depth);
                elements[pixel_index] = triangle.source.element_index;
                normals[pixel_index] = shading_normal(triangle, hit)
                    .to_array()
                    .map(|value| value as f32);
                albedo[pixel_index] = linear_color_to_f32(triangle.source.color);
                if let (Some(bvh), Some(center_ray)) = (&ao_bvh, center_ray) {
                    let geometric = if triangle.normal.dot(hit.direction) > 0.0 {
                        -triangle.normal
                    } else {
                        triangle.normal
                    };
                    let mut shading = shading_normal(triangle, hit);
                    if shading.dot(geometric) < 0.0 {
                        shading = -shading;
                    }
                    // The hit's distance along its ray places the point; its depth is a view-axis or panorama measure.
                    let point = center_ray.origin + center_ray.direction * hit.distance;
                    let visibility = ambient_visibility(
                        bvh,
                        triangles,
                        prepared_materials,
                        point,
                        geometric,
                        shading,
                        1.0,
                    );
                    ambient_occlusion[pixel_index] =
                        linear_color_to_f32([visibility, visibility, visibility, 1.0])[0];
                }
            }

            if matches!(mode, Mode::Solid | Mode::Beauty | Mode::Wire) {
                let mut average = [0.0_f64; 4];
                for sample_index in 0..samples {
                    // Without a film the pixel keeps its own pattern, whose first sample is the centre. A film sample is
                    // jittered over its pixel, the first included, as EEVEE jitters every sample (`pixel_jitter_get`).
                    let (sample_x, sample_y) = match film.as_ref() {
                        Some(film) => film.jitter(film_sample_offset(sample_index)),
                        None => sample_offset(seed, x, y, sample_index),
                    };
                    let (sample_ray, sample_hit) = if sample_index == 0 && film.is_none() {
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
                        let hit = match ray {
                            Some(ray) => nearest_visible_hit(
                                &prepared,
                                &visibility,
                                &mut candidates,
                                ray,
                                &frame,
                                shader,
                            )?,
                            None => None,
                        };
                        (ray, hit)
                    };
                    let sample_color = if matches!(mode, Mode::Wire) {
                        sample_hit.map_or(background_linear, |visible_hit| {
                            let triangle = &prepared[visible_hit.triangle_index];
                            if is_wire_edge(triangle, visible_hit.hit, &frame, width, height) {
                                triangle.source.color
                            } else {
                                background_linear
                            }
                        })
                    } else if let Some(sample_ray) = sample_ray {
                        // The composite walks every surface along the ray, nearest first. A transparent pane that only
                        // tints sits in front of an opaque surface and is not the opacity hit, so the opacity hit alone
                        // cannot decide the colour: an opaque front layer ends the walk inside `composite_layers`.
                        let mut volume_sample = volume_pass
                            .as_deref_mut()
                            .map(|volume_pass| {
                                volume_pass.begin_sample(super::path::RealtimeVolumeSampleInfo {
                                    camera,
                                    frame: &frame,
                                    width,
                                    height,
                                    x,
                                    y,
                                    sample_x,
                                    sample_y,
                                    ray: (sample_ray.origin, sample_ray.direction),
                                    sample_index,
                                })
                            })
                            .transpose()?;
                        composite_sample(
                            &prepared,
                            &visibility,
                            &mut candidates,
                            sample_ray,
                            &frame,
                            camera,
                            background_linear,
                            world,
                            shader,
                            sample_hit,
                            camera_lights,
                            volume_sample.as_mut(),
                        )?
                    } else {
                        sample_background(background_linear, world, None)?
                    };
                    let count = f64::from(sample_index + 1);
                    let previous_weight = 1.0 - 1.0 / count;
                    for channel in 0..4 {
                        average[channel] =
                            average[channel] * previous_weight + sample_color[channel] / count;
                    }
                    if let Some(film) = film.as_mut() {
                        film.splat(x, y, sample_x, sample_y, sample_index, sample_color);
                    }
                }
                let color = color_to_srgba(average);
                rgba[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&color);
                linear_rgba[pixel_index] = linear_color_to_f32(average);
            } else if let Some(visible_hit) = center_hit {
                let triangle = &prepared[visible_hit.triangle_index];
                let hit = visible_hit.hit;
                let (color, linear_color) = match mode {
                    Mode::Normal => {
                        let color = normal_to_rgba(shading_normal(triangle, hit));
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

    if let Some(film) = &film {
        for (pixel_index, (colour, weight)) in film.colour.iter().zip(&film.weight).enumerate() {
            let average = colour.map(|sum| sum / weight.max(f64::MIN_POSITIVE));
            rgba[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&color_to_srgba(average));
            linear_rgba[pixel_index] = linear_color_to_f32(average);
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
pub(super) fn rasterize_with_lines(
    triangles: &[Triangle],
    lines: &[Line],
    camera: &Camera,
    width: u32,
    height: u32,
    mode: Mode,
    background: [u8; 4],
    world: Option<WorldRadiance<'_>>,
    samples: u32,
    seed: u32,
    filter_size: f64,
    shader: Option<&dyn SurfaceShader>,
    prepared_materials: &[crate::shader::PreparedMaterial],
    camera_lights: Option<&super::path::RealtimeCameraLightContext<'_>>,
    volume_pass: Option<&mut super::path::RealtimeVolumePass<'_, '_>>,
) -> Result<RasterOutput> {
    let mut output = rasterize_with_samples(
        triangles,
        camera,
        width,
        height,
        mode,
        background,
        world,
        samples,
        seed,
        filter_size,
        shader,
        prepared_materials,
        camera_lights,
        volume_pass,
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
        if matches!(mode, Mode::Beauty) && line.ray_mask & RayKind::Camera.bit() == 0 {
            continue;
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
                // A pixel with no camera ray (outside a fisheye's image circle, or past its field of view) shows nothing, as
                // Blender leaves it black; a stroke there must not reach the frame's corners.
                if ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?.is_none() {
                    continue;
                }
                let world_position = line.start.lerp(line.end, fraction);
                let depth = frame_depth(&frame, camera.position, world_position);
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
/// The wireframe preview (`render/wireframe.rs`): the object outline of the surfaces' ids, then the edges and
/// points over it. `triangles` and `lines` only feed the pick, depth and element buffers, which keep their
/// usual meaning; the RGBA and linear colour are the wireframe overlay's.
pub(crate) fn rasterize_wireframe(
    edges: &[WireEdge],
    points: &[WirePoint],
    triangles: &[Triangle],
    lines: &[Line],
    camera: &Camera,
    width: u32,
    height: u32,
) -> Result<RasterOutput> {
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image height exceeds platform limits",
        )
    })?;
    // Workbench's object id buffer holds the surfaces alone: loose edges and points are not objects.
    let surfaces = rasterize_with_samples(
        triangles,
        camera,
        width,
        height,
        Mode::Wire,
        [0; 4],
        None,
        1,
        0,
        // Wire is never a film Beauty render, so no film filter applies.
        0.0,
        None,
        &[],
        None,
        None,
    )?;
    let mut output = rasterize_with_lines(
        triangles,
        lines,
        camera,
        width,
        height,
        Mode::Wire,
        wireframe::background_rgba(),
        None,
        1,
        0,
        0.0,
        None,
        &[],
        None,
        None,
    )?;
    let frame = camera_frame(camera, width, height)?;
    let orthographic = matches!(frame.projection, ProjectionFrame::Orthographic { .. });
    let (rim, interior) = wireframe::theme_linear();
    let facing = |normal: DVec3, point: DVec3| {
        // An orthographic view has one direction; a perspective view looks from the eye to the point.
        let toward_viewer = if orthographic {
            -frame.forward
        } else {
            (camera.position - point).normalize_or_zero()
        };
        normal.dot(toward_viewer)
    };
    // Window depth in [0, 1], where the cleared buffer is 1.
    let window_depth = |point: DVec3| {
        ((frame_depth(&frame, camera.position, point) - camera.near) / (camera.far - camera.near))
            .clamp(0.0, 1.0)
    };
    let mut segments = Vec::with_capacity(edges.len());
    for edge in edges {
        if !is_finite_vec3(edge.start)
            || !is_finite_vec3(edge.end)
            || !edge.wd.is_finite()
            || !edge.colour.iter().all(|channel| channel.is_finite())
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "wireframe edge contains invalid geometry or color",
            ));
        }
        if !wireframe::edge_visible(edge.wd, wireframe::WIRE_THRESHOLD) {
            continue;
        }
        // Clip to the near and far planes along the edge first. Depth is linear along a segment for the
        // planar projections; a panorama clips by distance, so its edges keep their endpoints.
        let (start, end, normals) = if frame.depth_is_distance {
            (edge.start, edge.end, edge.normals)
        } else {
            let depth_at = |point: DVec3| (point - camera.position).dot(frame.forward);
            let Some((t0, t1)) = clip_range(
                depth_at(edge.start),
                depth_at(edge.end),
                camera.near,
                camera.far,
            ) else {
                continue;
            };
            let at = |t: f64| edge.start.lerp(edge.end, t);
            let normals = edge.normals.map(|[normal_a, normal_b]| {
                [
                    normal_a.lerp(normal_b, t0).normalize_or_zero(),
                    normal_a.lerp(normal_b, t1).normalize_or_zero(),
                ]
            });
            (at(t0), at(t1), normals)
        };
        let (Some(start_projected), Some(end_projected)) = (
            project_unclipped(camera, &frame, width, height, start),
            project_unclipped(camera, &frame, width, height, end),
        ) else {
            continue;
        };
        let (colour_a, colour_b) = match normals {
            Some([normal_a, normal_b]) => (
                wireframe::facing_colour(facing(normal_a, start), rim, interior),
                wireframe::facing_colour(facing(normal_b, end), rim, interior),
            ),
            None => (edge.colour, edge.colour),
        };
        // `project_world` puts pixel centres on integers; the overlay model takes pixel `i` as `[i, i + 1)`,
        // so its centres sit at `i + 0.5`.
        segments.push(WireSegment {
            a: ScreenPoint {
                x: start_projected.x + 0.5,
                y: start_projected.y + 0.5,
                depth: window_depth(start),
            },
            b: ScreenPoint {
                x: end_projected.x + 0.5,
                y: end_projected.y + 0.5,
                depth: window_depth(end),
            },
            colour_a,
            colour_b,
        });
    }
    let mut projected_points = Vec::with_capacity(points.len());
    for point in points {
        if !is_finite_vec3(point.position) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "wireframe point contains invalid geometry",
            ));
        }
        let Some(projected) = project_world(camera, &frame, width, height, point.position) else {
            continue;
        };
        projected_points.push((*point, projected));
    }
    // A point-cloud point's sphere enters the object id buffer before the outline is drawn, wherever it is not
    // behind a surface (its view depth is at most the surface's).
    let mut ids = surfaces.ids.clone();
    for (point, projected) in &projected_points {
        let Some(sphere) = point.sphere else {
            continue;
        };
        let radius_px = sphere.radius * projected.pixel_scale;
        let depth = frame_depth(&frame, camera.position, point.position);
        let (centre_x, centre_y) = (projected.x + 0.5, projected.y + 0.5);
        let min_x = (centre_x - radius_px).floor().max(0.0) as usize;
        let max_x =
            ((centre_x + radius_px).ceil().max(0.0) as usize).min(width_usize.saturating_sub(1));
        let min_y = (centre_y - radius_px).floor().max(0.0) as usize;
        let max_y =
            ((centre_y + radius_px).ceil().max(0.0) as usize).min(height_usize.saturating_sub(1));
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let inside = (px as f64 + 0.5 - centre_x).abs()
                    + (py as f64 + 0.5 - centre_y).abs()
                    <= radius_px;
                let index = py * width_usize + px;
                if inside && (ids[index] == 0 || depth <= f64::from(surfaces.depths[index])) {
                    ids[index] = sphere.object_index;
                }
            }
        }
    }
    let background = wireframe::outline_background(&ids, width_usize, height_usize);
    let mut squares = Vec::with_capacity(projected_points.len());
    for (_, projected) in &projected_points {
        squares.push(PointSquare {
            centre_x: projected.x + 0.5,
            centre_y: projected.y + 0.5,
            side: 2.0 * wireframe::vertex_size(),
        });
    }
    let linear = wireframe::composite(&segments, &squares, &background, width_usize, height_usize);
    for (index, colour) in linear.iter().enumerate() {
        let encoded = [
            wireframe::display_byte(colour[0]),
            wireframe::display_byte(colour[1]),
            wireframe::display_byte(colour[2]),
            255,
        ];
        output.rgba[index * 4..index * 4 + 4].copy_from_slice(&encoded);
        output.linear_rgba[index] = [colour[0] as f32, colour[1] as f32, colour[2] as f32, 1.0];
    }
    Ok(output)
}

/// The part `[t0, t1]` of a segment whose depth runs linearly from `depth_a` to `depth_b` that lies within
/// `[near, far]`, as fractions of the segment; `None` when no part does.
fn clip_range(depth_a: f64, depth_b: f64, near: f64, far: f64) -> Option<(f64, f64)> {
    let slope = depth_b - depth_a;
    if slope.abs() < 1e-12 * (1.0 + depth_a.abs()) {
        return (depth_a >= near && depth_a <= far).then_some((0.0, 1.0));
    }
    let t_near = (near - depth_a) / slope;
    let t_far = (far - depth_a) / slope;
    let (lo, hi) = (t_near.min(t_far).max(0.0), t_near.max(t_far).min(1.0));
    (lo <= hi).then_some((lo, hi))
}

/// The depth a world point records under `frame`: its distance from the camera for a panorama, and its view-axis distance
/// otherwise. Clipping, the depth buffer, and picking all use this one definition.
fn frame_depth(frame: &CameraFrame, position: DVec3, point: DVec3) -> f64 {
    let relative = point - position;
    if frame.depth_is_distance {
        relative.length()
    } else {
        relative.dot(frame.forward)
    }
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
    let clip_depth = clip_depth_of(camera, frame, point);
    if !clip_depth.is_finite() || clip_depth < frame.near || clip_depth > frame.far {
        return None;
    }
    project_unclipped(camera, frame, width, height, point)
}

/// The depth a point is clipped by: its view-axis depth, or its distance from the camera for a panorama,
/// which covers every direction.
fn clip_depth_of(camera: &Camera, frame: &CameraFrame, point: DVec3) -> f64 {
    let relative = point - camera.position;
    match frame.projection {
        ProjectionFrame::Orthographic { .. } | ProjectionFrame::Perspective { .. } => {
            relative.dot(frame.forward)
        }
        ProjectionFrame::Equirectangular { .. }
        | ProjectionFrame::FisheyeEquidistant { .. }
        | ProjectionFrame::FisheyeEquisolid { .. } => relative.length(),
    }
}

/// `project_world` without the clip-range test, for a point the caller has already clipped to the planes.
fn project_unclipped(
    camera: &Camera,
    frame: &CameraFrame,
    width: u32,
    height: u32,
    point: DVec3,
) -> Option<ProjectedPoint> {
    let relative = point - camera.position;
    let depth = relative.dot(frame.forward);
    let clip_depth = clip_depth_of(camera, frame, point);
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
        ProjectionFrame::Equirectangular {
            longitude,
            latitude,
        } => {
            let lon_span = longitude[1] - longitude[0];
            let lat_span = latitude[1] - latitude[0];
            // The longitude of the point, wrapped to the turn that lies nearest the range's centre.
            let mid = f64::midpoint(longitude[0], longitude[1]);
            let raw = right.atan2(depth);
            let point_longitude = raw + TAU * ((mid - raw) / TAU).round();
            let point_latitude = (up / clip_depth).clamp(-1.0, 1.0).asin();
            (
                2.0 * (point_longitude - longitude[0]) / lon_span - 1.0,
                2.0 * (point_latitude - latitude[0]) / lat_span - 1.0,
                f64::from(width) / (lon_span.abs() * clip_depth),
                f64::from(height) / (lat_span.abs() * clip_depth),
            )
        }
        ProjectionFrame::FisheyeEquidistant { fov } => {
            let angle = (relative / clip_depth)
                .dot(frame.forward)
                .clamp(-1.0, 1.0)
                .acos();
            let radial = DVec3::new(right, up, 0.0).normalize_or_zero();
            let radius = 2.0 * angle / fov;
            (
                radial.x * radius,
                radial.y * radius,
                f64::from(width) / (fov * clip_depth),
                f64::from(height) / (fov * clip_depth),
            )
        }
        ProjectionFrame::FisheyeEquisolid {
            fov: _,
            fisheye_lens,
            sensor_width_mm,
            sensor_height_mm,
        } => {
            let angle = (relative / clip_depth)
                .dot(frame.forward)
                .clamp(-1.0, 1.0)
                .acos();
            let image_radius = 2.0 * fisheye_lens * (0.5 * angle).sin();
            let radial = DVec3::new(right, up, 0.0).normalize_or_zero();
            (
                radial.x * image_radius * 2.0 / sensor_width_mm,
                radial.y * image_radius * 2.0 / sensor_height_mm,
                f64::from(width) * fisheye_lens / (sensor_width_mm * clip_depth),
                f64::from(height) * fisheye_lens / (sensor_height_mm * clip_depth),
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

/// The camera's `forward`, `right`, and `up` directions in world space, from its position, target
/// and up hint.
fn view_axes(camera: &Camera) -> Result<(DVec3, DVec3, DVec3)> {
    let forward = safe_normalize(camera.target - camera.position)
        .ok_or_else(|| PotError::invalid_argument("camera position and target must differ"))?;
    let up_hint = safe_normalize(camera.up)
        .ok_or_else(|| PotError::invalid_argument("camera up vector must be non-zero"))?;
    let right = safe_normalize(forward.cross(up_hint)).ok_or_else(|| {
        PotError::invalid_argument("camera up vector must not be parallel to its view direction")
    })?;
    let up = safe_normalize(right.cross(forward))
        .ok_or_else(|| PotError::invalid_argument("camera orientation is invalid"))?;
    Ok((forward, right, up))
}

pub(crate) fn camera_frame(camera: &Camera, width: u32, height: u32) -> Result<CameraFrame> {
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

    let (forward, right, up) = view_axes(camera)?;

    let projection = match camera.projection {
        Projection::Orthographic {
            height: view_height,
        } if view_height.is_finite() && view_height > 0.0 => {
            // Blender's AUTO sensor fit (intern/camera.cc, `BKE_camera_sensor_fit`: HOR when the
            // render is at least as wide as tall) applies the orthographic scale to the larger
            // dimension. `BKE_camera_params_compute_viewplane` then scales the other dimension by
            // the aspect, so a landscape frame spans `scale` horizontally and a portrait frame
            // spans `scale` vertically.
            let half_scale = view_height * 0.5;
            let (half_width, half_height) = if view_width >= height {
                (
                    half_scale,
                    half_scale * (f64::from(height) / f64::from(view_width)),
                )
            } else {
                (
                    half_scale * (f64::from(view_width) / f64::from(height)),
                    half_scale,
                )
            };
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
        Projection::Equirectangular {
            longitude,
            latitude,
        } if longitude
            .iter()
            .chain(&latitude)
            .all(|value| value.is_finite())
            && longitude[1] > longitude[0]
            && longitude[1] - longitude[0] <= crate::model::BLENDER_TURN
            && latitude[1] > latitude[0]
            && latitude[0] >= -crate::model::BLENDER_RIGHT_ANGLE
            && latitude[1] <= crate::model::BLENDER_RIGHT_ANGLE =>
        {
            ProjectionFrame::Equirectangular {
                longitude,
                latitude,
            }
        }
        Projection::FisheyeEquidistant { fov } if is_valid_fov(fov) => {
            ProjectionFrame::FisheyeEquidistant { fov }
        }
        Projection::FisheyeEquisolid {
            fov,
            fisheye_lens,
            sensor_width_mm,
            sensor_height_mm,
            sensor_fit,
        } if is_valid_fov(fov)
            && [fisheye_lens, sensor_width_mm, sensor_height_mm]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0) =>
        {
            let (sensor_width_mm, sensor_height_mm) = equisolid_sensor(
                sensor_width_mm,
                sensor_height_mm,
                sensor_fit,
                f64::from(view_width),
                f64::from(height),
            );
            ProjectionFrame::FisheyeEquisolid {
                fov,
                fisheye_lens,
                sensor_width_mm,
                sensor_height_mm,
            }
        }
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
        panorama_pixel_angle: camera
            .projection
            .panorama_angle_per_pixel(view_width, height),
        depth_is_distance: camera.projection.measures_ray_distance(),
    })
}

fn is_valid_fov(fov: f64) -> bool {
    fov.is_finite() && fov > 0.0 && fov <= crate::model::FISHEYE_FOV_MAX
}

/// Fraction of the hemisphere above `shading` that is unoccluded at `point`. Shadow-visible surfaces occlude as
/// Cycles' ambient occlusion does, but a Volume-only material has no surface to block the ray.
pub(crate) fn ambient_visibility(
    bvh: &super::bvh::Bvh,
    triangles: &[Triangle],
    prepared_materials: &[crate::shader::PreparedMaterial],
    point: DVec3,
    geometric: DVec3,
    shading: DVec3,
    distance: f64,
) -> f64 {
    const SAMPLE_COUNT: u32 = 12;
    let normal = shading.normalize_or_zero();
    if normal == DVec3::ZERO || !distance.is_finite() || distance <= 0.0 {
        return 1.0;
    }
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent).normalize_or_zero();
    let geometric = geometric.normalize_or_zero();
    let origin = point + geometric * 1.0e-5;
    let mut visible = 0_u32;
    for index in 0..SAMPLE_COUNT {
        let fraction = (f64::from(index) + 0.5) / f64::from(SAMPLE_COUNT);
        let radius = (1.0 - fraction * fraction).sqrt();
        let angle = f64::from(index) * 2.399_963_229_728_653;
        let direction = (normal * fraction
            + tangent * (radius * angle.cos())
            + bitangent * (radius * angle.sin()))
        .normalize_or_zero();
        // Smooth shading can aim below the geometric surface; mirror such rays back above it.
        let below = direction.dot(geometric).min(0.0);
        let direction = if below < 0.0 {
            (direction - geometric * (2.0 * below)).normalize_or_zero()
        } else {
            direction
        };
        let mut occluded = false;
        bvh.walk_hits(
            triangles,
            super::bvh::WalkRay {
                origin,
                direction,
                epsilon: 1.0e-5,
                remaining: distance,
                kind: Some(RayKind::Shadow),
            },
            |hit, _| {
                let triangle = &triangles[hit.triangle_index];
                if prepared_materials
                    .get(triangle.material_index)
                    .is_some_and(crate::shader::PreparedMaterial::is_volume_only)
                {
                    true
                } else {
                    occluded = true;
                    false
                }
            },
        );
        if !occluded {
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
) -> Result<Option<Ray>> {
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
) -> Result<Option<Ray>> {
    if !pixel_in_bounds(width, height, x, y) || !sample_x.is_finite() || !sample_y.is_finite() {
        return Err(PotError::invalid_argument(
            "pixel sample offsets must be finite inside a non-empty image",
        ));
    }
    // Offsets are from the pixel's corner and may leave the pixel: a Cycles pixel filter with
    // support wider than one pixel places samples in the neighbours' area (`pixel_filter.rs`).

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
    let Some(mut ray) = (match frame.projection {
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => Some(Ray {
            origin: origin + frame.right * (ndc_x * half_width) + frame.up * (ndc_y * half_height),
            direction: frame.forward,
        }),
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
            Some(Ray { origin, direction })
        }
        ProjectionFrame::Equirectangular {
            longitude,
            latitude,
        } => {
            // Cycles' `equirectangular_range_to_direction`: u and v run from the range's minimum to its maximum.
            let u = f64::midpoint(ndc_x, 1.0);
            let v = f64::midpoint(ndc_y, 1.0);
            let point_longitude = longitude[0] + (longitude[1] - longitude[0]) * u;
            let point_latitude = latitude[0] + (latitude[1] - latitude[0]) * v;
            let direction = safe_normalize(
                frame.forward * (point_longitude.cos() * point_latitude.cos())
                    + frame.right * (point_longitude.sin() * point_latitude.cos())
                    + frame.up * point_latitude.sin(),
            )
            .ok_or_else(|| PotError::invalid_argument("panorama ray is out of range"))?;
            Some(Ray { origin, direction })
        }
        ProjectionFrame::FisheyeEquidistant { fov } => {
            let radius = ndc_x.hypot(ndc_y);
            (radius <= 1.0).then(|| {
                let angle = 0.5 * fov * radius;
                let direction = if radius <= f64::EPSILON {
                    frame.forward
                } else {
                    let radial = (frame.right * ndc_x + frame.up * ndc_y) / radius;
                    frame.forward * angle.cos() + radial * angle.sin()
                };
                Ray { origin, direction }
            })
        }
        ProjectionFrame::FisheyeEquisolid {
            fov,
            fisheye_lens,
            sensor_width_mm,
            sensor_height_mm,
        } => {
            let sensor_x = 0.5 * ndc_x * sensor_width_mm;
            let sensor_y = 0.5 * ndc_y * sensor_height_mm;
            let radius = sensor_x.hypot(sensor_y);
            // Blender's equisolid image circle ends at the field of view: 2·f·sin(fov/4), the radius at fov/2 (cycles'
            // `fisheye_equisolid_to_direction`). Beyond it a pixel has no ray.
            (radius <= 2.0 * fisheye_lens * (0.25 * fov).sin()).then(|| {
                let angle = 2.0 * (radius / (2.0 * fisheye_lens)).clamp(-1.0, 1.0).asin();
                let direction = if radius <= f64::EPSILON {
                    frame.forward
                } else {
                    let radial = (frame.right * sensor_x + frame.up * sensor_y) / radius;
                    frame.forward * angle.cos() + radial * angle.sin()
                };
                Ray { origin, direction }
            })
        }
    }) else {
        return Ok(None);
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
    Ok(Some(ray))
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

/// EEVEE's film reconstruction for realtime Beauty (`eevee_film.cc`): every shaded sample is splatted into the output
/// pixels whose centres lie within `radius` of it, weighted by `film_filter_weight`, and an output pixel is the weighted
/// mean of the samples that reached it. `colour` and `weight` are per output pixel.
struct FilmFilter {
    radius: f64,
    width: usize,
    height: usize,
    /// The column where the right eye starts, in a side-by-side stereo frame. A sample splats only into its own eye's
    /// columns, as each eye has its own film in EEVEE (`ShadingView::update_view` per view).
    eye_split: Option<usize>,
    colour: Vec<[f64; 4]>,
    weight: Vec<f64>,
}

impl FilmFilter {
    /// The sub-pixel position of a film sample from its offset in the pixel. EEVEE's `pixel_jitter_get` gives the
    /// offset `j` of its film weights: below `FRAC_1_SQRT_2` px a disk of half the radius, mapped by
    /// `Sampling::sample_disk` as `sqrt(u)` and `2*pi*v` (`eevee_film.cc`, `eevee_sampling.cc`), otherwise uniform over
    /// the pixel. The camera's window shifts by `+j` in NDC (`ShadingView::update_view`, `eevee_view.cc`;
    /// `window_translate_m4`, `math_geom.cc`), and content moves by that shift. So a pixel sees the scene at `-j` in NDC
    /// x. NDC y points up but rows here point down (`ndc_y = 1 - 2 (y + sy) / h`), so the same shift is `+j` in row
    /// units: the raster position is `(0.5 - j_x, 0.5 + j_y)`.
    fn jitter(&self, offset: (f64, f64)) -> (f64, f64) {
        if self.radius >= std::f64::consts::FRAC_1_SQRT_2 {
            return ((1.0 - offset.0).fract(), offset.1);
        }
        let scale = 0.5 * self.radius * offset.0.sqrt();
        let theta = std::f64::consts::TAU * offset.1;
        (0.5 - scale * theta.cos(), 0.5 + scale * theta.sin())
    }

    fn new(radius: f64, width: u32, height: u32, eye_split: Option<usize>) -> Result<Self> {
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
        let pixel_count = width.checked_mul(height).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "pixel count exceeds platform limits",
            )
        })?;
        Ok(Self {
            radius,
            width,
            height,
            eye_split,
            colour: allocate_filled(pixel_count, [0.0_f64; 4], "film colour")?,
            weight: allocate_filled(pixel_count, 0.0_f64, "film weight")?,
        })
    }

    /// The output columns a sample in column `x` may reach: its own eye's columns in a side-by-side frame.
    fn eye_columns(&self, x: u32) -> (i64, i64) {
        let width = self.width as i64;
        match self.eye_split {
            Some(split) => {
                let split = split as i64;
                if i64::from(x) < split {
                    (0, split)
                } else {
                    (split, width)
                }
            }
            None => (0, width),
        }
    }

    fn accumulate(&mut self, px: usize, py: usize, weight: f64, shaded: [f64; 4]) {
        let index = py * self.width + px;
        for (sum, value) in self.colour[index].iter_mut().zip(shaded) {
            *sum += weight * value;
        }
        self.weight[index] += weight;
    }

    /// Splats the shaded sample of frame `sample_index`, taken at `(sample_x, sample_y)` inside pixel `(x, y)`. Below
    /// `LARGE_FILTER_RADIUS` it reaches the output pixels whose centres lie within the radius. From there on it reaches
    /// EEVEE's 16 table taps (`large_filter_taps`), each weighted from its distance to the sample, with no radius test.
    /// Either way the output stays in the sample's own eye's columns.
    fn splat(
        &mut self,
        x: u32,
        y: u32,
        sample_x: f64,
        sample_y: f64,
        sample_index: u32,
        shaded: [f64; 4],
    ) {
        let sample = (f64::from(x) + sample_x, f64::from(y) + sample_y);
        let (lo, hi) = self.eye_columns(x);
        let height = self.height as i64;
        if self.radius < LARGE_FILTER_RADIUS {
            let reach = (self.radius + 0.5).ceil() as i64;
            let radius_squared = self.radius * self.radius;
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let px = i64::from(x) + dx;
                    let py = i64::from(y) + dy;
                    if px < lo || px >= hi || py < 0 || py >= height {
                        continue;
                    }
                    let centre = (px as f64 + 0.5, py as f64 + 0.5);
                    let distance_squared =
                        (centre.0 - sample.0).powi(2) + (centre.1 - sample.1).powi(2);
                    if distance_squared >= radius_squared {
                        continue;
                    }
                    let weight = film_filter_weight(self.radius, distance_squared);
                    self.accumulate(px as usize, py as usize, weight, shaded);
                }
            }
        } else {
            for (tap_x, tap_y) in large_filter_taps(self.radius, sample_index) {
                // EEVEE's table is in texel space with y up, and rows here point down.
                let px = i64::from(x) - tap_x;
                let py = i64::from(y) + tap_y;
                if px < lo || px >= hi || py < 0 || py >= height {
                    continue;
                }
                let centre = (px as f64 + 0.5, py as f64 + 0.5);
                let distance_squared =
                    (centre.0 - sample.0).powi(2) + (centre.1 - sample.1).powi(2);
                let weight = film_filter_weight(self.radius, distance_squared);
                self.accumulate(px as usize, py as usize, weight, shaded);
            }
        }
    }
}

/// The film radius of realtime Beauty, or `None` for the one-sample render, where EEVEE's filter is zero and the sample
/// sits at the pixel centre (`update_sample_table`, `sample_count() == 1`). Otherwise EEVEE clamps `scene.r.gauss` to
/// 0..=100 (`eevee_film.cc`, `filter_radius`). The floor keeps the weights finite; below 0.01 px the samples stay within
/// a disk of 0.5 px radius, so the filter is a point sample at the centre, as EEVEE renders it.
fn film_radius(mode: Mode, samples: u32, filter_size: f64) -> Option<f64> {
    if !matches!(mode, Mode::Beauty) || samples <= 1 {
        return None;
    }
    Some(filter_size.clamp(0.0, 100.0).max(1.0e-4))
}

/// The filter radius from which EEVEE switches to its 16-tap table (`Film::update_sample_table`, `filter_radius < 2.20f`).
const LARGE_FILTER_RADIUS: f64 = 2.2;

/// EEVEE's large-filter table for frame `sample_index`: 16 integer texel offsets. For tap `i` the offset is the floor of
/// `sample_spiral` at `((u + i) / 16, v)`, times the radius. The pair `(u, v)` is the SSS Halton pair at index `5 * k + 1`
/// (`Sampling::step`, `SAMPLING_SSS_U` and `SAMPLING_SSS_V`, leap 5, primes 2 and 3). The offsets are in EEVEE's texel
/// space, whose y points up.
fn large_filter_taps(radius: f64, sample_index: u32) -> [(i64, i64); 16] {
    let index = 5 * u64::from(sample_index) + 1;
    let u = halton_radical_inverse(2, index);
    let v = halton_radical_inverse(3, index);
    std::array::from_fn(|i| {
        let (x, y) = sample_spiral((u + i as f64) / 16.0, v);
        ((x * radius).floor() as i64, (y * radius).floor() as i64)
    })
}

/// `Sampling::sample_spiral` (`eevee_sampling.cc`): a Fibonacci spiral point for `rand_x`, rotated by `rand_y`.
fn sample_spiral(rand_x: f64, rand_y: f64) -> (f64, f64) {
    let omega = 4.0 * std::f64::consts::PI * (1.0 + 5.0_f64.sqrt()) * rand_x
        + rand_y * std::f64::consts::TAU;
    let radius = rand_x.sqrt();
    (radius * omega.cos(), radius * omega.sin())
}

/// A film sample's offset inside its pixel: EEVEE's Halton (2, 3) sequence at index `sample_index + 1`, each dimension
/// rotated by a fixed offset, `fract(halton + 1/2)` and `fract(halton + 2/3)` (`eevee_sampling.cc`, `Sampling::step`,
/// with `BLI_halton_2d` from `blenlib/intern/rand.cc`). The pattern is the same for every pixel and needs no seed.
fn film_sample_offset(sample_index: u32) -> (f64, f64) {
    let index = u64::from(sample_index) + 1;
    let x_offset = (halton_radical_inverse(2, index) + 0.5).fract();
    let y_offset = (halton_radical_inverse(3, index) + 2.0 / 3.0).fract();
    (x_offset, y_offset)
}

/// The radical inverse of `index` in `base`: its digits reversed about the radix point.
fn halton_radical_inverse(base: u64, mut index: u64) -> f64 {
    let inverse_base = 1.0 / base as f64;
    let mut scale = 1.0;
    let mut value = 0.0;
    while index > 0 {
        scale *= inverse_base;
        value += scale * (index % base) as f64;
        index /= base;
    }
    value
}

/// EEVEE's `film_filter_weight`: a Gaussian fitted to Blackman-Harris, with a standard deviation of 0.284 radii
/// (`eevee_film_shared.hh`).
fn film_filter_weight(radius: f64, distance_squared: f64) -> f64 {
    const SIGMA: f64 = 0.284;
    let r = distance_squared / (radius * radius);
    (-0.5 / (SIGMA * SIGMA) * r).exp()
}

fn mix_sample_hash(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn hit_opacity(triangle: &Triangle, color: [f64; 4]) -> f64 {
    match triangle.alpha_mode {
        AlphaMode::Opaque => 1.0,
        AlphaMode::Blend => color[3].clamp(0.0, 1.0),
        AlphaMode::Clip => {
            if color[3] >= triangle.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Shading normal at a hit: the blended corner normal for smooth triangles, else the geometric one.
fn shading_normal(triangle: &PreparedTriangle, hit: Hit) -> DVec3 {
    triangle.source.normals.map_or(triangle.normal, |corners| {
        blend_corner_normals(corners, hit.barycentric, triangle.normal)
    })
}

/// The surface sample at a visible hit: the surface shader's when one is present, otherwise the triangle's colour.
fn hit_color(
    triangle: &PreparedTriangle,
    hit: Hit,
    ray: Ray,
    shader: Option<&dyn SurfaceShader>,
) -> Result<SurfaceSample> {
    match shader {
        Some(shader) => {
            let point = ray.origin + ray.direction * hit.distance;
            let surface = SurfaceHit {
                point,
                shadow_point: smooth_shadow_origin(&triangle.source, point, hit.barycentric),
                normal: shading_normal(triangle, hit),
                geometric: triangle.normal,
                barycentric: hit.barycentric,
                view: -ray.direction,
            };
            shader.shade(&triangle.source, &surface)
        }
        _ => Ok(SurfaceSample::opaque(triangle.source.color)),
    }
}

fn nearest_visible_hit(
    triangles: &[PreparedTriangle],
    visibility: &super::bvh::Bvh,
    candidates: &mut Vec<usize>,
    ray: Ray,
    frame: &CameraFrame,
    shader: Option<&dyn SurfaceShader>,
) -> Result<Option<VisibleHit>> {
    let mut cursor = None;
    // The ray is fixed, so its candidates are too. The nearest is the least (depth, index) pair
    // among them, so their order does not matter.
    visibility.overlapping(ray.origin, ray.direction, candidates);
    loop {
        let mut nearest: Option<(usize, Hit)> = None;
        for &triangle_index in candidates.iter() {
            let Some(hit) = intersect_triangle(&triangles[triangle_index], ray, frame) else {
                continue;
            };
            if cursor.is_some_and(|(depth, index)| {
                hit.depth < depth || (equal_f64(hit.depth, depth) && triangle_index <= index)
            }) {
                continue;
            }
            if nearest.is_none_or(|(closest_index, closest_hit)| {
                hit.depth < closest_hit.depth
                    || (equal_f64(hit.depth, closest_hit.depth) && triangle_index < closest_index)
            }) {
                nearest = Some((triangle_index, hit));
            }
        }
        let Some((triangle_index, hit)) = nearest else {
            return Ok(None);
        };
        let triangle = &triangles[triangle_index];
        let sample = hit_color(triangle, hit, ray, shader)?;
        // The nearest surface with opacity is the pixel's data-pass hit (ids, depth, normals, albedo, AO). A transparent
        // pane that only tints is not, so it cannot change those passes; the Beauty composite finds it separately.
        if sample.has_surface && hit_opacity(&triangle.source, sample.color) > 0.0 {
            return Ok(Some(VisibleHit {
                triangle_index,
                hit,
                color: sample.color,
                transparent_tint: sample.transparent_tint,
            }));
        }
        cursor = Some((hit.depth, triangle_index));
    }
}

/// Whether a transparent tint passes every channel unchanged.
fn is_white(tint: [f64; 3]) -> bool {
    tint.iter().all(|&channel| (channel - 1.0).abs() <= 1.0e-12)
}

fn composite_sample(
    triangles: &[PreparedTriangle],
    visibility: &super::bvh::Bvh,
    candidates: &mut Vec<usize>,
    ray: Ray,
    frame: &CameraFrame,
    camera: &Camera,
    background: [f64; 4],
    world: Option<WorldRadiance<'_>>,
    shader: Option<&dyn SurfaceShader>,
    cached_hit: Option<VisibleHit>,
    camera_lights: Option<&super::path::RealtimeCameraLightContext<'_>>,
    mut volume_sample: Option<&mut super::path::RealtimeVolumeSample<'_, '_>>,
) -> Result<[f64; 4]> {
    const MAX_ALPHA_LAYERS: usize = 16;
    let mut hits: [Option<VisibleHit>; MAX_ALPHA_LAYERS] = [None; MAX_ALPHA_LAYERS];
    let mut hit_count = 0;
    visibility.overlapping(ray.origin, ray.direction, candidates);
    // Equal depths keep the first layer inserted, so visit the candidates in triangle order.
    candidates.sort_unstable();
    for &triangle_index in candidates.iter() {
        let triangle = &triangles[triangle_index];
        let Some(hit) = intersect_triangle(triangle, ray, frame) else {
            continue;
        };
        let sample = cached_hit
            .filter(|cached| {
                cached.triangle_index == triangle_index && equal_f64(cached.hit.depth, hit.depth)
            })
            .map_or_else(
                || hit_color(triangle, hit, ray, shader),
                |cached| {
                    Ok(SurfaceSample {
                        color: cached.color,
                        has_surface: true,
                        transparent_tint: cached.transparent_tint,
                    })
                },
            )?;
        if !sample.has_surface
            || (hit_opacity(&triangle.source, sample.color) <= 0.0
                && is_white(sample.transparent_tint))
        {
            continue;
        }
        let visible_hit = VisibleHit {
            triangle_index,
            hit,
            color: sample.color,
            transparent_tint: sample.transparent_tint,
        };
        let mut insert_at = hit_count;
        while insert_at > 0
            && hits[insert_at - 1].is_some_and(|previous| previous.hit.depth > hit.depth)
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
        hits[insert_at] = Some(visible_hit);
        hit_count = next_count;
    }
    let mut color = if let Some(volume_sample) = volume_sample.as_deref_mut() {
        let mut transmittance = DVec3::ONE;
        let mut alpha_transmittance = 1.0;
        let mut premultiplied = DVec3::ZERO;
        let mut start_distance = volume_sample.near_distance;
        for visible_hit in hits.iter().take(hit_count).flatten() {
            let (radiance, segment_transmittance) =
                volume_sample.integrate_segment(start_distance, visible_hit.hit.distance)?;
            premultiplied += transmittance * radiance;
            transmittance *= segment_transmittance;
            alpha_transmittance *= segment_transmittance.element_sum() / 3.0;
            let triangle = &triangles[visible_hit.triangle_index];
            let color = visible_hit.color;
            let opacity = hit_opacity(&triangle.source, color);
            premultiplied += DVec3::new(color[0], color[1], color[2]) * (transmittance * opacity);
            let passed = 1.0 - opacity;
            transmittance *= DVec3::from_array(visible_hit.transparent_tint) * passed;
            alpha_transmittance *= visible_hit.transparent_tint.iter().sum::<f64>() / 3.0 * passed;
            start_distance = visible_hit.hit.distance;
            if transmittance.max_element() <= 1.0e-6 && alpha_transmittance <= 1.0e-6 {
                break;
            }
        }
        if transmittance.max_element() > 1.0e-6 || alpha_transmittance > 1.0e-6 {
            let (radiance, segment_transmittance) =
                volume_sample.integrate_segment(start_distance, volume_sample.far_distance)?;
            premultiplied += transmittance * radiance;
            transmittance *= segment_transmittance;
            alpha_transmittance *= segment_transmittance.element_sum() / 3.0;
        }
        let background_alpha = background[3].clamp(0.0, 1.0);
        let background_rgb = if transmittance == DVec3::ZERO {
            [0.0; 3]
        } else {
            let color = sample_background(background, world, Some(ray.direction))?;
            [color[0], color[1], color[2]]
        };
        premultiplied += DVec3::from_array(background_rgb) * (transmittance * background_alpha);
        let alpha = 1.0 - alpha_transmittance * (1.0 - background_alpha);
        let color = if alpha <= f64::MIN_POSITIVE {
            premultiplied + DVec3::from_array(background_rgb)
        } else {
            premultiplied / alpha
        };
        [color.x, color.y, color.z, alpha]
    } else {
        let layers = hits.iter().take(hit_count).flatten().map(|visible_hit| {
            let triangle = &triangles[visible_hit.triangle_index];
            let color = visible_hit.color;
            Layer {
                rgb: [color[0], color[1], color[2]],
                opacity: hit_opacity(&triangle.source, color),
                tint: visible_hit.transparent_tint,
            }
        });
        composite_layers_with_background(layers, background[3], || {
            let background = sample_background(background, world, Some(ray.direction))?;
            Ok([background[0], background[1], background[2]])
        })?
    };

    if let Some(camera_lights) = camera_lights {
        let near_distance = camera.ray_distance_for_depth(ray.direction, camera.near);
        let far_distance = camera.ray_distance_for_depth(ray.direction, camera.far);
        let layers = &hits[..hit_count];
        let mut light_radiance = DVec3::ZERO;
        for event in camera_lights.local_events(camera, ray.origin, ray.direction, far_distance) {
            light_radiance += camera_light_event_radiance(
                triangles,
                layers,
                event.distance,
                event.radiance,
                near_distance,
                &mut volume_sample,
            )?;
        }
        light_radiance += camera_light_event_radiance(
            triangles,
            layers,
            far_distance,
            camera_lights.sun_radiance(ray.direction),
            near_distance,
            &mut volume_sample,
        )?;
        color[0] += light_radiance.x;
        color[1] += light_radiance.y;
        color[2] += light_radiance.z;
    }
    Ok(color)
}

fn camera_light_event_radiance(
    triangles: &[PreparedTriangle],
    layers: &[Option<VisibleHit>],
    distance: f64,
    radiance: DVec3,
    near_distance: f64,
    volume_sample: &mut Option<&mut super::path::RealtimeVolumeSample<'_, '_>>,
) -> Result<DVec3> {
    if distance < near_distance || radiance.max_element() <= 0.0 {
        return Ok(DVec3::ZERO);
    }
    let mut surface_transmittance = DVec3::ONE;
    for visible_hit in layers.iter().flatten() {
        if visible_hit.hit.distance >= distance {
            break;
        }
        let triangle = &triangles[visible_hit.triangle_index];
        let passed = 1.0 - hit_opacity(&triangle.source, visible_hit.color);
        surface_transmittance *= DVec3::from_array(visible_hit.transparent_tint) * passed;
        if surface_transmittance == DVec3::ZERO {
            break;
        }
    }
    if surface_transmittance == DVec3::ZERO {
        return Ok(DVec3::ZERO);
    }
    let volume_transmittance = match volume_sample {
        Some(volume_sample) => volume_sample.transmittance_to(distance)?,
        None => DVec3::ONE,
    };
    Ok(radiance * surface_transmittance * volume_transmittance)
}

/// One surface of a pixel's stack, for `composite_layers`.
#[derive(Clone, Copy)]
struct Layer {
    rgb: [f64; 3],
    /// The share of the light reaching this surface that it adds.
    opacity: f64,
    /// The colour of the share it passes: `(1 - opacity) * tint` is its transmittance per channel.
    tint: [f64; 3],
}

/// Composites `layers` (nearest first) over `background`, as a pixel's film does. A layer adds its colour weighted by
/// the transmittance reaching it and its opacity, then passes `(1 - opacity) * tint` of the light, so a Transparent
/// BSDF's tint reaches what lies behind it channel by channel (`path.rs`, `throughput *= tint`).
///
/// The film's alpha is EEVEE's separate alpha channel, not the colour's transmittance: each layer multiplies one scalar
/// by the mean of its passed transmittance, `mean(tint) * (1 - opacity)` (`eevee_surf_forward.bsl.hh`, `combined_a` is
/// `average(transmittance)` per surface). Overlapping coloured layers therefore multiply their means, not the mean of
/// their products. The film alpha is `1 - alpha_transmittance * (1 - background alpha)`. With white tints this is the
/// scalar composite of the layers. Returns the un-premultiplied colour and that alpha.
fn composite_layers_with_background(
    layers: impl IntoIterator<Item = Layer>,
    background_alpha: f64,
    background_rgb: impl FnOnce() -> Result<[f64; 3]>,
) -> Result<[f64; 4]> {
    let mut transmittance = DVec3::ONE;
    let mut alpha_transmittance = 1.0;
    let mut premultiplied = DVec3::ZERO;
    for layer in layers {
        premultiplied += DVec3::from_array(layer.rgb) * (transmittance * layer.opacity);
        let passed = 1.0 - layer.opacity;
        transmittance *= DVec3::from_array(layer.tint) * passed;
        alpha_transmittance *= layer.tint.iter().sum::<f64>() / 3.0 * passed;
        // A layer that passes nothing adds nothing to the colour, but the alpha still counts later layers.
        if transmittance.max_element() <= 1.0e-6 && alpha_transmittance <= 1.0e-6 {
            break;
        }
    }
    let background_alpha = background_alpha.clamp(0.0, 1.0);
    // Opaque surfaces block the world, so avoid evaluating its graph for these samples.
    let background_rgb = if transmittance == DVec3::ZERO {
        [0.0; 3]
    } else {
        background_rgb()?
    };
    premultiplied += DVec3::from_array(background_rgb) * (transmittance * background_alpha);
    let alpha = 1.0 - alpha_transmittance * (1.0 - background_alpha);
    if alpha <= f64::MIN_POSITIVE {
        return Ok([
            background_rgb[0],
            background_rgb[1],
            background_rgb[2],
            alpha,
        ]);
    }
    let color = premultiplied / alpha;
    Ok([color.x, color.y, color.z, alpha])
}

/// Primary-ray triangles. Realtime Beauty leaves out objects hidden from camera rays; the other
/// modes draw every triangle, as the viewport does.
fn prepare_triangles(triangles: &[Triangle], mode: Mode) -> Result<Vec<PreparedTriangle>> {
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
        if matches!(mode, Mode::Beauty) && triangle.ray_mask & RayKind::Camera.bit() == 0 {
            continue;
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
            source: triangle.shading_copy(),
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

    // A panorama's depth is the hit's distance: a view-axis depth is negative behind the camera, where it still has pixels.
    let depth = if frame.depth_is_distance {
        distance
    } else {
        distance * ray.direction.dot(frame.forward)
    };
    if !depth.is_finite() || depth < frame.near || depth > frame.far {
        return None;
    }
    Some(Hit {
        depth,
        distance,
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
        _ => match frame.panorama_pixel_angle {
            Some(angle) => angle * hit.depth,
            None => return false,
        },
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

pub(super) fn color_to_srgba(color: [f64; 4]) -> [u8; 4] {
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
    use super::{
        Camera, CameraEngine, Line, Mode, Projection, SensorFit, StereoMode, camera_frame,
        pixel_in_bounds, projection_for_camera, rasterize_with_lines, ray_for_pixel,
    };
    use crate::model::{CameraData, CameraProjection};
    use glam::DVec3;
    use serde_json::Value;
    use std::error::Error;

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
        let (Ok(Some((_, baseline))), Ok(frame)) = (baseline, frame) else {
            return;
        };
        camera.shift = [0.25, -0.25];
        let shifted = ray_for_pixel(&camera, width, height, 4, 4);
        assert!(shifted.is_ok());
        let Ok(Some((_, shifted))) = shifted else {
            return;
        };
        let delta = shifted - baseline;
        assert!(delta.dot(frame.right) > 0.0);
        assert!(delta.dot(frame.up) < 0.0);
    }

    fn panorama_projections() -> [Projection; 3] {
        [
            Projection::Equirectangular {
                longitude: [-1.2, 2.0],
                latitude: [-0.7, 1.1],
            },
            Projection::FisheyeEquidistant { fov: 2.0 },
            Projection::FisheyeEquisolid {
                fov: 2.5,
                fisheye_lens: 12.0,
                sensor_width_mm: 20.0,
                sensor_height_mm: 15.0,
                sensor_fit: SensorFit::Auto,
            },
        ]
    }

    #[test]
    fn pixel_rays_round_trip_to_their_pixel_centers() {
        let width = 7;
        let height = 5;
        let mut projections = vec![
            Projection::Orthographic { height: 4.0 },
            Projection::Perspective {
                lens_mm: 50.0,
                sensor_width_mm: 36.0,
            },
        ];
        projections.extend(panorama_projections());
        for projection in projections {
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
            let Ok(Some((origin, direction))) = ray else {
                panic!("pixel {x},{y} has no ray for {projection:?}");
            };
            let point = origin + direction * 5.0;
            let relative = point - camera.position;
            let camera_x = relative.dot(frame.right);
            let camera_y = relative.dot(frame.up);
            let camera_depth = relative.dot(frame.forward);
            let radius_to_point = relative.length();
            let (ndc_x, ndc_y) = match frame.projection {
                super::ProjectionFrame::Orthographic {
                    half_width,
                    half_height,
                } => (camera_x / half_width, camera_y / half_height),
                super::ProjectionFrame::Perspective {
                    horizontal_scale,
                    vertical_scale,
                } => (
                    camera_x / camera_depth / horizontal_scale,
                    camera_y / camera_depth / vertical_scale,
                ),
                super::ProjectionFrame::Equirectangular {
                    longitude,
                    latitude,
                } => {
                    let point_longitude = camera_x.atan2(camera_depth);
                    let point_latitude = (camera_y / radius_to_point).asin();
                    (
                        2.0 * (point_longitude - longitude[0]) / (longitude[1] - longitude[0])
                            - 1.0,
                        2.0 * (point_latitude - latitude[0]) / (latitude[1] - latitude[0]) - 1.0,
                    )
                }
                super::ProjectionFrame::FisheyeEquidistant { fov } => {
                    let angle = (camera_depth / radius_to_point).clamp(-1.0, 1.0).acos();
                    let radial = glam::DVec2::new(camera_x, camera_y).normalize_or_zero();
                    (radial.x * 2.0 * angle / fov, radial.y * 2.0 * angle / fov)
                }
                super::ProjectionFrame::FisheyeEquisolid {
                    fisheye_lens,
                    sensor_width_mm,
                    sensor_height_mm,
                    ..
                } => {
                    let angle = (camera_depth / radius_to_point).clamp(-1.0, 1.0).acos();
                    let image_radius = 2.0 * fisheye_lens * (0.5 * angle).sin();
                    let radial = glam::DVec2::new(camera_x, camera_y).normalize_or_zero();
                    (
                        radial.x * image_radius * 2.0 / sensor_width_mm,
                        radial.y * image_radius * 2.0 / sensor_height_mm,
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

    fn number(value: &Value) -> Result<f64, Box<dyn Error>> {
        value
            .as_f64()
            .ok_or_else(|| "oracle value is not a number".into())
    }

    fn pair(value: &Value) -> Result<[f64; 2], Box<dyn Error>> {
        Ok([number(&value[0])?, number(&value[1])?])
    }

    /// The camera of a Blender `camera_to_world` matrix: its position, its -Z forward, and its +Y up.
    fn oracle_camera(matrix: &Value, projection: Projection) -> Result<Camera, Box<dyn Error>> {
        let mut rows = [[0.0_f64; 4]; 3];
        for (index, row) in rows.iter_mut().enumerate() {
            let entries = matrix[index]
                .as_array()
                .ok_or("camera matrix row is missing")?;
            for (slot, entry) in row.iter_mut().zip(entries) {
                *slot = number(entry)?;
            }
        }
        let column = |index: usize| DVec3::new(rows[0][index], rows[1][index], rows[2][index]);
        let position = DVec3::new(rows[0][3], rows[1][3], rows[2][3]);
        let forward = -column(2);
        Ok(Camera {
            position,
            target: position + forward,
            up: column(1),
            near: 0.1,
            far: 100.0,
            projection,
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        })
    }

    /// The ray through `(x, y)` must match Blender's direction, or both must have no ray. Blender samples each pixel
    /// centre to within about 0.0005 px, which moves a direction by up to 1e-4 rad, so the tolerance is 2e-4 rad
    /// (about a thousandth of a pixel). Each oracle is a Blender render of the same camera (see its generator).
    fn assert_ray_matches(
        camera: &Camera,
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        expected: Option<[f64; 3]>,
    ) -> Result<(), Box<dyn Error>> {
        const TOLERANCE: f64 = 2.0e-4;
        let ray = ray_for_pixel(camera, width, height, x, y)?;
        match (ray, expected) {
            (Some((_, direction)), Some(expected)) => {
                let error = (direction - DVec3::from_array(expected))
                    .abs()
                    .max_element();
                assert!(
                    error <= TOLERANCE,
                    "pixel ({x}, {y}) ray {direction:?} differs from Blender's {expected:?} by {error}"
                );
            }
            (None, None) => {}
            (ray, expected) => {
                panic!("pixel ({x}, {y}) ray {ray:?} does not match Blender's {expected:?}")
            }
        }
        Ok(())
    }

    #[test]
    fn equirectangular_range_rays_match_blenders_render() -> Result<(), Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/panorama_camera/blender_panorama_camera_range.json"
        ))?;
        let width = u32::try_from(oracle["width"].as_u64().ok_or("oracle width is missing")?)?;
        let height = u32::try_from(
            oracle["height"]
                .as_u64()
                .ok_or("oracle height is missing")?,
        )?;
        let longitude = pair(&oracle["longitude"])?;
        let latitude = pair(&oracle["latitude"])?;
        let data = CameraData {
            projection: CameraProjection::Panorama,
            panorama_type: "equirectangular".to_owned(),
            longitude_min: longitude[0],
            longitude_max: longitude[1],
            latitude_min: latitude[0],
            latitude_max: latitude[1],
            ..CameraData::default()
        };
        let projection = projection_for_camera(&data, CameraEngine::Path)?;
        let camera = oracle_camera(&oracle["camera_to_world"], projection)?;
        let records = oracle["records"]
            .as_array()
            .ok_or("oracle records are missing")?;
        assert_eq!(records.len(), usize::try_from(width * height)?);
        for record in records {
            let x = u32::try_from(record[0].as_u64().ok_or("oracle column is missing")?)?;
            // The oracle's rows are bottom-up, Potter's pixel rows count from the top.
            let row = u32::try_from(record[1].as_u64().ok_or("oracle row is missing")?)?;
            let expected = [
                number(&record[2])?,
                number(&record[3])?,
                number(&record[4])?,
            ];
            assert_ray_matches(&camera, width, height, x, height - 1 - row, Some(expected))?;
        }
        Ok(())
    }

    #[test]
    fn fisheye_rays_match_blenders_render_and_leave_outside_pixels_rayless()
    -> Result<(), Box<dyn Error>> {
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/panorama_camera/blender_fisheye_camera.json"
        ))?;
        let width = u32::try_from(oracle["width"].as_u64().ok_or("oracle width is missing")?)?;
        let height = u32::try_from(
            oracle["height"]
                .as_u64()
                .ok_or("oracle height is missing")?,
        )?;
        for oracle_camera_data in oracle["cameras"]
            .as_array()
            .ok_or("oracle cameras are missing")?
        {
            let equisolid = match oracle_camera_data["panorama_type"].as_str() {
                Some("FISHEYE_EQUISOLID") => true,
                Some("FISHEYE_EQUIDISTANT") => false,
                _ => return Err("oracle panorama type is unknown".into()),
            };
            let mut data = CameraData {
                projection: CameraProjection::Panorama,
                panorama_type: if equisolid {
                    "fisheye_equisolid".to_owned()
                } else {
                    "fisheye_equidistant".to_owned()
                },
                fisheye_fov: number(&oracle_camera_data["fisheye_fov"])?,
                ..CameraData::default()
            };
            // The equisolid lens and sensor are stored only for equisolid cameras; an equidistant camera has none.
            if equisolid {
                data.fisheye_lens = number(&oracle_camera_data["fisheye_lens"])?;
                data.sensor_width_mm = number(&oracle_camera_data["sensor_width"])?;
                data.sensor_height_mm = number(&oracle_camera_data["sensor_height"])?;
                data.sensor_fit = oracle_camera_data["sensor_fit"]
                    .as_str()
                    .ok_or("oracle sensor fit is missing")?
                    .to_owned();
            }
            let projection = projection_for_camera(&data, CameraEngine::Path)?;
            let camera = oracle_camera(&oracle_camera_data["camera_to_world"], projection)?;
            let records = oracle_camera_data["records"]
                .as_array()
                .ok_or("oracle records are missing")?;
            assert_eq!(records.len(), usize::try_from(width * height)?);
            for record in records {
                let x = u32::try_from(record[0].as_u64().ok_or("oracle column is missing")?)?;
                let y = u32::try_from(record[1].as_u64().ok_or("oracle row is missing")?)?;
                // Blender writes `[x, y, has_ray, dx, dy, dz, r, g, b, a]`: a pixel without a ray stores its colour and alpha.
                let has_ray = record[2].as_bool() == Some(true);
                let expected = if has_ray {
                    Some([
                        number(&record[3])?,
                        number(&record[4])?,
                        number(&record[5])?,
                    ])
                } else {
                    None
                };
                assert_ray_matches(&camera, width, height, x, y, expected)?;
                if !has_ray {
                    // Blender's film is opaque in this render, so a pixel with no ray is black with alpha 1, and
                    // Potter adds no radiance there.
                    let colour = [
                        number(&record[6])?,
                        number(&record[7])?,
                        number(&record[8])?,
                        number(&record[9])?,
                    ];
                    assert_eq!(colour, [0.0, 0.0, 0.0, 1.0], "pixel ({x}, {y}) has no ray");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn panorama_ray_directions_are_unit_vectors() -> Result<(), Box<dyn Error>> {
        for projection in panorama_projections() {
            let camera = camera(projection);
            for y in 0..5 {
                for x in 0..7 {
                    let ray = ray_for_pixel(&camera, 7, 5, x, y)?;
                    if let Some((_, direction)) = ray {
                        assert!((direction.length() - 1.0).abs() < 1.0e-9);
                    }
                }
            }
        }
        Ok(())
    }

    /// Blender's default full-sphere panorama and fisheye limits are float32 values (`rna_camera.cc`), so their spans sit a
    /// hair beyond the f64 turn. Each default must be accepted and give a ray for every pixel, or a default camera fails.
    #[test]
    fn blender_default_panorama_limits_are_accepted_and_ray_every_pixel()
    -> Result<(), Box<dyn Error>> {
        assert!(crate::float::equal_f64(
            crate::model::BLENDER_TURN,
            f64::from(std::f32::consts::TAU)
        ));
        assert!(crate::float::equal_f64(
            crate::model::BLENDER_RIGHT_ANGLE,
            f64::from(std::f32::consts::FRAC_PI_2)
        ));
        let full = f64::from(std::f32::consts::PI);
        let defaults = [
            Projection::Equirectangular {
                longitude: [-full, full],
                latitude: [-0.5 * full, 0.5 * full],
            },
            Projection::FisheyeEquidistant { fov: full },
        ];
        for projection in defaults {
            let camera = camera(projection);
            for y in 0..5 {
                for x in 0..7 {
                    let ray = ray_for_pixel(&camera, 7, 5, x, y)?;
                    let Some((_, direction)) = ray else {
                        // Only a fisheye's corners lie outside its circle; the full sphere covers every pixel.
                        assert!(
                            matches!(projection, Projection::FisheyeEquidistant { .. }),
                            "{projection:?} pixel {x},{y} has no ray"
                        );
                        continue;
                    };
                    assert!((direction.length() - 1.0).abs() < 1.0e-9);
                }
            }
        }
        Ok(())
    }
    /// Cycles draws a panorama camera with its panorama projection; the rasterizer draws it as Blender's EEVEE and Workbench
    /// do, a perspective camera with `lens_mm`, and the fisheye's `fisheye_lens` belongs to the path tracer only.
    #[test]
    fn the_engine_picks_a_panorama_cameras_projection() -> Result<(), Box<dyn Error>> {
        let data = CameraData {
            projection: CameraProjection::Panorama,
            panorama_type: "fisheye_equisolid".to_owned(),
            lens_mm: 35.0,
            fisheye_lens: 12.0,
            fisheye_fov: 3.0,
            ..CameraData::default()
        };
        assert!(matches!(
            projection_for_camera(&data, CameraEngine::Path)?,
            Projection::FisheyeEquisolid { fisheye_lens, .. } if fisheye_lens == 12.0
        ));
        assert!(matches!(
            projection_for_camera(&data, CameraEngine::Raster)?,
            Projection::Perspective { lens_mm, .. } if lens_mm == 35.0
        ));
        let planar = CameraData {
            projection: CameraProjection::Perspective,
            ..data
        };
        assert!(matches!(
            projection_for_camera(&planar, CameraEngine::Path)?,
            Projection::Perspective { lens_mm, .. } if lens_mm == 35.0
        ));
        Ok(())
    }

    fn fisheye_camera(fov: f64) -> Camera {
        Camera {
            position: DVec3::ZERO,
            target: -DVec3::Z,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::FisheyeEquidistant { fov },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        }
    }

    /// A panorama covers every direction, so a stroke 120 degrees off its view axis is drawn; a view-axis clip would drop it.
    #[test]
    fn a_stroke_behind_a_full_sphere_panorama_is_drawn() -> Result<(), Box<dyn Error>> {
        let camera = Camera {
            projection: Projection::Equirectangular {
                longitude: [-std::f64::consts::PI, std::f64::consts::PI],
                latitude: [-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2],
            },
            ..fisheye_camera(0.0)
        };
        // 120 degrees from the forward axis (-Z): its view-axis depth is negative.
        let theta = 120.0_f64.to_radians();
        let point = DVec3::new(3.0 * theta.sin(), 0.0, -3.0 * theta.cos());
        let line = Line {
            start: point,
            end: point + DVec3::new(0.0, 0.2, 0.0),
            radius_start: 0.2,
            radius_end: 0.2,
            color: [1.0, 0.0, 0.0, 1.0],
            object_index: 7,
            element_index: 0,
            ray_mask: crate::model::RayVisibility::default().mask(),
        };
        let output = rasterize_with_lines(
            &[],
            &[line],
            &camera,
            64,
            32,
            Mode::Id,
            [0; 4],
            None,
            1,
            0,
            1.5,
            None,
            &[],
            None,
            None,
        )?;
        assert!(
            output.ids.contains(&7),
            "the stroke behind the view axis must reach the image"
        );
        Ok(())
    }

    /// A pixel with no camera ray (outside a fisheye's image circle) shows no stroke, so no stroke reaches the corners.
    #[test]
    fn strokes_leave_pixels_without_a_camera_ray_untouched() -> Result<(), Box<dyn Error>> {
        let camera = fisheye_camera(std::f64::consts::FRAC_PI_2);
        let (width, height) = (48_u32, 48_u32);
        // From 40 degrees (inside the 45 degree circle) out to 80 degrees (beyond it), along the image's horizontal axis.
        let at = |theta: f64| DVec3::new(3.0 * theta.sin(), 0.0, -3.0 * theta.cos());
        let line = Line {
            start: at(40.0_f64.to_radians()),
            end: at(80.0_f64.to_radians()),
            radius_start: 0.3,
            radius_end: 0.3,
            color: [1.0, 0.0, 0.0, 1.0],
            object_index: 7,
            element_index: 0,
            ray_mask: crate::model::RayVisibility::default().mask(),
        };
        let output = rasterize_with_lines(
            &[],
            &[line],
            &camera,
            width,
            height,
            Mode::Id,
            [0; 4],
            None,
            1,
            0,
            1.5,
            None,
            &[],
            None,
            None,
        )?;
        let mut drawn = 0_usize;
        for y in 0..height {
            for x in 0..width {
                let index = usize::try_from(y * width + x)?;
                let has_ray = ray_for_pixel(&camera, width, height, x, y)?.is_some();
                if output.ids[index] == 7 {
                    drawn += 1;
                    assert!(
                        has_ray,
                        "pixel ({x}, {y}) has no camera ray but shows the stroke"
                    );
                }
                if !has_ray {
                    assert_eq!(output.ids[index], 0, "pixel ({x}, {y}) has no camera ray");
                }
            }
        }
        assert!(
            drawn > 0,
            "the part of the stroke inside the circle must be drawn"
        );
        Ok(())
    }
}

#[cfg(test)]
mod line_tests {
    use super::{Camera, Line, Mode, Projection, StereoMode, rasterize_with_lines};
    use crate::model::RayVisibility;
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
            ray_mask: RayVisibility::default().mask(),
        };
        let output = rasterize_with_lines(
            &[],
            &[line],
            &camera,
            7,
            5,
            Mode::Beauty,
            [242, 242, 242, 255],
            None,
            1,
            0,
            1.5,
            None,
            &[],
            None,
            None,
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

#[cfg(test)]
mod smooth_shading_tests {
    #![expect(clippy::expect_used, reason = "small raster fixtures")]

    use std::cell::RefCell;

    use glam::DVec3;

    use super::{
        AlphaMode, Camera, Mode, Projection, RasterOutput, StereoMode, SurfaceHit, SurfaceSample,
        SurfaceShader, Triangle, rasterize_with_samples,
    };
    use crate::error::Result;
    use crate::model::RayVisibility;

    const SIZE: u32 = 64;
    /// Pixel (32, 32) has its centre at world (1/64, -1/64, 0) for this camera.
    const CENTER: usize = 32 * SIZE as usize + 32;
    const CORNERS: [DVec3; 3] = [
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(0.6, 0.0, 0.8),
        DVec3::new(0.0, 0.6, 0.8),
    ];

    struct RecordingShader {
        calls: RefCell<Vec<(DVec3, DVec3, DVec3)>>,
    }

    impl SurfaceShader for RecordingShader {
        fn shade(&self, _triangle: &Triangle, hit: &SurfaceHit) -> Result<SurfaceSample> {
            self.calls
                .borrow_mut()
                .push((hit.point, hit.normal, hit.geometric));
            Ok(SurfaceSample::opaque([0.25, 0.5, 0.75, 1.0]))
        }
    }

    fn camera() -> Camera {
        Camera {
            position: DVec3::new(0.0, 0.0, 5.0),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 2.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        }
    }

    fn triangle(normals: Option<[DVec3; 3]>) -> Triangle {
        Triangle {
            positions: [
                DVec3::new(-2.0, -2.0, 0.0),
                DVec3::new(2.0, -2.0, 0.0),
                DVec3::new(0.0, 2.0, 0.0),
            ],
            generated: crate::geom::GeneratedCorners::default(),
            object_index: 1,
            element_index: 1,
            color: [0.8, 0.4, 0.2, 1.0],
            alpha_mode: AlphaMode::Opaque,
            alpha_threshold: 0.5,
            double_sided: false,
            material_index: 0,
            uv: [[0.0; 2]; 3],
            layer_uvs: Vec::new(),
            normals,
            layer_colors: Vec::new(),
            lit_per_pixel: false,
            ray_mask: RayVisibility::default().mask(),
        }
    }

    fn render(
        triangles: &[Triangle],
        mode: Mode,
        shader: Option<&dyn SurfaceShader>,
    ) -> RasterOutput {
        rasterize_with_samples(
            triangles,
            &camera(),
            SIZE,
            SIZE,
            mode,
            [0, 0, 0, 255],
            None,
            1,
            0,
            1.5,
            shader,
            &[],
            None,
            None,
        )
        .expect("triangle rasterizes")
    }

    struct FrontShader {
        rgba: [f64; 4],
    }

    impl SurfaceShader for FrontShader {
        fn shade(&self, triangle: &Triangle, _hit: &SurfaceHit) -> Result<SurfaceSample> {
            if triangle.object_index == 1 {
                Ok(SurfaceSample::opaque(self.rgba))
            } else {
                Ok(SurfaceSample::opaque(triangle.color))
            }
        }
    }

    fn layered_triangles(alpha_mode: AlphaMode) -> [Triangle; 2] {
        let mut front = triangle(Some([DVec3::Z; 3]));
        front.alpha_mode = alpha_mode;
        let mut back = triangle(None);
        back.object_index = 2;
        back.positions = back.positions.map(|position| position - DVec3::Z);
        back.color = [0.0, 0.0, 1.0, 1.0];
        [front, back]
    }

    /// Barycentric weights of the centre pixel's world point, in triangle corner order.
    fn center_weights() -> [f64; 3] {
        let point = DVec3::new(1.0 / 64.0, -1.0 / 64.0, 0.0);
        let positions = triangle(None).positions;
        let (a, b, c) = (positions[0], positions[1], positions[2]);
        let (v0, v1, v2) = (b - a, c - a, point - a);
        let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
        let (d20, d21) = (v2.dot(v0), v2.dot(v1));
        let denominator = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denominator;
        let w = (d00 * d21 - d01 * d20) / denominator;
        [1.0 - v - w, v, w]
    }

    #[test]
    fn smooth_normal_pass_interpolates_corner_normals_per_pixel() {
        let output = render(&[triangle(Some(CORNERS))], Mode::Normal, None);
        let weights = center_weights();
        let expected =
            (CORNERS[0] * weights[0] + CORNERS[1] * weights[1] + CORNERS[2] * weights[2])
                .normalize();
        let actual = DVec3::from_array(output.normals[CENTER].map(f64::from));
        assert!(
            actual.abs_diff_eq(expected, 1e-5),
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn flat_normal_pass_keeps_the_geometric_normal() {
        let output = render(&[triangle(None)], Mode::Normal, None);
        let actual = DVec3::from_array(output.normals[CENTER].map(f64::from));
        assert!(actual.abs_diff_eq(DVec3::Z, 1e-6), "got {actual:?}");
    }

    #[test]
    fn smooth_triangles_are_lit_per_pixel_with_shading_and_geometric_normals() {
        let shader = RecordingShader {
            calls: RefCell::new(Vec::new()),
        };
        let output = render(&[triangle(Some(CORNERS))], Mode::Solid, Some(&shader));
        let color = output.linear_rgba[CENTER];
        for (channel, expected) in color.iter().zip([0.25, 0.5, 0.75, 1.0]) {
            assert!(
                (f64::from(*channel) - expected).abs() < 1e-5,
                "color {color:?}"
            );
        }
        let calls = shader.calls.borrow();
        let center_point = DVec3::new(1.0 / 64.0, -1.0 / 64.0, 0.0);
        let (_, normal, geometric) = *calls
            .iter()
            .find(|(point, _, _)| point.abs_diff_eq(center_point, 1e-9))
            .expect("shader ran at the centre pixel");
        let weights = center_weights();
        let expected =
            (CORNERS[0] * weights[0] + CORNERS[1] * weights[1] + CORNERS[2] * weights[2])
                .normalize();
        assert!(normal.abs_diff_eq(expected, 1e-6), "normal {normal:?}");
        assert!(
            geometric.abs_diff_eq(DVec3::Z, 1e-9),
            "geometric {geometric:?}"
        );
    }

    #[test]
    fn flat_triangles_are_shaded_per_hit_with_their_geometric_normal() {
        let shader = RecordingShader {
            calls: RefCell::new(Vec::new()),
        };
        let output = render(&[triangle(None)], Mode::Solid, Some(&shader));
        let color = output.linear_rgba[CENTER];
        for (channel, expected) in color.iter().zip([0.25, 0.5, 0.75, 1.0]) {
            assert!(
                (f64::from(*channel) - expected).abs() < 1e-5,
                "color {color:?}"
            );
        }
        let calls = shader.calls.borrow();
        let center_point = DVec3::new(1.0 / 64.0, -1.0 / 64.0, 0.0);
        let (_, normal, geometric) = *calls
            .iter()
            .find(|(point, _, _)| point.abs_diff_eq(center_point, 1e-9))
            .expect("shader ran at the centre pixel");
        assert!(normal.abs_diff_eq(DVec3::Z, 1e-9), "normal {normal:?}");
        assert!(
            geometric.abs_diff_eq(DVec3::Z, 1e-9),
            "geometric {geometric:?}"
        );
    }

    #[test]
    fn zero_shader_alpha_does_not_let_a_blend_hit_mask_geometry() {
        let shader = FrontShader {
            rgba: [1.0, 0.0, 0.0, 0.0],
        };
        let output = render(
            &layered_triangles(AlphaMode::Blend),
            Mode::Solid,
            Some(&shader),
        );
        assert_eq!(output.ids[CENTER], 2);
        let color = output.linear_rgba[CENTER];
        for (actual, expected) in color.iter().zip([0.0, 0.0, 1.0, 1.0]) {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-5,
                "color {color:?}"
            );
        }
    }

    #[test]
    fn shader_alpha_controls_blend_compositing() {
        let shader = FrontShader {
            rgba: [1.0, 0.0, 0.0, 0.25],
        };
        let output = render(
            &layered_triangles(AlphaMode::Blend),
            Mode::Solid,
            Some(&shader),
        );
        assert_eq!(output.ids[CENTER], 1);
        let color = output.linear_rgba[CENTER];
        for (actual, expected) in color.iter().zip([0.25, 0.0, 0.75, 1.0]) {
            assert!(
                (f64::from(*actual) - expected).abs() < 1e-5,
                "color {color:?}"
            );
        }
    }

    #[test]
    fn clip_threshold_and_opaque_mode_use_shader_rgba_alpha() {
        let below_threshold = FrontShader {
            rgba: [1.0, 0.0, 0.0, 0.49],
        };
        let output = render(
            &layered_triangles(AlphaMode::Clip),
            Mode::Solid,
            Some(&below_threshold),
        );
        assert_eq!(output.ids[CENTER], 2);

        let at_threshold = FrontShader {
            rgba: [1.0, 0.0, 0.0, 0.5],
        };
        let output = render(
            &layered_triangles(AlphaMode::Clip),
            Mode::Solid,
            Some(&at_threshold),
        );
        assert_eq!(output.ids[CENTER], 1);
        assert!((f64::from(output.linear_rgba[CENTER][3]) - 1.0).abs() < 1e-5);

        let transparent_opaque = FrontShader {
            rgba: [1.0, 0.0, 0.0, 0.0],
        };
        let output = render(
            &layered_triangles(AlphaMode::Opaque),
            Mode::Solid,
            Some(&transparent_opaque),
        );
        assert_eq!(output.ids[CENTER], 1);
        assert!((f64::from(output.linear_rgba[CENTER][3]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ambient_visibility_mirrors_smooth_rays_back_above_the_geometric_surface() {
        let plane = triangle(None);
        let bvh =
            crate::render::bvh::Bvh::build(std::slice::from_ref(&plane)).expect("plane builds");
        // The shading normal leans 30 degrees from the plane, so some rays dip below it.
        let visibility = super::ambient_visibility(
            &bvh,
            std::slice::from_ref(&plane),
            &[],
            DVec3::ZERO,
            DVec3::Z,
            DVec3::new(0.866, 0.0, 0.5),
            1.0,
        );
        assert!((visibility - 1.0).abs() < 1e-12, "visibility {visibility}");
    }

    #[test]
    fn smooth_shadow_origin_lifts_flat_hits_onto_the_curved_surface() {
        let sphere = Triangle {
            positions: [DVec3::X, DVec3::Y, DVec3::Z],
            normals: Some([DVec3::X, DVec3::Y, DVec3::Z]),
            ..triangle(None)
        };
        let centre = DVec3::splat(1.0 / 3.0);
        let lifted = super::smooth_shadow_origin(&sphere, centre, [1.0 / 3.0; 3]);
        assert!(lifted.length() > centre.length(), "lifted {lifted:?}");
        let flat = triangle(None);
        assert_eq!(
            super::smooth_shadow_origin(&flat, centre, [1.0 / 3.0; 3]),
            centre
        );
    }
}

#[cfg(test)]
mod visibility_tests {
    #![expect(clippy::expect_used, reason = "small raster fixtures")]

    use glam::DVec3;

    use super::{
        AlphaMode, Camera, CameraFrame, Hit, Layer, Mode, PreparedTriangle, Projection, Ray,
        StereoMode, Triangle, camera_frame, composite_layers_with_background, composite_sample,
        hit_opacity, intersect_triangle, nearest_visible_hit, prepare_triangles,
        ray_for_pixel_in_frame,
    };
    use crate::float::equal_f64;
    use crate::model::RayVisibility;
    use crate::render::bvh::Bvh;

    const SIZE: u32 = 48;
    const GRID: usize = 10;
    const BACKGROUND: [f64; 4] = [0.2, 0.3, 0.4, 1.0];

    fn corner(i: usize, j: usize) -> DVec3 {
        let x = i as f64 / GRID as f64 * 2.0 - 1.0;
        let y = j as f64 / GRID as f64 * 2.0 - 1.0;
        let z = 0.1 * ((i * 7 + j * 5) % 4) as f64;
        DVec3::new(x, y, z)
    }

    fn triangle(positions: [DVec3; 3]) -> Triangle {
        Triangle {
            positions,
            generated: crate::geom::GeneratedCorners::default(),
            object_index: 1,
            element_index: 1,
            color: [0.5, 0.5, 0.5, 1.0],
            alpha_mode: AlphaMode::Opaque,
            alpha_threshold: 0.5,
            double_sided: true,
            material_index: 0,
            uv: [[0.0; 2]; 3],
            layer_uvs: Vec::new(),
            normals: None,
            layer_colors: Vec::new(),
            lit_per_pixel: false,
            ray_mask: RayVisibility::default().mask(),
        }
    }

    fn translucent(mut triangle: Triangle, color: [f64; 4]) -> Triangle {
        triangle.alpha_mode = AlphaMode::Blend;
        triangle.color = color;
        triangle
    }

    /// A bumpy grid whose every triangle appears twice, so each hit ties on depth. The first copy
    /// is translucent green and the second translucent red, so the order of tied layers changes
    /// the composite. A zero-area triangle comes first: preparation drops it, so the visibility
    /// BVH's indices must count prepared triangles, not input triangles.
    fn tied_grid() -> Vec<Triangle> {
        let mut triangles = vec![triangle([DVec3::ZERO, DVec3::X, DVec3::X * 2.0])];
        for i in 0..GRID {
            for j in 0..GRID {
                let a = corner(i, j);
                let b = corner(i + 1, j);
                let c = corner(i, j + 1);
                let d = corner(i + 1, j + 1);
                triangles.push(translucent(triangle([a, b, c]), [0.2, 0.6, 0.2, 0.5]));
                triangles.push(translucent(triangle([b, d, c]), [0.2, 0.6, 0.2, 0.5]));
            }
        }
        let copies: Vec<Triangle> = triangles
            .iter()
            .map(|original| translucent(original.clone(), [0.9, 0.1, 0.1, 0.5]))
            .collect();
        triangles.extend(copies);
        triangles
    }

    fn test_camera() -> Camera {
        Camera {
            position: DVec3::new(0.0, 0.0, 2.5),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Perspective {
                lens_mm: 50.0,
                sensor_width_mm: 36.0,
            },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        }
    }

    /// The reference: every prepared triangle, least (depth, index) hit.
    fn scan_nearest(
        triangles: &[PreparedTriangle],
        ray: Ray,
        frame: &CameraFrame,
    ) -> Option<(usize, f64)> {
        let mut nearest: Option<(usize, Hit)> = None;
        for (index, triangle) in triangles.iter().enumerate() {
            let Some(hit) = intersect_triangle(triangle, ray, frame) else {
                continue;
            };
            if nearest.is_none_or(|(best, best_hit)| {
                hit.depth < best_hit.depth || (equal_f64(hit.depth, best_hit.depth) && index < best)
            }) {
                nearest = Some((index, hit));
            }
        }
        nearest.map(|(index, hit)| (index, hit.depth))
    }

    /// The composite before candidate culling: every prepared triangle in index order, inserted
    /// by depth (equal depths keep index order), the 16 nearest layers kept, then the same
    /// accumulation as `composite_sample`. Shading is the triangle colour, as with no shader.
    fn scan_composite(
        triangles: &[PreparedTriangle],
        ray: Ray,
        frame: &CameraFrame,
        background: [f64; 4],
    ) -> super::Result<[f64; 4]> {
        let mut layers: Vec<(usize, Hit)> = Vec::new();
        for (index, triangle) in triangles.iter().enumerate() {
            let Some(hit) = intersect_triangle(triangle, ray, frame) else {
                continue;
            };
            if hit_opacity(&triangle.source, triangle.source.color) <= 0.0 {
                continue;
            }
            let at = layers
                .iter()
                .position(|(_, previous)| previous.depth > hit.depth)
                .unwrap_or(layers.len());
            layers.insert(at, (index, hit));
            layers.truncate(16);
        }
        composite_layers_with_background(
            layers.iter().map(|(index, _)| {
                let triangle = &triangles[*index];
                let color = triangle.source.color;
                Layer {
                    rgb: [color[0], color[1], color[2]],
                    opacity: hit_opacity(&triangle.source, color),
                    tint: [1.0; 3],
                }
            }),
            background[3],
            || Ok([background[0], background[1], background[2]]),
        )
    }

    #[test]
    fn bounded_nearest_hit_matches_a_scan_of_every_triangle() {
        let prepared = prepare_triangles(&tied_grid(), Mode::Solid).expect("grid prepares");
        assert_eq!(
            prepared.len(),
            4 * GRID * GRID,
            "the zero-area triangle is dropped"
        );
        let visibility =
            Bvh::build_from_positions(prepared.iter().map(|triangle| triangle.source.positions))
                .expect("visibility BVH builds");
        let camera = test_camera();
        let frame = camera_frame(&camera, SIZE, SIZE).expect("frame");
        let mut candidates = Vec::new();
        let mut hit_pixels = 0_usize;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let ray = ray_for_pixel_in_frame(&frame, camera.position, SIZE, SIZE, x, y)
                    .expect("pixel ray")
                    .expect("the perspective test view has a ray");
                let expected = scan_nearest(&prepared, ray, &frame);
                let actual =
                    nearest_visible_hit(&prepared, &visibility, &mut candidates, ray, &frame, None)
                        .expect("visibility resolves")
                        .map(|visible| (visible.triangle_index, visible.hit.depth));
                assert_eq!(actual, expected, "pixel ({x}, {y})");
                hit_pixels += usize::from(expected.is_some());
            }
        }
        // The grid spans [-1, 1]; at z = 2.5 this lens sees about [-0.9, 0.9], so it fills the view.
        assert!(
            hit_pixels > (SIZE * SIZE) as usize * 9 / 10,
            "grid covers the view"
        );
    }

    #[test]
    fn translucent_composite_matches_a_scan_of_every_triangle() {
        let prepared = prepare_triangles(&tied_grid(), Mode::Solid).expect("grid prepares");
        let visibility =
            Bvh::build_from_positions(prepared.iter().map(|triangle| triangle.source.positions))
                .expect("visibility BVH builds");
        let camera = test_camera();
        let frame = camera_frame(&camera, SIZE, SIZE).expect("frame");
        let mut candidates = Vec::new();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let ray = ray_for_pixel_in_frame(&frame, camera.position, SIZE, SIZE, x, y)
                    .expect("pixel ray")
                    .expect("the perspective test view has a ray");
                let expected = scan_composite(&prepared, ray, &frame, BACKGROUND)
                    .expect("reference composite resolves");
                let actual = composite_sample(
                    &prepared,
                    &visibility,
                    &mut candidates,
                    ray,
                    &frame,
                    &camera,
                    BACKGROUND,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("composite resolves");
                assert_eq!(actual, expected, "pixel ({x}, {y})");
            }
        }
    }
}

#[cfg(test)]
mod composite_layer_tests {
    use super::{Layer, composite_layers_with_background};

    fn composite_layers(
        layers: impl IntoIterator<Item = Layer>,
        background: [f64; 4],
    ) -> super::Result<[f64; 4]> {
        composite_layers_with_background(layers, background[3], || {
            Ok([background[0], background[1], background[2]])
        })
    }

    fn assert_close(actual: [f64; 4], expected: [f64; 4]) {
        for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "channel {channel}: {actual} != {expected} (full {actual:?} vs {expected:?})"
            );
        }
    }

    /// Two coloured layers over a background of alpha 0.6, worked out by hand.
    /// Layer A (rgb (1,0,0), opacity 0.5, tint (1,0.5,0)): colour += (1,0,0) * 0.5 = (0.5,0,0); transmittance becomes
    /// (1,0.5,0) * 0.5 = (0.5,0.25,0); alpha transmittance becomes mean(1,0.5,0) * 0.5 = 0.25.
    /// Layer B (rgb (0,1,0), opacity 0, tint (0.2,0.4,0.8)): colour += 0; transmittance becomes (0.1,0.1,0); alpha
    /// transmittance becomes 0.25 * mean(0.2,0.4,0.8) = 0.25 * 1.4/3.
    /// Background (0.5,0.5,0.5), alpha 0.6: colour += (0.5,0.5,0.5) * (0.1,0.1,0) * 0.6 = (0.03,0.03,0).
    /// So the colour sums to (0.53,0.03,0) and the alpha is 1 - (0.25 * 1.4/3) * (1 - 0.6).
    #[test]
    fn coloured_layers_over_a_partly_transparent_background() -> super::Result<()> {
        let layers = [
            Layer {
                rgb: [1.0, 0.0, 0.0],
                opacity: 0.5,
                tint: [1.0, 0.5, 0.0],
            },
            Layer {
                rgb: [0.0, 1.0, 0.0],
                opacity: 0.0,
                tint: [0.2, 0.4, 0.8],
            },
        ];
        let alpha = 1.0 - (0.25 * 1.4 / 3.0) * (1.0 - 0.6);
        let actual = composite_layers(layers, [0.5, 0.5, 0.5, 0.6])?;
        assert_close(actual, [0.53 / alpha, 0.03 / alpha, 0.0, alpha]);
        Ok(())
    }

    /// Two panes that tint different channels over a transparent film. Each mean is 1/3, so the alpha is
    /// 1 - (1/3) * (1/3) = 8/9. A product of the colour vectors would give zero transmittance and alpha 1.
    #[test]
    fn overlapping_panes_multiply_their_alpha_means() -> super::Result<()> {
        let layers = [
            Layer {
                rgb: [0.0; 3],
                opacity: 0.0,
                tint: [1.0, 0.0, 0.0],
            },
            Layer {
                rgb: [0.0; 3],
                opacity: 0.0,
                tint: [0.0, 1.0, 0.0],
            },
        ];
        let actual = composite_layers(layers, [0.0; 4])?;
        assert_close(actual, [0.0, 0.0, 0.0, 8.0 / 9.0]);
        Ok(())
    }

    /// A white layer is the scalar composite: colour (0.8,0.2,0.4) at opacity 0.25 over an opaque background of
    /// (0.5,0.5,0.5). The colour is 0.25 * (0.8,0.2,0.4) + 0.75 * (0.5,0.5,0.5) = (0.575,0.425,0.475), alpha 1.
    #[test]
    fn white_layer_is_the_scalar_composite() -> super::Result<()> {
        let layers = [Layer {
            rgb: [0.8, 0.2, 0.4],
            opacity: 0.25,
            tint: [1.0; 3],
        }];
        let actual = composite_layers(layers, [0.5, 0.5, 0.5, 1.0])?;
        assert_close(actual, [0.575, 0.425, 0.475, 1.0]);
        Ok(())
    }

    /// No layer over a fully transparent background is that background, as the film has nothing to show.
    #[test]
    fn empty_stack_over_transparent_background_is_the_background() -> super::Result<()> {
        let actual = composite_layers(std::iter::empty(), [0.3, 0.4, 0.5, 0.0])?;
        assert_close(actual, [0.3, 0.4, 0.5, 0.0]);
        Ok(())
    }
}
