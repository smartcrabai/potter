use std::f64::consts::TAU;

use glam::{DMat4, DVec3, Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use super::eevee_advanced_bsdf::EeveeSubsurfaceSample;
use super::eevee_advanced_bsdf::{
    EeveeSubsurfaceFrame, EeveeSubsurfacePixel, EeveeSubsurfaceProjection,
    resolve_screen_space_subsurface,
};
use super::eevee_fast_gi as fast_gi;
use super::eevee_fast_gi::FastGiSh;
use super::eevee_raytrace::EeveeRayTracingSettings;
use super::eevee_raytrace::{
    HiZBuffer, HiZLevel, ScreenRadianceBuffer, ScreenThicknessParameters, ScreenTraceCamera,
    ScreenTraceError, ScreenTraceParameters, ScreenTraceRay, raytrace_sample_screen,
    raytrace_screen,
};
use super::eevee_raytrace_denoise as denoise;
use super::eevee_raytrace_generate::{
    RayGenerateError, RayGenerateInput, RayLobe, ThicknessMode, generate_ray,
};
use super::wireframe::{self, PointSquare, ScreenPoint, WireEdge, WirePoint, WireSegment};
use crate::error::{ErrorCode, PotError, Result};
use crate::float::equal_f64;
use crate::geom::GeneratedCorners;
use crate::model::{CameraData, CameraProjection, RayKind};
use crate::shader::HitDifferentials;

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
    Dithered,
    Clip,
}
#[derive(Debug, Clone)]
pub(crate) struct Triangle {
    pub(crate) positions: [DVec3; 3],
    /// The Generated coordinate of each corner and the Jacobian the corners share. A hit interpolates the corners over
    /// its barycentric weights, as Cycles stores Generated per vertex and interpolates it
    /// (`intern/cycles/blender/mesh.cpp`, `ATTR_STD_GENERATED`), so a corner value of zero stays zero.
    pub(crate) generated: GeneratedCorners,
    /// Source point slots for the three original mesh vertices of this triangle.
    pub(crate) geometry_vertex_slots: [usize; 3],
    /// Each current corner's barycentric coordinates in the original source triangle.
    pub(crate) geometry_vertex_weights: [[f64; 3]; 3],
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
    /// World-space per-corner Mikk tangents for each UV layer; absent when no UV tangent graph is used.
    pub(crate) layer_tangents: Vec<[DVec3; 3]>,
    /// World-space corner shading normals; realtime Source may provide them on flat faces too.
    pub(crate) normals: Option<[DVec3; 3]>,
    /// Source face smoothness, distinct from whether packed corner normals are present.
    pub(crate) smooth_shading: bool,
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
        layer_tangents: Vec<[DVec3; 3]>,
        layer_colors: Vec<[[f64; 4]; 3]>,
        geometry_vertex_weights: [[f64; 3]; 3],
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
            layer_tangents,
            normals: self.normals,
            smooth_shading: self.smooth_shading,
            layer_colors,
            geometry_vertex_slots: self.geometry_vertex_slots,
            geometry_vertex_weights,
            lit_per_pixel: self.lit_per_pixel,
            ray_mask: self.ray_mask,
        }
    }

    pub(crate) const fn geometry_attribute_frame(&self) -> ([usize; 3], [[f64; 3]; 3]) {
        (self.geometry_vertex_slots, self.geometry_vertex_weights)
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
                Vec::new(),
                self.geometry_vertex_weights,
            )
        }
    }
}

/// Where a visible hit lies on its triangle, for per-pixel shading.
pub(crate) struct SurfaceHit<'a> {
    /// Hit point in world space.
    pub(crate) point: DVec3,
    /// Shadow-ray origin (see [`smooth_shadow_origin`]).
    pub(crate) shadow_point: DVec3,
    /// Interpolated shading normal.
    pub(crate) normal: DVec3,
    /// Unit triangle normal.
    pub(crate) geometric: DVec3,
    /// Face-forwarded geometric normal supplied to the shader context.
    pub(crate) geometric_normal: DVec3,
    /// Whether the un-oriented face normal points away from the camera ray.
    pub(crate) back_facing: bool,
    /// The EEVEE screen-space derivatives at this hit.
    pub(crate) realtime_differentials: HitDifferentials,
    /// The camera matrices for the eye that produced this hit.
    pub(crate) camera_matrices: Option<&'a (DMat4, DMat4)>,
    /// Barycentric weights of the hit over the triangle's corners.
    pub(crate) barycentric: [f64; 3],
    /// EEVEE G-buffer fragment coordinate and per-sample dither, when rasterized.
    pub(crate) gbuffer_frag_coord: Option<[f32; 2]>,
    pub(crate) gbuffer_sample_offset: Option<[f32; 3]>,
    pub(crate) gbuffer_closure_random: Option<f32>,
    /// Unit direction from the hit back toward the camera.
    pub(crate) view: DVec3,
    pub(crate) eevee_raytrace_feedback: EeveeRaytraceFeedback,
}

/// One source-selected EEVEE G-buffer bin, before transport tracing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EeveeRaytraceBin {
    pub(crate) world_input: RayGenerateInput,
    pub(crate) dominant_direction: DVec3,
    pub(crate) transmission: bool,
}
#[derive(Clone, Copy, Default)]
pub(crate) struct EeveeRaytraceLayerFlags {
    /// Source closure bits after its empty-material EMISSION fallback.
    pub(crate) has_any_closure: bool,
    /// Source REFLECTION|DIFFUSE closure flags used to enable screen reflection.
    pub(crate) has_reflect_closure: bool,
    /// Source REFRACTION|TRANSLUCENT flags used to enable ray tracing on this layer.
    pub(crate) has_transmit_closure: bool,
    /// Source `has_transmission()` result used to decide whether the next layer refracts this one.
    pub(crate) has_transmission: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct EeveeRaytraceLayerSummary {
    pub(crate) opaque: EeveeRaytraceLayerFlags,
    pub(crate) refraction: Option<EeveeRaytraceLayerFlags>,
}

/// Per-sample camera data consumed by EEVEE screen tracing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EeveeRaytraceSample {
    pub(crate) point: DVec3,
    pub(crate) normal: DVec3,
    /// Whether the source material uses EEVEE's deferred G-buffer route.
    pub(crate) has_gbuffer_route: bool,
    /// Whole-material EEVEE screen-refraction-layer selection, independent of G-buffer bins.
    pub(crate) screen_refraction: Option<bool>,
    /// EEVEE's compacted storage-layer order, not graph-closure order.
    pub(crate) bins: [Option<EeveeRaytraceBin>; 3],
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum EeveeRaytraceLayer {
    Opaque,
    Refraction,
}

impl EeveeRaytraceLayer {
    fn contains(self, sample: Option<&EeveeRaytraceSample>) -> bool {
        let Some(sample) = sample else {
            return self == Self::Opaque;
        };
        sample.has_gbuffer_route
            && (sample.screen_refraction == Some(true)) == (self == Self::Refraction)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct EeveeRaytraceFeedback {
    /// Opaque-layer closure radiance, indexed by source storage layer.
    pub(crate) radiance: [Option<[f32; 3]>; 3],
    /// Refraction-layer radiance uses the same slots but must not alias them.
    pub(crate) refraction_radiance: [Option<[f32; 3]>; 3],
    pub(crate) refraction_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EeveeDitherSample {
    /// The raw average of EEVEE's `g_transmittance`, used by the hashed depth prepass.
    pub(crate) transparency: f32,
    /// Reciprocal of EEVEE's deferred G-buffer alpha, used to unassociate a kept fragment.
    pub(crate) alpha_reciprocal: f32,
}

/// What the raster hit evaluator returns: its linear RGBA, whether this is a camera surface,
/// transparent and holdout shares, and EEVEE closure data that needs a frame resolve.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SurfaceSample {
    pub(crate) color: [f64; 4],
    /// Whether this sample is an actual surface rather than a volume-only boundary.
    pub(crate) has_surface: bool,
    /// The Transparent closure's colour per channel, as `(1 - alpha) * tint` is its transmittance.
    pub(crate) transparent_tint: [f64; 3],
    /// EEVEE matte share, independent of physical transparency. It removes film alpha only when the film is transparent.
    pub(crate) holdout_weight: f64,
    /// EEVEE's deferred SSS data; its direct/indirect terms are resolved across this sample's complete screen frame.
    pub(crate) subsurface: Option<EeveeSubsurfaceSample>,
    /// Actual source-selected G-buffer bins for this camera sample.
    pub(crate) eevee_raytrace: Option<EeveeRaytraceSample>,
    /// Source EEVEE transparency and alpha normalization for a DITHERED fragment.
    pub(crate) eevee_dither: Option<EeveeDitherSample>,
}

impl SurfaceSample {
    /// A sample with no transparent, holdout, or subsurface share.
    pub(crate) fn opaque(color: [f64; 4]) -> Self {
        Self {
            color,
            has_surface: true,
            transparent_tint: [1.0; 3],
            holdout_weight: 0.0,
            subsurface: None,
            eevee_raytrace: None,
            eevee_dither: None,
        }
    }
}

/// Linear RGBA and the transparent tint at `hit` on `triangle`.
pub(crate) trait SurfaceShader {
    fn shade(&self, triangle: &Triangle, hit: &SurfaceHit<'_>) -> Result<SurfaceSample>;
    fn needs_eevee_gbuffer_context(&self) -> bool {
        false
    }
    fn eevee_raytracing_settings(&self) -> Result<Option<EeveeRayTracingSettings>> {
        Ok(None)
    }
    fn eevee_raytrace_layer_flags(&self) -> Result<Option<EeveeRaytraceLayerSummary>> {
        Ok(None)
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        _sample: &EeveeRaytraceSample,
        _bin: EeveeRaytraceBin,
        _world_direction: DVec3,
    ) -> Result<DVec3> {
        Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "EEVEE ray tracing requires a source probe-radiance evaluator",
        ))
    }
    fn eevee_fast_gi_volume_irradiance(&self, _point: DVec3) -> Result<Option<FastGiSh>> {
        Ok(None)
    }

    fn worker(&self) -> Result<Option<Box<dyn SurfaceShaderWorker + Send + '_>>> {
        Ok(None)
    }

    fn merge_worker_warnings(&self, _warnings: Vec<serde_json::Value>) {}
}

pub(crate) trait SurfaceShaderWorker: Send {
    fn shade(
        &self,
        triangle: &Triangle,
        hit: &SurfaceHit<'_>,
        warnings: &std::cell::RefCell<Vec<serde_json::Value>>,
    ) -> Result<SurfaceSample>;

    fn needs_eevee_gbuffer_context(&self) -> bool {
        false
    }
    fn eevee_raytracing_settings(&self) -> Result<Option<EeveeRayTracingSettings>> {
        Ok(None)
    }
    fn eevee_raytrace_layer_flags(&self) -> Result<Option<EeveeRaytraceLayerSummary>> {
        Ok(None)
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        _sample: &EeveeRaytraceSample,
        _bin: EeveeRaytraceBin,
        _world_direction: DVec3,
    ) -> Result<DVec3> {
        Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "EEVEE ray tracing requires a source probe-radiance evaluator",
        ))
    }
    fn eevee_fast_gi_volume_irradiance(&self, _point: DVec3) -> Result<Option<FastGiSh>> {
        Ok(None)
    }
}

#[derive(Clone, Copy)]
enum RasterShader<'a> {
    Main(&'a dyn SurfaceShader),
    Worker(
        &'a dyn SurfaceShaderWorker,
        &'a std::cell::RefCell<Vec<serde_json::Value>>,
    ),
}

impl RasterShader<'_> {
    fn shade(&self, triangle: &Triangle, hit: &SurfaceHit<'_>) -> Result<SurfaceSample> {
        match self {
            Self::Main(shader) => shader.shade(triangle, hit),
            Self::Worker(shader, warnings) => shader.shade(triangle, hit, warnings),
        }
    }
    fn needs_eevee_gbuffer_context(&self) -> bool {
        match self {
            Self::Main(shader) => shader.needs_eevee_gbuffer_context(),
            Self::Worker(shader, _) => shader.needs_eevee_gbuffer_context(),
        }
    }
    fn eevee_raytracing_settings(&self) -> Result<Option<EeveeRayTracingSettings>> {
        match self {
            Self::Main(shader) => shader.eevee_raytracing_settings(),
            Self::Worker(shader, _) => shader.eevee_raytracing_settings(),
        }
    }
    fn eevee_raytrace_layer_flags(&self) -> Result<Option<EeveeRaytraceLayerSummary>> {
        match self {
            Self::Main(shader) => shader.eevee_raytrace_layer_flags(),
            Self::Worker(shader, _) => shader.eevee_raytrace_layer_flags(),
        }
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        sample: &EeveeRaytraceSample,
        bin: EeveeRaytraceBin,
        world_direction: DVec3,
    ) -> Result<DVec3> {
        match self {
            Self::Main(shader) => {
                shader.eevee_raytrace_probe_radiance(sample, bin, world_direction)
            }
            Self::Worker(shader, _) => {
                shader.eevee_raytrace_probe_radiance(sample, bin, world_direction)
            }
        }
    }
    fn eevee_fast_gi_volume_irradiance(&self, point: DVec3) -> Result<Option<FastGiSh>> {
        match self {
            Self::Main(shader) => shader.eevee_fast_gi_volume_irradiance(point),
            Self::Worker(shader, _) => shader.eevee_fast_gi_volume_irradiance(point),
        }
    }
}

/// The camera's view axes in world space, in Blender's view-space order: `right`, `up`, and
/// `backward` (toward the viewer). Workbench's studio lights are fixed in this space.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewBasis {
    right: DVec3,
    up: DVec3,
    backward: DVec3,
    camera_to_world: Option<DMat4>,
    world_to_camera: Option<DMat4>,
}

impl ViewBasis {
    pub(crate) fn from_camera(camera: &Camera) -> Result<Self> {
        let (forward, right, up) = view_axes(camera)?;
        let camera_matrices = (camera.stereo_mode == StereoMode::None).then(|| {
            let camera_to_world = DMat4::from_cols(
                right.extend(0.0),
                up.extend(0.0),
                (-forward).extend(0.0),
                camera.position.extend(1.0),
            );
            (camera_to_world, camera_to_world.inverse())
        });
        Ok(Self {
            right,
            up,
            backward: -forward,
            camera_to_world: camera_matrices.map(|(camera_to_world, _)| camera_to_world),
            world_to_camera: camera_matrices.map(|(_, world_to_camera)| world_to_camera),
        })
    }

    pub(crate) fn camera_matrices(&self) -> Option<(&DMat4, &DMat4)> {
        match (&self.camera_to_world, &self.world_to_camera) {
            (Some(camera_to_world), Some(world_to_camera)) => {
                Some((camera_to_world, world_to_camera))
            }
            _ => None,
        }
    }

    /// `direction`, given in world space, expressed in view space.
    pub(crate) fn direction_to_view(&self, direction: DVec3) -> DVec3 {
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
    if !triangle.smooth_shading {
        return point;
    }
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
    /// Whether `linear_rgba` stores associated RGB with additive camera-shape radiance.
    pub(crate) linear_rgba_associated: bool,
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
pub(super) struct CameraLensSample {
    aperture: glam::DVec2,
    focus_distance: f64,
}
impl CameraLensSample {
    pub(super) fn is_zero(self) -> bool {
        self.aperture == glam::DVec2::ZERO
    }
}
impl CameraFrame {
    pub(super) fn is_side_by_side(&self) -> bool {
        self.stereo_mode == StereoMode::SideBySide
    }
    pub(super) fn lens_sample(&self, sample_x: f64, sample_y: f64) -> Option<CameraLensSample> {
        let dof = self
            .depth_of_field
            .filter(|dof| dof.aperture_radius > 0.0)?;
        Some(CameraLensSample {
            aperture: concentric_disk(sample_x, sample_y, dof.aperture_blades)
                * dof.aperture_radius,
            focus_distance: dof.focus_distance,
        })
    }
}

pub(crate) struct CameraFrustum {
    corners: [DVec3; 8],
    planes: [FrustumPlane; 6],
}

#[derive(Clone, Copy)]
struct FrustumPlane {
    normal: DVec3,
    offset: f64,
}

impl FrustumPlane {
    fn distance(self, point: DVec3) -> f64 {
        self.normal.dot(point) + self.offset
    }
}

impl CameraFrustum {
    pub(crate) fn intersects_sphere(&self, center: DVec3, radius: f64) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.distance(center) >= -radius)
    }

    pub(crate) fn intersects_pyramid(&self, corners: [DVec3; 5]) -> bool {
        if self
            .planes
            .iter()
            .any(|plane| corners.iter().all(|corner| plane.distance(*corner) <= 0.0))
        {
            return false;
        }

        let apex = corners[0];
        let first = corners[1];
        let second = corners[2];
        let third = corners[3];
        let fourth = corners[4];
        let pyramid_planes = [
            frustum_plane(apex, second - apex, first - apex),
            frustum_plane(apex, third - apex, second - apex),
            frustum_plane(apex, fourth - apex, third - apex),
            frustum_plane(apex, first - apex, fourth - apex),
            frustum_plane(first, second - first, fourth - first),
        ];
        !pyramid_planes.iter().any(|plane| {
            self.corners
                .iter()
                .all(|corner| plane.distance(*corner) <= 0.0)
        })
    }
}

fn frustum_plane(point: DVec3, first: DVec3, second: DVec3) -> FrustumPlane {
    let normal = first.cross(second).normalize_or_zero();
    FrustumPlane {
        normal,
        offset: -normal.dot(point),
    }
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
// EEVEE's projected shape quad interpolates local coordinates on the 8-bit subpixel raster grid.
const CAMERA_LIGHT_SUBPIXEL_SCALE: f32 = 256.0;
const CAMERA_LIGHT_QUAD_UV: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
const CAMERA_LIGHT_QUAD_TRIANGLES: [[usize; 3]; 2] = [[0, 1, 2], [1, 3, 2]];
// Near/far clipping can add one vertex per plane to each source triangle.
const CAMERA_LIGHT_CLIPPED_TRIANGLE_VERTICES: usize = 5;
const CAMERA_LIGHT_MAX_TRIANGLES: usize = 6;
const CAMERA_LIGHT_MAX_VERTICES: usize = CAMERA_LIGHT_MAX_TRIANGLES * 3;

#[derive(Clone, Copy, Default)]
struct CameraLightClipVertex {
    clip: [f32; 2],
    w: f32,
    depth: f32,
    local: [f32; 2],
}

#[derive(Clone, Copy, Default)]
struct CameraLightProjectedVertex {
    screen: [f32; 2],
    inverse_w: f32,
    local: [f32; 2],
}

pub(super) struct CameraLightProjectedQuad {
    vertices: [CameraLightProjectedVertex; CAMERA_LIGHT_MAX_VERTICES],
    triangle_count: usize,
}

pub(super) struct CameraLightRasterQuad {
    screen: [[i64; 2]; CAMERA_LIGHT_MAX_VERTICES],
    triangle_count: usize,
}

impl CameraLightProjectedQuad {
    pub(super) fn rasterize(&self, jitter_x: f64, jitter_y: f64) -> CameraLightRasterQuad {
        let vertex_count = self.triangle_count * 3;
        let jitter_x = 0.5 - jitter_x as f32;
        let jitter_y = 0.5 - jitter_y as f32;
        CameraLightRasterQuad {
            screen: std::array::from_fn(|index| {
                if index < vertex_count {
                    [
                        snap_camera_light_coordinate(self.vertices[index].screen[0] + jitter_x),
                        snap_camera_light_coordinate(self.vertices[index].screen[1] + jitter_y),
                    ]
                } else {
                    [0; 2]
                }
            }),
            triangle_count: self.triangle_count,
        }
    }

    pub(super) fn contains_pixel(&self, raster: &CameraLightRasterQuad, x: u32, y: u32) -> bool {
        raster.contains_pixel(x, y, &self.vertices)
    }
}

impl CameraLightRasterQuad {
    fn contains_pixel(
        &self,
        x: u32,
        y: u32,
        vertices: &[CameraLightProjectedVertex; CAMERA_LIGHT_MAX_VERTICES],
    ) -> bool {
        let pixel = [i64::from(x) * 256 + 128, i64::from(y) * 256 + 128];
        (0..self.triangle_count).any(|triangle| {
            self.local_position(pixel, triangle, vertices)
                .is_some_and(|local| local[0] * local[0] + local[1] * local[1] <= 1.0)
        })
    }

    fn local_position(
        &self,
        pixel: [i64; 2],
        triangle: usize,
        vertices: &[CameraLightProjectedVertex; CAMERA_LIGHT_MAX_VERTICES],
    ) -> Option<[f32; 2]> {
        let first = triangle * 3;
        let second = first + 1;
        let third = first + 2;
        let edge_a = [
            self.screen[second][0] - self.screen[first][0],
            self.screen[second][1] - self.screen[first][1],
        ];
        let edge_b = [
            self.screen[third][0] - self.screen[first][0],
            self.screen[third][1] - self.screen[first][1],
        ];
        let point = [
            pixel[0] - self.screen[first][0],
            pixel[1] - self.screen[first][1],
        ];
        let cross = |left: [i64; 2], right: [i64; 2]| {
            i128::from(left[0]) * i128::from(right[1]) - i128::from(left[1]) * i128::from(right[0])
        };
        let determinant = cross(edge_a, edge_b);
        if determinant == 0 {
            return None;
        }
        let numerator_second = cross(point, edge_b);
        let numerator_third = cross(edge_a, point);
        let numerator_first = determinant - numerator_second - numerator_third;
        let orientation = determinant.signum();
        if [numerator_first, numerator_second, numerator_third]
            .into_iter()
            .any(|weight| weight * orientation < 0)
        {
            return None;
        }

        for (numerator, (start, end)) in [
            (numerator_first, (second, third)),
            (numerator_second, (third, first)),
            (numerator_third, (first, second)),
        ] {
            if numerator == 0 {
                let mut dx = self.screen[end][0] - self.screen[start][0];
                let mut dy = self.screen[end][1] - self.screen[start][1];
                if orientation < 0 {
                    dx = -dx;
                    dy = -dy;
                }
                if !(dy < 0 || (dy == 0 && dx < 0)) {
                    return None;
                }
            }
        }

        let denominator = determinant.abs() as f32;
        let weight_second = (numerator_second * orientation) as f32 / denominator;
        let weight_third = (numerator_third * orientation) as f32 / denominator;
        let weight_first = 1.0 - weight_second - weight_third;
        let perspective_weights = [
            weight_first * vertices[first].inverse_w,
            weight_second * vertices[second].inverse_w,
            weight_third * vertices[third].inverse_w,
        ];
        let weight_sum = perspective_weights[0] + perspective_weights[1] + perspective_weights[2];
        if !weight_sum.is_finite() || weight_sum <= f32::MIN_POSITIVE {
            return None;
        }
        Some([
            (perspective_weights[0] * vertices[first].local[0]
                + perspective_weights[1] * vertices[second].local[0]
                + perspective_weights[2] * vertices[third].local[0])
                / weight_sum,
            (perspective_weights[0] * vertices[first].local[1]
                + perspective_weights[1] * vertices[second].local[1]
                + perspective_weights[2] * vertices[third].local[1])
                / weight_sum,
        ])
    }
}

fn interpolate_camera_light_vertex(
    first: CameraLightClipVertex,
    second: CameraLightClipVertex,
    t: f32,
    clipped_depth: f32,
) -> CameraLightClipVertex {
    let interpolate = |a: f32, b: f32| a + (b - a) * t;
    CameraLightClipVertex {
        clip: [
            interpolate(first.clip[0], second.clip[0]),
            interpolate(first.clip[1], second.clip[1]),
        ],
        w: interpolate(first.w, second.w),
        depth: clipped_depth,
        local: [
            interpolate(first.local[0], second.local[0]),
            interpolate(first.local[1], second.local[1]),
        ],
    }
}

fn clip_camera_light_polygon(
    input: [CameraLightClipVertex; CAMERA_LIGHT_CLIPPED_TRIANGLE_VERTICES],
    input_count: usize,
    depth: f32,
    keep_greater: bool,
) -> (
    [CameraLightClipVertex; CAMERA_LIGHT_CLIPPED_TRIANGLE_VERTICES],
    usize,
) {
    let mut output = [CameraLightClipVertex::default(); CAMERA_LIGHT_CLIPPED_TRIANGLE_VERTICES];
    if input_count == 0 {
        return (output, 0);
    }
    let inside = |vertex: CameraLightClipVertex| {
        if keep_greater {
            vertex.depth >= depth
        } else {
            vertex.depth <= depth
        }
    };
    let strictly_inside = |vertex: CameraLightClipVertex| {
        if keep_greater {
            vertex.depth > depth
        } else {
            vertex.depth < depth
        }
    };
    let mut output_count = 0;
    let mut previous = input[input_count - 1];
    let mut previous_inside = inside(previous);
    for current in input.into_iter().take(input_count) {
        let current_inside = inside(current);
        if current_inside {
            if !previous_inside && strictly_inside(current) {
                let t = (depth - previous.depth) / (current.depth - previous.depth);
                output[output_count] = interpolate_camera_light_vertex(previous, current, t, depth);
                output_count += 1;
            }
            output[output_count] = current;
            output_count += 1;
        } else if previous_inside && strictly_inside(previous) {
            let t = (depth - previous.depth) / (current.depth - previous.depth);
            output[output_count] = interpolate_camera_light_vertex(previous, current, t, depth);
            output_count += 1;
        }
        previous = current;
        previous_inside = current_inside;
    }
    (output, output_count)
}

fn snap_camera_light_coordinate(value: f32) -> i64 {
    (value * CAMERA_LIGHT_SUBPIXEL_SCALE).round() as i64
}

fn camera_light_clip_vertices(
    camera_position: DVec3,
    frame: &CameraFrame,
    positions: &[DVec3; 4],
    lens_sample: Option<CameraLensSample>,
) -> Option<[CameraLightClipVertex; 4]> {
    let camera_position = Vec3::new(
        camera_position.x as f32,
        camera_position.y as f32,
        camera_position.z as f32,
    );
    let forward = Vec3::new(
        frame.forward.x as f32,
        frame.forward.y as f32,
        frame.forward.z as f32,
    );
    let right = Vec3::new(
        frame.right.x as f32,
        frame.right.y as f32,
        frame.right.z as f32,
    );
    let up = Vec3::new(frame.up.x as f32, frame.up.y as f32, frame.up.z as f32);
    let shift = [frame.shift[0] as f32, frame.shift[1] as f32];
    let mut vertices = [CameraLightClipVertex::default(); 4];

    for (index, position) in positions.iter().enumerate() {
        let point = Vec3::new(position.x as f32, position.y as f32, position.z as f32);
        let offset = point - camera_position;
        let depth = offset.dot(forward);
        let (clip, w) = match frame.projection {
            ProjectionFrame::Orthographic {
                half_width,
                half_height,
            } => {
                let lens_correction = lens_sample.map_or(glam::DVec2::ZERO, |sample| {
                    sample.aperture * (1.0 - f64::from(depth) / sample.focus_distance)
                });
                (
                    [
                        offset.dot(right) / half_width as f32
                            - 2.0 * shift[0]
                            - lens_correction.x as f32 / half_width as f32,
                        offset.dot(up) / half_height as f32
                            - 2.0 * shift[1]
                            - lens_correction.y as f32 / half_height as f32,
                    ],
                    1.0,
                )
            }
            ProjectionFrame::Perspective {
                horizontal_scale,
                vertical_scale,
            } => {
                let horizontal_scale = horizontal_scale as f32;
                let vertical_scale = vertical_scale as f32;
                let lens_correction = lens_sample.map_or(glam::DVec2::ZERO, |sample| {
                    sample.aperture * (f64::from(depth) / sample.focus_distance - 1.0)
                });
                (
                    [
                        offset.dot(right) / horizontal_scale - 2.0 * shift[0] * depth
                            + lens_correction.x as f32 / horizontal_scale,
                        offset.dot(up) / vertical_scale - 2.0 * shift[1] * depth
                            + lens_correction.y as f32 / vertical_scale,
                    ],
                    depth,
                )
            }
            ProjectionFrame::Equirectangular { .. }
            | ProjectionFrame::FisheyeEquidistant { .. }
            | ProjectionFrame::FisheyeEquisolid { .. } => return None,
        };
        if !depth.is_finite() || clip.iter().any(|value| !value.is_finite()) {
            return None;
        }
        vertices[index] = CameraLightClipVertex {
            clip,
            w,
            depth,
            local: CAMERA_LIGHT_QUAD_UV[index],
        };
    }
    Some(vertices)
}

pub(super) fn project_camera_light_quad(
    camera_position: DVec3,
    frame: &CameraFrame,
    width: u32,
    height: u32,
    positions: &[DVec3; 4],
    lens_sample: Option<CameraLensSample>,
) -> Option<CameraLightProjectedQuad> {
    let clip_vertices = camera_light_clip_vertices(camera_position, frame, positions, lens_sample)?;
    let near = frame.near as f32;
    let far = frame.far as f32;
    if !near.is_finite() || !far.is_finite() || near <= 0.0 || far <= near {
        return None;
    }
    let mut vertices = [CameraLightProjectedVertex::default(); CAMERA_LIGHT_MAX_VERTICES];
    let mut triangle_count = 0;
    for triangle in CAMERA_LIGHT_QUAD_TRIANGLES {
        let mut polygon =
            [CameraLightClipVertex::default(); CAMERA_LIGHT_CLIPPED_TRIANGLE_VERTICES];
        for (slot, vertex_index) in triangle.into_iter().enumerate() {
            polygon[slot] = clip_vertices[vertex_index];
        }
        let (polygon, count) = clip_camera_light_polygon(polygon, 3, near, true);
        let (polygon, count) = clip_camera_light_polygon(polygon, count, far, false);
        for fan in 1..count.saturating_sub(1) {
            if triangle_count == CAMERA_LIGHT_MAX_TRIANGLES {
                return None;
            }
            let output = &mut vertices[triangle_count * 3..triangle_count * 3 + 3];
            for (slot, vertex) in [polygon[0], polygon[fan], polygon[fan + 1]]
                .into_iter()
                .enumerate()
            {
                let inverse_w = 1.0 / vertex.w;
                let ndc = [vertex.clip[0] / vertex.w, vertex.clip[1] / vertex.w];
                let screen = [
                    ndc[0].midpoint(1.0) * width as f32,
                    (-ndc[1]).midpoint(1.0) * height as f32,
                ];
                if !inverse_w.is_finite() || screen.iter().any(|value| !value.is_finite()) {
                    return None;
                }
                output[slot] = CameraLightProjectedVertex {
                    screen,
                    inverse_w,
                    local: vertex.local,
                };
            }
            triangle_count += 1;
        }
    }

    (triangle_count > 0).then_some(CameraLightProjectedQuad {
        vertices,
        triangle_count,
    })
}

pub(super) fn camera_light_display_quad(
    camera_position: DVec3,
    frame: &CameraFrame,
    light_position: DVec3,
    light_radius: f64,
    soft_falloff: bool,
    width: u32,
    height: u32,
) -> Option<(DVec3, DVec3, [DVec3; 4], Option<CameraLightProjectedQuad>)> {
    let to_light = Vec3::new(
        light_position.x as f32,
        light_position.y as f32,
        light_position.z as f32,
    ) - Vec3::new(
        camera_position.x as f32,
        camera_position.y as f32,
        camera_position.z as f32,
    );
    let distance = to_light.length();
    let (plane_normal, right, up) = if distance > 1.0e-8 {
        let normal = to_light / distance;
        let up_vector = if normal.z.abs() < 0.999_99 {
            Vec3::Z
        } else {
            Vec3::X
        };
        let right = up_vector.cross(normal).normalize();
        (normal, right, normal.cross(right))
    } else {
        (
            Vec3::new(
                frame.forward.x as f32,
                frame.forward.y as f32,
                frame.forward.z as f32,
            ),
            Vec3::new(
                frame.right.x as f32,
                frame.right.y as f32,
                frame.right.z as f32,
            ),
            Vec3::new(frame.up.x as f32, frame.up.y as f32, frame.up.z as f32),
        )
    };

    let mut radius = (light_radius as f32).max(0.001);
    if !soft_falloff {
        if distance > radius {
            let ratio = radius / distance;
            radius *= 1.0 / (1.0 - ratio * ratio).max(1.0e-8).sqrt();
        } else {
            radius = frame.far as f32;
        }
    }
    if !radius.is_finite() {
        return None;
    }

    let center = Vec3::new(
        light_position.x as f32,
        light_position.y as f32,
        light_position.z as f32,
    );
    let positions = CAMERA_LIGHT_QUAD_UV.map(|coordinate| {
        let point = center + (right * coordinate[0] + up * coordinate[1]) * radius;
        DVec3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z))
    });
    let center = DVec3::new(
        f64::from(center.x),
        f64::from(center.y),
        f64::from(center.z),
    );
    let plane_normal = DVec3::new(
        f64::from(plane_normal.x),
        f64::from(plane_normal.y),
        f64::from(plane_normal.z),
    );
    let quad = project_camera_light_quad(camera_position, frame, width, height, &positions, None);
    Some((center, plane_normal, positions, quad))
}
#[cfg(test)]
mod camera_light_raster_tests {
    use std::error::Error;

    use glam::DVec3;

    use super::{
        Camera, CameraFrame, CameraLightProjectedQuad, CameraLightProjectedVertex, DepthOfField,
        Projection, StereoMode, camera_frame, camera_light_display_quad, camera_light_eye_index,
        project_camera_light_quad, ray_for_subpixel_in_frame,
    };

    fn camera(depth_of_field: Option<DepthOfField>) -> Camera {
        Camera {
            position: DVec3::ZERO,
            target: -DVec3::Z,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Perspective {
                lens_mm: 50.0,
                sensor_width_mm: 36.0,
            },
            shift: [0.0; 2],
            depth_of_field,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.0,
        }
    }

    fn projected_center_x(quad: &CameraLightProjectedQuad) -> f32 {
        assert_eq!(quad.triangle_count, 2);
        quad.vertices[..6]
            .iter()
            .map(|vertex| vertex.screen[0])
            .sum::<f32>()
            / 6.0
    }

    fn projected_quad_positions(center: DVec3, radius: f64) -> [DVec3; 4] {
        [
            center + DVec3::new(-radius, -radius, 0.0),
            center + DVec3::new(radius, -radius, 0.0),
            center + DVec3::new(-radius, radius, 0.0),
            center + DVec3::new(radius, radius, 0.0),
        ]
    }

    #[test]
    fn film_sample_translation_uses_pixel_corner_offsets() {
        let offsets = [
            0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0,
        ];
        let quad = CameraLightProjectedQuad {
            vertices: std::array::from_fn(|index| CameraLightProjectedVertex {
                screen: [1.25 + offsets[index], 2.75 + offsets[index]],
                inverse_w: 1.0,
                local: [0.0; 2],
            }),
            triangle_count: 2,
        };
        let centered = quad.rasterize(0.5, 0.5);
        let shifted = quad.rasterize(0.25, 0.75);
        assert_eq!(centered.screen[0], [320, 704]);
        assert_eq!(shifted.screen[0], [384, 640]);
    }

    #[test]
    fn sample_routing_keeps_the_anaglyph_eye() {
        assert_eq!(camera_light_eye_index(StereoMode::Anaglyph, 0, 100, 1), 1);
        assert_eq!(
            camera_light_eye_index(StereoMode::SideBySide, 49, 100, 0),
            0
        );
        assert_eq!(
            camera_light_eye_index(StereoMode::SideBySide, 50, 100, 0),
            1
        );
    }

    #[test]
    fn hard_falloff_expands_the_sphere_but_soft_falloff_keeps_its_disk()
    -> Result<(), Box<dyn Error>> {
        let camera = camera(None);
        let frame = camera_frame(&camera, 100, 100)?;
        let light = DVec3::new(0.0, 0.0, -10.0);
        let (_, _, soft, _) =
            camera_light_display_quad(camera.position, &frame, light, 1.0, true, 100, 100)
                .ok_or("soft-falloff display quad was unavailable")?;
        let (_, _, hard, _) =
            camera_light_display_quad(camera.position, &frame, light, 1.0, false, 100, 100)
                .ok_or("hard-falloff display quad was unavailable")?;
        let center = DVec3::new(0.0, 0.0, -10.0);
        let radius = |positions: [DVec3; 4]| (positions[0] - center).length() / 2.0_f64.sqrt();
        let soft_radius = radius(soft);
        let hard_radius = radius(hard);
        assert!((soft_radius - 1.0).abs() < 1.0e-6);
        assert!(hard_radius > soft_radius);
        Ok(())
    }

    #[test]
    fn camera_inside_off_axis_light_keeps_the_near_clipped_disk() -> Result<(), Box<dyn Error>> {
        let camera = camera(None);
        let frame = camera_frame(&camera, 100, 100)?;
        let (_, _, _, projected) = camera_light_display_quad(
            camera.position,
            &frame,
            DVec3::new(0.5, 0.0, -1.0),
            2.0,
            false,
            100,
            100,
        )
        .ok_or("inside light display geometry was unavailable")?;
        let projected = projected.ok_or("inside light's visible quad was clipped away")?;
        let raster = projected.rasterize(0.5, 0.5);
        assert!(projected.contains_pixel(&raster, 50, 49));
        Ok(())
    }

    #[test]
    fn projected_dof_coverage_matches_the_aperture_ray() -> Result<(), Box<dyn Error>> {
        let camera = camera(Some(DepthOfField {
            focus_distance: 5.0,
            aperture_radius: 0.5,
            aperture_blades: 0,
        }));
        let frame: CameraFrame = camera_frame(&camera, 100, 100)?;
        let positions = projected_quad_positions(DVec3::new(0.0, 0.0, -2.0), 0.02);
        let pinhole =
            project_camera_light_quad(camera.position, &frame, 100, 100, &positions, None)
                .ok_or("pinhole quad projection was unavailable")?;
        let lens_sample = frame
            .lens_sample(0.75, 0.5)
            .ok_or("depth-of-field lens sample was unavailable")?;
        let aperture = project_camera_light_quad(
            camera.position,
            &frame,
            100,
            100,
            &positions,
            Some(lens_sample),
        )
        .ok_or("aperture quad projection was unavailable")?;
        assert!(projected_center_x(&aperture) < projected_center_x(&pinhole) - 5.0);

        let raster = aperture.rasterize(0.75, 0.5);
        let inside_ray =
            ray_for_subpixel_in_frame(&frame, camera.position, 100, 100, 39, 49, 0.75, 0.5)?
                .ok_or("inside aperture sample had no ray")?;
        let inside_distance = (-2.0 - inside_ray.origin.z) / inside_ray.direction.z;
        let inside_hit = inside_ray.origin + inside_ray.direction * inside_distance;
        assert!(inside_hit.x * inside_hit.x + inside_hit.y * inside_hit.y < 0.02_f64.powi(2));
        assert!(aperture.contains_pixel(&raster, 39, 49));

        let outside_ray =
            ray_for_subpixel_in_frame(&frame, camera.position, 100, 100, 50, 49, 0.75, 0.5)?
                .ok_or("outside aperture sample had no ray")?;
        let outside_distance = (-2.0 - outside_ray.origin.z) / outside_ray.direction.z;
        let outside_hit = outside_ray.origin + outside_ray.direction * outside_distance;
        assert!(outside_hit.x * outside_hit.x + outside_hit.y * outside_hit.y > 0.02_f64.powi(2));
        assert!(!aperture.contains_pixel(&raster, 50, 49));
        Ok(())
    }
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
    holdout_weight: f64,
    subsurface: Option<EeveeSubsurfaceSample>,
    eevee_raytrace: Option<EeveeRaytraceSample>,
    eevee_dither: Option<EeveeDitherSample>,
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

fn camera_matrices_for_raster(camera: &Camera) -> Result<[(DMat4, DMat4); 2]> {
    if camera.stereo_mode == StereoMode::SideBySide {
        let left = camera_matrices_for_camera(&camera_eye(camera, -1.0))?;
        let right = camera_matrices_for_camera(&camera_eye(camera, 1.0))?;
        Ok([left, right])
    } else {
        let matrices = camera_matrices_for_camera(camera)?;
        Ok([matrices, matrices])
    }
}

fn camera_matrices_for_camera(camera: &Camera) -> Result<(DMat4, DMat4)> {
    let view = ViewBasis::from_camera(camera)?;
    view.camera_matrices()
        .map(|(camera_to_world, world_to_camera)| (*camera_to_world, *world_to_camera))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "camera matrices are unavailable for this eye",
            )
        })
}

#[derive(Clone, Copy)]
struct EeveeWorldRayFrame {
    view_positions: [Vec3; 3],
    right: Vec3,
    up: Vec3,
    back: Vec3,
}

impl EeveeWorldRayFrame {
    fn new(camera: &Camera, frame: &CameraFrame, width: u32, height: u32) -> Option<Self> {
        let right = Vec3::new(
            frame.right.x as f32,
            frame.right.y as f32,
            frame.right.z as f32,
        );
        let up = Vec3::new(frame.up.x as f32, frame.up.y as f32, frame.up.z as f32);
        let back = Vec3::new(
            -frame.forward.x as f32,
            -frame.forward.y as f32,
            -frame.forward.z as f32,
        );
        let view_positions = match (camera.projection, frame.projection) {
            (Projection::Orthographic { .. }, ProjectionFrame::Orthographic { .. }) => {
                [Vec3::new(0.0, 0.0, -1.0); 3]
            }
            (
                Projection::Perspective {
                    lens_mm,
                    sensor_width_mm,
                },
                ProjectionFrame::Perspective { .. },
            ) => {
                let view_width = if frame.stereo_mode == StereoMode::SideBySide {
                    width / 2
                } else {
                    width
                };
                let near = camera.near as f32;
                let far = camera.far as f32;
                let pixel_size = (sensor_width_mm as f32 * near) / lens_mm as f32;
                let pixel_size = pixel_size / view_width as f32;
                let shift_x = camera.shift[0] as f32 * view_width as f32;
                let shift_y = camera.shift[1] as f32 * view_width as f32;
                let left = (-0.5 * view_width as f32 + shift_x) * pixel_size;
                let right = (0.5 * view_width as f32 + shift_x) * pixel_size;
                let bottom = (-0.5 * height as f32 + shift_y) * pixel_size;
                let top = (0.5 * height as f32 + shift_y) * pixel_size;
                let width = right - left;
                let height = top - bottom;
                let depth = far - near;
                let projection = Mat4::from_cols(
                    Vec4::new(2.0 * near / width, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, 2.0 * near / height, 0.0, 0.0),
                    Vec4::new(
                        (right + left) / width,
                        (top + bottom) / height,
                        -(far + near) / depth,
                        -1.0,
                    ),
                    Vec4::new(0.0, 0.0, -(2.0 * far * near) / depth, 0.0),
                );
                let inverse_projection = projection.inverse();
                if !inverse_projection.is_finite() {
                    return None;
                }
                // EEVEE inverse-projects the fullscreen-triangle vertices before GPU interpolation.
                [
                    Vec4::new(-1.0, -1.0, 1.0, 1.0),
                    Vec4::new(3.0, -1.0, 1.0, 1.0),
                    Vec4::new(-1.0, 3.0, 1.0, 1.0),
                ]
                .map(|ndc| {
                    let position = inverse_projection * ndc;
                    position.truncate() / position.w
                })
            }
            _ => return None,
        };
        Some(Self {
            view_positions,
            right,
            up,
            back,
        })
    }

    fn direction(
        self,
        camera: &Camera,
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        sample_x: f64,
        sample_y: f64,
    ) -> DVec3 {
        let (view_x, view_width) = if camera.stereo_mode == StereoMode::SideBySide {
            let split = width / 2;
            if x < split {
                (x, split)
            } else {
                (x - split, width - split)
            }
        } else {
            (x, width)
        };
        // Evaluate the fullscreen triangle's f32 attribute plane using raster-space gradients.
        let pixel_x = view_x as f32 + sample_x as f32;
        let pixel_y = height as f32 - y as f32 - sample_y as f32;
        let slope_x = (self.view_positions[1] - self.view_positions[0]) / (2.0 * view_width as f32);
        let slope_y = (self.view_positions[2] - self.view_positions[0]) / (2.0 * height as f32);
        let view_position = self.view_positions[0] + slope_x * pixel_x + slope_y * pixel_y;
        let view_incident = (-view_position).normalize();
        let world_incident =
            self.right * view_incident.x + self.up * view_incident.y + self.back * view_incident.z;
        DVec3::new(
            f64::from(-world_incident.x),
            f64::from(-world_incident.y),
            f64::from(-world_incident.z),
        )
    }
}
/// The scene world's radiance along a view ray. Realtime Beauty shows it where no surface covers a sample, as EEVEE
/// shows its world, and modes without one show their constant background colour.
pub(crate) type WorldRadiance<'a> = &'a (dyn Fn(DVec3) -> Result<DVec3> + Send + Sync);

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

struct RasterSamplesRequest<'scene, 'context, 'lights, 'volume, 'volume_scene, 'borrow> {
    triangles: &'scene [Triangle],
    camera: &'scene Camera,
    width: u32,
    height: u32,
    mode: Mode,
    background: [u8; 4],
    world: Option<WorldRadiance<'scene>>,
    eevee_world_rays: bool,
    samples: u32,
    seed: u32,
    filter_size: f64,
    exposure_scale: f32,
    shader: Option<&'scene dyn SurfaceShader>,
    raytrace_settings: Option<EeveeRayTracingSettings>,
    prepared_materials: &'scene [crate::shader::PreparedMaterial],
    camera_lights: Option<super::path::RealtimeCameraLightView<'context, 'lights>>,
    view_index: usize,
    band_count: usize,
    volume_pass: Option<&'borrow mut super::path::RealtimeVolumePass<'volume, 'volume_scene>>,
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
    let raytrace_settings = match shader {
        Some(shader) => RasterShader::Main(shader).eevee_raytracing_settings()?,
        None => None,
    };
    rasterize_with_samples_for_eye(RasterSamplesRequest {
        triangles,
        camera,
        width,
        height,
        mode,
        background,
        world,
        eevee_world_rays: false,
        samples,
        seed,
        filter_size,
        exposure_scale: 1.0,
        shader,
        raytrace_settings,
        prepared_materials,
        camera_lights: camera_lights.map(|lights| lights.view(0)),
        view_index: 0,
        band_count: std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        volume_pass,
    })
}

#[derive(Clone, Copy)]
struct CachedRasterHit {
    ray: Option<Ray>,
    hit: Option<VisibleHit>,
}

type RasterSampleOutput = (
    [f64; 4],
    [f64; 4],
    Option<EeveeSubsurfacePixel>,
    Option<EeveeRaytraceSample>,
);

struct RasterPixelSample {
    cached: Option<CachedRasterHit>,
    index: u32,
    count: u32,
    needs_eevee_gbuffer_context: bool,
    gbuffer_sample_offset: Option<[f32; 3]>,
    gbuffer_closure_sample: Option<f32>,
    eevee_raytrace_feedback: EeveeRaytraceFeedback,
    x: u32,
    y: u32,
    sample_x: f64,
    sample_y: f64,
}
#[derive(Clone, Copy)]
struct FilmSampleJitter {
    index: u32,
    x: f64,
    y: f64,
    gbuffer_sample_offset: Option<[f32; 3]>,
    gbuffer_closure_sample: Option<f32>,
}

impl FilmSampleJitter {
    fn new(index: u32, x: f64, y: f64, needs_eevee_gbuffer_context: bool) -> Self {
        let (gbuffer_sample_offset, gbuffer_closure_sample) = if needs_eevee_gbuffer_context {
            let source_sample = u64::from(index) + 1;
            (
                Some([
                    halton_radical_inverse(11, source_sample) as f32,
                    halton_radical_inverse(13, source_sample) as f32,
                    halton_radical_inverse(17, source_sample) as f32,
                ]),
                Some(halton_radical_inverse(3, source_sample) as f32),
            )
        } else {
            (None, None)
        };
        Self {
            index,
            x,
            y,
            gbuffer_sample_offset,
            gbuffer_closure_sample,
        }
    }
}

struct FilmBandWorker<'a> {
    candidates: Vec<usize>,
    shader: Option<Box<dyn SurfaceShaderWorker + Send + 'a>>,
    volume_scratch: super::path::RealtimeVolumeScratch,
    warnings: std::cell::RefCell<Vec<serde_json::Value>>,
    merged_warning_count: usize,
}

impl FilmBandWorker<'_> {
    fn take_new_warnings(&mut self) -> Vec<serde_json::Value> {
        let recorded = self.warnings.borrow();
        let warnings = recorded[self.merged_warning_count..].to_vec();
        self.merged_warning_count = recorded.len();
        warnings
    }
}

struct FilmCenterBandBuffers<'a> {
    ids: &'a mut [u32],
    depths: &'a mut [f32],
    elements: &'a mut [u32],
    normals: &'a mut [[f32; 3]],
    albedo: &'a mut [[f32; 4]],
    ambient_occlusion: &'a mut [f32],
}

struct FilmCenterRenderContext<'scene> {
    triangles: &'scene [Triangle],
    prepared: &'scene [PreparedTriangle],
    visibility: &'scene super::bvh::Bvh,
    ao_bvh: Option<&'scene super::bvh::Bvh>,
    frame: &'scene CameraFrame,
    camera: &'scene Camera,
    camera_matrices: &'scene [(DMat4, DMat4); 2],
    view_index: usize,
    width: u32,
    height: u32,
    width_usize: usize,
    prepared_materials: &'scene [crate::shader::PreparedMaterial],
    has_volume_only: bool,
}

impl FilmCenterRenderContext<'_> {
    fn render_rows(
        &self,
        rows: std::ops::Range<usize>,
        buffers: &mut FilmCenterBandBuffers<'_>,
        candidates: &mut Vec<usize>,
        shader: Option<RasterShader<'_>>,
    ) -> Result<()> {
        let center_sample = FilmSampleJitter::new(
            0,
            0.5,
            0.5,
            shader.is_some_and(|shader| shader.needs_eevee_gbuffer_context()),
        );
        for row in rows.clone() {
            let y = u32::try_from(row).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "pixel row exceeds platform limits",
                )
            })?;
            for x in 0..self.width {
                let x_usize = usize::try_from(x).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "pixel column exceeds platform limits",
                    )
                })?;
                let pixel = (row - rows.start)
                    .checked_mul(self.width_usize)
                    .and_then(|index| index.checked_add(x_usize))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "pixel index exceeds platform limits",
                        )
                    })?;
                let center_ray = ray_for_pixel_in_frame(
                    self.frame,
                    self.camera.position,
                    self.width,
                    self.height,
                    x,
                    y,
                )?;
                let hit_context = if shader.is_some() {
                    Some(raster_hit_context(
                        self.frame,
                        self.camera,
                        self.width,
                        self.height,
                        &RasterPixelSample {
                            cached: None,
                            index: 0,
                            count: 1,
                            needs_eevee_gbuffer_context: shader
                                .is_some_and(|shader| shader.needs_eevee_gbuffer_context()),
                            gbuffer_sample_offset: center_sample.gbuffer_sample_offset,
                            gbuffer_closure_sample: center_sample.gbuffer_closure_sample,
                            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
                            x,
                            y,
                            sample_x: 0.5,
                            sample_y: 0.5,
                        },
                        self.camera_matrices,
                        self.view_index,
                    )?)
                } else {
                    None
                };
                let center_hit = match center_ray {
                    Some(ray) => nearest_visible_hit(
                        self.prepared,
                        self.visibility,
                        candidates,
                        ray,
                        self.frame,
                        shader,
                        hit_context.as_ref(),
                        None,
                    )?,
                    None => None,
                };
                if let Some(visible_hit) = center_hit {
                    let triangle = &self.prepared[visible_hit.triangle_index];
                    let hit = visible_hit.hit;
                    buffers.ids[pixel] = triangle.source.object_index;
                    buffers.depths[pixel] = depth_to_f32(hit.depth);
                    buffers.elements[pixel] = triangle.source.element_index;
                    buffers.normals[pixel] = shading_normal(triangle, hit)
                        .to_array()
                        .map(|value| value as f32);
                    buffers.albedo[pixel] = linear_color_to_f32(triangle.source.color);
                    if let (Some(bvh), Some(center_ray)) = (self.ao_bvh, center_ray) {
                        let geometric = if triangle.normal.dot(hit.direction) > 0.0 {
                            -triangle.normal
                        } else {
                            triangle.normal
                        };
                        let mut shading = shading_normal(triangle, hit);
                        if shading.dot(geometric) < 0.0 {
                            shading = -shading;
                        }
                        let point = center_ray.origin + center_ray.direction * hit.distance;
                        let visibility = ambient_visibility(
                            bvh,
                            self.triangles,
                            self.prepared_materials,
                            point,
                            geometric,
                            shading,
                            1.0,
                            self.has_volume_only,
                        );
                        buffers.ambient_occlusion[pixel] =
                            linear_color_to_f32([visibility, visibility, visibility, 1.0])[0];
                    }
                }
            }
        }
        Ok(())
    }
}

fn make_film_band_workers(
    count: usize,
    shader: Option<&dyn SurfaceShader>,
) -> Result<Vec<FilmBandWorker<'_>>> {
    let mut workers = Vec::with_capacity(count);
    for _ in 0..count {
        let worker_shader = if let Some(shader) = shader {
            let Some(worker) = shader.worker()? else {
                return Ok(Vec::new());
            };
            Some(worker)
        } else {
            None
        };
        workers.push(FilmBandWorker {
            candidates: Vec::new(),
            shader: worker_shader,
            volume_scratch: super::path::RealtimeVolumeScratch::default(),
            warnings: std::cell::RefCell::new(Vec::new()),
            merged_warning_count: 0,
        });
    }
    Ok(workers)
}

fn render_film_center_bands<'worker>(
    context: &FilmCenterRenderContext<'_>,
    bands: &[std::ops::Range<usize>],
    workers: Vec<FilmBandWorker<'worker>>,
    buffers: FilmCenterBandBuffers<'_>,
) -> Result<Vec<FilmBandWorker<'worker>>> {
    std::thread::scope(|scope| {
        let mut ids_rest = buffers.ids;
        let mut depths_rest = buffers.depths;
        let mut elements_rest = buffers.elements;
        let mut normals_rest = buffers.normals;
        let mut albedo_rest = buffers.albedo;
        let mut ambient_occlusion_rest = buffers.ambient_occlusion;
        let mut handles = Vec::with_capacity(bands.len());
        for (rows, mut worker) in bands.iter().cloned().zip(workers) {
            let band_len = rows.len() * context.width_usize;
            let (ids, ids_tail) = std::mem::take(&mut ids_rest).split_at_mut(band_len);
            let (depths, depths_tail) = std::mem::take(&mut depths_rest).split_at_mut(band_len);
            let (elements, elements_tail) =
                std::mem::take(&mut elements_rest).split_at_mut(band_len);
            let (normals, normals_tail) = std::mem::take(&mut normals_rest).split_at_mut(band_len);
            let (albedo, albedo_tail) = std::mem::take(&mut albedo_rest).split_at_mut(band_len);
            let (ambient_occlusion, ambient_occlusion_tail) =
                std::mem::take(&mut ambient_occlusion_rest).split_at_mut(band_len);
            ids_rest = ids_tail;
            depths_rest = depths_tail;
            elements_rest = elements_tail;
            normals_rest = normals_tail;
            albedo_rest = albedo_tail;
            ambient_occlusion_rest = ambient_occlusion_tail;
            handles.push(
                std::thread::Builder::new()
                    .stack_size(super::path::WORKER_STACK_BYTES)
                    .spawn_scoped(scope, move || {
                        let shader = worker
                            .shader
                            .as_deref()
                            .map(|shader| RasterShader::Worker(shader, &worker.warnings));
                        let mut buffers = FilmCenterBandBuffers {
                            ids,
                            depths,
                            elements,
                            normals,
                            albedo,
                            ambient_occlusion,
                        };
                        let result =
                            context.render_rows(rows, &mut buffers, &mut worker.candidates, shader);
                        (result, worker)
                    }),
            );
        }
        let mut workers = Vec::with_capacity(handles.len());
        let mut first_error = None;
        for handle in handles {
            match handle {
                Ok(handle) => match handle.join() {
                    Ok((Ok(()), worker)) => workers.push(worker),
                    Ok((Err(error), worker)) => {
                        workers.push(worker);
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                    Err(_) if first_error.is_none() => {
                        first_error = Some(PotError::new(
                            ErrorCode::InternalError,
                            "render worker panicked",
                        ));
                    }
                    Err(_) => {}
                },
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(PotError::new(
                            ErrorCode::InternalError,
                            format!("cannot start a render worker: {error}"),
                        ));
                    }
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(workers),
        }
    })
}

struct FilmSampleRenderContext<'scene, 'light_context, 'lights> {
    triangles: &'scene [PreparedTriangle],
    visibility: &'scene super::bvh::Bvh,
    frame: &'scene CameraFrame,
    camera: &'scene Camera,
    camera_matrices: &'scene [(DMat4, DMat4); 2],
    width: u32,
    width_usize: usize,
    height: u32,
    mode: Mode,
    background: [f64; 4],
    world: Option<WorldRadiance<'scene>>,
    eevee_world_ray: Option<&'scene EeveeWorldRayFrame>,
    camera_lights: Option<super::path::RealtimeCameraLightView<'light_context, 'lights>>,
    camera_light_raster_samples: Option<&'scene super::path::CameraLightRasterSamples>,
    view_index: usize,
    sample_count: u32,
    linear_rgba_associated: bool,
}

impl FilmSampleRenderContext<'_, '_, '_> {
    fn render_rows(
        &self,
        rows: std::ops::Range<usize>,
        mut sample_color: Option<&mut [[f32; 4]]>,
        mut sample_display: Option<&mut [[f32; 4]]>,
        sample_subsurface: &mut [Option<EeveeSubsurfacePixel>],
        mut raytrace_samples: Option<&mut [Option<EeveeRaytraceSample>]>,
        raytrace_layer: Option<EeveeRaytraceLayer>,
        sample: FilmSampleJitter,
        sample_offset_seed: Option<u32>,
        raytrace_feedback: Option<&[EeveeRaytraceFeedback]>,
        candidates: &mut Vec<usize>,
        shader: Option<RasterShader<'_>>,
        volume_pass: &mut Option<&mut super::path::RealtimeVolumePass<'_, '_>>,
    ) -> Result<()> {
        // Source feedback is copied from deferred output before world, volume, and forward passes.
        let feedback_layer = raytrace_layer.is_some();
        let mut no_volume_pass = None;
        for row in rows.clone() {
            let y = u32::try_from(row).map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "pixel row exceeds platform limits",
                )
            })?;
            for x in 0..self.width {
                let x_usize = usize::try_from(x).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "pixel column exceeds platform limits",
                    )
                })?;
                let local_pixel = (row - rows.start)
                    .checked_mul(self.width_usize)
                    .and_then(|index| index.checked_add(x_usize))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "pixel index exceeds platform limits",
                        )
                    })?;
                let global_pixel = row * self.width_usize + x_usize;
                let (sample_x, sample_y) = sample_offset_seed
                    .map_or((sample.x, sample.y), |seed| {
                        sample_offset(seed, x, y, sample.index)
                    });
                let raytrace_feedback = raytrace_feedback
                    .and_then(|pixels| pixels.get(global_pixel))
                    .copied()
                    .unwrap_or_default();
                let (sample_straight, sample_linear, subsurface, eevee_raytrace) =
                    RasterSampleContext {
                        triangles: self.triangles,
                        visibility: self.visibility,
                        camera: self.camera,
                        camera_matrices: self.camera_matrices,
                        width: self.width,
                        height: self.height,
                        mode: self.mode,
                        candidates,
                        frame: self.frame,
                        background: if feedback_layer {
                            [0.0, 0.0, 0.0, 1.0]
                        } else {
                            self.background
                        },
                        world: if feedback_layer { None } else { self.world },
                        eevee_world_ray: if feedback_layer {
                            None
                        } else {
                            self.eevee_world_ray
                        },
                        shader,
                        camera_lights: self.camera_lights,
                        camera_light_raster_samples: self.camera_light_raster_samples,
                        view_index: self.view_index,
                        volume_pass: if feedback_layer {
                            &mut no_volume_pass
                        } else {
                            volume_pass
                        },
                        linear_rgba_associated: feedback_layer || self.linear_rgba_associated,
                        raytrace_layer,
                    }
                    .shade(&RasterPixelSample {
                        cached: None,
                        index: sample.index,
                        count: self.sample_count,
                        needs_eevee_gbuffer_context: shader
                            .is_some_and(|shader| shader.needs_eevee_gbuffer_context()),
                        gbuffer_sample_offset: sample.gbuffer_sample_offset,
                        gbuffer_closure_sample: sample.gbuffer_closure_sample,
                        eevee_raytrace_feedback: raytrace_feedback,
                        x,
                        y,
                        sample_x,
                        sample_y,
                    })?;
                if let Some(sample_color) = sample_color.as_deref_mut() {
                    sample_color[local_pixel] = sample_linear.map(|value| value as f32);
                }
                if let Some(sample_display) = sample_display.as_deref_mut() {
                    sample_display[local_pixel] = sample_straight.map(|value| value as f32);
                }
                sample_subsurface[local_pixel] = subsurface;
                if let Some(raytrace_samples) = raytrace_samples.as_deref_mut() {
                    raytrace_samples[local_pixel] = eevee_raytrace;
                }
            }
        }
        Ok(())
    }
}

struct RasterSampleContext<
    'scene,
    'light_context,
    'lights,
    'volume_pass,
    'volume_scene,
    'pass_borrow,
    'borrow,
> {
    triangles: &'scene [PreparedTriangle],
    visibility: &'scene super::bvh::Bvh,
    candidates: &'borrow mut Vec<usize>,
    frame: &'scene CameraFrame,
    camera: &'scene Camera,
    camera_matrices: &'scene [(DMat4, DMat4); 2],
    width: u32,
    height: u32,
    mode: Mode,
    background: [f64; 4],
    world: Option<WorldRadiance<'scene>>,
    eevee_world_ray: Option<&'scene EeveeWorldRayFrame>,
    shader: Option<RasterShader<'scene>>,
    camera_lights: Option<super::path::RealtimeCameraLightView<'light_context, 'lights>>,
    camera_light_raster_samples: Option<&'scene super::path::CameraLightRasterSamples>,
    view_index: usize,
    volume_pass: &'borrow mut Option<
        &'pass_borrow mut super::path::RealtimeVolumePass<'volume_pass, 'volume_scene>,
    >,
    linear_rgba_associated: bool,
    raytrace_layer: Option<EeveeRaytraceLayer>,
}

impl RasterSampleContext<'_, '_, '_, '_, '_, '_, '_> {
    fn shade(&mut self, sample: &RasterPixelSample) -> Result<RasterSampleOutput> {
        let mut hit_context = if self.shader.is_some() {
            Some(raster_hit_context(
                self.frame,
                self.camera,
                self.width,
                self.height,
                sample,
                self.camera_matrices,
                self.view_index,
            )?)
        } else {
            None
        };
        if let Some(hit_context) = hit_context.as_mut() {
            hit_context.raytrace_layer = self.raytrace_layer;
        }
        let (sample_ray, cached_hit, uncached_hit) = if let Some(cached) = sample.cached.as_ref() {
            (cached.ray, cached.hit.as_ref(), None)
        } else {
            let ray = ray_for_subpixel_in_frame(
                self.frame,
                self.camera.position,
                self.width,
                self.height,
                sample.x,
                sample.y,
                sample.sample_x,
                sample.sample_y,
            )?;
            let hit = match ray {
                Some(ray) => nearest_visible_hit(
                    self.triangles,
                    self.visibility,
                    self.candidates,
                    ray,
                    self.frame,
                    self.shader,
                    hit_context.as_ref(),
                    self.raytrace_layer,
                )?,
                None => None,
            };
            (ray, None, hit)
        };
        let sample_hit = cached_hit.or(uncached_hit.as_ref());
        if matches!(self.mode, Mode::Wire) {
            let color = sample_hit.map_or(self.background, |visible_hit| {
                let triangle = &self.triangles[visible_hit.triangle_index];
                if is_wire_edge(
                    triangle,
                    visible_hit.hit,
                    self.frame,
                    self.width,
                    self.height,
                ) {
                    triangle.source.color
                } else {
                    self.background
                }
            });
            return Ok((color, color, None, None));
        }
        let Some(sample_ray) = sample_ray else {
            let color = sample_background(self.background, self.world, None)?;
            return Ok((
                color,
                if self.linear_rgba_associated {
                    associated_rgba(color)
                } else {
                    color
                },
                None,
                None,
            ));
        };
        self.shade_surface(sample, sample_ray, sample_hit, hit_context.as_ref())
    }

    fn shade_surface(
        &mut self,
        sample: &RasterPixelSample,
        sample_ray: Ray,
        sample_hit: Option<&VisibleHit>,
        hit_context: Option<&RasterHitContext<'_>>,
    ) -> Result<RasterSampleOutput> {
        let eye_index = camera_light_eye_index(
            self.camera.stereo_mode,
            sample.x,
            self.width,
            self.view_index,
        );
        let mut volume_sample = self
            .volume_pass
            .as_deref_mut()
            .map(|volume_pass| {
                volume_pass.begin_sample(&super::path::RealtimeVolumeSampleInfo {
                    camera: self.camera,
                    frame: self.frame,
                    width: self.width,
                    height: self.height,
                    x: sample.x,
                    y: sample.y,
                    sample_x: sample.sample_x,
                    sample_y: sample.sample_y,
                    ray: (sample_ray.origin, sample_ray.direction),
                    sample_index: sample.index,
                    sample_count: sample.count,
                    view_index: eye_index,
                    camera_matrices: self.camera_matrices.get(eye_index).copied(),
                })
            })
            .transpose()?;
        let camera_light_view = self.camera_lights.map(|lights| {
            let eye = camera_light_eye_index(
                self.camera.stereo_mode,
                sample.x,
                self.width,
                self.view_index,
            );
            let pixel_x = if self.camera.stereo_mode == StereoMode::SideBySide
                && sample.x >= self.width / 2
            {
                sample.x - self.width / 2
            } else {
                sample.x
            };
            lights.for_eye(eye).at_pixel(
                self.camera_light_raster_samples,
                sample.index,
                sample.sample_x,
                sample.sample_y,
                pixel_x,
                sample.y,
            )
        });
        let world_direction = self
            .eevee_world_ray
            .map_or(sample_ray.direction, |world_ray| {
                world_ray.direction(
                    self.camera,
                    self.width,
                    self.height,
                    sample.x,
                    sample.y,
                    sample.sample_x,
                    sample.sample_y,
                )
            });
        let color = composite_sample_with_association(
            self.triangles,
            self.visibility,
            self.candidates,
            sample_ray,
            world_direction,
            self.frame,
            self.camera,
            self.background,
            self.world,
            self.shader,
            hit_context,
            sample_hit,
            camera_light_view,
            volume_sample.as_mut(),
            self.raytrace_layer,
        )?;
        let linear = if self.linear_rgba_associated {
            color.associated
        } else {
            color.straight
        };
        Ok((
            color.straight,
            linear,
            color.subsurface,
            color.eevee_raytrace,
        ))
    }
}

struct EeveeTemporalHistory {
    radiance: Vec<[f32; 3]>,
    variance: Vec<f32>,
    tile_mask: Vec<u8>,
    valid: bool,
    view_projection: Option<Mat4>,
}

struct EeveeRaytracePassOutput {
    feedback: Vec<EeveeRaytraceFeedback>,
    depth: Vec<f32>,
}

fn eevee_temporal_histories(
    pixel_count: usize,
    width: u32,
    height: u32,
    plan: denoise::StagePlan,
) -> Result<Vec<EeveeTemporalHistory>> {
    let mut histories = Vec::new();
    histories.try_reserve_exact(3).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EEVEE temporal history allocation failed",
        )
    })?;
    for _ in 0..3 {
        histories.push(EeveeTemporalHistory {
            radiance: if plan.temporal {
                allocate_filled(pixel_count, [0.0; 3], "EEVEE radiance history")?
            } else {
                Vec::new()
            },
            variance: if plan.temporal {
                allocate_filled(pixel_count, 0.0, "EEVEE variance history")?
            } else {
                Vec::new()
            },
            tile_mask: if plan.temporal {
                allocate_filled(
                    width.div_ceil(8) as usize * height.div_ceil(8) as usize,
                    0_u8,
                    "EEVEE tile-mask history",
                )?
            } else {
                Vec::new()
            },
            valid: false,
            view_projection: None,
        });
    }
    Ok(histories)
}

struct EeveeRaytracedSamplesRequest<
    'borrow,
    'scene,
    'light_context,
    'lights,
    'shader,
    'volume_ref,
    'volume_frame,
    'volume_scene,
> {
    render: &'borrow FilmSampleRenderContext<'scene, 'light_context, 'lights>,
    settings: EeveeRayTracingSettings,
    seed: u32,
    film: Option<FilmFilter>,
    subsurface_frame: Option<EeveeSubsurfaceFrame>,
    shader: Option<RasterShader<'shader>>,
    candidates: &'borrow mut Vec<usize>,
    volume_pass: &'borrow mut Option<
        &'volume_ref mut super::path::RealtimeVolumePass<'volume_frame, 'volume_scene>,
    >,
}

fn rasterize_eevee_raytraced_samples(
    mut output: RasterOutput,
    request: EeveeRaytracedSamplesRequest<'_, '_, '_, '_, '_, '_, '_, '_>,
) -> Result<RasterOutput> {
    let EeveeRaytracedSamplesRequest {
        render,
        settings,
        seed,
        mut film,
        subsurface_frame,
        shader,
        candidates,
        volume_pass,
    } = request;

    let pixel_count = output.linear_rgba.len();
    let source_layer_summary = if let Some(shader) = shader {
        let summary = shader.eevee_raytrace_layer_flags()?;
        if summary.is_none() && shader.needs_eevee_gbuffer_context() {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                "EEVEE ray tracing requires source deferred-layer closure flags",
            ));
        }
        summary
    } else {
        None
    };
    let source_refraction_layer_present =
        source_layer_summary.is_some_and(|summary| summary.refraction.is_some());
    let opaque_feedback_output = source_layer_summary.is_some_and(|summary| {
        let next_layer_has_transmission = summary
            .refraction
            .is_some_and(|layer| layer.has_transmission);
        eevee_source_feedback_output(
            summary.opaque,
            next_layer_has_transmission,
            summary.refraction.is_none(),
        )
    });
    let refraction_feedback_output = source_layer_summary
        .and_then(|summary| summary.refraction)
        .is_some_and(|layer| eevee_source_feedback_output(layer, false, true));
    let mut front_feedback_color: Option<Vec<[f32; 4]>> = None;
    let mut front_feedback_valid = false;
    let mut opaque_feedback_valid = false;
    let mut screen_feedback_view_projection = None;
    let mut prepass_subsurface =
        allocate_filled(pixel_count, None, "EEVEE ray-tracing prepass subsurface")?;
    let mut prepass_surfaces =
        allocate_filled(pixel_count, None, "EEVEE ray-tracing sample surfaces")?;
    let mut opaque_color: Option<Vec<[f32; 4]>> = None;
    let mut opaque_surfaces: Option<Vec<Option<EeveeRaytraceSample>>> = None;
    let mut refraction_surfaces: Option<Vec<Option<EeveeRaytraceSample>>> = None;
    let effective_trace_max = if settings.use_fast_gi {
        settings.trace_max_roughness as f32
    } else {
        1.0
    };
    let fast_gi_active = fast_gi::fast_gi_should_run(settings.use_fast_gi, effective_trace_max);
    let mut fast_gi_history_view_projection = None;
    let denoise_plan = denoise::resolve_stage_plan(
        settings.use_denoise,
        denoise::RequestedStages {
            spatial: settings.denoise_spatial,
            temporal: settings.denoise_temporal,
            bilateral: settings.denoise_bilateral,
        },
    );
    let mut temporal_histories =
        eevee_temporal_histories(pixel_count, render.width, render.height, denoise_plan)?;
    let mut refraction_temporal_histories = None;
    let mut sample_subsurface = if film.is_none() {
        Some(allocate_filled(
            pixel_count,
            None,
            "EEVEE ray-tracing subsurface frame",
        )?)
    } else {
        None
    };
    let mut sample_color = if film.is_none() {
        Some(allocate_filled(
            pixel_count,
            [0.0_f32; 4],
            "EEVEE ray-tracing combined sample",
        )?)
    } else {
        None
    };
    let mut sample_display = if film.is_none() {
        Some(allocate_filled(
            pixel_count,
            [0.0_f32; 4],
            "EEVEE ray-tracing display sample",
        )?)
    } else {
        None
    };
    let mut average_display = if film.is_none() {
        Some(allocate_filled(
            pixel_count,
            [0.0_f64; 4],
            "EEVEE ray-tracing display average",
        )?)
    } else {
        None
    };
    let mut average_linear = if film.is_none() {
        Some(allocate_filled(
            pixel_count,
            [0.0_f64; 4],
            "EEVEE ray-tracing linear average",
        )?)
    } else {
        None
    };
    let sample_offset_seed = film.is_none().then_some(seed);

    for sample_index in 0..render.sample_count {
        let (sample_x, sample_y) = film.as_ref().map_or((0.5, 0.5), |film| {
            film.jitter(film_sample_offset(sample_index))
        });
        let sample = FilmSampleJitter::new(
            sample_index,
            sample_x,
            sample_y,
            shader.is_some_and(|shader| shader.needs_eevee_gbuffer_context()),
        );
        let trace_view = eevee_trace_view(
            render.frame,
            render.camera_matrices,
            render.view_index,
            render.width,
            render.height,
            sample_x,
            sample_y,
        )?;
        let screen_feedback_enabled =
            settings.method == crate::model::EeveeRayTracingMethod::Screen;
        let (previous_combined_feedback, mut prepass_color) = if screen_feedback_enabled {
            let previous = if sample_index == 0 {
                None
            } else if let Some(film) = film.as_ref() {
                Some(film.sample_color.as_slice())
            } else {
                Some(sample_color.as_deref().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "missing EEVEE combined sample history",
                    )
                })?)
            };
            (previous, None)
        } else {
            let color = match film.as_mut() {
                Some(film) => film.sample_color.as_mut_slice(),
                None => sample_color.as_deref_mut().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "missing EEVEE prepass color")
                })?,
            };
            (None, Some(color))
        };
        if let Some(color) = prepass_color.as_deref_mut() {
            color.fill([0.0; 4]);
        }
        prepass_subsurface.fill(None);
        prepass_surfaces.fill(None);
        render.render_rows(
            0..render.height as usize,
            prepass_color.as_deref_mut(),
            None,
            &mut prepass_subsurface,
            Some(&mut prepass_surfaces),
            None,
            sample,
            sample_offset_seed,
            None,
            candidates,
            shader,
            volume_pass,
        )?;
        let has_refraction_layer = source_refraction_layer_present;
        let raytrace_feedback = if has_refraction_layer {
            if front_feedback_color.is_none() {
                let feedback_color = allocate_filled(
                    pixel_count,
                    [0.0_f32; 4],
                    "EEVEE previous refraction-layer color",
                )?;
                front_feedback_color = Some(feedback_color);
            }
            if opaque_color.is_none() {
                opaque_color = Some(allocate_filled(
                    pixel_count,
                    [0.0; 4],
                    "EEVEE opaque-layer feedback color",
                )?);
            }
            if opaque_surfaces.is_none() {
                opaque_surfaces = Some(allocate_filled(
                    pixel_count,
                    None,
                    "EEVEE opaque-layer ray-tracing surfaces",
                )?);
            }
            if refraction_surfaces.is_none() {
                refraction_surfaces = Some(allocate_filled(
                    pixel_count,
                    None,
                    "EEVEE refraction-layer ray-tracing surfaces",
                )?);
            }
            let opaque_color = opaque_color.as_mut().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "missing EEVEE opaque-layer color")
            })?;
            let opaque_surfaces = opaque_surfaces.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "missing EEVEE opaque-layer ray-tracing surfaces",
                )
            })?;
            opaque_surfaces.fill(None);
            prepass_subsurface.fill(None);
            render.render_rows(
                0..render.height as usize,
                None,
                None,
                &mut prepass_subsurface,
                Some(opaque_surfaces),
                Some(EeveeRaytraceLayer::Opaque),
                sample,
                sample_offset_seed,
                None,
                candidates,
                shader,
                volume_pass,
            )?;
            if opaque_surfaces
                .iter()
                .flatten()
                .any(|surface| surface.has_gbuffer_route && surface.screen_refraction.is_none())
            {
                eevee_screen_trace_enabled(settings.method, true, None)?;
            }
            let fast_gi_history_top_down = if opaque_feedback_output && opaque_feedback_valid {
                if screen_feedback_enabled {
                    previous_combined_feedback
                } else {
                    Some(opaque_color.as_slice())
                }
            } else {
                None
            };
            let opaque_trace = trace_eevee_sample_frame(
                render,
                &trace_view,
                settings,
                sample_index,
                sample.gbuffer_closure_sample.unwrap_or(0.0),
                fast_gi_history_top_down,
                fast_gi_history_view_projection,
                &mut temporal_histories,
                opaque_surfaces,
                previous_combined_feedback,
                None,
                None,
                screen_feedback_view_projection,
                shader,
            )?;
            opaque_color.fill([0.0; 4]);
            prepass_subsurface.fill(None);
            render.render_rows(
                0..render.height as usize,
                Some(opaque_color.as_mut_slice()),
                None,
                &mut prepass_subsurface,
                None,
                Some(EeveeRaytraceLayer::Opaque),
                sample,
                sample_offset_seed,
                Some(&opaque_trace.feedback),
                candidates,
                shader,
                volume_pass,
            )?;
            resolve_eevee_subsurface_frame(
                &prepass_subsurface,
                opaque_color,
                subsurface_frame,
                sample_index,
                [sample_x, sample_y],
            )?;

            let refraction_surfaces = refraction_surfaces.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "missing EEVEE refraction-layer ray-tracing surfaces",
                )
            })?;
            prepass_subsurface.fill(None);
            refraction_surfaces.fill(None);
            render.render_rows(
                0..render.height as usize,
                None,
                None,
                &mut prepass_subsurface,
                Some(refraction_surfaces),
                Some(EeveeRaytraceLayer::Refraction),
                sample,
                sample_offset_seed,
                None,
                candidates,
                shader,
                volume_pass,
            )?;
            let fast_gi_history_top_down = if refraction_feedback_output && front_feedback_valid {
                front_feedback_color.as_deref()
            } else {
                None
            };
            if refraction_temporal_histories.is_none() {
                refraction_temporal_histories = Some(eevee_temporal_histories(
                    pixel_count,
                    render.width,
                    render.height,
                    denoise_plan,
                )?);
            }
            let refraction_trace = trace_eevee_sample_frame(
                render,
                &trace_view,
                settings,
                sample_index,
                sample.gbuffer_closure_sample.unwrap_or(0.0),
                fast_gi_history_top_down,
                fast_gi_history_view_projection,
                refraction_temporal_histories.as_mut().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "missing EEVEE refraction-layer temporal history",
                    )
                })?,
                refraction_surfaces,
                Some(opaque_color.as_slice()),
                Some(&opaque_trace.depth),
                fast_gi_history_top_down,
                screen_feedback_view_projection,
                shader,
            )?;
            let mut feedback = opaque_trace.feedback;
            for (feedback, refraction) in feedback.iter_mut().zip(refraction_trace.feedback) {
                feedback.refraction_radiance = refraction.radiance;
                feedback.refraction_active = true;
            }
            let front_feedback_color = front_feedback_color.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "missing EEVEE previous refraction-layer color",
                )
            })?;
            front_feedback_color.fill([0.0; 4]);
            prepass_subsurface.fill(None);
            render.render_rows(
                0..render.height as usize,
                Some(front_feedback_color.as_mut_slice()),
                None,
                &mut prepass_subsurface,
                None,
                Some(EeveeRaytraceLayer::Refraction),
                sample,
                sample_offset_seed,
                Some(&feedback),
                candidates,
                shader,
                volume_pass,
            )?;
            for ((color, surface), opaque_color) in front_feedback_color
                .iter_mut()
                .zip(refraction_surfaces.iter())
                .zip(opaque_color.iter())
            {
                if surface.is_none() {
                    *color = *opaque_color;
                }
            }
            resolve_eevee_subsurface_frame(
                &prepass_subsurface,
                front_feedback_color.as_mut_slice(),
                subsurface_frame,
                sample_index,
                [sample_x, sample_y],
            )?;
            feedback
        } else {
            let fast_gi_history_top_down = if opaque_feedback_output && opaque_feedback_valid {
                if screen_feedback_enabled {
                    previous_combined_feedback
                } else {
                    opaque_color.as_deref()
                }
            } else {
                None
            };
            trace_eevee_sample_frame(
                render,
                &trace_view,
                settings,
                sample_index,
                sample.gbuffer_closure_sample.unwrap_or(0.0),
                fast_gi_history_top_down,
                fast_gi_history_view_projection,
                &mut temporal_histories,
                &prepass_surfaces,
                if screen_feedback_enabled {
                    previous_combined_feedback
                } else {
                    prepass_color.as_deref()
                },
                None,
                None,
                screen_feedback_view_projection,
                shader,
            )?
            .feedback
        };

        if let Some(film) = film.as_mut() {
            film.sample_color.fill([0.0; 4]);
            film.sample_subsurface.fill(None);
            render.render_rows(
                0..render.height as usize,
                Some(film.sample_color.as_mut_slice()),
                sample_display.as_deref_mut(),
                &mut film.sample_subsurface,
                None,
                None,
                sample,
                None,
                Some(&raytrace_feedback),
                candidates,
                shader,
                volume_pass,
            )?;
            resolve_eevee_subsurface_frame(
                &film.sample_subsurface,
                &mut film.sample_color,
                subsurface_frame,
                sample_index,
                [sample_x, sample_y],
            )?;
            for pixel in 0..film.sample_color.len() {
                let color = film.sample_color[pixel].map(f64::from);
                store_film_sample(&mut film.sample_color, pixel, color);
            }
            film.resolve_sample(sample_index, sample_x, sample_y, &mut output.linear_rgba);
        } else {
            let sample_color = sample_color.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "missing EEVEE sample color storage",
                )
            })?;
            let sample_display = sample_display.as_mut().ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "missing EEVEE display color storage",
                )
            })?;
            let sample_subsurface = sample_subsurface.as_mut().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "missing EEVEE SSS storage")
            })?;
            sample_color.fill([0.0; 4]);
            sample_display.fill([0.0; 4]);
            sample_subsurface.fill(None);
            render.render_rows(
                0..render.height as usize,
                Some(sample_color.as_mut_slice()),
                Some(sample_display),
                sample_subsurface,
                None,
                None,
                sample,
                sample_offset_seed,
                Some(&raytrace_feedback),
                candidates,
                shader,
                volume_pass,
            )?;
            resolve_eevee_subsurface_frame(
                sample_subsurface,
                sample_color,
                subsurface_frame,
                sample_index,
                [0.5, 0.5],
            )?;
            if sample_subsurface.iter().any(Option::is_some) {
                for (display, associated) in sample_display.iter_mut().zip(sample_color.iter()) {
                    let alpha = associated[3];
                    *display = if alpha > 0.0 {
                        [
                            associated[0] / alpha,
                            associated[1] / alpha,
                            associated[2] / alpha,
                            alpha,
                        ]
                    } else {
                        *associated
                    };
                }
            }
            let count = f64::from(sample_index + 1);
            let previous_weight = 1.0 - 1.0 / count;
            let average_display = average_display.as_mut().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "missing EEVEE display average")
            })?;
            let average_linear = average_linear.as_mut().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "missing EEVEE linear average")
            })?;
            for pixel in 0..pixel_count {
                for channel in 0..4 {
                    average_display[pixel][channel] = average_display[pixel][channel]
                        * previous_weight
                        + f64::from(sample_display[pixel][channel]) / count;
                    average_linear[pixel][channel] = average_linear[pixel][channel]
                        * previous_weight
                        + f64::from(sample_color[pixel][channel]) / count;
                }
            }
        }
        if screen_feedback_enabled {
            let (history_color, history_subsurface) = if let Some(film) = film.as_mut() {
                (
                    film.sample_color.as_mut_slice(),
                    film.sample_subsurface.as_mut_slice(),
                )
            } else {
                (
                    sample_color.as_deref_mut().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "missing EEVEE combined sample history",
                        )
                    })?,
                    sample_subsurface.as_deref_mut().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "missing EEVEE SSS storage")
                    })?,
                )
            };
            history_subsurface.fill(None);
            if has_refraction_layer {
                let opaque_color = opaque_color.as_deref().ok_or_else(|| {
                    PotError::new(ErrorCode::InternalError, "missing EEVEE opaque-layer color")
                })?;
                history_color.copy_from_slice(opaque_color);
            } else {
                history_color.fill([0.0; 4]);
                render.render_rows(
                    0..render.height as usize,
                    Some(&mut *history_color),
                    None,
                    &mut *history_subsurface,
                    None,
                    Some(EeveeRaytraceLayer::Opaque),
                    sample,
                    sample_offset_seed,
                    Some(&raytrace_feedback),
                    candidates,
                    shader,
                    volume_pass,
                )?;
                resolve_eevee_subsurface_frame(
                    history_subsurface,
                    history_color,
                    subsurface_frame,
                    sample_index,
                    [sample_x, sample_y],
                )?;
            }
            screen_feedback_view_projection = Some(trace_view.view_projection);
        }
        if screen_feedback_enabled {
            opaque_feedback_valid = opaque_feedback_output;
            front_feedback_valid = refraction_feedback_output;
        } else {
            front_feedback_valid = has_refraction_layer && refraction_feedback_output;
            opaque_feedback_valid = false;
            if fast_gi_active && opaque_feedback_output {
                if has_refraction_layer {
                    opaque_feedback_valid = true;
                } else {
                    if opaque_color.is_none() {
                        opaque_color = Some(allocate_filled(
                            pixel_count,
                            [0.0; 4],
                            "EEVEE opaque-layer feedback color",
                        )?);
                    }
                    let opaque_color = opaque_color.as_mut().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "missing EEVEE opaque-layer color")
                    })?;
                    opaque_color.fill([0.0; 4]);
                    prepass_subsurface.fill(None);
                    render.render_rows(
                        0..render.height as usize,
                        Some(opaque_color.as_mut_slice()),
                        None,
                        &mut prepass_subsurface,
                        None,
                        Some(EeveeRaytraceLayer::Opaque),
                        sample,
                        sample_offset_seed,
                        Some(&raytrace_feedback),
                        candidates,
                        shader,
                        volume_pass,
                    )?;
                    resolve_eevee_subsurface_frame(
                        &prepass_subsurface,
                        opaque_color,
                        subsurface_frame,
                        sample_index,
                        [sample_x, sample_y],
                    )?;
                    opaque_feedback_valid = true;
                }
            }
        }
        if opaque_feedback_valid || front_feedback_valid {
            fast_gi_history_view_projection = Some(trace_view.view_projection);
        }
    }

    if film.is_some() {
        for (pixel, associated) in output.linear_rgba.iter().copied().enumerate() {
            let associated = associated.map(f64::from);
            let alpha = associated[3];
            let display = if alpha > 0.0 {
                [
                    associated[0] / alpha,
                    associated[1] / alpha,
                    associated[2] / alpha,
                    alpha,
                ]
            } else {
                associated
            };
            output.rgba[pixel * 4..pixel * 4 + 4].copy_from_slice(&color_to_srgba(display));
        }
    } else {
        let average_display = average_display.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "missing EEVEE display average")
        })?;
        let average_linear = average_linear.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "missing EEVEE linear average")
        })?;
        for pixel in 0..pixel_count {
            output.linear_rgba[pixel] = linear_color_to_f32(if output.linear_rgba_associated {
                average_linear[pixel]
            } else {
                average_display[pixel]
            });
            output.rgba[pixel * 4..pixel * 4 + 4]
                .copy_from_slice(&color_to_srgba(average_display[pixel]));
        }
    }
    Ok(output)
}
#[derive(Clone, Copy)]
struct EeveeTraceView {
    view: Mat4,
    view_inverse: Mat4,
    projection: Mat4,
    projection_inverse: Mat4,
    view_projection: Mat4,
    screen_camera: ScreenTraceCamera,
    near: f32,
    far: f32,
}

#[derive(Clone, Copy)]
struct EeveeTraceFrame<'a> {
    width: u32,
    height: u32,
    surfaces: &'a [Option<EeveeRaytraceSample>],
    screen_depth: &'a [f32],
    view: &'a EeveeTraceView,
    parameters: ScreenTraceParameters,
    back_hiz: HiZBuffer<'a>,
    front_hiz: Option<HiZBuffer<'a>>,
    screen_feedback_view_projection: Option<Mat4>,
    back_radiance: ScreenRadianceBuffer<'a>,
    front_radiance: Option<ScreenRadianceBuffer<'a>>,
    settings: EeveeRayTracingSettings,
    trace_max_roughness: f32,
    trace_scale: u32,
    trace_offset: [u32; 2],
    sample_index: u32,
    raytrace_u: [f32; 2],
    raytrace_w: f32,
    closure_random: f32,
}

struct EeveeDenoisedClosure {
    radiance: Vec<[f32; 3]>,
    tile_mask: Vec<u8>,
}

fn eevee_projection_matrix(
    frame: &CameraFrame,
    width: u32,
    height: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<Mat4> {
    let near = frame.near as f32;
    let far = frame.far as f32;
    let depth = far - near;
    let shift_x = frame.shift[0] as f32;
    let shift_y = frame.shift[1] as f32;
    let pixel_jitter_x = 0.5_f32 - sample_x as f32;
    let pixel_jitter_y = sample_y as f32 - 0.5;
    let jitter_x = (pixel_jitter_x / width as f32) * 2.0;
    let jitter_y = (pixel_jitter_y / height as f32) * 2.0;
    let perspective = matches!(frame.projection, ProjectionFrame::Perspective { .. });
    let mut projection = match frame.projection {
        ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } => Mat4::from_cols(
            Vec4::new(1.0 / horizontal_scale as f32, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / vertical_scale as f32, 0.0, 0.0),
            Vec4::new(2.0 * shift_x, 2.0 * shift_y, -(far + near) / depth, -1.0),
            Vec4::new(0.0, 0.0, -(2.0 * far * near) / depth, 0.0),
        ),
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => Mat4::from_cols(
            Vec4::new(1.0 / half_width as f32, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0 / half_height as f32, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -2.0 / depth, 0.0),
            Vec4::new(-2.0 * shift_x, -2.0 * shift_y, -(far + near) / depth, 1.0),
        ),
        ProjectionFrame::Equirectangular { .. }
        | ProjectionFrame::FisheyeEquidistant { .. }
        | ProjectionFrame::FisheyeEquisolid { .. } => {
            return Err(PotError::new(
                ErrorCode::UnsupportedFeature,
                "EEVEE screen tracing requires its perspective or orthographic camera projection",
            ));
        }
    };
    // Match Blender's row-normalized perspective window translation.
    if perspective {
        let x_scale = projection.x_axis.x;
        let y_scale = projection.y_axis.y;
        let x_row_length = Vec3::new(
            projection.x_axis.x,
            projection.y_axis.x,
            projection.z_axis.x,
        )
        .length();
        let y_row_length = Vec3::new(
            projection.x_axis.y,
            projection.y_axis.y,
            projection.z_axis.y,
        )
        .length();
        projection.z_axis.x -= x_row_length.recip() * x_scale * jitter_x;
        projection.z_axis.y -= y_row_length.recip() * y_scale * jitter_y;
    } else {
        projection.w_axis.x += jitter_x;
        projection.w_axis.y += jitter_y;
    }
    if !projection.is_finite() || !projection.inverse().is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE screen-tracing projection is invalid",
        ));
    }
    Ok(projection)
}

fn eevee_trace_view(
    frame: &CameraFrame,
    camera_matrices: &[(DMat4, DMat4); 2],
    view_index: usize,
    width: u32,
    height: u32,
    sample_x: f64,
    sample_y: f64,
) -> Result<EeveeTraceView> {
    let (camera_to_world, world_to_camera) = camera_matrices.get(view_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE ray tracing has no camera transform for this view",
        )
    })?;
    let view = world_to_camera.as_mat4();
    let view_inverse = camera_to_world.as_mat4();
    let projection = eevee_projection_matrix(frame, width, height, sample_x, sample_y)?;
    let projection_inverse = projection.inverse();
    let screen_camera =
        ScreenTraceCamera::new(projection, projection_inverse).map_err(|error| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                format!("EEVEE screen-tracing camera is invalid: {error:?}"),
            )
        })?;
    Ok(EeveeTraceView {
        view,
        view_inverse,
        projection,
        projection_inverse,
        view_projection: projection * view,
        near: frame.near as f32,
        far: frame.far as f32,
        screen_camera,
    })
}

fn eevee_view_depth_to_screen(projection: Mat4, view_z: f32) -> f32 {
    let clip = projection * Vec4::new(0.0, 0.0, view_z, 1.0);
    clip.z / clip.w * 0.5 + 0.5
}

fn eevee_source_resolution_scale(value: u32) -> u32 {
    let value = value.max(1);
    1_u32 << value.ilog2()
}

fn eevee_random_in_tile(sample_index: u32, tile_size: u32) -> [u32; 2] {
    if tile_size == 1 {
        return [0, 0];
    }
    let mut hash = sample_index;
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x85eb_ca6b);
    hash ^= hash >> 13;
    hash = hash.wrapping_mul(0xc2b2_ae35);
    hash ^= hash >> 16;
    let coordinate = hash % tile_size.saturating_mul(tile_size);
    [
        eevee_compact_bits(coordinate),
        eevee_compact_bits(coordinate >> 1),
    ]
}

fn eevee_compact_bits(mut value: u32) -> u32 {
    value &= 0x5555_5555;
    value = (value ^ (value >> 1)) & 0x3333_3333;
    value = (value ^ (value >> 2)) & 0x0f0f_0f0f;
    value = (value ^ (value >> 4)) & 0x00ff_00ff;
    value = (value ^ (value >> 8)) & 0x0000_ffff;
    value
}

fn eevee_hiz_extent(width: u32, height: u32, probe_extent: u32) -> Result<(u32, u32)> {
    const HIZ_MIP_ALIGNMENT: u32 = 64;
    let pad = |extent: u32| {
        extent
            .max(probe_extent)
            .div_ceil(HIZ_MIP_ALIGNMENT)
            .checked_mul(HIZ_MIP_ALIGNMENT)
    };
    let width = pad(width).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EEVEE HiZ width exceeds texture limits",
        )
    })?;
    let height = pad(height).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EEVEE HiZ height exceeds texture limits",
        )
    })?;
    Ok((width, height))
}

fn eevee_hiz_pyramid(
    depth: &[f32],
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
) -> Result<Vec<Vec<f32>>> {
    const HIZ_MIP_COUNT: usize = 7;
    let mut pyramid = Vec::new();
    pyramid
        .try_reserve_exact(HIZ_MIP_COUNT)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "EEVEE HiZ mip allocation failed"))?;
    let mut base = allocate_filled(
        checked_pixel_count(padded_width, padded_height)?,
        1.0_f32,
        "EEVEE HiZ base level",
    )?;
    for y in 0..padded_height {
        let source_y = y.min(height - 1);
        for x in 0..padded_width {
            let source_x = x.min(width - 1);
            base[y as usize * padded_width as usize + x as usize] =
                depth[source_y as usize * width as usize + source_x as usize];
        }
    }
    pyramid.push(base);
    let (mut previous_width, mut previous_height) = (padded_width, padded_height);
    for _ in 1..HIZ_MIP_COUNT {
        let mip_width = previous_width / 2;
        let mip_height = previous_height / 2;
        let mut mip = allocate_filled(
            checked_pixel_count(mip_width, mip_height)?,
            1.0_f32,
            "EEVEE HiZ mip level",
        )?;
        let previous = pyramid.last().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "EEVEE HiZ pyramid has no base level",
            )
        })?;
        for y in 0..mip_height {
            for x in 0..mip_width {
                let first = (y * 2) as usize * previous_width as usize + (x * 2) as usize;
                let row = previous_width as usize;
                mip[y as usize * mip_width as usize + x as usize] = previous[first]
                    .max(previous[first + 1])
                    .max(previous[first + row])
                    .max(previous[first + row + 1]);
            }
        }
        pyramid.push(mip);
        (previous_width, previous_height) = (mip_width, mip_height);
    }
    Ok(pyramid)
}
struct EeveeDenoiseOps<'a> {
    surfaces: &'a [Option<EeveeRaytraceSample>],
    closure_index: usize,
    width: u32,
    height: u32,
    trace_scale: u32,
    trace_offset: [u32; 2],
    view: &'a EeveeTraceView,
}

impl denoise::ColorTransfer for EeveeDenoiseOps<'_> {
    fn log_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3] {
        color.map(|channel| (1.0 + channel).log2())
    }

    fn scene_linear_from_log(&self, color: [f32; 3]) -> [f32; 3] {
        color.map(|channel| channel.exp2() - 1.0)
    }

    fn ycocg_from_scene_linear(&self, color: [f32; 3]) -> [f32; 3] {
        [
            color[0] + 2.0 * color[1] + color[2],
            2.0 * (color[0] - color[2]),
            -color[0] + 2.0 * color[1] - color[2],
        ]
    }

    fn scene_linear_from_ycocg(&self, color: [f32; 3]) -> [f32; 3] {
        [
            (color[0] + color[1] - color[2]) * 0.25,
            (color[0] + color[2]) * 0.25,
            (color[0] - color[1] - color[2]) * 0.25,
        ]
    }

    fn safe_color(&self, color: [f32; 3]) -> [f32; 3] {
        color.map(|channel| {
            if channel.is_nan() || channel <= 0.0 {
                0.0
            } else if channel >= 1.0e20 {
                1.0e20
            } else {
                channel
            }
        })
    }
}

impl denoise::SpatialKernelOps for EeveeDenoiseOps<'_> {
    fn closure_evaluate_pdf(
        &mut self,
        pixel_index: usize,
        ray_direction: [f32; 3],
        view_normal: [f32; 3],
        view_incident: [f32; 3],
    ) -> f32 {
        let Some(bin) = self
            .surfaces
            .get(pixel_index)
            .and_then(|surface| *surface)
            .and_then(|surface| surface.bins[self.closure_index])
        else {
            return 0.0;
        };
        eevee_closure_evaluate_pdf(
            bin.world_input,
            Vec3::from_array(view_normal),
            Vec3::from_array(ray_direction),
            Vec3::from_array(view_incident),
        )
    }

    fn depth_view_to_screen(&mut self, _pixel_index: usize, view_z: f32) -> f32 {
        eevee_view_depth_to_screen(self.view.projection, view_z)
    }

    fn upsample_sample_weight(&mut self, center_fullres: [u32; 2], trace_sample: [i32; 2]) -> f32 {
        let Some(center_index) = eevee_pixel_index(
            self.width,
            self.height,
            center_fullres[0],
            center_fullres[1],
        ) else {
            return 0.0;
        };
        let sample_x = trace_sample[0] * self.trace_scale as i32 + self.trace_offset[0] as i32;
        let sample_y = trace_sample[1] * self.trace_scale as i32 + self.trace_offset[1] as i32;
        if sample_x < 0 || sample_y < 0 {
            return 0.0;
        }
        let Some(sample_index) =
            eevee_pixel_index(self.width, self.height, sample_x as u32, sample_y as u32)
        else {
            return 0.0;
        };
        let Some(center) = self.surfaces[center_index] else {
            return 0.0;
        };
        let Some(sample) = self.surfaces[sample_index] else {
            return 0.0;
        };
        let Some(center_bin) = center.bins[self.closure_index] else {
            return 0.0;
        };
        let Some(sample_bin) = sample.bins[self.closure_index] else {
            return 0.0;
        };
        let normal = center_bin.world_input.normal;
        let sample_normal = sample_bin.world_input.normal;
        let normal_dot = normal.dot(sample_normal);
        let plane_distance = normal.dot(sample.point.as_vec3() - center.point.as_vec3());
        let depth_weight = (-10_000.0 * plane_distance * plane_distance).exp2();
        (depth_weight * normal_dot.powi(8).clamp(0.0, 1.0)).max(1.0e-4)
    }
}

// EEVEE's source coefficient 1.442_695_041 rounded to its exact f32 bit pattern.

const EEVEE_GAUSSIAN_FACTOR: f32 = f32::from_bits(0x3fb8_aa3b);

impl denoise::BilateralKernelOps for EeveeDenoiseOps<'_> {
    fn gaussian_factor(&self, filter_size: f32, sigma: f32) -> f32 {
        EEVEE_GAUSSIAN_FACTOR * sigma / (filter_size * filter_size)
    }

    fn gaussian_weight(&self, factor: f32, distance_squared: f32) -> f32 {
        (-factor * distance_squared).exp2()
    }

    fn planar_weight(
        &self,
        center_normal: [f32; 3],
        center_position: [f32; 3],
        sample_position: [f32; 3],
        scene_scale: f32,
    ) -> f32 {
        let distance = Vec3::from_array(center_normal)
            .dot(Vec3::from_array(sample_position) - Vec3::from_array(center_position));
        (-scene_scale * distance * distance).exp2()
    }

    fn angle_weight(&self, center_normal: [f32; 3], sample_normal: [f32; 3]) -> f32 {
        Vec3::from_array(center_normal)
            .dot(Vec3::from_array(sample_normal))
            .powi(8)
            .clamp(0.0, 1.0)
    }
}

fn eevee_closure_evaluate_pdf(
    input: RayGenerateInput,
    view_normal: Vec3,
    direction: Vec3,
    view: Vec3,
) -> f32 {
    match input.lobe {
        RayLobe::None => 0.0,
        RayLobe::Diffuse | RayLobe::BurleyBssrdf => view_normal.dot(direction).clamp(0.0, 1.0),
        RayLobe::Translucent => {
            if input.thickness.mode == ThicknessMode::Sphere && input.thickness.value != 0.0 {
                0.25 / std::f32::consts::PI
            } else {
                (-view_normal).dot(direction).clamp(0.0, 1.0)
            }
        }
        RayLobe::GgxReflection => eevee_ggx_reflection_pdf(
            view_normal,
            direction,
            view,
            input.roughness * input.roughness,
        ),
        RayLobe::ThinGlassTransmission => eevee_ggx_reflection_pdf(
            -view_normal,
            direction,
            eevee_reflect(view, view_normal),
            input.roughness * input.roughness,
        ),
        RayLobe::GgxRefraction => {
            let (normal, view, ior) = eevee_ggx_refraction_pdf_context(input, view_normal, view);
            eevee_ggx_refraction_pdf(
                normal,
                direction,
                view,
                input.roughness * input.roughness,
                ior,
                input.thickness.value != 0.0,
            )
        }
    }
}

fn eevee_ggx_reflection_pdf(normal: Vec3, direction: Vec3, view: Vec3, alpha: f32) -> f32 {
    let nv = normal.dot(view);
    let h = (direction + view).normalize_or_zero();
    let alpha = alpha.max(1.0e-3);
    let alpha_squared = alpha * alpha;
    let nv_squared = nv * nv;
    let one_minus_nv_squared = (1.0 - nv_squared).clamp(0.0, 1.0);
    let s_squared = (1.0 + one_minus_nv_squared.sqrt()).powi(2);
    let length_alpha_squared = alpha_squared * one_minus_nv_squared;
    let t = (length_alpha_squared + nv_squared).sqrt();
    let base_pdf = if nv >= 0.0 {
        let k = (1.0 - alpha_squared) * s_squared / (s_squared + alpha_squared * nv_squared);
        1.0 / (2.0 * (k * nv + t))
    } else {
        (t - nv) / (2.0 * length_alpha_squared)
    };
    let nh = normal.dot(h).max(1.0e-8);
    let ndf = eevee_ggx_ndf(nh, alpha_squared);
    base_pdf * ndf
}

fn eevee_ggx_refraction_pdf(
    normal: Vec3,
    direction: Vec3,
    view: Vec3,
    alpha: f32,
    mut ior: f32,
    thickness: bool,
) -> f32 {
    if thickness {
        ior = 1.0 / ior;
    }
    let lv = direction.dot(view);
    if (ior - 1.0).abs() <= 1.0e-4 {
        return 1.0e6;
    }
    if !if ior < 1.0 {
        lv < -ior
    } else {
        lv * ior < -1.0
    } {
        return 0.0;
    }
    let mut h = ior * direction + view;
    if ior >= 1.0 {
        h = -h;
    }
    let inverse_h_length = if h.length_squared() == 0.0 {
        0.0
    } else {
        1.0 / h.length()
    };
    h *= inverse_h_length;
    let nh = normal.dot(h).max(1.0e-8);
    let nv = normal.dot(view);
    let alpha = alpha.max(1.0e-3);
    let alpha_squared = alpha * alpha;
    let geometry_view = eevee_ggx_smith_g1(nv, alpha_squared);
    let ndf = eevee_ggx_ndf(nh, alpha_squared);
    let vh = view.dot(h).clamp(0.0, 1.0);
    let lh = (-direction).dot(h).clamp(0.0, 1.0);
    ndf * geometry_view * vh * lh * (ior * inverse_h_length).powi(2) / nv
}

fn eevee_ggx_refraction_pdf_context(
    input: RayGenerateInput,
    normal: Vec3,
    view: Vec3,
) -> (Vec3, Vec3, f32) {
    let ior = input.ior;
    if input.thickness.value == 0.0 {
        return (normal, view, ior);
    }
    let perceived_roughness = input.apparent_roughness();
    let m = perceived_roughness * perceived_roughness;
    let refracted = eevee_refract(-view, normal, 1.0 / ior);
    let smoothness = 1.0 - m;
    let factor = smoothness * (smoothness.sqrt() + m);
    let direction = (-normal).lerp(refracted, factor).normalize_or_zero();
    if input.thickness.mode == ThicknessMode::Sphere {
        let cos_alpha = direction.dot(-normal);
        let hit_normal = (normal + direction * (cos_alpha * 2.0)).normalize_or_zero();
        (-hit_normal, -direction, ior)
    } else {
        (normal, -direction, ior)
    }
}

fn eevee_ggx_ndf(nh: f32, alpha_squared: f32) -> f32 {
    // Preserve Blender d13 bxdf_ggx_D's grouped NH square in f32.
    let nh_squared = nh * nh;
    alpha_squared
        / (std::f32::consts::PI * ((1.0 - nh_squared) + alpha_squared * nh_squared).powi(2))
}

fn eevee_ggx_smith_g1(nx: f32, alpha_squared: f32) -> f32 {
    2.0 / (1.0 + (1.0 + alpha_squared * (1.0 / (nx * nx) - 1.0)).sqrt())
}

fn eevee_reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * normal.dot(incident) * normal
}

fn eevee_refract(incident: Vec3, normal: Vec3, eta: f32) -> Vec3 {
    let dot = normal.dot(incident);
    let k = 1.0 - eta * eta * (1.0 - dot * dot);
    if k < 0.0 {
        Vec3::ZERO
    } else {
        eta * incident - (eta * dot + k.sqrt()) * normal
    }
}

fn eevee_screen_thickness(
    frame: &CameraFrame,
    projection: &Mat4,
    width: u32,
    height: u32,
    minimum_thickness: f32,
) -> ScreenThicknessParameters {
    let (avg_pixel_radius_unit, is_perspective) = match frame.projection {
        ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } => {
            let x = 2.0 * horizontal_scale / f64::from(width);
            let y = 2.0 * vertical_scale / f64::from(height);
            ((x * x + y * y).sqrt() as f32, true)
        }
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => {
            let x = 2.0 * half_width / f64::from(width);
            let y = 2.0 * half_height / f64::from(height);
            ((x * x + y * y).sqrt() as f32, false)
        }
        ProjectionFrame::Equirectangular { .. }
        | ProjectionFrame::FisheyeEquidistant { .. }
        | ProjectionFrame::FisheyeEquisolid { .. } => (0.0, false),
    };
    let min_pixel_thickness = 3.0_f32;
    if is_perspective {
        ScreenThicknessParameters {
            vs_scale: avg_pixel_radius_unit * min_pixel_thickness,
            vs_bias: -minimum_thickness,
            ndc_scale: projection.w_axis.z * 0.5,
            ndc_bias: 0.0,
        }
    } else {
        ScreenThicknessParameters {
            vs_scale: 0.0,
            vs_bias: 1.0,
            ndc_scale: 0.0,
            ndc_bias: -(avg_pixel_radius_unit * min_pixel_thickness + minimum_thickness)
                * (-2.0 / (frame.far as f32 - frame.near as f32))
                * 0.5,
        }
    }
}
fn eevee_screen_trace_ray(
    view: Mat4,
    world_origin: Vec3,
    direction: Vec3,
    origin_offset_world: Vec3,
    max_time: f32,
) -> ScreenTraceRay {
    ScreenTraceRay {
        origin: view.transform_point3(world_origin) + view.transform_vector3(origin_offset_world),
        direction,
        max_time,
    }
}

fn eevee_screen_to_view(projection_inverse: Mat4, uv: Vec2, depth: f32) -> Vec3 {
    let ndc = Vec3::new(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth * 2.0 - 1.0);
    let homogeneous = projection_inverse * ndc.extend(1.0);
    homogeneous.truncate() / homogeneous.w
}

fn eevee_screen_to_world(view: &EeveeTraceView, uv: Vec2, depth: f32) -> Vec3 {
    view.view_inverse
        .transform_point3(eevee_screen_to_view(view.projection_inverse, uv, depth))
}

fn eevee_view_incident(view: &EeveeTraceView, position: Vec3) -> Vec3 {
    if view.projection.w_axis.w == 0.0 {
        (-position).normalize_or_zero()
    } else {
        Vec3::Z
    }
}

fn eevee_line_plane_intersection(origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) -> Vec3 {
    let denominator = direction.dot(normal);
    if denominator.abs() <= 1.0e-8 {
        point
    } else {
        origin + direction * ((point - origin).dot(normal) / denominator)
    }
}

/// Stores a world-space ray directly in world axes for RTZ, then returns the
/// decoded world and view forms.
fn store_ray_direction_rtz_in_world(view: &EeveeTraceView, direction_world: Vec3) -> (Vec3, Vec3) {
    let stored_world = [
        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(direction_world.x)).to_f32(),
        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(direction_world.y)).to_f32(),
        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(direction_world.z)).to_f32(),
    ];
    let direction_world = Vec3::from_array(stored_world);
    (
        direction_world,
        view.view.transform_vector3(direction_world),
    )
}
fn eevee_view_position_derivatives(
    surface: &EeveeRaytraceSample,
    view: &EeveeTraceView,
    view_normal: Vec3,
    uv: Vec2,
    depth: f32,
    width: u32,
    height: u32,
    trace_scale: u32,
) -> [[f32; 3]; 2] {
    let view_position = view.view.transform_point3(surface.point.as_vec3());
    let pixel_size = Vec2::new(
        trace_scale as f32 / width as f32,
        trace_scale as f32 / height as f32,
    );
    let view_position_x = eevee_screen_to_view(
        view.projection_inverse,
        uv + Vec2::new(pixel_size.x, 0.0),
        depth,
    );
    let view_position_y = eevee_screen_to_view(
        view.projection_inverse,
        uv + Vec2::new(0.0, pixel_size.y),
        depth,
    );
    let normal = view_normal.normalize_or_zero();
    let dx_view = eevee_line_plane_intersection(
        view_position_x,
        eevee_view_incident(view, view_position_x),
        view_position,
        normal,
    ) - view_position;
    let dy_view = eevee_line_plane_intersection(
        view_position_y,
        eevee_view_incident(view, view_position_y),
        view_position,
        normal,
    ) - view_position;
    [dx_view.to_array(), dy_view.to_array()]
}

// EEVEE's source coefficient 52.982_918_9 rounded to its exact f32 bit pattern.

const EEVEE_INTERLEAVED_GRADIENT_NOISE_SCALE: f32 = f32::from_bits(0x4253_ee82);

fn eevee_interleaved_gradient_noise(pixel: Vec2, seed: f32, offset: f32) -> f32 {
    let pixel = pixel + Vec2::new(47.0, 17.0) * (seed * 0.695);
    let inner = (0.067_110_56 * pixel.x + 0.005_837_15 * pixel.y).fract();
    (offset + EEVEE_INTERLEAVED_GRADIENT_NOISE_SCALE * inner).fract()
}

/// Mirrors `Camera::update_bounds`: the bound sphere of the eight view-space frustum corners.
fn eevee_camera_bound_radius(projection: Mat4, near: f32, far: f32) -> f32 {
    let corner = |ndc_x: f32, ndc_y: f32, z: f32| {
        let clip_w = projection.z_axis.w * z + projection.w_axis.w;
        let x =
            (ndc_x * clip_w - projection.z_axis.x * z - projection.w_axis.x) / projection.x_axis.x;
        let y =
            (ndc_y * clip_w - projection.z_axis.y * z - projection.w_axis.y) / projection.y_axis.y;
        Vec3::new(x, y, z)
    };
    let corners = [
        corner(-1.0, -1.0, -near),
        corner(-1.0, -1.0, -far),
        corner(-1.0, 1.0, -far),
        corner(-1.0, 1.0, -near),
        corner(1.0, -1.0, -near),
        corner(1.0, -1.0, -far),
        corner(1.0, 1.0, -far),
        corner(1.0, 1.0, -near),
    ];
    let center = corners
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, point| sum + point)
        / 8.0;
    corners
        .into_iter()
        .map(|point| point.distance_squared(center))
        .fold(0.0_f32, f32::max)
        .sqrt()
}

fn eevee_quantize_radiance(value: Vec3) -> [f32; 3] {
    std::array::from_fn(|channel| {
        super::eevee_transmission::truncate_ufloat(f64::from(value[channel]), [6, 6, 5][channel])
            as f32
    })
}

fn eevee_source_rng(sample_index: u32) -> ([f32; 2], f32, f32) {
    let source_index = u64::from(sample_index) * 13 + 1;
    (
        [
            halton_radical_inverse(5, source_index) as f32,
            halton_radical_inverse(7, source_index) as f32,
        ],
        halton_radical_inverse(11, source_index) as f32,
        halton_radical_inverse(2, u64::from(sample_index) + 1) as f32,
    )
}

fn eevee_pixel_index(width: u32, height: u32, x: u32, y: u32) -> Option<usize> {
    if x >= width || y >= height {
        return None;
    }
    usize::try_from(y)
        .ok()?
        .checked_mul(usize::try_from(width).ok()?)?
        .checked_add(usize::try_from(x).ok()?)
}

fn eevee_mat4_to_rows(matrix: Mat4) -> [[f32; 4]; 4] {
    let columns = matrix.to_cols_array();
    std::array::from_fn(|row| std::array::from_fn(|column| columns[column * 4 + row]))
}

fn eevee_screen_trace_error(error: ScreenTraceError) -> PotError {
    PotError::new(
        ErrorCode::EvaluationFailed,
        format!("EEVEE screen-space ray tracing failed: {error:?}"),
    )
}

fn eevee_utility_blue_noise() -> Result<Vec<[f32; 4]>> {
    let mut noise = allocate_filled(64 * 64, [0.0_f32; 4], "EEVEE utility blue noise")?;
    for y in 0..64 {
        for x in 0..64 {
            let index = y * 64 + x;
            let coordinate = [x, y];
            let ba = super::eevee_blue_noise_ba::raytrace_ba(coordinate);
            noise[index] = [
                super::eevee_blue_noise::closure_random([x as f32, y as f32], 0.0),
                super::eevee_blue_noise_green::raytrace_green(coordinate),
                ba[0],
                ba[1],
            ];
        }
    }
    Ok(noise)
}

fn eevee_screen_trace_enabled(
    method: crate::model::EeveeRayTracingMethod,
    transmission: bool,
    material_refraction: Option<bool>,
) -> Result<bool> {
    if method != crate::model::EeveeRayTracingMethod::Screen {
        return Ok(false);
    }
    if !transmission {
        return Ok(true);
    }
    material_refraction.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "EEVEE screen transmission needs the imported material's raytrace-refraction setting",
            serde_json::json!({"feature_id":"render.eevee.ray_tracing.refraction_material"}),
        )
    })
}

// Matches `DeferredLayer::end_sync` using compile-time layer closure flags.
fn eevee_source_feedback_output(
    layer: EeveeRaytraceLayerFlags,
    next_layer_has_transmission: bool,
    is_last_layer: bool,
) -> bool {
    let use_raytracing = layer.has_transmit_closure || layer.has_reflect_closure;
    let is_layer_refracted = next_layer_has_transmission && layer.has_any_closure;
    let use_screen_reflection = use_raytracing && layer.has_reflect_closure;
    (use_raytracing || is_layer_refracted) && (!is_last_layer || use_screen_reflection)
}

fn trace_eevee_sample_frame(
    render: &FilmSampleRenderContext<'_, '_, '_>,
    view: &EeveeTraceView,
    settings: EeveeRayTracingSettings,
    sample_index: u32,
    closure_random: f32,
    fast_gi_history_top_down: Option<&[[f32; 4]]>,
    fast_gi_history_view_projection: Option<Mat4>,
    temporal_histories: &mut [EeveeTemporalHistory],
    surfaces_top_down: &[Option<EeveeRaytraceSample>],
    sample_color_top_down: Option<&[[f32; 4]]>,
    back_depth: Option<&[f32]>,
    front_feedback_top_down: Option<&[[f32; 4]]>,
    screen_feedback_view_projection: Option<Mat4>,
    shader: Option<RasterShader<'_>>,
) -> Result<EeveeRaytracePassOutput> {
    let (width, height) = (render.width, render.height);
    let pixel_count = checked_pixel_count(width, height)?;
    if surfaces_top_down.len() != pixel_count
        || sample_color_top_down.is_some_and(|color| color.len() != pixel_count)
    {
        return Err(PotError::invalid_argument(
            "EEVEE ray-tracing sample buffers do not match the render extent",
        ));
    }
    if back_depth.is_some_and(|depth| depth.len() != pixel_count) {
        return Err(PotError::invalid_argument(
            "EEVEE ray-tracing back depth does not match the render extent",
        ));
    }
    if front_feedback_top_down.is_some_and(|feedback| feedback.len() != pixel_count) {
        return Err(PotError::invalid_argument(
            "EEVEE previous front-layer color does not match the render extent",
        ));
    }
    if fast_gi_history_top_down.is_some_and(|history| history.len() != pixel_count) {
        return Err(PotError::invalid_argument(
            "EEVEE FastGI history texture must match the current sample extent",
        ));
    }
    let trace_max_roughness = if settings.use_fast_gi {
        settings.trace_max_roughness as f32
    } else {
        1.0
    };
    let fast_gi_active = fast_gi::fast_gi_should_run(settings.use_fast_gi, trace_max_roughness);
    let (hiz_width, hiz_height) =
        eevee_hiz_extent(width, height, settings.sphere_probe_render_extent)?;
    let mut surfaces = allocate_filled(pixel_count, None, "EEVEE lower-left sample surfaces")?;
    let mut screen_depth = allocate_filled(pixel_count, 1.0_f32, "EEVEE sample depth")?;
    let mut world_normals =
        allocate_filled(pixel_count, [0.0_f32, 0.0, 1.0], "EEVEE sample normals")?;
    let mut processed = allocate_filled(pixel_count, false, "EEVEE G-buffer mask")?;
    let mut screen_radiance = allocate_filled(
        pixel_count,
        [exr::prelude::f16::from_f32(0.0); 4],
        "EEVEE screen radiance texture",
    )?;
    let mut front_screen_radiance = if back_depth.is_some() {
        Some(allocate_filled(
            pixel_count,
            [exr::prelude::f16::from_f32(0.0); 4],
            "EEVEE previous refraction feedback",
        )?)
    } else {
        None
    };
    let mut resolve_pixels = allocate_filled(
        pixel_count,
        fast_gi::FastGiResolvePixel {
            volume_irradiance: None,
            closures: [None; 3],
        },
        "EEVEE FastGI resolve pixels",
    )?;
    for y_gl in 0..height {
        let y_top = height - 1 - y_gl;
        for x in 0..width {
            let gl_index = y_gl as usize * width as usize + x as usize;
            let top_index = y_top as usize * width as usize + x as usize;
            let surface = surfaces_top_down[top_index];
            surfaces[gl_index] = surface;
            if let Some(surface) = surface {
                let view_position = view.view.transform_point3(surface.point.as_vec3());
                screen_depth[gl_index] =
                    eevee_view_depth_to_screen(view.projection, view_position.z).clamp(0.0, 1.0);
                world_normals[gl_index] = surface.normal.as_vec3().to_array();
                processed[gl_index] = surface.bins.iter().any(Option::is_some);
                if processed[gl_index] {
                    if let Some(shader) = shader {
                        resolve_pixels[gl_index].volume_irradiance =
                            shader.eevee_fast_gi_volume_irradiance(surface.point)?;
                    }
                    for (bin_index, bin) in surface.bins.iter().copied().enumerate() {
                        if let Some(bin) = bin {
                            resolve_pixels[gl_index].closures[bin_index] =
                                Some(fast_gi::FastGiResolveClosure {
                                    bin_index: bin_index as u8,
                                    roughness: bin.world_input.apparent_roughness(),
                                    dominant_direction: bin.dominant_direction.as_vec3(),
                                    transmission: bin.transmission,
                                });
                        }
                    }
                }
            }
            let history_radiance_top_down = if fast_gi_active
                && settings.method != crate::model::EeveeRayTracingMethod::Screen
            {
                fast_gi_history_top_down
            } else {
                sample_color_top_down
            };
            if let Some(history_radiance_top_down) = history_radiance_top_down {
                let color = history_radiance_top_down[top_index];
                let rgb = if settings.method == crate::model::EeveeRayTracingMethod::Screen
                    || fast_gi_active
                {
                    [color[0], color[1], color[2]]
                } else {
                    let alpha = color[3];
                    if render.linear_rgba_associated && alpha > 0.0 {
                        [color[0] / alpha, color[1] / alpha, color[2] / alpha]
                    } else {
                        [color[0], color[1], color[2]]
                    }
                };
                screen_radiance[gl_index] = [
                    exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[0])),
                    exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[1])),
                    exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[2])),
                    exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(0.0)),
                ];
            }
        }
    }

    let hiz_depth = back_depth.unwrap_or(&screen_depth);
    let hiz_pyramid = eevee_hiz_pyramid(hiz_depth, width, height, hiz_width, hiz_height)?;
    let hiz_scale = Vec2::new(
        width as f32 / hiz_width as f32,
        height as f32 / hiz_height as f32,
    );
    let mut hiz_levels = Vec::new();
    hiz_levels
        .try_reserve_exact(hiz_pyramid.len())
        .map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "EEVEE HiZ level view allocation failed",
            )
        })?;
    for (mip, depth) in hiz_pyramid.iter().enumerate() {
        hiz_levels.push(
            HiZLevel::new((hiz_width >> mip).max(1), (hiz_height >> mip).max(1), depth).map_err(
                |error| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        format!("EEVEE HiZ level is invalid: {error:?}"),
                    )
                },
            )?,
        );
    }
    let back_hiz = HiZBuffer::new(hiz_scale, &hiz_levels).map_err(|error| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            format!("EEVEE HiZ buffer is invalid: {error:?}"),
        )
    })?;
    let front_hiz_pyramid = if let Some(back_depth) = back_depth {
        let mut front_depth = allocate_filled(pixel_count, 1.0_f32, "EEVEE composed front depth")?;
        for index in 0..pixel_count {
            front_depth[index] = screen_depth[index].min(back_depth[index]);
        }
        Some(eevee_hiz_pyramid(
            &front_depth,
            width,
            height,
            hiz_width,
            hiz_height,
        )?)
    } else {
        None
    };
    let mut front_hiz_levels = Vec::new();
    let front_hiz = if let Some(front_hiz_pyramid) = front_hiz_pyramid.as_ref() {
        front_hiz_levels
            .try_reserve_exact(front_hiz_pyramid.len())
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "EEVEE front HiZ level view allocation failed",
                )
            })?;
        for (mip, depth) in front_hiz_pyramid.iter().enumerate() {
            front_hiz_levels.push(
                HiZLevel::new((hiz_width >> mip).max(1), (hiz_height >> mip).max(1), depth)
                    .map_err(|error| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            format!("EEVEE front HiZ level is invalid: {error:?}"),
                        )
                    })?,
            );
        }
        Some(
            HiZBuffer::new(hiz_scale, &front_hiz_levels).map_err(|error| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("EEVEE front HiZ buffer is invalid: {error:?}"),
                )
            })?,
        )
    } else {
        None
    };
    let back_radiance =
        ScreenRadianceBuffer::new(width, height, &screen_radiance).map_err(|error| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                format!("EEVEE screen radiance buffer is invalid: {error:?}"),
            )
        })?;

    let trace_scale = eevee_source_resolution_scale(settings.resolution_scale);
    let trace_offset = eevee_random_in_tile(sample_index, trace_scale);
    let (raytrace_u, raytrace_w, _) = eevee_source_rng(sample_index);
    let trace_frame = EeveeTraceFrame {
        width,
        height,
        surfaces: &surfaces,
        screen_depth: &screen_depth,
        view,
        parameters: ScreenTraceParameters {
            full_resolution: UVec2::new(width, height),
            quality: (1.0 - 0.95 * settings.screen_trace_quality) as f32,
            ray_thickness: eevee_screen_thickness(
                render.frame,
                &view.projection,
                width,
                height,
                settings.screen_trace_thickness as f32,
            ),
        },
        back_hiz,
        front_hiz,
        screen_feedback_view_projection,
        back_radiance,
        front_radiance: None,
        settings,
        trace_max_roughness,
        trace_scale,
        trace_offset,
        sample_index,
        raytrace_u,
        raytrace_w,
        closure_random,
    };
    let plan = denoise::resolve_stage_plan(
        settings.use_denoise,
        denoise::RequestedStages {
            spatial: settings.denoise_spatial,
            temporal: settings.denoise_temporal,
            bilateral: settings.denoise_bilateral,
        },
    );
    let mut closure_images = Vec::new();
    closure_images.try_reserve_exact(3).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EEVEE closure radiance image allocation failed",
        )
    })?;
    let mut feedback = allocate_filled(
        pixel_count,
        EeveeRaytraceFeedback::default(),
        "EEVEE per-sample ray-trace feedback",
    )?;
    let front_radiance = if let Some(front_screen_radiance) = front_screen_radiance.as_mut() {
        front_screen_radiance.fill([exr::prelude::f16::from_f32(0.0); 4]);
        if let Some(front_feedback) = front_feedback_top_down {
            for y_gl in 0..height {
                let y_top = height - 1 - y_gl;
                for x in 0..width {
                    let gl_index = y_gl as usize * width as usize + x as usize;
                    let top_index = y_top as usize * width as usize + x as usize;
                    let color = front_feedback[top_index];
                    let rgb = [color[0], color[1], color[2]];
                    front_screen_radiance[gl_index] = [
                        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[0])),
                        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[1])),
                        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(rgb[2])),
                        exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(0.0)),
                    ];
                }
            }
        }
        Some(
            ScreenRadianceBuffer::new(width, height, front_screen_radiance).map_err(|error| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    format!("EEVEE previous refraction radiance buffer is invalid: {error:?}"),
                )
            })?,
        )
    } else {
        None
    };
    let closure_trace_frame = EeveeTraceFrame {
        front_radiance,
        ..trace_frame
    };
    for closure_index in 0..3 {
        let history = temporal_histories.get_mut(closure_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "EEVEE temporal history has fewer than three closure bins",
            )
        })?;
        let denoised =
            trace_eevee_closure_frame(&closure_trace_frame, closure_index, plan, history, shader)?;
        for y_gl in 0..height {
            let y_top = height - 1 - y_gl;
            for x in 0..width {
                let gl_index = y_gl as usize * width as usize + x as usize;
                let top_index = y_top as usize * width as usize + x as usize;
                let tile_width = width.div_ceil(denoise::RAYTRACE_GROUP_SIZE);
                let tile_index = (y_gl / denoise::RAYTRACE_GROUP_SIZE) as usize
                    * tile_width as usize
                    + (x / denoise::RAYTRACE_GROUP_SIZE) as usize;
                if surfaces[gl_index].is_some_and(|surface| surface.bins[closure_index].is_some())
                    && denoised.tile_mask[tile_index] != 0
                {
                    feedback[top_index].radiance[closure_index] = Some(denoised.radiance[gl_index]);
                }
            }
        }
        closure_images.push(denoised.radiance);
    }

    if fast_gi::fast_gi_should_run(settings.use_fast_gi, trace_max_roughness) {
        let blue_noise = eevee_utility_blue_noise()?;
        let mut reversed_depth =
            allocate_filled(pixel_count, 0.0_f32, "EEVEE FastGI reverse-Z depth")?;
        for index in 0..pixel_count {
            reversed_depth[index] = 1.0 - screen_depth[index];
        }
        let fast_gi_hiz_pyramid = front_hiz_pyramid.as_deref().unwrap_or(&hiz_pyramid);
        let mut fast_hiz = Vec::new();
        fast_hiz
            .try_reserve_exact(fast_gi_hiz_pyramid.len())
            .map_err(|_| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "EEVEE FastGI HiZ view allocation failed",
                )
            })?;
        for (mip, pixels) in fast_gi_hiz_pyramid.iter().enumerate() {
            fast_hiz.push(fast_gi::FastGiTexture {
                width: (hiz_width >> mip).max(1),
                height: (hiz_height >> mip).max(1),
                pixels,
            });
        }
        let closure_textures = [
            fast_gi::FastGiTexture {
                width,
                height,
                pixels: &closure_images[0],
            },
            fast_gi::FastGiTexture {
                width,
                height,
                pixels: &closure_images[1],
            },
            fast_gi::FastGiTexture {
                width,
                height,
                pixels: &closure_images[2],
            },
        ];
        let ao_index = u64::from(sample_index) + 1;
        let fast_settings = fast_gi::FastGiSettings {
            resolution: settings.fast_gi_resolution,
            raytrace_resolution: settings.resolution_scale,
            ray_count: settings.fast_gi_ray_count,
            step_count: settings.fast_gi_step_count,
            quality: settings.fast_gi_quality as f32,
            bias: settings.fast_gi_bias as f32,
            distance: settings.fast_gi_distance as f32,
            camera_bound_radius: eevee_camera_bound_radius(view.projection, view.near, view.far),
            thickness_near: settings.fast_gi_thickness_near as f32,
            method: match settings.fast_gi_method {
                crate::model::EeveeFastGiMethod::AmbientOcclusionOnly => {
                    fast_gi::FastGiMethod::AoOnly
                }
                crate::model::EeveeFastGiMethod::GlobalIllumination => fast_gi::FastGiMethod::Full,
            },
            trace_max_roughness,
            backface_hit_enabled: settings.use_backface_hit,
            backface_radiance_scale: settings.backface_radiance_scale as f32,
            surface_indirect_clamp: settings.surface_indirect_clamp as f32,
            hiz_uv_scale: hiz_scale,
            sample_index,
            sampling_ao: Vec3::new(
                halton_radical_inverse(5, ao_index) as f32,
                halton_radical_inverse(7, ao_index) as f32,
                halton_radical_inverse(3, ao_index) as f32,
            ),
        };
        let history_pixels = front_screen_radiance.as_deref().unwrap_or(&screen_radiance);
        let output = fast_gi::run_fast_gi(&fast_gi::FastGiInput {
            width,
            height,
            depth: &reversed_depth,
            hiz: &fast_hiz,
            history_radiance: fast_gi::FastGiTexture {
                width,
                height,
                pixels: history_pixels,
            },
            history_valid: fast_gi_history_top_down.is_some(),
            world_normals: &world_normals,
            processed: &processed,
            blue_noise: fast_gi::FastGiTexture {
                width: 64,
                height: 64,
                pixels: &blue_noise,
            },
            closures: closure_textures,
            resolve: &resolve_pixels,
            view: fast_gi::FastGiView {
                view: view.view,
                view_inverse: view.view_inverse,
                projection: view.projection,
                projection_inverse: view.projection_inverse,
            },
            history_view_projection: fast_gi_history_view_projection
                .unwrap_or(view.view_projection),
            settings: fast_settings,
        })?;
        for y_gl in 0..height {
            let y_top = height - 1 - y_gl;
            for x in 0..width {
                let gl_index = y_gl as usize * width as usize + x as usize;
                let top_index = y_top as usize * width as usize + x as usize;
                if let Some(surface) = surfaces[gl_index] {
                    for closure_index in 0..3 {
                        if surface.bins[closure_index].is_some() {
                            feedback[top_index].radiance[closure_index] =
                                Some(output.closure_radiance[closure_index].pixels[gl_index]);
                        }
                    }
                }
            }
        }
    }
    Ok(EeveeRaytracePassOutput {
        feedback,
        depth: screen_depth,
    })
}
fn trace_eevee_closure_frame(
    frame: &EeveeTraceFrame<'_>,
    closure_index: usize,
    stage_plan: denoise::StagePlan,
    history: &mut EeveeTemporalHistory,
    shader: Option<RasterShader<'_>>,
) -> Result<EeveeDenoisedClosure> {
    let width = frame.width;
    let height = frame.height;
    let pixel_count = checked_pixel_count(width, height)?;
    let trace_extent = denoise::Extent {
        width: width.div_ceil(frame.trace_scale),
        height: height.div_ceil(frame.trace_scale),
    };
    let trace_count = checked_pixel_count(trace_extent.width, trace_extent.height)?;
    let tile_extent = denoise::Extent {
        width: width.div_ceil(denoise::RAYTRACE_GROUP_SIZE),
        height: height.div_ceil(denoise::RAYTRACE_GROUP_SIZE),
    };
    let tile_count = checked_pixel_count(tile_extent.width, tile_extent.height)?;
    let mut tile_mask = allocate_filled(tile_count, 0_u8, "EEVEE ray-tracing tile mask")?;
    let mut pixels = allocate_filled(
        pixel_count,
        denoise::SpatialPixel {
            closure_present: false,
            apparent_roughness: 1.0,
            noise: [0.0; 2],
            view_normal: Vec3::Z.to_array(),
            view_incident: Vec3::Z.to_array(),
            view_position_derivatives: [[0.0; 3]; 2],
            scene_view_z: 0.0,
        },
        "EEVEE spatial denoise inputs",
    )?;
    let roughness_end = frame.trace_max_roughness + 0.2;
    for y in 0..height {
        for x in 0..width {
            let index = y as usize * width as usize + x as usize;
            let Some(surface) = frame.surfaces.get(index).copied().flatten() else {
                continue;
            };
            let Some(bin) = surface.bins[closure_index] else {
                continue;
            };
            let blue_noise = super::eevee_blue_noise_ba::raytrace_ba([x as usize, y as usize]);
            let apparent_roughness = bin.world_input.apparent_roughness();
            let view_normal = frame.view.view.transform_vector3(bin.world_input.normal);
            let view_incident = frame.view.view.transform_vector3(bin.world_input.view);
            let view_position = frame.view.view.transform_point3(surface.point.as_vec3());
            pixels[index] = denoise::SpatialPixel {
                closure_present: true,
                apparent_roughness,
                noise: [
                    (blue_noise[0] + frame.closure_random).fract(),
                    (blue_noise[1] + frame.closure_random).fract(),
                ],
                view_normal: view_normal.to_array(),
                view_incident: view_incident.to_array(),
                view_position_derivatives: eevee_view_position_derivatives(
                    &surface,
                    frame.view,
                    view_normal,
                    Vec2::new(
                        (x as f32 + 0.5) / width as f32,
                        (y as f32 + 0.5) / height as f32,
                    ),
                    frame.screen_depth[index],
                    width,
                    height,
                    frame.trace_scale,
                ),
                scene_view_z: view_position.z,
            };
            if apparent_roughness < roughness_end {
                let tile_index = (y / denoise::RAYTRACE_GROUP_SIZE) as usize
                    * tile_extent.width as usize
                    + (x / denoise::RAYTRACE_GROUP_SIZE) as usize;
                tile_mask[tile_index] |= 1;
            }
        }
    }

    let zero_ray = denoise::RaySample {
        direction_and_pdf_inverse: [0.0; 4],
        time: 0.0,
        radiance: [0.0; 3],
    };
    let mut rays = allocate_filled(trace_count, zero_ray, "EEVEE ray-tracing ray samples")?;
    for trace_y in 0..trace_extent.height {
        for trace_x in 0..trace_extent.width {
            let full_x = trace_x * frame.trace_scale + frame.trace_offset[0];
            let full_y = trace_y * frame.trace_scale + frame.trace_offset[1];
            if full_x >= width || full_y >= height {
                continue;
            }
            let full_tile = (full_y / denoise::RAYTRACE_GROUP_SIZE) as usize
                * tile_extent.width as usize
                + (full_x / denoise::RAYTRACE_GROUP_SIZE) as usize;
            if tile_mask[full_tile] & 1 == 0 {
                continue;
            }
            let pixel_index = full_y as usize * width as usize + full_x as usize;
            let Some(surface) = frame.surfaces[pixel_index] else {
                continue;
            };
            let Some(bin) = surface.bins[closure_index] else {
                continue;
            };
            let noise = [
                super::eevee_blue_noise::closure_random(
                    [trace_x as f32, trace_y as f32],
                    frame.raytrace_u[0],
                ),
                (super::eevee_blue_noise_green::raytrace_green([
                    trace_x as usize,
                    trace_y as usize,
                ]) + frame.raytrace_u[1])
                    .fract(),
            ];
            let generated = match generate_ray(bin.world_input, noise) {
                Ok(generated) => generated,
                Err(RayGenerateError::NoSampleableClosure) => continue,
                Err(RayGenerateError::InvalidInput) => {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "EEVEE ray generation received invalid selected G-buffer data",
                    ));
                }
            };
            let inverse_pdf = if generated.pdf == 0.0 {
                0.0
            } else {
                (1.0 / generated.pdf).max(6.0e-8)
            };
            let (world_direction, ray_direction) =
                store_ray_direction_rtz_in_world(frame.view, generated.direction);
            let direction_and_pdf_inverse = [
                ray_direction.x,
                ray_direction.y,
                ray_direction.z,
                exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(inverse_pdf)).to_f32(),
            ];
            if direction_and_pdf_inverse[3] == 0.0 {
                continue;
            }
            let mut radiance = Vec3::ZERO;
            let mut hit_time = 0.0;
            let mut hit_valid = false;
            let is_transmission = bin.transmission;
            let screen_trace_enabled = eevee_screen_trace_enabled(
                frame.settings.method,
                is_transmission,
                surface.screen_refraction,
            )?;
            if screen_trace_enabled {
                let (hiz, screen_radiance) = if frame.front_hiz.is_some() && !is_transmission {
                    (
                        frame.front_hiz.ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "missing EEVEE composed front HiZ",
                            )
                        })?,
                        frame.front_radiance.ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "missing EEVEE previous refraction radiance",
                            )
                        })?,
                    )
                } else {
                    (frame.back_hiz, frame.back_radiance)
                };
                let apparent_roughness = bin.world_input.apparent_roughness();
                let ray = eevee_screen_trace_ray(
                    frame.view.view,
                    surface.point.as_vec3(),
                    ray_direction,
                    generated.origin_offset,
                    (frame.view.far - frame.view.near).abs(),
                );
                let stride_rand = eevee_interleaved_gradient_noise(
                    Vec2::new(trace_x as f32, trace_y as f32),
                    5.0,
                    frame.raytrace_w,
                );
                let hit = raytrace_screen(
                    frame.view.screen_camera,
                    frame.parameters,
                    hiz,
                    ray,
                    stride_rand,
                    apparent_roughness,
                    is_transmission,
                )
                .map_err(eevee_screen_trace_error)?;
                if hit.valid {
                    if is_transmission {
                        let sample_uv = hit.hit_uv;
                        radiance = raytrace_sample_screen(
                            frame.view.screen_camera,
                            frame.parameters,
                            screen_radiance,
                            hit,
                            apparent_roughness,
                            sample_uv,
                        )
                        .map_err(eevee_screen_trace_error)?;
                        hit_valid = true;
                        hit_time = hit.time;
                    } else {
                        let hit_world = frame
                            .view
                            .view_inverse
                            .transform_point3(hit.hit_view_position);
                        let history_view_projection = frame
                            .screen_feedback_view_projection
                            .unwrap_or(frame.view.view_projection);
                        let history_clip = history_view_projection * hit_world.extend(1.0);
                        if history_clip.w != 0.0 {
                            let ndc = history_clip.truncate() / history_clip.w;
                            if ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0 {
                                let history_uv = ndc.truncate() * 0.5 + Vec2::splat(0.5);
                                radiance = raytrace_sample_screen(
                                    frame.view.screen_camera,
                                    frame.parameters,
                                    screen_radiance,
                                    hit,
                                    apparent_roughness,
                                    history_uv,
                                )
                                .map_err(eevee_screen_trace_error)?;
                                hit_valid = true;
                                hit_time = hit.time;
                                if hit.hit_backface {
                                    radiance *= if frame.settings.use_backface_hit {
                                        frame.settings.backface_radiance_scale as f32
                                    } else {
                                        0.0
                                    };
                                    hit_valid = frame.settings.use_backface_hit;
                                }
                            }
                        }
                    }
                }
            }
            if !hit_valid {
                let shader = shader.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::UnsupportedFeature,
                        "EEVEE ray tracing requires the scene probe-radiance evaluator",
                    )
                })?;
                radiance = shader
                    .eevee_raytrace_probe_radiance(
                        &surface,
                        bin,
                        DVec3::new(
                            f64::from(world_direction.x),
                            f64::from(world_direction.y),
                            f64::from(world_direction.z),
                        ),
                    )?
                    .as_vec3();
                hit_time = if frame.settings.method == crate::model::EeveeRayTracingMethod::Probe {
                    1000.0
                } else {
                    10000.0
                };
            }
            let clamp = 1.0e20_f32;
            radiance *= (clamp / radiance.abs().max_element().max(1.0e-8)).clamp(0.0, 1.0);
            let ray_index = trace_y as usize * trace_extent.width as usize + trace_x as usize;
            rays[ray_index] = denoise::RaySample {
                direction_and_pdf_inverse,
                time: hit_time,
                radiance: eevee_quantize_radiance(radiance),
            };
        }
    }
    let spatial_pixels = pixels;
    let mut spatial_radiance = allocate_filled(
        pixel_count,
        denoise::INVALID_RADIANCE,
        "EEVEE spatial radiance",
    )?;
    let mut spatial_variance = allocate_filled(pixel_count, 0.0_f32, "EEVEE spatial variance")?;
    let mut hit_depth = allocate_filled(pixel_count, 0.0_f32, "EEVEE spatial hit depth")?;
    let mut ops = EeveeDenoiseOps {
        surfaces: frame.surfaces,
        closure_index,
        width,
        height,
        trace_scale: frame.trace_scale,
        trace_offset: frame.trace_offset,
        view: frame.view,
    };
    denoise::spatial_denoise(
        denoise::SpatialInput {
            extent: denoise::Extent { width, height },
            trace_extent,
            trace_scale: frame.trace_scale,
            trace_offset: frame.trace_offset,
            tile_mask: &tile_mask,
            pixels: &spatial_pixels,
            rays: &rays,
            skip_denoise: !stage_plan.spatial_denoising,
        },
        denoise::SpatialOutput {
            radiance: &mut spatial_radiance,
            variance: &mut spatial_variance,
            hit_depth: &mut hit_depth,
        },
        &mut ops,
    )?;
    let mut temporal_radiance = spatial_radiance.clone();
    let mut temporal_variance = spatial_variance.clone();
    if stage_plan.temporal {
        let mut temporal_pixels = allocate_filled(
            pixel_count,
            denoise::TemporalPixel {
                surface_world: [0.0; 3],
                hit_world: [0.0; 3],
            },
            "EEVEE temporal pixel positions",
        )?;
        for y in 0..height {
            for x in 0..width {
                let index = y as usize * width as usize + x as usize;
                let uv = Vec2::new(
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                );
                let surface_world = frame.surfaces[index]
                    .map_or([0.0; 3], |surface| surface.point.as_vec3().to_array());
                let hit_world = eevee_screen_to_world(frame.view, uv, hit_depth[index]).to_array();
                temporal_pixels[index] = denoise::TemporalPixel {
                    surface_world,
                    hit_world,
                };
            }
        }
        let zero_history_mask = allocate_filled(
            history.tile_mask.len(),
            0_u8,
            "EEVEE invalid temporal history tile mask",
        )?;
        let history_radiance = &history.radiance;
        let history_variance = &history.variance;
        let history_tile_mask = if history.valid {
            &history.tile_mask
        } else {
            &zero_history_mask
        };
        denoise::temporal_denoise(
            denoise::TemporalInput {
                extent: denoise::Extent { width, height },
                tile_mask: &tile_mask,
                spatial_radiance: &spatial_radiance,
                spatial_variance: &spatial_variance,
                pixels: &temporal_pixels,
                history_extent: denoise::Extent { width, height },
                history_radiance,
                history_variance,
                history_tile_mask,
                history_view_projection: eevee_mat4_to_rows(
                    history
                        .view_projection
                        .unwrap_or(frame.view.view_projection),
                ),
            },
            denoise::TemporalOutput {
                radiance: &mut temporal_radiance,
                variance: &mut temporal_variance,
            },
            &ops,
        )?;
        history.radiance.clone_from(&temporal_radiance);
        history.variance.clone_from(&temporal_variance);
        history.tile_mask.clone_from(&tile_mask);
        history.view_projection = Some(frame.view.view_projection);
        history.valid = true;
    } else {
        history.valid = false;
    }
    let mut radiance = temporal_radiance;
    if stage_plan.bilateral {
        let mut bilateral_pixels = allocate_filled(
            pixel_count,
            denoise::BilateralPixel {
                closure_present: false,
                depth: 0.0,
                apparent_roughness: 1.0,
                world_position: [0.0; 3],
                world_normal: [0.0; 3],
                noise: [0.0; 2],
            },
            "EEVEE bilateral filter pixels",
        )?;
        let (_, raytrace_w, raytrace_x) = eevee_source_rng(frame.sample_index);
        for (index, bilateral_pixel) in bilateral_pixels.iter_mut().enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            let surface = frame.surfaces[index];
            let bin = surface.and_then(|surface| surface.bins[closure_index]);
            let position = surface.map_or([0.0; 3], |surface| surface.point.as_vec3().to_array());
            let noise = [
                eevee_interleaved_gradient_noise(
                    Vec2::new(x as f32 + 0.5, y as f32 + 0.5),
                    3.0,
                    0.0,
                ) + raytrace_w,
                eevee_interleaved_gradient_noise(
                    Vec2::new(x as f32 + 0.5, y as f32 + 0.5),
                    5.0,
                    0.0,
                ) + raytrace_x,
            ];
            *bilateral_pixel = denoise::BilateralPixel {
                closure_present: bin.is_some(),
                depth: 1.0 - frame.screen_depth[index],
                apparent_roughness: bin.map_or(1.0, |bin| bin.world_input.apparent_roughness()),
                world_position: position,
                world_normal: bin.map_or([0.0; 3], |bin| bin.world_input.normal.to_array()),
                noise,
            };
        }
        let mut bilateral_radiance = radiance.clone();
        denoise::bilateral_denoise(
            denoise::BilateralInput {
                extent: denoise::Extent { width, height },
                tile_mask: &tile_mask,
                pixels: &bilateral_pixels,
                temporal_radiance: &radiance,
                temporal_variance: &temporal_variance,
            },
            denoise::BilateralOutput {
                radiance: &mut bilateral_radiance,
            },
            &ops,
        )?;
        radiance = bilateral_radiance;
    }
    Ok(EeveeDenoisedClosure {
        radiance,
        tile_mask,
    })
}

fn rasterize_with_samples_for_eye(
    request: RasterSamplesRequest<'_, '_, '_, '_, '_, '_>,
) -> Result<RasterOutput> {
    let RasterSamplesRequest {
        triangles,
        camera,
        width,
        height,
        mode,
        background,
        world,
        eevee_world_rays,
        samples,
        seed,
        filter_size,
        exposure_scale,
        shader,
        raytrace_settings,
        prepared_materials,
        camera_lights,
        view_index,
        volume_pass,
        band_count,
    } = request;
    if samples == 0 {
        return Err(PotError::invalid_argument(
            "sample count must be greater than zero",
        ));
    }
    let mut volume_pass = volume_pass;
    if camera.stereo_mode == StereoMode::Anaglyph {
        let left_camera = camera_eye(camera, -1.0);
        let right_camera = camera_eye(camera, 1.0);
        let mut left_output = rasterize_with_samples_for_eye(RasterSamplesRequest {
            triangles,
            camera: &left_camera,
            width,
            height,
            mode,
            background,
            world,
            eevee_world_rays,
            raytrace_settings,
            samples,
            seed,
            filter_size,
            exposure_scale,
            shader,
            prepared_materials,
            camera_lights: camera_lights.map(|lights| lights.for_eye(0)),
            view_index: 0,
            band_count,
            volume_pass: volume_pass.as_deref_mut(),
        })?;
        let right_output = rasterize_with_samples_for_eye(RasterSamplesRequest {
            triangles,
            camera: &right_camera,
            width,
            height,
            mode,
            background,
            world,
            eevee_world_rays,
            raytrace_settings,
            samples,
            seed,
            filter_size,
            exposure_scale,
            shader,
            prepared_materials,
            camera_lights: camera_lights.map(|lights| lights.for_eye(1)),
            view_index: 1,
            band_count,
            volume_pass: volume_pass.as_deref_mut(),
        })?;
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
    let eevee_sss_frame = matches!(mode, Mode::Beauty)
        .then(|| eevee_subsurface_frame(camera, &frame, width, height))
        .flatten();
    let camera_matrices = if shader.is_some() || volume_pass.is_some() {
        camera_matrices_for_raster(camera)?
    } else {
        [(DMat4::IDENTITY, DMat4::IDENTITY); 2]
    };
    let eevee_world_ray = (eevee_world_rays && world.is_some())
        .then(|| EeveeWorldRayFrame::new(camera, &frame, width, height))
        .flatten();
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

    let has_volume_only = ao_bvh.is_some()
        && prepared_materials
            .iter()
            .any(crate::shader::PreparedMaterial::is_volume_only);

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
    let linear_rgba_associated = matches!(mode, Mode::Beauty);
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
            FilmFilter::new(radius, width, height, eye_split, exposure_scale)
        })
        .transpose()?;
    let mut single_sample_subsurface = if film.is_none() && matches!(mode, Mode::Beauty) {
        Some(allocate_filled(
            pixel_count,
            None,
            "EEVEE subsurface closure frame",
        )?)
    } else {
        None
    };
    // Film jitter is shared across pixels; other sample patterns vary per pixel.
    let camera_light_raster_samples = match (film.as_ref(), camera_lights) {
        (Some(film), Some(lights)) => Some(lights.prepare_raster_samples(
            (0..samples).map(|sample_index| film.jitter(film_sample_offset(sample_index))),
            camera.stereo_mode == StereoMode::SideBySide,
        )?),
        _ => None,
    };

    let bands = film.as_ref().map_or_else(Vec::new, |film| {
        super::path::band_rows(film.height, band_count)
    });
    let worker_count = if pixel_count > 0 && bands.len() > 1 {
        bands.len()
    } else {
        0
    };
    let mut band_workers = make_film_band_workers(worker_count, shader)?;
    let parallel = pixel_count > 0 && band_workers.len() == bands.len() && bands.len() > 1;

    let mut candidates = Vec::new();
    if parallel {
        let center_context = FilmCenterRenderContext {
            triangles,
            prepared: &prepared,
            visibility: &visibility,
            ao_bvh: ao_bvh.as_ref(),
            frame: &frame,
            camera,
            camera_matrices: &camera_matrices,
            view_index,
            width,
            height,
            width_usize,
            prepared_materials,
            has_volume_only,
        };
        band_workers = render_film_center_bands(
            &center_context,
            &bands,
            band_workers,
            FilmCenterBandBuffers {
                ids: &mut ids,
                depths: &mut depths,
                elements: &mut elements,
                normals: &mut normals,
                albedo: &mut albedo,
                ambient_occlusion: &mut ambient_occlusion,
            },
        )?;
        if let Some(shader) = shader {
            for worker in &mut band_workers {
                shader.merge_worker_warnings(worker.take_new_warnings());
            }
        }
    }
    let center_rows = if parallel { 0..0 } else { 0..height };
    let center_sample = FilmSampleJitter::new(
        0,
        0.5,
        0.5,
        shader.is_some_and(SurfaceShader::needs_eevee_gbuffer_context),
    );
    for y in center_rows {
        for x in 0..width {
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
            let center_ray = ray_for_pixel_in_frame(&frame, camera.position, width, height, x, y)?;
            let hit_context = if shader.is_some() {
                Some(raster_hit_context(
                    &frame,
                    camera,
                    width,
                    height,
                    &RasterPixelSample {
                        cached: None,
                        index: 0,
                        count: 1,
                        needs_eevee_gbuffer_context: shader
                            .is_some_and(SurfaceShader::needs_eevee_gbuffer_context),
                        gbuffer_sample_offset: center_sample.gbuffer_sample_offset,
                        gbuffer_closure_sample: center_sample.gbuffer_closure_sample,
                        eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
                        x,
                        y,
                        sample_x: 0.5,
                        sample_y: 0.5,
                    },
                    &camera_matrices,
                    view_index,
                )?)
            } else {
                None
            };
            let center_hit = match center_ray {
                Some(ray) => nearest_visible_hit(
                    &prepared,
                    &visibility,
                    &mut candidates,
                    ray,
                    &frame,
                    shader.map(RasterShader::Main),
                    hit_context.as_ref(),
                    None,
                )?,
                None => None,
            };
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
                    let point = center_ray.origin + center_ray.direction * hit.distance;
                    let visibility = ambient_visibility(
                        bvh,
                        triangles,
                        prepared_materials,
                        point,
                        geometric,
                        shading,
                        1.0,
                        has_volume_only,
                    );
                    ambient_occlusion[pixel_index] =
                        linear_color_to_f32([visibility, visibility, visibility, 1.0])[0];
                }
            }
            if raytrace_settings.is_none()
                && film.is_none()
                && matches!(mode, Mode::Solid | Mode::Beauty | Mode::Wire)
            {
                let mut average = [0.0_f64; 4];
                let display_average_needed = film.is_none();
                let associated_average_needed = linear_rgba_associated && display_average_needed;
                let mut average_associated = [0.0_f64; 4];
                for sample_index in 0..samples {
                    let (sample_x, sample_y) = sample_offset(seed, x, y, sample_index);
                    let gbuffer_sample = FilmSampleJitter::new(
                        sample_index,
                        sample_x,
                        sample_y,
                        shader.is_some_and(SurfaceShader::needs_eevee_gbuffer_context),
                    );
                    let (sample_color, sample_linear, subsurface, _) = RasterSampleContext {
                        triangles: &prepared,
                        visibility: &visibility,
                        candidates: &mut candidates,
                        frame: &frame,
                        camera,
                        camera_matrices: &camera_matrices,
                        width,
                        height,
                        mode,
                        background: background_linear,
                        world,
                        eevee_world_ray: eevee_world_ray.as_ref(),
                        shader: shader.map(RasterShader::Main),
                        camera_lights,
                        camera_light_raster_samples: camera_light_raster_samples.as_ref(),
                        view_index,
                        volume_pass: &mut volume_pass,
                        linear_rgba_associated,
                        raytrace_layer: None,
                    }
                    .shade(&RasterPixelSample {
                        cached: (sample_index == 0).then_some(CachedRasterHit {
                            ray: center_ray,
                            hit: center_hit,
                        }),
                        index: sample_index,
                        count: samples,
                        needs_eevee_gbuffer_context: shader
                            .is_some_and(SurfaceShader::needs_eevee_gbuffer_context),
                        gbuffer_sample_offset: gbuffer_sample.gbuffer_sample_offset,
                        gbuffer_closure_sample: gbuffer_sample.gbuffer_closure_sample,
                        eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
                        x,
                        y,
                        sample_x,
                        sample_y,
                    })?;
                    if let Some(records) = single_sample_subsurface.as_mut() {
                        records[pixel_index] = subsurface;
                    }
                    let count = f64::from(sample_index + 1);
                    let previous_weight = 1.0 - 1.0 / count;
                    for channel in 0..4 {
                        if display_average_needed {
                            average[channel] =
                                average[channel] * previous_weight + sample_color[channel] / count;
                        }
                        if associated_average_needed {
                            average_associated[channel] = average_associated[channel]
                                * previous_weight
                                + sample_linear[channel] / count;
                        }
                    }
                }
                if display_average_needed {
                    let color = color_to_srgba(average);
                    rgba[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&color);
                    linear_rgba[pixel_index] = linear_color_to_f32(if associated_average_needed {
                        average_associated
                    } else {
                        average
                    });
                }
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
    if let Some(settings) = raytrace_settings {
        let render_context = FilmSampleRenderContext {
            triangles: &prepared,
            visibility: &visibility,
            frame: &frame,
            camera,
            camera_matrices: &camera_matrices,
            width,
            width_usize,
            height,
            mode,
            background: background_linear,
            world,
            eevee_world_ray: eevee_world_ray.as_ref(),
            camera_lights,
            camera_light_raster_samples: camera_light_raster_samples.as_ref(),
            view_index,
            sample_count: samples,
            linear_rgba_associated,
        };
        return rasterize_eevee_raytraced_samples(
            RasterOutput {
                rgba,
                linear_rgba,
                linear_rgba_associated,
                ids,
                depths,
                elements,
                normals,
                albedo,
                emission,
                ambient_occlusion,
            },
            EeveeRaytracedSamplesRequest {
                render: &render_context,
                settings,
                seed,
                film,
                subsurface_frame: eevee_sss_frame,
                shader: shader.map(RasterShader::Main),
                candidates: &mut candidates,
                volume_pass: &mut volume_pass,
            },
        );
    }
    if let Some(sample_subsurface) = single_sample_subsurface.as_ref() {
        let has_subsurface = sample_subsurface.iter().any(Option::is_some);
        resolve_eevee_subsurface_frame(
            sample_subsurface,
            &mut linear_rgba,
            eevee_sss_frame,
            0,
            [0.5, 0.5],
        )?;
        if has_subsurface {
            for (pixel, associated) in linear_rgba.iter().copied().enumerate() {
                let alpha = associated[3];
                let display = if alpha > 0.0 {
                    [
                        associated[0] / alpha,
                        associated[1] / alpha,
                        associated[2] / alpha,
                        alpha,
                    ]
                } else {
                    associated
                };
                rgba[pixel * 4..pixel * 4 + 4]
                    .copy_from_slice(&color_to_srgba(display.map(f64::from)));
            }
        }
    }

    if let Some(film) = film.as_mut() {
        linear_rgba.fill([0.0; 4]);
        let render_context = FilmSampleRenderContext {
            triangles: &prepared,
            visibility: &visibility,
            frame: &frame,
            camera,
            camera_matrices: &camera_matrices,
            width,
            height,
            width_usize,
            mode,
            background: background_linear,
            world,
            eevee_world_ray: eevee_world_ray.as_ref(),
            camera_lights,
            camera_light_raster_samples: camera_light_raster_samples.as_ref(),
            view_index,
            sample_count: samples,
            linear_rgba_associated,
        };
        let volume_template = if parallel {
            volume_pass
                .as_deref()
                .map(super::path::RealtimeVolumePass::worker_template)
        } else {
            None
        };
        for sample_index in 0..samples {
            film.sample_subsurface.fill(None);
            let (sample_x, sample_y) = film.jitter(film_sample_offset(sample_index));
            let sample_jitter = FilmSampleJitter::new(
                sample_index,
                sample_x,
                sample_y,
                shader.is_some_and(SurfaceShader::needs_eevee_gbuffer_context),
            );
            if parallel {
                let worker_outcomes = std::thread::scope(|scope| {
                    let mut handles = Vec::with_capacity(bands.len());
                    let mut sample_rest = film.sample_color.as_mut_slice();
                    let mut subsurface_rest = film.sample_subsurface.as_mut_slice();
                    for (rows, mut worker) in bands.iter().cloned().zip(band_workers) {
                        let band_len = rows.len() * width_usize;
                        let (sample_rows, sample_tail) =
                            std::mem::take(&mut sample_rest).split_at_mut(band_len);
                        sample_rest = sample_tail;
                        let (subsurface_rows, subsurface_tail) =
                            std::mem::take(&mut subsurface_rest).split_at_mut(band_len);
                        subsurface_rest = subsurface_tail;
                        let render_context = &render_context;
                        handles.push(
                            std::thread::Builder::new()
                                .stack_size(super::path::WORKER_STACK_BYTES)
                                .spawn_scoped(scope, move || {
                                    let mut volume_owned = volume_template.map(|template| {
                                        template
                                            .new_pass(&mut worker.volume_scratch, &worker.warnings)
                                    });
                                    let mut volume_pass = volume_owned.as_mut();
                                    let sample_shader = worker.shader.as_deref().map(|shader| {
                                        RasterShader::Worker(shader, &worker.warnings)
                                    });
                                    let result = render_context.render_rows(
                                        rows,
                                        Some(sample_rows),
                                        None,
                                        subsurface_rows,
                                        None,
                                        None,
                                        sample_jitter,
                                        None,
                                        None,
                                        &mut worker.candidates,
                                        sample_shader,
                                        &mut volume_pass,
                                    );
                                    (result, worker)
                                }),
                        );
                    }
                    let mut workers = Vec::with_capacity(handles.len());
                    let mut first_error = None;
                    for handle in handles {
                        match handle {
                            Ok(handle) => match handle.join() {
                                Ok((Ok(()), worker)) => workers.push(worker),
                                Ok((Err(error), worker)) => {
                                    workers.push(worker);
                                    if first_error.is_none() {
                                        first_error = Some(error);
                                    }
                                }
                                Err(_) if first_error.is_none() => {
                                    first_error = Some(PotError::new(
                                        ErrorCode::InternalError,
                                        "render worker panicked",
                                    ));
                                }
                                Err(_) => {}
                            },
                            Err(error) => {
                                if first_error.is_none() {
                                    first_error = Some(PotError::new(
                                        ErrorCode::InternalError,
                                        format!("cannot start a render worker: {error}"),
                                    ));
                                }
                            }
                        }
                    }
                    match first_error {
                        Some(error) => Err(error),
                        None => Ok(workers),
                    }
                })?;
                band_workers = worker_outcomes;
                for worker in &mut band_workers {
                    let warnings = worker.take_new_warnings();
                    if let Some(shader) = shader {
                        shader.merge_worker_warnings(warnings);
                    } else if let Some(pass) = volume_pass.as_deref() {
                        pass.merge_worker_warnings(warnings);
                    }
                }
            } else {
                render_context.render_rows(
                    0..film.height,
                    Some(film.sample_color.as_mut_slice()),
                    None,
                    &mut film.sample_subsurface,
                    None,
                    None,
                    sample_jitter,
                    None,
                    None,
                    &mut candidates,
                    shader.map(RasterShader::Main),
                    &mut volume_pass,
                )?;
            }
            resolve_eevee_subsurface_frame(
                &film.sample_subsurface,
                &mut film.sample_color,
                eevee_sss_frame,
                sample_index,
                [sample_x, sample_y],
            )?;
            for pixel in 0..film.sample_color.len() {
                let color = film.sample_color[pixel].map(f64::from);
                store_film_sample(&mut film.sample_color, pixel, color);
            }
            film.resolve_sample(sample_index, sample_x, sample_y, &mut linear_rgba);
        }
        for (pixel_index, associated) in linear_rgba.iter().copied().enumerate() {
            let associated = associated.map(f64::from);
            let alpha = associated[3];
            let display = if alpha > 0.0 {
                [
                    associated[0] / alpha,
                    associated[1] / alpha,
                    associated[2] / alpha,
                    alpha,
                ]
            } else {
                associated
            };
            rgba[pixel_index * 4..pixel_index * 4 + 4].copy_from_slice(&color_to_srgba(display));
        }
    }

    Ok(RasterOutput {
        rgba,
        linear_rgba,
        linear_rgba_associated,
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
    exposure_scale: f32,
    shader: Option<&dyn SurfaceShader>,
    prepared_materials: &[crate::shader::PreparedMaterial],
    camera_lights: Option<&super::path::RealtimeCameraLightContext<'_>>,
    volume_pass: Option<&mut super::path::RealtimeVolumePass<'_, '_>>,
    eevee_world_rays: bool,
) -> Result<RasterOutput> {
    let mut output = rasterize_with_samples_for_eye(RasterSamplesRequest {
        triangles,
        camera,
        width,
        height,
        mode,
        background,
        eevee_world_rays,
        world,
        samples,
        seed,
        filter_size,
        exposure_scale,
        shader,
        raytrace_settings: match shader {
            Some(shader) => RasterShader::Main(shader).eevee_raytracing_settings()?,
            None => None,
        },
        prepared_materials,
        camera_lights: camera_lights.map(|lights| lights.view(0)),
        view_index: 0,
        band_count: std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get),
        volume_pass,
    })?;
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
                        linear_color_to_f32(if output.linear_rgba_associated {
                            associated_rgba(color)
                        } else {
                            color
                        })
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
        1.0,
        None,
        &[],
        None,
        None,
        false,
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
fn camera_light_eye_index(stereo_mode: StereoMode, x: u32, width: u32, view_index: usize) -> usize {
    if stereo_mode == StereoMode::SideBySide {
        usize::from(x >= width / 2)
    } else {
        view_index
    }
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

fn eevee_subsurface_frame(
    camera: &Camera,
    frame: &CameraFrame,
    width: u32,
    height: u32,
) -> Option<EeveeSubsurfaceFrame> {
    let projection = match frame.projection {
        ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } => EeveeSubsurfaceProjection::Perspective {
            horizontal_scale,
            vertical_scale,
        },
        ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } => EeveeSubsurfaceProjection::Orthographic {
            half_width,
            half_height,
        },
        ProjectionFrame::Equirectangular { .. }
        | ProjectionFrame::FisheyeEquidistant { .. }
        | ProjectionFrame::FisheyeEquisolid { .. } => return None,
    };
    Some(EeveeSubsurfaceFrame {
        camera_position: camera.position,
        forward: frame.forward,
        right: frame.right,
        up: frame.up,
        near: frame.near,
        far: frame.far,
        shift: frame.shift,
        projection,
        width,
        height,
        eye_split: (frame.stereo_mode == StereoMode::SideBySide).then_some(width / 2),
    })
}

/// The camera's geometric frustum for one rendered view; unbounded projections conservatively
/// return `None`.
pub(crate) fn camera_frustum(
    camera: &Camera,
    width: u32,
    height: u32,
) -> Result<Option<CameraFrustum>> {
    let frame = camera_frame(camera, width, height)?;
    let point = |depth: f64, x: f64, y: f64| {
        let (half_width, half_height) = match frame.projection {
            ProjectionFrame::Orthographic {
                half_width,
                half_height,
            } => (half_width, half_height),
            ProjectionFrame::Perspective {
                horizontal_scale,
                vertical_scale,
            } => (horizontal_scale * depth, vertical_scale * depth),
            _ => unreachable!("non-planar projections have no culling frustum"),
        };
        camera.position
            + frame.forward * depth
            + frame.right * ((x + 2.0 * frame.shift[0]) * half_width)
            + frame.up * ((y + 2.0 * frame.shift[1]) * half_height)
    };
    if !matches!(
        frame.projection,
        ProjectionFrame::Orthographic { .. } | ProjectionFrame::Perspective { .. }
    ) {
        return Ok(None);
    }
    let corners = [
        point(frame.near, -1.0, -1.0),
        point(frame.near, -1.0, 1.0),
        point(frame.far, -1.0, 1.0),
        point(frame.far, -1.0, -1.0),
        point(frame.near, 1.0, -1.0),
        point(frame.near, 1.0, 1.0),
        point(frame.far, 1.0, 1.0),
        point(frame.far, 1.0, -1.0),
    ];
    if corners
        .iter()
        .copied()
        .any(|corner| !is_finite_vec3(corner))
    {
        return Ok(None);
    }
    let center = corners.iter().copied().sum::<DVec3>() / corners.len() as f64;
    let inward_plane = |point: DVec3, first: DVec3, second: DVec3| {
        let mut plane = frustum_plane(point, first - point, second - point);
        if plane.distance(center) < 0.0 {
            plane.normal = -plane.normal;
            plane.offset = -plane.offset;
        }
        plane
    };
    let planes = [
        inward_plane(corners[0], corners[1], corners[3]),
        inward_plane(corners[0], corners[4], corners[3]),
        inward_plane(corners[0], corners[1], corners[4]),
        inward_plane(corners[4], corners[5], corners[7]),
        inward_plane(corners[1], corners[2], corners[5]),
        inward_plane(corners[3], corners[2], corners[7]),
    ];
    if planes.iter().any(|plane| {
        plane.normal == DVec3::ZERO || !is_finite_vec3(plane.normal) || !plane.offset.is_finite()
    }) {
        return Ok(None);
    }
    Ok(Some(CameraFrustum { corners, planes }))
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
    has_volume_only: bool,
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
        let occluded = if has_volume_only {
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
            occluded
        } else {
            bvh.any_hit_where(triangles, origin, direction, 1.0e-5, distance, &|mask| {
                mask & RayKind::Shadow.bit() != 0
            })
        };
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

#[derive(Clone, Copy)]
struct RaySampleOffsets {
    pixel: [f64; 2],
    lens: [f64; 2],
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
    ray_for_subpixel_sample_in_frame(
        frame,
        camera_position,
        width,
        height,
        x,
        y,
        RaySampleOffsets {
            pixel: [sample_x, sample_y],
            lens: [sample_x, sample_y],
        },
    )
}

fn ray_for_subpixel_sample_in_frame(
    frame: &CameraFrame,
    camera_position: DVec3,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    sample: RaySampleOffsets,
) -> Result<Option<Ray>> {
    let sample_x = sample.pixel[0];
    let sample_y = sample.pixel[1];
    let lens_sample_x = sample.lens[0];
    let lens_sample_y = sample.lens[1];
    if !pixel_in_bounds(width, height, x, y)
        || !sample
            .pixel
            .iter()
            .chain(sample.lens.iter())
            .all(|value| value.is_finite())
    {
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
    if let Some(lens_sample) = frame.lens_sample(lens_sample_x, lens_sample_y) {
        let plane_distance =
            lens_sample.focus_distance / ray.direction.dot(frame.forward).max(1.0e-6);
        let focus_point = ray.origin + ray.direction * plane_distance;
        ray.origin += frame.right * lens_sample.aperture.x + frame.up * lens_sample.aperture.y;
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

/// EEVEE's combined-film resolve uses one RGBA16F input frame and one RGBA16F history frame;
/// spatial weights and their history are float32.
#[derive(Clone, Copy, Default)]
struct FilmTap {
    x: i64,
    y: i64,
    weight: f32,
}

const MAX_FILM_TAPS: usize = 16;

struct FilmFilter {
    radius: f64,
    width: usize,
    height: usize,
    exposure_scale: f32,
    /// The current unquantized Combined sample, quantized after the SSS frame pass.
    sample_color: Vec<[f32; 4]>,
    /// One reusable SSS closure record, including its own depth, for each TAA sample.
    sample_subsurface: Vec<Option<EeveeSubsurfacePixel>>,
    /// The accumulation-weight image persists between film samples.
    history_weight: Vec<f32>,
    /// The column where the right eye starts; each eye has an independent film.
    eye_split: Option<usize>,
}

impl FilmFilter {
    /// EEVEE's pixel jitter, converted to a position inside the rendered pixel.
    fn jitter(&self, offset: (f64, f64)) -> (f64, f64) {
        if self.radius >= std::f64::consts::FRAC_1_SQRT_2 {
            return (1.0 - offset.0, offset.1);
        }
        let scale = 0.5 * self.radius * offset.0.sqrt();
        let theta = std::f64::consts::TAU * offset.1;
        (0.5 - scale * theta.cos(), 0.5 + scale * theta.sin())
    }

    fn new(
        radius: f64,
        width: u32,
        height: u32,
        eye_split: Option<usize>,
        exposure_scale: f32,
    ) -> Result<Self> {
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
            exposure_scale,
            sample_color: allocate_filled(pixel_count, [0.0_f32; 4], "film input frame")?,
            sample_subsurface: allocate_filled(
                pixel_count,
                None,
                "EEVEE subsurface closure frame",
            )?,
            history_weight: allocate_filled(pixel_count, 0.0_f32, "film weight history")?,
            eye_split,
        })
    }

    fn sample_taps(
        &self,
        sample_index: u32,
        sample_x: f64,
        sample_y: f64,
    ) -> ([FilmTap; MAX_FILM_TAPS], usize, f32) {
        let mut taps = [FilmTap::default(); MAX_FILM_TAPS];
        let mut sample_weight_total = 0.0_f32;
        let tap_count = if self.radius < LARGE_FILTER_RADIUS {
            if self.radius < 0.01 {
                taps[0] = FilmTap {
                    x: 0,
                    y: 0,
                    weight: 1.0,
                };
                sample_weight_total = 1.0;
                1
            } else {
                let radius = self.radius as f32;
                let jitter_x = 0.5 - sample_x as f32;
                let jitter_y = sample_y as f32 - 0.5;
                let reach = radius.ceil() as i64;
                let radius_squared = radius * radius;
                let mut count = 0;
                let mut closest = 0;
                let mut closest_distance = f32::MAX;
                for y in -reach..=reach {
                    for x in -reach..=reach {
                        let distance_x = x as f32 - jitter_x;
                        let distance_y = y as f32 - jitter_y;
                        let distance_squared = distance_x * distance_x + distance_y * distance_y;
                        if distance_squared >= radius_squared {
                            continue;
                        }
                        let weight = film_filter_weight(radius, distance_squared);
                        sample_weight_total += weight;
                        taps[count] = FilmTap { x, y, weight };
                        if distance_squared < closest_distance {
                            closest = count;
                            closest_distance = distance_squared;
                        }
                        count += 1;
                    }
                }
                if closest != 0 {
                    taps.swap(0, closest);
                }
                if count <= 9 { 9 } else { MAX_FILM_TAPS }
            }
        } else {
            let jitter_x = 0.5 - sample_x as f32;
            let jitter_y = sample_y as f32 - 0.5;
            for (tap, (x, y)) in taps
                .iter_mut()
                .zip(large_filter_taps(self.radius, sample_index))
            {
                let distance_x = x as f32 - jitter_x;
                let distance_y = y as f32 - jitter_y;
                let weight = film_filter_weight(
                    self.radius as f32,
                    distance_x * distance_x + distance_y * distance_y,
                );
                *tap = FilmTap { x, y, weight };
                sample_weight_total += weight;
            }
            MAX_FILM_TAPS
        };
        (taps, tap_count, sample_weight_total)
    }

    /// Resolves one TAA sample into the float32 weight image and RGBA16F combined history.
    fn resolve_sample(
        &mut self,
        sample_index: u32,
        sample_x: f64,
        sample_y: f64,
        history: &mut [[f32; 4]],
    ) {
        let (taps, tap_count, sample_weight_total) =
            self.sample_taps(sample_index, sample_x, sample_y);
        let (width, height, exposure_scale, eye_split) =
            (self.width, self.height, self.exposure_scale, self.eye_split);
        let sample_color = &self.sample_color;
        let history_weight = &mut self.history_weight;
        for y in 0..height {
            for x in 0..width {
                let pixel = y * width + x;
                let width_i64 = width as i64;
                let (column_start, column_end) = match eye_split {
                    Some(split) if x < split => (0, split as i64),
                    Some(split) => (split as i64, width_i64),
                    None => (0, width_i64),
                };
                let mut color_sum = [0.0_f32; 4];
                let mut color_weight = 0.0_f32;
                for tap in taps[..tap_count].iter().rev() {
                    if tap.weight == 0.0 {
                        continue;
                    }
                    let source_x = (x as i64 + tap.x).clamp(column_start, column_end - 1);
                    let source_y = (y as i64 - tap.y).clamp(0, height as i64 - 1);
                    let source = sample_color[source_y as usize * width + source_x as usize];
                    let color = ycocg_from_sample(source);
                    let weight = tap.weight * film_luma_weight(color[0], exposure_scale);
                    for channel in 0..4 {
                        color_sum[channel] += color[channel] * weight;
                    }
                    color_weight += weight;
                }

                let previous = history[pixel];
                let previous_ycocg = ycocg_from_scene_linear(previous);
                let previous_weight = history_weight[pixel];
                let weight_dst =
                    film_luma_weight(previous_ycocg[0], exposure_scale) * previous_weight;
                let color_src = color_sum.map(|channel| channel / color_weight);
                let weight_sum = weight_dst + color_weight;
                let blended = std::array::from_fn(|channel| {
                    (previous_ycocg[channel] * weight_dst + color_src[channel] * color_weight)
                        / weight_sum
                });
                let mut color = scene_linear_from_ycocg(blended);
                if color[3] > 0.995 {
                    color[3] = 1.0;
                }
                if color.iter().any(|channel| channel.is_nan()) {
                    color = [0.0, 0.0, 0.0, 1.0];
                }
                for channel in &mut color {
                    if (*channel).partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                        *channel = 0.0;
                    }
                }
                history[pixel] = color.map(store_film_history_f16);
                history_weight[pixel] = previous_weight + sample_weight_total;
            }
        }
    }
}

fn store_film_sample(destination: &mut [[f32; 4]], pixel: usize, color: [f64; 4]) {
    let mut stored = color.map(|channel| store_film_f16_truncated(channel as f32));
    stored[3] = store_film_f16_truncated(1.0_f32 - color[3] as f32);
    destination[pixel] = stored;
}

fn ycocg_from_scene_linear(color: [f32; 4]) -> [f32; 4] {
    [
        color[0] + 2.0 * color[1] + color[2],
        2.0 * color[0] - 2.0 * color[2],
        -color[0] + 2.0 * color[1] - color[2],
        color[3],
    ]
}
fn ycocg_from_sample(color: [f32; 4]) -> [f32; 4] {
    let mut color = ycocg_from_scene_linear(color);
    color[3] = (1.0 - color[3]).clamp(0.0, 1.0);
    color
}

fn scene_linear_from_ycocg(color: [f32; 4]) -> [f32; 4] {
    [
        (color[0] + color[1] - color[2]) * 0.25,
        (color[0] + color[2]) * 0.25,
        (color[0] - color[1] - color[2]) * 0.25,
        color[3],
    ]
}

fn film_luma_weight(luma: f32, exposure_scale: f32) -> f32 {
    1.0 / (4.0 + luma * exposure_scale)
}

fn store_film_history_f16(value: f32) -> f32 {
    let patched = f32::from_bits(value.to_bits().wrapping_add(0x1000));
    store_film_f16_truncated(patched)
}

/// Native EEVEE 16-bit render targets and image stores truncate float mantissas.
fn store_film_f16_truncated(value: f32) -> f32 {
    exr::prelude::f16::from_bits(denoise::store_f16_bits_rtz(value)).to_f32()
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

/// A film sample's offset inside its pixel: EEVEE's Halton (2, 3) sequence at index `sample_index + 1`; it adds
/// `0.5_f32` and `2.0_f32 / 3.0_f32` to the f64 dimensions, casts each sum to f32, then applies `fractf` before exposing
/// the filter dimensions (`eevee_sampling.cc`, `Sampling::step`, with `BLI_halton_2d` from `blenlib/intern/rand.cc`).
/// The pattern is the same for every pixel and needs no seed.
fn film_sample_offset(sample_index: u32) -> (f64, f64) {
    let index = u64::from(sample_index) + 1;
    let x = (halton_radical_inverse(2, index) + f64::from(0.5_f32)) as f32;
    let y = (halton_radical_inverse(3, index) + f64::from(2.0_f32 / 3.0_f32)) as f32;
    (f64::from(x - x.floor()), f64::from(y - y.floor()))
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
fn film_filter_weight(radius: f32, distance_squared: f32) -> f32 {
    const SIGMA: f32 = 0.284;
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
        AlphaMode::Blend | AlphaMode::Dithered => color[3].clamp(0.0, 1.0),
        AlphaMode::Clip => {
            if color[3] >= triangle.alpha_threshold {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn raster_radiance_weight(opaque_share: f64, dither: Option<EeveeDitherSample>) -> f64 {
    opaque_share.clamp(0.0, 1.0) * dither.map_or(1.0, |dither| f64::from(dither.alpha_reciprocal))
}

/// EEVEE 5.2.2 `hashed_transparency::alpha_threshold` from the pinned 55-line shader kernel.
fn eevee_dither_alpha_threshold(
    point: DVec3,
    differentials: HitDifferentials,
    hash_offset: f32,
) -> f32 {
    fn fract(value: f32) -> f32 {
        value - value.floor()
    }
    fn hash(point: [f32; 2]) -> f32 {
        let x = 17.0 * point[0] + 0.1 * point[1];
        let y = 13.0 * point[1] + point[0];
        fract(1.0e4 * x.sin() * (0.1 + y.sin().abs()))
    }

    let point = point.to_array().map(|value| value as f32);
    let dx = Vec3::new(
        differentials.world_dx.x as f32,
        differentials.world_dx.y as f32,
        differentials.world_dx.z as f32,
    );
    let dy = Vec3::new(
        differentials.world_dy.x as f32,
        differentials.world_dy.y as f32,
        differentials.world_dy.z as f32,
    );
    let dx_length = dx.length();
    let dy_length = dy.length();
    let max_deriv = dx_length.max(dy_length);
    let hash_scale = 0.1_f32;
    let pix_scale = 1.0 / (hash_scale * max_deriv);
    let pix_scale_log = pix_scale.log2();
    let pix_scales = [pix_scale_log.floor().exp2(), pix_scale_log.ceil().exp2()];
    let low_cell = [
        (pix_scales[0] * point[0]).floor(),
        (pix_scales[0] * point[1]).floor(),
        (pix_scales[0] * point[2]).floor(),
    ];
    let high_cell = [
        (pix_scales[1] * point[0]).floor(),
        (pix_scales[1] * point[1]).floor(),
        (pix_scales[1] * point[2]).floor(),
    ];
    let low_inner_hash_arg = [low_cell[0], low_cell[1]];
    let high_inner_hash_arg = [high_cell[0], high_cell[1]];
    let low_inner_hash = hash(low_inner_hash_arg);
    let high_inner_hash = hash(high_inner_hash_arg);
    let low_outer_hash_arg = [low_inner_hash, low_cell[2]];
    let high_outer_hash_arg = [high_inner_hash, high_cell[2]];
    let alpha = [hash(low_outer_hash_arg), hash(high_outer_hash_arg)];
    let fac = fract(pix_scale_log);
    let x = alpha[0] * (1.0 - fac) + alpha[1] * fac;
    let a = fac.min(1.0 - fac);
    let one_a = 1.0 - a;
    let denom = 1.0 / (2.0 * a * one_a);
    let one_x = 1.0 - x;
    let cases = [
        (x * x) * denom,
        (x - 0.5 * a) / one_a,
        1.0 - (one_x * one_x * denom),
    ];
    let selected_case = usize::from(x >= a) + usize::from(x >= one_a);
    let threshold_before_jitter = cases[selected_case];
    let threshold_after_jitter = fract(threshold_before_jitter + hash_offset);
    threshold_after_jitter.clamp(1.0e-6, 1.0)
}

/// The physical share that can reach the next surface after opacity and the zero-transmittance Holdout closure.
/// EEVEE attenuates the current surface radiance by `1 - g_holdout` separately from film alpha and transmittance.
fn physical_surface_pass(opacity: f64, holdout: f64) -> f64 {
    (1.0 - opacity.clamp(0.0, 1.0) - holdout.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Shading normal at a hit: the blended corner normal for smooth triangles, else the geometric one.
fn shading_normal(triangle: &PreparedTriangle, hit: Hit) -> DVec3 {
    triangle.source.normals.map_or(triangle.normal, |corners| {
        blend_corner_normals(corners, hit.barycentric, triangle.normal)
    })
}

#[derive(Clone, Copy)]
struct RasterHitContext<'a> {
    sample_index: u32,
    raytrace_layer: Option<EeveeRaytraceLayer>,
    camera_matrices: Option<&'a (DMat4, DMat4)>,
    x_neighbor: Option<(Ray, f64)>,
    y_neighbor: Option<(Ray, f64)>,
    gbuffer_frag_coord: Option<[f32; 2]>,
    gbuffer_sample_offset: Option<[f32; 3]>,
    gbuffer_closure_random: Option<f32>,
    eevee_raytrace_feedback: EeveeRaytraceFeedback,
}

fn raster_hit_context<'a>(
    frame: &CameraFrame,
    camera: &Camera,
    width: u32,
    height: u32,
    sample: &RasterPixelSample,
    camera_matrices: &'a [(DMat4, DMat4); 2],
    view_index: usize,
) -> Result<RasterHitContext<'a>> {
    let (view_start, view_end) = if camera.stereo_mode == StereoMode::SideBySide {
        let split = width / 2;
        if sample.x < split {
            (0, split)
        } else {
            (split, width)
        }
    } else {
        (0, width)
    };
    // EEVEE's fine derivatives use the adjacent lane in each 2x2 fragment pair.
    // Shift subpixel offsets for helper lanes outside the view instead of crossing stereo eyes.
    let local_x = sample.x - view_start;
    let x_neighbor = if local_x.is_multiple_of(2) {
        if sample.x + 1 < view_end {
            (
                sample.x + 1,
                sample.y,
                sample.sample_x,
                sample.sample_y,
                1.0,
            )
        } else {
            (
                sample.x,
                sample.y,
                sample.sample_x + 1.0,
                sample.sample_y,
                1.0,
            )
        }
    } else {
        (
            sample.x - 1,
            sample.y,
            sample.sample_x,
            sample.sample_y,
            -1.0,
        )
    };
    let y_neighbor = if sample.y.is_multiple_of(2) {
        if sample.y + 1 < height {
            (
                sample.x,
                sample.y + 1,
                sample.sample_x,
                sample.sample_y,
                1.0,
            )
        } else {
            (
                sample.x,
                sample.y,
                sample.sample_x,
                sample.sample_y + 1.0,
                1.0,
            )
        }
    } else {
        (
            sample.x,
            sample.y - 1,
            sample.sample_x,
            sample.sample_y,
            -1.0,
        )
    };
    let (neighbor_x, neighbor_y, neighbor_sample_x, neighbor_sample_y, x_sign) = x_neighbor;
    let x_neighbor = ray_for_subpixel_sample_in_frame(
        frame,
        camera.position,
        width,
        height,
        neighbor_x,
        neighbor_y,
        RaySampleOffsets {
            pixel: [neighbor_sample_x, neighbor_sample_y],
            lens: [sample.sample_x, sample.sample_y],
        },
    )?
    .map(|ray| (ray, x_sign));
    let (neighbor_x, neighbor_y, neighbor_sample_x, neighbor_sample_y, y_sign) = y_neighbor;
    let y_neighbor = ray_for_subpixel_sample_in_frame(
        frame,
        camera.position,
        width,
        height,
        neighbor_x,
        neighbor_y,
        RaySampleOffsets {
            pixel: [neighbor_sample_x, neighbor_sample_y],
            lens: [sample.sample_x, sample.sample_y],
        },
    )?
    .map(|ray| (ray, y_sign));
    let eye_index = camera_light_eye_index(camera.stereo_mode, sample.x, width, view_index);
    let (gbuffer_frag_coord, gbuffer_sample_offset, gbuffer_closure_random) =
        if sample.needs_eevee_gbuffer_context {
            let frag_coord = [local_x as f32 + 0.5, (height - 1 - sample.y) as f32 + 0.5];
            let sample_offset = sample.gbuffer_sample_offset.ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "EEVEE G-buffer sample has no precomputed sampling offset",
                )
            })?;
            let sampling_closure = sample.gbuffer_closure_sample.ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "EEVEE G-buffer sample has no precomputed closure offset",
                )
            })?;
            (
                Some(frag_coord),
                Some(sample_offset),
                Some(super::eevee_blue_noise::closure_random(
                    frag_coord,
                    sampling_closure,
                )),
            )
        } else {
            (None, None, None)
        };
    Ok(RasterHitContext {
        sample_index: sample.index,
        raytrace_layer: None,
        camera_matrices: camera_matrices.get(eye_index),
        x_neighbor,
        y_neighbor,
        gbuffer_frag_coord,
        gbuffer_sample_offset,
        gbuffer_closure_random,
        eevee_raytrace_feedback: sample.eevee_raytrace_feedback,
    })
}

fn ray_barycentric_on_triangle(ray: Ray, triangle: &Triangle) -> Option<(f64, [f64; 3])> {
    let edge1 = triangle.positions[1] - triangle.positions[0];
    let edge2 = triangle.positions[2] - triangle.positions[0];
    let p_vector = ray.direction.cross(edge2);
    let determinant = edge1.dot(p_vector);
    if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let from_vertex = ray.origin - triangle.positions[0];
    let barycentric_1 = from_vertex.dot(p_vector) * inverse_determinant;
    let q_vector = from_vertex.cross(edge1);
    let barycentric_2 = ray.direction.dot(q_vector) * inverse_determinant;
    let distance = edge2.dot(q_vector) * inverse_determinant;
    let barycentric = [
        1.0 - barycentric_1 - barycentric_2,
        barycentric_1,
        barycentric_2,
    ];
    (distance.is_finite() && barycentric.iter().all(|weight| weight.is_finite()))
        .then_some((distance, barycentric))
}

fn differential_on_neighbor(
    triangle: &Triangle,
    hit: Hit,
    ray: Ray,
    neighbor: Option<(Ray, f64)>,
) -> (DVec3, [f64; 2]) {
    let Some((neighbor_ray, sign)) = neighbor else {
        return (DVec3::ZERO, [0.0; 2]);
    };
    let Some((distance, barycentric)) = ray_barycentric_on_triangle(neighbor_ray, triangle) else {
        return (DVec3::ZERO, [0.0; 2]);
    };
    let point = ray.origin + ray.direction * hit.distance;
    let neighbor_point = neighbor_ray.origin + neighbor_ray.direction * distance;
    let uv = blend_uv(&triangle.uv, hit.barycentric);
    let neighbor_uv = blend_uv(&triangle.uv, barycentric);
    (
        (neighbor_point - point) * sign,
        [
            (neighbor_uv[0] - uv[0]) * sign,
            (neighbor_uv[1] - uv[1]) * sign,
        ],
    )
}

fn realtime_hit_differentials(
    triangle: &Triangle,
    hit: Hit,
    ray: Ray,
    context: Option<&RasterHitContext<'_>>,
) -> HitDifferentials {
    let (world_dx, uv_dx) = differential_on_neighbor(
        triangle,
        hit,
        ray,
        context.and_then(|context| context.x_neighbor),
    );
    let (world_dy, uv_dy) = differential_on_neighbor(
        triangle,
        hit,
        ray,
        context.and_then(|context| context.y_neighbor),
    );
    HitDifferentials {
        world_dx,
        world_dy,
        uv_dx,
        uv_dy,
    }
}

/// The surface sample at a visible hit: the surface shader's when one is present, otherwise the triangle's colour.
/// A DITHERED surface discarded by EEVEE's hashed depth prepass returns `None`.
fn hit_color(
    triangle: &PreparedTriangle,
    hit: Hit,
    ray: Ray,
    shader: Option<RasterShader<'_>>,
    hit_context: Option<&RasterHitContext<'_>>,
) -> Result<Option<SurfaceSample>> {
    let point = ray.origin + ray.direction * hit.distance;
    let mut hit_differentials = None;
    let eevee_gbuffer_context = shader.is_some_and(|shader| shader.needs_eevee_gbuffer_context());
    let mut sample = match shader {
        Some(shader) => {
            let back_facing = triangle.normal.dot(-ray.direction) < 0.0;
            let geometric_normal = if back_facing {
                -triangle.normal
            } else {
                triangle.normal
            };
            let differentials = realtime_hit_differentials(&triangle.source, hit, ray, hit_context);
            hit_differentials = Some(differentials);
            let surface = SurfaceHit {
                point,
                shadow_point: smooth_shadow_origin(&triangle.source, point, hit.barycentric),
                normal: shading_normal(triangle, hit),
                geometric: triangle.normal,
                geometric_normal,
                back_facing,
                realtime_differentials: differentials,
                camera_matrices: hit_context.and_then(|context| context.camera_matrices),
                barycentric: hit.barycentric,
                gbuffer_frag_coord: hit_context.and_then(|context| context.gbuffer_frag_coord),
                gbuffer_sample_offset: hit_context
                    .and_then(|context| context.gbuffer_sample_offset),
                gbuffer_closure_random: hit_context
                    .and_then(|context| context.gbuffer_closure_random),
                view: -ray.direction,
                eevee_raytrace_feedback: hit_context
                    .map_or_else(EeveeRaytraceFeedback::default, |context| {
                        context.eevee_raytrace_feedback
                    }),
            };
            shader.shade(&triangle.source, &surface)?
        }
        _ => SurfaceSample::opaque(triangle.source.color),
    };
    if !eevee_gbuffer_context {
        sample.eevee_dither = None;
    }
    if eevee_gbuffer_context
        && triangle.source.alpha_mode == AlphaMode::Dithered
        && sample.has_surface
    {
        let dither = sample.eevee_dither.ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE DITHERED material has no deferred transparency data",
            )
        })?;
        if dither.transparency > 0.0 {
            let context = hit_context.ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "EEVEE DITHERED coverage requires the per-sample raster context",
                )
            })?;
            let differentials = hit_differentials.unwrap_or_else(|| {
                realtime_hit_differentials(&triangle.source, hit, ray, Some(context))
            });
            let hash_offset = halton_radical_inverse(7, u64::from(context.sample_index) + 1) as f32;
            let threshold_value = eevee_dither_alpha_threshold(point, differentials, hash_offset);
            let passed = dither.transparency <= threshold_value;
            if dither.transparency > 0.0 && !passed {
                return Ok(None);
            }
        }
    }
    Ok(Some(sample))
}

fn blocks_refraction_layer(triangle: &Triangle, sample: &SurfaceSample) -> bool {
    let Some(raytrace) = sample.eevee_raytrace else {
        return false;
    };
    if !sample.has_surface
        || !raytrace.has_gbuffer_route
        || raytrace.screen_refraction != Some(false)
        || sample.holdout_weight > 0.0
        || !is_white(sample.transparent_tint)
    {
        return false;
    }
    match triangle.alpha_mode {
        AlphaMode::Opaque => true,
        AlphaMode::Clip => hit_opacity(triangle, sample.color) > 0.0,
        AlphaMode::Dithered => sample.eevee_dither.is_some(),
        AlphaMode::Blend => false,
    }
}

fn is_visible_surface(triangle: &Triangle, sample: &SurfaceSample) -> bool {
    sample.has_surface
        && ((triangle.alpha_mode == AlphaMode::Dithered && sample.eevee_dither.is_some())
            || hit_opacity(triangle, sample.color) > 0.0
            || sample.holdout_weight > 0.0)
}

fn nearest_visible_hit(
    triangles: &[PreparedTriangle],
    visibility: &super::bvh::Bvh,
    candidates: &mut Vec<usize>,
    ray: Ray,
    frame: &CameraFrame,
    shader: Option<RasterShader<'_>>,
    hit_context: Option<&RasterHitContext<'_>>,
    raytrace_layer: Option<EeveeRaytraceLayer>,
) -> Result<Option<VisibleHit>> {
    // The ray is fixed, so its candidates are too. The nearest is the least (depth, index) pair
    // among them, so their order does not matter.
    visibility.overlapping(ray.origin, ray.direction, candidates);
    let mut cursor: Option<(f64, usize)> = None;
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
        let Some(sample) = hit_color(triangle, hit, ray, shader, hit_context)? else {
            cursor = Some((hit.depth, triangle_index));
            continue;
        };
        let visible = is_visible_surface(&triangle.source, &sample);
        if visible
            && raytrace_layer == Some(EeveeRaytraceLayer::Refraction)
            && blocks_refraction_layer(&triangle.source, &sample)
        {
            return Ok(None);
        }
        if visible
            && raytrace_layer.is_some_and(|layer| !layer.contains(sample.eevee_raytrace.as_ref()))
        {
            cursor = Some((hit.depth, triangle_index));
            continue;
        }
        if visible {
            return Ok(Some(VisibleHit {
                triangle_index,
                hit,
                color: sample.color,
                transparent_tint: sample.transparent_tint,
                holdout_weight: sample.holdout_weight,
                subsurface: sample.subsurface,
                eevee_raytrace: sample.eevee_raytrace,
                eevee_dither: sample.eevee_dither,
            }));
        }
        cursor = Some((hit.depth, triangle_index));
    }
}

/// Whether a transparent tint passes every channel unchanged.
fn is_white(tint: [f64; 3]) -> bool {
    tint.iter().all(|&channel| (channel - 1.0).abs() <= 1.0e-12)
}

fn composite_sample_with_association(
    triangles: &[PreparedTriangle],
    visibility: &super::bvh::Bvh,
    candidates: &mut Vec<usize>,
    ray: Ray,
    world_direction: DVec3,
    frame: &CameraFrame,
    camera: &Camera,
    background: [f64; 4],
    world: Option<WorldRadiance<'_>>,
    shader: Option<RasterShader<'_>>,
    hit_context: Option<&RasterHitContext<'_>>,
    cached_hit: Option<&VisibleHit>,
    camera_lights: Option<super::path::RealtimeCameraLightSample<'_, '_, '_>>,
    mut volume_sample: Option<&mut super::path::RealtimeVolumeSample<'_, '_>>,
    raytrace_layer: Option<EeveeRaytraceLayer>,
) -> Result<CompositeColor> {
    const MAX_ALPHA_LAYERS: usize = 16;
    let mut hits: [Option<VisibleHit>; MAX_ALPHA_LAYERS] = [None; MAX_ALPHA_LAYERS];
    let mut hit_count = 0;
    let mut refraction_blocker: Option<(f64, usize)> = None;
    visibility.overlapping(ray.origin, ray.direction, candidates);
    // Equal depths keep the first layer inserted, so visit the candidates in triangle order.
    candidates.sort_unstable();
    for &triangle_index in candidates.iter() {
        let triangle = &triangles[triangle_index];
        let Some(hit) = intersect_triangle(triangle, ray, frame) else {
            continue;
        };
        let sample = if let Some(cached) = cached_hit.filter(|cached| {
            cached.triangle_index == triangle_index && equal_f64(cached.hit.depth, hit.depth)
        }) {
            SurfaceSample {
                color: cached.color,
                has_surface: true,
                transparent_tint: cached.transparent_tint,
                holdout_weight: cached.holdout_weight,
                subsurface: cached.subsurface,
                eevee_raytrace: cached.eevee_raytrace,
                eevee_dither: cached.eevee_dither,
            }
        } else {
            let Some(sample) = hit_color(triangle, hit, ray, shader, hit_context)? else {
                continue;
            };
            sample
        };
        let dither = if triangle.source.alpha_mode == AlphaMode::Dithered {
            sample.eevee_dither
        } else {
            None
        };
        if !sample.has_surface
            || (dither.is_none()
                && hit_opacity(&triangle.source, sample.color) <= 0.0
                && is_white(sample.transparent_tint)
                && sample.holdout_weight <= 0.0)
        {
            continue;
        }
        let visible_hit = VisibleHit {
            triangle_index,
            hit,
            color: sample.color,
            transparent_tint: sample.transparent_tint,
            holdout_weight: sample.holdout_weight,
            subsurface: sample.subsurface,
            eevee_raytrace: sample.eevee_raytrace,
            eevee_dither: dither,
        };
        if raytrace_layer == Some(EeveeRaytraceLayer::Refraction)
            && blocks_refraction_layer(&triangle.source, &sample)
        {
            let blocker = (hit.depth, triangle_index);
            if refraction_blocker.is_none_or(|(depth, index)| {
                blocker.0 < depth || (equal_f64(blocker.0, depth) && blocker.1 < index)
            }) {
                refraction_blocker = Some(blocker);
            }
            continue;
        }
        if raytrace_layer.is_some_and(|layer| !layer.contains(visible_hit.eevee_raytrace.as_ref()))
        {
            continue;
        }
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
    if let Some((blocker_depth, blocker_index)) = refraction_blocker {
        let mut retained = 0;
        while retained < hit_count
            && hits[retained].is_some_and(|visible_hit| {
                visible_hit.hit.depth < blocker_depth
                    || (equal_f64(visible_hit.hit.depth, blocker_depth)
                        && visible_hit.triangle_index < blocker_index)
            })
        {
            retained += 1;
        }
        hits[retained..hit_count].fill(None);
        hit_count = retained;
    }
    let eevee_raytrace = hits
        .iter()
        .take(hit_count)
        .flatten()
        .find_map(|visible_hit| visible_hit.eevee_raytrace);
    let mut color = if let Some(volume_sample) = volume_sample.as_deref_mut() {
        let mut transmittance = DVec3::ONE;
        let mut alpha_transmittance = 1.0;
        let mut alpha_coverage = 0.0;
        let mut premultiplied = DVec3::ZERO;
        let mut subsurface = None;
        let mut start_distance = volume_sample.near_distance;
        for visible_hit in hits.iter().take(hit_count).flatten() {
            let holdout = visible_hit.holdout_weight.clamp(0.0, 1.0);
            let (radiance, segment_transmittance) =
                volume_sample.integrate_segment(start_distance, visible_hit.hit.distance)?;
            // EEVEE attenuates the volume-composited radiance by (1 - g_holdout) at this surface.
            premultiplied += transmittance * radiance * (1.0 - holdout);
            transmittance *= segment_transmittance;
            let segment_alpha_pass = segment_transmittance.element_sum() / 3.0;
            alpha_coverage += alpha_transmittance * (1.0 - segment_alpha_pass);
            alpha_transmittance *= segment_alpha_pass;
            let triangle = &triangles[visible_hit.triangle_index];
            let color = visible_hit.color;
            let dither = if triangle.source.alpha_mode == AlphaMode::Dithered {
                visible_hit.eevee_dither
            } else {
                None
            };
            let opacity = if dither.is_some() {
                1.0
            } else {
                hit_opacity(&triangle.source, color)
            };
            let holdout_alpha = matte_alpha_weight(holdout, background[3]);
            let radiance_weight = raster_radiance_weight(color[3], dither);
            let surface_weight = transmittance * radiance_weight;
            premultiplied += DVec3::new(color[0], color[1], color[2]) * surface_weight;
            if subsurface.is_none()
                && let Some(sample) = visible_hit.subsurface
            {
                subsurface = Some(EeveeSubsurfacePixel {
                    sample,
                    weight: surface_weight.to_array(),
                });
            }
            let passed = physical_surface_pass(opacity, holdout);
            transmittance *= DVec3::from_array(visible_hit.transparent_tint) * passed;
            let alpha_pass = visible_hit.transparent_tint.iter().sum::<f64>() / 3.0 * passed;
            alpha_coverage +=
                alpha_transmittance * (1.0 - alpha_pass - holdout_alpha).clamp(0.0, 1.0);
            alpha_transmittance *= alpha_pass;
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
            let segment_alpha_pass = segment_transmittance.element_sum() / 3.0;
            alpha_coverage += alpha_transmittance * (1.0 - segment_alpha_pass);
            alpha_transmittance *= segment_alpha_pass;
        }
        let background_alpha = background[3].clamp(0.0, 1.0);
        let background_rgb = if transmittance == DVec3::ZERO {
            [0.0; 3]
        } else {
            let color = sample_background(background, world, Some(world_direction))?;
            [color[0], color[1], color[2]]
        };
        premultiplied += DVec3::from_array(background_rgb) * (transmittance * background_alpha);
        let alpha = alpha_coverage + alpha_transmittance * background_alpha;
        let straight = if alpha <= f64::MIN_POSITIVE {
            premultiplied + DVec3::from_array(background_rgb)
        } else {
            premultiplied / alpha
        };
        CompositeColor {
            straight: [straight.x, straight.y, straight.z, alpha],
            associated: [premultiplied.x, premultiplied.y, premultiplied.z, alpha],
            subsurface,
            eevee_raytrace,
        }
    } else {
        let layers = hits.iter().take(hit_count).flatten().map(|visible_hit| {
            let triangle = &triangles[visible_hit.triangle_index];
            let color = visible_hit.color;
            let dither = if triangle.source.alpha_mode == AlphaMode::Dithered {
                visible_hit.eevee_dither
            } else {
                None
            };
            Layer {
                rgb: [color[0], color[1], color[2]],
                opacity: if dither.is_some() {
                    1.0
                } else {
                    hit_opacity(&triangle.source, color)
                },
                radiance_weight: raster_radiance_weight(color[3], dither),
                tint: visible_hit.transparent_tint,
                holdout_weight: visible_hit.holdout_weight,
                subsurface: visible_hit.subsurface,
                eevee_raytrace: visible_hit.eevee_raytrace,
            }
        });
        composite_layers_with_association(layers, background[3], || {
            let background = sample_background(background, world, Some(world_direction))?;
            Ok([background[0], background[1], background[2]])
        })?
    };

    if let Some(camera_lights) = camera_lights {
        let near_distance = camera.ray_distance_for_depth(ray.direction, camera.near);
        let far_distance = camera.ray_distance_for_depth(ray.direction, camera.far);
        let layers = &hits[..hit_count];
        let mut light_radiance = DVec3::ZERO;
        for event in camera_lights.local_events(ray.origin, ray.direction, far_distance) {
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
        color.straight[0] += light_radiance.x;
        color.straight[1] += light_radiance.y;
        color.straight[2] += light_radiance.z;
        color.associated[0] += light_radiance.x;
        color.associated[1] += light_radiance.y;
        color.associated[2] += light_radiance.z;
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
        let holdout = visible_hit.holdout_weight.clamp(0.0, 1.0);
        let opacity = hit_opacity(&triangle.source, visible_hit.color);
        if triangle.source.alpha_mode == AlphaMode::Dithered && visible_hit.eevee_dither.is_none() {
            if holdout > 0.0 {
                let pass = physical_surface_pass(opacity, holdout);
                let tint = visible_hit.transparent_tint.iter().sum::<f64>() / 3.0;
                surface_transmittance *= DVec3::splat(tint * pass);
            }
            continue;
        }
        match triangle.source.alpha_mode {
            AlphaMode::Blend if holdout > 0.0 => {
                let pass = physical_surface_pass(opacity, holdout);
                let tint = visible_hit.transparent_tint.iter().sum::<f64>() / 3.0;
                surface_transmittance *= DVec3::splat(tint * pass);
            }
            AlphaMode::Blend => continue,
            AlphaMode::Opaque | AlphaMode::Dithered => {
                surface_transmittance = DVec3::ZERO;
                break;
            }
            AlphaMode::Clip if opacity > 0.0 => {
                surface_transmittance = DVec3::ZERO;
                break;
            }
            AlphaMode::Clip => {
                surface_transmittance *= DVec3::splat(1.0 - holdout);
            }
        }
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
/// Both the straight color used by display consumers and its associated Combined render-target value.
#[derive(Clone, Copy)]
struct CompositeColor {
    straight: [f64; 4],
    associated: [f64; 4],
    subsurface: Option<EeveeSubsurfacePixel>,
    eevee_raytrace: Option<EeveeRaytraceSample>,
}

fn associated_rgba(color: [f64; 4]) -> [f64; 4] {
    [
        color[0] * color[3],
        color[1] * color[3],
        color[2] * color[3],
        color[3],
    ]
}

/// One surface of a pixel's stack, for `composite_layers_with_association`.
#[derive(Clone, Copy)]
struct Layer {
    /// Source sample radiance after the selected forward/deferred Holdout association, unassociated by O.
    rgb: [f64; 3],
    /// Physical opacity used by the visibility and film-alpha pass.
    opacity: f64,
    /// `SurfaceSample`'s opaque share O, separate from physical opacity and Holdout pass.
    radiance_weight: f64,
    /// RGB transmittance tint for the remaining physical pass.
    tint: [f64; 3],
    holdout_weight: f64,
    subsurface: Option<EeveeSubsurfaceSample>,
    eevee_raytrace: Option<EeveeRaytraceSample>,
}

/// Composites `layers` (nearest first) over `background`. `rgb` already carries the source pipeline's Holdout/O
/// association; radiance uses O exactly once, while the physical pass and film alpha retain their separate Holdout terms.
fn composite_layers_with_association(
    layers: impl IntoIterator<Item = Layer>,
    background_alpha: f64,
    background_rgb: impl FnOnce() -> Result<[f64; 3]>,
) -> Result<CompositeColor> {
    let mut eevee_raytrace = None;
    let background_alpha = background_alpha.clamp(0.0, 1.0);
    let mut transmittance = DVec3::ONE;
    let mut alpha_transmittance = 1.0;
    let mut alpha_coverage = 0.0;
    let mut premultiplied = DVec3::ZERO;
    let mut subsurface = None;
    for layer in layers {
        if eevee_raytrace.is_none() {
            eevee_raytrace = layer.eevee_raytrace;
        }
        let opacity = layer.opacity.clamp(0.0, 1.0);
        let holdout = layer.holdout_weight.clamp(0.0, 1.0);
        let holdout_alpha = matte_alpha_weight(holdout, background_alpha);
        let radiance_weight = layer.radiance_weight.max(0.0);
        let surface_weight = transmittance * radiance_weight;
        premultiplied += DVec3::from_array(layer.rgb) * surface_weight;
        if subsurface.is_none()
            && let Some(sample) = layer.subsurface
        {
            subsurface = Some(EeveeSubsurfacePixel {
                sample,
                weight: surface_weight.to_array(),
            });
        }
        let passed = physical_surface_pass(opacity, holdout);
        transmittance *= DVec3::from_array(layer.tint) * passed;
        let alpha_pass = layer.tint.iter().sum::<f64>() / 3.0 * passed;
        alpha_coverage += alpha_transmittance * (1.0 - alpha_pass - holdout_alpha).clamp(0.0, 1.0);
        alpha_transmittance *= alpha_pass;
        if transmittance.max_element() <= 1.0e-6 && alpha_transmittance <= 1.0e-6 {
            break;
        }
    }
    // Opaque surfaces block the world, so avoid evaluating its graph for these samples.
    let background_rgb = if transmittance == DVec3::ZERO {
        [0.0; 3]
    } else {
        background_rgb()?
    };
    premultiplied += DVec3::from_array(background_rgb) * (transmittance * background_alpha);
    let alpha = alpha_coverage + alpha_transmittance * background_alpha;
    let straight = if alpha <= f64::MIN_POSITIVE {
        [
            background_rgb[0],
            background_rgb[1],
            background_rgb[2],
            alpha,
        ]
    } else {
        let color = premultiplied / alpha;
        [color.x, color.y, color.z, alpha]
    };
    Ok(CompositeColor {
        straight,
        associated: [premultiplied.x, premultiplied.y, premultiplied.z, alpha],
        subsurface,
        eevee_raytrace,
    })
}

/// EEVEE's holdout changes film alpha only when film transparency is enabled.
fn matte_alpha_weight(holdout: f64, background_alpha: f64) -> f64 {
    if background_alpha < 1.0 {
        holdout.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn resolve_eevee_subsurface_frame(
    samples: &[Option<EeveeSubsurfacePixel>],
    sample_color: &mut [[f32; 4]],
    frame: Option<EeveeSubsurfaceFrame>,
    sample_index: u32,
    pixel_jitter: [f64; 2],
) -> Result<()> {
    if !samples.iter().any(Option::is_some) {
        return Ok(());
    }
    let Some(frame) = frame else {
        return Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            "EEVEE screen-space subsurface requires a perspective or orthographic raster camera",
        ));
    };
    resolve_screen_space_subsurface(samples, sample_color, frame, sample_index, pixel_jitter);
    Ok(())
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
    let barycentric = [barycentric_0, barycentric_1, barycentric_2];
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
    // Barycentric epsilon narrows candidates; the edge determinant, not this tolerance, decides ownership.
    for (opposite_vertex, weight) in barycentric.iter().enumerate() {
        if weight.abs() <= BARYCENTRIC_EPSILON
            && ray_is_on_triangle_edge(triangle, opposite_vertex, ray)
            && !triangle_owns_edge(triangle, opposite_vertex, ray, frame, determinant)
        {
            return None;
        }
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
        barycentric: barycentric.map(|weight| weight.clamp(0.0, 1.0)),
        direction: ray.direction,
    })
}

fn ray_is_on_triangle_edge(triangle: &PreparedTriangle, opposite_vertex: usize, ray: Ray) -> bool {
    let (start, end) = match opposite_vertex {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let start = triangle.source.positions[start] - ray.origin;
    let end = triangle.source.positions[end] - ray.origin;
    let direction = ray.direction;
    // Scale the edge-plane determinant's rounding bound to its own products, not the triangle's barycentric area.
    let determinant = direction.dot(start.cross(end));
    let error_scale = direction.x.abs()
        * (start.y.abs() * end.z.abs() + start.z.abs() * end.y.abs())
        + direction.y.abs() * (start.z.abs() * end.x.abs() + start.x.abs() * end.z.abs())
        + direction.z.abs() * (start.x.abs() * end.y.abs() + start.y.abs() * end.x.abs());
    determinant.is_finite()
        && error_scale.is_finite()
        && determinant.abs() <= 16.0 * f64::EPSILON * error_scale
}

fn triangle_owns_edge(
    triangle: &PreparedTriangle,
    opposite_vertex: usize,
    ray: Ray,
    frame: &CameraFrame,
    determinant: f64,
) -> bool {
    let (start, end) = match opposite_vertex {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let mut edge = triangle.source.positions[end] - triangle.source.positions[start];
    // Normalize the edge direction to screen-space clockwise winding before applying top-left.
    if determinant > 0.0 {
        edge = -edge;
    }
    let (dx, dy) = projected_edge_direction(edge, ray, frame);
    if dy.abs() > f64::MIN_POSITIVE {
        return dy < 0.0;
    }
    if dx.abs() > f64::MIN_POSITIVE {
        return dx < 0.0;
    }

    // A projected point edge has no top-left direction; use a stable ray-space tie-break instead.
    if edge.x.abs() > f64::MIN_POSITIVE {
        return edge.x > 0.0;
    }
    if edge.y.abs() > f64::MIN_POSITIVE {
        return edge.y > 0.0;
    }
    edge.z > 0.0
}

fn projected_edge_direction(edge: DVec3, ray: Ray, frame: &CameraFrame) -> (f64, f64) {
    let direction = ray.direction;
    match frame.projection {
        ProjectionFrame::Orthographic { .. } => (edge.dot(frame.right), -edge.dot(frame.up)),
        ProjectionFrame::Perspective { .. } => {
            let forward = direction.dot(frame.forward);
            let right = direction.dot(frame.right);
            let up = direction.dot(frame.up);
            let edge_forward = edge.dot(frame.forward);
            let projected = (
                edge.dot(frame.right) * forward - right * edge_forward,
                -(edge.dot(frame.up) * forward - up * edge_forward),
            );
            if forward.abs() > f64::MIN_POSITIVE {
                projected
            } else {
                ray_space_edge_direction(edge, ray, frame)
            }
        }
        ProjectionFrame::Equirectangular { .. }
        | ProjectionFrame::FisheyeEquidistant { .. }
        | ProjectionFrame::FisheyeEquisolid { .. } => ray_space_edge_direction(edge, ray, frame),
    }
}

fn ray_space_edge_direction(edge: DVec3, ray: Ray, frame: &CameraFrame) -> (f64, f64) {
    let direction = ray.direction;
    let tangent = edge - direction * direction.dot(edge);
    let mut up = frame.up - direction * direction.dot(frame.up);
    if up.length_squared() <= f64::MIN_POSITIVE {
        up = frame.right - direction * direction.dot(frame.right);
    }
    let up = safe_normalize(up).unwrap_or(frame.up);
    let right = safe_normalize(direction.cross(up)).unwrap_or(frame.right);
    (tangent.dot(right), -tangent.dot(up))
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
        film_sample_offset, pixel_in_bounds, projection_for_camera, rasterize_with_lines,
        ray_for_pixel, store_film_f16_truncated, store_film_history_f16,
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
            1.0,
            None,
            &[],
            None,
            None,
            false,
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
            1.0,
            None,
            &[],
            None,
            None,
            false,
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
    #[test]
    fn film_filter_dimensions_match_source_sampling_uniform_bits() {
        for (sample_index, expected_bits) in [
            (0_u32, [0, 0]),
            (1, [1_061_158_912, 1_051_372_204]),
            (63, [1_057_095_680, 1_031_591_648]),
        ] {
            let (x, y) = film_sample_offset(sample_index);
            assert_eq!([x as f32, y as f32].map(f32::to_bits), expected_bits);
        }
    }

    #[test]
    fn film_half_storage_matches_native_truncation_and_single_rounding() {
        let input = store_film_f16_truncated(0.3334);
        assert_eq!(f64::from(input), 0.333_251_953_125_f64);
        assert_eq!(store_film_history_f16(input), input);
        assert_eq!(
            f64::from(store_film_history_f16(0.3334)),
            0.333_496_093_75_f64
        );
    }
}

#[cfg(test)]
mod eevee_derivative_tests {
    use super::{
        EeveeRaytraceSample, EeveeTraceView, ScreenTraceCamera, eevee_view_position_derivatives,
    };
    use glam::{DVec3, Mat4, Vec2, Vec3, Vec4};

    #[test]
    fn closure_derivatives_are_view_space_deltas_under_rotated_perspective_camera() {
        // A 90-degree perspective projection with near/far planes at 1 and 10.
        let projection = Mat4::from_cols(
            Vec4::X,
            Vec4::Y,
            Vec4::new(0.0, 0.0, -11.0 / 9.0, -1.0),
            Vec4::new(0.0, 0.0, -20.0 / 9.0, 0.0),
        );
        let projection_inverse = projection.inverse();
        let screen_camera = match ScreenTraceCamera::new(projection, projection_inverse) {
            Ok(camera) => camera,
            Err(error) => panic!("perspective projection is a valid screen camera: {error:?}"),
        };
        let identity = EeveeTraceView {
            view: Mat4::IDENTITY,
            view_inverse: Mat4::IDENTITY,
            projection,
            projection_inverse,
            view_projection: projection,
            screen_camera,
            near: 1.0,
            far: 10.0,
        };
        let rotated_view =
            Mat4::from_cols(Vec4::new(0.0, 0.0, -1.0, 0.0), Vec4::Y, Vec4::X, Vec4::W);
        let rotated = EeveeTraceView {
            view: rotated_view,
            view_inverse: rotated_view.inverse(),
            projection,
            projection_inverse,
            view_projection: projection * rotated_view,
            screen_camera,
            near: 1.0,
            far: 10.0,
        };
        let view_point = Vec3::new(0.4, -0.25, -2.5);
        let view_normal = Vec3::new(0.0, 0.6, 0.8);
        let make_surface = |point: Vec3, normal: Vec3| EeveeRaytraceSample {
            point: DVec3::new(f64::from(point.x), f64::from(point.y), f64::from(point.z)),
            normal: DVec3::new(
                f64::from(normal.x),
                f64::from(normal.y),
                f64::from(normal.z),
            ),
            has_gbuffer_route: true,
            screen_refraction: None,
            bins: [None; 3],
        };
        let identity_surface = make_surface(view_point, view_normal);
        let rotated_world_point = rotated.view_inverse.transform_point3(view_point);
        let rotated_world_normal = rotated.view_inverse.transform_vector3(view_normal);
        let rotated_surface = make_surface(rotated_world_point, rotated_world_normal);
        let uv = Vec2::splat(0.5);
        let identity_deltas = eevee_view_position_derivatives(
            &identity_surface,
            &identity,
            view_normal,
            uv,
            0.5,
            16,
            16,
            1,
        );
        let rotated_view_normal = rotated.view.transform_vector3(rotated_world_normal);
        let rotated_deltas = eevee_view_position_derivatives(
            &rotated_surface,
            &rotated,
            rotated_view_normal,
            uv,
            0.5,
            16,
            16,
            1,
        );

        let expected_deltas = [
            Vec3::new(-0.064_062_5, 0.25, -0.1875),
            Vec3::new(-0.4, 0.620_689_63, -0.465_517_28),
        ];
        for (actual, expected) in identity_deltas.iter().zip(expected_deltas) {
            assert!(Vec3::from_array(*actual).abs_diff_eq(expected, 1.0e-6));
        }
        for (rotated, identity) in rotated_deltas.iter().zip(identity_deltas) {
            assert!(Vec3::from_array(*rotated).abs_diff_eq(Vec3::from_array(identity), 1.0e-6));
        }
        assert_ne!(rotated_world_point, view_point);
        assert_ne!(rotated_world_normal, view_normal);
        assert!(rotated_view_normal.abs_diff_eq(view_normal, 1.0e-6));
    }

    #[test]
    fn ray_data_half_store_uses_world_axes_before_view_readback() {
        let projection = Mat4::IDENTITY;
        let screen_camera = match ScreenTraceCamera::new(projection, projection) {
            Ok(camera) => camera,
            Err(error) => panic!("identity projection is a valid screen camera: {error:?}"),
        };
        let view_matrix = Mat4::from_rotation_y(0.37);
        let view = EeveeTraceView {
            view: view_matrix,
            view_inverse: view_matrix.inverse(),
            projection,
            projection_inverse: projection,
            view_projection: projection * view_matrix,
            screen_camera,
            near: 1.0,
            far: 10.0,
        };
        let direction_world = Vec3::new(0.796_875, 0.234_375, -0.4375);
        let store = |value| {
            exr::prelude::f16::from_bits(super::denoise::store_f16_bits_rtz(value)).to_f32()
        };
        let expected_world = Vec3::new(
            store(direction_world.x),
            store(direction_world.y),
            store(direction_world.z),
        );
        let (stored_world, stored_view) =
            super::store_ray_direction_rtz_in_world(&view, direction_world);
        assert_eq!(stored_world, expected_world);
        assert_eq!(stored_view, view.view.transform_vector3(expected_world));

        let direction_view = view.view.transform_vector3(direction_world);
        let incorrectly_stored_view = Vec3::new(
            store(direction_view.x),
            store(direction_view.y),
            store(direction_view.z),
        );
        assert_ne!(
            stored_view,
            view.view.transform_vector3(incorrectly_stored_view)
        );
    }

    #[test]
    fn unit_ior_refraction_pdf_is_independent_of_sample_direction() {
        assert_eq!(
            super::eevee_ggx_refraction_pdf(Vec3::Z, Vec3::X, Vec3::Z, 0.5, 1.0, false),
            1.0e6
        );
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
            1.0,
            None,
            &[],
            None,
            None,
            false,
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
    use crate::graph::{GraphKind, GraphLink, GraphNode, NodeGroup};
    use crate::model::{Id, Material, RayKind, RayVisibility, Registry};
    use crate::shader::PreparedMaterial;

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
        fn shade(&self, _triangle: &Triangle, hit: &SurfaceHit<'_>) -> Result<SurfaceSample> {
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
            layer_tangents: Vec::new(),
            geometry_vertex_slots: [0; 3],
            geometry_vertex_weights: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            smooth_shading: normals.is_some(),
            layer_colors: Vec::new(),
            lit_per_pixel: false,
            ray_mask: RayVisibility::default().mask(),
            normals,
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
    #[test]
    fn basic_fixed_rgba_dithered_surface_uses_its_alpha_without_eevee_metadata() {
        let mut triangle = triangle(None);
        triangle.alpha_mode = AlphaMode::Dithered;
        triangle.color = [0.25, 0.5, 0.75, 0.5];

        let output = render(&[triangle], Mode::Beauty, None);
        assert_eq!(output.linear_rgba[CENTER], [0.125, 0.25, 0.375, 1.0]);
    }

    fn translucent(mut triangle: Triangle) -> Triangle {
        triangle.alpha_mode = AlphaMode::Blend;
        triangle.color = [0.25, 0.5, 0.75, 0.5];
        triangle
    }

    fn render_beauty(triangles: &[Triangle]) -> RasterOutput {
        rasterize_with_samples(
            triangles,
            &camera(),
            SIZE,
            SIZE,
            Mode::Beauty,
            [0, 0, 0, 0],
            None,
            1,
            0,
            1.5,
            None,
            &[],
            None,
            None,
        )
        .expect("Beauty triangles rasterize")
    }

    #[test]
    fn transparent_quad_shared_diagonal_has_one_beauty_owner_for_either_winding() {
        let center = DVec3::new(1.0 / 64.0, -1.0 / 64.0, 0.0);
        let a = center + DVec3::new(-1.0, -1.0, 0.0);
        let b = center + DVec3::new(1.0, -1.0, 0.0);
        let c = center + DVec3::new(1.0, 1.0, 0.0);
        let d = center + DVec3::new(-1.0, 1.0, 0.0);

        for reverse_second_winding in [false, true] {
            let mut first = translucent(triangle(None));
            first.positions = [a, b, c];
            first.double_sided = true;
            let mut second = translucent(triangle(None));
            second.positions = [a, c, d];
            second.double_sided = true;
            if reverse_second_winding {
                second.positions.swap(1, 2);
            }

            let actual = render_beauty(&[first, second]).linear_rgba[CENTER].map(f64::from);
            for (channel, expected) in actual.iter().zip([0.125, 0.25, 0.375, 0.5]) {
                assert!(
                    (*channel - expected).abs() < 1.0e-6,
                    "reverse_second_winding={reverse_second_winding}, got {actual:?}"
                );
            }
        }
    }

    #[test]
    fn opaque_quad_matches_native_top_left_boundary_ownership() {
        let probes = [
            ("bottom", 10, 10, 0.0, 0.25, 1.0),
            ("top", 20, 20, 0.0, -0.25, 0.0),
            ("left", 40, 40, 0.25, 0.0, 1.0),
            ("right", 50, 50, -0.25, 0.0, 0.0),
        ];
        let mut triangles = Vec::with_capacity(probes.len() * 2);
        for (_, x, y, offset_x, offset_y, _) in probes {
            let world_x = -1.0 + (x as f64 + 0.5) * 2.0 / f64::from(SIZE);
            let world_y = 1.0 - (y as f64 + 0.5) * 2.0 / f64::from(SIZE);
            let center = DVec3::new(world_x + offset_x, world_y + offset_y, 0.0);
            let a = center + DVec3::new(-0.25, -0.25, 0.0);
            let b = center + DVec3::new(0.25, -0.25, 0.0);
            let c = center + DVec3::new(0.25, 0.25, 0.0);
            let d = center + DVec3::new(-0.25, 0.25, 0.0);
            let mut first = triangle(None);
            first.positions = [a, b, c];
            let mut second = triangle(None);
            second.positions = [a, c, d];
            triangles.extend([first, second]);
        }

        let output = rasterize_with_samples(
            &triangles,
            &camera(),
            SIZE,
            SIZE,
            Mode::Beauty,
            [0, 0, 0, 0],
            None,
            1,
            0,
            0.0,
            None,
            &[],
            None,
            None,
        )
        .expect("opaque boundary probes rasterize");
        for (edge, x, y, _, _, expected_alpha) in probes {
            let actual = f64::from(output.linear_rgba[y as usize * SIZE as usize + x as usize][3]);
            assert_eq!(actual, expected_alpha, "{edge} edge");
        }
    }

    #[test]
    fn beauty_retains_interior_samples_with_small_barycentric_weights() {
        let center = DVec3::new(1.0 / 64.0, -1.0 / 64.0, 0.0);
        let mut triangle = translucent(triangle(None));
        triangle.positions = [
            center + DVec3::new(-1.0, -0.5, 0.0),
            center + DVec3::new(1.0, -0.5, 0.0),
            center + DVec3::new(0.0, 1.0e12 - 0.5, 0.0),
        ];

        let actual = render_beauty(&[triangle]).linear_rgba[CENTER].map(f64::from);
        for (channel, expected) in actual.iter().zip([0.125, 0.25, 0.375, 0.5]) {
            assert!(
                (*channel - expected).abs() < 1.0e-6,
                "expected a covered interior sample, got {actual:?}"
            );
        }
    }

    #[test]
    fn distinct_transparent_surfaces_still_compose_as_separate_beauty_layers() {
        let mut front = translucent(triangle(None));
        let mut back = translucent(triangle(None));
        front.double_sided = true;
        back.double_sided = true;
        back.object_index = 2;
        back.positions = back.positions.map(|position| position - DVec3::Z);

        let actual = render_beauty(&[front, back]).linear_rgba[CENTER].map(f64::from);
        for (channel, expected) in actual.iter().zip([0.1875, 0.375, 0.5625, 0.75]) {
            assert!(
                (*channel - expected).abs() < 1.0e-6,
                "expected two transparent layers, got {actual:?}"
            );
        }
    }

    struct FrontShader {
        rgba: [f64; 4],
    }

    impl SurfaceShader for FrontShader {
        fn shade(&self, triangle: &Triangle, _hit: &SurfaceHit<'_>) -> Result<SurfaceSample> {
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
            false,
        );
        assert!((visibility - 1.0).abs() < 1e-12, "visibility {visibility}");
    }

    fn horizontal_triangle(z: f64) -> Triangle {
        let mut plane = triangle(None);
        plane.positions = plane.positions.map(|point| point + DVec3::Z * z);
        plane
    }

    fn ao_visibility(
        triangles: &[Triangle],
        materials: &[PreparedMaterial],
        distance: f64,
        has_volume_only: bool,
    ) -> f64 {
        let bvh = crate::render::bvh::Bvh::build(triangles).expect("AO BVH builds");
        super::ambient_visibility(
            &bvh,
            triangles,
            materials,
            DVec3::ZERO,
            DVec3::Z,
            DVec3::Z,
            distance,
            has_volume_only,
        )
    }

    fn volume_only_material() -> PreparedMaterial {
        let mut graph = NodeGroup::new("AO volume", GraphKind::Shader);
        let output = Id::new("output").expect("valid output id");
        let volume = Id::new("volume").expect("valid volume id");
        graph
            .nodes
            .insert(output.clone(), GraphNode::new("ShaderNodeOutputMaterial"));
        graph
            .nodes
            .insert(volume.clone(), GraphNode::new("ShaderNodeVolumePrincipled"));
        graph.links.push(GraphLink {
            from_node: volume,
            from_socket: "Volume".to_owned(),
            to_node: output,
            to_socket: "Volume".to_owned(),
        });
        let graph_id = Id::new("ao_volume").expect("valid graph id");
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        PreparedMaterial::new(
            &Material {
                node_tree: Some(graph_id),
                ..Material::default()
            },
            &groups,
        )
    }

    #[test]
    fn ambient_visibility_uses_any_hit_without_changing_surface_boundaries() {
        assert_eq!(ao_visibility(&[], &[], 1.0, false), 1.0);

        let opaque = horizontal_triangle(0.001);
        assert_eq!(
            ao_visibility(std::slice::from_ref(&opaque), &[], 1.0, false),
            0.0
        );

        let mut filtered = horizontal_triangle(0.001);
        filtered.ray_mask &= !RayKind::Shadow.bit();
        assert_eq!(
            ao_visibility(std::slice::from_ref(&filtered), &[], 1.0, false),
            1.0
        );

        let mut transparent = horizontal_triangle(0.001);
        transparent.alpha_mode = AlphaMode::Blend;
        transparent.color[3] = 0.0;
        assert_eq!(
            ao_visibility(std::slice::from_ref(&transparent), &[], 1.0, false),
            0.0
        );

        let distant = horizontal_triangle(1.0);
        assert_eq!(
            ao_visibility(std::slice::from_ref(&distant), &[], 1.0, false),
            1.0
        );
        assert_eq!(
            ao_visibility(std::slice::from_ref(&distant), &[], 1.1, false),
            11.0 / 12.0
        );
    }

    #[test]
    fn ambient_visibility_passes_through_volume_only_shells() {
        let materials = [volume_only_material(), PreparedMaterial::default()];
        let mut shell = horizontal_triangle(0.001);
        shell.material_index = 0;
        assert_eq!(
            ao_visibility(std::slice::from_ref(&shell), &materials[..1], 1.0, true,),
            1.0
        );

        let mut blocker = horizontal_triangle(0.002);
        blocker.material_index = 1;
        assert_eq!(ao_visibility(&[shell, blocker], &materials, 1.0, true), 0.0);
    }

    #[test]
    fn smooth_shadow_origin_lifts_flat_hits_onto_the_curved_surface() {
        let sphere = Triangle {
            positions: [DVec3::X, DVec3::Y, DVec3::Z],
            normals: Some([DVec3::X, DVec3::Y, DVec3::Z]),
            smooth_shading: true,
            ..triangle(None)
        };
        let centre = DVec3::splat(1.0 / 3.0);
        let lifted = super::smooth_shadow_origin(&sphere, centre, [1.0 / 3.0; 3]);
        assert!(lifted.length() > centre.length(), "lifted {lifted:?}");

        let source_packed_normal =
            DVec3::new(0.0, f64::from(-457_f32 / 511.0), f64::from(228_f32 / 511.0)).normalize();
        let packed_flat = Triangle {
            positions: [DVec3::ZERO, DVec3::X, DVec3::new(0.0, 1.0, 2.0)],
            normals: Some([source_packed_normal; 3]),
            smooth_shading: false,
            ..triangle(None)
        };
        let flat_point = packed_flat.positions.iter().copied().sum::<DVec3>() / 3.0;
        assert_eq!(
            super::smooth_shadow_origin(&packed_flat, flat_point, [1.0 / 3.0; 3]),
            flat_point
        );
    }
}

#[cfg(test)]
mod visibility_tests {
    #![expect(clippy::expect_used, reason = "small raster fixtures")]

    use glam::DVec3;

    use super::{
        AlphaMode, Camera, CameraFrame, EeveeDitherSample, EeveeRaytraceFeedback,
        EeveeRaytraceSample, FilmSampleJitter, Hit, Layer, Mode, PreparedTriangle, Projection,
        RasterHitContext, RasterPixelSample, Ray, StereoMode, SurfaceHit, SurfaceSample,
        SurfaceShader, Triangle, camera_frame, camera_matrices_for_raster,
        composite_layers_with_association, composite_sample_with_association, hit_opacity,
        intersect_triangle, nearest_visible_hit, prepare_triangles, raster_hit_context,
        ray_barycentric_on_triangle, ray_for_pixel_in_frame, ray_for_subpixel_in_frame,
        realtime_hit_differentials,
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
            layer_tangents: Vec::new(),
            normals: None,
            smooth_shading: false,
            geometry_vertex_slots: [0; 3],
            geometry_vertex_weights: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
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

    fn layer_triangle(
        z: f64,
        material_index: usize,
        alpha_mode: AlphaMode,
        alpha: f64,
    ) -> Triangle {
        let mut triangle = triangle([
            DVec3::new(-10.0, -10.0, z),
            DVec3::new(10.0, -10.0, z),
            DVec3::new(0.0, 10.0, z),
        ]);
        triangle.material_index = material_index;
        triangle.alpha_mode = alpha_mode;
        triangle.color = [0.9, 0.1, 0.2, alpha];
        triangle
    }

    struct RefractionLayerShader;

    impl SurfaceShader for RefractionLayerShader {
        fn shade(
            &self,
            triangle: &Triangle,
            _hit: &SurfaceHit<'_>,
        ) -> super::Result<SurfaceSample> {
            let material_index = triangle.material_index;
            let (has_gbuffer_route, screen_refraction) = match material_index {
                0 => (true, Some(true)),
                1 => (false, Some(false)),
                6 => (true, None),
                _ => (true, Some(false)),
            };
            Ok(SurfaceSample {
                color: triangle.color,
                has_surface: material_index != 8,
                transparent_tint: if material_index == 3 {
                    [0.5; 3]
                } else {
                    [1.0; 3]
                },
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: (material_index != 8).then_some(EeveeRaytraceSample {
                    point: DVec3::ZERO,
                    normal: DVec3::Z,
                    has_gbuffer_route,
                    screen_refraction,
                    bins: [None; 3],
                }),
                eevee_dither: match material_index {
                    4 => Some(EeveeDitherSample {
                        transparency: 1.0,
                        alpha_reciprocal: 1.0,
                    }),
                    7 => Some(EeveeDitherSample {
                        transparency: 0.0,
                        alpha_reciprocal: 1.0,
                    }),
                    _ => None,
                },
            })
        }

        fn needs_eevee_gbuffer_context(&self) -> bool {
            true
        }
    }

    fn dither_context(ray: Ray) -> RasterHitContext<'static> {
        RasterHitContext {
            sample_index: 0,
            camera_matrices: None,
            x_neighbor: Some((
                Ray {
                    origin: ray.origin + DVec3::X * 0.01,
                    direction: ray.direction,
                },
                1.0,
            )),
            y_neighbor: Some((
                Ray {
                    origin: ray.origin + DVec3::Y * 0.01,
                    direction: ray.direction,
                },
                1.0,
            )),
            gbuffer_frag_coord: None,
            gbuffer_sample_offset: None,
            gbuffer_closure_random: None,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
            raytrace_layer: None,
        }
    }

    #[test]
    fn refraction_layer_consumers_stop_at_visible_deferred_opaque() {
        let camera = test_camera();
        let frame = camera_frame(&camera, SIZE, SIZE).expect("frame");
        let ray = ray_for_pixel_in_frame(&frame, camera.position, SIZE, SIZE, SIZE / 2, SIZE / 2)
            .expect("pixel ray")
            .expect("center ray");
        let shader = RefractionLayerShader;
        let base = [
            layer_triangle(0.0, 0, AlphaMode::Opaque, 1.0),
            layer_triangle(1.5, 1, AlphaMode::Opaque, 1.0),
        ];
        let prepared = prepare_triangles(&base, Mode::Solid).expect("base layers prepare");
        let visibility =
            Bvh::build_from_positions(prepared.iter().map(|triangle| triangle.source.positions))
                .expect("base visibility builds");
        let mut candidates = Vec::new();
        let target = nearest_visible_hit(
            &prepared,
            &visibility,
            &mut candidates,
            ray,
            &frame,
            Some(super::RasterShader::Main(&shader)),
            None,
            Some(super::EeveeRaytraceLayer::Refraction),
        )
        .expect("forward-only surface passes through");
        let target = target.expect("flagged layer behind forward surface remains visible");
        assert_eq!(target.triangle_index, 0);
        let baseline = composite_sample_with_association(
            &prepared,
            &visibility,
            &mut candidates,
            ray,
            ray.direction,
            &frame,
            &camera,
            BACKGROUND,
            None,
            Some(super::RasterShader::Main(&shader)),
            None,
            Some(&target),
            None,
            None,
            Some(super::EeveeRaytraceLayer::Refraction),
        )
        .expect("refraction composite resolves");
        assert_eq!(baseline.straight, [0.9, 0.1, 0.2, 1.0]);
        assert!(baseline.eevee_raytrace.is_some());

        let blocked = [
            base[0].clone(),
            base[1].clone(),
            layer_triangle(1.0, 2, AlphaMode::Opaque, 1.0),
        ];
        let prepared = prepare_triangles(&blocked, Mode::Solid).expect("blocked layers prepare");
        let visibility =
            Bvh::build_from_positions(prepared.iter().map(|triangle| triangle.source.positions))
                .expect("blocked visibility builds");
        assert!(
            nearest_visible_hit(
                &prepared,
                &visibility,
                &mut candidates,
                ray,
                &frame,
                Some(super::RasterShader::Main(&shader)),
                None,
                Some(super::EeveeRaytraceLayer::Refraction),
            )
            .expect("deferred blocker resolves")
            .is_none()
        );
        let blocked_composite = composite_sample_with_association(
            &prepared,
            &visibility,
            &mut candidates,
            ray,
            ray.direction,
            &frame,
            &camera,
            BACKGROUND,
            None,
            Some(super::RasterShader::Main(&shader)),
            None,
            Some(&target),
            None,
            None,
            Some(super::EeveeRaytraceLayer::Refraction),
        )
        .expect("cached target is occluded during composite");
        assert_eq!(blocked_composite.straight, BACKGROUND);
        assert!(blocked_composite.eevee_raytrace.is_none());

        let context = dither_context(ray);
        for (material, alpha_mode, alpha) in [
            (1, AlphaMode::Opaque, 1.0),
            (3, AlphaMode::Blend, 1.0),
            (4, AlphaMode::Dithered, 1.0),
            (5, AlphaMode::Clip, 0.25),
            (6, AlphaMode::Opaque, 1.0),
            (8, AlphaMode::Opaque, 1.0),
        ] {
            let pass_through = [
                layer_triangle(0.0, 0, AlphaMode::Opaque, 1.0),
                layer_triangle(1.5, material, alpha_mode, alpha),
            ];
            let prepared =
                prepare_triangles(&pass_through, Mode::Solid).expect("pass-through prepares");
            let visibility = Bvh::build_from_positions(
                prepared.iter().map(|triangle| triangle.source.positions),
            )
            .expect("pass-through visibility builds");
            let hit = nearest_visible_hit(
                &prepared,
                &visibility,
                &mut candidates,
                ray,
                &frame,
                Some(super::RasterShader::Main(&shader)),
                Some(&context),
                Some(super::EeveeRaytraceLayer::Refraction),
            )
            .expect("pass-through classification resolves");
            assert_eq!(
                hit.map(|visible| visible.triangle_index),
                Some(0),
                "material {material} must not occlude the flagged layer"
            );
        }

        let dithered_opaque = [
            layer_triangle(0.0, 0, AlphaMode::Opaque, 1.0),
            layer_triangle(1.5, 7, AlphaMode::Dithered, 1.0),
        ];
        let prepared =
            prepare_triangles(&dithered_opaque, Mode::Solid).expect("kept DITHER prepares");
        let visibility =
            Bvh::build_from_positions(prepared.iter().map(|triangle| triangle.source.positions))
                .expect("kept DITHER visibility builds");
        assert!(
            nearest_visible_hit(
                &prepared,
                &visibility,
                &mut candidates,
                ray,
                &frame,
                Some(super::RasterShader::Main(&shader)),
                Some(&context),
                Some(super::EeveeRaytraceLayer::Refraction),
            )
            .expect("kept DITHER blocks")
            .is_none()
        );
    }

    #[test]
    fn realtime_hit_differentials_follow_neighbor_rays_and_eye_matrices() {
        let mut camera = test_camera();
        camera.projection = Projection::Orthographic { height: 2.0 };
        let frame = camera_frame(&camera, 4, 4).expect("frame");
        let matrices = camera_matrices_for_raster(&camera).expect("camera matrices");
        let mut triangle = triangle([
            DVec3::new(-10.0, -10.0, 0.0),
            DVec3::new(10.0, -10.0, 0.0),
            DVec3::new(-10.0, 10.0, 0.0),
        ]);
        triangle.uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let sample_jitter = FilmSampleJitter::new(0, 0.5, 0.5, true);
        let sample = RasterPixelSample {
            cached: None,
            index: 0,
            count: 1,
            needs_eevee_gbuffer_context: false,
            gbuffer_sample_offset: sample_jitter.gbuffer_sample_offset,
            gbuffer_closure_sample: sample_jitter.gbuffer_closure_sample,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
            x: 1,
            y: 1,
            sample_x: 0.5,
            sample_y: 0.5,
        };
        let ray = ray_for_subpixel_in_frame(
            &frame,
            camera.position,
            4,
            4,
            sample.x,
            sample.y,
            sample.sample_x,
            sample.sample_y,
        )
        .expect("pixel ray")
        .expect("orthographic ray");
        let (distance, barycentric) =
            ray_barycentric_on_triangle(ray, &triangle).expect("plane intersection");
        let hit = Hit {
            depth: distance,
            distance,
            barycentric,
            direction: ray.direction,
        };
        let context =
            raster_hit_context(&frame, &camera, 4, 4, &sample, &matrices, 0).expect("hit context");
        let mut gbuffer_sample = sample;
        gbuffer_sample.needs_eevee_gbuffer_context = true;
        let gbuffer_context =
            raster_hit_context(&frame, &camera, 4, 4, &gbuffer_sample, &matrices, 0)
                .expect("EEVEE G-buffer context");
        let frag_coord = gbuffer_context
            .gbuffer_frag_coord
            .expect("EEVEE G-buffer fragment coordinates");
        assert_eq!(
            frag_coord.map(f32::to_bits),
            [1.5_f32, 2.5].map(f32::to_bits)
        );
        let sample_offset = gbuffer_context
            .gbuffer_sample_offset
            .expect("EEVEE G-buffer sample offset");
        assert_eq!(
            sample_offset.map(f32::to_bits),
            [1.0_f32 / 11.0, 1.0_f32 / 13.0, 1.0_f32 / 17.0].map(f32::to_bits)
        );
        assert!(
            gbuffer_context
                .gbuffer_closure_random
                .is_some_and(|value| (0.0..1.0).contains(&value))
        );
        let differentials = realtime_hit_differentials(&triangle, hit, ray, Some(&context));
        assert!(differentials.world_dx.abs_diff_eq(DVec3::X * 0.5, 1.0e-12));
        assert!(
            differentials
                .world_dy
                .abs_diff_eq(DVec3::NEG_Y * 0.5, 1.0e-12)
        );
        assert!((differentials.uv_dx[0] - 0.025).abs() < 1.0e-12);
        assert!(differentials.uv_dx[1].abs() < 1.0e-12);
        assert!(differentials.uv_dy[0].abs() < 1.0e-12);
        assert!((differentials.uv_dy[1] + 0.025).abs() < 1.0e-12);

        camera.stereo_mode = StereoMode::SideBySide;
        let stereo = camera_matrices_for_raster(&camera).expect("stereo matrices");
        assert!((stereo[0].0.w_axis.x + camera.interocular_distance * 0.5).abs() < 1.0e-12);
        assert!((stereo[1].0.w_axis.x - camera.interocular_distance * 0.5).abs() < 1.0e-12);
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
    /// accumulation as `composite_sample_with_association`. Shading is the triangle colour, as with no shader.
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
        composite_layers_with_association(
            layers.iter().map(|(index, _)| {
                let triangle = &triangles[*index];
                let color = triangle.source.color;
                Layer {
                    rgb: [color[0], color[1], color[2]],
                    opacity: hit_opacity(&triangle.source, color),
                    radiance_weight: color[3].clamp(0.0, 1.0),
                    tint: [1.0; 3],
                    holdout_weight: 0.0,
                    subsurface: None,
                    eevee_raytrace: None,
                }
            }),
            background[3],
            || Ok([background[0], background[1], background[2]]),
        )
        .map(|color| color.straight)
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
                let actual = nearest_visible_hit(
                    &prepared,
                    &visibility,
                    &mut candidates,
                    ray,
                    &frame,
                    None,
                    None,
                    None,
                )
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
                let actual = composite_sample_with_association(
                    &prepared,
                    &visibility,
                    &mut candidates,
                    ray,
                    ray.direction,
                    &frame,
                    &camera,
                    BACKGROUND,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .expect("composite resolves");
                assert_eq!(actual.straight, expected, "pixel ({x}, {y})");
            }
        }
    }
}

#[cfg(test)]
mod composite_layer_tests {
    use crate::error::{ErrorCode, PotError};

    use super::super::eevee_advanced_bsdf::{EeveeSubsurfaceClosure, EeveeSubsurfaceSample};
    use super::{Layer, composite_layers_with_association};

    fn composite_layers(
        layers: impl IntoIterator<Item = Layer>,
        background: [f64; 4],
    ) -> super::Result<[f64; 4]> {
        composite_layers_with_association(layers, background[3], || {
            Ok([background[0], background[1], background[2]])
        })
        .map(|color| color.straight)
    }

    fn assert_close(actual: [f64; 4], expected: [f64; 4]) {
        for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "channel {channel}: {actual} != {expected} (full {actual:?} vs {expected:?})"
            );
        }
    }

    #[test]
    fn screen_refraction_setting_is_consumed_only_for_screen_transmission() -> Result<(), String> {
        use crate::model::EeveeRayTracingMethod;

        let trace = |method, transmission, setting| {
            super::eevee_screen_trace_enabled(method, transmission, setting)
                .map_err(|error| format!("{error:?}"))
        };
        assert!(trace(EeveeRayTracingMethod::Screen, false, None)?);
        assert!(trace(EeveeRayTracingMethod::Screen, true, Some(true))?);
        assert!(!trace(EeveeRayTracingMethod::Screen, true, Some(false))?);
        assert!(!trace(EeveeRayTracingMethod::Probe, true, None)?);
        assert!(trace(EeveeRayTracingMethod::Screen, true, None).is_err());
        Ok(())
    }
    #[test]
    fn material_refraction_layer_membership_does_not_depend_on_gbuffer_bins() {
        let sample = super::EeveeRaytraceSample {
            point: glam::DVec3::ZERO,
            normal: glam::DVec3::Z,
            has_gbuffer_route: true,
            screen_refraction: Some(true),
            bins: [None; 3],
        };
        assert!(super::EeveeRaytraceLayer::Refraction.contains(Some(&sample)));
        assert!(!super::EeveeRaytraceLayer::Opaque.contains(Some(&sample)));
        assert!(super::EeveeRaytraceLayer::Opaque.contains(None));
        let forward = super::EeveeRaytraceSample {
            has_gbuffer_route: false,
            ..sample
        };
        assert!(!super::EeveeRaytraceLayer::Refraction.contains(Some(&forward)));
        assert!(!super::EeveeRaytraceLayer::Opaque.contains(Some(&forward)));
    }

    #[test]
    fn eevee_dither_hash_matches_the_pinned_source_kernel() {
        use glam::DVec3;

        use crate::shader::HitDifferentials;

        let differentials = HitDifferentials {
            world_dx: DVec3::new(0.01, 0.002, 0.0),
            world_dy: DVec3::new(0.003, 0.02, 0.0),
            uv_dx: [0.0; 2],
            uv_dy: [0.0; 2],
        };
        let actual = super::eevee_dither_alpha_threshold(
            DVec3::new(0.123, -0.456, 1.7),
            differentials,
            1.0 / 7.0,
        );
        assert!(
            (actual - 0.113_164_78).abs() < 2.0e-5,
            "source hashed threshold: {actual}"
        );
    }

    #[test]
    fn dithered_kept_surface_uses_gbuffer_alpha_unassociation() {
        let dither = super::EeveeDitherSample {
            transparency: 0.5,
            alpha_reciprocal: 2.0,
        };
        assert_eq!(super::raster_radiance_weight(0.5, Some(dither)), 1.0);
        assert_eq!(super::raster_radiance_weight(0.5, None), 0.5);
    }
    #[test]
    fn gbuffer_alpha_unassociation_keeps_source_radiance_weight_in_composite_and_sss()
    -> super::Result<()> {
        let dither = super::EeveeDitherSample {
            transparency: 0.5,
            alpha_reciprocal: 2.0,
        };
        let radiance_weight = super::raster_radiance_weight(1.0, Some(dither));
        assert_eq!(radiance_weight, 2.0);
        let subsurface = EeveeSubsurfaceSample {
            closures: [
                Some(EeveeSubsurfaceClosure {
                    color: [1.0; 3],
                    radius: [1.0; 3],
                }),
                None,
                None,
            ],
            direct: [1.0; 3],
            indirect: [0.0; 3],
            association_scale: 1.0,
            object_index: 0,
            depth: 1.0,
            split_radiance: false,
        };
        let composite = super::composite_layers_with_association(
            [Layer {
                rgb: [0.25, 0.5, 0.75],
                opacity: 1.0,
                radiance_weight,
                tint: [1.0; 3],
                holdout_weight: 0.0,
                subsurface: Some(subsurface),
                eevee_raytrace: None,
            }],
            1.0,
            || Ok([0.0; 3]),
        )?;
        assert_close(composite.straight, [0.5, 1.0, 1.5, 1.0]);
        assert_eq!(
            composite
                .subsurface
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "unassociated SSS sample was lost in the composite",
                    )
                })?
                .weight,
            [2.0; 3],
        );
        Ok(())
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
                radiance_weight: 0.5,
                tint: [1.0, 0.5, 0.0],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
            },
            Layer {
                rgb: [0.0, 1.0, 0.0],
                opacity: 0.0,
                radiance_weight: 0.0,
                tint: [0.2, 0.4, 0.8],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
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
                radiance_weight: 0.0,
                tint: [1.0, 0.0, 0.0],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
            },
            Layer {
                rgb: [0.0; 3],
                opacity: 0.0,
                radiance_weight: 0.0,
                tint: [0.0, 1.0, 0.0],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
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
            radiance_weight: 0.25,
            tint: [1.0; 3],
            holdout_weight: 0.0,
            subsurface: None,
            eevee_raytrace: None,
        }];
        let actual = composite_layers(layers, [0.5, 0.5, 0.5, 1.0])?;
        assert_close(actual, [0.575, 0.425, 0.475, 1.0]);
        Ok(())
    }

    #[test]
    fn holdout_hides_the_backplate_without_showing_geometry_behind_it() -> super::Result<()> {
        let layers = [Layer {
            rgb: [0.8, 0.4, 0.2],
            opacity: 0.0,
            radiance_weight: 0.0,
            tint: [1.0; 3],
            holdout_weight: 1.0,
            subsurface: None,
            eevee_raytrace: None,
        }];
        let actual = composite_layers(layers, [0.2, 0.4, 0.6, 0.0])?;
        assert_close(actual, [0.0, 0.0, 0.0, 0.0]);
        Ok(())
    }
    #[test]
    fn mixed_holdout_shares_block_only_the_holdout_and_opaque_parts() -> super::Result<()> {
        // Scene shading has already applied the forward source's (1 - Holdout) / opaque-share association.
        let diffuse = composite_layers(
            [Layer {
                rgb: [0.4, 0.2, 0.1],
                opacity: 0.5,
                radiance_weight: 0.5,
                tint: [1.0; 3],
                holdout_weight: 0.5,
                subsurface: None,
                eevee_raytrace: None,
            }],
            [0.2, 0.4, 0.6, 0.0],
        )?;
        assert_close(diffuse, [0.4, 0.2, 0.1, 0.5]);

        let transparent = composite_layers(
            [
                Layer {
                    rgb: [0.0; 3],
                    opacity: 0.0,
                    radiance_weight: 0.0,
                    tint: [1.0; 3],
                    holdout_weight: 0.5,
                    subsurface: None,
                    eevee_raytrace: None,
                },
                Layer {
                    rgb: [0.6, 0.2, 0.1],
                    opacity: 1.0,
                    radiance_weight: 1.0,
                    tint: [1.0; 3],
                    holdout_weight: 0.0,
                    subsurface: None,
                    eevee_raytrace: None,
                },
            ],
            [0.0; 4],
        )?;
        assert_close(transparent, [0.6, 0.2, 0.1, 0.5]);
        Ok(())
    }
    #[test]
    fn holdout_does_not_attenuate_resolved_sss_weight() -> super::Result<()> {
        let sample = EeveeSubsurfaceSample {
            closures: [
                Some(EeveeSubsurfaceClosure {
                    color: [1.0; 3],
                    radius: [0.1; 3],
                }),
                None,
                None,
            ],
            direct: [0.0; 3],
            indirect: [0.0; 3],
            association_scale: 2.0,
            object_index: 0,
            depth: 1.0,
            split_radiance: false,
        };
        let composite = composite_layers_with_association(
            [Layer {
                rgb: [0.0; 3],
                opacity: 0.5,
                radiance_weight: 0.5,
                tint: [1.0; 3],
                holdout_weight: 0.5,
                subsurface: Some(sample),
                eevee_raytrace: None,
            }],
            0.0,
            || Ok([0.0; 3]),
        )?;
        let subsurface = composite.subsurface.ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "mixed Holdout and SSS surface lost its resolved sample",
            )
        })?;
        assert_eq!(subsurface.weight, [0.5; 3]);
        assert_eq!(subsurface.sample.association_scale, 2.0);
        Ok(())
    }

    #[test]
    fn foreground_pane_preserves_sss_payload_depth_and_transmitted_weight() -> super::Result<()> {
        let sample = EeveeSubsurfaceSample {
            closures: [
                Some(EeveeSubsurfaceClosure {
                    color: [1.0; 3],
                    radius: [0.1; 3],
                }),
                None,
                None,
            ],
            direct: [0.0; 3],
            indirect: [0.0; 3],
            association_scale: 1.0,
            object_index: 3,
            depth: 5.0,
            split_radiance: false,
        };
        let composite = composite_layers_with_association(
            [
                Layer {
                    rgb: [0.2, 0.4, 0.6],
                    opacity: 0.25,
                    radiance_weight: 0.25,
                    tint: [1.0; 3],
                    holdout_weight: 0.0,
                    subsurface: None,
                    eevee_raytrace: None,
                },
                Layer {
                    rgb: [0.0; 3],
                    opacity: 1.0,
                    radiance_weight: 1.0,
                    tint: [1.0; 3],
                    holdout_weight: 0.0,
                    subsurface: Some(sample),
                    eevee_raytrace: None,
                },
            ],
            1.0,
            || Ok([0.0; 3]),
        )?;
        let subsurface = composite.subsurface.ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "background SSS payload was not preserved through the pane",
            )
        })?;
        assert_eq!(subsurface.sample.depth, 5.0);
        assert_eq!(subsurface.weight, [0.75; 3]);
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
#[cfg(test)]
mod film_band_tests {
    use std::collections::BTreeMap;

    use glam::DVec3;
    use serde_json::json;

    use super::{
        Camera, DepthOfField, Mode, Projection, RasterOutput, RasterSamplesRequest, StereoMode,
        ViewBasis, rasterize_with_samples_for_eye,
    };
    use crate::{
        error::{ErrorCode, PotError, Result},
        eval::{EvaluationContext, Snapshot},
        model::SceneDoc,
        render::path::{RealtimeVolumePass, RealtimeVolumeScratch},
    };
    const WIDTH: u32 = 12;
    const HEIGHT: u32 = 8;
    const SAMPLES: u32 = 4;

    fn render(band_count: usize) -> Result<(RasterOutput, Vec<serde_json::Value>)> {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
        let doc = crate::ops::apply_batch(
            &doc,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {
                        "op":"material.create","id":"warning",
                        "graph":{
                            "kind":"shader",
                            "nodes":{
                                "surface":{"type":"ShaderNodeBsdfPrincipled","inputs":{
                                    "Thin Wall":true,"Subsurface Weight":0.8
                                }},
                                "output":{"type":"OutputMaterial"}
                            },
                            "links":[{"from_node":"surface","from_socket":"BSDF",
                                "to_node":"output","to_socket":"Surface"}]
                        }
                    },
                    {
                        "op":"material.create","id":"fog",
                        "graph":{
                            "kind":"shader",
                            "nodes":{
                                "volume":{"type":"ShaderNodeVolumePrincipled","inputs":{
                                    "Color":[0.0,0.0,0.0,1.0],"Density":0.8
                                }},
                                "output":{"type":"OutputMaterial"}
                            },
                            "links":[{"from_node":"volume","from_socket":"Volume",
                                "to_node":"output","to_socket":"Volume"}]
                        }
                    },
                    {"op":"node.create","id":"warning_plane","kind":"plane",
                        "params":{"size":2.5},"material":"warning"},
                    {"op":"node.create","id":"fog_volume","kind":"sphere",
                        "params":{"radius":0.7,"segments":12,"ring_count":8},
                        "transform":{"translation":[0.0,0.0,1.2]},"material":"fog"},
                    {"op":"light.create","id":"camera_light","light_type":"point","energy":100.0,
                        "radius":0.2,"transform":{"translation":[0.0,0.0,3.5]}},
                    {"op":"node.update","target":{"id":"camera_light"},
                        "set":{"ray_visibility":{"camera":true}}}
                ]
            }),
        )?
        .doc;
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let images = BTreeMap::new();
        let geometry =
            crate::render::scene::extract_geometry(&doc, &snapshot, Mode::Beauty, false, &images)?;
        let camera = Camera {
            position: DVec3::new(0.0, 0.0, 5.0),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 2.5 },
            shift: [0.0; 2],
            depth_of_field: Some(DepthOfField {
                focus_distance: 5.0,
                aperture_radius: 0.05,
                aperture_blades: 6,
            }),
            stereo_mode: StereoMode::SideBySide,
            interocular_distance: 0.065,
        };
        let view = ViewBasis::from_camera(&camera)?;
        let shader = crate::render::scene::surface_shader(
            &doc,
            &snapshot,
            &images,
            &geometry,
            Mode::Beauty,
            &view,
        );
        let transport = crate::render::path::prepare_realtime_volume_transport(
            &doc,
            &snapshot,
            &geometry,
            &images,
            &geometry.shader_warnings,
            None,
        )?
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "test volume was not prepared"))?;
        let camera_lights = crate::render::path::prepare_realtime_camera_light_context(
            &doc,
            &snapshot,
            Some(&transport),
            &camera,
            WIDTH,
            HEIGHT,
        )?;
        let mut scratch = RealtimeVolumeScratch::default();
        let mut volume_pass = Some(RealtimeVolumePass::new(
            &transport,
            &geometry,
            &mut scratch,
            7,
        ));
        let output = rasterize_with_samples_for_eye(RasterSamplesRequest {
            triangles: &geometry.triangles,
            camera: &camera,
            width: WIDTH,
            height: HEIGHT,
            mode: Mode::Beauty,
            background: [0, 0, 0, 0],
            world: None,
            eevee_world_rays: false,
            samples: SAMPLES,
            seed: 7,
            filter_size: 1.5,
            exposure_scale: 1.0,
            shader: shader.as_deref(),
            raytrace_settings: None,
            prepared_materials: &geometry.prepared,
            camera_lights: Some(camera_lights.view(0)),
            view_index: 0,
            band_count,
            volume_pass: volume_pass.as_mut(),
        })?;
        let warnings = geometry.shader_warnings.borrow().clone();
        Ok((output, warnings))
    }

    #[test]
    fn beauty_film_pixels_and_warnings_match_across_band_counts() -> Result<()> {
        let (serial, serial_warnings) = render(1)?;
        let (banded, banded_warnings) = render(3)?;

        assert_eq!(serial.linear_rgba, banded.linear_rgba);
        assert_eq!(serial.rgba, banded.rgba);
        assert_eq!(serial.linear_rgba_associated, banded.linear_rgba_associated);
        assert_eq!(serial.ids, banded.ids);
        assert_eq!(serial.depths, banded.depths);
        assert_eq!(serial.elements, banded.elements);
        assert_eq!(serial.normals, banded.normals);
        assert_eq!(serial.albedo, banded.albedo);
        assert_eq!(serial.emission, banded.emission);
        assert_eq!(serial.ambient_occlusion, banded.ambient_occlusion);
        assert_eq!(serial_warnings, banded_warnings);
        assert!(
            serial_warnings
                .iter()
                .any(|warning| warning["feature_id"] == "shader.principled.thin_wall"),
            "warning-producing material did not emit its realtime warning: {serial_warnings:?}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod eevee_camera_bound_tests {
    use super::eevee_camera_bound_radius;
    use glam::{Mat4, Vec4};

    #[test]
    fn camera_bound_uses_the_frustum_corners() {
        assert!(
            (eevee_camera_bound_radius(Mat4::IDENTITY, 1.0, 3.0) - 3.0_f32.sqrt()).abs() < 1.0e-6
        );
        let perspective = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, -2.0, -1.0),
            Vec4::new(0.0, 0.0, -3.0, 0.0),
        );
        assert!(
            (eevee_camera_bound_radius(perspective, 1.0, 3.0) - 19.0_f32.sqrt()).abs() < 1.0e-5
        );
    }
}
#[cfg(test)]
mod eevee_screen_thickness_tests {
    use super::{
        Camera, Projection, ProjectionFrame, StereoMode, camera_frame, eevee_projection_matrix,
        eevee_screen_thickness, eevee_screen_trace_ray,
    };
    use crate::error::Result;
    use glam::{DVec3, Mat4, Quat, Vec3};

    fn camera(projection: Projection) -> Camera {
        Camera {
            position: DVec3::ZERO,
            target: DVec3::new(0.0, 0.0, -1.0),
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection,
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.0,
        }
    }

    #[test]
    fn perspective_screen_thickness_matches_source_near_and_far_depths() -> Result<()> {
        let width = 64;
        let height = 64;
        let frame = camera_frame(
            &camera(Projection::Perspective {
                lens_mm: 50.0,
                sensor_width_mm: 36.0,
            }),
            width,
            height,
        )?;
        let projection = eevee_projection_matrix(&frame, width, height, 0.5, 0.5)?;
        let minimum_thickness = 0.02_f32;
        let parameters =
            eevee_screen_thickness(&frame, &projection, width, height, minimum_thickness);

        let ProjectionFrame::Perspective {
            horizontal_scale,
            vertical_scale,
        } = frame.projection
        else {
            return Err(crate::error::PotError::new(
                crate::error::ErrorCode::InternalError,
                "perspective test camera did not produce a perspective frame",
            ));
        };
        let pixel_x = 2.0 * horizontal_scale / f64::from(width);
        let pixel_y = 2.0 * vertical_scale / f64::from(height);
        let source_view_scale = ((pixel_x * pixel_x + pixel_y * pixel_y).sqrt() as f32) * 3.0;
        let near = frame.near as f32;
        let far = frame.far as f32;
        let source_ndc_scale = -(near * far) / (far - near);
        assert!((parameters.ndc_scale - source_ndc_scale).abs() < 1.0e-6);

        for view_z in [-near, -far] {
            let source_view_thickness = view_z * source_view_scale - minimum_thickness;
            let source_ndc_thickness =
                source_view_thickness / (view_z * (view_z + source_view_thickness));
            let expected = source_ndc_thickness * source_ndc_scale;

            let view_thickness = view_z * parameters.vs_scale + parameters.vs_bias;
            let actual = view_thickness / (view_z * (view_z + view_thickness))
                * parameters.ndc_scale
                + parameters.ndc_bias;
            assert!((actual - expected).abs() < 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn orthographic_screen_thickness_keeps_source_bias_and_gl_range() -> Result<()> {
        let width = 64;
        let height = 64;
        let frame = camera_frame(
            &camera(Projection::Orthographic { height: 4.0 }),
            width,
            height,
        )?;
        let projection = eevee_projection_matrix(&frame, width, height, 0.5, 0.5)?;
        let minimum_thickness = 0.02_f32;
        let parameters =
            eevee_screen_thickness(&frame, &projection, width, height, minimum_thickness);
        let ProjectionFrame::Orthographic {
            half_width,
            half_height,
        } = frame.projection
        else {
            return Err(crate::error::PotError::new(
                crate::error::ErrorCode::InternalError,
                "orthographic test camera did not produce an orthographic frame",
            ));
        };
        let pixel_x = 2.0 * half_width / f64::from(width);
        let pixel_y = 2.0 * half_height / f64::from(height);
        let average_pixel_radius = ((pixel_x * pixel_x + pixel_y * pixel_y).sqrt() as f32) * 3.0;
        let expected_bias = -(average_pixel_radius + minimum_thickness)
            * (-2.0 / (frame.far as f32 - frame.near as f32))
            * 0.5;
        assert_eq!(parameters.vs_scale, 0.0);
        assert_eq!(parameters.vs_bias, 1.0);
        assert_eq!(parameters.ndc_scale, 0.0);
        assert!((parameters.ndc_bias - expected_bias).abs() < 1.0e-7);
        Ok(())
    }

    #[test]
    fn screen_trace_ray_applies_world_origin_offset_in_view_space() {
        let view =
            Mat4::from_rotation_translation(Quat::from_rotation_y(0.7), Vec3::new(3.0, -2.0, 1.0));
        let world_origin = Vec3::new(-1.0, 0.5, -4.0);
        let direction = Vec3::new(0.2, -0.1, -0.9);
        let origin_offset_world = Vec3::new(-0.4, 0.0, -1.8);
        let ray = eevee_screen_trace_ray(view, world_origin, direction, origin_offset_world, 12.0);
        assert_eq!(
            ray.origin,
            view.transform_point3(world_origin) + view.transform_vector3(origin_offset_world)
        );
        assert_eq!(ray.direction, direction);
        assert_eq!(ray.max_time, 12.0);
    }
}
