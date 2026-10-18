// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   intern/cycles/subd/split.cpp and intern/cycles/subd/dice.cpp, SPDX-License-Identifier: Apache-2.0.

use std::{
    collections::{BTreeMap, HashMap},
    mem::size_of,
};

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    model::CameraProjection,
};

use super::subdivision_limit::{LimitSurface, PatchCache, Sample, Samples};

use super::{MAX_SUBDIVISION_MEMORY_BYTES, MAX_SUBDIVISION_TRIANGLES};

/// Dicing measure used by Cycles adaptive subdivision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdaptiveSpace {
    Pixel,
    Object,
}

/// Camera projection inputs needed to estimate a world-space pixel width.
#[derive(Clone, Debug)]
pub(super) struct AdaptiveCamera {
    pub(super) world_to_camera: DMat4,
    pub(super) projection: CameraProjection,
    pub(super) lens_mm: f64,
    pub(super) sensor_width_mm: f64,
    pub(super) sensor_height_mm: f64,
    pub(super) sensor_fit: String,
    pub(super) panorama_type: String,
    pub(super) clip_start: f64,
    pub(super) ortho_scale: f64,
    pub(super) shift_x: f64,
    pub(super) shift_y: f64,
    pub(super) resolution: [u32; 2],
}

/// Render-time inputs already resolved by the modifier/render caller.
#[derive(Clone, Debug)]
pub(super) struct AdaptiveDicingConfig {
    pub(super) space: AdaptiveSpace,
    pub(super) object_to_world: DMat4,
    /// Per-modifier target size multiplied by the preview/render dicing rate.
    pub(super) target_size: f64,
    pub(super) max_subdivisions: u32,
    pub(super) offscreen_scale: f64,
    pub(super) camera: Option<AdaptiveCamera>,
}

/// Cycles' camera metric is constant for a render; precompute its raster differentials and
/// frustum planes once rather than in every adaptive edge test.
/// Pixel measures follow Cycles `scene/camera.cpp:722-878`; panorama uses the ray-differential
/// transfer from `kernel/camera/camera.h:381-432` and `kernel/util/differential.h:130-154`.
struct CameraDicingFrame {
    world_to_camera: DMat4,
    orthographic_pixel_width: f64,
    orthographic_v1: DVec3,
    orthographic_v2: DVec3,
    perspective_pixel_scale: f64,
    perspective_frustum: [DVec3; 4],
    panorama_pixel_scale: f64,
}

impl CameraDicingFrame {
    fn build(camera: &AdaptiveCamera) -> Result<Self> {
        let camera_to_world = camera.world_to_camera.inverse();
        if !camera_to_world.is_finite() {
            return Err(PotError::invalid_argument(
                "adaptive dicing camera transform must be invertible",
            ));
        }
        let camera_position = camera_to_world.transform_point3(DVec3::ZERO);
        let forward = normalized(camera_to_world.transform_vector3(-DVec3::Z));
        let up_hint = normalized(camera_to_world.transform_vector3(DVec3::Y));
        let right = normalized(forward.cross(up_hint));
        let up = normalized(right.cross(forward));
        // Match the unit camera axes used by preview/raster while retaining the evaluated origin.
        let camera_to_world = DMat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-forward).extend(0.0),
            camera_position.extend(1.0),
        );
        let world_to_camera = camera_to_world.inverse();

        let width = f64::from(camera.resolution[0]);
        let height = f64::from(camera.resolution[1]);
        let ycor = height / width;
        let horizontal_fit = match camera.sensor_fit.as_str() {
            "HORIZONTAL" => true,
            "VERTICAL" => false,
            _ => ycor <= 1.0,
        };
        let view_factor = if horizontal_fit { 1.0 } else { ycor };
        let sensor_size = if horizontal_fit {
            camera.sensor_width_mm
        } else {
            camera.sensor_height_mm
        };
        let mut frame = Self {
            world_to_camera,
            orthographic_pixel_width: 0.0,
            orthographic_v1: DVec3::ZERO,
            orthographic_v2: DVec3::ZERO,
            perspective_pixel_scale: 0.0,
            perspective_frustum: [DVec3::ZERO; 4],
            panorama_pixel_scale: 0.0,
        };

        match camera.projection {
            CameraProjection::Orthographic => {
                let pixel_step = camera.ortho_scale / (width * view_factor);
                let dx = camera_to_world.transform_vector3(DVec3::X * pixel_step);
                let dy = camera_to_world.transform_vector3(DVec3::Y * pixel_step);
                frame.orthographic_pixel_width = dx.length().min(dy.length());
                frame.orthographic_v1 = orthographic_camera_point(camera, width, height);
                frame.orthographic_v2 = orthographic_camera_point(camera, 0.0, 0.0);
            }
            CameraProjection::Perspective => {
                let center = perspective_camera_point(camera, width * 0.5, height * 0.5);
                let center_world = camera_to_world.transform_vector3(center);
                let direction = normalized(center_world);
                let dx_local = DVec3::new(
                    sensor_size / (view_factor * camera.lens_mm * width),
                    0.0,
                    0.0,
                );
                let dy_local = DVec3::new(
                    0.0,
                    -ycor * sensor_size / (view_factor * camera.lens_mm * height),
                    0.0,
                );
                let dx_world = camera_to_world.transform_vector3(dx_local);
                let dy_world = camera_to_world.transform_vector3(dy_local);
                let dx = if dx_world.length_squared() < dy_world.length_squared() {
                    dx_world
                } else {
                    dy_world
                };
                let d_direction = normalized(center_world + dx) - direction;
                let tangent = d_direction - direction * d_direction.dot(direction);
                frame.perspective_pixel_scale = tangent.length();
                let right =
                    positive_camera_point(perspective_camera_point(camera, width, height * 0.5));
                let left =
                    positive_camera_point(perspective_camera_point(camera, 0.0, height * 0.5));
                let top = positive_camera_point(perspective_camera_point(camera, width * 0.5, 0.0));
                let bottom =
                    positive_camera_point(perspective_camera_point(camera, width * 0.5, height));
                frame.perspective_frustum = [
                    normalized(DVec3::new(right.z, 0.0, -right.x)),
                    normalized(DVec3::new(0.0, top.z, -top.y)),
                    normalized(DVec3::new(-left.z, 0.0, left.x)),
                    normalized(DVec3::new(0.0, -bottom.z, bottom.y)),
                ];
            }
            CameraProjection::Panorama | CameraProjection::Fisheye => {
                let center = normalized(camera_to_world.transform_vector3(
                    panorama_camera_direction(camera, width * 0.5, height * 0.5),
                ));
                if center == DVec3::ZERO {
                    frame.panorama_pixel_scale = 0.0;
                } else {
                    let right = normalized(camera_to_world.transform_vector3(
                        panorama_camera_direction(camera, width * 0.5 + 1.0, height * 0.5),
                    ));
                    let down = normalized(camera_to_world.transform_vector3(
                        panorama_camera_direction(camera, width * 0.5, height * 0.5 + 1.0),
                    ));
                    frame.panorama_pixel_scale =
                        0.5 * (right.distance(center) + down.distance(center));
                }
            }
        }

        let positive_metric = match camera.projection {
            CameraProjection::Orthographic => frame.orthographic_pixel_width,
            CameraProjection::Perspective => frame.perspective_pixel_scale,
            CameraProjection::Panorama | CameraProjection::Fisheye => frame.panorama_pixel_scale,
        };
        if !positive_metric.is_finite() || positive_metric < 0.0 {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "adaptive camera differential is not finite",
            ));
        }
        Ok(frame)
    }
}

fn normalized(vector: DVec3) -> DVec3 {
    let length = vector.length();
    if length.is_finite() && length > f64::EPSILON {
        vector / length
    } else {
        DVec3::ZERO
    }
}

fn camera_view_factors(camera: &AdaptiveCamera) -> (f64, f64) {
    let height_over_width = f64::from(camera.resolution[1]) / f64::from(camera.resolution[0]);
    let horizontal_fit = match camera.sensor_fit.as_str() {
        "HORIZONTAL" => true,
        "VERTICAL" => false,
        _ => height_over_width <= 1.0,
    };
    (
        height_over_width,
        if horizontal_fit {
            1.0
        } else {
            height_over_width
        },
    )
}

fn orthographic_camera_point(camera: &AdaptiveCamera, x: f64, y: f64) -> DVec3 {
    let width = f64::from(camera.resolution[0]);
    let height = f64::from(camera.resolution[1]);
    let (ycor, view_factor) = camera_view_factors(camera);
    let ndc_x = 2.0 * x / width - 1.0;
    let ndc_y = 1.0 - 2.0 * y / height;
    DVec3::new(
        (ndc_x + 2.0 * camera.shift_x * view_factor) * camera.ortho_scale / (2.0 * view_factor),
        (ndc_y + 2.0 * camera.shift_y * view_factor / ycor) * ycor * camera.ortho_scale
            / (2.0 * view_factor),
        0.0,
    )
}

fn perspective_camera_point(camera: &AdaptiveCamera, x: f64, y: f64) -> DVec3 {
    let width = f64::from(camera.resolution[0]);
    let height = f64::from(camera.resolution[1]);
    let (ycor, view_factor) = camera_view_factors(camera);
    let horizontal_fit = match camera.sensor_fit.as_str() {
        "HORIZONTAL" => true,
        "VERTICAL" => false,
        _ => ycor <= 1.0,
    };
    let sensor_size = if horizontal_fit {
        camera.sensor_width_mm
    } else {
        camera.sensor_height_mm
    };
    let ndc_x = 2.0 * x / width - 1.0;
    let ndc_y = 1.0 - 2.0 * y / height;
    DVec3::new(
        (ndc_x + 2.0 * camera.shift_x * view_factor) * sensor_size
            / (2.0 * view_factor * camera.lens_mm),
        (ndc_y + 2.0 * camera.shift_y * view_factor / ycor) * ycor * sensor_size
            / (2.0 * view_factor * camera.lens_mm),
        -1.0,
    )
}

fn positive_camera_point(point: DVec3) -> DVec3 {
    DVec3::new(point.x, point.y, -point.z)
}

fn panorama_camera_direction(camera: &AdaptiveCamera, x: f64, y: f64) -> DVec3 {
    let width = f64::from(camera.resolution[0]);
    let height = f64::from(camera.resolution[1]);
    let ndc_x = 2.0 * x / width - 1.0 + 2.0 * camera.shift_x;
    let ndc_y = 1.0 - 2.0 * y / height + 2.0 * camera.shift_y;
    if camera_is_fisheye(camera) {
        let radius = ndc_x.hypot(ndc_y);
        if radius > 1.0 {
            return DVec3::ZERO;
        }
        let angle = radius * std::f64::consts::PI;
        if radius <= f64::EPSILON {
            return DVec3::new(0.0, 0.0, -1.0);
        }
        let sin_angle = angle.sin();
        DVec3::new(
            ndc_x / radius * sin_angle,
            ndc_y / radius * sin_angle,
            -angle.cos(),
        )
    } else {
        let longitude = ndc_x * std::f64::consts::PI;
        let latitude = ndc_y * std::f64::consts::FRAC_PI_2;
        DVec3::new(
            longitude.sin() * latitude.cos(),
            latitude.sin(),
            -longitude.cos() * latitude.cos(),
        )
    }
}

/// Diced faces address the unique output vertex table described by `samples`.
#[derive(Clone, Debug)]
pub(super) struct DicedTopology {
    pub(super) faces: Vec<Vec<usize>>,
    pub(super) samples: Samples,
    pub(super) positions: Vec<DVec3>,
    /// Unit `dPdu × dPdv` per output vertex, aligned with `positions`.
    pub(super) normals: Vec<DVec3>,
}

// Match the Cycles 5.2.2 `split_threshold` used by `DiagSplit::T` (split.cpp:105-114).
const SPLIT_THRESHOLD: f64 = 1.0;
const PARAM_EPSILON: f64 = 1.0e-12;

// Match modifiers.rs:3069-3173, including the dicer's retained samples and temporary topology buffers.
fn diced_vertex_bytes() -> u128 {
    64 + size_of::<Option<Sample>>() as u128 + size_of::<DVec3>() as u128 + 64
}

fn diced_triangle_bytes() -> u128 {
    3 * 128
        + 128
        + size_of::<Vec<usize>>() as u128
        + size_of::<Vec<Sample>>() as u128
        + 3 * size_of::<usize>() as u128
        + 3 * (32 + size_of::<Sample>() as u128)
}

fn check_diced_capacity(vertices: usize, triangles: usize, extra_bytes: u128) -> Result<()> {
    let vertices = u128::try_from(vertices).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive dicing capacity overflows",
        )
    })?;
    let triangles = u128::try_from(triangles).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive dicing capacity overflows",
        )
    })?;
    if triangles > MAX_SUBDIVISION_TRIANGLES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive subdivision exceeds the 100-million triangle budget",
        ));
    }
    let estimated_bytes = vertices
        .checked_mul(diced_vertex_bytes())
        .and_then(|bytes| bytes.checked_add(triangles.checked_mul(diced_triangle_bytes())?))
        .and_then(|bytes| bytes.checked_add(extra_bytes))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive dicing memory estimate overflows",
            )
        })?;
    if estimated_bytes > MAX_SUBDIVISION_MEMORY_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive dicing exceeds the 8-GiB estimated memory budget",
        ));
    }
    Ok(())
}
fn dicing_buffer_bytes(count: usize, element_size: usize) -> Result<u128> {
    u128::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(element_size as u128))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive dicing temporary buffer estimate overflows",
            )
        })
}

fn curve_temporary_bytes(
    patch_bytes: u128,
    curve_count: usize,
    retained_points: usize,
    candidate_points: usize,
    candidate_count: usize,
) -> Result<u128> {
    let point_count = retained_points
        .checked_add(candidate_points)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive curve point count overflows",
            )
        })?;
    let candidates = dicing_buffer_bytes(candidate_count, size_of::<(f64, f64, Vec<f64>)>() * 2)?;
    let curve_map_bytes = dicing_buffer_bytes(curve_count, 128)?;
    let point_bytes = dicing_buffer_bytes(point_count, size_of::<f64>())?
        .checked_mul(2)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive curve vector capacity estimate overflows",
            )
        })?;
    patch_bytes
        .checked_add(curve_map_bytes)
        .and_then(|bytes| bytes.checked_add(point_bytes))
        .and_then(|bytes| bytes.checked_add(candidates))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive curve memory estimate overflows",
            )
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CurveKey {
    BaseEdge(usize, usize),
    NgonSpoke(usize, usize),
}

#[derive(Clone, Copy, Debug)]
struct PatchEdge {
    curve: CurveKey,
    /// Parameter at this edge's first corner in the curve's canonical direction.
    start: f64,
    /// Parameter at this edge's second corner in the curve's canonical direction.
    end: f64,
}

#[derive(Clone, Debug)]
struct Patch {
    face: usize,
    ptex: usize,
    edges: [PatchEdge; 4],
}

#[derive(Clone, Copy, Debug)]
struct EdgeRequest {
    patch: usize,
    edge: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DicedVertexKey {
    BaseVertex(usize),
    NgonCenter(usize),
    Curve(CurveKey, usize),
    Interior(usize, usize, usize, usize),
}

#[derive(Clone, Copy, Debug)]
struct GridVertex {
    index: usize,
    sample: Sample,
}

struct DicedBuilder {
    faces: Vec<Vec<usize>>,
    samples: Samples,
    local_positions: Vec<DVec3>,
    normals: Vec<DVec3>,
    shared_vertices: HashMap<DicedVertexKey, usize>,
    estimated_bytes: u128,
    /// Leaf patches of the single `LimitSurface` this builder dices (one builder per `dice` call).
    patches: PatchCache,
}

impl DicedBuilder {
    fn new(curve_bytes: u128) -> Self {
        Self {
            faces: Vec::new(),
            samples: Samples {
                vertices: Vec::new(),
                corners: Vec::new(),
            },
            local_positions: Vec::new(),
            normals: Vec::new(),
            shared_vertices: HashMap::new(),
            estimated_bytes: curve_bytes,
            patches: PatchCache::default(),
        }
    }

    fn check_capacity(
        &self,
        extra_vertices: usize,
        extra_triangles: usize,
        transient_bytes: u128,
    ) -> Result<()> {
        self.samples
            .vertices
            .len()
            .checked_add(extra_vertices)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "adaptive dicing vertex count overflows",
                )
            })?;
        let triangles = self
            .faces
            .len()
            .checked_add(extra_triangles)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "adaptive dicing triangle count overflows",
                )
            })?;
        if u128::try_from(triangles).unwrap_or(u128::MAX) > MAX_SUBDIVISION_TRIANGLES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive subdivision exceeds the 100-million triangle budget",
            ));
        }
        let added_bytes = u128::try_from(extra_vertices)
            .ok()
            .and_then(|count| count.checked_mul(diced_vertex_bytes()))
            .and_then(|bytes| {
                u128::try_from(extra_triangles)
                    .ok()
                    .and_then(|count| count.checked_mul(diced_triangle_bytes()))
                    .and_then(|triangles| bytes.checked_add(triangles))
            })
            .and_then(|bytes| bytes.checked_add(transient_bytes))
            .and_then(|bytes| self.estimated_bytes.checked_add(bytes))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "adaptive dicing memory estimate overflows",
                )
            })?;
        if added_bytes > MAX_SUBDIVISION_MEMORY_BYTES {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive dicing exceeds the 8-GiB estimated memory budget",
            ));
        }

        Ok(())
    }

    fn vertex(
        &mut self,
        key: DicedVertexKey,
        sample: Sample,
        surface: &LimitSurface,
    ) -> Result<usize> {
        if let Some(&index) = self.shared_vertices.get(&key) {
            return Ok(index);
        }
        self.insert_vertex(key, sample, surface)
    }

    fn insert_vertex(
        &mut self,
        key: DicedVertexKey,
        sample: Sample,
        surface: &LimitSurface,
    ) -> Result<usize> {
        self.check_capacity(1, 0, 0)?;
        let (position, du, dv) = surface.point_and_derivatives_cached(sample, &mut self.patches)?;
        if !position.is_finite() {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "adaptive subdivision produced a non-finite limit sample",
            ));
        }
        let index = self.samples.vertices.len();
        self.samples.vertices.push(Some(sample));
        self.local_positions.push(position);
        // Cycles 5.2.2 EdgeDice::set_vertex stores the patch's normalized dPdu × dPdv.
        self.normals.push(du.cross(dv).normalize_or_zero());
        self.shared_vertices.insert(key, index);
        self.estimated_bytes += diced_vertex_bytes();
        Ok(index)
    }

    fn emit_triangle(&mut self, a: GridVertex, b: GridVertex, c: GridVertex) -> Result<()> {
        if a.index == b.index || b.index == c.index || c.index == a.index {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "adaptive dicing collapsed a patch triangle",
            ));
        }
        self.check_capacity(0, 1, 0)?;
        self.faces.push(vec![a.index, b.index, c.index]);
        self.samples
            .corners
            .push(vec![a.sample, b.sample, c.sample]);
        self.estimated_bytes += diced_triangle_bytes();
        Ok(())
    }

    fn zip_boundaries(&mut self, first: &[GridVertex], second: &[GridVertex]) -> Result<()> {
        if first.is_empty() || second.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "adaptive dicing encountered an empty patch edge",
            ));
        }
        let (mut i, mut j) = (0, 0);
        while i + 1 < first.len() || j + 1 < second.len() {
            let first_index = i;
            let second_index = j;
            let advance_first = if j + 1 == second.len() {
                true
            } else if i + 1 == first.len() {
                false
            } else {
                let diagonal_a = self.local_positions[first[i].index]
                    .distance_squared(self.local_positions[second[j + 1].index]);
                let diagonal_b = self.local_positions[second[j].index]
                    .distance_squared(self.local_positions[first[i + 1].index]);
                diagonal_a >= diagonal_b
            };
            let next = if advance_first {
                i += 1;
                first[i]
            } else {
                j += 1;
                second[j]
            };
            self.emit_triangle(first[first_index], second[second_index], next)?;
        }
        Ok(())
    }
}

/// Builds shared edge factors first, then dices each ptex quad with a stitched interior grid.
///
/// The edge test and cap follow Cycles 5.2.2 `subd/split.cpp:89-147`; the patch grid dimensions and
/// zipper stitching follow `subd/dice.cpp:124-260, 365-421`. Quad faces have one ptex patch; every
/// non-quad base face uses corner-indexed ptex domains from
/// `subdivision_limit.rs:113-120, 302-323`.
pub(super) fn dice(
    base_faces: &[Vec<usize>],
    surface: &LimitSurface,
    config: &AdaptiveDicingConfig,
) -> Result<DicedTopology> {
    let target_size = validate_config(base_faces, config)?;
    if base_faces.is_empty() {
        return Ok(DicedTopology {
            faces: Vec::new(),
            samples: Samples {
                vertices: Vec::new(),
                corners: Vec::new(),
            },
            positions: Vec::new(),
            normals: Vec::new(),
        });
    }
    let camera_frame = config
        .camera
        .as_ref()
        .map(CameraDicingFrame::build)
        .transpose()?;

    let max_segments = 1usize.checked_shl(config.max_subdivisions).ok_or_else(|| {
        PotError::invalid_argument("adaptive subdivision maximum exceeds addressable edge factors")
    })?;
    let patch_count = base_faces
        .iter()
        .try_fold(0usize, |count, face| {
            count.checked_add(if face.len() == 4 { 1 } else { face.len() })
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive ptex patch count overflows",
            )
        })?;
    let edge_request_count = patch_count.checked_mul(4).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive edge request count overflows",
        )
    })?;
    let minimum_triangles = patch_count.checked_mul(2).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive patch triangle estimate overflows",
        )
    })?;
    let patch_storage = dicing_buffer_bytes(patch_count, size_of::<Patch>())?;
    let request_storage = dicing_buffer_bytes(edge_request_count, size_of::<EdgeRequest>() + 128)?;
    let tree_storage = dicing_buffer_bytes(patch_count, 128)?;
    let patch_bytes = patch_storage
        .checked_add(request_storage)
        .and_then(|bytes| bytes.checked_add(tree_storage))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive patch memory estimate overflows",
            )
        })?;
    check_diced_capacity(0, minimum_triangles, patch_bytes)?;
    let mut patches = Vec::with_capacity(patch_count);
    let mut curve_requests: BTreeMap<CurveKey, Vec<EdgeRequest>> = BTreeMap::new();
    for (face_index, face) in base_faces.iter().enumerate() {
        let ptex_count = if face.len() == 4 { 1 } else { face.len() };
        for ptex in 0..ptex_count {
            let patch_index = patches.len();
            let patch = make_patch(face_index, face, ptex)?;
            for (edge_index, edge) in patch.edges.iter().enumerate() {
                curve_requests
                    .entry(edge.curve)
                    .or_default()
                    .push(EdgeRequest {
                        patch: patch_index,
                        edge: edge_index,
                    });
            }
            patches.push(patch);
        }
    }

    let curve_count = curve_requests.len();
    let mut retained_curve_points = 0usize;
    let mut curve_points = BTreeMap::new();
    for (curve, requests) in &curve_requests {
        let mut candidates = Vec::new();
        let mut candidate_point_count = 0usize;
        for request in requests {
            let patch = &patches[request.patch];
            let edge = patch.edges[request.edge];
            let lo = edge.start.min(edge.end);
            let hi = edge.start.max(edge.end);
            let (edge_depth, edge_cap) = if base_faces[patch.face].len() == 4 {
                (config.max_subdivisions, max_segments)
            } else {
                // Cycles split.cpp:130-147 applies the `from_ngon` half-factor cap per ptex edge.
                (
                    config.max_subdivisions.saturating_sub(1),
                    (max_segments / 2).max(1),
                )
            };
            let temporary_base_bytes = curve_temporary_bytes(
                patch_bytes,
                curve_count,
                retained_curve_points,
                candidate_point_count,
                candidates.len(),
            )?;
            let edge_buffer_bytes = temporary_base_bytes
                .checked_add(size_of::<Vec<f64>>() as u128)
                .and_then(|bytes| bytes.checked_add(size_of::<f64>() as u128))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "adaptive edge memory estimate overflows",
                    )
                })?;
            check_diced_capacity(0, minimum_triangles, edge_buffer_bytes)?;
            let mut edge_points = vec![lo];
            append_edge_dicing(
                surface,
                patch,
                request.edge,
                edge,
                config,
                camera_frame.as_ref(),
                target_size,
                edge_depth,
                edge_cap,
                minimum_triangles,
                temporary_base_bytes,
                lo,
                hi,
                &mut edge_points,
            )?;
            candidate_point_count = candidate_point_count
                .checked_add(edge_points.len())
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "adaptive curve point count overflows",
                    )
                })?;
            check_diced_capacity(
                0,
                minimum_triangles,
                curve_temporary_bytes(
                    patch_bytes,
                    curve_count,
                    retained_curve_points,
                    candidate_point_count,
                    candidates.len() + 1,
                )?,
            )?;
            candidates.push((lo, hi, edge_points));
        }

        let selected_copy_budget = candidate_point_count.checked_mul(2).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive curve copy estimate overflows",
            )
        })?;
        check_diced_capacity(
            0,
            minimum_triangles,
            curve_temporary_bytes(
                patch_bytes,
                curve_count,
                retained_curve_points,
                selected_copy_budget,
                candidates.len(),
            )?,
        )?;
        let mut points = if let Some((_, _, full_edge)) = candidates
            .iter()
            .filter(|(lo, hi, _)| *lo <= PARAM_EPSILON && *hi >= 1.0 - PARAM_EPSILON)
            .max_by_key(|(_, _, points)| points.len())
        {
            let mut points = (*full_edge).clone();
            let has_ngon_half = candidates.iter().any(|(lo, hi, _)| {
                (*lo <= PARAM_EPSILON && (*hi - 0.5).abs() <= PARAM_EPSILON)
                    || ((*lo - 0.5).abs() <= PARAM_EPSILON && *hi >= 1.0 - PARAM_EPSILON)
            });
            if matches!(curve, CurveKey::BaseEdge(_, _))
                && has_ngon_half
                && points
                    .iter()
                    .all(|parameter| (*parameter - 0.5).abs() > PARAM_EPSILON)
            {
                let insertion = points.partition_point(|parameter| *parameter < 0.5);
                points.insert(insertion, 0.5);
            }
            points
        } else {
            let intervals: &[(f64, f64)] = if matches!(curve, CurveKey::BaseEdge(_, _)) {
                &[(0.0, 0.5), (0.5, 1.0)]
            } else {
                &[(0.0, 1.0)]
            };
            let mut points = Vec::new();
            for &(lo, hi) in intervals {
                if let Some((_, _, best)) = candidates
                    .iter()
                    .filter(|(candidate_lo, candidate_hi, _)| {
                        (*candidate_lo - lo).abs() <= PARAM_EPSILON
                            && (*candidate_hi - hi).abs() <= PARAM_EPSILON
                    })
                    .max_by_key(|(_, _, points)| points.len())
                {
                    points.extend(best.iter().copied());
                }
            }
            points
        };
        points.sort_by(f64::total_cmp);
        points.dedup_by(|left, right| (*left - *right).abs() <= PARAM_EPSILON);
        retained_curve_points =
            retained_curve_points
                .checked_add(points.len())
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "adaptive retained curve point count overflows",
                    )
                })?;
        check_diced_capacity(
            0,
            minimum_triangles,
            curve_temporary_bytes(
                patch_bytes,
                curve_count,
                retained_curve_points,
                candidate_point_count,
                candidates.len(),
            )?,
        )?;
        curve_points.insert(*curve, points);
    }

    let curve_bytes = curve_temporary_bytes(patch_bytes, curve_count, retained_curve_points, 0, 0)?;
    let mut builder = DicedBuilder::new(curve_bytes);
    for patch in &patches {
        dice_patch(patch, base_faces, &curve_points, surface, &mut builder)?;
    }
    Ok(DicedTopology {
        faces: builder.faces,
        samples: builder.samples,
        positions: builder.local_positions,
        normals: builder.normals,
    })
}

fn validate_config(base_faces: &[Vec<usize>], config: &AdaptiveDicingConfig) -> Result<f64> {
    if !config.object_to_world.is_finite()
        || !config.target_size.is_finite()
        || config.target_size <= 0.0
        || !config.offscreen_scale.is_finite()
        || config.offscreen_scale <= 0.0
    {
        return Err(PotError::invalid_argument(
            "adaptive dicing transforms, target size, and offscreen scale must be finite and positive",
        ));
    }
    for face in base_faces {
        if face.len() < 3 || (0..face.len()).any(|i| face[i] == face[(i + 1) % face.len()]) {
            return Err(PotError::invalid_argument(
                "adaptive dicing requires faces with at least three distinct adjacent vertices",
            ));
        }
    }
    match (config.space, config.camera.as_ref()) {
        (AdaptiveSpace::Object, None) => {}
        (AdaptiveSpace::Object, Some(_)) => {
            return Err(PotError::invalid_argument(
                "object-space adaptive dicing does not take a camera",
            ));
        }
        (AdaptiveSpace::Pixel, None) => {
            return Err(PotError::invalid_argument(
                "pixel-space adaptive dicing requires a camera",
            ));
        }
        (AdaptiveSpace::Pixel, Some(camera)) => {
            validate_camera(camera)?;
        }
    }
    // Cycles clamps adaptive pixel size to 0.1 before applying the global dicing rate
    // (blender/mesh.cpp:916-923); `target_size` has already included that rate.
    Ok(if config.space == AdaptiveSpace::Pixel {
        config.target_size.max(0.1)
    } else {
        config.target_size
    })
}

fn validate_camera(camera: &AdaptiveCamera) -> Result<()> {
    if !camera.world_to_camera.is_finite()
        || !camera.clip_start.is_finite()
        || camera.clip_start <= 0.0
        || camera.resolution[0] == 0
        || camera.resolution[1] == 0
        || !camera.shift_x.is_finite()
        || !camera.shift_y.is_finite()
    {
        return Err(PotError::invalid_argument(
            "adaptive dicing camera transform, shift, and raster dimensions are invalid",
        ));
    }
    match camera.projection {
        CameraProjection::Perspective => {
            if !camera.lens_mm.is_finite()
                || camera.lens_mm <= 0.0
                || !camera.sensor_width_mm.is_finite()
                || camera.sensor_width_mm <= 0.0
                || !camera.sensor_height_mm.is_finite()
                || camera.sensor_height_mm <= 0.0
            {
                return Err(PotError::invalid_argument(
                    "perspective adaptive dicing requires positive camera intrinsics",
                ));
            }
        }
        CameraProjection::Orthographic => {
            if !camera.ortho_scale.is_finite() || camera.ortho_scale <= 0.0 {
                return Err(PotError::invalid_argument(
                    "orthographic adaptive dicing requires a positive camera scale",
                ));
            }
        }
        // preview.rs:462-465 maps every non-fisheye stored panorama type to equirectangular.
        CameraProjection::Panorama => {}
        CameraProjection::Fisheye => {}
    }
    Ok(())
}

fn make_patch(face_index: usize, face: &[usize], ptex: usize) -> Result<Patch> {
    let n = face.len();
    if n == 4 {
        let edges: [PatchEdge; 4] = std::array::from_fn(|edge| {
            let a = face[edge];
            let b = face[(edge + 1) % n];
            let (curve, start, end) = base_edge_curve(a, b);
            PatchEdge { curve, start, end }
        });
        return Ok(Patch {
            face: face_index,
            ptex: 0,
            edges,
        });
    }
    if ptex >= n {
        return Err(PotError::invalid_argument(
            "adaptive dicing ptex index exceeds the base n-gon",
        ));
    }
    let current = face[ptex];
    let next = face[(ptex + 1) % n];
    let previous = face[(ptex + n - 1) % n];
    // Samples::base/next place each n-gon corner at ptex (0,0), its outgoing edge midpoint at (1,0),
    // and the shared face center at (1,1) (subdivision_limit.rs:529-588).
    let (edge_curve, edge_start, _) = base_edge_curve(current, next);
    let (_, _, current_t) = base_edge_curve(previous, current);
    Ok(Patch {
        face: face_index,
        ptex,
        edges: [
            PatchEdge {
                curve: edge_curve,
                start: edge_start,
                end: 0.5,
            },
            PatchEdge {
                curve: CurveKey::NgonSpoke(face_index, ptex),
                start: 0.0,
                end: 1.0,
            },
            PatchEdge {
                curve: CurveKey::NgonSpoke(face_index, (ptex + n - 1) % n),
                start: 1.0,
                end: 0.0,
            },
            PatchEdge {
                curve: CurveKey::BaseEdge(previous.min(current), previous.max(current)),
                start: 0.5,
                end: current_t,
            },
        ],
    })
}

fn base_edge_curve(a: usize, b: usize) -> (CurveKey, f64, f64) {
    let curve = CurveKey::BaseEdge(a.min(b), a.max(b));
    let start = if a < b { 0.0 } else { 1.0 };
    let end = 1.0 - start;
    (curve, start, end)
}

fn sample_on_edge(patch: &Patch, edge_index: usize, parameter: f64) -> Sample {
    let edge = patch.edges[edge_index];
    let local = ((parameter - edge.start) / (edge.end - edge.start)).clamp(0.0, 1.0);
    let (u, v) = match edge_index {
        0 => (local, 0.0),
        1 => (1.0, local),
        2 => (1.0 - local, 1.0),
        _ => (0.0, 1.0 - local),
    };
    Sample {
        face: patch.face,
        ptex: patch.ptex,
        u,
        v,
    }
}

fn append_edge_dicing(
    surface: &LimitSurface,
    patch: &Patch,
    edge_index: usize,
    edge: PatchEdge,
    config: &AdaptiveDicingConfig,
    camera_frame: Option<&CameraDicingFrame>,
    target_size: f64,
    remaining_depth: u32,
    max_segments: usize,
    minimum_triangles: usize,
    temporary_base_bytes: u128,
    start: f64,
    end: f64,
    output: &mut Vec<f64>,
) -> Result<()> {
    let midpoint = (start + end) * 0.5;
    let samples = [
        sample_on_edge(patch, edge_index, start),
        sample_on_edge(patch, edge_index, midpoint),
        sample_on_edge(patch, edge_index, end),
    ];
    let evaluated = surface.evaluate(&samples.map(Some));
    let [Some(first), Some(middle), Some(last)] = evaluated.as_slice() else {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "adaptive edge evaluation did not return all limit samples",
        ));
    };
    let points = [*first, *middle, *last];
    if points.iter().any(|point| !point.is_finite()) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "adaptive edge evaluation produced a non-finite limit sample",
        ));
    }
    let first_length = edge_measure(points[0], points[1], config, camera_frame)?;
    let second_length = edge_measure(points[1], points[2], config, camera_frame)?;
    let l_sum = first_length + second_length;
    let l_max = first_length.max(second_length);
    let t_min = (l_sum / target_size).ceil();
    let t_max = (2.0 * l_max / target_size).ceil();
    if !t_min.is_finite() || !t_max.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "adaptive edge factor is not finite",
        ));
    }

    // Cycles `DiagSplit::T` samples each edge at three points and recursively resolves a
    // non-uniform `tmax - tmin` (split.cpp:89-125). Its max-level edge cap is applied in
    // `limit_edge_factor` (split.cpp:130-147); the remaining-depth budget below keeps each branch
    // within the same power-of-two cap while allowing a nonuniform stitched parameter list.
    if t_max - t_min > SPLIT_THRESHOLD && remaining_depth > 0 {
        append_edge_dicing(
            surface,
            patch,
            edge_index,
            edge,
            config,
            camera_frame,
            target_size,
            remaining_depth - 1,
            max_segments,
            minimum_triangles,
            temporary_base_bytes,
            start,
            midpoint,
            output,
        )?;
        return append_edge_dicing(
            surface,
            patch,
            edge_index,
            edge,
            config,
            camera_frame,
            target_size,
            remaining_depth - 1,
            max_segments,
            minimum_triangles,
            temporary_base_bytes,
            midpoint,
            end,
            output,
        );
    }

    let branch_cap = 1usize
        .checked_shl(remaining_depth)
        .unwrap_or(max_segments)
        .min(max_segments);
    let segments = if t_max >= branch_cap as f64 {
        branch_cap
    } else {
        (t_max as usize).max(1)
    };
    let resulting_points = output.len().checked_add(segments).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive edge point count overflows",
        )
    })?;
    let temporary_bytes = temporary_base_bytes
        .checked_add(dicing_buffer_bytes(resulting_points, 2 * size_of::<f64>())?)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive edge memory estimate overflows",
            )
        })?;
    check_diced_capacity(0, minimum_triangles, temporary_bytes)?;
    for step in 1..=segments {
        output.push(start + (end - start) * (step as f64 / segments as f64));
    }
    Ok(())
}

fn edge_measure(
    first: DVec3,
    second: DVec3,
    config: &AdaptiveDicingConfig,
    camera_frame: Option<&CameraDicingFrame>,
) -> Result<f64> {
    let local_delta = second - first;
    match config.space {
        AdaptiveSpace::Object => Ok(local_delta.length()),
        AdaptiveSpace::Pixel => {
            let camera = config
                .camera
                .as_ref()
                .expect("pixel dicing camera validated");
            let first_world = config.object_to_world.transform_point3(first);
            let second_world = config.object_to_world.transform_point3(second);
            let midpoint = (first_world + second_world) * 0.5;
            let frame = camera_frame.expect("pixel dicing camera frame validated");
            let pixel_width = world_pixel_width(camera, frame, midpoint, config.offscreen_scale)?;
            let length = (second_world - first_world).length() / pixel_width;
            if length.is_finite() {
                Ok(length)
            } else {
                Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "adaptive pixel edge measure is not finite",
                ))
            }
        }
    }
}

fn world_pixel_width(
    camera: &AdaptiveCamera,
    frame: &CameraDicingFrame,
    world: DVec3,
    offscreen_scale: f64,
) -> Result<f64> {
    let local = frame.world_to_camera.transform_point3(world);
    if !local.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "adaptive camera transform produced a non-finite point",
        ));
    }
    let distance = local.length();
    let (mut pixel_width, offscreen_distance, scale_per_distance) = match camera.projection {
        CameraProjection::Orthographic => {
            let point = positive_camera_point(local);
            let v1 = frame.orthographic_v1;
            let v2 = frame.orthographic_v2;
            let closest = DVec3::new(
                point.x.clamp(v1.x.min(v2.x), v1.x.max(v2.x)),
                point.y.clamp(v1.y.min(v2.y), v1.y.max(v2.y)),
                point.z.max(0.0),
            );
            let denominator = (0.5 * (v1.x * v1.x + v1.y * v1.y)).sqrt();
            (
                frame.orthographic_pixel_width,
                (point - closest).length() / denominator,
                frame.orthographic_pixel_width,
            )
        }
        CameraProjection::Perspective => {
            let point = positive_camera_point(local);
            let offscreen = if offscreen_scale > 1.0 {
                perspective_offscreen_distance(point, frame)
            } else {
                0.0
            };
            (
                distance * frame.perspective_pixel_scale,
                offscreen,
                frame.perspective_pixel_scale,
            )
        }
        CameraProjection::Panorama | CameraProjection::Fisheye => (
            (distance + camera.clip_start) * frame.panorama_pixel_scale,
            0.0,
            0.0,
        ),
    };
    if offscreen_scale > 1.0 {
        pixel_width += scale_per_distance * offscreen_distance * (offscreen_scale - 1.0);
    }
    if pixel_width.is_finite() && pixel_width > f64::EPSILON {
        Ok(pixel_width)
    } else {
        Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "adaptive camera differential has no finite positive pixel width",
        ))
    }
}

/// Cycles camera.cpp:779-826 measures perspective distance to the four frustum planes in camera space.
fn perspective_offscreen_distance(point: DVec3, frame: &CameraDicingFrame) -> f64 {
    let [right_normal, top_normal, left_normal, bottom_normal] = frame.perspective_frustum;
    let r = point.dot(right_normal);
    let t = point.dot(top_normal);
    let l = point.dot(left_normal);
    let b = point.dot(bottom_normal);
    if r <= 0.0 && l <= 0.0 && t <= 0.0 && b <= 0.0 {
        return 0.0;
    }
    if r > 0.0 && l > 0.0 && t > 0.0 && b > 0.0 {
        return point.length();
    }

    let along = [
        DVec3::new(-right_normal.z, 0.0, right_normal.x),
        DVec3::new(left_normal.z, 0.0, -left_normal.x),
        DVec3::new(0.0, -top_normal.z, top_normal.y),
        DVec3::new(0.0, bottom_normal.z, -bottom_normal.y),
    ];
    let distances = [r, l, t, b];
    let mut f_dist: f64 = 0.0;
    let mut test_origin = false;
    for index in 0..4 {
        if distances[index] > 0.0 {
            if point.dot(along[index]) >= 0.0 {
                f_dist = f_dist.max(distances[index]);
            } else {
                test_origin = true;
            }
        }
    }
    if test_origin {
        f_dist = if f_dist > 0.0 {
            f_dist.min(point.length())
        } else {
            point.length()
        };
    }
    f_dist
}

fn dice_patch(
    patch: &Patch,
    base_faces: &[Vec<usize>],
    curve_points: &BTreeMap<CurveKey, Vec<f64>>,
    surface: &LimitSurface,
    builder: &mut DicedBuilder,
) -> Result<()> {
    let mut boundary_point_count = 0usize;
    for edge in patch.edges {
        let points = curve_points.get(&edge.curve).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "adaptive dicing is missing a shared patch-edge factor",
            )
        })?;
        let edge_count = points
            .iter()
            .filter(|&&parameter| {
                parameter >= edge.start.min(edge.end) - PARAM_EPSILON
                    && parameter <= edge.start.max(edge.end) + PARAM_EPSILON
            })
            .count();
        boundary_point_count = boundary_point_count
            .checked_add(edge_count)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "adaptive patch edge point count overflows",
                )
            })?;
    }
    let edge_temp_bytes = dicing_buffer_bytes(
        boundary_point_count,
        2 * size_of::<GridVertex>() + size_of::<usize>(),
    )?
    .checked_add(
        (4 * (size_of::<Vec<GridVertex>>() + size_of::<Vec<usize>>())
            + size_of::<Vec<GridVertex>>()) as u128,
    )
    .ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive patch boundary memory estimate overflows",
        )
    })?;
    builder.check_capacity(boundary_point_count, 0, edge_temp_bytes)?;
    let mut edge_vertices: [Vec<GridVertex>; 4] = std::array::from_fn(|_| Vec::new());
    for edge_index in 0..4 {
        let edge = patch.edges[edge_index];
        let points = curve_points.get(&edge.curve).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "adaptive dicing is missing a shared patch-edge factor",
            )
        })?;
        let mut indices: Vec<usize> = points
            .iter()
            .enumerate()
            .filter_map(|(index, &parameter)| {
                let lo = edge.start.min(edge.end) - PARAM_EPSILON;
                let hi = edge.start.max(edge.end) + PARAM_EPSILON;
                (parameter >= lo && parameter <= hi).then_some(index)
            })
            .collect();
        if edge.end < edge.start {
            indices.reverse();
        }
        if indices.len() < 2 {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "adaptive patch edge has fewer than two stitch points",
            ));
        }
        for &curve_index in &indices[..indices.len() - 1] {
            let parameter = points[curve_index];
            let sample = sample_on_edge(patch, edge_index, parameter);
            let key =
                shared_vertex_key(edge.curve, parameter, curve_index, curve_points, base_faces)?;
            let index = builder.vertex(key, sample, surface)?;
            edge_vertices[edge_index].push(GridVertex { index, sample });
        }
        let last_index = *indices.last().expect("at least two edge samples");
        let parameter = points[last_index];
        let sample = sample_on_edge(patch, edge_index, parameter);
        let key = shared_vertex_key(edge.curve, parameter, last_index, curve_points, base_faces)?;
        let index = builder.vertex(key, sample, surface)?;
        edge_vertices[edge_index].push(GridVertex { index, sample });
    }
    let edge_segments: [usize; 4] = std::array::from_fn(|edge| edge_vertices[edge].len() - 1);
    // Cycles dice.cpp:370-380 uses the larger factor on each pair of opposing patch edges.
    let mu = edge_segments[0].max(edge_segments[2]);
    let mv = edge_segments[1].max(edge_segments[3]);
    if mu == 1 && mv == 1 {
        builder.check_capacity(0, 2, edge_temp_bytes)?;
        return dice_single_quad(&edge_vertices, builder);
    }
    if mu == 1 {
        check_strip_capacity(builder, edge_segments, 3, 1, edge_temp_bytes)?;
        let left: Vec<GridVertex> = edge_vertices[3].iter().rev().copied().collect();
        return builder.zip_boundaries(&left, &edge_vertices[1]);
    }
    if mv == 1 {
        check_strip_capacity(builder, edge_segments, 2, 0, edge_temp_bytes)?;
        let top: Vec<GridVertex> = edge_vertices[2].iter().rev().copied().collect();
        return builder.zip_boundaries(&top, &edge_vertices[0]);
    }

    // Cycles dice.cpp:370-396 always stitches the interior grid at the shared edge factors; its
    // area-scaled grid is compiled out there (`#if 0`).
    let grid_inner_vertices = (mu - 1).checked_mul(mv - 1).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive interior vertex estimate overflows",
        )
    })?;
    let interior_cells = (mu - 2).checked_mul(mv - 2).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive interior cell estimate overflows",
        )
    })?;
    let interior_triangles = interior_cells.checked_mul(2).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive interior triangle estimate overflows",
        )
    })?;
    let inner_stitches = (mu - 2)
        .checked_mul(2)
        .and_then(|count| count.checked_add((mv - 2).checked_mul(2)?))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive stitch triangle estimate overflows",
            )
        })?;
    let outer_stitches = edge_segments
        .iter()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive boundary triangle estimate overflows",
            )
        })?;
    let patch_triangles = interior_triangles
        .checked_add(inner_stitches)
        .and_then(|count| count.checked_add(outer_stitches))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive patch triangle estimate overflows",
            )
        })?;
    let grid_rows = mv.checked_add(1).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive grid row count overflows",
        )
    })?;
    let grid_columns = mu.checked_add(1).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive grid column count overflows",
        )
    })?;
    let grid_cells = grid_rows.checked_mul(grid_columns).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "adaptive grid cell count overflows",
        )
    })?;
    let mut grid_temporary_bytes = dicing_buffer_bytes(grid_cells, size_of::<GridVertex>())?
        .checked_add(dicing_buffer_bytes(
            grid_rows,
            size_of::<Vec<GridVertex>>(),
        )?)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive grid memory estimate overflows",
            )
        })?;
    let inner_edge_points = (mu - 1).max(mv - 1);
    grid_temporary_bytes = grid_temporary_bytes
        .checked_add(dicing_buffer_bytes(
            inner_edge_points,
            2 * size_of::<GridVertex>(),
        )?)
        .and_then(|bytes| bytes.checked_add(size_of::<Vec<GridVertex>>() as u128))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive stitch memory estimate overflows",
            )
        })?;
    let patch_temporary_bytes = edge_temp_bytes
        .checked_add(grid_temporary_bytes)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive patch temporary memory estimate overflows",
            )
        })?;
    builder.check_capacity(grid_inner_vertices, patch_triangles, patch_temporary_bytes)?;
    let placeholder = GridVertex {
        index: 0,
        sample: Sample {
            face: patch.face,
            ptex: patch.ptex,
            u: 0.0,
            v: 0.0,
        },
    };
    let mut grid = vec![vec![placeholder; mu + 1]; mv + 1];
    for j in 1..mv {
        for i in 1..mu {
            let sample = Sample {
                face: patch.face,
                ptex: patch.ptex,
                u: i as f64 / mu as f64,
                v: j as f64 / mv as f64,
            };
            let key = DicedVertexKey::Interior(patch.face, patch.ptex, i, j);
            let index = builder.insert_vertex(key, sample, surface)?;
            grid[j][i] = GridVertex { index, sample };
        }
    }
    for j in 1..(mv - 1) {
        for i in 1..(mu - 1) {
            let a = grid[j][i];
            let b = grid[j][i + 1];
            let c = grid[j + 1][i + 1];
            let d = grid[j + 1][i];
            builder.emit_triangle(a, b, c)?;
            builder.emit_triangle(a, c, d)?;
        }
    }

    for edge in 0..4 {
        let mut inner = Vec::new();
        match edge {
            0 => inner.extend((1..mu).map(|i| grid[1][i])),
            1 => inner.extend((1..mv).map(|j| grid[j][mu - 1])),
            2 => inner.extend((1..mu).rev().map(|i| grid[mv - 1][i])),
            _ => inner.extend((1..mv).rev().map(|j| grid[j][1])),
        }
        builder.zip_boundaries(&inner, &edge_vertices[edge])?;
    }
    Ok(())
}

fn check_strip_capacity(
    builder: &DicedBuilder,
    edge_segments: [usize; 4],
    first_edge: usize,
    second_edge: usize,
    edge_temp_bytes: u128,
) -> Result<()> {
    let triangles = edge_segments[first_edge]
        .checked_add(edge_segments[second_edge])
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "adaptive strip triangle count overflows",
            )
        })?;
    builder.check_capacity(0, triangles, edge_temp_bytes)
}

// Match render/preview.rs:462-465: stored panorama type selects equirectangular or equidistant;
// the separate Fisheye projection maps to equidistant.
fn camera_is_fisheye(camera: &AdaptiveCamera) -> bool {
    camera.projection == CameraProjection::Fisheye
        || (camera.projection == CameraProjection::Panorama
            && camera.panorama_type == "fisheye_equidistant")
}

fn shared_vertex_key(
    curve: CurveKey,
    parameter: f64,
    index: usize,
    curve_points: &BTreeMap<CurveKey, Vec<f64>>,
    base_faces: &[Vec<usize>],
) -> Result<DicedVertexKey> {
    match curve {
        CurveKey::BaseEdge(a, _) if parameter <= PARAM_EPSILON => Ok(DicedVertexKey::BaseVertex(a)),
        CurveKey::BaseEdge(_, b) if parameter >= 1.0 - PARAM_EPSILON => {
            Ok(DicedVertexKey::BaseVertex(b))
        }
        CurveKey::BaseEdge(_, _) => Ok(DicedVertexKey::Curve(curve, index)),
        CurveKey::NgonSpoke(face, _) if parameter >= 1.0 - PARAM_EPSILON => {
            Ok(DicedVertexKey::NgonCenter(face))
        }
        CurveKey::NgonSpoke(face, edge) if parameter <= PARAM_EPSILON => {
            let corners = base_faces.get(face).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "n-gon spoke references a missing face",
                )
            })?;
            let a = corners[edge];
            let b = corners[(edge + 1) % corners.len()];
            let base_curve = CurveKey::BaseEdge(a.min(b), a.max(b));
            let base_samples = curve_points.get(&base_curve).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "n-gon spoke endpoint has no base-edge sample list",
                )
            })?;
            let midpoint = base_samples
                .iter()
                .position(|sample| (*sample - 0.5).abs() <= PARAM_EPSILON)
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "n-gon base edge is missing its shared midpoint sample",
                    )
                })?;
            Ok(DicedVertexKey::Curve(base_curve, midpoint))
        }
        CurveKey::NgonSpoke(_, _) => Ok(DicedVertexKey::Curve(curve, index)),
    }
}

fn dice_single_quad(
    edge_vertices: &[Vec<GridVertex>; 4],
    builder: &mut DicedBuilder,
) -> Result<()> {
    let c0 = edge_vertices[0][0];
    let c1 = *edge_vertices[0].last().expect("patch edge has a corner");
    let c2 = *edge_vertices[1].last().expect("patch edge has a corner");
    let c3 = *edge_vertices[2].last().expect("patch edge has a corner");
    let diagonal_02 =
        builder.local_positions[c0.index].distance_squared(builder.local_positions[c2.index]);
    let diagonal_13 =
        builder.local_positions[c1.index].distance_squared(builder.local_positions[c3.index]);
    if diagonal_02 <= diagonal_13 {
        builder.emit_triangle(c0, c1, c2)?;
        builder.emit_triangle(c0, c2, c3)
    } else {
        builder.emit_triangle(c0, c1, c3)?;
        builder.emit_triangle(c1, c2, c3)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "small deterministic adaptive camera checks"
)]
mod tests {
    use super::{AdaptiveCamera, CameraDicingFrame, world_pixel_width};
    use crate::model::CameraProjection;
    use glam::{DMat4, DVec3};

    fn camera(projection: CameraProjection, camera_to_world: DMat4) -> AdaptiveCamera {
        AdaptiveCamera {
            world_to_camera: camera_to_world.inverse(),
            projection,
            lens_mm: 50.0,
            sensor_width_mm: 36.0,
            sensor_height_mm: 24.0,
            sensor_fit: "AUTO".to_owned(),
            panorama_type: "equirectangular".to_owned(),
            clip_start: 0.1,
            ortho_scale: 8.0,
            shift_x: 0.0,
            shift_y: 0.0,
            resolution: [100, 80],
        }
    }

    #[test]
    fn camera_object_scale_does_not_change_pixel_metrics_or_origin() {
        let position = DVec3::new(4.0, -2.0, 3.0);
        let unit = DMat4::from_cols(
            DVec3::X.extend(0.0),
            DVec3::Y.extend(0.0),
            DVec3::Z.extend(0.0),
            position.extend(1.0),
        );
        let scaled = DMat4::from_cols(
            (DVec3::X * 2.0).extend(0.0),
            (DVec3::Y * 3.0).extend(0.0),
            (DVec3::Z * 4.0).extend(0.0),
            position.extend(1.0),
        );

        for projection in [
            CameraProjection::Orthographic,
            CameraProjection::Perspective,
            CameraProjection::Panorama,
            CameraProjection::Fisheye,
        ] {
            let unit_camera = camera(projection, unit);
            let scaled_camera = camera(projection, scaled);
            let unit_frame = CameraDicingFrame::build(&unit_camera).unwrap();
            let scaled_frame = CameraDicingFrame::build(&scaled_camera).unwrap();
            assert_eq!(
                unit_frame.orthographic_pixel_width,
                scaled_frame.orthographic_pixel_width
            );
            assert_eq!(
                unit_frame.perspective_pixel_scale,
                scaled_frame.perspective_pixel_scale
            );
            assert_eq!(
                unit_frame.perspective_frustum,
                scaled_frame.perspective_frustum
            );
            assert_eq!(
                unit_frame.panorama_pixel_scale,
                scaled_frame.panorama_pixel_scale
            );

            let world = position + DVec3::new(0.25, 0.5, -4.0);
            let unit_local = unit_frame.world_to_camera.transform_point3(world);
            let scaled_local = scaled_frame.world_to_camera.transform_point3(world);
            assert!((unit_local - DVec3::new(0.25, 0.5, -4.0)).length() < 1.0e-12);
            assert!((scaled_local - unit_local).length() < 1.0e-12);
            let unit_width = world_pixel_width(&unit_camera, &unit_frame, world, 1.0).unwrap();
            let scaled_width =
                world_pixel_width(&scaled_camera, &scaled_frame, world, 1.0).unwrap();
            assert!((scaled_width - unit_width).abs() < 1.0e-12);
        }
    }
}
