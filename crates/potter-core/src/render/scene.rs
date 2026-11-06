use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::{Arc, OnceLock},
};

use glam::{DMat3, DMat4, DVec3, Mat3, Vec3};
use serde::Serialize;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::{
        GeneratedCorners, GeneratedVertex, Mesh, TextureSpace, attribute_keys,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        normals::{render_corner_normals, smooth_face_flags},
        uv_layers::{self, UvStore},
    },
    image::ImageData,
    model::{
        CyclesCurveShape, CyclesHairShadowGeometryKey, CyclesHairShadowGroup,
        CyclesHairShadowMember, Id, LightData, LightType, Material, Node, RayKind, RayVisibility,
        SceneDoc, World,
    },
    render::eevee_advanced_bsdf::{
        MAX_EEVEE_SUBSURFACE_CLOSURES, eevee_brdf_f82_tint_lut, eevee_gbuffer_closure_color,
    },
    shader::{
        BsdfModel, BsdfParams, EEVEE_CLOSURE_DIFFUSE, EEVEE_CLOSURE_EMISSION,
        EEVEE_CLOSURE_REFLECTION, EEVEE_CLOSURE_REFRACTION, EEVEE_CLOSURE_SHADER_TO_RGBA,
        EEVEE_CLOSURE_SUBSURFACE, EEVEE_CLOSURE_TRANSLUCENT, EEVEE_CLOSURE_TRANSPARENCY,
        EeveeGBufferBin, EeveeGBufferCandidate, EeveeGBufferLayout, EeveeGBufferLobe,
        EeveeGBufferMode, EvaluatorWorkspace, GeometryAttribute, GeometryAttributeValues,
        GlassDistribution, HitContext, HitFootprint, HitUvLayers, PreparedMaterial, ShaderClosure,
        SurfaceShading, dielectric_f0, evaluate_displacement_with_workspace_semantics,
        evaluate_surface_with_workspace, evaluate_surface_with_workspace_semantics,
        fill_color_layers, heaviest, make_orthonormals,
    },
};

use super::eevee_advanced_bsdf::{
    EeveeGBufferContext, EeveeSubsurfaceClosure, EeveeSubsurfaceSample, eevee_gbuffer_ior,
    eevee_gbuffer_normal, eevee_gbuffer_roughness, eevee_gbuffer_thickness, eevee_holdout_weight,
    eevee_metallic_brdf, eevee_subsurface_gbuffer_profile, eevee_subsurface_radius,
};
use super::eevee_fast_gi::FastGiSh;
use super::eevee_raytrace::EeveeRayTracingSettings;
use super::eevee_raytrace_generate::{RayGenerateInput, RayLobe, RayThickness, ThicknessMode};
use super::hair_intersect::{HairBvh, HairCurveRenderMode};
use super::world_probe::{EeveeWorldSun, WorldProbe, WorldProbeMode};
use super::{
    bvh::{Bvh, BvhHit, WalkRay},
    eevee_ltc::Emitter,
    eevee_transmission::{self, GlassClosures, GlassLight, Thickness, irradiance_probe_radiance},
    light_model::{
        AreaShape, area_irradiance_per_radiance, area_radiance, area_shape,
        area_visibility_samples, emitter_radiance, local_light_influence, point_intensity,
        spot_cone, sun_disc_radius,
    },
    path::{TraceGeometry, hit_params, object_for_index},
    principled_bsdf::{
        Layers, layers as principled_layers, reflected_lobes_coefficient,
        thin_subsurface_transmission_weight, without_subsurface,
    },
    raster::{
        AlphaMode, EeveeDitherSample, EeveeRaytraceBin, EeveeRaytraceFeedback,
        EeveeRaytraceLayerFlags, EeveeRaytraceLayerSummary, EeveeRaytraceSample, Line, Mode,
        SurfaceHit, SurfaceSample, SurfaceShader, SurfaceShaderWorker, Triangle, ViewBasis,
        ambient_visibility, blend_color, blend_corner_normals, blend_uv,
    },
    studio,
    wireframe::{PointSphere, WireEdge, WireEdgeBuilder, WirePoint},
};

/// Most color layers one object carries, and one hit can read (Blender's per-mesh limit).
pub(super) const MAX_COLOR_LAYERS: usize = 8;

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub(super) struct ObjectRecord {
    pub index: u32,
    pub node_id: Id,
    pub data_id: Option<Id>,
    pub instance_path: Vec<Id>,
    /// UV layer names in layer order, matching `Triangle::layer_uvs`. Not part of the output.
    #[serde(skip)]
    pub uv_layers: Vec<String>,
    /// Index of the render layer in `uv_layers`; `None` without layers. Not part of the output.
    #[serde(skip)]
    pub uv_render: Option<usize>,
    /// Color layer names in layer order, matching `Triangle::layer_colors`. Not part of the output.
    #[serde(skip)]
    pub color_layers: Vec<String>,
    /// Slot-specific render-color position and member color availability, per `Triangle::material_index`.
    #[serde(skip)]
    pub material_colors: MaterialColors,
    /// Joined color registry indices, matching `color_layers`.
    #[serde(skip)]
    pub color_layer_indices: Vec<usize>,
    /// Typed generic mesh attributes, in the source mesh's point order.
    #[serde(skip)]
    pub geometry_attributes: Vec<GeometryAttribute>,
    /// EEVEE's default refraction thickness of the object (`transmission_thickness`). Not part of the output.
    #[serde(skip)]
    pub thickness: f64,
    /// Evaluated world transform for shader nodes. Not part of the output.
    #[serde(skip)]
    pub object_to_world: DMat4,
    /// Inverse of the evaluated world matrix; `None` when it is not finite, and the point then stays unchanged.
    #[serde(skip)]
    pub world_to_local: Option<DMat4>,
    #[serde(skip)]
    pub object_color: [f64; 4],
    #[serde(skip)]
    pub object_pass_index: f64,
    #[serde(skip)]
    pub object_random_number: Option<f64>,
    #[serde(skip)]
    pub is_holdout: bool,
    #[serde(skip)]
    pub shadow_terminator_normal_offset: f64,
    #[serde(skip)]
    pub shadow_terminator_geometry_offset: f64,
    #[serde(skip)]
    pub shadow_terminator_shading_offset: f64,
    #[serde(skip)]
    pub view_layer_use_solid: bool,
    #[serde(skip)]
    pub view_layer_use_volumes: bool,
}

/// Color state per material of one object, keyed by `Triangle::material_index` and sorted by it, so a hit
/// finds its entry by binary search. A material the object does not draw has no entry and reads as `None`.
#[derive(Clone, Debug, Default)]
pub(super) struct MaterialColors {
    entries: Vec<MaterialColor>,
}

#[derive(Clone, Debug)]
struct MaterialColor {
    material_index: usize,
    /// Identifies the cached prepared material for deferred-layer closure-flag reuse.
    material_id: Option<Id>,
    material_pass_index: Option<f64>,
    render_layer: Option<usize>,
    available_mask: Option<Arc<[u64]>>,
}

impl MaterialColors {
    /// Records the entry of `material_index`; a later face of the same material replaces it.
    fn insert(
        &mut self,
        material_index: usize,
        material_id: Option<&Id>,
        material_pass_index: Option<f64>,
        render_layer: Option<usize>,
        available_mask: Option<Arc<[u64]>>,
    ) {
        match self
            .entries
            .binary_search_by_key(&material_index, |stored| stored.material_index)
        {
            Ok(position) => {
                let entry = &mut self.entries[position];
                if entry.material_id.is_none() {
                    entry.material_id = material_id.cloned();
                }
                entry.material_pass_index = material_pass_index;
                entry.render_layer = render_layer;
                entry.available_mask = available_mask;
            }
            Err(position) => self.entries.insert(
                position,
                MaterialColor {
                    material_index,
                    material_id: material_id.cloned(),
                    material_pass_index,
                    render_layer,
                    available_mask,
                },
            ),
        }
    }

    /// The color state of `material_index`, or `None` when the object does not draw that material.
    fn get(&self, material_index: usize) -> Option<&MaterialColor> {
        self.entries
            .binary_search_by_key(&material_index, |stored| stored.material_index)
            .ok()
            .map(|position| &self.entries[position])
    }
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub(super) struct ElementRecord {
    pub index: u32,
    pub node_id: Id,
    pub domain: String,
    pub element_id: String,
    /// Per-face UV binding metadata read by triangles through their retained `element_index`.
    #[serde(skip)]
    pub uv_render: Option<usize>,
    /// Source-member layers available to this face; `None` for ordinary meshes.
    #[serde(skip)]
    pub uv_available_mask: Option<u8>,
}

pub(super) struct HairCurveInstance {
    pub curve: super::hair_intersect::HairCurveGeometry,
    pub point_normals: super::hair_geometry::HairCurvePointNormals,
    pub shadow_transparency: Arc<super::hair_geometry::HairCurveShadowTransparency>,
    pub material_index: usize,
    pub object_index: u32,
    pub element_index: u32,
    pub curve_random: Option<f64>,
    pub ray_mask: u8,
}

pub(super) struct Geometry {
    pub hair_curve_instances: Vec<HairCurveInstance>,
    pub hair_curve_bvh: HairBvh,
    pub triangles: Vec<Triangle>,
    pub lines: Vec<Line>,
    pub wire_edges: Vec<WireEdge>,
    /// Loose vertices of a mesh with no edges or faces, and point-cloud points, for the wireframe preview.
    pub wire_points: Vec<WirePoint>,
    pub materials: Vec<Material>,
    pub prepared: Vec<PreparedMaterial>,
    pub objects: Vec<ObjectRecord>,
    pub elements: Vec<ElementRecord>,
    pub bounds: Option<(DVec3, DVec3)>,
    pub warnings: Vec<serde_json::Value>,
    /// Warnings recorded while shading hits, such as `uv.layer_missing`.
    pub shader_warnings: RefCell<Vec<serde_json::Value>>,
    pub shadows: Option<RealtimeShadows>,
}

impl Geometry {
    /// Extraction warnings, then each distinct warning recorded while shading hits.
    pub(super) fn all_warnings(&self) -> Vec<serde_json::Value> {
        let mut all = self.warnings.clone();
        for warning in self.shader_warnings.borrow().iter() {
            if !all.contains(warning) {
                all.push(warning.clone());
            }
        }
        all
    }
    /// The shared realtime BVH contains every geometry triangle; callers apply their own ray-kind filter.
    pub(super) fn realtime_bvh(&self) -> Option<&Bvh> {
        self.shadows.as_ref().map(|shadows| &shadows.bvh)
    }
}

/// Shadow acceleration for the realtime Beauty engine; built once per geometry and shared by the per-pixel shading of
/// every Beauty hit, flat triangles included.
pub(super) struct RealtimeShadows {
    bvh: Bvh,
    point_maps: BTreeMap<Id, PointShadowMap>,
    /// Whether some triangle can cast a transparent shadow, so shadow rays must evaluate the occluders they hit.
    transparent_casters: bool,
    has_volume_only: bool,
}

const SHADOW_MAP_SIZE: u32 = 64;
const SHADOW_MAP_SIZE_USIZE: usize = 64;

struct ShadowFace {
    forward: DVec3,
    right: DVec3,
    up: DVec3,
    depth: Vec<f64>,
}

struct PointShadowMap {
    position: DVec3,
    faces: Vec<ShadowFace>,
}

/// The shadow rays of one light cross the BVH over every triangle. Point and spot lights take the opaque triangles
/// from their depth map, which holds only triangles that cannot cast a transparent shadow.
#[derive(Clone, Copy)]
struct ShadowScene<'a> {
    bvh: &'a Bvh,
    triangles: &'a [Triangle],
    objects: &'a [ObjectRecord],
    prepared_materials: &'a [PreparedMaterial],
    has_volume_only: bool,
    point_maps: &'a BTreeMap<Id, PointShadowMap>,
    occluders: Option<&'a ShadowOccluders<'a>>,
}

type ShadowContext<'a> = Option<ShadowScene<'a>>;

/// Tolerance of shadow rays along their own surface and at a hit (the epsilon of the realtime shadow code).
const SHADOW_EPSILON: f64 = 1.0e-5;

/// Whether a surface casts a transparent shadow. Volume-only geometry has no surface to occlude a light.
fn casts_transparent_shadow(material: &Material, prepared: &PreparedMaterial) -> bool {
    prepared.is_volume_only() || (material.use_transparent_shadow && prepared.is_transparent())
}

/// EEVEE's transparent shadow pass for one hit (`eevee_surf_shadow.bsl.hh`): a fragment is kept with probability
/// `1 - transparency`, where `transparency` is the sum over the closures of `weight × (1 - alpha) × tint`: the
/// Transparent BSDF's colour, or a Principled Alpha's `1 - alpha` at white tint. The shadow ray passes with the same
/// average. Refraction is no transparent closure (`closure_eval(ClosureTransparency)` is the only writer
/// of `g_transmittance`), so glass casts an opaque shadow.
/// A clip material passes the binarised alpha of its closures, as its visible surface does. This is a divergence, not
/// parity: `blend_bridge.py` `_set_principled_values` exports a clip material as a DITHERED Principled with Alpha equal
/// to its base alpha, so EEVEE's pass on that file is fractional (`1 - alpha` per fragment, stochastically), where
/// Potter's realtime shadow is 0 or 1 per hit.
fn transparent_shadow_pass(closures: &[ShaderClosure]) -> f64 {
    closures
        .iter()
        .map(|closure| {
            let tint = closure.params.transparent_tint;
            let mean_tint = (tint[0] + tint[1] + tint[2]) / 3.0;
            closure.weight * (1.0 - closure.params.alpha.clamp(0.0, 1.0)) * mean_tint
        })
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Evaluates the transparent pass of the triangles a shadow ray hits. A failed evaluation is kept and reported by
/// [`ShadowOccluders::finish`], because the light loops that ask for a pass return no `Result`.
struct ShadowOccluders<'a> {
    doc: &'a SceneDoc,
    geometry: &'a TraceGeometry<'a>,
    images: &'a BTreeMap<String, ImageData>,
    workspace: &'a RefCell<EvaluatorWorkspace>,
    camera_matrices: Option<(&'a DMat4, &'a DMat4)>,
    error: RefCell<Option<PotError>>,
}

impl<'a> ShadowOccluders<'a> {
    fn new(
        doc: &'a SceneDoc,
        geometry: &'a TraceGeometry<'a>,
        images: &'a BTreeMap<String, ImageData>,
        workspace: &'a RefCell<EvaluatorWorkspace>,
        camera_matrices: Option<(&'a DMat4, &'a DMat4)>,
    ) -> Self {
        Self {
            doc,
            geometry,
            images,
            workspace,
            camera_matrices,
            error: RefCell::new(None),
        }
    }

    /// The transparent pass of a hit, or `None` when the triangle cannot cast a transparent shadow and so blocks like
    /// opaque geometry. A failed evaluation is recorded and passes nothing.
    fn pass(&self, hit: BvhHit, origin: DVec3, direction: DVec3) -> Option<f64> {
        match self.try_pass(hit, origin, direction) {
            Ok(pass) => pass,
            Err(error) => {
                let mut slot = self.error.borrow_mut();
                if slot.is_none() {
                    *slot = Some(error);
                }
                Some(0.0)
            }
        }
    }

    fn try_pass(&self, hit: BvhHit, origin: DVec3, direction: DVec3) -> Result<Option<f64>> {
        let triangle = self
            .geometry
            .triangles
            .get(hit.triangle_index)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "BVH returned an invalid triangle index",
                )
            })?;
        let material = self
            .geometry
            .materials
            .get(triangle.material_index)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "triangle references an unmapped material",
                )
            })?;
        let Some(prepared) = self.geometry.prepared.get(triangle.material_index) else {
            return Ok(None);
        };
        if prepared.is_volume_only() {
            return Ok(Some(1.0));
        }
        if !casts_transparent_shadow(material, prepared) {
            return Ok(None);
        }
        let shading = hit_params(
            self.doc,
            self.geometry,
            hit,
            origin,
            direction,
            self.images,
            self.camera_matrices,
            &mut self.workspace.borrow_mut(),
        )?;
        Ok(Some(transparent_shadow_pass(&shading.closures)))
    }

    /// The first failed evaluation, if any shadow ray hit one.
    fn finish(self) -> Result<()> {
        match self.error.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Expected transmittance of a shadow ray from `origin` along `direction`, up to `distance`. A hit on a triangle that
/// cannot cast a transparent shadow blocks the ray when `opaque_blocks` (sun and area lights); a point or spot light
/// passes `false`, because its depth map already holds those triangles.
fn ray_transmittance(
    shadow: ShadowScene<'_>,
    origin: DVec3,
    direction: DVec3,
    distance: f64,
    opaque_blocks: bool,
) -> f64 {
    if opaque_blocks
        && shadow.occluders.is_none()
        && shadow
            .objects
            .iter()
            .all(|object| object.view_layer_use_solid)
    {
        return if shadow.bvh.any_hit_where(
            shadow.triangles,
            origin,
            direction,
            SHADOW_EPSILON,
            distance,
            &|mask| mask & RayKind::Shadow.bit() != 0,
        ) {
            0.0
        } else {
            1.0
        };
    }
    let mut transmittance = 1.0;
    shadow.bvh.walk_hits(
        shadow.triangles,
        WalkRay {
            origin,
            direction,
            epsilon: SHADOW_EPSILON,
            remaining: distance,
            kind: Some(RayKind::Shadow),
        },
        |hit, origin| {
            let surface_visible = shadow
                .triangles
                .get(hit.triangle_index)
                .and_then(|triangle| object_for_index(shadow.objects, triangle.object_index))
                .is_none_or(|object| object.view_layer_use_solid);
            if !surface_visible {
                return true;
            }
            match shadow
                .occluders
                .and_then(|occluders| occluders.pass(hit, origin, direction))
            {
                Some(pass) => transmittance *= pass,
                None if opaque_blocks => {
                    transmittance = 0.0;
                    return false;
                }
                None => {}
            }
            transmittance > 1.0e-5
        },
    );
    if transmittance <= 1.0e-5 {
        0.0
    } else {
        transmittance
    }
}

/// The origin of a shadow ray leaving `point`, lifted to the side of `normal` that `direction` points to.
fn shadow_origin(point: DVec3, normal: DVec3, direction: DVec3) -> DVec3 {
    let side = if normal.dot(direction) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    point + normal * (side * SHADOW_EPSILON)
}

fn cycles_hash_uint2(kx: u32, ky: u32) -> u32 {
    let mut a = 0xdead_beef_u32.wrapping_add((2 << 2) + 13);
    let mut b = a;
    let mut c = a;
    a = a.wrapping_add(kx);
    b = b.wrapping_add(ky);
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(14));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(11));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(25));
    c ^= b;
    c = c.wrapping_sub(b.rotate_left(16));
    a ^= c;
    a = a.wrapping_sub(c.rotate_left(4));
    b ^= a;
    b = b.wrapping_sub(a.rotate_left(14));
    c ^= b;
    c.wrapping_sub(b.rotate_left(24))
}

fn cycles_curve_random(curve_index: usize) -> Option<f64> {
    let index = u32::try_from(curve_index).ok()?;
    let random = cycles_hash_uint2(index, 0);
    Some(f64::from((random as f32) * (1.0_f32 / u32::MAX as f32)))
}

fn hair_curve_geometry_error(
    node_id: &Id,
    error: &super::hair_geometry::HairCurveGeometryError,
) -> PotError {
    use super::hair_geometry::HairCurveGeometryError;

    match error {
        HairCurveGeometryError::Attribute(attribute) => PotError::new(
            ErrorCode::SceneInvalid,
            format!("hair curve attribute {attribute} is invalid"),
        ),
        HairCurveGeometryError::Curve(curve_index) => PotError::with_details(
            ErrorCode::EvaluationFailed,
            "hair curve spline data is invalid",
            serde_json::json!({
                "feature_id":"render.hair.curve_data",
                "node_id":node_id.as_str(),
                "curve_index":curve_index,
            }),
        ),
        HairCurveGeometryError::Transform => PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "path-traced hair requires an invertible affine object transform",
            serde_json::json!({
                "feature_id":"render.hair.object_transform",
                "node_id":node_id.as_str(),
            }),
        ),
    }
}
struct HairCurveBakeObject {
    node_id: Id,
    object_to_world: DMat4,
    world_to_local: Option<DMat4>,
    object_color: [f64; 4],
    object_pass_index: f64,
    object_random_number: Option<f64>,
}

impl HairCurveBakeObject {
    fn from_record(object: &ObjectRecord) -> Self {
        Self {
            node_id: object.node_id.clone(),
            object_to_world: object.object_to_world,
            world_to_local: object.world_to_local,
            object_color: object.object_color,
            object_pass_index: object.object_pass_index,
            object_random_number: object.object_random_number,
        }
    }

    fn from_live_node(doc: &SceneDoc, snapshot: &Snapshot, node_id: &Id) -> Result<Self> {
        let node = doc.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "Cycles Hair representative node is missing",
            )
        })?;
        let evaluated = snapshot.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "Cycles Hair representative transform is missing",
            )
        })?;
        let object_to_world = DMat4::from_cols_array(&evaluated.world_matrix);
        let world_to_local = object_inverse(object_to_world).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Cycles Hair shadow bake requires an invertible representative transform",
                serde_json::json!({
                    "feature_id":"render.hair.object_transform",
                    "node_id":node_id.as_str(),
                }),
            )
        })?;
        let object_color = object_info_color(node);
        if object_color.iter().any(|value| !value.is_finite()) {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "Cycles Hair representative Object Info color is non-finite",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":node_id.as_str(),
                }),
            ));
        }
        Ok(Self {
            node_id: node_id.clone(),
            object_to_world,
            world_to_local: Some(world_to_local),
            object_color,
            object_pass_index: object_info_pass_index(node),
            object_random_number: Some(f64::from(crate::blender_rng::cycles_object_random_number(
                &node.name,
            ))),
        })
    }

    fn from_member_context(member: &crate::model::CyclesHairShadowMember) -> Result<Self> {
        let node_id = &member.instance.real_object_node_id;
        let matrix = cycles_hair_object_matrix(member.object_to_world);
        if matrix
            .to_cols_array()
            .iter()
            .any(|value| !value.is_finite())
            || member.object_color.iter().any(|value| !value.is_finite())
        {
            return Err(PotError::with_details(
                ErrorCode::SceneInvalid,
                "Cycles Hair member context contains non-finite object data",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":node_id.as_str(),
                }),
            ));
        }
        let world_to_local = object_inverse(matrix).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Cycles Hair member context requires an invertible object transform",
                serde_json::json!({
                    "feature_id":"render.hair.object_transform",
                    "node_id":node_id.as_str(),
                }),
            )
        })?;
        Ok(Self {
            node_id: node_id.clone(),
            object_to_world: matrix,
            world_to_local: Some(world_to_local),
            object_color: member.object_color,
            object_pass_index: f64::from(member.object_pass_index),
            object_random_number: Some(f64::from(
                crate::blender_rng::cycles_object_random_number_from_signed_id(
                    member.object_random_id,
                ),
            )),
        })
    }
}

fn cycles_hair_object_matrix(rows: [[f64; 4]; 4]) -> DMat4 {
    DMat4::from_cols_array(&[
        rows[0][0], rows[1][0], rows[2][0], rows[3][0], rows[0][1], rows[1][1], rows[2][1],
        rows[3][1], rows[0][2], rows[1][2], rows[2][2], rows[3][2], rows[0][3], rows[1][3],
        rows[2][3], rows[3][3],
    ])
}
type CyclesHairMemberIndices = BTreeMap<usize, BTreeMap<Id, Vec<usize>>>;

fn cycles_hair_shadow_group_index(
    scene: &crate::model::Scene,
    snapshot: &Snapshot,
    node_id: &Id,
    data_id: Option<&Id>,
    member_indices: &CyclesHairMemberIndices,
) -> Result<Option<usize>> {
    let Some(groups) = scene.cycles_hair_shadow_groups.as_deref() else {
        return Ok(None);
    };
    let mut first: Option<usize> = None;
    for (index, group) in groups.iter().enumerate() {
        if snapshot
            .view_layer
            .as_ref()
            .is_some_and(|view_layer| &group.view_layer_id != view_layer)
            || !member_indices
                .get(&index)
                .is_some_and(|members| members.contains_key(node_id))
        {
            continue;
        }
        let geometry_matches = match &group.geometry_key {
            CyclesHairShadowGeometryKey::Data {
                data_id: Some(source_data_id),
            } => data_id == Some(source_data_id),
            CyclesHairShadowGeometryKey::Data { data_id: None } => true,
            CyclesHairShadowGeometryKey::Object {
                node_id: object_node_id,
            } => object_node_id == node_id,
        };
        if !geometry_matches {
            continue;
        }
        if let Some(first_index) = first {
            if !equivalent_cycles_hair_shadow_groups(&groups[first_index], group) {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "multiple distinct Cycles Hair shadow groups match the current object context",
                    serde_json::json!({
                        "feature_id":"render.hair.shadow_cache_provenance",
                        "node_id":node_id.as_str(),
                        "view_layers":[
                            groups[first_index].view_layer_id.as_str(),
                            group.view_layer_id.as_str(),
                        ],
                    }),
                ));
            }
        } else {
            first = Some(index);
        }
    }
    let Some(first) = first else {
        if member_indices
            .values()
            .any(|members| members.contains_key(node_id))
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Cycles Hair shadow provenance no longer matches the current node data",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":node_id.as_str(),
                    "data_id":data_id.map(Id::as_str),
                }),
            ));
        }
        return Ok(None);
    };
    Ok(Some(first))
}

fn equivalent_cycles_hair_shadow_groups(
    left: &CyclesHairShadowGroup,
    right: &CyclesHairShadowGroup,
) -> bool {
    left.geometry_key == right.geometry_key
        && left.representative_node_id == right.representative_node_id
        && left.member_node_ids == right.member_node_ids
        && left.members == right.members
        && left.source_curve_indices == right.source_curve_indices
        && left.evaluated_hair_curves == right.evaluated_hair_curves
        && left.source_input_fingerprint == right.source_input_fingerprint
}

/// Cycles uses an object-keyed Hair geometry cache for modified objects, metaballs, or object-linked material slots.
fn cycles_hair_geometry_key_is_object(
    node: &Node,
    data_block: Option<&crate::model::DataBlock>,
) -> bool {
    data_block.is_some_and(|data| data.data_type == "metaball")
        || node
            .modifiers
            .iter()
            .any(|modifier| modifier.is_active(crate::eval::EvaluationPurpose::Render))
        || (0..crate::model::material_slot_count(
            node.materials.len(),
            data_block.map_or(0, |data| data.materials.len()),
        ))
            .any(|slot| {
                crate::model::material_link_at(node.materials.len(), &node.material_links, slot)
                    == crate::model::MaterialLink::Object
            })
}

struct HairPathMember {
    object_index: u32,
    material_index: usize,
    ray_mask: u8,
    prepared_curves: Vec<Option<super::hair_geometry::PreparedHairCurve>>,
}

fn append_hair_instance_object(
    objects: &mut Vec<ObjectRecord>,
    base_object_index: u32,
    member: &CyclesHairShadowMember,
    material_index: usize,
    materials: &[Material],
    is_holdout: bool,
    shadow_terminator_normal_offset: f64,
    shadow_terminator_geometry_offset: f64,
    shadow_terminator_shading_offset: f64,
    force_context: bool,
) -> Result<u32> {
    if !member.instance.is_instance && !force_context {
        return Ok(base_object_index);
    }
    let base_index = usize::try_from(base_object_index.checked_sub(1).ok_or_else(|| {
        PotError::new(
            ErrorCode::InternalError,
            "hair base object index is invalid",
        )
    })?)
    .map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "hair object index exceeds platform limits",
        )
    })?;
    let object_to_world = cycles_hair_object_matrix(member.object_to_world);
    if !object_to_world
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
        || member.object_color.iter().any(|value| !value.is_finite())
    {
        return Err(PotError::with_details(
            ErrorCode::SceneInvalid,
            "Cycles Hair member context contains non-finite object data",
            serde_json::json!({
                "feature_id":"render.hair.shadow_cache_provenance",
                "node_id":member.instance.real_object_node_id.as_str(),
            }),
        ));
    }
    let world_to_local = object_inverse(object_to_world).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Cycles Hair member transform must be invertible",
            serde_json::json!({
                "feature_id":"render.hair.object_transform",
                "node_id":member.instance.real_object_node_id.as_str(),
            }),
        )
    })?;
    let index = if member.instance.is_instance {
        u32::try_from(objects.len() + 1).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "too many objects for the ID buffer",
            )
        })?
    } else {
        base_object_index
    };
    if member.instance.is_instance {
        let mut object = objects.get(base_index).cloned().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "hair instance base object record is missing",
            )
        })?;
        update_hair_member_object_record(
            &mut object,
            index,
            member,
            object_to_world,
            world_to_local,
            material_index,
            materials,
            is_holdout,
            shadow_terminator_normal_offset,
            shadow_terminator_geometry_offset,
            shadow_terminator_shading_offset,
        );
        objects.push(object);
    } else {
        let object = objects.get_mut(base_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "hair base object record is missing",
            )
        })?;
        update_hair_member_object_record(
            object,
            index,
            member,
            object_to_world,
            world_to_local,
            material_index,
            materials,
            is_holdout,
            shadow_terminator_normal_offset,
            shadow_terminator_geometry_offset,
            shadow_terminator_shading_offset,
        );
    }
    Ok(index)
}

fn update_hair_member_object_record(
    object: &mut ObjectRecord,
    index: u32,
    member: &CyclesHairShadowMember,
    object_to_world: DMat4,
    world_to_local: DMat4,
    material_index: usize,
    materials: &[Material],
    is_holdout: bool,
    shadow_terminator_normal_offset: f64,
    shadow_terminator_geometry_offset: f64,
    shadow_terminator_shading_offset: f64,
) {
    object.index = index;
    object.node_id = member.instance.real_object_node_id.clone();
    object.object_to_world = object_to_world;
    object.world_to_local = Some(world_to_local);
    object.object_color = member.object_color;
    object.object_pass_index = f64::from(member.object_pass_index);
    object.object_random_number = Some(f64::from(
        crate::blender_rng::cycles_object_random_number_from_signed_id(member.object_random_id),
    ));
    object.is_holdout = is_holdout;
    object.shadow_terminator_normal_offset = shadow_terminator_normal_offset;
    object.shadow_terminator_geometry_offset = shadow_terminator_geometry_offset;
    object.shadow_terminator_shading_offset = shadow_terminator_shading_offset;
    if let Some(material) = materials.get(material_index) {
        object.material_colors.insert(
            material_index,
            None,
            material.pass_index.map(f64::from),
            None,
            None,
        );
    }
}

#[derive(Clone, Copy)]
struct HairCurveBakeContext<'a> {
    doc: &'a SceneDoc,
    object: &'a HairCurveBakeObject,
    material: &'a Material,
    prepared: &'a PreparedMaterial,
    images: &'a BTreeMap<String, ImageData>,
    warnings: &'a RefCell<Vec<serde_json::Value>>,
}

fn bake_cycles_hair_shadow_transparency(
    context: HairCurveBakeContext<'_>,
    curve: &super::hair_geometry::PreparedHairCurve,
    key_count: usize,
    curve_random: Option<f64>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<super::hair_geometry::HairCurveShadowTransparency> {
    if !context.material.use_transparent_shadow || !context.prepared.is_transparent() {
        return Ok(super::hair_geometry::HairCurveShadowTransparency::new(
            None, false,
        ));
    }
    let world_to_local = context.object.world_to_local.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "hair shader evaluation requires an invertible object transform",
            serde_json::json!({
                "feature_id":"render.hair.object_transform",
                "node_id":context.object.node_id.as_str(),
            }),
        )
    })?;
    let mut values = Vec::new();
    values.try_reserve_exact(key_count).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "hair shadow transparency key allocation failed",
        )
    })?;
    for key_index in 0..key_count {
        let (span, u) = if key_index + 1 == key_count {
            (key_count - 2, 1.0)
        } else {
            (key_index, 0.0)
        };
        let sample = curve
            .curve
            .sample_local_for_shadow_bake(span, u)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "Cycles hair shadow bake could not sample a source key",
                )
            })?;
        let point = context
            .object
            .object_to_world
            .transform_point3(sample.position);
        let tangent = context
            .object
            .object_to_world
            .transform_vector3(sample.tangent);
        let curve_normal = curve.point_normals.at(span, u).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "Cycles hair shadow bake could not sample a source point normal",
            )
        })?;
        let view = DVec3::Y.cross(tangent).normalize_or_zero();
        let mut hit_context = HitContext::new(&context.doc.node_groups)
            .with_images(context.images)
            .with_warnings(context.warnings);
        hit_context.position = point;
        hit_context.object = world_to_local.transform_point3(point);
        hit_context.object_to_world = context.object.object_to_world;
        hit_context.world_to_object = world_to_local;
        hit_context.object_color = context.object.object_color;
        hit_context.object_pass_index = context.object.object_pass_index;
        hit_context.object_random_number = context.object.object_random_number;
        hit_context.material_pass_index = context.material.pass_index.map(f64::from);
        hit_context.normal = view;
        hit_context.geometric_normal = view;
        hit_context.view_direction = view;
        hit_context.light_falloff_distance = Some(0.0);
        hit_context.path_traced = true;
        hit_context.hair_curve_hit = true;
        hit_context = hit_context
            .with_footprint(HitFootprint {
                dp: 0.0,
                geometric_normal: view,
            })
            .with_hair_context(tangent, Some(curve_normal), curve_random);
        let shading = evaluate_surface_with_workspace(
            context.material,
            context.prepared,
            &hit_context,
            evaluator_workspace,
        )?;
        values.push(cycles_curve_shadow_value(&shading));
    }
    Ok(super::hair_geometry::HairCurveShadowTransparency::new(
        Some(values),
        false,
    ))
}

fn cycles_curve_shadow_value(shading: &SurfaceShading) -> f32 {
    let (extinction, has_transparent_flag) = super::path::cycles_shadow_extinction(
        shading
            .closures
            .iter()
            .map(|closure| (closure.weight, closure.params)),
    );
    if !has_transparent_flag {
        return 0.0;
    }
    ((extinction[0] + extinction[1] + extinction[2]) / 3.0).clamp(0.0, 1.0)
}

fn validate_cycles_hair_shadow_group_freshness(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    group: &CyclesHairShadowGroup,
) -> Result<()> {
    let captured = group.source_input_fingerprint.as_deref();
    let current = doc.cycles_hair_shadow_source_fingerprint(&snapshot.scene_id, group)?;
    if let Some(captured) = captured {
        if current.as_deref() == Some(captured) {
            return Ok(());
        }
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Cycles Hair evaluated geometry dependencies changed; reimport from Blender",
            serde_json::json!({
                "feature_id":"render.hair.shadow_cache_provenance",
                "node_id":group.representative_node_id.as_str(),
                "captured_fingerprint":captured,
                "current_fingerprint":current,
            }),
        ));
    }
    let has_instance = group
        .members
        .iter()
        .any(|member| member.instance.is_instance && !member.instance.use_particle_hair);
    let has_render_modifier = group.members.iter().any(|member| {
        doc.nodes
            .get(&member.instance.real_object_node_id)
            .is_some_and(|node| {
                node.modifiers
                    .iter()
                    .any(|modifier| modifier.is_active(crate::eval::EvaluationPurpose::Render))
            })
    });
    if group.evaluated_hair_curves.is_some()
        || has_instance
        || has_render_modifier
        || group
            .members
            .iter()
            .filter(|member| !member.instance.use_particle_hair)
            .count()
            > 1
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Cycles Hair geometry dependency closure was not captured; reimport from Blender",
            serde_json::json!({
                "feature_id":"render.hair.shadow_cache_provenance",
                "node_id":group.representative_node_id.as_str(),
            }),
        ));
    }
    Ok(())
}
fn bake_cycles_hair_shadow_group(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    group: &CyclesHairShadowGroup,
    hair_data: &crate::model::HairCurvesData,
    evaluated_to_source: Option<&[u32]>,
    scene_render_mode: HairCurveRenderMode,
    ribbon_subdivisions: usize,
    default_surface: &Material,
    images: &BTreeMap<String, ImageData>,
    warnings: &RefCell<Vec<serde_json::Value>>,
    evaluator_workspace: &mut EvaluatorWorkspace,
) -> Result<Vec<Arc<super::hair_geometry::HairCurveShadowTransparency>>> {
    validate_cycles_hair_shadow_group_freshness(doc, snapshot, group)?;
    let representative = group.members.first().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::SceneInvalid,
            "Cycles Hair shadow group has no representative member",
            serde_json::json!({
                "feature_id":"render.hair.shadow_cache_provenance",
                "node_id":group.representative_node_id.as_str(),
            }),
        )
    })?;
    let representative_node_id = &representative.instance.real_object_node_id;
    let bake_object = if representative.instance.is_instance {
        HairCurveBakeObject::from_member_context(representative)?
    } else {
        HairCurveBakeObject::from_live_node(doc, snapshot, representative_node_id)?
    };
    let prepared_curves = super::hair_geometry::prepare_cycles_hair_curve_geometry(
        hair_data,
        bake_object.object_to_world,
        scene_render_mode,
        ribbon_subdivisions,
    )
    .map_err(|error| hair_curve_geometry_error(&group.representative_node_id, &error))?;
    let representative_node = doc.nodes.get(representative_node_id).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "Cycles Hair representative object is missing",
        )
    })?;
    let representative_data = representative_node
        .data
        .as_ref()
        .and_then(|data_id| doc.data_blocks.get(data_id));
    let material_id = if group.evaluated_hair_curves.is_some() {
        representative.slot_0_material_id.as_ref()
    } else {
        snapshot
            .slot_material(
                representative_node_id,
                representative_node,
                representative_data,
                0,
            )
            .or(representative.slot_0_material_id.as_ref())
    };
    let material = match material_id {
        Some(material_id) => doc.materials.get(material_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::SceneInvalid,
                "Cycles Hair shadow representative material is missing",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":group.representative_node_id.as_str(),
                    "material_id":material_id.as_str(),
                }),
            )
        })?,
        None => default_surface,
    };
    let prepared_material = PreparedMaterial::new(material, &doc.node_groups);
    let curve_count = evaluated_to_source.map_or(hair_data.curves.len(), <[u32]>::len);
    let mut shadow_transparency = Vec::new();
    shadow_transparency
        .try_reserve_exact(curve_count)
        .map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Cycles Hair shadow cache allocation failed",
            )
        })?;
    for evaluated_curve_index in 0..curve_count {
        let source_curve_index = evaluated_to_source
            .map(|indices| indices[evaluated_curve_index])
            .map_or(Some(evaluated_curve_index), |index| {
                usize::try_from(index).ok()
            })
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Cycles Hair source curve index cannot be represented",
                    serde_json::json!({
                        "feature_id":"render.hair.shadow_cache_provenance",
                        "node_id":group.representative_node_id.as_str(),
                        "evaluated_curve_index":evaluated_curve_index,
                    }),
                )
            })?;
        let curve_data = hair_data.curves.get(source_curve_index).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Cycles Hair source curve mapping references unavailable geometry",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":group.representative_node_id.as_str(),
                    "source_curve_index":source_curve_index,
                }),
            )
        })?;
        let prepared_curve = prepared_curves.get(source_curve_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "Cycles Hair prepared source curve is missing",
            )
        })?;
        let curve_random = cycles_curve_random(evaluated_curve_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "hair curve index exceeds Cycles' random attribute range",
            )
        })?;
        let transparency = if curve_data.points.len() > 1 {
            bake_cycles_hair_shadow_transparency(
                HairCurveBakeContext {
                    doc,
                    object: &bake_object,
                    material,
                    prepared: &prepared_material,
                    images,
                    warnings,
                },
                prepared_curve,
                curve_data.points.len(),
                Some(curve_random),
                evaluator_workspace,
            )?
        } else {
            super::hair_geometry::HairCurveShadowTransparency::new(None, false)
        };
        shadow_transparency.push(Arc::new(transparency));
    }
    Ok(shadow_transparency)
}

fn eevee_world_volume_absorption_refusal(scene_id: &Id) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "EEVEE World.Volume is not evaluated, so its absorption cannot be checked for WorldSun extraction; reimport without a World.Volume connection",
        serde_json::json!({"feature_id":"render.world.volume_absorption","scene_id":scene_id.as_str()}),
    )
}
pub(super) fn selected_view_layer<'a>(
    doc: &'a SceneDoc,
    snapshot: &Snapshot,
) -> Option<&'a crate::model::ViewLayer> {
    let scene = doc.scenes.get(&snapshot.scene_id)?;
    snapshot
        .view_layer
        .as_ref()
        .and_then(|id| scene.view_layers.get(id))
}

pub(super) fn selected_world_id<'a>(doc: &'a SceneDoc, snapshot: &Snapshot) -> Option<&'a Id> {
    let scene = doc.scenes.get(&snapshot.scene_id)?;
    selected_view_layer(doc, snapshot)
        .and_then(|layer| layer.world_override.as_ref())
        .or(scene.world.as_ref())
}

pub(super) fn extract_geometry(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    mode: Mode,
    path_traced: bool,
    images: &BTreeMap<String, ImageData>,
) -> Result<Geometry> {
    extract_geometry_with_semantics(
        doc,
        snapshot,
        mode,
        path_traced,
        images,
        crate::node_kernels::MathSemantics::Cycles,
    )
}

pub(super) fn extract_geometry_with_semantics(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    mode: Mode,
    path_traced: bool,
    images: &BTreeMap<String, ImageData>,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<Geometry> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let source_mesh_attribute_precision = if math_semantics
        == crate::node_kernels::MathSemantics::Eevee
        && matches!(mode, Mode::Beauty)
    {
        scene.render.use_high_quality_normals.map_or(
            MeshAttributePrecision::Unknown,
            MeshAttributePrecision::Known,
        )
    } else {
        MeshAttributePrecision::NotApplicable
    };
    if matches!(mode, Mode::Beauty)
        && math_semantics == crate::node_kernels::MathSemantics::Eevee
        && super::path::scene_has_world_volume(doc, snapshot)
    {
        return Err(eevee_world_volume_absorption_refusal(&snapshot.scene_id));
    }

    let blender_imported = doc.compatibility.contains_key("blender_adapter");
    let curve_render_mode = match scene.render.cycles_curves_shape.as_ref() {
        Some(CyclesCurveShape::Ribbons) => HairCurveRenderMode::Ribbon,
        Some(CyclesCurveShape::Thick) | None => HairCurveRenderMode::Thick,
        Some(CyclesCurveShape::ThickLinear) => HairCurveRenderMode::ThickLinear,
    };
    let curve_ribbon_subdivisions = scene.render.cycles_curves_subdivisions.unwrap_or(2) as usize;
    let view_layer = selected_view_layer(doc, snapshot);
    let view_layer_use_solid = view_layer.is_none_or(|layer| layer.use_solid);
    let view_layer_use_volumes = view_layer.is_none_or(|layer| layer.use_volumes);
    let view_layer_use_strand = view_layer.is_none_or(|layer| layer.use_strand);
    let material_override = view_layer.and_then(|layer| layer.material_override.as_ref());
    let excluded: HashSet<Id> = view_layer
        .map(|layer| layer.excluded_collections.iter().cloned().collect())
        .unwrap_or_default();
    let mut visible_nodes = HashSet::new();
    let mut visited = HashSet::new();
    collect_collection_nodes(
        doc,
        &scene.root_collection,
        &excluded,
        &mut visited,
        &mut visible_nodes,
    );
    let mut instanced_hair_nodes = HashSet::new();
    let mut hair_member_indices = CyclesHairMemberIndices::new();
    let mut hair_group_nodes = HashSet::new();
    if path_traced
        && matches!(mode, Mode::Beauty)
        && view_layer_use_strand
        && let Some(groups) = scene.cycles_hair_shadow_groups.as_deref()
    {
        for (group_index, group) in groups.iter().enumerate() {
            if snapshot
                .view_layer
                .as_ref()
                .is_some_and(|view_layer_id| view_layer_id != &group.view_layer_id)
            {
                continue;
            }
            let mut members_by_node = BTreeMap::<Id, Vec<usize>>::new();
            for (member_index, member) in group.members.iter().enumerate() {
                if member.instance.use_particle_hair {
                    continue;
                }
                hair_group_nodes.insert(member.instance.real_object_node_id.clone());
                members_by_node
                    .entry(member.instance.real_object_node_id.clone())
                    .or_default()
                    .push(member_index);
                if member.instance.is_instance {
                    let source_has_hair = doc
                        .nodes
                        .get(&member.instance.real_object_node_id)
                        .and_then(|node| node.data.as_ref())
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .is_some_and(|data| data.hair_curves.is_some());
                    if source_has_hair || group.evaluated_hair_curves.is_some() {
                        instanced_hair_nodes.insert(member.instance.real_object_node_id.clone());
                    }
                }
            }
            if !members_by_node.is_empty() {
                hair_member_indices.insert(group_index, members_by_node);
            }
        }
    }
    if path_traced
        && matches!(mode, Mode::Beauty)
        && view_layer_use_strand
        && scene.cycles_hair_shadow_groups.is_some()
    {
        for (node_id, node) in &doc.nodes {
            let has_hair = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .is_some_and(|data| data.hair_curves.is_some());
            let has_instance_path = snapshot
                .instance_paths
                .get(node_id)
                .is_some_and(|paths| paths.iter().any(|path| !path.is_empty()));
            let has_group = scene
                .cycles_hair_shadow_groups
                .as_deref()
                .is_some_and(|groups| {
                    groups.iter().any(|group| {
                        snapshot
                            .view_layer
                            .as_ref()
                            .is_none_or(|view_layer_id| &group.view_layer_id == view_layer_id)
                            && group.member_node_ids.contains(node_id)
                    })
                });
            if has_hair && has_instance_path && !has_group {
                instanced_hair_nodes.insert(node_id.clone());
            }
        }
    }
    if path_traced
        && matches!(mode, Mode::Beauty)
        && blender_imported
        && scene.cycles_hair_shadow_groups.is_none()
    {
        for (node_id, node) in &doc.nodes {
            let has_hair = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .is_some_and(|data| data.hair_curves.is_some());
            let has_instance_path = snapshot
                .instance_paths
                .get(node_id)
                .is_some_and(|paths| paths.iter().any(|path| !path.is_empty()));
            if has_hair && has_instance_path {
                instanced_hair_nodes.insert(node_id.clone());
            }
        }
    }

    // Workbench's Solid shades faces with no material as Blender's empty default material:
    // colour 0.8, roughness 0.4, metallic 0 (`DNA_material_types.h`; `BKE_material_default_empty`
    // via `workbench_engine.cc` `get_material`). Beauty keeps Potter's default material.
    let default_surface = if matches!(mode, Mode::Solid) {
        Material {
            base_color: [0.8, 0.8, 0.8, 1.0],
            roughness: 0.4,
            ..Material::default()
        }
    } else {
        Material::default()
    };
    let default_prepared = PreparedMaterial::new(&default_surface, &doc.node_groups);
    let mut materials = vec![default_surface];
    let mut prepared = vec![default_prepared];
    let mut material_indices =
        std::collections::BTreeMap::<(Id, Option<usize>, Option<Arc<[u64]>>), usize>::new();
    let mut hair_curve_instances = Vec::new();
    let mut hair_shadow_caches =
        BTreeMap::<usize, Vec<Arc<super::hair_geometry::HairCurveShadowTransparency>>>::new();
    let mut legacy_hair_shadow_caches = BTreeMap::<
        Id,
        (
            Id,
            Vec<Arc<super::hair_geometry::HairCurveShadowTransparency>>,
        ),
    >::new();
    let mut triangles = Vec::new();
    let mut lines = Vec::new();
    let mut wire_edges = Vec::new();
    let mut wire_points = Vec::new();
    let mut objects = Vec::new();
    let mut elements = Vec::new();
    let mut bounds: Option<(DVec3, DVec3)> = None;
    let mut warnings = Vec::new();
    let shader_warnings: RefCell<Vec<serde_json::Value>> = RefCell::new(Vec::new());
    let mut evaluator_workspace = EvaluatorWorkspace::default();
    let has_uv_map_tangent = doc.node_groups.values().any(|group| {
        group.nodes.values().any(|node| {
            node.node_type == "ShaderNodeTangent"
                && node
                    .properties
                    .get("direction_type")
                    .and_then(serde_json::Value::as_str)
                    == Some("UV_MAP")
        })
    });
    for (node_id, node) in &doc.nodes {
        let base_visible = visible_nodes.contains(node_id)
            && snapshot.node_is_visible(node_id, node)
            && (!matches!(mode, Mode::Beauty)
                || crate::exchange::evaluated_visibility(snapshot, node_id, node).1);
        let hair_instance_only = path_traced
            && matches!(mode, Mode::Beauty)
            && !base_visible
            && instanced_hair_nodes.contains(node_id);
        if !base_visible && !hair_instance_only {
            continue;
        }
        let ray_visibility = node.resolved_ray_visibility();
        if !path_traced && matches!(mode, Mode::Beauty) {
            ray_visibility_warnings(node_id, node, &mut warnings);
        }
        let data_block = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        let path_curves_enabled =
            path_traced && matches!(mode, Mode::Beauty) && view_layer_use_strand;
        let source_hair = if view_layer_use_strand {
            data_block.and_then(|data| data.hair_curves.as_ref())
        } else {
            None
        };
        let source_material_id = snapshot.slot_material(node_id, node, data_block, 0);
        let hair_shadow_group_index = if path_curves_enabled {
            cycles_hair_shadow_group_index(
                scene,
                snapshot,
                node_id,
                node.data.as_ref(),
                &hair_member_indices,
            )?
        } else {
            None
        };
        let hair_shadow_group = hair_shadow_group_index
            .and_then(|index| scene.cycles_hair_shadow_groups.as_ref()?.get(index));
        let evaluated_hair =
            hair_shadow_group.and_then(|group| group.evaluated_hair_curves.as_ref());
        let member_material_id = hair_shadow_group.and_then(|group| {
            group
                .members
                .iter()
                .find(|member| {
                    member.instance.real_object_node_id == *node_id
                        && !member.instance.use_particle_hair
                })
                .and_then(|member| member.slot_0_material_id.as_ref())
        });
        let material_id = if let Some(override_id) = material_override {
            Some(override_id)
        } else if evaluated_hair.is_some() {
            member_material_id.or(source_material_id)
        } else {
            source_material_id.or(member_material_id)
        };
        let hair_shadow_curve_indices = if evaluated_hair.is_some() {
            None
        } else {
            hair_shadow_group.and_then(|group| group.source_curve_indices.as_deref())
        };
        let hair = evaluated_hair.or(source_hair);
        if path_curves_enabled
            && blender_imported
            && hair_shadow_group_index.is_none()
            && source_hair.is_some()
            && node
                .modifiers
                .iter()
                .any(|modifier| modifier.is_active(crate::eval::EvaluationPurpose::Render))
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "renderable Blender Hair modifier geometry was not captured; reimport from Blender",
                serde_json::json!({
                    "feature_id":"render.hair.evaluated_geometry",
                    "node_id":node_id.as_str(),
                }),
            ));
        }
        if path_curves_enabled
            && blender_imported
            && hair.is_some()
            && snapshot
                .instance_paths
                .get(node_id)
                .is_some_and(|paths| paths.iter().any(|path| !path.is_empty()))
            && (scene.cycles_hair_shadow_groups.is_none() || hair_shadow_group_index.is_none())
        {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Blender Hair instance transforms were not captured; reimport from Blender",
                serde_json::json!({
                    "feature_id":"render.hair.shadow_cache_provenance",
                    "node_id":node_id.as_str(),
                }),
            ));
        }
        let volume = node
            .data
            .as_ref()
            .and_then(|data_id| snapshot.volume_data.get(data_id))
            .or_else(|| data_block.and_then(|data| data.volume.as_ref()));
        let strokes = snapshot.strokes.get(node_id);
        if let Some(crate::geom::volume::VolumeSource::File(source)) =
            volume.map(|volume| &volume.source)
            && source.format != "vdb"
        {
            let code = "volume.file_format";
            if !matches!(mode, Mode::Solid | Mode::Wire) {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!(
                        "file-backed volume format `{}` cannot be rendered",
                        source.format
                    ),
                    serde_json::json!({"feature_id":code,"node_id":node_id}),
                ));
            }
            warnings.push(serde_json::json!({
                "code": code,
                "message": "unsupported file-backed volume content was skipped; Blender-reported bounds were retained",
                "data_id": node.data.as_ref().map(Id::as_str),
            }));
        }
        let raster_volume_mesh = if matches!(mode, Mode::Solid | Mode::Wire)
            && volume.is_some_and(|volume| match &volume.source {
                crate::geom::volume::VolumeSource::Generated(_) => true,
                crate::geom::volume::VolumeSource::File(source) => {
                    source.format == "vdb" && volume.decoded_vdb.is_some()
                }
            }) {
            let volume = volume.ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "volume data disappeared during render",
                )
            })?;
            let iso_level = volume
                .decoded_vdb
                .as_ref()
                .and_then(|decoded| decoded.density_grid())
                .and_then(|grid| grid.class.as_deref())
                .filter(|class| class.to_ascii_lowercase().contains("level set"))
                .map_or(0.5, |_| 0.0);
            Some(crate::geom::volume::volume_to_mesh(volume, iso_level)?)
        } else {
            None
        };
        let mesh = (!(hair_instance_only || evaluated_hair.is_some()))
            .then(|| snapshot.meshes.get(node_id).or(raster_volume_mesh.as_ref()))
            .flatten();
        if mesh.is_none()
            && strokes.is_none()
            && hair.is_none()
            && volume.is_none()
            && !hair_instance_only
            && !hair_group_nodes.contains(node_id)
        {
            continue;
        }
        let evaluated = snapshot.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "render node is missing from snapshot",
            )
        })?;
        let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
        // Match Cycles' `triangle_normal` for `object_negative_scale_applied`: reverse baked winding,
        // not corner normals.
        let mirrored = matrix.determinant() < 0.0;
        if volume.is_some()
            && let Some(volume_bounds) = evaluated.bounds
        {
            bounds = Some(match bounds {
                Some((current_min, current_max)) => (
                    current_min.min(volume_bounds.min),
                    current_max.max(volume_bounds.max),
                ),
                None => (volume_bounds.min, volume_bounds.max),
            });
        }
        let object_index = u32::try_from(objects.len() + 1).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "too many objects for the ID buffer",
            )
        })?;
        let uv_store = mesh.map_or_else(UvStore::default, uv_layers::read);
        if uv_store.layers.len() > uv_layers::MAX_LAYERS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "mesh has more UV layers than the supported maximum",
            ));
        }
        let uv_names: Vec<String> = uv_store
            .layers
            .iter()
            .map(|layer| layer.name.clone())
            .collect();
        let uv_render = (!uv_names.is_empty()).then_some(uv_store.render_index().unwrap_or(0));
        let color_store = mesh.map_or_else(ColorStore::default, color_layers::read);
        let geometry_attributes = mesh.map_or_else(Vec::new, mesh_geometry_attributes);
        let has_geometry_attributes = !geometry_attributes.is_empty();
        let color_slot_count = snapshot.slot_count(node_id, node, data_block).max(1);
        let mut color_slots = Vec::with_capacity(color_slot_count);
        for slot in 0..color_slot_count {
            let binding = snapshot.slot_binding(node_id, node, slot);
            let material_id = snapshot.slot_material(node_id, node, data_block, slot);
            let references = material_id
                .map(|material| color_layer_references(doc, material))
                .unwrap_or_default();
            let available_mask = binding.map(|binding| Arc::clone(&binding.color_available_mask));
            let render_layer = if references.uses_render_layer {
                let layer = binding.map_or_else(
                    || color_store.render_index(),
                    |binding| binding.color_render_layer,
                );
                layer.filter(|layer| {
                    *layer < color_store.layers.len()
                        && source_has_color_layer(available_mask.as_deref(), *layer)
                })
            } else {
                None
            };
            color_slots.push(ColorSlotReferences {
                names: references.names,
                render_layer,
                available_mask,
                requires_hit_shading: references.requires_hit_shading,
                reads_colors: references.reads_colors,
            });
        }
        let material_reads_colors = color_slots
            .iter()
            .map(|slot| slot.reads_colors)
            .collect::<Vec<_>>();
        let material_requires_hit_shading = color_slots
            .iter()
            .map(|slot| slot.requires_hit_shading)
            .collect::<Vec<_>>();
        // Node graphs and native texture fields require per-hit sampling in realtime Beauty.
        let (carried, color_render_layers) = carried_color_layers(&color_store, &color_slots);
        if carried.len() > MAX_COLOR_LAYERS {
            return Err(PotError::new(
                ErrorCode::LimitExceeded,
                "mesh has more color layers than one object can carry",
            ));
        }
        let color_available_masks = color_slots
            .iter()
            .map(|slot| slot.available_mask.clone())
            .collect::<Vec<_>>();
        let carried_layers: Vec<&ColorLayer> = carried
            .iter()
            .map(|&index| &color_store.layers[index])
            .collect();
        let thickness = mesh.map_or(0.0, |mesh| transmission_thickness(mesh, matrix));
        // Blender's texture space is computed on the undeformed mesh (`BKE_mesh_eval_geometry` calls
        // `BKE_mesh_texspace_calc` on the original), and modifiers copy it to their result. The data block's mesh is
        // that original, so modifiers that rebuild or shrink the evaluated mesh do not move the texture space.
        let undeformed = data_block.and_then(|data| data.mesh.as_ref()).or(mesh);
        let texture_space = undeformed
            .and_then(Mesh::manual_texture_space)
            .unwrap_or_else(|| TextureSpace::automatic(undeformed.and_then(Mesh::bounds)));
        let instance_path = snapshot
            .instance_paths
            .get(node_id)
            .and_then(|paths| paths.first())
            .cloned()
            .unwrap_or_default();
        let object_color = object_info_color(node);
        let object_pass_index = object_info_pass_index(node);
        let object_random_number = instance_path
            .is_empty()
            .then(|| f64::from(crate::blender_rng::cycles_object_random_number(&node.name)));
        objects.push(ObjectRecord {
            index: object_index,
            node_id: node_id.clone(),
            data_id: node.data.clone(),
            instance_path,
            uv_layers: uv_names.clone(),
            uv_render,
            color_layers: carried_layers
                .iter()
                .map(|layer| layer.name.clone())
                .collect(),
            color_layer_indices: carried,
            geometry_attributes,
            thickness,
            material_colors: MaterialColors::default(),
            object_to_world: matrix,
            world_to_local: object_inverse(matrix),
            object_color,
            object_pass_index,
            object_random_number,
            is_holdout: node.is_holdout,
            shadow_terminator_normal_offset: node.shadow_terminator_normal_offset,
            shadow_terminator_geometry_offset: node.shadow_terminator_geometry_offset,
            shadow_terminator_shading_offset: node.shadow_terminator_shading_offset,
            view_layer_use_solid,
            view_layer_use_volumes,
        });
        if let Some(mesh) = mesh
            && !mesh.vertices.is_empty()
        {
            let mut min = DVec3::splat(f64::INFINITY);
            let mut max = DVec3::splat(f64::NEG_INFINITY);
            for vertex in &mesh.vertices {
                let point = matrix.transform_point3(vertex.co);
                min = min.min(point);
                max = max.max(point);
            }
            bounds = Some(match bounds {
                Some((current_min, current_max)) => (current_min.min(min), current_max.max(max)),
                None => (min, max),
            });

            let vertex_positions: std::collections::BTreeMap<u32, DVec3> = mesh
                .vertices
                .iter()
                .map(|vertex| (vertex.id, vertex.co))
                .collect();
            // A collection instance's joined mesh carries each member's Generated per vertex, in its own texture space.
            let generated_by_vertex: Option<std::collections::BTreeMap<u32, GeneratedVertex>> =
                match snapshot.generated_vertices.get(node_id) {
                    Some(table) => {
                        if table.len() != mesh.vertices.len() {
                            return Err(PotError::new(
                                ErrorCode::InternalError,
                                "collection instance Generated does not match its vertices",
                            ));
                        }
                        Some(
                            mesh.vertices
                                .iter()
                                .zip(table)
                                .map(|(vertex, value)| (vertex.id, *value))
                                .collect(),
                        )
                    }
                    None => None,
                };
            let mesh_triangles = mesh.tessellate().map_err(|error| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    format!("mesh triangulation failed: {error}"),
                    serde_json::json!({ "node_id": node_id }),
                )
            })?;
            let face_smooth_flags = smooth_face_flags(mesh)?;
            let face_normal_data = corner_normals(
                mesh,
                &vertex_positions,
                matrix,
                node_id,
                snapshot.adaptive_normals.get(node_id).map(Vec::as_slice),
                has_uv_map_tangent,
                source_mesh_attribute_precision,
                &mut warnings,
            );
            if !face_normal_data.precision_matches {
                return Err(unknown_mesh_attribute_precision_error(
                    &snapshot.scene_id,
                    node_id,
                ));
            }
            let mut mikk_tangents = if has_uv_map_tangent && !uv_store.layers.is_empty() {
                Some(mesh_mikk_tangents(
                    mesh,
                    &mesh_triangles,
                    &vertex_positions,
                    &uv_store,
                    face_normal_data.local.as_deref(),
                    matrix,
                    source_mesh_attribute_precision,
                )?)
            } else {
                None
            };
            if mikk_tangents
                .as_ref()
                .is_some_and(|tangents| !tangents.precision_matches)
            {
                return Err(unknown_mesh_attribute_precision_error(
                    &snapshot.scene_id,
                    node_id,
                ));
            }
            let mut mikk_tangents = mikk_tangents
                .take()
                .map(|tangents| tangents.by_triangle)
                .map(Vec::into_iter);
            let face_normals = face_normal_data.world;
            let mut triangle_offset = 0_usize;
            let mut corner_base = 0_usize;
            let mesh_edge_pairs: Vec<[u32; 2]> = if matches!(mode, Mode::Wire) {
                mesh.edges.iter().map(|edge| edge.vertices).collect()
            } else {
                Vec::new()
            };
            let mut wire_builder = WireEdgeBuilder::new(&mesh_edge_pairs);
            let vertex_slots: BTreeMap<u32, usize> =
                if carried_layers.is_empty() && !has_geometry_attributes {
                    BTreeMap::new()
                } else {
                    mesh.vertices
                        .iter()
                        .enumerate()
                        .map(|(slot, vertex)| (vertex.id, slot))
                        .collect()
                };
            for (face_index, face) in mesh.faces.iter().enumerate() {
                let material_slot = usize::try_from(face.material_index).ok();
                let binding_slot =
                    material_slot.map_or(0, |slot| slot.min(color_slot_count.saturating_sub(1)));
                let binding =
                    material_slot.and_then(|slot| snapshot.slot_binding(node_id, node, slot));
                let render_layer = binding.map_or(uv_render, |binding| {
                    binding.uv.render_index.map(usize::from)
                });
                let uv_available_mask = binding.map(|binding| binding.uv.available_mask);
                let face_normal_values = face_normals[face_index].as_ref();
                let face_corner_normals = face_normal_values.and_then(FaceNormalValues::as_corners);
                let face_smooth = face_smooth_flags[face_index];
                if matches!(mode, Mode::Wire) {
                    let positions: Option<Vec<DVec3>> = face
                        .vertices
                        .iter()
                        .map(|vertex_id| {
                            vertex_positions
                                .get(vertex_id)
                                .map(|local| matrix.transform_point3(*local))
                        })
                        .collect();
                    if let Some(positions) = &positions {
                        wire_builder.add_polygon(&face.vertices, positions, face_corner_normals);
                    }
                }
                // Corners are numbered across faces in order, as `color_layers::read` lays them
                // out.
                let face_corner_base = corner_base;
                corner_base += face.vertices.len();
                let face_reads_colors = material_reads_colors
                    .get(binding_slot)
                    .copied()
                    .unwrap_or(false);
                let face_requires_hit_shading = material_requires_hit_shading
                    .get(binding_slot)
                    .copied()
                    .unwrap_or(false);
                let element_index = u32::try_from(elements.len() + 1).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "too many elements for the ID buffer",
                    )
                })?;
                elements.push(ElementRecord {
                    index: element_index,
                    node_id: node_id.clone(),
                    domain: "face".to_owned(),
                    element_id: format!("f{}", face.id),
                    uv_render: render_layer,
                    uv_available_mask,
                });
                let material_id = material_override
                    .or_else(|| snapshot.slot_material(node_id, node, data_block, binding_slot));
                let slot_color_render = color_render_layers.get(binding_slot).copied().flatten();
                let slot_color_mask = if face_reads_colors {
                    color_available_masks.get(binding_slot).cloned().flatten()
                } else {
                    None
                };
                let material_index = if let Some(id) = material_id {
                    let key = (id.clone(), slot_color_render, slot_color_mask.clone());
                    if let Some(index) = material_indices.get(&key).copied() {
                        index
                    } else if let Some(material) = doc.materials.get(id) {
                        let index = materials.len();
                        materials.push(material.clone());
                        prepared.push(PreparedMaterial::new(material, &doc.node_groups));
                        material_indices.insert(key, index);
                        index
                    } else {
                        0
                    }
                } else {
                    0
                };
                {
                    let object = objects.last_mut().ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "object record was not stored")
                    })?;
                    object.material_colors.insert(
                        material_index,
                        material_id.filter(|_| material_index != 0),
                        materials
                            .get(material_index)
                            .and_then(|material| material.pass_index.map(f64::from)),
                        slot_color_render,
                        slot_color_mask,
                    );
                }
                let material = materials.get(material_index);
                let prepared_material = prepared.get(material_index);
                let base_color = material.map_or([0.6, 0.6, 0.6, 1.0], |value| value.base_color);
                let face_triangle_count = face.vertices.len().checked_sub(2).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "mesh face has fewer than three vertices",
                    )
                })?;
                let face_triangle_end = triangle_offset
                    .checked_add(face_triangle_count)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "triangle offset exceeds the supported index range",
                        )
                    })?;
                let face_triangles = mesh_triangles
                    .get(triangle_offset..face_triangle_end)
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "triangulated face ranges are inconsistent",
                        )
                    })?;
                for triangle in face_triangles {
                    let mut triangle_vertices = *triangle;
                    if mirrored {
                        triangle_vertices.swap(1, 2);
                    }
                    let mut positions = [DVec3::ZERO; 3];
                    let mut vertices = [GeneratedVertex::default(); 3];
                    let corners = triangle_corners(face, &triangle_vertices);
                    let layer_uvs = triangle_layer_uvs(&uv_store, face_index, corners);
                    for (index, vertex_id) in triangle_vertices.iter().enumerate() {
                        let local = vertex_positions.get(vertex_id).copied().ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "triangle references a missing vertex",
                            )
                        })?;
                        positions[index] = matrix.transform_point3(local);
                        vertices[index] = match &generated_by_vertex {
                            Some(table) => *table.get(vertex_id).ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "collection instance Generated is missing a vertex",
                                )
                            })?,
                            None => texture_space.generated_vertex(local),
                        };
                    }
                    let generated = GeneratedCorners {
                        corners: vertices.map(|vertex| vertex.generated),
                        per_local: vertices[0].per_local,
                    };
                    let uv = render_layer
                        .and_then(|index| layer_uvs.get(index).copied())
                        .unwrap_or([[0.0, 1.0]; 3]);
                    let layer_colors = triangle_layer_colors(
                        &carried_layers,
                        face_index,
                        face_corner_base,
                        corners,
                        triangle_vertices,
                        &vertex_slots,
                    );
                    let geometry_vertex_slots = if has_geometry_attributes {
                        [
                            *vertex_slots.get(&triangle_vertices[0]).ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::SceneInvalid,
                                    "triangle references a missing generic attribute point",
                                )
                            })?,
                            *vertex_slots.get(&triangle_vertices[1]).ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::SceneInvalid,
                                    "triangle references a missing generic attribute point",
                                )
                            })?,
                            *vertex_slots.get(&triangle_vertices[2]).ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::SceneInvalid,
                                    "triangle references a missing generic attribute point",
                                )
                            })?,
                        ]
                    } else {
                        [0; 3]
                    };
                    let mut layer_tangents = mikk_tangents
                        .as_mut()
                        .and_then(Iterator::next)
                        .unwrap_or_default();
                    if mirrored {
                        for tangents in &mut layer_tangents {
                            tangents.swap(1, 2);
                        }
                    }
                    let raw_normal =
                        (positions[1] - positions[0]).cross(positions[2] - positions[0]);
                    if raw_normal.length_squared() <= f64::EPSILON || !raw_normal.is_finite() {
                        continue;
                    }
                    let corner_normals =
                        face_normal_values.and_then(|values| values.triangle_values(corners));
                    let normal = raw_normal.normalize();
                    let center = (positions[0] + positions[1] + positions[2]) / 3.0;
                    let lit_per_pixel = !path_traced
                        && matches!(mode, Mode::Beauty)
                        && (face_reads_colors || face_requires_hit_shading);
                    // Beauty shades each hit at its own point with its normal, Solid from its view ray, and the path tracer shades its own hits,
                    // so their triangles keep the base colour, flat faces included (EEVEE decides shadows per pixel).
                    // Other passes take a flat face's colour from its material.
                    let color =
                        if corner_normals.is_some() || matches!(mode, Mode::Beauty | Mode::Solid) {
                            base_color
                        } else {
                            let params = shading_params(material, base_color, normal);
                            let closures = [ShaderClosure {
                                weight: 1.0,
                                params,
                            }];
                            shade(&SurfaceShadeContext {
                                doc,
                                snapshot,
                                receiver: Receiver {
                                    point: center,
                                    shadow_point: center,
                                    geometric: normal,
                                    view: normal,
                                },
                                normal,
                                closures: &closures,
                                mode,
                                shadow: None,
                                object_thickness: thickness,
                                world: None,
                                world_sun: None,
                                gbuffer: None,
                                eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
                                material_refraction: Some(false),
                            })?
                        };
                    let triangle = Triangle {
                        positions,
                        generated,
                        geometry_vertex_slots,
                        geometry_vertex_weights: [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        object_index,
                        element_index,
                        color,
                        alpha_mode: material.map_or(AlphaMode::Opaque, |material| {
                            match (mode, &material.viewport_display) {
                                // Workbench draws a viewport-displayed material through its transparent pass only
                                // below full alpha (`Material::is_transparent`).
                                (Mode::Solid, Some(viewport))
                                    if viewport.diffuse_color[3] < 1.0 =>
                                {
                                    AlphaMode::Blend
                                }
                                (Mode::Solid, Some(_)) => AlphaMode::Opaque,
                                _ => match material.alpha_mode.as_str() {
                                    "blend" => AlphaMode::Blend,
                                    "clip" => AlphaMode::Clip,
                                    // EEVEE DITHERED graph materials use the source per-sample hashed depth-prepass coverage.
                                    "opaque"
                                        if matches!(mode, Mode::Beauty)
                                            && material.node_tree.is_some() =>
                                    {
                                        AlphaMode::Dithered
                                    }
                                    _ => AlphaMode::Opaque,
                                },
                            }
                        }),
                        alpha_threshold: material.map_or(0.5, |material| material.alpha_threshold),
                        // Only Beauty culls by the material's backface setting (EEVEE). Solid, Wire and the
                        // id, normal and depth passes draw the Workbench geometry with back faces, as Blender does.
                        // A face without a material shades with Blender's Default Surface, which is not culled.
                        double_sided: !matches!(mode, Mode::Beauty)
                            || material.is_none_or(|value| value.double_sided),
                        material_index,
                        uv,
                        layer_uvs,
                        layer_tangents,
                        normals: corner_normals,
                        smooth_shading: face_smooth,
                        layer_colors,
                        lit_per_pixel,
                        ray_mask: ray_visibility.mask(),
                    };
                    if let (Some(material), Some(prepared_material)) = (
                        material.filter(|material| {
                            material.displacement_method == "displacement"
                                && material.node_tree.is_some()
                        }),
                        prepared_material,
                    ) {
                        let layers = HitUvLayers {
                            names: &uv_names,
                            corners: &triangle.layer_uvs,
                            positions: triangle.positions,
                            barycentric: [1.0, 0.0, 0.0],
                            render: render_layer,
                            available_mask: uv_available_mask,
                        };
                        let object = objects.last().ok_or_else(|| {
                            PotError::new(ErrorCode::InternalError, "object record was not stored")
                        })?;
                        triangles.extend(subdivide_displaced_triangle(
                            doc,
                            material,
                            prepared_material,
                            matrix,
                            path_traced,
                            &triangle,
                            images,
                            object,
                            layers,
                            &shader_warnings,
                            &mut evaluator_workspace,
                            math_semantics,
                        )?);
                    } else {
                        triangles.push(triangle);
                    }
                }
                triangle_offset = face_triangle_end;
            }
            if matches!(mode, Mode::Wire) {
                // Blender's overlay draws the face edges (mesh-edge order), then the loose edges (mesh order),
                // then strokes and points. Loose edges have no facing and never hide.
                let loose: Vec<WireEdge> = mesh
                    .edges
                    .iter()
                    .filter(|edge| !wire_builder.contains(edge.vertices[0], edge.vertices[1]))
                    .filter_map(|edge| {
                        let local_a = vertex_positions.get(&edge.vertices[0])?;
                        let local_b = vertex_positions.get(&edge.vertices[1])?;
                        Some(WireEdge {
                            start: matrix.transform_point3(*local_a),
                            end: matrix.transform_point3(*local_b),
                            normals: None,
                            colour: [0.0; 3],
                            wd: 0.0,
                        })
                    })
                    .collect();
                // The subdivision's optimal-display mask hides internal edges of subdivided faces (`wd = 1`).
                let optimal = mesh
                    .attributes
                    .get(crate::geom::modifiers::OPTIMAL_DISPLAY_EDGES)
                    .and_then(|attribute| attribute.get("values"))
                    .and_then(serde_json::Value::as_object);
                let hidden: Vec<bool> = mesh
                    .edges
                    .iter()
                    .map(|edge| {
                        optimal.is_some_and(|values| {
                            values
                                .get(&format!("e{}", edge.id))
                                .and_then(serde_json::Value::as_bool)
                                == Some(false)
                        })
                    })
                    .collect();
                wire_edges
                    .extend(wire_builder.finish(|slot| hidden.get(slot).copied().unwrap_or(false)));
                wire_edges.extend(loose);
            }
            let used_vertices: HashSet<u32> = mesh
                .faces
                .iter()
                .flat_map(|face| face.vertices.iter().copied())
                .collect();
            let pointcloud = node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .is_some_and(|data| {
                    matches!(data.data_type.as_str(), "pointcloud" | "point_cloud")
                });
            let default_point_color = snapshot
                .slot_material(node_id, node, data_block, 0)
                .and_then(|material_id| doc.materials.get(material_id))
                .map_or([0.6, 0.6, 0.6, 1.0], |material| material.base_color);
            let world_scale = [
                matrix.transform_vector3(DVec3::X).length(),
                matrix.transform_vector3(DVec3::Y).length(),
                matrix.transform_vector3(DVec3::Z).length(),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            let point_colors = color_store
                .render_index()
                .map(|index| &color_store.layers[index])
                .filter(|layer| layer.domain == ColorDomain::Point)
                .map(|layer| layer.values.as_slice());
            for (slot, vertex) in mesh.vertices.iter().enumerate().filter(|(_, vertex)| {
                !used_vertices.contains(&vertex.id) && (pointcloud || mesh.edges.is_empty())
            }) {
                let position = matrix.transform_point3(vertex.co);
                let element_index = u32::try_from(elements.len() + 1).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "too many elements for the ID buffer",
                    )
                })?;
                elements.push(ElementRecord {
                    index: element_index,
                    node_id: node_id.clone(),
                    domain: if pointcloud {
                        "point".to_owned()
                    } else {
                        "vertex".to_owned()
                    },
                    element_id: format!("{}{}", if pointcloud { "p" } else { "v" }, vertex.id),
                    uv_render: None,
                    uv_available_mask: None,
                });
                let radius = read_vertex_number(mesh, "point_radius", vertex.id)
                    .unwrap_or(0.02)
                    .max(0.0)
                    * world_scale;
                let color = point_colors
                    .and_then(|colors| colors.get(slot))
                    .copied()
                    .unwrap_or(default_point_color);
                let extent = DVec3::splat(radius);
                bounds = Some(match bounds {
                    Some((current_min, current_max)) => (
                        current_min.min(position - extent),
                        current_max.max(position + extent),
                    ),
                    None => (position - extent, position + extent),
                });
                if matches!(mode, Mode::Wire) {
                    // Loose vertices and point-cloud points are squares in the overlay (`wire_points`); a point
                    // cloud's points also have spheres, which the object outline sees.
                    wire_points.push(WirePoint {
                        position,
                        sphere: pointcloud.then_some(PointSphere {
                            object_index,
                            radius,
                        }),
                    });
                }
                lines.push(Line {
                    start: position,
                    end: position,
                    radius_start: radius,
                    radius_end: radius,
                    color,
                    object_index,
                    element_index,
                    ray_mask: ray_visibility.mask(),
                });
            }
        }
        if let Some(strokes) = strokes {
            for (stroke_index, stroke) in strokes.iter().enumerate() {
                if stroke.radius.len() != stroke.points_world.len()
                    || stroke.color.iter().any(|channel| !channel.is_finite())
                {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "evaluated stroke has inconsistent point, radius, or color data",
                    ));
                }
                if stroke.points_world.is_empty() {
                    continue;
                }
                let stroke_number = u32::try_from(stroke_index).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "too many evaluated strokes on one object",
                    )
                })?;
                for (point_index, coordinates) in stroke.points_world.iter().copied().enumerate() {
                    let position = DVec3::from_array(coordinates);
                    let radius = stroke.radius[point_index];
                    if !position.is_finite() || !radius.is_finite() || radius < 0.0 {
                        return Err(PotError::new(
                            ErrorCode::EvaluationFailed,
                            "evaluated stroke point has invalid position or radius",
                        ));
                    }
                    let element_index = u32::try_from(elements.len() + 1).map_err(|_| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "too many elements for the ID buffer",
                        )
                    })?;
                    let point_number = u32::try_from(point_index).map_err(|_| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "too many points in one evaluated stroke",
                        )
                    })?;
                    elements.push(ElementRecord {
                        index: element_index,
                        node_id: node_id.clone(),
                        domain: "point".to_owned(),
                        element_id: format!("p{stroke_number}_{point_number}"),
                        uv_render: None,
                        uv_available_mask: None,
                    });
                    let extent = DVec3::splat(radius);
                    bounds = Some(match bounds {
                        Some((current_min, current_max)) => (
                            current_min.min(position - extent),
                            current_max.max(position + extent),
                        ),
                        None => (position - extent, position + extent),
                    });
                    lines.push(Line {
                        start: position,
                        end: position,
                        radius_start: radius,
                        radius_end: radius,
                        color: stroke.color,
                        object_index,
                        element_index,
                        ray_mask: ray_visibility.mask(),
                    });
                }
                let stroke_element_index = u32::try_from(elements.len() + 1).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "too many elements for the ID buffer",
                    )
                })?;
                elements.push(ElementRecord {
                    index: stroke_element_index,
                    node_id: node_id.clone(),
                    domain: "stroke".to_owned(),
                    element_id: format!("s{stroke_index}"),
                    uv_render: None,
                    uv_available_mask: None,
                });
                let mut segment_count = stroke.points_world.len().saturating_sub(1);
                if stroke.cyclic && stroke.points_world.len() > 2 {
                    segment_count = segment_count.saturating_add(1);
                }
                for point_index in 0..segment_count {
                    let next_index = if point_index + 1 == stroke.points_world.len() {
                        0
                    } else {
                        point_index + 1
                    };
                    if matches!(mode, Mode::Wire) {
                        // Grease Pencil strokes are drawn as their polyline (a zero-length segment draws nothing).
                        let start = DVec3::from_array(stroke.points_world[point_index]);
                        let end = DVec3::from_array(stroke.points_world[next_index]);
                        if start != end {
                            wire_edges.push(WireEdge {
                                start,
                                end,
                                normals: None,
                                colour: [0.0; 3],
                                wd: 0.0,
                            });
                        }
                    }
                    lines.push(Line {
                        start: DVec3::from_array(stroke.points_world[point_index]),
                        end: DVec3::from_array(stroke.points_world[next_index]),
                        radius_start: stroke.radius[point_index],
                        radius_end: stroke.radius[next_index],
                        color: stroke.color,
                        object_index,
                        element_index: stroke_element_index,
                        ray_mask: ray_visibility.mask(),
                    });
                }
            }
        }
        if let Some(hair) = hair {
            let radius_scale = [
                matrix.transform_vector3(DVec3::X).length(),
                matrix.transform_vector3(DVec3::Y).length(),
                matrix.transform_vector3(DVec3::Z).length(),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            let material_index = if let Some(id) = material_id {
                let key = (id.clone(), None, None);
                if let Some(index) = material_indices.get(&key).copied() {
                    index
                } else if let Some(material) = doc.materials.get(id) {
                    let index = materials.len();
                    materials.push(material.clone());
                    prepared.push(PreparedMaterial::new(material, &doc.node_groups));
                    material_indices.insert(key, index);
                    index
                } else {
                    0
                }
            } else {
                0
            };
            if let Some(object) = objects.last_mut() {
                object.material_colors.insert(
                    material_index,
                    material_id.filter(|_| material_index != 0),
                    materials
                        .get(material_index)
                        .and_then(|material| material.pass_index.map(f64::from)),
                    None,
                    None,
                );
            }
            let color = materials
                .get(material_index)
                .map_or([0.6, 0.6, 0.6, 1.0], |material| material.base_color);
            let legacy_shared_hair_shadow = path_curves_enabled
                && blender_imported
                && scene.cycles_hair_shadow_groups.is_none()
                && node.data.is_some()
                && !cycles_hair_geometry_key_is_object(node, data_block);
            let point_count = hair
                .curves
                .iter()
                .try_fold(0usize, |count, curve| count.checked_add(curve.points.len()));
            let point_count = point_count.ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "hair point count exceeds platform limits",
                )
            })?;
            let curve_types = hair.attributes.get("curve_type");
            if curve_types.is_some_and(|attribute| {
                attribute.domain != "CURVE"
                    || attribute.data_type != "INT8"
                    || attribute.values.len() != hair.curves.len()
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair curve_type attribute has an invalid domain, type, or length",
                ));
            }
            let radius_values = hair.attributes.get("radius");
            if radius_values.is_some_and(|attribute| {
                attribute.domain != "POINT"
                    || attribute.data_type != "FLOAT"
                    || attribute.values.len() != point_count
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair radius attribute has an invalid domain, type, or length",
                ));
            }
            let cyclic_values = hair.attributes.get("cyclic");
            if cyclic_values.is_some_and(|attribute| {
                attribute.domain != "CURVE"
                    || attribute.data_type != "BOOLEAN"
                    || attribute.values.len() != hair.curves.len()
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair cyclic attribute has an invalid domain, type, or length",
                ));
            }
            let normal_modes = hair.attributes.get("normal_mode");
            if normal_modes.is_some_and(|attribute| {
                attribute.domain != "CURVE"
                    || attribute.data_type != "INT8"
                    || attribute.values.len() != hair.curves.len()
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair normal_mode attribute has an invalid domain, type, or length",
                ));
            }
            let resolutions = hair.attributes.get("resolution");
            if resolutions.is_some_and(|attribute| {
                attribute.domain != "CURVE"
                    || attribute.data_type != "INT"
                    || attribute.values.len() != hair.curves.len()
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair resolution attribute has an invalid domain, type, or length",
                ));
            }
            let tilt_values = hair.attributes.get("tilt");
            if tilt_values.is_some_and(|attribute| {
                attribute.domain != "POINT"
                    || attribute.data_type != "FLOAT"
                    || attribute.values.len() != point_count
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair tilt attribute has an invalid domain, type, or length",
                ));
            }
            let custom_normal_values = hair.attributes.get("custom_normal");
            if custom_normal_values.is_some_and(|attribute| {
                attribute.domain != "POINT"
                    || attribute.data_type != "FLOAT_VECTOR"
                    || attribute.values.len() != point_count
            }) {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "hair custom_normal attribute has an invalid domain, type, or length",
                ));
            }
            let hair_path_matrix = if hair_instance_only {
                let member = hair_shadow_group
                    .and_then(|group| {
                        group.members.iter().find(|member| {
                            member.instance.real_object_node_id == *node_id
                                && !member.instance.use_particle_hair
                        })
                    })
                    .ok_or_else(|| {
                        PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            "instanced Cycles Hair transform was not captured",
                            serde_json::json!({
                                "feature_id":"render.hair.shadow_cache_provenance",
                                "node_id":node_id.as_str(),
                            }),
                        )
                    })?;
                cycles_hair_object_matrix(member.object_to_world)
            } else {
                matrix
            };
            let mut prepared_cycles_curves = if path_curves_enabled {
                Some(
                    super::hair_geometry::prepare_cycles_hair_curve_geometry(
                        hair,
                        hair_path_matrix,
                        curve_render_mode,
                        curve_ribbon_subdivisions,
                    )
                    .map_err(|error| hair_curve_geometry_error(node_id, &error))?,
                )
            } else {
                None
            };
            let path_render_mode = prepared_cycles_curves
                .as_ref()
                .and_then(|curves| curves.first())
                .map(|prepared| prepared.curve.render_mode);
            let has_renderable_curve = hair.curves.iter().any(|curve| curve.points.len() > 1);
            let needs_curve_settings = has_renderable_curve
                && path_render_mode
                    .is_some_and(|render_mode| render_mode != HairCurveRenderMode::ThickLinear);
            let missing_shape = scene.render.cycles_curves_shape.is_none();
            let missing_ribbon_subdivisions = has_renderable_curve
                && path_render_mode
                    .is_some_and(|render_mode| render_mode == HairCurveRenderMode::Ribbon)
                && scene.render.cycles_curves_subdivisions.is_none();
            if blender_imported
                && needs_curve_settings
                && (missing_shape || missing_ribbon_subdivisions)
            {
                let mut required_settings = Vec::new();
                if missing_shape {
                    required_settings.push("cycles_curves.shape");
                }
                if missing_ribbon_subdivisions {
                    required_settings.push("cycles_curves.subdivisions");
                }
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "Cycles Hair Curves settings affecting the render shape were not captured; reimport from Blender",
                    serde_json::json!({
                        "feature_id":"render.cycles_curves.settings",
                        "node_id":node_id.as_str(),
                        "required_settings":required_settings,
                    }),
                ));
            }
            if let Some(group_index) = hair_shadow_group_index
                && !hair_shadow_caches.contains_key(&group_index)
            {
                let group = hair_shadow_group.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "Cycles Hair shadow group disappeared during extraction",
                    )
                })?;
                let shadow_cache = bake_cycles_hair_shadow_group(
                    doc,
                    snapshot,
                    group,
                    hair,
                    hair_shadow_curve_indices,
                    curve_render_mode,
                    curve_ribbon_subdivisions,
                    &materials[0],
                    images,
                    &shader_warnings,
                    &mut evaluator_workspace,
                )?;
                hair_shadow_caches.insert(group_index, shadow_cache);
            }
            let cached_hair_shadow =
                hair_shadow_group_index.and_then(|index| hair_shadow_caches.get(&index));
            let local_hair_bake_object = if path_curves_enabled && cached_hair_shadow.is_none() {
                let object = objects.last().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "hair curve has no owning object record",
                    )
                })?;
                Some(HairCurveBakeObject::from_record(object))
            } else {
                None
            };
            let mut path_members = Vec::new();
            if path_curves_enabled {
                if let Some(group) = hair_shadow_group {
                    let group_index = hair_shadow_group_index.ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "Cycles Hair path member group index is missing",
                        )
                    })?;
                    let members_for_node = hair_member_indices
                        .get(&group_index)
                        .and_then(|members| members.get(node_id))
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "Cycles Hair path member projection is missing",
                            )
                        })?;
                    path_members
                        .try_reserve_exact(members_for_node.len())
                        .map_err(|_| {
                            PotError::new(
                                ErrorCode::LimitExceeded,
                                "Cycles Hair instance allocation failed",
                            )
                        })?;
                    let mut use_prepared_for_first_member = hair_instance_only;
                    for &member_index in members_for_node {
                        let member = group.members.get(member_index).ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "Cycles Hair member index is out of range",
                            )
                        })?;
                        if material_override.is_none()
                            && evaluated_hair.is_some()
                            && member
                                .slot_0_material_id
                                .as_ref()
                                .is_some_and(|member_id| Some(member_id) != material_id)
                        {
                            return Err(PotError::with_details(
                                ErrorCode::UnsupportedFeature,
                                "Cycles Hair member material differs from its cache group",
                                serde_json::json!({
                                    "feature_id":"render.hair.shadow_cache_provenance",
                                    "node_id":node_id.as_str(),
                                }),
                            ));
                        }
                        let member_object = doc
                            .nodes
                            .get(&member.instance.real_object_node_id)
                            .ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "Cycles Hair member object record is missing",
                                )
                            })?;
                        let member_object_index = append_hair_instance_object(
                            &mut objects,
                            object_index,
                            member,
                            material_index,
                            &materials,
                            member_object.is_holdout,
                            member_object.shadow_terminator_normal_offset,
                            member_object.shadow_terminator_geometry_offset,
                            member_object.shadow_terminator_shading_offset,
                            hair_instance_only,
                        )?;
                        let use_prepared_curves = if hair_instance_only {
                            let use_prepared = use_prepared_for_first_member;
                            use_prepared_for_first_member = false;
                            use_prepared
                        } else {
                            !member.instance.is_instance
                        };
                        let member_curves = if use_prepared_curves {
                            prepared_cycles_curves.take().ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "Cycles Hair member geometry was not prepared for a path hit",
                                )
                            })?
                        } else {
                            super::hair_geometry::prepare_cycles_hair_curve_geometry(
                                hair,
                                cycles_hair_object_matrix(member.object_to_world),
                                curve_render_mode,
                                curve_ribbon_subdivisions,
                            )
                            .map_err(|error| hair_curve_geometry_error(node_id, &error))?
                        };
                        path_members.push(HairPathMember {
                            object_index: member_object_index,
                            material_index,
                            ray_mask: if member.instance.is_instance {
                                member.ray_visibility.mask()
                            } else {
                                ray_visibility.mask()
                            },
                            prepared_curves: member_curves.into_iter().map(Some).collect(),
                        });
                    }
                    if path_members.is_empty() {
                        return Err(PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            "Cycles Hair group has no renderable curve member for this object",
                            serde_json::json!({
                                "feature_id":"render.hair.shadow_cache_provenance",
                                "node_id":node_id.as_str(),
                            }),
                        ));
                    }
                } else {
                    let member_curves = prepared_cycles_curves.take().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "Cycles hair geometry was not prepared for a path hit",
                        )
                    })?;
                    path_members.push(HairPathMember {
                        object_index,
                        material_index,
                        ray_mask: ray_visibility.mask(),
                        prepared_curves: member_curves.into_iter().map(Some).collect(),
                    });
                }
            }
            let curve_count = hair_shadow_curve_indices.map_or(hair.curves.len(), <[u32]>::len);
            let mut legacy_shadow_values = if legacy_shared_hair_shadow {
                Some(Vec::new())
            } else {
                None
            };
            if let Some(values) = &mut legacy_shadow_values {
                values.try_reserve_exact(curve_count).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "legacy hair shadow cache allocation failed",
                    )
                })?;
            }
            let identity_curve_order = curve_count == hair.curves.len()
                && hair_shadow_curve_indices.is_none_or(|indices| {
                    indices.iter().enumerate().all(|(index, source_index)| {
                        usize::try_from(*source_index).ok() == Some(index)
                    })
                });
            if let Some(indices) = hair_shadow_curve_indices {
                let mut seen = BTreeSet::new();
                if indices
                    .iter()
                    .any(|source_index| !seen.insert(*source_index))
                {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        "Cycles Hair evaluated curves duplicate source indices without evaluated geometry",
                        serde_json::json!({
                            "feature_id":"render.hair.shadow_cache_provenance",
                            "node_id":node_id.as_str(),
                        }),
                    ));
                }
            }
            let source_point_offsets = if identity_curve_order {
                None
            } else {
                let mut offsets = Vec::new();
                offsets.try_reserve_exact(hair.curves.len()).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "hair source point-offset allocation failed",
                    )
                })?;
                let mut offset = 0usize;
                for source_curve in &hair.curves {
                    offsets.push(offset);
                    offset = offset
                        .checked_add(source_curve.points.len())
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::LimitExceeded,
                                "hair point offset exceeds platform limits",
                            )
                        })?;
                }
                Some(offsets)
            };
            let mut identity_point_offset = 0usize;
            for evaluated_curve_index in 0..curve_count {
                let source_curve_index = if let Some(indices) = hair_shadow_curve_indices {
                    usize::try_from(indices[evaluated_curve_index]).map_err(|_| {
                        PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            "Cycles Hair source curve index cannot be represented",
                            serde_json::json!({
                                "feature_id":"render.hair.shadow_cache_provenance",
                                "node_id":node_id.as_str(),
                                "evaluated_curve_index":evaluated_curve_index,
                            }),
                        )
                    })?
                } else {
                    evaluated_curve_index
                };
                let curve = hair.curves.get(source_curve_index).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        "Cycles Hair source curve mapping references unavailable geometry",
                        serde_json::json!({
                            "feature_id":"render.hair.shadow_cache_provenance",
                            "node_id":node_id.as_str(),
                            "source_curve_index":source_curve_index,
                        }),
                    )
                })?;
                let point_offset = if let Some(offsets) = &source_point_offsets {
                    *offsets.get(source_curve_index).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::InternalError,
                            "hair source point offset is missing",
                        )
                    })?
                } else {
                    identity_point_offset
                };
                if !curve.radius.is_finite()
                    || curve.radius < 0.0
                    || curve
                        .points
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "hair curve contains invalid position or radius data",
                    ));
                }
                if curve.points.is_empty() {
                    if let Some(values) = &mut legacy_shadow_values {
                        values.push(Arc::new(
                            super::hair_geometry::HairCurveShadowTransparency::new(None, false),
                        ));
                    }
                    continue;
                }
                let _curve_type = curve_types
                    .and_then(|attribute| attribute.values.get(source_curve_index))
                    .map_or(Some(0), serde_json::Value::as_i64)
                    .and_then(|curve_type| u8::try_from(curve_type).ok())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "hair curve_type is invalid")
                    })?;
                let curve_random =
                    Some(cycles_curve_random(evaluated_curve_index).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "hair curve index exceeds Cycles' random attribute range",
                        )
                    })?);
                let _cyclic = cyclic_values
                    .and_then(|attribute| attribute.values.get(source_curve_index))
                    .map_or(Some(false), serde_json::Value::as_bool)
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "hair cyclic value is invalid")
                    })?;
                let _normal_mode = normal_modes
                    .and_then(|attribute| attribute.values.get(source_curve_index))
                    .map_or(Some(0), serde_json::Value::as_i64)
                    .and_then(|value| u8::try_from(value).ok())
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "hair normal_mode is invalid")
                    })?;
                let _resolution = resolutions
                    .and_then(|attribute| attribute.values.get(source_curve_index))
                    .map_or(Some(12usize), |value| {
                        value.as_i64().and_then(|value| usize::try_from(value).ok())
                    })
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::SceneInvalid, "hair resolution is invalid")
                    })?;
                if radius_values.is_some_and(|attribute| {
                    attribute.values[point_offset..point_offset + curve.points.len()]
                        .iter()
                        .any(|value| {
                            value
                                .as_f64()
                                .is_none_or(|radius| !radius.is_finite() || radius < 0.0)
                        })
                }) {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "hair curve contains invalid radius data",
                    ));
                }
                let element_index = if hair_instance_only {
                    None
                } else {
                    let index = u32::try_from(elements.len() + 1).map_err(|_| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "too many elements for the ID buffer",
                        )
                    })?;
                    elements.push(ElementRecord {
                        index,
                        node_id: node_id.clone(),
                        domain: "hair_curve".to_owned(),
                        element_id: format!("h{evaluated_curve_index}"),
                        uv_render: None,
                        uv_available_mask: None,
                    });
                    Some(index)
                };
                let radius = curve.radius * radius_scale;
                if !hair_instance_only {
                    for point in &curve.points {
                        let position = matrix.transform_point3(DVec3::from_array(*point));
                        let extent = DVec3::splat(radius);
                        bounds = Some(match bounds {
                            Some((current_min, current_max)) => (
                                current_min.min(position - extent),
                                current_max.max(position + extent),
                            ),
                            None => (position - extent, position + extent),
                        });
                    }
                    let segment_count = curve.points.len().saturating_sub(1).max(1);
                    for segment in 0..segment_count {
                        let start =
                            matrix.transform_point3(DVec3::from_array(curve.points[segment]));
                        let end = if segment + 1 < curve.points.len() {
                            matrix.transform_point3(DVec3::from_array(curve.points[segment + 1]))
                        } else {
                            start
                        };
                        lines.push(Line {
                            start,
                            end,
                            radius_start: radius,
                            radius_end: radius,
                            color,
                            object_index,
                            element_index: element_index.ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "hair line element is missing",
                                )
                            })?,
                            ray_mask: ray_visibility.mask(),
                        });
                    }
                }
                if path_curves_enabled && curve.points.len() > 1 {
                    for path_member in &mut path_members {
                        let prepared_curve_data = path_member
                            .prepared_curves
                            .get_mut(source_curve_index)
                            .and_then(Option::take)
                            .ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "Cycles hair member geometry was not prepared for a path hit",
                                )
                            })?;
                        let shadow_transparency = if let Some(cache) = cached_hair_shadow {
                            cache.get(evaluated_curve_index).cloned().ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::InternalError,
                                    "Cycles Hair shadow cache is missing an evaluated curve",
                                )
                            })?
                        } else {
                            let hair_bake_object =
                                local_hair_bake_object.as_ref().ok_or_else(|| {
                                    PotError::new(
                                        ErrorCode::InternalError,
                                        "hair curve bake context was not prepared",
                                    )
                                })?;
                            Arc::new(bake_cycles_hair_shadow_transparency(
                                HairCurveBakeContext {
                                    doc,
                                    object: hair_bake_object,
                                    material: materials
                                        .get(path_member.material_index)
                                        .ok_or_else(|| {
                                            PotError::new(
                                                ErrorCode::InternalError,
                                                "hair curve material was not prepared",
                                            )
                                        })?,
                                    prepared: prepared.get(path_member.material_index).ok_or_else(
                                        || {
                                            PotError::new(
                                                ErrorCode::InternalError,
                                                "hair curve prepared material was not prepared",
                                            )
                                        },
                                    )?,
                                    images,
                                    warnings: &shader_warnings,
                                },
                                &prepared_curve_data,
                                curve.points.len(),
                                curve_random,
                                &mut evaluator_workspace,
                            )?)
                        };
                        let curve_geometry = prepared_curve_data.curve;
                        let point_normals = prepared_curve_data.point_normals;
                        let (curve_min, curve_max) =
                            curve_geometry.world_bounds().ok_or_else(|| {
                                PotError::new(
                                    ErrorCode::EvaluationFailed,
                                    "Blender hair curve bounds evaluation failed",
                                )
                            })?;
                        bounds = Some(match bounds {
                            Some((current_min, current_max)) => {
                                (current_min.min(curve_min), current_max.max(curve_max))
                            }
                            None => (curve_min, curve_max),
                        });
                        let path_element_index =
                            if !hair_instance_only && path_member.object_index == object_index {
                                element_index.ok_or_else(|| {
                                    PotError::new(
                                        ErrorCode::InternalError,
                                        "hair base element is missing",
                                    )
                                })?
                            } else {
                                let index = u32::try_from(elements.len() + 1).map_err(|_| {
                                    PotError::new(
                                        ErrorCode::LimitExceeded,
                                        "too many elements for the ID buffer",
                                    )
                                })?;
                                elements.push(ElementRecord {
                                    index,
                                    node_id: node_id.clone(),
                                    domain: "hair_curve".to_owned(),
                                    element_id: format!("h{evaluated_curve_index}"),
                                    uv_render: None,
                                    uv_available_mask: None,
                                });
                                index
                            };
                        if let Some(values) = &mut legacy_shadow_values {
                            values.push(Arc::clone(&shadow_transparency));
                        }
                        hair_curve_instances.push(HairCurveInstance {
                            curve: curve_geometry,
                            point_normals,
                            shadow_transparency,
                            material_index: path_member.material_index,
                            object_index: path_member.object_index,
                            element_index: path_element_index,
                            curve_random,
                            ray_mask: path_member.ray_mask,
                        });
                    }
                } else if path_curves_enabled && let Some(values) = &mut legacy_shadow_values {
                    values.push(Arc::new(
                        super::hair_geometry::HairCurveShadowTransparency::new(None, false),
                    ));
                }
                if identity_curve_order {
                    identity_point_offset = identity_point_offset
                        .checked_add(curve.points.len())
                        .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "hair point offset exceeds platform limits",
                        )
                    })?;
                }
            }
            if let Some(values) = legacy_shadow_values {
                let data_id = node.data.as_ref().ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "shared legacy hair cache has no source data ID",
                    )
                })?;
                if let Some((first_node_id, first_values)) = legacy_hair_shadow_caches.get(data_id)
                {
                    if first_values != &values {
                        return Err(PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            "shared Blender Hair shadow values differ but their Cycles representative was not captured; reimport from Blender",
                            serde_json::json!({
                                "feature_id":"render.hair.shadow_cache_provenance",
                                "data_id":data_id.as_str(),
                                "first_node_id":first_node_id.as_str(),
                                "node_id":node_id.as_str(),
                            }),
                        ));
                    }
                } else {
                    legacy_hair_shadow_caches.insert(data_id.clone(), (node_id.clone(), values));
                }
            }
        }
    }
    let hair_curve_bvh =
        HairBvh::build_splines(hair_curve_instances.iter().map(|curve| &curve.curve));
    let mut geometry = Geometry {
        hair_curve_instances,
        hair_curve_bvh,
        triangles,
        lines,
        wire_edges,
        wire_points,
        materials,
        prepared,
        objects,
        elements,
        bounds,
        warnings,
        shader_warnings,
        shadows: None,
    };
    if !path_traced && matches!(mode, Mode::Beauty) && !geometry.triangles.is_empty() {
        // Per material: whether its surfaces cast transparent shadows, which EEVEE decides once at compile time.
        let caster_materials: Vec<bool> = geometry
            .materials
            .iter()
            .zip(&geometry.prepared)
            .map(|(material, prepared)| casts_transparent_shadow(material, prepared))
            .collect();
        let shadow_maps = build_point_shadow_maps(
            doc,
            snapshot,
            &geometry.triangles,
            &geometry.objects,
            &caster_materials,
        )?;
        let bvh = Bvh::build(&geometry.triangles)?;
        let transparent_casters = geometry.triangles.iter().any(|triangle| {
            caster_materials
                .get(triangle.material_index)
                .copied()
                .unwrap_or(false)
        });
        geometry.shadows = Some(RealtimeShadows {
            bvh,
            point_maps: shadow_maps,
            transparent_casters,
            has_volume_only: geometry
                .prepared
                .iter()
                .any(PreparedMaterial::is_volume_only),
        });
    }
    Ok(geometry)
}

fn build_point_shadow_maps(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    triangles: &[Triangle],
    objects: &[ObjectRecord],
    caster_materials: &[bool],
) -> Result<BTreeMap<Id, PointShadowMap>> {
    // Transparent and view-layer-hidden surfaces are handled by shadow-ray traversal, not the opaque depth map.
    let opaque: Vec<&Triangle> = triangles
        .iter()
        .filter(|triangle| {
            object_for_index(objects, triangle.object_index)
                .is_none_or(|object| object.view_layer_use_solid)
                && !caster_materials
                    .get(triangle.material_index)
                    .copied()
                    .unwrap_or(false)
        })
        .collect();
    let mut maps = BTreeMap::new();
    for (id, node) in &doc.nodes {
        let Some(evaluated) = snapshot.nodes.get(id) else {
            continue;
        };
        let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
            node.data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|block| block.light.as_ref())
        }) else {
            continue;
        };
        if light.use_shadow == Some(false)
            || !matches!(
                light.light_type,
                crate::model::LightType::Point | crate::model::LightType::Spot
            )
        {
            continue;
        }
        let position =
            DMat4::from_cols_array(&evaluated.world_matrix).transform_point3(DVec3::ZERO);
        maps.insert(id.clone(), PointShadowMap::build(position, &opaque)?);
    }
    Ok(maps)
}

impl PointShadowMap {
    fn build(position: DVec3, triangles: &[&Triangle]) -> Result<Self> {
        let bases = [
            (DVec3::X, DVec3::Z),
            (-DVec3::X, DVec3::Z),
            (DVec3::Y, DVec3::Z),
            (-DVec3::Y, DVec3::Z),
            (DVec3::Z, DVec3::Y),
            (-DVec3::Z, DVec3::Y),
        ];
        let mut faces = Vec::new();
        faces.try_reserve_exact(bases.len()).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "point shadow-map faces exceed available memory",
            )
        })?;
        for (forward, up_hint) in bases {
            let right = forward.cross(up_hint).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            let mut depth = Vec::new();
            depth
                .try_reserve_exact(SHADOW_MAP_SIZE_USIZE * SHADOW_MAP_SIZE_USIZE)
                .map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "point shadow-map depth buffer exceeds available memory",
                    )
                })?;
            depth.resize(SHADOW_MAP_SIZE_USIZE * SHADOW_MAP_SIZE_USIZE, f64::INFINITY);
            faces.push(ShadowFace {
                forward,
                right,
                up,
                depth,
            });
        }
        let casters = triangles
            .iter()
            .filter(|triangle| triangle.ray_mask & RayKind::Shadow.bit() != 0);
        for triangle in casters {
            for face in &mut faces {
                rasterize_shadow_triangle(face, position, triangle);
            }
        }
        Ok(Self { position, faces })
    }

    fn visibility(&self, point: DVec3) -> f64 {
        const PCF_OFFSETS: [[i32; 2]; 9] = [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ];
        let relative = point - self.position;
        let absolute = relative.abs();
        let face_index = if absolute.x >= absolute.y && absolute.x >= absolute.z {
            usize::from(relative.x < 0.0)
        } else if absolute.y >= absolute.z {
            2 + usize::from(relative.y < 0.0)
        } else if relative.z >= 0.0 {
            4
        } else {
            5
        };
        let Some(face) = self.faces.get(face_index) else {
            return 1.0;
        };
        let depth = relative.dot(face.forward);
        if depth <= f64::EPSILON {
            return 1.0;
        }
        let reciprocal_depth = depth.recip();
        let x = relative.dot(face.right) * reciprocal_depth;
        let y = relative.dot(face.up) * reciprocal_depth;
        let pixel_x = (x * 0.5 + 0.5) * f64::from(SHADOW_MAP_SIZE) - 0.5;
        let pixel_y = (0.5 - y * 0.5) * f64::from(SHADOW_MAP_SIZE) - 0.5;
        let center_x = pixel_x.round() as i32;
        let center_y = pixel_y.round() as i32;
        // One texel's world size at this depth: a face spans `2 · depth` across (its 90° field of view). A receiver on a
        // curved surface sees its neighbouring texels at a depth one texel's slope away, so a bias below a texel
        // self-shadows the lit side. `tests/fixtures/native_oracle/point_shadow_bias.py` measures it against exact rays:
        // 90 of 250 lit sphere points are dark at 0.001 · depth, none at one texel, and the plane's shadow is unchanged.
        let bias = 2.0 * depth / f64::from(SHADOW_MAP_SIZE);
        let mut visible = 0.0;
        for offset in PCF_OFFSETS {
            let sample_x = center_x + offset[0];
            let sample_y = center_y + offset[1];
            if sample_x < 0
                || sample_y < 0
                || sample_x >= SHADOW_MAP_SIZE as i32
                || sample_y >= SHADOW_MAP_SIZE as i32
            {
                visible += 1.0;
                continue;
            }
            let index = sample_y as usize * SHADOW_MAP_SIZE_USIZE + sample_x as usize;
            if face
                .depth
                .get(index)
                .is_none_or(|blocker| depth <= blocker + bias)
            {
                visible += 1.0;
            }
        }
        visible / PCF_OFFSETS.len() as f64
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded shadow-map screen coordinates are converted to texel indices"
)]
fn rasterize_shadow_triangle(face: &mut ShadowFace, origin: DVec3, triangle: &Triangle) {
    let mut projected = [[0.0; 3]; 3];
    for (index, position) in triangle.positions.iter().enumerate() {
        let relative = *position - origin;
        let depth = relative.dot(face.forward);
        if depth <= 1.0e-6 {
            return;
        }
        projected[index] = [
            (relative.dot(face.right) / depth * 0.5 + 0.5) * f64::from(SHADOW_MAP_SIZE),
            (0.5 - relative.dot(face.up) / depth * 0.5) * f64::from(SHADOW_MAP_SIZE),
            depth,
        ];
    }
    let min_x = projected
        .iter()
        .map(|vertex| vertex[0])
        .fold(f64::INFINITY, f64::min);
    let max_x = projected
        .iter()
        .map(|vertex| vertex[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = projected
        .iter()
        .map(|vertex| vertex[1])
        .fold(f64::INFINITY, f64::min);
    let max_y = projected
        .iter()
        .map(|vertex| vertex[1])
        .fold(f64::NEG_INFINITY, f64::max);
    if max_x < 0.0
        || max_y < 0.0
        || min_x >= f64::from(SHADOW_MAP_SIZE)
        || min_y >= f64::from(SHADOW_MAP_SIZE)
    {
        return;
    }
    let min_x = min_x.floor().clamp(0.0, f64::from(SHADOW_MAP_SIZE - 1)) as u32;
    let max_x = max_x.ceil().clamp(0.0, f64::from(SHADOW_MAP_SIZE - 1)) as u32;
    let min_y = min_y.floor().clamp(0.0, f64::from(SHADOW_MAP_SIZE - 1)) as u32;
    let max_y = max_y.ceil().clamp(0.0, f64::from(SHADOW_MAP_SIZE - 1)) as u32;
    let [first, second, third] = projected;
    let area = shadow_edge(first, second, third);
    if area.abs() <= f64::EPSILON {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = [f64::from(x) + 0.5, f64::from(y) + 0.5, 0.0];
            let weights = [
                shadow_edge(second, third, sample) / area,
                shadow_edge(third, first, sample) / area,
                shadow_edge(first, second, sample) / area,
            ];
            if weights.iter().any(|weight| *weight < -1.0e-9) {
                continue;
            }
            let reciprocal_depth =
                weights[0] / first[2] + weights[1] / second[2] + weights[2] / third[2];
            if reciprocal_depth <= f64::EPSILON {
                continue;
            }
            let depth = reciprocal_depth.recip();
            let index = y as usize * SHADOW_MAP_SIZE_USIZE + x as usize;
            if let Some(current) = face.depth.get_mut(index) {
                *current = current.min(depth);
            }
        }
    }
}

fn shadow_edge(first: [f64; 3], second: [f64; 3], point: [f64; 3]) -> f64 {
    (second[0] - first[0]) * (point[1] - first[1]) - (second[1] - first[1]) * (point[0] - first[0])
}

fn subdivide_displaced_triangle(
    doc: &SceneDoc,
    material: &Material,
    prepared: &PreparedMaterial,
    object_to_world: DMat4,
    path_traced: bool,
    triangle: &Triangle,
    images: &BTreeMap<String, ImageData>,
    object: &ObjectRecord,
    layers: HitUvLayers<'_>,
    warnings: &RefCell<Vec<serde_json::Value>>,
    evaluator_workspace: &mut EvaluatorWorkspace,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Result<[Triangle; 4]> {
    subdivide_displaced(triangle, object_to_world, |sample| {
        let mut tables = LayerTables::new();
        let context = layered_context(
            doc,
            images,
            warnings,
            object,
            triangle,
            HitPose {
                point: sample.point,
                object: sample.object,
                normal: sample.normal,
                geometric_normal: (triangle.positions[1] - triangle.positions[0])
                    .cross(triangle.positions[2] - triangle.positions[0])
                    .normalize_or_zero(),
                back_facing: false,
                path_traced,
                view: DVec3::Z,
                barycentric: sample.barycentric,
            },
            Some(sample.uv),
            HitUvLayers {
                barycentric: sample.barycentric,
                ..layers
            },
            &mut tables,
        )?;
        let context = context.with_path_traced(path_traced);
        evaluate_displacement_with_workspace_semantics(
            material,
            prepared,
            &context,
            evaluator_workspace,
            math_semantics,
        )
    })
}

/// One displacement sample of a triangle being subdivided.
struct DisplacementSample {
    point: DVec3,
    object: DVec3,
    uv: [f64; 2],
    /// Barycentric weights of the sample over the triangle's corners.
    barycentric: [f64; 3],
    /// Blended corner normal the sample moves along, or the face normal on flat triangles.
    normal: DVec3,
}

/// Splits a triangle into four and moves its six sample points by `displacement`.
///
/// Samples move along the blended corner normal, so neighbouring smooth triangles agree on
/// shared edges. Each sub-triangle then carries its own displaced face normal, flipped toward
/// the blended sample normals, so the relief still shades instead of following the undisplaced
/// normals.
fn subdivide_displaced(
    triangle: &Triangle,
    object_to_world: DMat4,
    mut displacement: impl FnMut(&DisplacementSample) -> Result<f64>,
) -> Result<[Triangle; 4]> {
    let normal = (triangle.positions[1] - triangle.positions[0])
        .cross(triangle.positions[2] - triangle.positions[0])
        .normalize_or_zero();
    let barycentric = [
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ];
    let mut positions = [DVec3::ZERO; 6];
    let mut geometry_samples = [[0.0; 3]; 6];
    let mut generated = [DVec3::ZERO; 6];
    let mut uvs = [[0.0; 2]; 6];
    let mut layer_samples = vec![[[0.0; 2]; 6]; triangle.layer_uvs.len()];
    let mut color_samples = vec![[[0.0; 4]; 6]; triangle.layer_colors.len()];
    let mut tangent_samples = vec![[DVec3::ZERO; 6]; triangle.layer_tangents.len()];
    let mut sample_normals = [normal; 6];
    let world_to_object = object_to_world.inverse();
    let inverse_is_finite = world_to_object
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite());
    for index in 0..6 {
        let weights = barycentric[index];
        let point = triangle.positions[0] * weights[0]
            + triangle.positions[1] * weights[1]
            + triangle.positions[2] * weights[2];
        let uv = blend_uv(&triangle.uv, weights);
        let object = if inverse_is_finite {
            world_to_object.transform_point3(point)
        } else {
            point
        };
        let sample_normal = triangle.normals.map_or(normal, |corners| {
            blend_corner_normals(corners, weights, normal)
        });
        let sample = DisplacementSample {
            point,
            object,
            uv,
            barycentric: weights,
            normal: sample_normal,
        };
        let height = displacement(&sample)?;
        positions[index] = point + sample_normal * height;
        // Cycles moves only positions and normals when it displaces (`mesh_displace.cpp`), so a sample keeps the
        // Generated of its undisplaced point, blended from the parent corners.
        let [first, second, third] = triangle.generated.corners;
        generated[index] = first * weights[0] + second * weights[1] + third * weights[2];
        for (component, sample) in geometry_samples[index].iter_mut().enumerate() {
            *sample = triangle.geometry_vertex_weights[0][component] * weights[0]
                + triangle.geometry_vertex_weights[1][component] * weights[1]
                + triangle.geometry_vertex_weights[2][component] * weights[2];
        }
        sample_normals[index] = sample_normal;
        uvs[index] = uv;
        for (table, corners) in layer_samples.iter_mut().zip(&triangle.layer_uvs) {
            table[index] = blend_uv(corners, weights);
        }
        for (table, corners) in color_samples.iter_mut().zip(&triangle.layer_colors) {
            table[index] = blend_color(corners, weights);
        }
        for (table, corners) in tangent_samples.iter_mut().zip(&triangle.layer_tangents) {
            table[index] =
                corners[0] * weights[0] + corners[1] * weights[1] + corners[2] * weights[2];
        }
    }
    let indices: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 4], [2, 4, 5], [1, 4, 2]];
    Ok(indices.map(|corners| {
        let mut subtriangle = triangle.with_corners(
            corners.map(|vertex| positions[vertex]),
            GeneratedCorners {
                corners: corners.map(|vertex| generated[vertex]),
                per_local: triangle.generated.per_local,
            },
            corners.map(|vertex| uvs[vertex]),
            layer_samples
                .iter()
                .map(|table| corners.map(|vertex| table[vertex]))
                .collect(),
            tangent_samples
                .iter()
                .map(|table| corners.map(|vertex| table[vertex]))
                .collect(),
            color_samples
                .iter()
                .map(|table| corners.map(|vertex| table[vertex]))
                .collect(),
            corners.map(|vertex| geometry_samples[vertex]),
        );
        if triangle.normals.is_some() {
            let average = corners
                .iter()
                .fold(DVec3::ZERO, |sum, &vertex| sum + sample_normals[vertex]);
            let [a, b, c] = corners.map(|vertex| positions[vertex]);
            let mut face = (b - a)
                .cross(c - a)
                .normalize_or(average.normalize_or(normal));
            if face.dot(average) < 0.0 {
                face = -face;
            }
            subtriangle.normals = Some([face; 3]);
        }
        subtriangle
    }))
}

fn collect_collection_nodes(
    doc: &SceneDoc,
    collection_id: &Id,
    excluded: &HashSet<Id>,
    visited: &mut HashSet<Id>,
    nodes: &mut HashSet<Id>,
) {
    if excluded.contains(collection_id) || !visited.insert(collection_id.clone()) {
        return;
    }
    let Some(collection) = doc.collections.get(collection_id) else {
        return;
    };
    nodes.extend(collection.objects.iter().cloned());
    for child in &collection.children {
        collect_collection_nodes(doc, child, excluded, visited, nodes);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshAttributePrecision {
    NotApplicable,
    Unknown,
    Known(bool),
}

#[derive(Clone, Debug, PartialEq)]
enum FaceNormalValues {
    Constant(DVec3),
    Corners(Vec<DVec3>),
}

impl FaceNormalValues {
    fn as_corners(&self) -> Option<&[DVec3]> {
        match self {
            Self::Constant(_) => None,
            Self::Corners(normals) => Some(normals),
        }
    }

    fn constant(&self) -> Option<DVec3> {
        match self {
            Self::Constant(normal) => Some(*normal),
            Self::Corners(_) => None,
        }
    }

    fn triangle_values(&self, corners: [Option<usize>; 3]) -> Option<[DVec3; 3]> {
        match self {
            Self::Constant(normal) => Some([*normal; 3]),
            Self::Corners(normals) => corner_values(normals, corners),
        }
    }
}

struct FaceNormalData {
    /// Flat source normals stay a single value; smooth faces keep their corner array.
    world: Vec<Option<FaceNormalValues>>,
    local: Option<Vec<Option<FaceNormalValues>>>,
    precision_matches: bool,
}

struct MeshTangentData {
    by_triangle: Vec<Vec<[DVec3; 3]>>,
    precision_matches: bool,
}

fn unknown_mesh_attribute_precision_error(scene_id: &Id, node_id: &Id) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        "realtime mesh shading requires the source mesh attribute precision setting",
        serde_json::json!({
            "feature_id": "render.eevee.mesh_attribute_precision",
            "scene_id": scene_id.as_str(),
            "node_id": node_id.as_str(),
            "setting": "use_high_quality_normals",
        }),
    )
}

/// Corner normals in object and world space. Packed flat normals remain a per-face constant.
fn corner_normals(
    mesh: &crate::geom::Mesh,
    vertex_positions: &BTreeMap<u32, DVec3>,
    matrix: DMat4,
    node_id: &Id,
    adaptive_normals: Option<&[Option<DVec3>]>,
    keep_local: bool,
    precision: MeshAttributePrecision,
    warnings: &mut Vec<serde_json::Value>,
) -> FaceNormalData {
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let mut local = match render_corner_normals(mesh, &positions) {
        Ok(faces) => faces
            .into_iter()
            .map(|face| face.map(FaceNormalValues::Corners))
            .collect::<Vec<_>>(),
        Err(error) => {
            warnings.push(serde_json::json!({
                "code": "shading.corner_normals_failed",
                "message": format!(
                    "smooth shading normals could not be computed ({error}); flat face normals were used"
                ),
                "node_id": node_id,
            }));
            let flat_faces = vec![None; mesh.faces.len()];
            if precision == MeshAttributePrecision::NotApplicable {
                return FaceNormalData {
                    world: flat_faces.clone(),
                    local: keep_local.then_some(flat_faces),
                    precision_matches: true,
                };
            }
            flat_faces
        }
    };
    if precision != MeshAttributePrecision::NotApplicable && local.iter().any(Option::is_none) {
        let mut face_points = Vec::with_capacity(
            mesh.faces
                .iter()
                .map(|face| face.vertices.len())
                .max()
                .unwrap_or(0),
        );
        for (face_index, face) in mesh.faces.iter().enumerate() {
            if local[face_index].is_some() {
                continue;
            }
            face_points.clear();
            for vertex_id in &face.vertices {
                let Some(position) = vertex_positions.get(vertex_id).copied() else {
                    face_points.clear();
                    break;
                };
                face_points.push(position);
            }
            if face_points.len() == face.vertices.len() {
                local[face_index] = Some(FaceNormalValues::Constant(
                    crate::geom::normals::face_normal(&face_points),
                ));
            }
        }
    }
    let mut vertex_indices: Option<BTreeMap<u32, usize>> = None;
    if let Some(normals) = adaptive_normals.filter(|normals| normals.len() == mesh.vertices.len())
        && let Ok(smooth_faces) = smooth_face_flags(mesh)
        && smooth_faces.iter().any(|smooth| *smooth)
    {
        let indices = vertex_indices.get_or_insert_with(|| {
            mesh.vertices
                .iter()
                .enumerate()
                .map(|(index, vertex)| (vertex.id, index))
                .collect::<BTreeMap<_, _>>()
        });
        for (face_index, (face, smooth)) in mesh.faces.iter().zip(smooth_faces).enumerate() {
            if !smooth {
                continue;
            }
            if let Some(normal) = local[face_index]
                .as_ref()
                .and_then(FaceNormalValues::constant)
            {
                local[face_index] =
                    Some(FaceNormalValues::Corners(vec![normal; face.vertices.len()]));
            }
            let Some(FaceNormalValues::Corners(face_normals)) =
                local.get_mut(face_index).and_then(Option::as_mut)
            else {
                continue;
            };
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                if let Some(normal) = indices
                    .get(vertex_id)
                    .and_then(|index| normals.get(*index))
                    .copied()
                    .flatten()
                {
                    face_normals[corner] = normal;
                }
            }
        }
    }
    // Source keeps these raw for MikkTSpace; its normal VBO is quantized before the world transform.
    let normal_matrix = DMat3::from_mat4(matrix).inverse().transpose();
    let transform = |normal: DVec3, high_quality: Option<bool>| {
        let normal = high_quality.map_or(normal, |high_quality| {
            source_quantized_mesh_attribute(normal, high_quality)
        });
        (normal_matrix * normal).normalize_or_zero()
    };
    let mut world = vec![None; mesh.faces.len()];
    let mut precision_matches = true;
    let mut degenerate = false;
    for face_index in 0..mesh.faces.len() {
        match local[face_index].as_ref() {
            Some(FaceNormalValues::Corners(local_normals)) => {
                let face_world: Vec<DVec3> = local_normals
                    .iter()
                    .map(|normal| {
                        transform(
                            *normal,
                            match precision {
                                MeshAttributePrecision::Known(high_quality) => Some(high_quality),
                                MeshAttributePrecision::Unknown => Some(false),
                                MeshAttributePrecision::NotApplicable => None,
                            },
                        )
                    })
                    .collect();
                if precision == MeshAttributePrecision::Unknown {
                    let high_quality_world: Vec<DVec3> = local_normals
                        .iter()
                        .map(|normal| transform(*normal, Some(true)))
                        .collect();
                    if face_world != high_quality_world {
                        precision_matches = false;
                        continue;
                    }
                }
                let valid = face_world
                    .iter()
                    .all(|normal| normal.is_finite() && *normal != DVec3::ZERO);
                degenerate |= !valid;
                if valid {
                    world[face_index] = Some(FaceNormalValues::Corners(face_world));
                }
            }
            Some(FaceNormalValues::Constant(local_normal)) => {
                let face_world = transform(
                    *local_normal,
                    match precision {
                        MeshAttributePrecision::Known(high_quality) => Some(high_quality),
                        MeshAttributePrecision::Unknown => Some(false),
                        MeshAttributePrecision::NotApplicable => None,
                    },
                );
                if precision == MeshAttributePrecision::Unknown
                    && face_world != transform(*local_normal, Some(true))
                {
                    precision_matches = false;
                    continue;
                }
                let valid = face_world.is_finite() && face_world != DVec3::ZERO;
                degenerate |= !valid;
                if valid {
                    world[face_index] = Some(FaceNormalValues::Constant(face_world));
                }
            }
            None => {}
        }
    }
    if degenerate {
        warnings.push(serde_json::json!({
            "code": "shading.corner_normals_failed",
            "message": "mesh shading normals became zero-length under the object transform; affected faces were shaded flat",
            "node_id": node_id,
        }));
    }
    FaceNormalData {
        world,
        local: keep_local.then_some(local),
        precision_matches,
    }
}
/// Blender's mesh VBO truncates normalized f32 components before SNORM decoding.
fn source_quantized_mesh_attribute(value: DVec3, high_quality: bool) -> DVec3 {
    DVec3::from_array(value.to_array().map(|component| {
        let component = component as f32;
        let decoded = if high_quality {
            let code =
                ((component * 32767.0) as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            (code as f32 / 32767.0).max(-1.0)
        } else {
            let code = ((component * 511.0) as i32).clamp(-512, 511);
            (code as f32 / 511.0).max(-1.0)
        };
        f64::from(decoded)
    }))
}

#[cfg(test)]
fn mesh_vertex_positions(mesh: &Mesh) -> BTreeMap<u32, DVec3> {
    mesh.vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect()
}

#[cfg(test)]
fn world_face_normals(
    mesh: &crate::geom::Mesh,
    matrix: DMat4,
    node_id: &Id,
    adaptive_normals: Option<&[Option<DVec3>]>,
    warnings: &mut Vec<serde_json::Value>,
) -> Vec<Option<Vec<DVec3>>> {
    let vertex_positions = mesh_vertex_positions(mesh);
    let data = corner_normals(
        mesh,
        &vertex_positions,
        matrix,
        node_id,
        adaptive_normals,
        false,
        MeshAttributePrecision::NotApplicable,
        warnings,
    );
    data.world
        .into_iter()
        .enumerate()
        .map(|(face_index, normals)| match normals {
            Some(FaceNormalValues::Corners(corners)) => Some(corners),
            Some(FaceNormalValues::Constant(normal)) => {
                Some(vec![normal; mesh.faces[face_index].vertices.len()])
            }
            None => None,
        })
        .collect()
}
struct MikkGeometry<'a> {
    positions: &'a [[[f32; 3]; 3]],
    normals: &'a [[[f32; 3]; 3]],
    uvs: &'a [[[f32; 2]; 3]],
    tangents: Vec<[[f32; 3]; 3]>,
}

impl mikktspace::Geometry for MikkGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.positions.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[face][vert]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[face][vert]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[face][vert]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        self.tangents[face][vert] = [tangent[0], tangent[1], tangent[2]];
    }
}

/// Calculates raw-local Mikk tangents for every UV layer, then packs the output mesh tangent
/// attribute before applying its object-normal transform for hit interpolation.
fn mesh_mikk_tangents(
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    vertex_positions: &BTreeMap<u32, DVec3>,
    uv_store: &UvStore,
    local_face_normals: Option<&[Option<FaceNormalValues>]>,
    matrix: DMat4,
    precision: MeshAttributePrecision,
) -> Result<MeshTangentData> {
    if triangles.is_empty() {
        return Ok(MeshTangentData {
            by_triangle: Vec::new(),
            precision_matches: true,
        });
    }
    let mut positions = Vec::with_capacity(triangles.len());
    let mut normals = Vec::with_capacity(triangles.len());
    let mut uvs_by_layer: Vec<Vec<[[f32; 2]; 3]>> = (0..uv_store.layers.len())
        .map(|_| Vec::with_capacity(triangles.len()))
        .collect();
    let mut triangle_offset = 0_usize;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let triangle_count = face.vertices.len().checked_sub(2).ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "mesh face has fewer than three vertices",
            )
        })?;
        let triangle_end = triangle_offset.checked_add(triangle_count).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "triangle offset exceeds the supported index range",
            )
        })?;
        let face_triangles = triangles
            .get(triangle_offset..triangle_end)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "triangulated face ranges are inconsistent",
                )
            })?;
        for triangle in face_triangles {
            let corners = triangle_corners(face, triangle);
            let mut local_positions = [DVec3::ZERO; 3];
            for (index, vertex_id) in triangle.iter().enumerate() {
                local_positions[index] =
                    vertex_positions.get(vertex_id).copied().ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "triangle references a missing vertex",
                        )
                    })?;
            }
            let mikk_positions = local_positions.map(mikk_vec3);
            let flat_normal = mikk_face_normal(mikk_positions);
            let mikk_normals = local_face_normals
                .and_then(|faces| faces.get(face_index))
                .and_then(Option::as_ref)
                .and_then(|face_normals| face_normals.triangle_values(corners))
                .map_or([flat_normal; 3], |normals| normals.map(mikk_vec3));
            positions.push(mikk_positions);
            normals.push(mikk_normals);
            for (uvs, layer) in uvs_by_layer.iter_mut().zip(&uv_store.layers) {
                let values = layer.faces.get(face_index).and_then(Option::as_deref);
                uvs.push(corners.map(|corner| {
                    corner
                        .and_then(|corner| values.and_then(|values| values.get(corner)))
                        .copied()
                        .unwrap_or([0.0, 1.0])
                        .map(|coordinate| coordinate as f32)
                }));
            }
        }
        triangle_offset = triangle_end;
    }
    if triangle_offset != triangles.len() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "triangulated face ranges are inconsistent",
        ));
    }

    let normal_matrix = DMat3::from_mat4(matrix).inverse().transpose();
    let transform = |tangent: DVec3, high_quality: Option<bool>| {
        let tangent = high_quality.map_or(tangent, |high_quality| {
            source_quantized_mesh_attribute(tangent, high_quality)
        });
        normal_matrix * tangent
    };
    let mut precision_matches = true;
    let mut tangents_by_triangle: Vec<Vec<[DVec3; 3]>> = (0..triangles.len())
        .map(|_| Vec::with_capacity(uv_store.layers.len()))
        .collect();
    for uvs in &uvs_by_layer {
        let mut geometry = MikkGeometry {
            positions: &positions,
            normals: &normals,
            uvs,
            tangents: vec![[[0.0; 3]; 3]; triangles.len()],
        };
        let _ = mikktspace::generate_tangents(&mut geometry);
        for (target, corners) in tangents_by_triangle.iter_mut().zip(geometry.tangents) {
            target.push(corners.map(|tangent| {
                let tangent = DVec3::new(tangent[0] as f64, tangent[1] as f64, tangent[2] as f64);
                let world = match precision {
                    MeshAttributePrecision::NotApplicable => transform(tangent, None),
                    MeshAttributePrecision::Known(high_quality) => {
                        transform(tangent, Some(high_quality))
                    }
                    MeshAttributePrecision::Unknown => {
                        let low_quality = transform(tangent, Some(false));
                        let high_quality = transform(tangent, Some(true));
                        precision_matches &= low_quality == high_quality;
                        low_quality
                    }
                };
                if world.is_finite() {
                    world
                } else {
                    DVec3::ZERO
                }
            }));
        }
    }
    Ok(MeshTangentData {
        by_triangle: tangents_by_triangle,
        precision_matches,
    })
}

fn mikk_vec3(value: DVec3) -> [f32; 3] {
    [value.x as f32, value.y as f32, value.z as f32]
}
/// Cycles' `Mesh::Triangle::compute_normal` for the flat-normal input to `MikkTSpace`.
fn mikk_face_normal(points: [[f32; 3]; 3]) -> [f32; 3] {
    let [first, second, third] = points;
    let edge1 = [
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    ];
    let edge2 = [
        third[0] - first[0],
        third[1] - first[1],
        third[2] - first[2],
    ];
    let normal = [
        edge1[1] * edge2[2] - edge1[2] * edge2[1],
        edge1[2] * edge2[0] - edge1[0] * edge2[2],
        edge1[0] * edge2[1] - edge1[1] * edge2[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length == 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    }
}

/// Ray-visibility flags that realtime Beauty does not honour, one warning per flag switched away from
/// Blender's default for a light. Blender's panel shows the flags each engine exposes
/// (`scripts/startup/bl_ui/properties_object.py`, `OBJECT_PT_visibility`, EEVEE branch): a mesh has
/// Camera, Shadow and Raycast, and a light has Camera, Diffuse, Glossy, Transmission and Volume Scatter.
/// EEVEE has no mesh diffuse, glossy, transmission or volume flag, so realtime warns about none of
/// those. A light's camera visibility shows its emitter in realtime as in EEVEE (`eevee_light_shape_display`), so it
/// warns about none of the light's flags except the two below.
/// EEVEE applies a light's transmission and volume scatter to its power (`eevee_light.cc`,
/// `transmission_visibility` and `volume_visibility`), and realtime does not. The path tracer honours
/// every flag. A light's `visible_shadow` is stored without a warning: Blender reads it nowhere
/// (Cycles' `light_object_visibility_flags`, EEVEE's `eevee_light.cc`).
fn ray_visibility_warnings(node_id: &Id, node: &Node, warnings: &mut Vec<serde_json::Value>) {
    const REALTIME_LIGHT: [(&str, RayKind); 2] = [
        ("transmission", RayKind::Transmission),
        ("volume_scatter", RayKind::VolumeScatter),
    ];
    if node.kind != "light" {
        return;
    }
    let visibility = node.resolved_ray_visibility();
    let default = RayVisibility::default_for_kind(&node.kind);
    for &(name, ray) in &REALTIME_LIGHT {
        let value = visibility.allows(ray);
        let expected = default.allows(ray);
        if value == expected {
            continue;
        }
        warnings.push(serde_json::json!({
            "code": "UNSUPPORTED_FEATURE",
            "feature_id": format!("object.ray_visibility.{name}"),
            "node_id": node_id,
            "message": format!(
                "visible_{name} is {value}, which realtime Beauty does not honour for this {}; it renders as if visible_{name} were {expected}",
                node.kind,
            ),
        }));
    }
}

/// Principled inputs of a surface shaded with `material`: the path tracer's own construction
/// (`BsdfParams::from_material`), with the surface colour `base` in place of the material's. An
/// unmapped slot takes the default material's.
fn shading_params(material: Option<&Material>, base: [f64; 4], normal: DVec3) -> BsdfParams {
    let mut params = material.map_or_else(
        || BsdfParams::from_material(&Material::default(), normal),
        |material| BsdfParams::from_material(material, normal),
    );
    params.base_color = base;
    params.alpha = base[3];
    params
}

/// Per-pixel lighting for every hit of raster Solid, which lights it in the camera's view space, and of realtime
/// Beauty, which lights it in world space with the hit's own normal: the geometric normal on flat faces, so their
/// shadows are decided per pixel as EEVEE decides them.
pub(super) struct SceneShader<'a> {
    core: SceneShaderCore<'a>,
    warnings: &'a RefCell<Vec<serde_json::Value>>,
}

struct SceneShaderWorker<'a> {
    core: SceneShaderCore<'a>,
}

struct EeveeWorldSunLight {
    light: LightData,
    volume_energy: f64,
    matrix: DMat4,
    ray_visibility: RayVisibility,
    use_shadows: bool,
    shadow_radius: Option<f64>,
}
#[derive(Clone, Copy)]
pub(super) struct EeveeWorldSunVolumeLight {
    pub(super) radiance: DVec3,
    pub(super) matrix: DMat4,
    pub(super) volume_energy: f64,
    pub(super) use_sun_shadow: bool,
    pub(super) shadow_radius: Option<f64>,
}

#[derive(Default)]
struct EeveeWorldSunState {
    light: Option<EeveeWorldSunLight>,
    unsupported_shadow_map_settings: bool,
}

fn eevee_world_sun_settings_refusal(scene_id: &Id, setting: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "EEVEE extracted WorldSun needs source setting `{setting}`; reimport the Blender scene or set it explicitly"
        ),
        serde_json::json!({
            "feature_id":"render.eevee.world_sun.settings",
            "scene_id":scene_id.as_str(),
            "setting":setting,
        }),
    )
}
pub(super) fn ensure_eevee_global_world_probe_available(
    scene_id: &Id,
    local_volume_probe_grids: Option<bool>,
) -> Result<()> {
    if local_volume_probe_grids == Some(false) {
        return Ok(());
    }
    let message = if local_volume_probe_grids == Some(true) {
        "EEVEE global World SH cannot substitute for visible local VOLUME probe grids; local volume-probe irradiance is not represented"
    } else {
        "EEVEE world-probe irradiance cannot be used because source participation in local VOLUME probe grids is unknown; reimport the Blender scene to recover it"
    };
    Err(PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        serde_json::json!({
            "feature_id":"render.eevee.world_probe.local_volume_grids",
            "scene_id":scene_id.as_str(),
            "setting":"eevee_has_local_volume_probe_grids",
        }),
    ))
}

// Transform::operator float4x4 reconstructs the Duff basis columns from the BSL's transposed rows.
fn eevee_world_sun_matrix(direction: Vec3) -> DMat4 {
    let z_sign = if direction.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (z_sign + direction.z);
    let b = direction.x * direction.y * a;
    let basis = Mat3::from_cols(
        Vec3::new(
            1.0 + z_sign * direction.x * direction.x * a,
            z_sign * b,
            -z_sign * direction.x,
        ),
        Vec3::new(b, z_sign + direction.y * direction.y * a, -direction.y),
        direction,
    );
    let to_double =
        |axis: Vec3| DVec3::new(f64::from(axis.x), f64::from(axis.y), f64::from(axis.z));
    DMat4::from_cols(
        to_double(basis.x_axis).extend(0.0),
        to_double(basis.y_axis).extend(0.0),
        to_double(basis.z_axis).extend(0.0),
        DVec3::ZERO.extend(1.0),
    )
}
fn eevee_world_sun_light(
    world: &World,
    source: EeveeWorldSun,
) -> std::result::Result<EeveeWorldSunLight, &'static str> {
    let sun_angle = world.eevee_sun_angle.ok_or("eevee_sun_angle")?;
    let use_shadows = world.eevee_use_sun_shadow.ok_or("eevee_use_sun_shadow")?;
    let shadow_radius = if use_shadows {
        let jitter = world
            .eevee_use_sun_shadow_jitter
            .ok_or("eevee_use_sun_shadow_jitter")?;
        let overblur = if jitter {
            world
                .eevee_sun_shadow_jitter_overblur
                .ok_or("eevee_sun_shadow_jitter_overblur")?
                / 100.0
        } else {
            1.0
        };
        let half_angle = sun_angle.min(179.9_f64.to_radians()) * 0.5;
        let shadow_angle = if half_angle > 0.0 {
            (half_angle * overblur).max(1.0e-8)
        } else {
            0.0
        };
        Some(shadow_angle.tan())
    } else {
        None
    };
    let mut ray_visibility = RayVisibility::default();
    ray_visibility.set(RayKind::Camera, false);
    Ok(EeveeWorldSunLight {
        light: LightData {
            light_type: LightType::Sun,
            color: source.radiance.to_array().map(f64::from),
            energy: std::f64::consts::FRAC_1_PI * (1.0 + 1.0 / sun_disc_radius(sun_angle).powi(2)),
            sun_angle,
            use_shadow: Some(use_shadows),
            ..LightData::default()
        },
        matrix: eevee_world_sun_matrix(source.direction),
        ray_visibility,
        volume_energy: 1.0,
        use_shadows,
        shadow_radius,
    })
}

struct SceneShaderCore<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: super::path::SharedGeometry<'a>,
    mode: Mode,
    math_semantics: crate::node_kernels::MathSemantics,
    /// Captured Blender EEVEE settings, or a precise refusal when imported state is unknown.
    eevee_raytracing_settings: Result<Option<EeveeRayTracingSettings>>,
    /// Cached source-compiled flags for the frame's two deferred material layers.
    eevee_raytrace_layer_flags: Result<Option<EeveeRaytraceLayerSummary>>,
    /// Forward EEVEE chooses RGB transparency targets scene-wide from compiled material flags.
    eevee_colored_transparency: bool,
    /// The camera's view space, in which Solid's studio lights are fixed.
    view: &'a ViewBasis,
    evaluator_workspace: RefCell<EvaluatorWorkspace>,
    /// Lazily populated once per render and shared by the serial pass and all band workers.
    world_probe: Arc<OnceLock<Result<Option<Arc<WorldProbe>>>>>,
    /// Source `WorldSun` as an ephemeral native direct light, shared by the render workers.
    world_sun: Arc<OnceLock<Result<EeveeWorldSunState>>>,
}

/// The per-pixel shader for `mode`, or `None` when the raster outputs unlit or per-triangle colour.
/// Beauty shades in world space and ignores `view`.
pub(super) fn surface_shader<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: &'a Geometry,
    mode: Mode,
    view: &'a ViewBasis,
) -> Option<Box<dyn SurfaceShader + 'a>> {
    surface_shader_with_semantics(
        doc,
        snapshot,
        images,
        geometry,
        mode,
        view,
        crate::node_kernels::MathSemantics::Cycles,
    )
}

pub(super) fn surface_shader_with_semantics<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: &'a Geometry,
    mode: Mode,
    view: &'a ViewBasis,
    math_semantics: crate::node_kernels::MathSemantics,
) -> Option<Box<dyn SurfaceShader + 'a>> {
    match mode {
        Mode::Solid => {}
        Mode::Beauty if geometry.shadows.is_some() => {}
        _ => return None,
    }
    scene_shader_with_semantics(doc, snapshot, images, geometry, mode, view, math_semantics)
        .into_surface_shader()
}

pub(super) fn scene_shader_with_semantics<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: &'a Geometry,
    mode: Mode,
    view: &'a ViewBasis,
    math_semantics: crate::node_kernels::MathSemantics,
) -> SceneShader<'a> {
    let eevee_colored_transparency = math_semantics == crate::node_kernels::MathSemantics::Eevee
        && geometry_uses_eevee_colored_transparency(geometry);
    let eevee_raytracing_settings = doc
        .scenes
        .get(&snapshot.scene_id)
        .map_or(Ok(None), |scene| {
            if !matches!(mode, Mode::Beauty)
                || math_semantics != crate::node_kernels::MathSemantics::Eevee
            {
                return Ok(None);
            }
            let blender_imported = doc
                .compatibility
                .get("blender_adapter")
                .is_some_and(serde_json::Value::is_object);
            EeveeRayTracingSettings::from_render_settings(
                &scene.render,
                blender_imported,
                snapshot.scene_id.as_str(),
            )
        });
    let eevee_raytrace_layer_flags = if matches!(mode, Mode::Beauty)
        && math_semantics == crate::node_kernels::MathSemantics::Eevee
    {
        eevee_registered_raytrace_layer_summary(doc, snapshot, geometry).map(Some)
    } else {
        Ok(None)
    };
    SceneShader {
        core: SceneShaderCore {
            doc,
            snapshot,
            images,
            geometry: super::path::SharedGeometry::of(geometry),
            mode,
            math_semantics,
            eevee_colored_transparency,
            eevee_raytracing_settings,
            eevee_raytrace_layer_flags,
            view,
            evaluator_workspace: RefCell::new(EvaluatorWorkspace::default()),
            world_probe: Arc::new(OnceLock::new()),
            world_sun: Arc::new(OnceLock::new()),
        },
        warnings: &geometry.shader_warnings,
    }
}

impl<'a> SceneShader<'a> {
    pub(super) fn into_surface_shader(self) -> Option<Box<dyn SurfaceShader + 'a>> {
        match self.core.mode {
            Mode::Solid => Some(Box::new(self)),
            Mode::Beauty if self.core.geometry.shadows.is_some() => Some(Box::new(self)),
            _ => None,
        }
    }

    pub(super) fn eevee_world_sun_volume_light(&self) -> Result<Option<EeveeWorldSunVolumeLight>> {
        let sun = self.core.world_sun(self.warnings)?.light.as_ref();
        Ok(sun.map(|sun| EeveeWorldSunVolumeLight {
            radiance: DVec3::from_array(sun.light.color),
            matrix: sun.matrix,
            volume_energy: sun.volume_energy,
            use_sun_shadow: sun.use_shadows,
            shadow_radius: sun.shadow_radius,
        }))
    }
}

impl SurfaceShader for SceneShader<'_> {
    fn shade(&self, triangle: &Triangle, hit: &SurfaceHit<'_>) -> Result<SurfaceSample> {
        self.core.shade(self.warnings, triangle, hit)
    }
    fn needs_eevee_gbuffer_context(&self) -> bool {
        matches!(self.core.mode, Mode::Beauty)
            && self.core.math_semantics == crate::node_kernels::MathSemantics::Eevee
    }
    fn eevee_raytracing_settings(&self) -> Result<Option<EeveeRayTracingSettings>> {
        self.core.eevee_raytracing_settings.clone()
    }
    fn eevee_raytrace_layer_flags(&self) -> Result<Option<EeveeRaytraceLayerSummary>> {
        self.core.eevee_raytrace_layer_flags.clone()
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        sample: &EeveeRaytraceSample,
        bin: EeveeRaytraceBin,
        world_direction: DVec3,
    ) -> Result<DVec3> {
        self.core
            .eevee_raytrace_probe_radiance(sample, bin, world_direction)
    }
    fn eevee_fast_gi_volume_irradiance(&self, _point: DVec3) -> Result<Option<FastGiSh>> {
        self.core.eevee_fast_gi_volume_irradiance()
    }

    fn worker(&self) -> Result<Option<Box<dyn SurfaceShaderWorker + Send + '_>>> {
        Ok(Some(Box::new(SceneShaderWorker {
            core: self.core.worker(),
        })))
    }

    fn merge_worker_warnings(&self, warnings: Vec<serde_json::Value>) {
        super::path::merge_shader_warnings(self.warnings, warnings);
    }
}

/// Whether a camera-visible BLENDED surface submitted to EEVEE's forward pass enables RGB transparency targets.
fn geometry_uses_eevee_colored_transparency(geometry: &Geometry) -> bool {
    let enables_rgb_transparency = |material_index: usize| {
        geometry
            .materials
            .get(material_index)
            .zip(geometry.prepared.get(material_index))
            .is_some_and(|(material, prepared)| {
                material.alpha_mode == "blend"
                    && (prepared.eevee_transparent_maybe_colored() || prepared.eevee_has_holdout())
            })
    };
    geometry.triangles.iter().any(|triangle| {
        triangle.ray_mask & RayKind::Camera.bit() != 0
            && enables_rgb_transparency(triangle.material_index)
    }) || geometry.hair_curve_instances.iter().any(|curve| {
        curve.ray_mask & RayKind::Camera.bit() != 0
            && enables_rgb_transparency(curve.material_index)
    })
}

/// Mesh sync registers each used-with-fallback slot; hair curves and point clouds use only slot zero.
/// (`material.cc:870-879`, `eevee_material.cc:443-455`). `Snapshot.slot_material` applies the
/// evaluated object/data bindings, clamped exactly like `BKE_object_material_get_eval`.
fn eevee_registered_raytrace_layer_summary(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    geometry: &Geometry,
) -> Result<EeveeRaytraceLayerSummary> {
    let default_flags = geometry
        .prepared
        .first()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "deferred geometry has no prepared default surface material",
            )
        })?
        .eevee_closure_flags()?;
    let mut prepared_flags = BTreeMap::new();
    for object in &geometry.objects {
        for entry in &object.material_colors.entries {
            if let Some(material_id) = &entry.material_id
                && !prepared_flags.contains_key(material_id)
                && let Some(prepared) = geometry.prepared.get(entry.material_index)
            {
                prepared_flags.insert(material_id.clone(), prepared.eevee_closure_flags()?);
            }
        }
    }
    let (mut opaque, mut refraction, mut has_refraction_material) = (0, 0, false);
    for object in &geometry.objects {
        let node = doc.nodes.get(&object.node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "deferred geometry references an unmapped object",
            )
        })?;
        let data = object
            .data_id
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id));
        for slot in 0..eevee_deferred_material_slot_count(snapshot, object, data) {
            let material_id = snapshot.slot_material(&object.node_id, node, data, slot);
            let Some(material) = material_id.and_then(|id| doc.materials.get(id)) else {
                // Blender's empty slot uses its default surface material; it is not a refraction layer.
                opaque |= match default_flags {
                    0 => EEVEE_CLOSURE_EMISSION,
                    flags => flags,
                };
                continue;
            };
            if material.alpha_mode == "blend" {
                continue;
            }
            let use_refraction = material.use_raytrace_refraction.ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    "EEVEE ray tracing needs the material's raytrace-refraction setting to identify its deferred layer",
                    serde_json::json!({
                        "feature_id":"render.eevee.ray_tracing.refraction_material",
                    }),
                )
            })?;
            let material_id = material_id.ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "resolved deferred material has no material ID",
                )
            })?;
            let flags = if let Some(flags) = prepared_flags.get(material_id).copied() {
                flags
            } else {
                let flags =
                    PreparedMaterial::new(material, &doc.node_groups).eevee_closure_flags()?;
                prepared_flags.insert(material_id.clone(), flags);
                flags
            };
            let flags = match flags {
                0 => EEVEE_CLOSURE_EMISSION,
                flags => flags,
            };
            if use_refraction {
                refraction |= flags;
                has_refraction_material = true;
            } else {
                opaque |= flags;
            }
        }
    }
    Ok(eevee_raytrace_layer_summary_from_masks(
        opaque,
        refraction,
        has_refraction_material,
    ))
}

fn eevee_deferred_material_slot_count(
    snapshot: &Snapshot,
    object: &ObjectRecord,
    data: Option<&crate::model::DataBlock>,
) -> usize {
    if data.is_some_and(|data| data.data_type == "volume" || data.volume.is_some()) {
        // Source volume passes use MAT_GEOM_VOLUME and never register deferred-layer materials.
        return 0;
    }
    if data.is_some_and(|data| data.hair_curves.is_some() || data.pointcloud.is_some()) {
        // EEVEE sync_curves/sync_pointcloud compile their single material_get slot zero.
        return 1;
    }
    let Some(mesh) = snapshot.meshes.get(&object.node_id) else {
        return 0;
    };
    // `Mesh::material_index_max` clamps to MAXMAT and returns None on empty faces; BKE's
    // used-with-fallback count still registers slot zero. sync_mesh obtains this array before
    // returning for an empty material geometry span (`eevee_sync.cc:222-227`).
    let max_index = mesh
        .faces
        .iter()
        .map(|face| face.material_index)
        .max()
        .unwrap_or(0)
        .min(i16::MAX as u32);
    usize::try_from(max_index).unwrap_or(0).saturating_add(1)
}

fn eevee_raytrace_layer_summary_from_masks(
    opaque: u32,
    refraction: u32,
    has_refraction_material: bool,
) -> EeveeRaytraceLayerSummary {
    let layer_flags = |flags: u32| {
        // DeferredLayer::end_sync omits SSS here; its broader transmission predicate includes it.
        let has_transmit_closure =
            flags & (EEVEE_CLOSURE_REFRACTION | EEVEE_CLOSURE_TRANSLUCENT) != 0;
        EeveeRaytraceLayerFlags {
            has_any_closure: flags != 0,
            has_reflect_closure: flags & (EEVEE_CLOSURE_DIFFUSE | EEVEE_CLOSURE_REFLECTION) != 0,
            has_transmit_closure,
            // CLOSURE_TRANSMISSION includes SSS, while this separate pair handles transparency + ShaderToRGB.
            has_transmission: flags
                & (EEVEE_CLOSURE_SUBSURFACE | EEVEE_CLOSURE_REFRACTION | EEVEE_CLOSURE_TRANSLUCENT)
                != 0
                || flags & EEVEE_CLOSURE_TRANSPARENCY != 0
                    && flags & EEVEE_CLOSURE_SHADER_TO_RGBA != 0,
        }
    };
    EeveeRaytraceLayerSummary {
        opaque: layer_flags(opaque),
        refraction: has_refraction_material.then(|| layer_flags(refraction)),
    }
}

/// The transparent tint the Beauty composite applies to a surface. EEVEE DITHERED uses mean transmittance; BLENDED
/// forward uses the scene-wide target choice (`eevee_surf_forward.bsl.hh`): red-only when monochromatic, full RGB when
/// `ForwardPipeline::use_colored_transparency` is enabled.
fn beauty_tint(
    material: Option<&Material>,
    tint: [f64; 3],
    eevee_monochromatic_transparency: bool,
) -> [f64; 3] {
    // A transmittance is a fraction of the light: EEVEE clamps the Color at zero and saturates the result
    // (`gpu_shader_material_transparent.glsl`, `eevee_surf_deferred.bsl.hh`), so a tint above one or below zero is cut to
    // [0, 1] before it can amplify or invert the background.
    let tint = tint.map(|channel| channel.clamp(0.0, 1.0));
    match material {
        Some(material) if material.alpha_mode != "blend" => {
            let mean = (tint[0] + tint[1] + tint[2]) / 3.0;
            [mean; 3]
        }
        Some(_) if eevee_monochromatic_transparency => [tint[0]; 3],
        _ => tint,
    }
}

impl SceneShaderCore<'_> {
    fn worker(&self) -> Self {
        Self {
            doc: self.doc,
            snapshot: self.snapshot,
            images: self.images,
            geometry: self.geometry,
            mode: self.mode,
            math_semantics: self.math_semantics,
            eevee_colored_transparency: self.eevee_colored_transparency,
            eevee_raytracing_settings: self.eevee_raytracing_settings.clone(),
            eevee_raytrace_layer_flags: self.eevee_raytrace_layer_flags.clone(),
            view: self.view,
            evaluator_workspace: RefCell::new(EvaluatorWorkspace::default()),
            world_probe: self.world_probe.clone(),
            world_sun: self.world_sun.clone(),
        }
    }

    /// The render's world probe holds radiance at every texel of a cube map. The first shade that needs it builds it
    /// lazily, then all band workers share it because the world cannot change during a render
    /// (`LightProbeModule::sync_world`, `eevee_lightprobe.cc`). `None` means the scene has no world.
    fn world_probe(&self) -> Result<Option<&WorldProbe>> {
        let cached = self.world_probe.get_or_init(|| {
            let scene = self.doc.scenes.get(&self.snapshot.scene_id);
            let world = selected_world_id(self.doc, self.snapshot)
                .and_then(|id| self.doc.worlds.get(id));
            if self.math_semantics == crate::node_kernels::MathSemantics::Eevee
                && super::path::scene_has_world_volume(self.doc, self.snapshot)
            {
                return Err(eevee_world_volume_absorption_refusal(
                    &self.snapshot.scene_id,
                ));
            }
            let local_volume_probe_grids =
                scene.and_then(|scene| scene.eevee_has_local_volume_probe_grids);
            if self.math_semantics == crate::node_kernels::MathSemantics::Eevee {
                ensure_eevee_global_world_probe_available(
                    &self.snapshot.scene_id,
                    local_volume_probe_grids,
                )?;
            }
            world
                .map(|world| {
                    let mode = if self.math_semantics == crate::node_kernels::MathSemantics::Eevee {
                        let resolution = scene
                            .and_then(|scene| scene.render.eevee_gi_cubemap_resolution)
                            .ok_or_else(|| {
                                super::path::realtime_world_probe_resolution_refusal(
                                    &self.snapshot.scene_id,
                                )
                            })?;
                        let surface_indirect_clamp = scene
                            .and_then(|scene| scene.render.eevee_clamp_surface_indirect)
                            .ok_or_else(|| {
                                PotError::with_details(
                                    ErrorCode::UnsupportedFeature,
                                    "EEVEE realtime world lighting needs the source clamp_surface_indirect setting; reimport the Blender scene to recover it",
                                    serde_json::json!({"feature_id":"render.eevee.clamp_surface_indirect","scene_id":self.snapshot.scene_id.as_str()}),
                                )
                            })?;
                        WorldProbeMode::Eevee {
                            cubemap_resolution: resolution,
                            surface_indirect_clamp: surface_indirect_clamp as f32,
                        }
                    } else {
                        WorldProbeMode::Path
                    };
                    WorldProbe::build(world, &self.doc.node_groups, self.images, mode).map(Arc::new)
                })
                .transpose()
        });
        match cached {
            Ok(probe) => Ok(probe.as_deref()),
            Err(error) => Err(error.clone()),
        }
    }
    fn eevee_fast_gi_volume_irradiance(&self) -> Result<Option<FastGiSh>> {
        let probe = self.world_probe()?;
        Ok(match probe {
            Some(probe) => probe.fast_gi_sh(),
            None if self.math_semantics == crate::node_kernels::MathSemantics::Eevee => {
                Some(FastGiSh::ZERO)
            }
            None => None,
        })
    }

    fn world_sun(&self, warnings: &RefCell<Vec<serde_json::Value>>) -> Result<&EeveeWorldSunState> {
        let cached = self.world_sun.get_or_init(|| {
            let state = self.build_world_sun()?;
            if state.unsupported_shadow_map_settings {
                let warning = serde_json::json!({
                    "code":"UNSUPPORTED_FEATURE",
                    "feature_id":"render.eevee.world_sun.shadow_map_settings",
                    "message":"Native WorldSun shadows use the scene BVH; World.sun_shadow_maximum_resolution and World.sun_shadow_filter_radius are not represented.",
                });
                let mut recorded = warnings.borrow_mut();
                if !recorded.contains(&warning) {
                    recorded.push(warning);
                }
            }
            Ok(state)
        });

        match cached {
            Ok(state) => Ok(state),
            Err(error) => Err(error.clone()),
        }
    }
    fn build_world_sun(&self) -> Result<EeveeWorldSunState> {
        if !matches!(self.mode, Mode::Beauty)
            || self.math_semantics != crate::node_kernels::MathSemantics::Eevee
        {
            return Ok(EeveeWorldSunState::default());
        }
        let Some(world) = selected_world_id(self.doc, self.snapshot)
            .and_then(|world_id| self.doc.worlds.get(world_id))
        else {
            return Ok(EeveeWorldSunState::default());
        };
        if !world
            .eevee_sun_threshold
            .is_some_and(|threshold| threshold > 0.0)
        {
            return Ok(EeveeWorldSunState::default());
        }
        let Some(probe) = self.world_probe()? else {
            return Ok(EeveeWorldSunState::default());
        };
        let Some(source_sun) = probe.world_sun() else {
            return Ok(EeveeWorldSunState::default());
        };
        let light = eevee_world_sun_light(world, source_sun).map_err(|setting| {
            eevee_world_sun_settings_refusal(&self.snapshot.scene_id, setting)
        })?;
        let unsupported_shadow_map_settings = light.use_shadows;
        Ok(EeveeWorldSunState {
            light: Some(light),
            unsupported_shadow_map_settings,
        })
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        sample: &EeveeRaytraceSample,
        bin: EeveeRaytraceBin,
        world_direction: DVec3,
    ) -> Result<DVec3> {
        let Some(world) = self.world_probe()? else {
            return Ok(DVec3::ZERO);
        };
        match bin.world_input.lobe {
            RayLobe::Diffuse | RayLobe::BurleyBssrdf => Ok(prefiltered_world(
                world,
                sample.normal,
                1.0,
                irradiance_probe_radiance,
            )),
            RayLobe::Translucent
                if bin.world_input.thickness.mode == ThicknessMode::Sphere
                    && bin.world_input.thickness.value != 0.0 =>
            {
                Ok(world.average_irradiance())
            }
            RayLobe::Translucent => Ok(prefiltered_world(
                world,
                -sample.normal,
                1.0,
                irradiance_probe_radiance,
            )),
            _ => eevee_deferred_indirect_world(
                world,
                world_direction,
                f64::from(bin.world_input.apparent_roughness()),
            ),
        }
    }

    fn shade(
        &self,
        warnings: &RefCell<Vec<serde_json::Value>>,
        triangle: &Triangle,
        hit: &SurfaceHit<'_>,
    ) -> Result<SurfaceSample> {
        let material = self.geometry.materials.get(triangle.material_index);
        let prepared = self.geometry.prepared.get(triangle.material_index);
        let object = object_for_index(self.geometry.objects, triangle.object_index);
        if object.is_some_and(|object| !object.view_layer_use_solid) {
            return Ok(SurfaceSample {
                color: [0.0; 4],
                has_surface: false,
                transparent_tint: [1.0; 3],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
                eevee_dither: None,
            });
        }
        if matches!(self.mode, Mode::Beauty)
            && prepared.is_some_and(PreparedMaterial::is_volume_only)
        {
            return Ok(SurfaceSample {
                color: [0.0; 4],
                has_surface: false,
                transparent_tint: [1.0; 3],
                holdout_weight: 0.0,
                subsurface: None,
                eevee_raytrace: None,
                eevee_dither: None,
            });
        }
        let graph_shading = if triangle.lit_per_pixel {
            material
                .zip(prepared)
                .map(|(material, prepared)| {
                    self.graph_shading(warnings, triangle, material, prepared, hit)
                })
                .transpose()?
        } else {
            None
        };
        let fallback;
        let closures: &[ShaderClosure] = if let Some(shading) = &graph_shading {
            &shading.closures
        } else {
            let alpha = material.map_or(1.0, |material| material.base_color[3]);
            let base = [
                triangle.color[0],
                triangle.color[1],
                triangle.color[2],
                alpha,
            ];
            let mut params = shading_params(material, base, hit.normal);
            if matches!(self.mode, Mode::Beauty) && hit.back_facing {
                // EEVEE's `init_globals` face-forwards N and keeps the back-facing IOR context.
                params.normal = -params.normal;
                params.back_facing = true;
            }
            fallback = [ShaderClosure {
                weight: 1.0,
                params,
            }];
            &fallback
        };
        if matches!(self.mode, Mode::Beauty)
            && closures
                .iter()
                .any(|closure| closure.params.thin_wall && closure.params.subsurface_weight > 0.0)
        {
            let warning = serde_json::json!({
                "code": "UNSUPPORTED_FEATURE",
                "feature_id": "shader.principled.thin_wall",
                "message": "Realtime EEVEE Thin Wall subsurface lighting is approximate; Blender 5.2.2's diffuse/translucent closure and thickness-dependent direct/probe behavior are not fully matched",
            });
            let mut recorded = warnings.borrow_mut();
            if !recorded.contains(&warning) {
                recorded.push(warning);
            }
        }
        if matches!(self.mode, Mode::Beauty)
            && self.math_semantics == crate::node_kernels::MathSemantics::Eevee
            && object.is_some_and(|object| object.is_holdout)
        {
            let tint = beauty_tint(
                material,
                surface_tint(closures),
                !self.eevee_colored_transparency,
            );
            let transparency = if material.is_some_and(|material| material.alpha_mode == "blend") {
                let opacity =
                    (opaque_share(closures) + eevee_holdout_weight(closures)).clamp(0.0, 1.0);
                (1.0 - opacity) * (tint[0] + tint[1] + tint[2]) / 3.0
            } else {
                0.0
            };
            return Ok(SurfaceSample {
                color: [0.0; 4],
                has_surface: true,
                transparent_tint: tint,
                holdout_weight: (1.0 - transparency).clamp(0.0, 1.0),
                subsurface: None,
                eevee_raytrace: None,
                eevee_dither: None,
            });
        }
        if matches!(self.mode, Mode::Solid) {
            // A back face is lit as its front: Workbench turns the normal toward the viewer
            // before it encodes it (`workbench_normal_encode`, `front_face`).
            let facing = if hit.view.dot(hit.geometric) < 0.0 {
                -hit.normal
            } else {
                hit.normal
            };
            let viewport = material.and_then(|material| material.viewport_display.as_ref());
            // Workbench shades a material with its viewport display values when present; Potter-native
            // materials have none and keep their own (`workbench_engine.cc` `get_material`).
            let dominant =
                heaviest(closures, |closure| closure.weight).map(|closure| &closure.params);
            let (albedo, roughness, metallic) = match viewport {
                Some(viewport) => (
                    DVec3::new(
                        viewport.diffuse_color[0],
                        viewport.diffuse_color[1],
                        viewport.diffuse_color[2],
                    ),
                    viewport.roughness,
                    viewport.metallic,
                ),
                None => dominant.map_or((DVec3::ZERO, 0.0, 0.0), |params| {
                    (base_albedo(params), params.roughness, params.metallic)
                }),
            };
            // Workbench transparency reads viewport alpha (`workbench_private.hh`, `Material::is_transparent`), not the
            // node tree's; a Potter-native material without viewport values keeps its opaque share.
            let alpha =
                viewport.map_or(opaque_share(closures), |viewport| viewport.diffuse_color[3]);
            let lit = studio::shade(
                albedo,
                roughness,
                metallic,
                self.view.direction_to_view(facing),
                self.view.direction_to_view(hit.view),
            );
            return Ok(SurfaceSample::opaque([lit.x, lit.y, lit.z, alpha]));
        }
        let shading_normal = if matches!(self.mode, Mode::Beauty)
            && self.math_semantics == crate::node_kernels::MathSemantics::Eevee
            && hit.back_facing
        {
            -hit.normal
        } else {
            hit.normal
        };
        let thickness = object.map_or(0.0, |object| object.thickness);
        // Select BLENDED from the source Material before graph opacity is routed through Triangle coverage.
        // Other EEVEE surfaces use deferred resolution; Triangle retains its separate mask semantics.
        let gbuffer_route = if matches!(self.mode, Mode::Beauty)
            && self.math_semantics == crate::node_kernels::MathSemantics::Eevee
            && material.is_none_or(|material| material.alpha_mode != "blend")
        {
            Some(eevee_gbuffer_route(
                graph_shading.as_ref(),
                closures,
                hit,
                thickness,
            )?)
        } else {
            None
        };
        let camera_matrices = hit
            .camera_matrices
            .map(|matrices| (&matrices.0, &matrices.1))
            .or_else(|| self.view.camera_matrices());
        let trace_geometry = TraceGeometry::with_warnings(self.geometry, warnings);
        let occluders = ShadowOccluders::new(
            self.doc,
            &trace_geometry,
            self.images,
            &self.evaluator_workspace,
            camera_matrices,
        );
        let shadow = self.geometry.shadows.map(|shadows| ShadowScene {
            bvh: &shadows.bvh,
            triangles: self.geometry.triangles,
            objects: self.geometry.objects,
            prepared_materials: self.geometry.prepared,
            has_volume_only: shadows.has_volume_only,
            point_maps: &shadows.point_maps,
            occluders: shadows.transparent_casters.then_some(&occluders),
        });
        let needs_world_probe = self.math_semantics != crate::node_kernels::MathSemantics::Eevee
            || closures.iter().any(|closure| {
                let features = closure.params.eevee_gbuffer_features();
                features.has_diffuse()
                    || features.has_reflection()
                    || features.has_refraction()
                    || features.has_subsurface()
                    || features.has_translucent()
                    || features.has_clearcoat()
            });
        let world = if needs_world_probe {
            self.world_probe()?
        } else {
            None
        };
        let world_sun = if needs_world_probe {
            self.world_sun(warnings)?.light.as_ref()
        } else {
            None
        };
        let mut color = shade(&SurfaceShadeContext {
            doc: self.doc,
            snapshot: self.snapshot,
            receiver: Receiver {
                point: hit.point,
                shadow_point: hit.shadow_point,
                geometric: hit.geometric,
                view: hit.view,
            },
            normal: shading_normal,
            closures,
            mode: self.mode,
            shadow,
            object_thickness: thickness,
            world,
            world_sun,
            gbuffer: gbuffer_route.as_ref(),
            eevee_raytrace_feedback: hit.eevee_raytrace_feedback,
            material_refraction: material
                .map_or(Some(false), |material| material.use_raytrace_refraction),
        })?;
        let holdout_weight = eevee_holdout_weight(closures);
        if gbuffer_route.is_none() {
            // Forward and native visualization samples carry their own Holdout association before raster composition.
            let radiance_share = 1.0 - holdout_weight;
            for channel in &mut color[..3] {
                *channel *= radiance_share;
            }
        }
        let subsurface = if matches!(self.mode, Mode::Beauty) {
            eevee_subsurface_sample(
                self.doc,
                self.snapshot,
                hit.point,
                hit.shadow_point,
                shading_normal,
                hit.geometric,
                hit.view,
                closures,
                gbuffer_route.as_ref(),
                Some(&hit.eevee_raytrace_feedback),
                shadow,
                world,
                world_sun,
                triangle.object_index,
                camera_matrices,
                material.map_or(Some(false), |material| material.use_raytrace_refraction),
            )?
        } else {
            None
        };
        let eevee_raytrace = self
            .eevee_raytracing_settings
            .as_ref()
            .is_ok_and(Option::is_some)
            .then(|| {
                eevee_raytrace_sample(
                    gbuffer_route.as_ref(),
                    closures,
                    hit,
                    material.map_or(Some(false), |material| material.use_raytrace_refraction),
                )
            })
            .transpose()?;
        occluders.finish()?;
        Ok(SurfaceSample {
            color,
            has_surface: true,
            transparent_tint: beauty_tint(
                material,
                surface_tint(closures),
                self.math_semantics == crate::node_kernels::MathSemantics::Eevee
                    && !self.eevee_colored_transparency,
            ),
            holdout_weight,
            eevee_dither: gbuffer_route.as_ref().map(|route| EeveeDitherSample {
                transparency: route.gbuffer_transparency,
                alpha_reciprocal: route.gbuffer_alpha_reciprocal,
            }),
            subsurface,
            eevee_raytrace,
        })
    }

    /// The material graph evaluated at `hit`, so vertex colors and textures shade the surface.
    fn graph_shading(
        &self,
        warnings: &RefCell<Vec<serde_json::Value>>,
        triangle: &Triangle,
        material: &Material,
        prepared: &PreparedMaterial,
        hit: &SurfaceHit<'_>,
    ) -> Result<SurfaceShading> {
        let object =
            object_for_index(self.geometry.objects, triangle.object_index).ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "triangle references an unmapped object",
                )
            })?;
        let element = triangle
            .element_index
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| self.geometry.elements.get(index));
        let uv_layers = HitUvLayers {
            names: &object.uv_layers,
            corners: &triangle.layer_uvs,
            positions: triangle.positions,
            barycentric: hit.barycentric,
            render: element.map_or(object.uv_render, |element| element.uv_render),
            available_mask: element.and_then(|element| element.uv_available_mask),
        };
        let mut tables = LayerTables::new();
        let mut context = layered_context(
            self.doc,
            self.images,
            warnings,
            object,
            triangle,
            HitPose {
                point: hit.point,
                object: object_space(object, hit.point),
                normal: hit.normal,
                geometric_normal: hit.geometric_normal,
                back_facing: hit.back_facing,
                path_traced: false,
                view: hit.view,
                barycentric: hit.barycentric,
            },
            None,
            uv_layers,
            &mut tables,
        )?;
        if let Some((camera_to_world, world_to_camera)) = hit.camera_matrices {
            context.set_camera_matrices(*camera_to_world, *world_to_camera);
        } else if let Some((camera_to_world, world_to_camera)) = self.view.camera_matrices() {
            context.set_camera_matrices(*camera_to_world, *world_to_camera);
        }
        context.realtime_differentials = Some(hit.realtime_differentials);
        let mut workspace = self.evaluator_workspace.borrow_mut();
        evaluate_surface_with_workspace_semantics(
            material,
            prepared,
            &context,
            &mut workspace,
            self.math_semantics,
        )
    }
}

impl SurfaceShaderWorker for SceneShaderWorker<'_> {
    fn shade(
        &self,
        triangle: &Triangle,
        hit: &SurfaceHit<'_>,
        warnings: &RefCell<Vec<serde_json::Value>>,
    ) -> Result<SurfaceSample> {
        self.core.shade(warnings, triangle, hit)
    }

    fn needs_eevee_gbuffer_context(&self) -> bool {
        matches!(self.core.mode, Mode::Beauty)
            && self.core.math_semantics == crate::node_kernels::MathSemantics::Eevee
    }
    fn eevee_raytracing_settings(&self) -> Result<Option<EeveeRayTracingSettings>> {
        self.core.eevee_raytracing_settings.clone()
    }
    fn eevee_raytrace_layer_flags(&self) -> Result<Option<EeveeRaytraceLayerSummary>> {
        self.core.eevee_raytrace_layer_flags.clone()
    }
    fn eevee_raytrace_probe_radiance(
        &self,
        sample: &EeveeRaytraceSample,
        bin: EeveeRaytraceBin,
        world_direction: DVec3,
    ) -> Result<DVec3> {
        self.core
            .eevee_raytrace_probe_radiance(sample, bin, world_direction)
    }
    fn eevee_fast_gi_volume_irradiance(&self, _point: DVec3) -> Result<Option<FastGiSh>> {
        self.core.eevee_fast_gi_volume_irradiance()
    }
}
/// EEVEE's default refraction thickness of an object: the smallest dimension of its bounds, in world units. EEVEE takes
/// the bounds' size along each axis times the object's scale along it, then the minimum (`eevee_shader.cc`,
/// `nodetree_thickness`, `ws_dimensions`). It is a sphere model's diameter (`eevee_thickness_lib.bsl.hh`).
fn transmission_thickness(mesh: &crate::geom::Mesh, matrix: DMat4) -> f64 {
    let Some(first) = mesh.vertices.first() else {
        return 0.0;
    };
    let (mut min, mut max) = (first.co, first.co);
    for vertex in &mesh.vertices {
        min = min.min(vertex.co);
        max = max.max(vertex.co);
    }
    let scale = DVec3::new(
        matrix.x_axis.truncate().length(),
        matrix.y_axis.truncate().length(),
        matrix.z_axis.truncate().length(),
    );
    ((max - min) * scale).min_element().max(0.0)
}

/// Face-corner index of each triangle vertex; the UV and shading-normal lookups share it.
fn triangle_corners(face: &crate::geom::Face, triangle: &[u32; 3]) -> [Option<usize>; 3] {
    triangle.map(|vertex_id| {
        face.vertices
            .iter()
            .position(|face_vertex| *face_vertex == vertex_id)
    })
}

/// Corner UVs of every UV layer for a triangle whose vertices sit at face corners `corners`, in
/// layer order. A layer with no entry for the face samples canonical UV `[0, 1]` at each corner.
fn triangle_layer_uvs(
    store: &UvStore,
    face_index: usize,
    corners: [Option<usize>; 3],
) -> Vec<[[f64; 2]; 3]> {
    store
        .layers
        .iter()
        .map(|layer| {
            let values = layer
                .faces
                .get(face_index)
                .and_then(|entry| entry.as_deref());
            corners.map(|corner| {
                corner
                    .and_then(|corner| values.and_then(|values| values.get(corner)))
                    .copied()
                    .unwrap_or([0.0, 1.0])
            })
        })
        .collect()
}

/// Corner colors of `triangle` in each carried color layer, in carried order. Point layers read the
/// vertex values, face layers copy their face value, and corner layers read face-corner values.
fn triangle_layer_colors(
    layers: &[&ColorLayer],
    face_index: usize,
    corner_base: usize,
    corners: [Option<usize>; 3],
    vertices: [u32; 3],
    vertex_slots: &BTreeMap<u32, usize>,
) -> Vec<[[f64; 4]; 3]> {
    const BLACK: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
    layers
        .iter()
        .map(|layer| match layer.domain {
            ColorDomain::Point => vertices.map(|vertex| {
                vertex_slots
                    .get(&vertex)
                    .and_then(|slot| layer.values.get(*slot))
                    .copied()
                    .unwrap_or(BLACK)
            }),
            ColorDomain::Face => layer
                .values
                .get(face_index)
                .copied()
                .map_or([BLACK; 3], |color| [color; 3]),
            ColorDomain::Corner => corners.map(|corner| {
                corner
                    .and_then(|corner| layer.values.get(corner_base + corner))
                    .copied()
                    .unwrap_or(BLACK)
            }),
        })
        .collect()
}

fn mesh_geometry_attributes(mesh: &Mesh) -> Vec<GeometryAttribute> {
    let mut attributes = Vec::new();
    for (key, value) in &mesh.attributes {
        let Some(name) = attribute_keys::user_name(key) else {
            continue;
        };
        let domain = value
            .get("domain")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let data_type = value
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        attributes.push(GeometryAttribute {
            name: name.to_owned(),
            domain: domain.to_owned(),
            data_type: data_type.to_owned(),
            values: point_attribute_values(mesh, value.get("values"), domain, data_type),
        });
    }
    for value in mesh
        .attributes
        .get("blender_attributes")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(name) = value.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if attributes.iter().any(|attribute| attribute.name == name) {
            continue;
        }
        let domain = value
            .get("domain")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let data_type = value
            .get("data_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        attributes.push(GeometryAttribute {
            name: name.to_owned(),
            domain: domain.to_ascii_lowercase(),
            data_type: data_type.to_owned(),
            values: point_attribute_values(mesh, value.get("values"), domain, data_type),
        });
    }
    attributes
}

fn point_attribute_values(
    mesh: &Mesh,
    values: Option<&serde_json::Value>,
    domain: &str,
    data_type: &str,
) -> GeometryAttributeValues {
    if !matches!(domain, "point" | "POINT") {
        return GeometryAttributeValues::Unsupported;
    }
    let Some(values) = values else {
        return GeometryAttributeValues::Unsupported;
    };
    let value_at = |slot: usize| {
        if let Some(values) = values.as_array() {
            if values.len() == mesh.vertices.len() {
                values.get(slot)
            } else {
                None
            }
        } else if let Some(values) = values.as_object() {
            let vertex = mesh.vertices.get(slot)?;
            values.get(&format!("v{}", vertex.id))
        } else {
            None
        }
    };
    match data_type {
        "float" | "FLOAT" => {
            let mut result = Vec::with_capacity(mesh.vertices.len());
            for slot in 0..mesh.vertices.len() {
                let Some(value) = value_at(slot)
                    .and_then(serde_json::Value::as_f64)
                    .filter(|value| value.is_finite())
                else {
                    return GeometryAttributeValues::Unsupported;
                };
                result.push(value);
            }
            GeometryAttributeValues::Float(result)
        }
        "int" | "INT" => {
            let mut result = Vec::with_capacity(mesh.vertices.len());
            for slot in 0..mesh.vertices.len() {
                let Some(value) = value_at(slot)
                    .and_then(serde_json::Value::as_i64)
                    .and_then(|value| i32::try_from(value).ok())
                else {
                    return GeometryAttributeValues::Unsupported;
                };
                result.push(value);
            }
            GeometryAttributeValues::Integer(result)
        }
        "float3" | "FLOAT_VECTOR" => {
            let mut result = Vec::with_capacity(mesh.vertices.len());
            for slot in 0..mesh.vertices.len() {
                let Some(values) = value_at(slot).and_then(serde_json::Value::as_array) else {
                    return GeometryAttributeValues::Unsupported;
                };
                if values.len() != 3 {
                    return GeometryAttributeValues::Unsupported;
                }
                let mut vector = [0.0; 3];
                for (component, value) in vector.iter_mut().zip(values) {
                    let Some(value) = value.as_f64().filter(|value| value.is_finite()) else {
                        return GeometryAttributeValues::Unsupported;
                    };
                    *component = value;
                }
                result.push(vector);
            }
            GeometryAttributeValues::Vector(result)
        }
        _ => GeometryAttributeValues::Unsupported,
    }
}

#[derive(Default)]
struct ColorLayerReferences {
    names: BTreeSet<String>,
    uses_render_layer: bool,
    requires_hit_shading: bool,
    reads_colors: bool,
}

struct ColorSlotReferences {
    names: BTreeSet<String>,
    render_layer: Option<usize>,
    available_mask: Option<Arc<[u64]>>,
    requires_hit_shading: bool,
    reads_colors: bool,
}

fn color_layer_references(doc: &SceneDoc, material: &Id) -> ColorLayerReferences {
    let mut references = ColorLayerReferences::default();
    let Some(material_value) = doc.materials.get(material) else {
        return references;
    };
    references.requires_hit_shading = material_value.node_tree.is_some()
        || material_value.base_color_texture.is_some()
        || material_value.roughness_texture.is_some()
        || material_value.metallic_texture.is_some()
        || material_value.normal_texture.is_some();
    let graph = material_value
        .node_tree
        .as_ref()
        .and_then(|graph| doc.node_groups.get(graph));
    for node in graph.into_iter().flat_map(|graph| graph.nodes.values()) {
        if node.node_type == "ShaderNodeVertexColor" {
            references.reads_colors = true;
            let name = node
                .properties
                .get("layer_name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if name.is_empty() {
                references.uses_render_layer = true;
            } else {
                references.names.insert(name.to_owned());
            }
            continue;
        }
        if node.node_type == "ShaderNodeAttribute"
            && let Some(name) = node
                .properties
                .get("attribute_name")
                .or_else(|| node.properties.get("name"))
                .and_then(serde_json::Value::as_str)
        {
            references.reads_colors = true;
            if !name.is_empty() {
                references.names.insert(name.to_owned());
            }
        }
    }
    references
}

fn source_has_color_layer(mask: Option<&[u64]>, layer: usize) -> bool {
    mask.is_none_or(|mask| {
        mask.get(layer / u64::BITS as usize)
            .is_some_and(|bits| bits & (1_u64 << (layer % u64::BITS as usize)) != 0)
    })
}

/// Carried source layers and each slot's render-layer position within that carried list.
fn carried_color_layers(
    store: &ColorStore,
    slots: &[ColorSlotReferences],
) -> (Vec<usize>, Vec<Option<usize>>) {
    let mut carried = BTreeSet::new();
    for slot in slots {
        for name in &slot.names {
            if let Some(layer) = store.index_of(name)
                && source_has_color_layer(slot.available_mask.as_deref(), layer)
            {
                carried.insert(layer);
            }
        }
        if let Some(layer) = slot.render_layer {
            carried.insert(layer);
        }
    }
    let carried = carried.into_iter().collect::<Vec<_>>();
    let render_layers = slots
        .iter()
        .map(|slot| {
            slot.render_layer
                .and_then(|layer| carried.iter().position(|&index| index == layer))
        })
        .collect();
    (carried, render_layers)
}

/// Color values for a hit, filled in place so that no memory is allocated per ray.
pub(super) struct LayerTables<'a> {
    color: [(&'a str, [f64; 4]); MAX_COLOR_LAYERS],
}

impl LayerTables<'_> {
    pub(super) fn new() -> Self {
        Self {
            color: [("", [0.0; 4]); MAX_COLOR_LAYERS],
        }
    }
}

/// Where one hit lies, for [`layered_context`].
#[derive(Clone, Copy)]
pub(super) struct HitPose {
    pub(super) point: DVec3,
    /// The hit point in the object's local space.
    pub(super) object: DVec3,
    pub(super) normal: DVec3,
    pub(super) geometric_normal: DVec3,
    pub(super) back_facing: bool,
    pub(super) path_traced: bool,
    /// Unit direction from the hit back toward the camera.
    pub(super) view: DVec3,
    pub(super) barycentric: [f64; 3],
}

/// The graph context of one hit on `triangle`, with UV layers borrowed and color layers filled in
/// `tables`.
pub(super) fn layered_context<'a>(
    doc: &'a SceneDoc,
    images: &'a BTreeMap<String, ImageData>,
    warnings: &'a RefCell<Vec<serde_json::Value>>,
    object: &'a ObjectRecord,
    triangle: &'a Triangle,
    pose: HitPose,
    uv_override: Option<[f64; 2]>,
    uv_layers: HitUvLayers<'a>,
    tables: &'a mut LayerTables<'a>,
) -> Result<HitContext<'a>> {
    let material_colors = object.material_colors.get(triangle.material_index);
    let available_mask = material_colors.and_then(|colors| colors.available_mask.as_deref());
    let color_maps = fill_color_layers(
        &mut tables.color,
        object
            .color_layers
            .iter()
            .zip(&object.color_layer_indices)
            .zip(&triangle.layer_colors)
            .filter(|((_, source_layer), _)| source_has_color_layer(available_mask, **source_layer))
            .map(|((name, _), corners)| (name.as_str(), blend_color(corners, pose.barycentric))),
    )?;
    let color_render = material_colors
        .and_then(|colors| colors.render_layer)
        .and_then(|index| triangle.layer_colors.get(index))
        .map(|corners| blend_color(corners, pose.barycentric));
    let (geometry_vertices, geometry_corner_weights) = triangle.geometry_attribute_frame();
    let mut context = HitContext::new(&doc.node_groups)
        .with_images(images)
        .with_uv_layers(HitUvLayers {
            barycentric: pose.barycentric,
            ..uv_layers
        })
        .with_uv_tangents(&triangle.layer_tangents)
        .with_color_layers(color_maps, color_render)
        .with_geometry_attributes(
            &object.geometry_attributes,
            geometry_vertices,
            geometry_corner_weights,
            pose.barycentric,
        )
        .with_warnings(warnings)
        .with_path_traced(false);
    context.uv = uv_override.unwrap_or_else(|| blend_uv(&triangle.uv, pose.barycentric));
    context = context.with_generated_corners(triangle.generated, pose.barycentric);
    context.position = pose.point;
    context.object = pose.object;
    context.object_color = object.object_color;
    context.object_pass_index = object.object_pass_index;
    context.object_random_number = object.object_random_number;
    context.material_pass_index = material_colors.and_then(|colors| colors.material_pass_index);
    context.normal = pose.normal;
    context.geometric_normal = pose.geometric_normal;
    context.path_traced = pose.path_traced;
    if !pose.path_traced {
        context = context.with_realtime_position_f32(triangle.positions, pose.barycentric);
    }
    context.light_falloff_distance = None;
    context.view_direction = pose.view;
    context.object_to_world = object.object_to_world;
    context.world_to_object = object.world_to_local.unwrap_or(DMat4::IDENTITY);
    context.back_facing = pose.back_facing;
    Ok(context)
}

/// `point` in the object's local space, through the inverse of its evaluated world matrix.
pub(super) fn object_space(object: &ObjectRecord, point: DVec3) -> DVec3 {
    match object.world_to_local {
        Some(world_to_local) => world_to_local.transform_point3(point),
        None => point,
    }
}

/// The inverse of an evaluated world matrix, or `None` when that inverse is not finite.
fn object_inverse(world_matrix: DMat4) -> Option<DMat4> {
    let inverse = world_matrix.inverse();
    inverse
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite())
        .then_some(inverse)
}
pub(super) fn object_info_color(node: &Node) -> [f64; 4] {
    let default = [1.0; 4];
    let Some(values) = node
        .properties
        .get("blender_object_metadata")
        .and_then(|metadata| metadata.get("rna_properties"))
        .and_then(|properties| properties.get("color"))
        .and_then(serde_json::Value::as_array)
        .filter(|values| values.len() >= 3)
    else {
        return default;
    };
    let Some(red) = values.first().and_then(serde_json::Value::as_f64) else {
        return default;
    };
    let Some(green) = values.get(1).and_then(serde_json::Value::as_f64) else {
        return default;
    };
    let Some(blue) = values.get(2).and_then(serde_json::Value::as_f64) else {
        return default;
    };
    let alpha = values
        .get(3)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    let color = [red, green, blue, alpha];
    if color.iter().all(|channel| channel.is_finite()) {
        color
    } else {
        default
    }
}

pub(super) fn object_info_pass_index(node: &Node) -> f64 {
    node.properties
        .get("blender_object_metadata")
        .and_then(|metadata| metadata.get("rna_properties"))
        .and_then(|properties| properties.get("pass_index"))
        .and_then(serde_json::Value::as_i64)
        .map_or(0.0, |pass_index| {
            i32::try_from(pass_index).map_or(0.0, f64::from)
        })
}

/// Per-corner values for a triangle, or `None` when any corner is not on the face.
fn corner_values<T: Copy>(values: &[T], corners: [Option<usize>; 3]) -> Option<[T; 3]> {
    Some([
        values.get(corners[0]?).copied()?,
        values.get(corners[1]?).copied()?,
        values.get(corners[2]?).copied()?,
    ])
}

fn read_vertex_number(mesh: &crate::geom::Mesh, name: &str, vertex_id: u32) -> Option<f64> {
    let key = format!("v{vertex_id}");
    let value = mesh
        .attributes
        .get(name)?
        .get("values")?
        .get(key)?
        .as_f64()?;
    value.is_finite().then_some(value)
}

/// The opaque share of a hit: each closure's weight times its alpha, summed and clamped to one.
fn opaque_share(closures: &[ShaderClosure]) -> f64 {
    closures
        .iter()
        .map(|closure| closure.weight * closure.params.alpha.clamp(0.0, 1.0))
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// EEVEE's physical transparent tint excludes Holdout, whose closure has zero transmittance and a separate film-matte
/// share (`gpu_shader_material_holdout.glsl`, `node_holdout`).
fn surface_tint(closures: &[ShaderClosure]) -> [f64; 3] {
    if closures.is_empty() {
        return [1.0; 3];
    }
    let holdout = eevee_holdout_weight(closures);
    let physical_opacity = (opaque_share(closures) + holdout).clamp(0.0, 1.0);
    weighted_tint(
        closures
            .iter()
            .filter(|closure| !matches!(closure.params.model, BsdfModel::Holdout))
            .map(|closure| {
                (
                    closure.weight,
                    closure.params.alpha,
                    closure.params.transparent_tint,
                )
            }),
        physical_opacity,
    )
}

/// The tint for `(weight, alpha, tint)` closures that together have `opacity`: the summed transmittance
/// `sum w (1 - alpha) tint` over the pass chance `1 - opacity`. A surface with no pass chance passes nothing, so its tint
/// is white.
fn weighted_tint(terms: impl Iterator<Item = (f64, f64, [f64; 3])>, opacity: f64) -> [f64; 3] {
    let chance = 1.0 - opacity;
    if chance <= 0.0 {
        return [1.0; 3];
    }
    let mut transmittance = [0.0; 3];
    for (weight, alpha, tint) in terms {
        let passed = weight * (1.0 - alpha.clamp(0.0, 1.0));
        for (sum, channel) in transmittance.iter_mut().zip(tint) {
            *sum += passed * channel;
        }
    }
    transmittance.map(|sum| sum / chance)
}

fn base_albedo(params: &BsdfParams) -> DVec3 {
    DVec3::new(
        params.base_color[0],
        params.base_color[1],
        params.base_color[2],
    )
}

fn eevee_gbuffer_candidates(
    closures: &[ShaderClosure],
    normal: DVec3,
    view: DVec3,
) -> Result<Vec<EeveeGBufferCandidate>> {
    let mut candidates = Vec::with_capacity(closures.len().saturating_mul(5));
    for (source_closure_index, closure) in closures.iter().enumerate() {
        let params = &closure.params;
        // EEVEE Hair preview's GPU Weight can be sourced from Secondary Reflection; graph weight still governs coverage.
        let weight =
            params.eevee_diffuse_weight.unwrap_or(closure.weight) * params.alpha.clamp(0.0, 1.0);
        if params.standalone_glass {
            if let Some(glass) = eevee_transmission::glass_closures(
                params,
                params.normal,
                view,
                DVec3::ONE,
                Thickness::None,
            ) {
                push_eevee_gbuffer_candidate(
                    &mut candidates,
                    source_closure_index,
                    EeveeGBufferLobe::Reflection,
                    weight,
                    glass.reflection,
                );
                push_eevee_gbuffer_candidate(
                    &mut candidates,
                    source_closure_index,
                    if params.thin_wall {
                        EeveeGBufferLobe::ThinRefraction
                    } else {
                        EeveeGBufferLobe::Refraction
                    },
                    weight,
                    glass.transmission,
                );
            }
            continue;
        }
        let features = params.eevee_gbuffer_features();
        if features.compile_bin_count() == 0 {
            continue;
        }
        if matches!(params.model, BsdfModel::Principled)
            && features.has_diffuse()
            && !features.has_reflection()
        {
            push_eevee_gbuffer_candidate(
                &mut candidates,
                source_closure_index,
                EeveeGBufferLobe::Diffuse,
                weight,
                base_albedo(params),
            );
            continue;
        }

        match params.model {
            BsdfModel::Principled => eevee_principled_gbuffer_candidates(
                &mut candidates,
                source_closure_index,
                closure,
                normal,
                view,
            ),
            BsdfModel::Metallic(_) => {
                let metallic_normal = normalize_fallback(params.normal, normal);
                let color =
                    eevee_metallic_brdf(params, metallic_normal, view).ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "EEVEE metallic closure has no conductor Fresnel inputs",
                        )
                    })?;
                push_eevee_gbuffer_candidate(
                    &mut candidates,
                    source_closure_index,
                    EeveeGBufferLobe::Reflection,
                    weight,
                    color,
                );
            }
            BsdfModel::Refraction(_) => push_eevee_gbuffer_candidate(
                &mut candidates,
                source_closure_index,
                if params.thin_wall {
                    EeveeGBufferLobe::ThinRefraction
                } else {
                    EeveeGBufferLobe::Refraction
                },
                weight,
                base_albedo(params),
            ),
            BsdfModel::Subsurface => {
                if eevee_subsurface_radius(params).is_some() {
                    push_eevee_gbuffer_candidate(
                        &mut candidates,
                        source_closure_index,
                        EeveeGBufferLobe::Subsurface,
                        weight,
                        base_albedo(params),
                    );
                }
            }
            BsdfModel::Translucent => push_eevee_gbuffer_candidate(
                &mut candidates,
                source_closure_index,
                EeveeGBufferLobe::Translucent,
                weight,
                base_albedo(params),
            ),
            BsdfModel::Sheen(_) => push_eevee_gbuffer_candidate(
                &mut candidates,
                source_closure_index,
                EeveeGBufferLobe::Diffuse,
                weight,
                base_albedo(params).max(DVec3::ZERO),
            ),
            BsdfModel::Toon(_)
            | BsdfModel::Holdout
            | BsdfModel::Hair(_)
            | BsdfModel::PrincipledHair(_)
            | BsdfModel::RayPortal(_) => {}
        }
    }
    Ok(candidates)
}

fn push_eevee_gbuffer_candidate(
    candidates: &mut Vec<EeveeGBufferCandidate>,
    source_closure_index: usize,
    lobe: EeveeGBufferLobe,
    weight: f64,
    color: DVec3,
) {
    candidates.push(EeveeGBufferCandidate {
        source_closure_index,
        lobe,
        weight: weight as f32,
        color: color.to_array().map(|channel| channel as f32),
    });
}

fn eevee_principled_gbuffer_candidates(
    candidates: &mut Vec<EeveeGBufferCandidate>,
    source_closure_index: usize,
    closure: &ShaderClosure,
    normal: DVec3,
    view: DVec3,
) {
    let params = &closure.params;
    let features = params.eevee_gbuffer_features();
    let mut weight = closure.weight * params.alpha.clamp(0.0, 1.0);
    let normal = normalize_fallback(params.normal, normal);
    let coat_normal = normalize_fallback(params.coat_normal.unwrap_or(normal), normal);
    let n_dot_v = normal.dot(view);
    let base_color = base_albedo(params).max(DVec3::ZERO);
    let clamped_base_color = base_color.min(DVec3::ONE);
    let mut coat_tint = DVec3::from_array(params.coat_tint).max(DVec3::ZERO);

    let mut sheen_data_color = DVec3::ZERO;
    let sheen_weight = params.sheen_weight.max(0.0);
    if sheen_weight > 0.0 {
        let sheen_normal = if params.coat_weight > 0.0 {
            normalize_fallback(
                normal.lerp(coat_normal, params.coat_weight.clamp(0.0, 1.0)),
                normal,
            )
        } else {
            normal
        };
        let sheen_nv = sheen_normal.dot(view).clamp(0.0, 1.0);
        let sheen_color = DVec3::from_array(params.sheen_tint).max(DVec3::ZERO)
            * (sheen_weight * eevee_principled_sheen(sheen_nv, params.sheen_roughness));
        sheen_data_color = sheen_color * weight;
        weight *= (1.0 - sheen_color.max_element()).max(0.0);
    }

    let coat_weight = params.coat_weight.max(0.0);
    let coat_roughness = params.coat_roughness.clamp(0.0, 1.0);
    let coat_ior = params.coat_ior.max(1.0);
    if features.has_clearcoat() && coat_weight > 0.0 {
        let reflectance = eevee_transmission::eevee_bsdf_lut(
            DVec3::splat(dielectric_f0(coat_ior, 0.5)),
            DVec3::ONE,
            coat_normal.dot(view),
            coat_roughness,
            coat_ior,
            false,
        )
        .x;
        push_eevee_gbuffer_candidate(
            candidates,
            source_closure_index,
            EeveeGBufferLobe::Clearcoat,
            weight * coat_weight * reflectance,
            DVec3::ONE,
        );
        weight *= (1.0 - reflectance * coat_weight).max(0.0);
        if coat_tint != DVec3::ONE {
            let path_length = coat_ior / (coat_ior * coat_ior - (1.0 - n_dot_v * n_dot_v)).sqrt();
            let tint = coat_tint.powf(path_length);
            coat_tint = DVec3::ONE.lerp(tint, coat_weight.clamp(0.0, 1.0));
        }
    } else {
        coat_tint = DVec3::ONE;
    }

    let metallic = params.metallic.clamp(0.0, 1.0);
    let roughness = params.roughness.clamp(0.0, 1.0);
    let transmission_weight = params.transmission.clamp(0.0, 1.0);
    let mut reflection_color = DVec3::ZERO;
    if metallic > 0.0 {
        let metallic_reflectance =
            eevee_principled_metallic_brdf(params, clamped_base_color, normal, view);
        reflection_color = metallic_reflectance * (weight * metallic);
        weight *= (1.0 - metallic).max(0.0);
    }

    if features.has_refraction() && transmission_weight > 0.0 {
        if let Some(glass) =
            eevee_transmission::glass_closures(params, normal, view, DVec3::ONE, Thickness::None)
        {
            let split_weight = (1.0 - metallic) * transmission_weight;
            let reflectance = glass.reflection / split_weight;
            let transmittance = glass.transmission / split_weight;
            let lobe = if params.thin_wall {
                EeveeGBufferLobe::ThinRefraction
            } else {
                EeveeGBufferLobe::Refraction
            };
            push_eevee_gbuffer_candidate(
                candidates,
                source_closure_index,
                lobe,
                weight * transmission_weight,
                transmittance * coat_tint,
            );
            reflection_color += reflectance * (weight * transmission_weight);
        }
        weight *= (1.0 - transmission_weight).max(0.0);
    }

    let eta = eevee_principled_specular_eta(params);
    let f0 = (DVec3::splat(dielectric_f0(
        params.ior.max(1.0e-5),
        params.specular_ior_level,
    )) * DVec3::from_array(params.specular_tint).max(DVec3::ZERO))
    .clamp(DVec3::ZERO, DVec3::ONE);
    let reflectance = eevee_transmission::eevee_bsdf_lut(
        f0,
        DVec3::ONE,
        n_dot_v,
        roughness,
        eta,
        matches!(params.glass_distribution, GlassDistribution::MultiGgx),
    );
    push_eevee_gbuffer_candidate(
        candidates,
        source_closure_index,
        EeveeGBufferLobe::Reflection,
        1.0,
        (reflection_color + reflectance * weight) * coat_tint,
    );
    weight *= (1.0 - reflectance.max_element()).max(0.0);

    let mut diffuse_weight = 0.0;
    let subsurface_weight = params.subsurface_weight.clamp(0.0, 1.0);
    if subsurface_weight > 0.0 {
        if params.thin_wall {
            diffuse_weight = subsurface_weight
                * weight
                * (0.5 * (1.0 - params.subsurface_anisotropy)).clamp(0.0, 1.0);
            if features.has_translucent() {
                let anisotropy_plus_one = 1.0 + params.subsurface_anisotropy;
                push_eevee_gbuffer_candidate(
                    candidates,
                    source_closure_index,
                    EeveeGBufferLobe::Translucent,
                    subsurface_weight * weight * (0.5 * anisotropy_plus_one).clamp(0.0, 1.0),
                    base_color * coat_tint,
                );
            }
        } else if features.has_subsurface() && eevee_subsurface_radius(params).is_some() {
            let color =
                clamped_base_color * (subsurface_weight * weight) * coat_tint + sheen_data_color;
            push_eevee_gbuffer_candidate(
                candidates,
                source_closure_index,
                EeveeGBufferLobe::Subsurface,
                1.0,
                color,
            );
        }
        weight *= (1.0 - subsurface_weight).max(0.0);
    }
    if features.has_diffuse() {
        push_eevee_gbuffer_candidate(
            candidates,
            source_closure_index,
            EeveeGBufferLobe::Diffuse,
            1.0,
            base_color * (diffuse_weight + weight) * coat_tint + sheen_data_color,
        );
    }
}

fn eevee_principled_metallic_brdf(
    params: &BsdfParams,
    base_color: DVec3,
    normal: DVec3,
    view: DVec3,
) -> DVec3 {
    let f0 = base_color.clamp(DVec3::ZERO, DVec3::ONE);
    let f82 = DVec3::from_array(params.specular_tint)
        .max(DVec3::ZERO)
        .min(DVec3::ONE);
    eevee_brdf_f82_tint_lut(
        f0,
        f82,
        normal.dot(view),
        params.roughness,
        matches!(params.glass_distribution, GlassDistribution::MultiGgx),
    )
}

fn eevee_principled_specular_eta(params: &BsdfParams) -> f64 {
    let ior = params.ior.max(1.0e-5);
    let mut f0 = dielectric_f0(ior, 0.5);
    let mut eta = ior;
    let specular_ior_level = params.specular_ior_level.max(0.0);
    if specular_ior_level.partial_cmp(&0.5) != Some(std::cmp::Ordering::Equal) {
        f0 *= 2.0 * specular_ior_level;
        let f = f0.clamp(0.0, 0.99).sqrt();
        eta = (-f - 1.0) / (f - 1.0);
        if ior < 1.0 {
            eta = 1.0 / eta;
        }
    }
    eta
}

fn eevee_principled_sheen(normal_dot_view: f64, roughness: f64) -> f64 {
    let nv = normal_dot_view;
    let roughness = roughness.clamp(0.0, 1.0);
    let denominator = 35.6694 * roughness * roughness - 24.4269 * roughness * nv - 0.1405 * nv * nv
        + 6.1211 * roughness
        + 0.28105 * nv
        - 0.1405;
    let numerator = 58.5299 * roughness * roughness - 85.0941 * roughness * nv
        + 9.8955 * nv * nv
        + 1.9250 * roughness
        + 74.2268 * nv
        - 0.2246;
    (denominator / numerator).clamp(0.0, 1.0)
}

fn normalize_fallback(value: DVec3, fallback: DVec3) -> DVec3 {
    let normalized = value.normalize_or_zero();
    if normalized == DVec3::ZERO {
        fallback.normalize_or_zero()
    } else {
        normalized
    }
}
fn eevee_gbuffer_alpha(closures: &[ShaderClosure]) -> (f32, f32, f32) {
    let mut transmittance = [0.0_f32; 3];
    for closure in closures {
        let params = &closure.params;
        if matches!(params.model, BsdfModel::Holdout) {
            continue;
        }
        let (alpha, tint) = if matches!(params.model, BsdfModel::RayPortal(_)) {
            (
                0.0,
                [
                    params.base_color[0],
                    params.base_color[1],
                    params.base_color[2],
                ],
            )
        } else if params.transparent_bsdf {
            (0.0, params.transparent_tint.map(|channel| channel.max(0.0)))
        } else {
            (params.alpha.clamp(0.0, 1.0), params.transparent_tint)
        };
        let weight = closure.weight as f32;
        let pass = 1.0 - alpha as f32;
        for (sum, channel) in transmittance.iter_mut().zip(tint) {
            *sum += (channel as f32 * pass) * weight;
        }
    }
    let transparency = (transmittance[0] + transmittance[1] + transmittance[2]) / 3.0;
    let alpha = 1.0 - transparency;
    let alpha_reciprocal = if alpha.is_finite() && alpha != 0.0 {
        alpha.recip()
    } else {
        0.0
    };
    (transparency, alpha, alpha_reciprocal)
}

struct EeveeGBufferRoute {
    candidates: Vec<EeveeGBufferCandidate>,
    layout: EeveeGBufferLayout,
    frag_coord: [f32; 2],
    sample_offset: [f32; 3],
    normal: DVec3,
    view: DVec3,
    thickness: Thickness,
    gbuffer_transparency: f32,
    gbuffer_alpha: f32,
    gbuffer_alpha_reciprocal: f32,
}

impl EeveeGBufferRoute {
    fn context(&self, data_layer: usize) -> EeveeGBufferContext {
        EeveeGBufferContext {
            frag_coord: self.frag_coord,
            sample_offset: self.sample_offset,
            data_layer: data_layer as u32,
        }
    }
}

struct DecodedEeveeGBufferBin<'a> {
    closure: &'a ShaderClosure,
    color: DVec3,
    normal: DVec3,
    roughness: Option<f64>,
    ior: Option<f64>,
}

fn eevee_gbuffer_route(
    shading: Option<&SurfaceShading>,
    closures: &[ShaderClosure],
    hit: &SurfaceHit<'_>,
    object_thickness: f64,
) -> Result<EeveeGBufferRoute> {
    let frag_coord = hit.gbuffer_frag_coord.ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE shading requires the raster G-buffer fragment coordinate",
        )
    })?;
    let sample_offset = hit.gbuffer_sample_offset.ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE shading requires the raster G-buffer sample offset",
        )
    })?;
    let closure_random = hit.gbuffer_closure_random.ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE shading requires the raster G-buffer closure random value",
        )
    })?;
    let (gbuffer_transparency, gbuffer_alpha, gbuffer_alpha_reciprocal) =
        eevee_gbuffer_alpha(closures);
    let normal = if hit.back_facing {
        -hit.normal
    } else {
        hit.normal
    };
    let candidates = eevee_gbuffer_candidates(closures, normal, hit.view)?;
    let layout = if let Some(shading) = shading {
        shading.eevee_gbuffer_layout(&candidates, closure_random, gbuffer_alpha_reciprocal)
    } else {
        let closure = closures.first().ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE shading has no closure for its compiled G-buffer layout",
            )
        })?;
        closure
            .params
            .eevee_gbuffer_features()
            .eevee_gbuffer_layout(&candidates, closure_random, gbuffer_alpha_reciprocal)
    }
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "EEVEE G-buffer candidates do not match the compiled material layout",
            serde_json::json!({"feature_id":"render.eevee.gbuffer.layout","error":format!("{error:?}")}),
        )
    })?;
    let thickness = layout
        .bins
        .iter()
        .take(layout.bin_count)
        .flatten()
        .find(|bin| {
            matches!(
                bin.mode,
                EeveeGBufferMode::Refraction
                    | EeveeGBufferMode::RefractionColorless
                    | EeveeGBufferMode::Translucent
                    | EeveeGBufferMode::Subsurface
                    | EeveeGBufferMode::ThinRefraction
                    | EeveeGBufferMode::ThinRefractionColorless
            )
        })
        .map_or(Thickness::None, |bin| {
            let params = &closures[bin.source_closure_index].params;
            eevee_gbuffer_thickness(Thickness::new(
                params.thickness.unwrap_or(object_thickness),
                params.thickness_slab,
            ))
        });
    Ok(EeveeGBufferRoute {
        candidates,
        layout,
        frag_coord,
        sample_offset,
        normal,
        view: hit.view,
        gbuffer_transparency,
        gbuffer_alpha,
        thickness,
        gbuffer_alpha_reciprocal,
    })
}

fn decode_eevee_gbuffer_bin<'a>(
    route: &EeveeGBufferRoute,
    closures: &'a [ShaderClosure],
    bin: EeveeGBufferBin,
) -> Result<DecodedEeveeGBufferBin<'a>> {
    let candidate = route.candidates.get(bin.candidate_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE G-buffer bin references a missing source candidate",
        )
    })?;
    let closure = closures.get(bin.source_closure_index).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE G-buffer bin references a missing source closure",
        )
    })?;
    if candidate.source_closure_index != bin.source_closure_index || candidate.lobe != bin.lobe {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE G-buffer bin does not match its selected source candidate",
        ));
    }
    let data0_context = route.context(bin.data0_layer);
    let resolved_color_scale = bin.weight * route.gbuffer_alpha_reciprocal;
    let resolved_color = DVec3::from_array(
        bin.color
            .map(|channel| f64::from(channel * resolved_color_scale)),
    );
    let parameter_context = if matches!(
        bin.mode,
        EeveeGBufferMode::Reflection
            | EeveeGBufferMode::Refraction
            | EeveeGBufferMode::ThinRefraction
    ) {
        let data1_layer = bin.data1_layer.ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE colored closure has no G-buffer parameter storage layer",
            )
        })?;
        route.context(data1_layer)
    } else {
        data0_context
    };
    let packed_color = eevee_gbuffer_closure_color(resolved_color, data0_context);
    let color = if matches!(
        bin.mode,
        EeveeGBufferMode::ReflectionColorless
            | EeveeGBufferMode::RefractionColorless
            | EeveeGBufferMode::ThinRefractionColorless
    ) {
        DVec3::splat(packed_color.z)
    } else {
        packed_color
    };
    let params = &closure.params;
    let normal = if bin.lobe == EeveeGBufferLobe::Clearcoat {
        params.coat_normal.unwrap_or(params.normal)
    } else {
        params.normal
    };
    let normal = eevee_gbuffer_normal(normalize_fallback(normal, route.normal));
    let roughness = match bin.lobe {
        EeveeGBufferLobe::Clearcoat => Some(params.coat_roughness),
        EeveeGBufferLobe::Reflection | EeveeGBufferLobe::Refraction => Some(params.roughness),
        EeveeGBufferLobe::ThinRefraction => {
            let glass = eevee_transmission::glass_closures(
                params,
                params.normal,
                route.view,
                DVec3::ONE,
                Thickness::None,
            );
            Some(
                glass
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "EEVEE thin refraction bin has no source transmission roughness",
                        )
                    })?
                    .transmission_roughness,
            )
        }
        EeveeGBufferLobe::Diffuse
        | EeveeGBufferLobe::Translucent
        | EeveeGBufferLobe::Subsurface => None,
    }
    .map(|roughness| eevee_gbuffer_roughness(roughness, parameter_context));
    let ior = (bin.lobe == EeveeGBufferLobe::Refraction)
        .then_some(params.ior.max(1.0e-5))
        .map(|ior| eevee_gbuffer_ior(ior, parameter_context));
    Ok(DecodedEeveeGBufferBin {
        closure,
        color,
        normal,
        roughness,
        ior,
    })
}

fn eevee_raytrace_sample(
    route: Option<&EeveeGBufferRoute>,
    closures: &[ShaderClosure],
    hit: &SurfaceHit<'_>,
    material_refraction: Option<bool>,
) -> Result<EeveeRaytraceSample> {
    let mut bins = [None; 3];
    if let Some(route) = route {
        for bin in route
            .layout
            .bins
            .iter()
            .take(route.layout.bin_count)
            .flatten()
            .copied()
        {
            let decoded = decode_eevee_gbuffer_bin(route, closures, bin)?;
            let lobe = match bin.mode {
                EeveeGBufferMode::Diffuse => RayLobe::Diffuse,
                EeveeGBufferMode::Reflection | EeveeGBufferMode::ReflectionColorless => {
                    RayLobe::GgxReflection
                }
                EeveeGBufferMode::Refraction | EeveeGBufferMode::RefractionColorless => {
                    RayLobe::GgxRefraction
                }
                EeveeGBufferMode::ThinRefraction | EeveeGBufferMode::ThinRefractionColorless => {
                    RayLobe::ThinGlassTransmission
                }
                EeveeGBufferMode::Translucent => RayLobe::Translucent,
                EeveeGBufferMode::Subsurface => RayLobe::BurleyBssrdf,
            };
            let roughness = decoded.roughness.unwrap_or(1.0) as f32;
            let input = RayGenerateInput {
                lobe,
                normal: Vec3::new(
                    decoded.normal.x as f32,
                    decoded.normal.y as f32,
                    decoded.normal.z as f32,
                ),
                view: Vec3::new(hit.view.x as f32, hit.view.y as f32, hit.view.z as f32),
                roughness,
                ior: decoded.ior.unwrap_or(1.0) as f32,
                thickness: RayThickness {
                    mode: match route.thickness {
                        Thickness::Slab(_) | Thickness::None => ThicknessMode::Slab,
                        Thickness::Sphere(_) => ThicknessMode::Sphere,
                    },
                    value: route.thickness.value() as f32,
                },
            };
            let source_thickness = route.thickness;
            let dominant_direction = match bin.lobe {
                EeveeGBufferLobe::Diffuse | EeveeGBufferLobe::Subsurface => decoded.normal,
                EeveeGBufferLobe::Translucent => {
                    if matches!(source_thickness, Thickness::Sphere(value) if value != 0.0) {
                        DVec3::ZERO
                    } else {
                        -decoded.normal
                    }
                }
                EeveeGBufferLobe::Reflection | EeveeGBufferLobe::Clearcoat => {
                    eevee_transmission::reflection_direction(
                        decoded.normal,
                        route.view,
                        f64::from(roughness),
                    )
                }
                EeveeGBufferLobe::ThinRefraction => {
                    let oriented_normal = if decoded.normal.dot(route.view) < 0.0 {
                        -decoded.normal
                    } else {
                        decoded.normal
                    };
                    eevee_transmission::reflection_direction(
                        -oriented_normal,
                        eevee_transmission::reflect(route.view, oriented_normal),
                        f64::from(roughness),
                    )
                }
                EeveeGBufferLobe::Refraction => {
                    let ior = decoded.ior.unwrap_or(1.0);
                    super::eevee_advanced_bsdf::eevee_refraction_probe_direction(
                        decoded.normal,
                        route.view,
                        ior,
                        f64::from(roughness),
                        source_thickness,
                    )
                }
            };
            let Some(storage_bin) = bins.get_mut(bin.storage_layer) else {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "EEVEE ray-tracing G-buffer layer exceeds the source closure count",
                ));
            };
            *storage_bin = Some(EeveeRaytraceBin {
                world_input: input,
                dominant_direction,
                transmission: matches!(
                    lobe,
                    RayLobe::GgxRefraction | RayLobe::ThinGlassTransmission | RayLobe::Translucent
                ),
            });
        }
    }
    Ok(EeveeRaytraceSample {
        point: hit.point,
        normal: route.map_or(hit.normal, |route| route.normal),
        has_gbuffer_route: route.is_some(),
        screen_refraction: material_refraction,
        bins,
    })
}

fn standalone_eevee_lighting(
    context: &SurfaceShadeContext<'_>,
    receiver: Receiver,
    params: &BsdfParams,
    thickness: Thickness,
    ambient: f64,
) -> Result<Option<DVec3>> {
    let doc = context.doc;
    let snapshot = context.snapshot;
    let normal = context.normal;
    let shadow = context.shadow;
    let world = context.world;
    let scene = doc.scenes.get(&snapshot.scene_id);
    let light_threshold = scene.map_or(0.01, |scene| scene.render.eevee_light_threshold);
    let mut result = DVec3::ZERO;
    match params.model {
        BsdfModel::Translucent => {
            let mut color = base_albedo(params).max(DVec3::ZERO);
            if thickness != Thickness::None {
                color *= color;
            }
            if let Some(world) = world {
                let environment = if matches!(thickness, Thickness::Sphere(value) if value != 0.0) {
                    world.average_irradiance()
                } else {
                    prefiltered_world(world, -normal, 1.0, irradiance_probe_radiance)
                };
                result += color * environment * ambient;
            }
            for (id, light_node) in &doc.nodes {
                let Some(evaluated) = snapshot.nodes.get(id) else {
                    continue;
                };
                let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                    light_node
                        .data
                        .as_ref()
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .and_then(|block| block.light.as_ref())
                }) else {
                    continue;
                };
                let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                let influence = light_influence_at(light, matrix, receiver.point, light_threshold);
                if influence <= 0.0 {
                    continue;
                }
                let contribution =
                    with_light_shadow_setting(light, shadow, None, |shadow, point_shadow| {
                        let direct_context = DirectLightContext {
                            light,
                            matrix,
                            receiver,
                            normal,
                            ray_visibility: light_node.resolved_ray_visibility(),
                            shadow,
                            point_shadow,
                            sun_shadow_radius: None,
                        };
                        translucent_direct_contribution(&direct_context, color, thickness)
                    })?;
                result += contribution * influence;
            }
            if let Some(sun) = context.world_sun {
                let direct_context = world_sun_direct_context(sun, receiver, normal, shadow);
                result += translucent_direct_contribution(&direct_context, color, thickness);
            }
        }
        BsdfModel::Refraction(_) => {
            let raw_color = base_albedo(params);
            let mut color = raw_color.max(DVec3::ZERO);
            if thickness != Thickness::None {
                color *= color;
            }
            let roughness = params.roughness.clamp(0.0, 1.0);
            let ior = params.ior.max(1.0e-5);
            if let Some(world) = world {
                let direction = super::eevee_advanced_bsdf::eevee_refraction_probe_direction(
                    normal,
                    receiver.view,
                    ior,
                    roughness,
                    thickness,
                );
                let perceived = eevee_transmission::transmission_roughness(roughness, ior);
                result += eevee_prefiltered_world(world, direction, perceived)? * color;
            }
            let glass = eevee_transmission::refraction_light(
                normal,
                receiver.view,
                raw_color,
                roughness,
                ior,
                thickness,
            );
            let mut direct_radiance = (DVec3::ZERO, DVec3::ZERO);
            for (id, light_node) in &doc.nodes {
                let Some(evaluated) = snapshot.nodes.get(id) else {
                    continue;
                };
                let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                    light_node
                        .data
                        .as_ref()
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .and_then(|block| block.light.as_ref())
                }) else {
                    continue;
                };
                let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                let influence = light_influence_at(light, matrix, receiver.point, light_threshold);
                let visibility = light_node.resolved_ray_visibility();
                if influence <= 0.0 || !visibility.allows(RayKind::Transmission) {
                    continue;
                }
                let point_shadow = shadow.and_then(|scene| scene.point_maps.get(id));
                let (reflection, transmission) = with_light_shadow_setting(
                    light,
                    shadow,
                    point_shadow,
                    |shadow, point_shadow| {
                        glass_direct_contribution(
                            light,
                            matrix,
                            receiver,
                            &glass,
                            thickness,
                            visibility,
                            shadow,
                            point_shadow,
                        )
                    },
                )?;
                direct_radiance.0 += reflection * influence;
                direct_radiance.1 += transmission * influence;
            }
            result += glass.reflection * rgb9e5_roundtrip(direct_radiance.0)
                + glass.transmission * rgb9e5_roundtrip(direct_radiance.1);
        }
        BsdfModel::Metallic(_) => {
            let metallic_normal = normalize_fallback(params.normal, normal);
            let Some(reflectance) = eevee_metallic_brdf(params, metallic_normal, receiver.view)
            else {
                return Ok(None);
            };
            let roughness = params.roughness.clamp(0.0, 1.0);
            if let Some(world) = world {
                let direction = super::eevee_advanced_bsdf::eevee_metallic_probe_direction(
                    metallic_normal,
                    receiver.view,
                    roughness,
                );
                result += eevee_prefiltered_world(world, direction, roughness)? * reflectance;
            }
            for (id, light_node) in &doc.nodes {
                let Some(evaluated) = snapshot.nodes.get(id) else {
                    continue;
                };
                let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                    light_node
                        .data
                        .as_ref()
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .and_then(|block| block.light.as_ref())
                }) else {
                    continue;
                };
                let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                let influence = light_influence_at(light, matrix, receiver.point, light_threshold);
                if influence <= 0.0 {
                    continue;
                }
                let point_shadow = shadow.and_then(|scene| scene.point_maps.get(id));
                let contribution = with_light_shadow_setting(
                    light,
                    shadow,
                    point_shadow,
                    |shadow, point_shadow| {
                        let direct_context = DirectLightContext {
                            light,
                            matrix,
                            receiver,
                            normal: metallic_normal,
                            ray_visibility: light_node.resolved_ray_visibility(),
                            shadow,
                            point_shadow,
                            sun_shadow_radius: None,
                        };
                        metallic_direct_contribution(&direct_context, roughness, reflectance)
                    },
                )?;
                result += contribution * influence;
            }
            if let Some(sun) = context.world_sun {
                let direct_context =
                    world_sun_direct_context(sun, receiver, metallic_normal, shadow);
                result += metallic_direct_contribution(&direct_context, roughness, reflectance);
            }
        }
        _ => return Ok(None),
    }
    Ok(Some(result))
}

struct SurfaceShadeContext<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    receiver: Receiver,
    normal: DVec3,
    closures: &'a [ShaderClosure],
    mode: Mode,
    shadow: ShadowContext<'a>,
    object_thickness: f64,
    world: Option<&'a WorldProbe>,
    world_sun: Option<&'a EeveeWorldSunLight>,
    gbuffer: Option<&'a EeveeGBufferRoute>,
    eevee_raytrace_feedback: EeveeRaytraceFeedback,
    material_refraction: Option<bool>,
}

fn shade_eevee_gbuffer(
    context: &SurfaceShadeContext<'_>,
    route: &EeveeGBufferRoute,
) -> Result<[f64; 4]> {
    let doc = context.doc;
    let snapshot = context.snapshot;
    let closures = context.closures;
    let surface_receiver = &context.receiver;
    let normal = context.normal;
    let geometric = surface_receiver.geometric;
    let point = surface_receiver.point;
    let view = surface_receiver.view.try_normalize().unwrap_or(normal);
    let opaque = opaque_share(closures);
    let scene = doc.scenes.get(&snapshot.scene_id);
    let light_threshold = scene.map_or(0.01, |scene| scene.render.eevee_light_threshold);
    let receiver = Receiver {
        view,
        ..*surface_receiver
    };
    let mut reflected = DVec3::ZERO;
    // The resolved layout has at most three bins; keep diffuse AO results on the stack and reuse only for identical
    // decoded normals at this receiver.
    let mut ambient_cache: [Option<(DVec3, f64)>; 3] = [None; 3];
    let mut ambient_cache_len = 0;
    let material_uses_refraction_layer = context.eevee_raytrace_feedback.refraction_active
        && context.material_refraction == Some(true);

    for bin in route
        .layout
        .bins
        .iter()
        .take(route.layout.bin_count)
        .flatten()
        .copied()
    {
        let decoded = decode_eevee_gbuffer_bin(route, closures, bin)?;
        let feedback_radiance = if material_uses_refraction_layer {
            &context.eevee_raytrace_feedback.refraction_radiance
        } else {
            &context.eevee_raytrace_feedback.radiance
        };
        let raytrace_radiance = feedback_radiance
            .get(bin.storage_layer)
            .copied()
            .flatten()
            .map(|radiance| {
                DVec3::new(
                    f64::from(radiance[0]),
                    f64::from(radiance[1]),
                    f64::from(radiance[2]),
                )
            });
        let params = &decoded.closure.params;
        let lobe_normal = decoded.normal;
        let color = decoded.color;
        let thickness = route.thickness;
        // EEVEE's deferred light-probe path has no geometric-AO multiplier; evaluate local ambient visibility only
        // for diffuse-family closures that consume it, and only once per identical receiver/normal input.
        let ambient = if context.world.is_some()
            && matches!(
                bin.lobe,
                EeveeGBufferLobe::Diffuse | EeveeGBufferLobe::Subsurface
            ) {
            if let Some(cached) = ambient_cache[..ambient_cache_len]
                .iter()
                .flatten()
                .find(|(cached_normal, _)| *cached_normal == lobe_normal)
                .map(|(_, value)| *value)
            {
                cached
            } else {
                let ambient = context.shadow.map_or(1.0, |shadow| {
                    let facing = if geometric.dot(view) < 0.0 {
                        -geometric
                    } else {
                        geometric
                    };
                    ambient_visibility(
                        shadow.bvh,
                        shadow.triangles,
                        shadow.prepared_materials,
                        point,
                        facing,
                        lobe_normal,
                        1.0,
                        shadow.has_volume_only,
                    )
                });
                ambient_cache[ambient_cache_len] = Some((lobe_normal, ambient));
                ambient_cache_len += 1;
                ambient
            }
        } else {
            1.0
        };

        match bin.lobe {
            EeveeGBufferLobe::Diffuse | EeveeGBufferLobe::Subsurface => {
                if bin.lobe == EeveeGBufferLobe::Diffuse
                    || !matches!(params.model, BsdfModel::Subsurface)
                {
                    let direct =
                        eevee_gbuffer_diffuse_direct(context, lobe_normal, view, light_threshold)?;
                    let indirect = raytrace_radiance.unwrap_or_else(|| {
                        context.world.map_or(DVec3::ZERO, |world| {
                            prefiltered_world(world, lobe_normal, 1.0, irradiance_probe_radiance)
                                * ambient
                        })
                    });
                    reflected += (direct + indirect) * color;
                }
            }
            EeveeGBufferLobe::Translucent => {
                let environment = raytrace_radiance.unwrap_or_else(|| {
                    context.world.map_or(DVec3::ZERO, |world| {
                        if matches!(thickness, Thickness::Sphere(value) if value != 0.0) {
                            world.average_irradiance()
                        } else {
                            prefiltered_world(world, -lobe_normal, 1.0, irradiance_probe_radiance)
                        }
                    })
                });
                let translucent_color = if thickness == Thickness::None {
                    color
                } else {
                    color * color
                };
                let mut lobe = translucent_color * environment * ambient;
                for (id, light_node) in &doc.nodes {
                    let Some(evaluated) = snapshot.nodes.get(id) else {
                        continue;
                    };
                    let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                        light_node
                            .data
                            .as_ref()
                            .and_then(|data_id| doc.data_blocks.get(data_id))
                            .and_then(|block| block.light.as_ref())
                    }) else {
                        continue;
                    };
                    let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                    let influence = light_influence_at(light, matrix, point, light_threshold);
                    if influence <= 0.0 {
                        continue;
                    }
                    let point_shadow = context.shadow.and_then(|shadow| shadow.point_maps.get(id));
                    let contribution = with_light_shadow_setting(
                        light,
                        context.shadow,
                        point_shadow,
                        |shadow, point_shadow| {
                            let direct_context = DirectLightContext {
                                light,
                                matrix,
                                receiver,
                                normal: lobe_normal,
                                ray_visibility: light_node.resolved_ray_visibility(),
                                shadow,
                                point_shadow,
                                sun_shadow_radius: None,
                            };
                            translucent_direct_contribution(
                                &direct_context,
                                translucent_color,
                                thickness,
                            )
                        },
                    )?;
                    lobe += contribution * influence;
                }
                if let Some(sun) = context.world_sun {
                    let direct_context =
                        world_sun_direct_context(sun, receiver, lobe_normal, context.shadow);
                    lobe += translucent_direct_contribution(
                        &direct_context,
                        translucent_color,
                        thickness,
                    );
                }
                reflected += lobe;
            }
            EeveeGBufferLobe::Reflection | EeveeGBufferLobe::Clearcoat => {
                let roughness = decoded.roughness.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "EEVEE reflection bin has no packed roughness",
                    )
                })?;
                let direction =
                    eevee_transmission::reflection_direction(lobe_normal, view, roughness);
                let environment = if let Some(radiance) = raytrace_radiance {
                    radiance
                } else {
                    context.world.map_or(Ok(DVec3::ZERO), |world| {
                        eevee_deferred_indirect_world(world, direction, roughness)
                    })?
                };
                let mut lobe = environment * color;
                let mut direct_radiance = DVec3::ZERO;
                for (id, light_node) in &doc.nodes {
                    let Some(evaluated) = snapshot.nodes.get(id) else {
                        continue;
                    };
                    let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                        light_node
                            .data
                            .as_ref()
                            .and_then(|data_id| doc.data_blocks.get(data_id))
                            .and_then(|block| block.light.as_ref())
                    }) else {
                        continue;
                    };
                    let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                    let influence = light_influence_at(light, matrix, point, light_threshold);
                    if influence <= 0.0 {
                        continue;
                    }
                    let point_shadow = context.shadow.and_then(|shadow| shadow.point_maps.get(id));
                    let raw_contribution = with_light_shadow_setting(
                        light,
                        context.shadow,
                        point_shadow,
                        |shadow, point_shadow| {
                            let direct_context = DirectLightContext {
                                light,
                                matrix,
                                receiver,
                                normal: lobe_normal,
                                ray_visibility: light_node.resolved_ray_visibility(),
                                shadow,
                                point_shadow,
                                sun_shadow_radius: None,
                            };
                            metallic_direct_contribution(&direct_context, roughness, DVec3::ONE)
                        },
                    )?;
                    let contribution = raw_contribution * influence;
                    direct_radiance += contribution;
                }
                if let Some(sun) = context.world_sun {
                    let direct_context =
                        world_sun_direct_context(sun, receiver, lobe_normal, context.shadow);
                    let contribution =
                        metallic_direct_contribution(&direct_context, roughness, DVec3::ONE);
                    direct_radiance += contribution;
                }
                let direct_store = rgb9e5_roundtrip(direct_radiance);
                lobe += color * direct_store;
                reflected += lobe;
            }
            EeveeGBufferLobe::Refraction | EeveeGBufferLobe::ThinRefraction => {
                let roughness = decoded.roughness.ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "EEVEE refraction bin has no packed roughness",
                    )
                })?;
                let thin = bin.lobe == EeveeGBufferLobe::ThinRefraction;
                let lobe_thickness = if thin { Thickness::None } else { thickness };
                let (glass, direction, probe_roughness) = if thin {
                    let oriented_normal = if lobe_normal.dot(view) < 0.0 {
                        -lobe_normal
                    } else {
                        lobe_normal
                    };
                    let matrix =
                        super::eevee_ltc::sample_matrix(oriented_normal.dot(view), roughness);
                    let glass = GlassLight {
                        oriented_normal,
                        reflection: DVec3::ZERO,
                        reflection_matrix: matrix,
                        transmission: color,
                        transmission_normal: -oriented_normal,
                        transmission_matrix: matrix,
                    };
                    (
                        glass,
                        eevee_transmission::reflection_direction(
                            -oriented_normal,
                            eevee_transmission::reflect(view, oriented_normal),
                            roughness,
                        ),
                        roughness,
                    )
                } else {
                    let ior = decoded.ior.ok_or_else(|| {
                        PotError::new(
                            ErrorCode::EvaluationFailed,
                            "EEVEE refraction bin has no packed IOR",
                        )
                    })?;
                    let glass = eevee_transmission::refraction_light(
                        lobe_normal,
                        view,
                        color,
                        roughness,
                        ior,
                        lobe_thickness,
                    );
                    (
                        glass,
                        super::eevee_advanced_bsdf::eevee_refraction_probe_direction(
                            lobe_normal,
                            view,
                            ior,
                            roughness,
                            lobe_thickness,
                        ),
                        eevee_transmission::transmission_roughness(roughness, ior),
                    )
                };
                let environment = if let Some(radiance) = raytrace_radiance {
                    radiance
                } else {
                    context.world.map_or(Ok(DVec3::ZERO), |world| {
                        eevee_deferred_indirect_world(world, direction, probe_roughness)
                    })?
                };
                let mut lobe = environment * glass.transmission;
                let mut direct_radiance = DVec3::ZERO;
                for (id, light_node) in &doc.nodes {
                    let Some(evaluated) = snapshot.nodes.get(id) else {
                        continue;
                    };
                    let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                        light_node
                            .data
                            .as_ref()
                            .and_then(|data_id| doc.data_blocks.get(data_id))
                            .and_then(|block| block.light.as_ref())
                    }) else {
                        continue;
                    };
                    let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                    let influence = light_influence_at(light, matrix, point, light_threshold);
                    let ray_visibility = light_node.resolved_ray_visibility();
                    if influence <= 0.0 || !ray_visibility.allows(RayKind::Transmission) {
                        continue;
                    }
                    let point_shadow = context.shadow.and_then(|shadow| shadow.point_maps.get(id));
                    let (_, transmission) = with_light_shadow_setting(
                        light,
                        context.shadow,
                        point_shadow,
                        |shadow, point_shadow| {
                            glass_direct_contribution(
                                light,
                                matrix,
                                receiver,
                                &glass,
                                lobe_thickness,
                                ray_visibility,
                                shadow,
                                point_shadow,
                            )
                        },
                    )?;
                    direct_radiance += transmission * influence;
                }
                let direct_store = rgb9e5_roundtrip(direct_radiance);
                let direct_color = glass.transmission * direct_store;
                lobe += direct_color;

                reflected += lobe;
            }
        }
    }
    let mut emitted = DVec3::ZERO;
    for closure in closures {
        let params = &closure.params;
        if matches!(params.model, BsdfModel::Holdout) {
            continue;
        }
        let lobe = BsdfParams {
            reflectance: 1.0,
            ..*params
        };
        let layers = principled_layers(&lobe, normal, view);
        emitted += DVec3::from_array(params.emission_color)
            * (closure.weight
                * params.alpha.clamp(0.0, 1.0)
                * params.emission_strength
                * layers.lower());
    }
    let holdout = eevee_holdout_weight(closures);
    // Deferred EEVEE adds resolved surface lighting without a Holdout multiplier; only raw emission carries (1 - H).
    let reflected = reflected.max(DVec3::ZERO).min(DVec3::splat(1.0e20));
    let result = if opaque > 0.0 {
        (reflected * f64::from(route.gbuffer_alpha) + emitted * (1.0 - holdout)) / opaque
    } else {
        DVec3::ZERO
    };
    Ok([result.x, result.y, result.z, opaque])
}

fn eevee_gbuffer_diffuse_direct(
    context: &SurfaceShadeContext<'_>,
    normal: DVec3,
    view: DVec3,
    light_threshold: f64,
) -> Result<DVec3> {
    eevee_diffuse_direct(
        context.doc,
        context.snapshot,
        Receiver {
            view,
            ..context.receiver
        },
        normal,
        context.shadow,
        context.world_sun,
        light_threshold,
    )
}

fn eevee_diffuse_direct(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    receiver: Receiver,
    normal: DVec3,
    shadow: ShadowContext<'_>,
    world_sun: Option<&EeveeWorldSunLight>,
    light_threshold: f64,
) -> Result<DVec3> {
    eevee_direct_lighting(
        doc,
        snapshot,
        receiver,
        normal,
        shadow,
        world_sun,
        light_threshold,
        diffuse_direct_contribution,
    )
}

fn eevee_subsurface_transmission_direct(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    receiver: Receiver,
    normal: DVec3,
    thickness: Thickness,
    shadow: ShadowContext<'_>,
    world_sun: Option<&EeveeWorldSunLight>,
    light_threshold: f64,
) -> Result<DVec3> {
    eevee_direct_lighting(
        doc,
        snapshot,
        receiver,
        normal,
        shadow,
        world_sun,
        light_threshold,
        |context| transmission_direct_contribution(context, DVec3::ONE, thickness, false),
    )
}

pub(super) fn evaluate_light_shadow_setting<T: PartialEq>(
    use_shadow: Option<bool>,
    mut evaluate: impl FnMut(bool) -> Result<T>,
) -> Result<T> {
    if let Some(casts_shadows) = use_shadow {
        evaluate(casts_shadows)
    } else {
        let shadowed = evaluate(true)?;
        if shadowed == evaluate(false)? {
            Ok(shadowed)
        } else {
            Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "direct lighting needs the imported light's source shadow setting; reimport from Blender to recover it",
                serde_json::json!({"feature_id":"render.light.use_shadow"}),
            ))
        }
    }
}

fn with_light_shadow_setting<T: PartialEq>(
    light: &crate::model::LightData,
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    mut evaluate: impl FnMut(ShadowContext<'_>, Option<&PointShadowMap>) -> T,
) -> Result<T> {
    if light.use_shadow.is_none() && shadow.is_none() && point_shadow.is_none() {
        return Ok(evaluate(None, None));
    }
    evaluate_light_shadow_setting(light.use_shadow, |casts_shadows| {
        let (shadow, point_shadow) = if casts_shadows {
            (shadow, point_shadow)
        } else {
            (None, None)
        };
        Ok(evaluate(shadow, point_shadow))
    })
}

fn eevee_direct_lighting(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    receiver: Receiver,
    normal: DVec3,
    shadow: ShadowContext<'_>,
    world_sun: Option<&EeveeWorldSunLight>,
    light_threshold: f64,
    mut evaluate: impl FnMut(&DirectLightContext<'_>) -> DVec3,
) -> Result<DVec3> {
    let mut result = DVec3::ZERO;
    for (id, light_node) in &doc.nodes {
        let Some(evaluated) = snapshot.nodes.get(id) else {
            continue;
        };
        let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
            light_node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|block| block.light.as_ref())
        }) else {
            continue;
        };
        let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
        let influence = light_influence_at(light, matrix, receiver.point, light_threshold);
        if influence <= 0.0 {
            continue;
        }
        let point_shadow = shadow.and_then(|scene| scene.point_maps.get(id));
        let contribution =
            with_light_shadow_setting(light, shadow, point_shadow, |shadow, point_shadow| {
                let direct = DirectLightContext {
                    light,
                    matrix,
                    receiver,
                    normal,
                    ray_visibility: light_node.resolved_ray_visibility(),
                    shadow,
                    point_shadow,
                    sun_shadow_radius: None,
                };
                evaluate(&direct)
            })?;
        result += contribution * influence;
    }
    if let Some(sun) = world_sun {
        let direct = world_sun_direct_context(sun, receiver, normal, shadow);
        result += evaluate(&direct);
    }
    Ok(result)
}

fn shade(context: &SurfaceShadeContext<'_>) -> Result<[f64; 4]> {
    if let Some(route) = context.gbuffer {
        return shade_eevee_gbuffer(context, route);
    }
    let doc = context.doc;
    let snapshot = context.snapshot;
    let surface_receiver = &context.receiver;
    let normal = context.normal;
    let closures = context.closures;
    let mode = context.mode;
    let shadow = context.shadow;
    let object_thickness = context.object_thickness;
    let world = context.world;
    let point = surface_receiver.point;
    let shadow_point = surface_receiver.shadow_point;
    let geometric = surface_receiver.geometric;
    let opaque = opaque_share(closures);
    if !matches!(mode, Mode::Beauty) {
        let albedo = heaviest(closures, |closure| closure.weight)
            .map_or(DVec3::ZERO, |closure| base_albedo(&closure.params));
        return Ok([albedo.x, albedo.y, albedo.z, opaque]);
    }
    let scene = doc.scenes.get(&snapshot.scene_id);
    // The view is the camera ray's reverse. An orthographic camera's rays are parallel, which is EEVEE's view too
    // (`world_incident_vector`); a perspective ray points at the camera. Without one, the normal.
    let view = surface_receiver.view.try_normalize().unwrap_or(normal);
    // EEVEE's `eevee.light_threshold` sets each local light's influence radius; Blender's default is 0.01.
    let light_threshold = scene.map_or(0.01, |scene| scene.render.eevee_light_threshold);
    let receiver = Receiver {
        view,
        ..*surface_receiver
    };
    // Ambient visibility modulates only diffuse-family indirect light; pure reflection/refraction and worldless
    // materials have no consumer for this hemisphere query.
    let needs_ambient = world.is_some()
        && closures.iter().any(|closure| {
            !matches!(closure.params.model, BsdfModel::Subsurface) && {
                let features = closure.params.eevee_gbuffer_features();
                features.has_diffuse() || features.has_subsurface() || features.has_translucent()
            }
        });
    let ambient = if needs_ambient {
        shadow.map_or(1.0, |scene| {
            let facing = if geometric.dot(view) < 0.0 {
                -geometric
            } else {
                geometric
            };
            ambient_visibility(
                scene.bvh,
                scene.triangles,
                scene.prepared_materials,
                point,
                facing,
                normal,
                1.0,
                scene.has_volume_only,
            )
        })
    } else {
        1.0
    };
    // EEVEE Hair preview may source diffuse radiance weight separately; raster coverage remains based on graph weights.
    // `share` scales radiance and emission, while `opaque_share` independently supplies physical/film coverage.
    let mut reflected = DVec3::ZERO;
    let mut emitted = DVec3::ZERO;
    for closure in closures {
        let params = &closure.params;
        let share =
            params.eevee_diffuse_weight.unwrap_or(closure.weight) * params.alpha.clamp(0.0, 1.0);
        if matches!(params.model, BsdfModel::Holdout | BsdfModel::Subsurface) {
            continue;
        }
        // Reflectance weights the whole sum once, below, so the lobes take unit reflectance.
        let lobe = BsdfParams {
            reflectance: 1.0,
            ..*params
        };
        // The lobes' layers are set up once here for the camera view, shared by every light below.
        let shading = principled_layers(&lobe, normal, view);
        let albedo = base_albedo(params);
        let metallic = params.metallic.clamp(0.0, 1.0);
        let roughness = params.roughness.clamp(0.0, 1.0);
        // The dielectric layer and the diffuse base share what the transmission closures leave (EEVEE's `weight`).
        let dielectric = (1.0 - metallic) * (1.0 - params.transmission.clamp(0.0, 1.0));
        // The coat attenuates every layer below it, including the environment terms.
        let lower = shading.lower();
        // The Material Output's linked Thickness socket decides the thickness when present (EEVEE's
        // `nodetree_thickness`), and the object's bounds otherwise.
        let thickness = Thickness::new(
            params.thickness.unwrap_or(object_thickness),
            params.thickness_slab,
        );
        let lobes = if let Some(lobes) =
            standalone_eevee_lighting(context, receiver, params, thickness, ambient)?
        {
            lobes
        } else {
            let mut lobes = DVec3::ZERO;
            // EEVEE packs each glass closure's summed direct radiance before multiplying its closure colour.
            let mut glass_direct_radiance = (DVec3::ZERO, DVec3::ZERO);
            if let Some(world) = world {
                let diffuse_environment =
                    prefiltered_world(world, normal, 1.0, irradiance_probe_radiance);
                let diffuse_albedo = if params.thin_wall {
                    let subsurface = params.subsurface_weight.clamp(0.0, 1.0);
                    let reflection =
                        f64::midpoint(1.0, -params.subsurface_anisotropy).clamp(0.0, 1.0);
                    albedo * (1.0 - subsurface + subsurface * reflection)
                } else {
                    albedo
                };
                lobes += diffuse_albedo * diffuse_environment * dielectric * lower * ambient;
                let reflection = (-view + normal * (2.0 * normal.dot(view))).normalize_or_zero();
                let specular_environment = eevee_prefiltered_world(world, reflection, roughness)?;
                let specular_f0 = dielectric_f0(params.ior, params.specular_ior_level);
                let f0 = albedo * metallic + DVec3::splat(specular_f0 * dielectric);
                lobes += specular_environment * f0 * (1.0_f64 - roughness).powi(2) * lower;
                if params.thin_wall && params.subsurface_weight > 0.0 {
                    let sphere_thickness =
                        matches!(thickness, Thickness::Sphere(value) if value > 0.0);
                    let translucent_environment = if sphere_thickness {
                        world.average_irradiance()
                    } else {
                        prefiltered_world(world, -normal, 1.0, irradiance_probe_radiance)
                    };
                    let color = if sphere_thickness {
                        albedo * albedo
                    } else {
                        albedo
                    };
                    lobes += color
                        * thin_subsurface_transmission_weight(&lobe, &shading, false)
                        * translucent_environment
                        * ambient;
                }
                // Refraction takes no occlusion: EEVEE's deferred combine multiplies no visibility into the transmission
                // closure's probe read (`eevee_deferred_combine.bsl.hh`), so the glass environment is not attenuated.
                if let Some(glass) =
                    eevee_transmission::glass_closures(params, normal, view, lower, thickness)
                {
                    lobes += glass_environment(world, &glass)?;
                }
            }
            let glass = eevee_transmission::glass_light(params, normal, view, lower, thickness);
            for (id, light_node) in &doc.nodes {
                let Some(evaluated) = snapshot.nodes.get(id) else {
                    continue;
                };
                let Some(light) = evaluated.evaluated_light.as_ref().or_else(|| {
                    light_node
                        .data
                        .as_ref()
                        .and_then(|data_id| doc.data_blocks.get(data_id))
                        .and_then(|block| block.light.as_ref())
                }) else {
                    continue;
                };
                let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
                let point_shadow = shadow.and_then(|scene| scene.point_maps.get(id));
                let influence = light_influence_at(light, matrix, point, light_threshold);
                let ray_visibility = light_node.resolved_ray_visibility();
                let contribution = with_light_shadow_setting(
                    light,
                    shadow,
                    point_shadow,
                    |shadow, point_shadow| {
                        light_contribution(
                            light,
                            matrix,
                            point,
                            shadow_point,
                            normal,
                            geometric,
                            &lobe,
                            &shading,
                            shadow,
                            point_shadow,
                            ray_visibility,
                            thickness,
                            None,
                        )
                    },
                )?;
                lobes += contribution * influence;
                if let Some(glass) = &glass {
                    let (reflection, transmission) = with_light_shadow_setting(
                        light,
                        shadow,
                        point_shadow,
                        |shadow, point_shadow| {
                            glass_direct_contribution(
                                light,
                                matrix,
                                receiver,
                                glass,
                                thickness,
                                ray_visibility,
                                shadow,
                                point_shadow,
                            )
                        },
                    )?;
                    glass_direct_radiance.0 += reflection * influence;
                    glass_direct_radiance.1 += transmission * influence;
                }
            }
            if let Some(sun) = context.world_sun {
                lobes += world_sun_light_contribution(
                    sun, receiver, normal, geometric, &lobe, &shading, shadow, thickness,
                );
            }
            if let Some(glass) = &glass {
                let reflection = rgb9e5_roundtrip(glass_direct_radiance.0);
                let transmission = rgb9e5_roundtrip(glass_direct_radiance.1);
                lobes += glass.reflection * reflection + glass.transmission * transmission;
            }
            lobes
        };
        // Reflectance weights every reflective closure (ambient, direct, environment); Emission has none.
        reflected += lobes * (share * params.reflectance.clamp(0.0, 1.0));
        // Emission sits outside the reflectance weight and the clamp: Cycles' `EmissionNode` emits
        // NODE_CLOSURE_EMISSION unscaled (intern/cycles/scene/shader_nodes.cpp:3071-3098), and neither
        // EEVEE nor Cycles clamps the strength (gpu_shader_material_principled.glsl:179,
        // intern/cycles/kernel/svm/closure.h:206-207).
        emitted += DVec3::from_array(params.emission_color)
            * (share * params.emission_strength * shading.lower());
    }
    // The straight colour is the premultiplied sum over the opaque share. With no opaque share nothing is visible.
    let (reflected, emitted) = if opaque > 0.0 {
        (reflected / opaque, emitted / opaque)
    } else {
        (DVec3::ZERO, DVec3::ZERO)
    };
    // EEVEE's safe_color preserves HDR radiance and caps it after all closures are combined.
    let result = (reflected + emitted)
        .max(DVec3::ZERO)
        .min(DVec3::splat(1.0e20));
    Ok([result.x, result.y, result.z, opaque])
}

fn eevee_subsurface_gbuffer_bin<'a>(
    route: &EeveeGBufferRoute,
    closures: &'a [ShaderClosure],
) -> Result<Option<(EeveeGBufferBin, &'a ShaderClosure)>> {
    for bin in route
        .layout
        .bins
        .iter()
        .take(route.layout.bin_count)
        .flatten()
        .copied()
    {
        if bin.lobe != EeveeGBufferLobe::Subsurface {
            continue;
        }
        let candidate = route.candidates.get(bin.candidate_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE subsurface bin references a missing source candidate",
            )
        })?;
        let closure = closures.get(bin.source_closure_index).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE subsurface bin references a missing source closure",
            )
        })?;
        if candidate.source_closure_index != bin.source_closure_index || candidate.lobe != bin.lobe
        {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE subsurface bin does not match its selected source candidate",
            ));
        }
        if matches!(closure.params.model, BsdfModel::Subsurface) {
            return Ok(Some((bin, closure)));
        }
    }
    Ok(None)
}

fn eevee_subsurface_shading_normal(evaluated: DVec3, geometric: DVec3) -> DVec3 {
    let normal = evaluated.normalize_or_zero();
    if normal == DVec3::ZERO {
        geometric.normalize_or_zero()
    } else {
        normal
    }
}

fn eevee_subsurface_sample(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    point: DVec3,
    shadow_point: DVec3,
    normal: DVec3,
    geometric: DVec3,
    view: DVec3,
    closures: &[ShaderClosure],
    route: Option<&EeveeGBufferRoute>,
    raytrace_feedback: Option<&EeveeRaytraceFeedback>,
    shadow: ShadowContext<'_>,
    world: Option<&WorldProbe>,
    world_sun: Option<&EeveeWorldSunLight>,
    object_index: u32,
    camera_matrices: Option<(&DMat4, &DMat4)>,
    material_refraction: Option<bool>,
) -> Result<Option<EeveeSubsurfaceSample>> {
    let Some(route) = route else {
        return Ok(None);
    };
    let Some((selected_bin, selected_closure)) = eevee_subsurface_gbuffer_bin(route, closures)?
    else {
        return Ok(None);
    };
    let params = &selected_closure.params;
    let radius = eevee_subsurface_radius(params).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "selected EEVEE subsurface bin has no source profile radius",
        )
    })?;
    let Some((_, world_to_camera)) = camera_matrices else {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "EEVEE subsurface requires the per-eye camera transform for view-space depth",
            serde_json::json!({"feature_id":"render.eevee.subsurface.camera_transform"}),
        ));
    };
    let depth = -world_to_camera.transform_point3(point).z;
    if !depth.is_finite() || depth <= 0.0 {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE subsurface hit has invalid view-space depth",
        ));
    }

    let scene = doc.scenes.get(&snapshot.scene_id);
    let light_threshold = scene.map_or(0.01, |scene| scene.render.eevee_light_threshold);
    let view = view.try_normalize().unwrap_or(normal);
    let shading_normal =
        eevee_gbuffer_normal(eevee_subsurface_shading_normal(params.normal, geometric));
    let opaque = opaque_share(closures);
    let profile_color_scale = selected_bin.weight * route.gbuffer_alpha_reciprocal;
    let profile_color = selected_bin
        .color
        .map(|channel| f64::from(channel * profile_color_scale));
    let profile_color_before_pack = profile_color;
    let profile_association_scale = if opaque > 0.0 {
        f64::from(route.gbuffer_alpha) / opaque
    } else {
        0.0
    };
    let radius_context = route.context(selected_bin.data1_layer.ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "EEVEE subsurface bin has no radius storage layer",
        )
    })?);
    let (profile_color, profile_radius) = eevee_subsurface_gbuffer_profile(
        profile_color_before_pack,
        radius,
        route.context(selected_bin.data0_layer),
        radius_context,
    );
    let mut profiles = [None; MAX_EEVEE_SUBSURFACE_CLOSURES];
    let profile = profiles
        .get_mut(selected_bin.storage_layer)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE subsurface bin exceeds the profile storage capacity",
            )
        })?;
    *profile = Some(EeveeSubsurfaceClosure {
        color: profile_color,
        radius: profile_radius,
    });

    let thickness = route.thickness;
    let receiver = Receiver {
        point,
        shadow_point,
        geometric,
        view,
    };
    let direct_front = eevee_diffuse_direct(
        doc,
        snapshot,
        receiver,
        shading_normal,
        shadow,
        world_sun,
        light_threshold,
    )?;
    let transmission = super::eevee_advanced_bsdf::eevee_subsurface_transmission(
        profile_radius,
        thickness.value() as f32,
    )
    .map(f64::from);
    let direct_back = if transmission.iter().any(|value| *value > 0.0) {
        eevee_subsurface_transmission_direct(
            doc,
            snapshot,
            receiver,
            shading_normal,
            thickness,
            shadow,
            world_sun,
            light_threshold,
        )?
    } else {
        DVec3::ZERO
    };
    let direct = direct_front + direct_back * DVec3::from_array(transmission);
    let ambient = if world.is_some() {
        shadow.map_or(1.0, |scene| {
            let facing = if geometric.dot(view) < 0.0 {
                -geometric
            } else {
                geometric
            };
            ambient_visibility(
                scene.bvh,
                scene.triangles,
                scene.prepared_materials,
                point,
                facing,
                shading_normal,
                1.0,
                scene.has_volume_only,
            )
        })
    } else {
        1.0
    };
    let indirect = raytrace_feedback
        .and_then(|feedback| {
            let radiance = if feedback.refraction_active && material_refraction == Some(true) {
                &feedback.refraction_radiance
            } else {
                &feedback.radiance
            };
            radiance.get(selected_bin.storage_layer).copied().flatten()
        })
        .map_or_else(
            || {
                world.map_or(DVec3::ZERO, |world| {
                    let Some(sh) = world.fast_gi_sh() else {
                        return DVec3::ZERO;
                    };
                    let transmission = super::eevee_advanced_bsdf::eevee_subsurface_transmission(
                        profile_radius,
                        thickness.value() as f32,
                    )
                    .map(f64::from);
                    let normal = shading_normal.as_vec3();
                    let front = sh
                        .evaluate_lambert(normal)
                        .truncate()
                        .to_array()
                        .map(f64::from);
                    let back = sh
                        .evaluate_lambert(-normal)
                        .truncate()
                        .to_array()
                        .map(f64::from);
                    (DVec3::from_array(front)
                        + DVec3::from_array(back) * DVec3::from_array(transmission))
                        * ambient
                })
            },
            |radiance| DVec3::from_array(radiance.map(f64::from)),
        );
    Ok(Some(EeveeSubsurfaceSample {
        closures: profiles,
        direct: direct.to_array().map(|value| value as f32),
        indirect: indirect.to_array().map(|value| value as f32),
        association_scale: profile_association_scale,
        object_index,
        depth,
        split_radiance: false,
    }))
}

/// The environment's share of the glass closures: the reflection at its mirror direction and the transmission at the
/// probe direction, each blurred with its roughness as EEVEE's light probe is (`eevee_transmission`).
fn glass_environment(world: &WorldProbe, closures: &GlassClosures) -> Result<DVec3> {
    let reflected = eevee_prefiltered_world(
        world,
        closures.reflection_direction,
        closures.reflection_roughness,
    )?;
    let transmitted = eevee_prefiltered_world(
        world,
        closures.transmission_direction,
        closures.transmission_roughness,
    )?;
    Ok(reflected * closures.reflection + transmitted * closures.transmission)
}

/// EEVEE specular reads use the roughness-prefiltered sphere-probe atlas, which is constructed in EEVEE mode.
fn eevee_prefiltered_world(world: &WorldProbe, direction: DVec3, roughness: f64) -> Result<DVec3> {
    world
        .prefiltered_radiance(direction, roughness)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "EEVEE specular shading requires an EEVEE-filtered world probe",
            )
        })
}

/// The deferred split-indirect target stores the filtered, uncolored probe value in `UFLOAT_11_11_10` before the
/// deferred combine multiplies by the closure color. Forward shading continues to use the unquantized atlas sample.
fn eevee_deferred_indirect_world(
    world: &WorldProbe,
    direction: DVec3,
    roughness: f64,
) -> Result<DVec3> {
    let radiance = eevee_prefiltered_world(world, direction, roughness)?;
    let radiance = DVec3::from_array(radiance.to_array().map(|channel| f64::from(channel as f32)));
    Ok(eevee_transmission::probe_radiance(radiance))
}

/// The world's filtered radiance along `axis` for `roughness`: five taps spread over a lobe of that roughness, read from
/// the render's world probe. EEVEE cube texels are already quantized at their SFLOAT16 store; `stored` models only a
/// later per-tap radiance conversion needed by the caller (for example, diffuse irradiance), not a second probe store.
fn prefiltered_world(
    world: &WorldProbe,
    axis: DVec3,
    roughness: f64,
    stored: impl Fn(DVec3) -> DVec3,
) -> DVec3 {
    const FILTER_OFFSETS: [[f64; 2]; 5] =
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let axis = axis.normalize_or_zero();
    if axis == DVec3::ZERO {
        return DVec3::ZERO;
    }
    let tangent = if axis.z.abs() < 0.999 {
        axis.cross(DVec3::Z).normalize_or_zero()
    } else {
        axis.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = axis.cross(tangent).normalize_or_zero();
    let spread = roughness.clamp(0.0, 1.0).powi(2) * 0.5;
    let mut irradiance = DVec3::ZERO;
    for offset in FILTER_OFFSETS {
        let direction = (axis + tangent * (spread * offset[0]) + bitangent * (spread * offset[1]))
            .normalize_or_zero();
        irradiance += stored(world.radiance(direction));
    }
    irradiance / 5.0
}

#[expect(
    clippy::too_many_arguments,
    reason = "light shading carries explicit material and visibility inputs"
)]
fn light_contribution(
    light: &crate::model::LightData,
    matrix: DMat4,
    point: DVec3,
    shadow_point: DVec3,
    normal: DVec3,
    geometric: DVec3,
    params: &BsdfParams,
    shading: &Layers,
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    lobes: RayVisibility,
    thickness: Thickness,
    sun_shadow_radius: Option<f64>,
) -> DVec3 {
    let color = DVec3::from_array(light.color).max(DVec3::ZERO);
    let energy = light.energy.max(0.0);
    let position = matrix.transform_point3(DVec3::ZERO);
    let sphere_thickness = matches!(thickness, Thickness::Sphere(value) if value > 0.0);
    let mut light_params = *params;
    light_params.thickness = Some(thickness.value());
    light_params.thickness_slab = matches!(thickness, Thickness::Slab(_));
    let contribution = match light.light_type {
        crate::model::LightType::Sun => {
            let toward_light = matrix.transform_vector3(DVec3::Z).normalize_or_zero();
            // The disc's visibility is already inside `sun_lighting`, one ray per disc sample, so the sun is not shadowed again.
            sun_lighting(
                &light_params,
                shading,
                shadow,
                shadow_point,
                geometric,
                toward_light,
                color,
                light,
                lobes,
                sun_shadow_radius,
            )
        }
        crate::model::LightType::Point => {
            let delta = position - point;
            let distance_squared = delta.length_squared().max(1.0e-6);
            let toward_light = delta.normalize_or_zero();
            punctual_radiance(
                &light_params,
                shading,
                toward_light,
                color,
                point_intensity(energy) / distance_squared,
                LightFactors::of(light),
                lobes,
            ) * visibility_for_light(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                toward_light,
                distance_squared.sqrt(),
                0.0,
            )
        }
        crate::model::LightType::Spot => {
            let delta = position - point;
            let distance_squared = delta.length_squared().max(1.0e-6);
            let toward_light = delta.normalize_or_zero();
            let falloff = spot_cone(&matrix, toward_light, light.spot_size, light.spot_blend);
            if falloff <= 0.0 {
                return DVec3::ZERO;
            }
            punctual_radiance(
                &light_params,
                shading,
                toward_light,
                color,
                (point_intensity(energy) / distance_squared) * falloff,
                LightFactors::of(light),
                lobes,
            ) * visibility_for_light(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                toward_light,
                distance_squared.sqrt(),
                0.0,
            )
        }
        crate::model::LightType::Area => {
            let shape = area_shape(light, &matrix);
            if shape.area <= 0.0 {
                return DVec3::ZERO;
            }
            let light_direction = matrix.transform_vector3(-DVec3::Z).normalize_or_zero();
            let radiance = area_radiance(energy, shape.area);
            // The lit hemisphere is the receiver's normal side; the transmitted share is the reverse side, where the
            // emitter's outline clips the same way (`area_irradiance_per_radiance` and `area_visibility_samples`).
            let front = area_side_weight(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                &shape,
                position,
                light_direction,
                point,
                normal,
            ) * radiance;
            let back = if params.thin_wall && !sphere_thickness {
                area_side_weight(
                    shadow,
                    point_shadow,
                    shadow_point,
                    -geometric,
                    &shape,
                    position,
                    light_direction,
                    point,
                    -normal,
                ) * radiance
            } else {
                0.0
            };
            // Integrated Lambert irradiance is exact for rectangles, and the visibility weight scales it. The specular,
            // sheen and coat lobes are evaluated towards the emitter centre on the lit side.
            let toward_centre = (position - point).normalize_or_zero();
            let front_params = if params.thin_wall {
                *params
            } else {
                without_subsurface(params)
            };
            let factors = LightFactors::of(light);
            let diffuse = reflected_lobes_coefficient(
                &front_params,
                shading,
                toward_centre,
                lobes.mask() & RayKind::Diffuse.bit(),
            );
            let specular = reflected_lobes_coefficient(
                &front_params,
                shading,
                toward_centre,
                lobes.mask() & RayKind::Glossy.bit(),
            );
            let translucent = if params.thin_wall && !sphere_thickness {
                reflected_lobes_coefficient(
                    params,
                    shading,
                    -normal,
                    lobes.mask() & RayKind::Diffuse.bit(),
                )
            } else {
                DVec3::ZERO
            };
            (diffuse * factors.diffuse + specular * factors.specular) * front * color
                + translucent * (factors.diffuse * back) * color
        }
    };
    let sphere_translucent = if sphere_thickness
        && params.thin_wall
        && params.subsurface_weight > 0.0
        && lobes.mask() & RayKind::Diffuse.bit() != 0
    {
        let Some(direct) =
            direct_emitter_with_sun_shadow_radius(light, matrix, point, sun_shadow_radius)
        else {
            return contribution;
        };
        let origin = transmission_origin(
            point,
            geometric,
            direct.direction,
            direct.shadow_distance,
            thickness,
        );
        let visibility = if let Some(shape) = direct.area {
            area_transmitted_fraction(
                shadow,
                origin,
                &shape,
                (direct.centre, -direct.axis),
                direct.direction,
            )
        } else {
            transmitted_visibility(shadow, origin, &direct)
        };
        let albedo = base_albedo(params).max(DVec3::ZERO);
        let weight = thin_subsurface_transmission_weight(params, shading, false);
        let form = direct.emitter.form_factor(
            direct.direction,
            shading.view,
            [1.0, 0.0, 0.0, 1.0],
            direct.to_centre,
        );
        albedo
            * albedo
            * weight
            * direct.radiance
            * (direct.attenuation * visibility * form * LightFactors::of(light).diffuse
                / std::f64::consts::PI)
    } else {
        DVec3::ZERO
    };
    contribution + sphere_translucent
}

/// The weight of an area light over the hemisphere on `side` of a receiver: the Lambert irradiance of the emitter's
/// outline clipped to that hemisphere, per unit radiance, times the share of its outline samples that reach the receiver.
/// `geometric` is the geometric normal on the same side, which offsets the shadow rays to it.
fn area_side_weight(
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    shadow_point: DVec3,
    geometric: DVec3,
    shape: &AreaShape,
    position: DVec3,
    light_direction: DVec3,
    point: DVec3,
    side: DVec3,
) -> f64 {
    let irradiance = area_irradiance_per_radiance(shape, position, light_direction, point, side);
    if irradiance <= 0.0 {
        return 0.0;
    }
    irradiance
        * area_visible_fraction(
            shadow,
            point_shadow,
            shadow_point,
            geometric,
            shape,
            position,
            light_direction,
            point,
            side,
        )
}

/// The share of an area light's outline samples that reach the receiver, over the hemisphere on `side`: the sample
/// weights of `area_visibility_samples`, times their visibility, over their total weight.
fn area_visible_fraction(
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    shadow_point: DVec3,
    geometric: DVec3,
    shape: &AreaShape,
    position: DVec3,
    light_direction: DVec3,
    point: DVec3,
    side: DVec3,
) -> f64 {
    let (samples, count) = area_visibility_samples(shape, position, light_direction, point, side);
    let mut total_weight = 0.0;
    let mut visible_weight = 0.0;
    for sample in &samples[..count] {
        total_weight += sample.weight;
        let visibility = visibility_for_light(
            shadow,
            point_shadow,
            shadow_point,
            geometric,
            sample.direction,
            sample.distance,
            0.0,
        );
        visible_weight += sample.weight * visibility;
    }
    if total_weight <= 0.0 || visible_weight <= 0.0 {
        return 0.0;
    }
    visible_weight / total_weight
}

/// The visibility of a light from `point`: its depth map for a point or spot light, else the scene's rays. `sun_radius` is a
/// sun's disc radius (`sun_disc_radius`) and is read only at an infinite distance, where the light is a sun.
fn visibility_for_light(
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
    sun_radius: f64,
) -> f64 {
    let Some(shadow_map) = point_shadow else {
        return shadow_visibility(shadow, point, normal, direction, distance, sun_radius);
    };
    let opaque = shadow_map.visibility(point);
    match shadow {
        Some(scene) if scene.occluders.is_some() && opaque > 0.0 => {
            opaque * point_light_transparency(scene, point, normal, direction, distance)
        }
        _ => opaque,
    }
}

/// A point or spot light's transparent pass: one ray toward the light through the transparent casters, since the depth
/// map holds only the opaque triangles.
fn point_light_transparency(
    shadow: ShadowScene<'_>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
) -> f64 {
    let direction = direction.normalize_or_zero();
    if direction == DVec3::ZERO {
        return 0.0;
    }
    let maximum_distance = if distance.is_finite() {
        (distance - SHADOW_EPSILON).max(0.0)
    } else {
        f64::INFINITY
    };
    if maximum_distance <= SHADOW_EPSILON {
        return 1.0;
    }
    let origin = shadow_origin(point, normal, direction);
    ray_transmittance(shadow, origin, direction, maximum_distance, false)
}

fn shadow_visibility(
    shadow: ShadowContext<'_>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
    sun_radius: f64,
) -> f64 {
    const PCF_OFFSETS: [[f64; 2]; 9] = [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let Some(shadow) = shadow else {
        return 1.0;
    };
    let direction = direction.normalize_or_zero();
    if direction == DVec3::ZERO {
        return 0.0;
    }
    let normal = normal.normalize_or_zero();
    if !distance.is_finite() {
        return sun_disc_visibility(shadow, point, normal, direction, sun_radius);
    }
    let epsilon = SHADOW_EPSILON;
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent).normalize_or_zero();
    let angular_radius = (0.02 / distance.max(0.02)).min(0.02);
    let maximum_distance = (distance - epsilon).max(0.0);
    if maximum_distance <= epsilon {
        return 1.0;
    }
    let mut visible = 0.0;
    for offset in PCF_OFFSETS {
        let sample_direction = (direction
            + tangent * (offset[0] * angular_radius)
            + bitangent * (offset[1] * angular_radius))
            .normalize_or_zero();
        let origin = shadow_origin(point, normal, sample_direction);
        visible += ray_transmittance(shadow, origin, sample_direction, maximum_distance, true);
    }
    visible / 9.0
}

/// The directions of a sun's disc: `SUN_DISC_SAMPLES` unit vectors uniform over the disc of tangent radius `radius` about
/// `direction`, in a Vogel spiral.
fn sun_disc_directions(direction: DVec3, radius: f64) -> impl Iterator<Item = DVec3> {
    let (tangent, bitangent) = make_orthonormals(direction);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..SUN_DISC_SAMPLES).map(move |index| {
        let offset = ((index as f64 + 0.5) / SUN_DISC_SAMPLES as f64).sqrt() * radius;
        let angle = index as f64 * golden_angle;
        (direction + tangent * (offset * angle.cos()) + bitangent * (offset * angle.sin()))
            .normalize_or_zero()
    })
}

/// Keeps the direct-light disc and its independently jittered shadow disc on the same spiral samples.
fn sun_disc_direction_pairs(
    direction: DVec3,
    shape_radius: f64,
    shadow_radius: f64,
) -> impl Iterator<Item = (DVec3, DVec3)> {
    let (tangent, bitangent) = make_orthonormals(direction);
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..SUN_DISC_SAMPLES).map(move |index| {
        let radial = ((index as f64 + 0.5) / SUN_DISC_SAMPLES as f64).sqrt();
        let angle = index as f64 * golden_angle;
        let (cosine, sine) = (angle.cos(), angle.sin());
        let sample = |radius: f64| {
            (direction
                + tangent * (radial * radius * cosine)
                + bitangent * (radial * radius * sine))
                .normalize_or_zero()
        };
        let shape = sample(shape_radius);
        let shadow = if shape_radius
            .partial_cmp(&shadow_radius)
            .is_some_and(std::cmp::Ordering::is_eq)
        {
            shape
        } else {
            sample(shadow_radius)
        };
        (shape, shadow)
    })
}

/// A sun's visibility at a receiver: the mean over rays from the receiver to the sun's disc, each blocked by an opaque
/// occluder (`ray_transmittance`).
fn sun_disc_visibility(
    shadow: ShadowScene<'_>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    radius: f64,
) -> f64 {
    let total: f64 = sun_disc_directions(direction, radius)
        .map(|sample| {
            let origin = shadow_origin(point, normal, sample);
            ray_transmittance(shadow, origin, sample, f64::INFINITY, true)
        })
        .sum();
    total / SUN_DISC_SAMPLES as f64
}

/// The receiver of direct light at a shading point: the point and its shadow origin, the geometric normal, and the view.
#[derive(Clone, Copy)]
struct Receiver {
    point: DVec3,
    shadow_point: DVec3,
    geometric: DVec3,
    view: DVec3,
}

/// The smallest radius of a light's shape (`shape_parameters_set`: `max(0.001, radius)`).
const MIN_SHAPE_RADIUS: f64 = 0.001;
/// The samples of a sun's disc: a Vogel spiral that integrates the disc's shadow as EEVEE's soft shadow cone does
/// (`shadow_ray_generate_directional`). Sixteen match EEVEE's penumbra to within its noise, where nine taps do not.
const SUN_DISC_SAMPLES: usize = 16;

/// A light as EEVEE's light pass sees it from one point: its emitter, the direction and distance to its centre, the
/// radiance of its colour and power, and the attenuation the lobes share (`light_attenuation_surface`: the influence
/// radius, a spot's cone and an area light's emitting side).
struct DirectEmitter {
    emitter: Emitter,
    /// The unit direction from the point to the emitter's centre (a sun's direction).
    direction: DVec3,
    /// The vector from the point to the emitter's centre, whose length is the distance (a sun's is its direction).
    to_centre: DVec3,
    /// The emitter's distance, one for a sun.
    distance: f64,
    /// The distance a shadow ray reaches the light across, infinite for a sun.
    shadow_distance: f64,
    /// The light's colour times its radiance per unit of closure colour.
    radiance: DVec3,
    /// The attenuation shared by the lobes.
    attenuation: f64,
    /// The emitter's centre and emitting axis, for an area light's outline samples.
    centre: DVec3,
    axis: DVec3,
    /// The area light's shape, or `None` for a point, spot or sun light.
    area: Option<AreaShape>,
    sun: bool,
    /// A sun's disc radius in tangent units (`sun_disc_radius`), zero for the other lights.
    sun_radius: f64,
}

/// EEVEE's influence of a light at `point` (`light_attenuation_surface`), applied to every closure of the light. A sun
/// has none; a local light fades to zero at its influence radius (`local_light_influence`).
fn light_influence_at(
    light: &crate::model::LightData,
    matrix: DMat4,
    point: DVec3,
    threshold: f64,
) -> f64 {
    if matches!(light.light_type, crate::model::LightType::Sun) {
        return 1.0;
    }
    let distance_squared = (matrix.transform_point3(DVec3::ZERO) - point)
        .length_squared()
        .max(1.0e-6);
    local_light_influence(light, distance_squared, threshold)
}

/// A light as EEVEE's light pass sees it from `point`, or `None` for an area light with no area. The lobes' shared
/// attenuation is the cone of a spot and the emitting side of an area light; the influence is `light_influence_at`'s.
fn direct_emitter(
    light: &crate::model::LightData,
    matrix: DMat4,
    point: DVec3,
) -> Option<DirectEmitter> {
    direct_emitter_with_sun_shadow_radius(light, matrix, point, None)
}

fn direct_emitter_with_sun_shadow_radius(
    light: &crate::model::LightData,
    matrix: DMat4,
    point: DVec3,
    sun_shadow_radius: Option<f64>,
) -> Option<DirectEmitter> {
    let color = DVec3::from_array(light.color).max(DVec3::ZERO);
    let energy = light.energy.max(0.0);
    let position = matrix.transform_point3(DVec3::ZERO);
    let axis = matrix.transform_vector3(DVec3::Z).normalize_or_zero();
    match light.light_type {
        crate::model::LightType::Sun => {
            let radius = sun_disc_radius(light.sun_angle);
            let shadow_radius = sun_shadow_radius.unwrap_or(radius);
            // `shape_radiance_get` for a sun: `(1 + 1 / r²) / π` per unit of irradiance.
            let inv_sin_squared = 1.0 + 1.0 / (radius * radius);
            Some(DirectEmitter {
                emitter: Emitter::Disk { radius },
                direction: axis,
                to_centre: axis,
                distance: 1.0,
                shadow_distance: f64::INFINITY,
                radiance: color * (energy * inv_sin_squared / std::f64::consts::PI),
                attenuation: 1.0,
                centre: point + axis,
                axis,
                area: None,
                sun: true,
                sun_radius: shadow_radius,
            })
        }
        crate::model::LightType::Point | crate::model::LightType::Spot => {
            let radius = light.radius.max(MIN_SHAPE_RADIUS);
            let to_centre = position - point;
            let distance_squared = to_centre.length_squared().max(1.0e-6);
            let distance = distance_squared.sqrt();
            let direction = to_centre / distance;
            let emitter = if light.soft_falloff {
                Emitter::Disk { radius }
            } else {
                Emitter::Sphere { radius }
            };
            let attenuation = if matches!(light.light_type, crate::model::LightType::Spot) {
                spot_cone(&matrix, direction, light.spot_size, light.spot_blend)
            } else {
                1.0
            };
            Some(DirectEmitter {
                emitter,
                direction,
                to_centre,
                distance,
                shadow_distance: distance,
                radiance: color * emitter_radiance(energy, radius),
                attenuation,
                centre: position,
                axis,
                area: None,
                sun: false,
                sun_radius: 0.0,
            })
        }
        crate::model::LightType::Area => {
            let shape = area_shape(light, &matrix);
            if shape.area <= 0.0 {
                return None;
            }
            let to_centre = position - point;
            let distance_squared = to_centre.length_squared().max(1.0e-6);
            let distance = distance_squared.sqrt();
            let direction = to_centre / distance;
            // `light_attenuation_common`: an area light emits along its own axis, so its lit side is where `L · axis > 0`.
            let emitting_side = if direction.dot(axis) > 0.0 { 1.0 } else { 0.0 };
            Some(DirectEmitter {
                emitter: Emitter::Area {
                    axis_u: shape.axis_u,
                    axis_v: shape.axis_v,
                    half_u: shape.half_u,
                    half_v: shape.half_v,
                    ellipse: shape.ellipse,
                },
                direction,
                to_centre,
                distance,
                shadow_distance: distance,
                radiance: color * area_radiance(energy, shape.area),
                attenuation: emitting_side,
                centre: position,
                axis,
                area: Some(shape),
                sun: false,
                sun_radius: 0.0,
            })
        }
    }
}

/// EEVEE's `light_attenuation_facing`: fades an emitter that is near or below the horizon of `ng`. A transmission reads
/// its lobe's normal reversed, so the light behind the surface is lit.
fn facing_attenuation(
    shape_radius: f64,
    direction: DVec3,
    distance: f64,
    ng: DVec3,
    is_transmission: bool,
) -> f64 {
    let sin_solid_angle = shape_radius / distance;
    let sin_light_angle = direction.dot(if is_transmission { -ng } else { ng });
    let horizon = sin_solid_angle
        + if is_transmission {
            -sin_light_angle
        } else {
            sin_light_angle
        };
    ((horizon + 0.1) * 10.0).clamp(0.0, 1.0)
}

/// The origin of a transmission's shadow ray, as EEVEE's `shadow_eval` moves it for `is_transmission`: a receiver that
/// faces away from its light is moved through the object by its thickness toward the light (at most up to a punctual
/// light), and biased along `reflect(Ng, L)`. A receiver that faces the light is biased along `Ng`.
fn transmission_origin(
    point: DVec3,
    geometric: DVec3,
    direction: DVec3,
    distance: f64,
    thickness: Thickness,
) -> DVec3 {
    if geometric.dot(direction) > 0.0 {
        return shadow_origin(point, geometric, direction);
    }
    let reach = if distance.is_finite() {
        thickness.value().min(distance - 0.01)
    } else {
        thickness.value()
    };
    let moved = point + direction * reach.abs();
    shadow_origin(
        moved,
        eevee_transmission::reflect(geometric, direction),
        direction,
    )
}

/// The transmittance of a shadow ray from `origin` along `direction` up to `distance`, through every occluder: a
/// transmission's shadow is traced against the scene, because the light's depth map holds the view from the light and
/// the transmission starts behind the surface. `1` without shadows.
fn transmittance_toward(
    shadow: ShadowContext<'_>,
    origin: DVec3,
    direction: DVec3,
    distance: f64,
) -> f64 {
    let Some(shadow) = shadow else {
        return 1.0;
    };
    let direction = direction.normalize_or_zero();
    if direction == DVec3::ZERO {
        return 0.0;
    }
    let maximum = if distance.is_finite() {
        (distance - SHADOW_EPSILON).max(0.0)
    } else {
        f64::INFINITY
    };
    if maximum <= SHADOW_EPSILON {
        return 1.0;
    }
    ray_transmittance(shadow, origin, direction, maximum, true)
}

/// The visibility of a light for a transmission: one ray from the amended origin to a point light's centre, or the mean
/// over a sun's disc.
fn transmitted_visibility(shadow: ShadowContext<'_>, origin: DVec3, direct: &DirectEmitter) -> f64 {
    if direct.sun {
        let total: f64 = sun_disc_directions(direct.direction, direct.sun_radius)
            .map(|sample| transmittance_toward(shadow, origin, sample, f64::INFINITY))
            .sum();
        return total / SUN_DISC_SAMPLES as f64;
    }
    let to_centre = direct.centre - origin;
    transmittance_toward(shadow, origin, to_centre, to_centre.length())
}

/// The share of an area light's outline samples that reach a transmission's amended origin, over the hemisphere on
/// `side` (the lobe's own side), as `area_visible_fraction` gives it for the reflection.
fn area_transmitted_fraction(
    shadow: ShadowContext<'_>,
    origin: DVec3,
    shape: &AreaShape,
    emitter: (DVec3, DVec3),
    side: DVec3,
) -> f64 {
    let (centre, axis) = emitter;
    let (samples, count) = area_visibility_samples(shape, centre, axis, origin, side);
    let mut total_weight = 0.0;
    let mut visible_weight = 0.0;
    for sample in &samples[..count] {
        total_weight += sample.weight;
        visible_weight +=
            sample.weight * transmittance_toward(shadow, origin, sample.direction, sample.distance);
    }
    if total_weight <= 0.0 || visible_weight <= 0.0 {
        return 0.0;
    }
    visible_weight / total_weight
}

/// The per-light direct radiance for EEVEE's reflection and transmission closures. The caller sums each lobe over lights,
/// packs the sums as RGB9E5, then applies the closure colour. The reflection is shadowed from the shading point; the
/// transmission from the point moved through the object by its thickness (`transmission_origin`).
fn glass_direct_contribution(
    light: &crate::model::LightData,
    matrix: DMat4,
    receiver: Receiver,
    glass: &GlassLight,
    thickness: Thickness,
    ray_visibility: RayVisibility,
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
) -> (DVec3, DVec3) {
    let Some(direct) = direct_emitter(light, matrix, receiver.point) else {
        return (DVec3::ZERO, DVec3::ZERO);
    };
    let factors = LightFactors::of(light);
    let mut result = (DVec3::ZERO, DVec3::ZERO);

    // The reflection, lit from the normal's side.
    let facing = facing_attenuation(
        direct.emitter.shape_radius(),
        direct.direction,
        direct.distance,
        glass.oriented_normal,
        false,
    );
    if ray_visibility.allows(RayKind::Glossy) && facing > 0.0 {
        let visibility = match direct.area {
            Some(shape) => area_visible_fraction(
                shadow,
                point_shadow,
                receiver.shadow_point,
                receiver.geometric,
                &shape,
                direct.centre,
                -direct.axis,
                receiver.point,
                glass.oriented_normal,
            ),
            None => visibility_for_light(
                shadow,
                point_shadow,
                receiver.shadow_point,
                receiver.geometric,
                direct.direction,
                direct.shadow_distance,
                direct.sun_radius,
            ),
        };
        let form = direct.emitter.form_factor(
            glass.oriented_normal,
            receiver.view,
            glass.reflection_matrix,
            direct.to_centre,
        );
        result.0 +=
            direct.radiance * (direct.attenuation * facing * visibility * form * factors.specular);
    }

    // The transmission, lit from the reverse side, where the lobe's normal points.
    let facing = facing_attenuation(
        direct.emitter.shape_radius(),
        direct.direction,
        direct.distance,
        glass.transmission_normal,
        true,
    );
    if facing > 0.0 {
        let origin = transmission_origin(
            receiver.point,
            receiver.geometric,
            direct.direction,
            direct.shadow_distance,
            thickness,
        );
        let visibility = match direct.area {
            Some(shape) => area_transmitted_fraction(
                shadow,
                origin,
                &shape,
                (direct.centre, -direct.axis),
                -glass.oriented_normal,
            ),
            None => transmitted_visibility(shadow, origin, &direct),
        };
        let form = direct.emitter.form_factor(
            glass.transmission_normal,
            receiver.view,
            glass.transmission_matrix,
            direct.to_centre,
        );
        result.1 += direct.radiance
            * (direct.attenuation * facing * visibility * form * factors.transmission);
    }
    result
}
/// EEVEE's `rgb9e5_encode`/`rgb9e5_decode` boundary for a closure's summed direct radiance.
fn rgb9e5_roundtrip(radiance: DVec3) -> DVec3 {
    let color = [
        (radiance.x as f32).clamp(0.0, 65_408.0),
        (radiance.y as f32).clamp(0.0, 65_408.0),
        (radiance.z as f32).clamp(0.0, 65_408.0),
    ];
    let max_component = color[0].max(color[1]).max(color[2]);
    let log2_floored = ((max_component.to_bits() >> 23) as i32) - 127;
    let exponent = log2_floored.max(-16) + 16;
    let mut denominator = f32::from_bits(((exponent - 24 + 127) as u32) << 23);
    if (max_component / denominator + 0.5) as i32 == 512 {
        denominator *= 2.0;
    }

    DVec3::new(
        (((color[0] / denominator + 0.5) as u32) as f32 * denominator) as f64,
        (((color[1] / denominator + 0.5) as u32) as f32 * denominator) as f64,
        (((color[2] / denominator + 0.5) as u32) as f32 * denominator) as f64,
    )
}

struct DirectLightContext<'a> {
    light: &'a crate::model::LightData,
    matrix: DMat4,
    receiver: Receiver,
    normal: DVec3,
    ray_visibility: RayVisibility,
    shadow: ShadowContext<'a>,
    point_shadow: Option<&'a PointShadowMap>,
    sun_shadow_radius: Option<f64>,
}

fn world_sun_shadow<'a>(sun: &EeveeWorldSunLight, shadow: ShadowContext<'a>) -> ShadowContext<'a> {
    if sun.use_shadows { shadow } else { None }
}

fn world_sun_direct_context<'a>(
    sun: &'a EeveeWorldSunLight,
    receiver: Receiver,
    normal: DVec3,
    shadow: ShadowContext<'a>,
) -> DirectLightContext<'a> {
    DirectLightContext {
        light: &sun.light,
        matrix: sun.matrix,
        receiver,
        normal,
        ray_visibility: sun.ray_visibility,
        shadow: world_sun_shadow(sun, shadow),
        point_shadow: None,
        sun_shadow_radius: sun.shadow_radius,
    }
}

fn world_sun_light_contribution(
    sun: &EeveeWorldSunLight,
    receiver: Receiver,
    normal: DVec3,
    geometric: DVec3,
    params: &BsdfParams,
    shading: &Layers,
    shadow: ShadowContext<'_>,
    thickness: Thickness,
) -> DVec3 {
    light_contribution(
        &sun.light,
        sun.matrix,
        receiver.point,
        receiver.shadow_point,
        normal,
        geometric,
        params,
        shading,
        world_sun_shadow(sun, shadow),
        None,
        sun.ray_visibility,
        thickness,
        sun.shadow_radius,
    )
}

fn translucent_direct_contribution(
    context: &DirectLightContext<'_>,
    color: DVec3,
    thickness: Thickness,
) -> DVec3 {
    transmission_direct_contribution(context, color, thickness, true)
}

/// BSSRDF transmission uses a back-facing `LIGHT_DIFFUSE` lobe while retaining the material thickness for shadow origin.
fn transmission_direct_contribution(
    context: &DirectLightContext<'_>,
    color: DVec3,
    thickness: Thickness,
    is_translucent_with_thickness: bool,
) -> DVec3 {
    if !context.ray_visibility.allows(RayKind::Diffuse) {
        return DVec3::ZERO;
    }
    let Some(direct) = direct_emitter_with_sun_shadow_radius(
        context.light,
        context.matrix,
        context.receiver.point,
        context.sun_shadow_radius,
    ) else {
        return DVec3::ZERO;
    };
    let sphere = is_translucent_with_thickness
        && matches!(thickness, Thickness::Sphere(value) if value != 0.0);
    let lobe_normal = if sphere {
        direct.direction
    } else {
        -context.normal
    };
    let facing = if sphere {
        1.0
    } else {
        facing_attenuation(
            direct.emitter.shape_radius(),
            direct.direction,
            direct.distance,
            lobe_normal,
            true,
        )
    };
    let origin = transmission_origin(
        context.receiver.point,
        context.receiver.geometric,
        direct.direction,
        direct.shadow_distance,
        thickness,
    );
    let visibility = match direct.area {
        Some(shape) => area_transmitted_fraction(
            context.shadow,
            origin,
            &shape,
            (direct.centre, -direct.axis),
            lobe_normal,
        ),
        None => transmitted_visibility(context.shadow, origin, &direct),
    };
    let form = direct.emitter.form_factor(
        lobe_normal,
        context.receiver.view,
        [1.0, 0.0, 0.0, 1.0],
        direct.to_centre,
    );
    let thickness_scale = if sphere {
        1.0 / std::f64::consts::PI
    } else {
        1.0
    };
    color
        * direct.radiance
        * (direct.attenuation
            * facing
            * visibility
            * form
            * LightFactors::of(context.light).diffuse
            * thickness_scale)
}

fn diffuse_direct_contribution(context: &DirectLightContext<'_>) -> DVec3 {
    ltc_direct_contribution(
        context,
        super::eevee_ltc::IDENTITY,
        DVec3::ONE,
        RayKind::Diffuse,
    )
}

fn metallic_direct_contribution(
    context: &DirectLightContext<'_>,
    roughness: f64,
    color: DVec3,
) -> DVec3 {
    let cos_theta = context.normal.dot(context.receiver.view);
    let matrix = super::eevee_ltc::sample_matrix(cos_theta, roughness);
    ltc_direct_contribution(context, matrix, color, RayKind::Glossy)
}

fn ltc_direct_contribution(
    context: &DirectLightContext<'_>,
    matrix: super::eevee_ltc::PackedMatrix,
    color: DVec3,
    ray_kind: RayKind,
) -> DVec3 {
    if !context.ray_visibility.allows(ray_kind) {
        return DVec3::ZERO;
    }
    let factors = LightFactors::of(context.light);
    let light_factor = match ray_kind {
        RayKind::Diffuse => factors.diffuse,
        RayKind::Glossy => factors.specular,
        _ => return DVec3::ZERO,
    };
    let Some(direct) = direct_emitter_with_sun_shadow_radius(
        context.light,
        context.matrix,
        context.receiver.point,
        context.sun_shadow_radius,
    ) else {
        return DVec3::ZERO;
    };
    let facing = facing_attenuation(
        direct.emitter.shape_radius(),
        direct.direction,
        direct.distance,
        context.normal,
        false,
    );
    if facing <= 0.0 {
        return DVec3::ZERO;
    }
    let visibility = match direct.area {
        Some(shape) => area_visible_fraction(
            context.shadow,
            context.point_shadow,
            context.receiver.shadow_point,
            context.receiver.geometric,
            &shape,
            direct.centre,
            -direct.axis,
            context.receiver.point,
            context.normal,
        ),
        None => visibility_for_light(
            context.shadow,
            context.point_shadow,
            context.receiver.shadow_point,
            context.receiver.geometric,
            direct.direction,
            direct.shadow_distance,
            direct.sun_radius,
        ),
    };
    let form = direct.emitter.form_factor(
        context.normal,
        context.receiver.view,
        matrix,
        direct.to_centre,
    );
    let unshadowed = direct.radiance * (direct.attenuation * facing * form * light_factor);
    color * unshadowed * visibility
}

/// A sun's direct light, integrated over the sun's disc as EEVEE's sun is. The diffuse class is flat over the disc, so its
/// centre value times the disc's mean visibility is its integral. The specular class is resolved over the disc: each sample
/// of the disc (`sun_disc_directions`), weighted by its visibility, is one direction of the integral, so a glossy highlight
/// takes the disc's shape as EEVEE's LTC integral does.
fn sun_lighting(
    params: &BsdfParams,
    shading: &Layers,
    shadow: ShadowContext<'_>,
    shadow_point: DVec3,
    geometric: DVec3,
    toward_light: DVec3,
    color: DVec3,
    light: &crate::model::LightData,
    lobes: RayVisibility,
    sun_shadow_radius: Option<f64>,
) -> DVec3 {
    let front_params = if params.thin_wall {
        *params
    } else {
        without_subsurface(params)
    };
    let energy = light.energy.max(0.0);
    let sphere_thickness =
        !params.thickness_slab && params.thickness.is_some_and(|value| value > 0.0);
    let factors = LightFactors::of(light);
    let shape_radius = sun_disc_radius(light.sun_angle);
    let shadow_radius = sun_shadow_radius.unwrap_or(shape_radius);
    let weight = energy / SUN_DISC_SAMPLES as f64;
    let mut mean_visibility = 0.0;
    let mut specular = DVec3::ZERO;
    let mut translucent = DVec3::ZERO;
    for (sample, shadow_sample) in
        sun_disc_direction_pairs(toward_light, shape_radius, shadow_radius)
    {
        let visibility = match shadow {
            Some(shadow) => ray_transmittance(
                shadow,
                shadow_origin(shadow_point, geometric, shadow_sample),
                shadow_sample,
                f64::INFINITY,
                true,
            ),
            None => 1.0,
        };
        mean_visibility += visibility / SUN_DISC_SAMPLES as f64;
        let cosine = shading.normal.dot(sample);
        if cosine > 0.0 {
            let lobe = reflected_lobes_coefficient(
                &front_params,
                shading,
                sample,
                lobes.mask() & RayKind::Glossy.bit(),
            );
            specular += lobe * (weight * cosine * visibility);
        } else if params.thin_wall && !sphere_thickness && cosine < 0.0 {
            let lobe = reflected_lobes_coefficient(
                params,
                shading,
                sample,
                lobes.mask() & RayKind::Diffuse.bit(),
            );
            translucent += lobe * (weight * -cosine * visibility);
        }
    }
    let cosine = shading.normal.dot(toward_light);
    let diffuse = if cosine > 0.0 {
        reflected_lobes_coefficient(
            &front_params,
            shading,
            toward_light,
            lobes.mask() & RayKind::Diffuse.bit(),
        ) * (energy * cosine * mean_visibility)
    } else {
        DVec3::ZERO
    };
    (diffuse * factors.diffuse + specular * factors.specular + translucent * factors.diffuse)
        * color
}

/// Radiance a point, spot or sun light reflects toward the view of `shading`, plus Thin Wall's translucent lobe on the
/// reverse hemisphere. The glass transmission lobes are `glass_direct_contribution`'s.
fn punctual_radiance(
    params: &BsdfParams,
    shading: &Layers,
    light: DVec3,
    color: DVec3,
    intensity: f64,
    factors: LightFactors,
    lobes: RayVisibility,
) -> DVec3 {
    let sphere_thickness =
        !params.thickness_slab && params.thickness.is_some_and(|value| value > 0.0);
    let normal_light_cosine = shading.normal.dot(light).abs();
    if normal_light_cosine <= 0.0 {
        return DVec3::ZERO;
    }
    // The reflective lobes sit above the surface.
    if shading.normal.dot(light) > 0.0 {
        let front_params = if params.thin_wall {
            *params
        } else {
            without_subsurface(params)
        };
        let diffuse = reflected_lobes_coefficient(
            &front_params,
            shading,
            light,
            lobes.mask() & RayKind::Diffuse.bit(),
        );
        let specular = reflected_lobes_coefficient(
            &front_params,
            shading,
            light,
            lobes.mask() & RayKind::Glossy.bit(),
        );
        (diffuse * factors.diffuse + specular * factors.specular)
            * color
            * (intensity * normal_light_cosine)
    } else if params.thin_wall && !sphere_thickness {
        let translucent = reflected_lobes_coefficient(
            params,
            shading,
            light,
            lobes.mask() & RayKind::Diffuse.bit(),
        );
        translucent * factors.diffuse * color * (intensity * normal_light_cosine)
    } else {
        DVec3::ZERO
    }
}

/// The light's closure multipliers, which EEVEE applies per closure class (`eevee_light.cc`, `power[LIGHT_DIFFUSE]`,
/// `power[LIGHT_SPECULAR]`, `power[LIGHT_TRANSMISSION]`). The path tracer does not apply them, as Cycles does not.
#[derive(Clone, Copy)]
struct LightFactors {
    diffuse: f64,
    specular: f64,
    transmission: f64,
}

impl LightFactors {
    fn of(light: &crate::model::LightData) -> Self {
        Self {
            diffuse: light.diffuse_factor.max(0.0),
            specular: light.specular_factor.max(0.0),
            transmission: light.transmission_factor.max(0.0),
        }
    }
}

#[cfg(test)]
mod shading_normal_tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use std::collections::BTreeMap;

    use glam::{DMat4, DVec3};
    use serde_json::json;

    use crate::error::{ErrorCode, PotError};

    use super::{
        ColorSlotReferences, FaceNormalValues, MaterialColors, MeshAttributePrecision,
        ObjectRecord, ShadowScene, blend_corner_normals, carried_color_layers,
        color_layer_references, corner_normals, corner_values,
        eevee_registered_raytrace_layer_summary, extract_geometry, extract_geometry_with_semantics,
        mesh_mikk_tangents, mesh_vertex_positions, ray_transmittance, scene_shader_with_semantics,
        source_has_color_layer, source_quantized_mesh_attribute, subdivide_displaced,
        surface_shader, triangle_corners, triangle_layer_colors, triangle_layer_uvs,
        world_face_normals,
    };
    use crate::render::bvh::Bvh;
    use crate::render::principled_bsdf::{
        diffuse_coefficient, layers as principled_layers, specular_coefficient,
    };
    use crate::render::raster::{
        Camera, EeveeRaytraceFeedback, Projection, StereoMode, SurfaceHit, SurfaceSample,
        SurfaceShader, ViewBasis,
    };
    use crate::shader::BsdfParams;
    use crate::{
        eval::{EvaluationContext, Snapshot},
        geom::{
            Mesh,
            color_layers::{ColorDomain, ColorLayer, ColorStore},
            uv_layers::{UvLayer, UvStore},
        },
        graph::{GraphInterface, GraphKind, GraphLink, GraphNode, GraphSocket, NodeGroup},
        image::{ImageData, ImageInterpolation},
        model::{Id, Material, MaterialLink, RayKind, RayVisibility, SceneDoc, ViewportDisplay},
        render::{
            light_model::{
                AreaShape, area_irradiance_per_radiance, area_radiance, area_visibility_samples,
            },
            raster::{AlphaMode, Mode, Triangle},
        },
    };

    fn node_id() -> Id {
        Id::new("node").expect("valid id")
    }

    #[test]
    fn vertex_color_without_a_layer_name_collects_the_render_layer() {
        let mut doc = SceneDoc::new("scene".to_owned());
        let material_id = Id::new("material").expect("valid material ID");
        let graph_id = Id::new("graph").expect("valid graph ID");
        let mut graph = NodeGroup::new("Graph", GraphKind::Shader);
        graph.nodes.insert(
            Id::new("vertex").expect("valid node ID"),
            GraphNode::new("ShaderNodeVertexColor"),
        );
        doc.node_groups.insert(graph_id.clone(), graph);
        doc.materials.insert(
            material_id.clone(),
            Material {
                node_tree: Some(graph_id.clone()),
                ..Material::default()
            },
        );
        assert!(color_layer_references(&doc, &material_id).uses_render_layer);
        let mut attribute = GraphNode::new("ShaderNodeAttribute");
        attribute
            .properties
            .insert("attribute_type".to_owned(), serde_json::json!("GEOMETRY"));
        attribute
            .properties
            .insert("attribute_name".to_owned(), serde_json::json!("Ghost"));
        doc.node_groups
            .get_mut(&graph_id)
            .expect("graph exists")
            .nodes
            .insert(Id::new("vertex").expect("valid node ID"), attribute);
        let references = color_layer_references(&doc, &material_id);
        assert!(!references.uses_render_layer);
        assert!(references.names.contains("Ghost"));
    }

    #[test]
    fn color_render_designations_and_availability_are_member_local() {
        let store = ColorStore {
            layers: vec![
                ColorLayer {
                    name: "A".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![[0.2, 0.4, 0.8, 1.0]],
                },
                ColorLayer {
                    name: "B".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![[0.7, 0.3, 0.1, 1.0]],
                },
            ],
            active: Some("A".to_owned()),
            render: Some("A".to_owned()),
        };
        let slots = vec![
            ColorSlotReferences {
                names: std::collections::BTreeSet::new(),
                render_layer: Some(0),
                available_mask: Some(std::sync::Arc::from(vec![0b01_u64])),
                requires_hit_shading: true,
                reads_colors: true,
            },
            ColorSlotReferences {
                names: std::collections::BTreeSet::new(),
                render_layer: Some(1),
                available_mask: Some(std::sync::Arc::from(vec![0b10_u64])),
                requires_hit_shading: true,
                reads_colors: true,
            },
            ColorSlotReferences {
                names: std::collections::BTreeSet::from(["Missing".to_owned()]),
                render_layer: None,
                available_mask: Some(std::sync::Arc::from(vec![0b01_u64])),
                requires_hit_shading: true,
                reads_colors: true,
            },
        ];
        let (carried, render_layers) = carried_color_layers(&store, &slots);
        assert_eq!(carried, [0, 1]);
        assert_eq!(render_layers, [Some(0), Some(1), None]);
        assert!(source_has_color_layer(
            slots[0].available_mask.as_deref(),
            0
        ));
        assert!(!source_has_color_layer(
            slots[1].available_mask.as_deref(),
            0
        ));
    }

    fn single_face_mesh(points: [DVec3; 3], smooth: bool) -> Mesh {
        let mut mesh = Mesh::from_positions_and_faces(points.to_vec(), vec![vec![0, 1, 2]])
            .expect("triangle is valid");
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain": "face", "type": "bool", "values": {"f0": smooth}}),
        );
        mesh
    }

    fn unknown_precision_scene(points: [DVec3; 3]) -> crate::error::Result<(SceneDoc, Snapshot)> {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"node.create","id":"body","kind":"plane","params":{"size":1.0}}
                ]
            }),
        )?;
        let mut doc = applied.doc;
        let node_id = Id::new("body")?;
        let data_id = doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane data is missing"))?;
        let mesh = Mesh::from_positions_and_faces(points.to_vec(), vec![vec![0, 1, 2]])
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.to_string()))?;
        let data = doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "plane data block is missing")
        })?;
        data.mesh = Some(mesh);
        for scene in doc.scenes.values_mut() {
            scene.render.use_high_quality_normals = None;
        }
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        Ok((doc, snapshot))
    }

    #[test]
    fn unknown_source_precision_keeps_common_packed_axis_normals() -> crate::error::Result<()> {
        let (doc, snapshot) = unknown_precision_scene([DVec3::ZERO, DVec3::X, DVec3::Y])?;
        let geometry = extract_geometry_with_semantics(
            &doc,
            &snapshot,
            Mode::Beauty,
            false,
            &BTreeMap::new(),
            crate::node_kernels::MathSemantics::Eevee,
        )?;
        let triangle = geometry.triangles.first().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "axis mesh triangle is missing")
        })?;
        assert_eq!(triangle.normals, Some([DVec3::Z; 3]));
        assert!(!triangle.smooth_shading);
        Ok(())
    }

    #[test]
    fn unknown_source_precision_refuses_different_world_normals() -> crate::error::Result<()> {
        let (doc, snapshot) =
            unknown_precision_scene([DVec3::ZERO, DVec3::X, DVec3::new(0.0, 1.0, 2.0)])?;
        let error = extract_geometry_with_semantics(
            &doc,
            &snapshot,
            Mode::Beauty,
            false,
            &BTreeMap::new(),
            crate::node_kernels::MathSemantics::Eevee,
        )
        .err()
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "different packed world normals were accepted without source precision",
            )
        })?;
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            json!("render.eevee.mesh_attribute_precision")
        );
        Ok(())
    }

    #[test]
    fn unknown_source_precision_does_not_change_path_or_solid_mesh_normals()
    -> crate::error::Result<()> {
        let (doc, snapshot) =
            unknown_precision_scene([DVec3::ZERO, DVec3::X, DVec3::new(0.0, 1.0, 2.0)])?;
        let solid = extract_geometry_with_semantics(
            &doc,
            &snapshot,
            Mode::Solid,
            false,
            &BTreeMap::new(),
            crate::node_kernels::MathSemantics::Eevee,
        )?;
        let path = extract_geometry(&doc, &snapshot, Mode::Beauty, true, &BTreeMap::new())?;
        assert!(
            solid
                .triangles
                .iter()
                .chain(&path.triangles)
                .all(|triangle| triangle.normals.is_none() && !triangle.smooth_shading)
        );
        Ok(())
    }

    #[test]
    fn unknown_source_precision_checks_mikk_tangent_outputs() -> crate::error::Result<()> {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0), DVec3::Y], true);
        let triangles = mesh
            .tessellate()
            .map_err(|error| PotError::new(ErrorCode::SceneInvalid, error.to_string()))?;
        let vertex_positions = mesh_vertex_positions(&mesh);
        let uv_store = UvStore {
            layers: vec![UvLayer::new(
                "UVMap",
                vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
            )],
            ..UvStore::default()
        };
        let mut warnings = Vec::new();
        let normals = corner_normals(
            &mesh,
            &vertex_positions,
            DMat4::IDENTITY,
            &node_id(),
            None,
            true,
            MeshAttributePrecision::Unknown,
            &mut warnings,
        );
        assert!(normals.precision_matches);
        let tangents = mesh_mikk_tangents(
            &mesh,
            &triangles,
            &vertex_positions,
            &uv_store,
            normals.local.as_deref(),
            DMat4::from_scale(DVec3::new(4.0, 1.0, 1.0)),
            MeshAttributePrecision::Unknown,
        )?;
        assert!(!tangents.precision_matches);
        Ok(())
    }

    #[test]
    fn absent_uv_rows_use_the_canonical_storage_default() {
        let store = UvStore {
            layers: vec![UvLayer::new("UVMap", vec![None])],
            ..UvStore::default()
        };
        assert_eq!(
            triangle_layer_uvs(&store, 0, [Some(0), Some(1), Some(2)]),
            vec![[[0.0, 1.0]; 3]],
        );
    }

    #[test]
    fn flat_faces_keep_geometric_normals_without_warnings() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)], false);
        let mut warnings = Vec::new();
        let faces = world_face_normals(&mesh, DMat4::IDENTITY, &node_id(), None, &mut warnings);
        assert_eq!(faces, vec![None]);
        assert_eq!(warnings, Vec::<serde_json::Value>::new());
    }

    #[test]
    fn smooth_corner_normals_use_the_inverse_transpose_of_the_object_matrix() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)], true);
        let matrix = DMat4::from_scale(DVec3::new(1.0, 1.0, 3.0));
        let mut warnings = Vec::new();
        let faces = world_face_normals(&mesh, matrix, &node_id(), None, &mut warnings);
        let corners = faces[0].as_ref().expect("smooth face has corner normals");
        let expected = DVec3::new(-3.0, 0.0, -1.0).normalize();
        assert_eq!(corners.len(), 3);
        for corner in corners {
            assert!(corner.abs_diff_eq(expected, 1e-9), "got {corner:?}");
        }
        assert_eq!(warnings, Vec::<serde_json::Value>::new());
    }

    #[test]
    fn realtime_mesh_normals_quantize_each_vertex_before_fragment_interpolation() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)], true);
        let vertex_positions = mesh_vertex_positions(&mesh);
        let raw = [
            DVec3::new(-100.25, -26.75, 500.5).normalize(),
            DVec3::Z,
            DVec3::new(-103.25, 0.0, 500.25).normalize(),
        ];
        let adaptive = raw.map(Some);
        let mut warnings = Vec::new();
        let data = corner_normals(
            &mesh,
            &vertex_positions,
            DMat4::IDENTITY,
            &node_id(),
            Some(&adaptive),
            true,
            MeshAttributePrecision::Known(false),
            &mut warnings,
        );
        let corners = data.world[0]
            .as_ref()
            .and_then(FaceNormalValues::as_corners)
            .expect("smooth face has VBO normals");
        let codes = [[-100, -26, 500], [0, 0, 511], [-103, 0, 500]];
        let decoded = codes.map(|[x, y, z]| {
            DVec3::new(
                f64::from(x as f32 / 511.0),
                f64::from(y as f32 / 511.0),
                f64::from(z as f32 / 511.0),
            )
            .normalize()
        });
        for (actual, expected) in corners.iter().zip(decoded) {
            assert!(actual.abs_diff_eq(expected, 1e-12), "got {actual:?}");
        }
        let local = data.local.as_ref().expect("Mikk keeps raw local normals");
        assert_eq!(
            local[0].as_ref().and_then(FaceNormalValues::as_corners),
            Some(raw.as_slice())
        );

        let actual = blend_corner_normals(
            [corners[0], corners[1], corners[2]],
            [1.0 / 3.0; 3],
            DVec3::Z,
        );
        let expected = (decoded[0] + decoded[1] + decoded[2]).normalize();
        assert!(actual.abs_diff_eq(expected, 1e-12));
        let interpolated_before_packing =
            source_quantized_mesh_attribute((raw[0] + raw[1] + raw[2]) / 3.0, false).normalize();
        assert!(!actual.abs_diff_eq(interpolated_before_packing, 1e-6));
        let high_quality = corner_normals(
            &mesh,
            &vertex_positions,
            DMat4::IDENTITY,
            &node_id(),
            Some(&adaptive),
            false,
            MeshAttributePrecision::Known(true),
            &mut warnings,
        );
        let high_quality_codes = [[-6426, -1714, 32084], [0, 0, 32767], [-6623, 0, 32090]];
        for (actual, [x, y, z]) in high_quality.world[0]
            .as_ref()
            .and_then(FaceNormalValues::as_corners)
            .expect("high-quality normals use the 16-bit VBO")
            .iter()
            .zip(high_quality_codes)
        {
            let expected = DVec3::new(
                f64::from(x as f32 / 32767.0),
                f64::from(y as f32 / 32767.0),
                f64::from(z as f32 / 32767.0),
            )
            .normalize();
            assert!(actual.abs_diff_eq(expected, 1e-12), "got {actual:?}");
        }
        let high_quality_corners = high_quality.world[0]
            .as_ref()
            .and_then(FaceNormalValues::as_corners)
            .expect("high-quality face keeps VBO normals");
        assert!(!high_quality_corners[0].abs_diff_eq(raw[0], 1e-8));
    }

    #[test]
    fn realtime_mesh_precision_packs_flat_face_normals() {
        let points = [DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)];
        let mesh = single_face_mesh(points, false);
        let vertex_positions = mesh_vertex_positions(&mesh);
        let mut warnings = Vec::new();
        let data = corner_normals(
            &mesh,
            &vertex_positions,
            DMat4::IDENTITY,
            &node_id(),
            None,
            false,
            MeshAttributePrecision::Known(false),
            &mut warnings,
        );
        let actual = data.world[0]
            .as_ref()
            .and_then(FaceNormalValues::constant)
            .expect("flat face has one VBO normal");
        let face_normal = crate::geom::normals::face_normal(&points);
        let expected = source_quantized_mesh_attribute(face_normal, false).normalize();
        assert!(actual.abs_diff_eq(expected, 1e-12));
    }

    #[test]
    fn realtime_mesh_tangents_pack_mikk_output_after_local_normal_generation() {
        let points = [DVec3::ZERO, DVec3::new(3.0, 4.0, 0.0), DVec3::Y];
        let mesh = single_face_mesh(points, true);
        let triangles = mesh.tessellate().expect("triangle tessellates");
        let vertex_positions = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.id, vertex.co))
            .collect::<BTreeMap<_, _>>();
        let uv_store = UvStore {
            layers: vec![UvLayer::new(
                "UVMap",
                vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
            )],
            ..UvStore::default()
        };
        let mut warnings = Vec::new();
        let object_transform = DMat4::from_scale(DVec3::new(4.0, 1.0, 1.0));
        let normals = corner_normals(
            &mesh,
            &vertex_positions,
            DMat4::IDENTITY,
            &node_id(),
            None,
            true,
            MeshAttributePrecision::Known(false),
            &mut warnings,
        );
        let local_normals = normals.local.as_deref();
        let path_tangents = mesh_mikk_tangents(
            &mesh,
            &triangles,
            &vertex_positions,
            &uv_store,
            local_normals,
            object_transform,
            MeshAttributePrecision::NotApplicable,
        )
        .expect("Path keeps unquantized Mikk tangents")
        .by_triangle;
        let low_quality = mesh_mikk_tangents(
            &mesh,
            &triangles,
            &vertex_positions,
            &uv_store,
            local_normals,
            object_transform,
            MeshAttributePrecision::Known(false),
        )
        .expect("low-quality Source packs the Mikk output")
        .by_triangle;
        let high_quality = mesh_mikk_tangents(
            &mesh,
            &triangles,
            &vertex_positions,
            &uv_store,
            local_normals,
            object_transform,
            MeshAttributePrecision::Known(true),
        )
        .expect("high-quality Source packs the Mikk output")
        .by_triangle;

        let path_tangent = path_tangents[0][0][0];
        assert!(path_tangent.abs_diff_eq(DVec3::new(0.15, 0.8, 0.0), 1e-6));
        assert!(low_quality[0][0][0].abs_diff_eq(
            DVec3::new(
                f64::from(306_f32 / 511.0) / 4.0,
                f64::from(408_f32 / 511.0),
                0.0,
            ),
            1e-12,
        ));
        assert!(high_quality[0][0][0].abs_diff_eq(
            DVec3::new(
                f64::from(19_660_f32 / 32_767.0) / 4.0,
                f64::from(26_213_f32 / 32_767.0),
                0.0,
            ),
            1e-12,
        ));
    }

    #[test]
    fn adaptive_normals_override_smooth_faces_but_not_flat_faces() {
        let points = [DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)];
        let smooth = single_face_mesh(points, true);
        let flat = single_face_mesh(points, false);
        let adaptive = [Some(DVec3::X); 3];
        let mut warnings = Vec::new();

        let smooth_faces = world_face_normals(
            &smooth,
            DMat4::IDENTITY,
            &node_id(),
            Some(&adaptive),
            &mut warnings,
        );
        assert_eq!(smooth_faces, vec![Some(vec![DVec3::X; 3])]);

        let flat_faces = world_face_normals(
            &flat,
            DMat4::IDENTITY,
            &node_id(),
            Some(&adaptive),
            &mut warnings,
        );
        assert_eq!(flat_faces, vec![None]);
        assert_eq!(warnings, Vec::<serde_json::Value>::new());
    }

    #[test]
    fn degenerate_smooth_fans_fall_back_to_flat_with_a_warning() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::X, DVec3::new(2.0, 0.0, 0.0)], true);
        let mut warnings = Vec::new();
        let faces = world_face_normals(&mesh, DMat4::IDENTITY, &node_id(), None, &mut warnings);
        assert_eq!(faces, vec![None]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "shading.corner_normals_failed");
        assert_eq!(warnings[0]["node_id"], "node");
    }

    #[test]
    fn singular_object_transforms_warn_and_fall_back_to_flat() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::X, DVec3::Y], true);
        let mut warnings = Vec::new();
        let matrix = DMat4::from_scale(DVec3::new(1.0, 1.0, 0.0));
        let faces = world_face_normals(&mesh, matrix, &node_id(), None, &mut warnings);
        assert_eq!(faces, vec![None]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "shading.corner_normals_failed");
    }

    #[test]
    fn triangle_layer_colors_read_corners_from_their_face_and_points_from_their_slot() {
        let corner = ColorLayer {
            name: "Paint".to_owned(),
            domain: ColorDomain::Corner,
            // Face 0 owns storage corners 0..3 and face 1 owns 3..6.
            values: vec![
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.25, 0.25, 0.25, 1.0],
                [0.5, 0.5, 0.5, 1.0],
                [1.0, 1.0, 1.0, 0.5],
            ],
        };
        let point = ColorLayer {
            name: "Col".to_owned(),
            domain: ColorDomain::Point,
            values: vec![[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 1.0]],
        };
        // Vertex 9 is storage slot 0 and vertex 7 is slot 1.
        let slots = std::collections::BTreeMap::from([(9_u32, 0_usize), (7, 1)]);
        let colors = triangle_layer_colors(
            &[&corner, &point],
            0,
            3,
            [Some(2), Some(0), Some(1)],
            [9, 7, 7],
            &slots,
        );
        assert_eq!(
            colors[0],
            [
                [1.0, 1.0, 1.0, 0.5],
                [0.25, 0.25, 0.25, 1.0],
                [0.5, 0.5, 0.5, 1.0],
            ]
        );
        assert_eq!(
            colors[1],
            [
                [0.1, 0.2, 0.3, 1.0],
                [0.4, 0.5, 0.6, 1.0],
                [0.4, 0.5, 0.6, 1.0],
            ]
        );

        // A corner or vertex with no stored value reads as opaque black.
        let missing = triangle_layer_colors(
            &[&corner, &point],
            0,
            3,
            [None, Some(0), Some(1)],
            [9, 8, 7],
            &slots,
        );
        assert_eq!(missing[0][0], [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(missing[1][1], [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn flat_graph_triangle_renders_asymmetric_texture_and_emission_per_hit()
    -> crate::error::Result<()> {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"image.create","id":"asym","width":2,"height":2,"fill_color":[0.0,0.0,0.0,1.0]},
                    {
                        "op":"material.create","id":"paint",
                        "graph": {
                            "kind":"shader",
                            "nodes": {
                                "uv":{"type":"ShaderNodeUVMap","properties":{"uv_map":"Base"}},
                                "texture":{"type":"ShaderNodeTexImage","properties":{"image":"asym","interpolation":"Closest"}},
                                "surface":{"type":"ShaderNodeBsdfPrincipled","inputs":{"Emission Strength":1.0,"Metallic":0.0,"Roughness":1.0}},
                                "output":{"type":"OutputMaterial"}
                            },
                            "links": [
                                {"from_node":"uv","from_socket":"UV","to_node":"texture","to_socket":"Vector"},
                                {"from_node":"texture","from_socket":"Color","to_node":"surface","to_socket":"Base Color"},
                                {"from_node":"texture","from_socket":"Color","to_node":"surface","to_socket":"Emission Color"},
                                {"from_node":"surface","from_socket":"BSDF","to_node":"output","to_socket":"Surface"}
                            ]
                        }
                    },
                    {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0},"material":"paint"}
                ]
            }),
        )?;
        let mut doc = applied.doc;
        let node_id = Id::new("body")?;
        let data_id = doc.nodes[&node_id].data.clone().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "flat texture mesh data is missing",
            )
        })?;
        let mesh = doc
            .data_blocks
            .get_mut(&data_id)
            .and_then(|data| data.mesh.as_mut())
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "flat texture mesh is missing")
            })?;
        let uv = UvLayer::new(
            "Base",
            mesh.faces
                .iter()
                .map(|face| Some(vec![[0.75, 0.25]; face.vertices.len()]))
                .collect(),
        );
        crate::geom::uv_layers::write(
            mesh,
            &UvStore {
                layers: vec![uv],
                active: Some("Base".to_owned()),
                render: Some("Base".to_owned()),
            },
        );
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let pixel = [
            [0.05, 0.9, 0.2, 1.0],
            [0.7, 0.1, 0.2, 1.0],
            [0.25, 0.4, 0.8, 1.0],
            [0.8, 0.6, 0.1, 1.0],
        ];
        let images = BTreeMap::from([(
            "asym".to_owned(),
            ImageData {
                colour: crate::image::ColourCache::default(),
                storage: crate::image::ImageStorage::default(),
                width: 2,
                height: 2,
                pixels: pixel.to_vec(),
                tiles: BTreeMap::new(),
                interpolation: ImageInterpolation::Closest,
                srgb: false,
                udim: false,
            },
        )]);
        let geometry = extract_geometry(&doc, &snapshot, Mode::Beauty, false, &images)?;
        let triangle = geometry.triangles.first().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "flat texture triangle is missing")
        })?;
        assert!(
            triangle.normals.is_none(),
            "the regression requires a flat face"
        );
        assert!(
            triangle.lit_per_pixel,
            "flat graph materials must be evaluated per hit"
        );
        let view = ViewBasis::from_camera(&Camera {
            position: DVec3::new(0.0, 0.0, 2.5),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 2.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        })?;
        let shader = surface_shader(&doc, &snapshot, &images, &geometry, Mode::Beauty, &view)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "Beauty shader is missing"))?;
        let point = triangle.positions.iter().copied().sum::<DVec3>() / 3.0;
        let color = shader
            .shade(
                triangle,
                &SurfaceHit {
                    point,
                    shadow_point: point,
                    normal: DVec3::Z,
                    geometric: DVec3::Z,
                    geometric_normal: DVec3::Z,
                    back_facing: false,
                    realtime_differentials: crate::shader::HitDifferentials {
                        world_dx: DVec3::ZERO,
                        world_dy: DVec3::ZERO,
                        uv_dx: [0.0; 2],
                        uv_dy: [0.0; 2],
                    },
                    camera_matrices: None,
                    barycentric: [1.0 / 3.0; 3],
                    gbuffer_frag_coord: None,
                    gbuffer_sample_offset: None,
                    gbuffer_closure_random: None,
                    view: DVec3::Z,
                    eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
                },
            )?
            .color;
        // No lights and no world: the hit returns its emission alone, as Cycles does. The old
        // Potter-only `albedo * 0.08` ambient fallback is gone.
        let expected_rgb = [0.7, 0.1, 0.2];
        for (actual, expected) in color[..3].iter().zip(expected_rgb) {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "asymmetric texture/emission mismatch: {color:?}"
            );
        }
        assert_eq!(color[3], 1.0);
        Ok(())
    }

    #[test]
    fn beauty_triangles_take_double_sided_from_their_material_or_default_surface()
    -> crate::error::Result<()> {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"material.create","id":"single","double_sided":false},
                    {"op":"material.create","id":"double"},
                    {"op":"node.create","id":"single_body","kind":"plane","params":{"size":1.0},"material":"single"},
                    {"op":"node.create","id":"double_body","kind":"plane","params":{"size":1.0},"transform":{"translation":[3.0,0.0,0.0]},"material":"double"},
                    {"op":"node.create","id":"bare_body","kind":"plane","params":{"size":1.0},"transform":{"translation":[6.0,0.0,0.0]}}
                ]
            }),
        )?;
        let snapshot = Snapshot::evaluate(&applied.doc, &EvaluationContext::default())?;
        let geometry = extract_geometry(
            &applied.doc,
            &snapshot,
            Mode::Beauty,
            false,
            &BTreeMap::new(),
        )?;
        // Each node's triangles carry the object index recorded for that node.
        let expected = [
            ("single_body", false),
            ("double_body", true),
            ("bare_body", true),
        ];
        for (node, double_sided) in expected {
            let node_id = Id::new(node).expect("valid id");
            let object = geometry
                .objects
                .iter()
                .find(|object| object.node_id == node_id)
                .expect("node is rendered");
            let triangles = geometry
                .triangles
                .iter()
                .filter(|triangle| triangle.object_index == object.index)
                .collect::<Vec<_>>();
            // Beauty splits triangles that are not lit per pixel, so the count varies by material.
            assert!(!triangles.is_empty(), "{node} has no triangles");
            assert!(
                triangles
                    .iter()
                    .all(|triangle| triangle.double_sided == double_sided),
                "{node} double_sided should be {double_sided}"
            );
        }
        Ok(())
    }

    fn source_triangle(positions: [DVec3; 3], normals: [DVec3; 3]) -> Triangle {
        Triangle {
            positions,
            generated: crate::geom::GeneratedCorners::default(),
            geometry_vertex_slots: [0; 3],
            geometry_vertex_weights: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            object_index: 1,
            element_index: 1,
            color: [1.0; 4],
            alpha_mode: AlphaMode::Opaque,
            alpha_threshold: 0.5,
            double_sided: false,
            material_index: 0,
            uv: [[0.0; 2]; 3],
            layer_uvs: Vec::new(),
            layer_tangents: Vec::new(),
            layer_colors: Vec::new(),
            normals: Some(normals),
            smooth_shading: true,
            lit_per_pixel: false,
            ray_mask: RayVisibility::default().mask(),
        }
    }
    fn source_shader_material(
        doc: &mut SceneDoc,
        name: &str,
        node_type: &str,
        output_socket: &str,
        inputs: &serde_json::Value,
        use_raytrace_refraction: Option<bool>,
    ) -> Material {
        let graph_id = Id::new(format!("{name}_graph")).expect("valid graph id");
        let shader_id = Id::new(format!("{name}_shader")).expect("valid shader id");
        let output_id = Id::new(format!("{name}_output")).expect("valid output id");
        let mut graph = NodeGroup::new(name, GraphKind::Shader);
        let mut shader = GraphNode::new(node_type);
        shader.inputs = inputs
            .as_object()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        graph.nodes.insert(shader_id.clone(), shader);
        let mut output = GraphNode::new("ShaderNodeOutputMaterial");
        output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        graph.nodes.insert(output_id.clone(), output);
        graph.links.push(GraphLink {
            from_node: shader_id,
            from_socket: output_socket.to_owned(),
            to_node: output_id,
            to_socket: "Surface".to_owned(),
        });
        doc.node_groups.insert(graph_id.clone(), graph);
        Material {
            name: name.to_owned(),
            node_tree: Some(graph_id),
            use_raytrace_refraction,
            ..Material::default()
        }
    }
    #[test]
    fn eevee_registered_slots_route_unused_bindings_and_null_fallback() -> crate::error::Result<()>
    {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"node.create","id":"body","kind":"plane","params":{"size":1.0}}
                ]
            }),
        )?;
        let mut doc = applied.doc;
        let object_refraction_id = Id::new("object_refraction")?;
        let data_refraction_id = Id::new("data_refraction")?;
        let data_reflection_id = Id::new("data_reflection")?;
        let outputless_id = Id::new("outputless")?;
        let object_refraction = source_shader_material(
            &mut doc,
            "object_refraction_graph",
            "ShaderNodeBsdfRefraction",
            "BSDF",
            &json!({}),
            Some(false),
        );
        let data_refraction = source_shader_material(
            &mut doc,
            "data_refraction_graph",
            "ShaderNodeBsdfRefraction",
            "BSDF",
            &json!({}),
            Some(true),
        );
        let data_reflection = source_shader_material(
            &mut doc,
            "data_reflection_graph",
            "ShaderNodeBsdfGlossy",
            "BSDF",
            &json!({}),
            Some(true),
        );
        let outputless = source_volume_only_material(&mut doc, "outputless", true);
        doc.materials
            .insert(object_refraction_id.clone(), object_refraction);
        doc.materials
            .insert(data_refraction_id.clone(), data_refraction);
        doc.materials
            .insert(data_reflection_id.clone(), data_reflection);
        doc.materials.insert(outputless_id.clone(), outputless);

        let node_id = Id::new("body")?;
        let data_id = doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane data is missing"))?;
        let data = doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "plane data block is missing")
        })?;
        let face = data
            .mesh
            .as_mut()
            .and_then(|mesh| mesh.faces.first_mut())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane face is missing"))?;
        face.material_index = 5;
        data.materials = vec![
            Some(data_refraction_id),
            Some(data_reflection_id),
            None,
            Some(outputless_id),
        ];
        let node = doc
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane object is missing"))?;
        node.materials = vec![Some(object_refraction_id)];
        node.material_links = vec![MaterialLink::Object];

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let images = BTreeMap::new();
        let geometry = extract_geometry_with_semantics(
            &doc,
            &snapshot,
            Mode::Beauty,
            false,
            &images,
            crate::node_kernels::MathSemantics::Eevee,
        )?;
        let view = ViewBasis::from_camera(&Camera {
            position: DVec3::new(0.0, 0.0, 2.5),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 2.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        })?;
        let shader = scene_shader_with_semantics(
            &doc,
            &snapshot,
            &images,
            &geometry,
            Mode::Beauty,
            &view,
            crate::node_kernels::MathSemantics::Eevee,
        );
        let summary = shader.eevee_raytrace_layer_flags()?.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "EEVEE layer summary is missing")
        })?;
        assert!(summary.opaque.has_reflect_closure);
        assert!(summary.opaque.has_transmit_closure);
        let refraction = summary.refraction.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "refraction layer is missing")
        })?;
        assert!(refraction.has_any_closure);
        assert!(refraction.has_reflect_closure);
        assert!(!refraction.has_transmit_closure);
        assert!(!refraction.has_transmission);
        Ok(())
    }

    fn source_simple_principled_material(doc: &mut SceneDoc, mut material: Material) -> Material {
        let name = material.name.clone();
        let graph_id = Id::new(format!("{name}_graph")).expect("valid simple graph id");
        let principled_id = Id::new(format!("{name}_principled")).expect("valid principled id");
        let output_id = Id::new(format!("{name}_output")).expect("valid output id");
        let mut graph = NodeGroup::new(name, GraphKind::Shader);
        let mut principled = GraphNode::new("ShaderNodeBsdfPrincipled");
        principled
            .properties
            .insert("potter_simple_material".to_owned(), json!(true));
        graph.nodes.insert(principled_id.clone(), principled);
        let mut output = GraphNode::new("ShaderNodeOutputMaterial");
        output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        graph.nodes.insert(output_id.clone(), output);
        graph.links.push(GraphLink {
            from_node: principled_id,
            from_socket: "BSDF".to_owned(),
            to_node: output_id,
            to_socket: "Surface".to_owned(),
        });
        doc.node_groups.insert(graph_id.clone(), graph);
        material.node_tree = Some(graph_id);
        material
    }

    fn source_volume_only_material(
        doc: &mut SceneDoc,
        name: &str,
        use_raytrace_refraction: bool,
    ) -> Material {
        let graph_id = Id::new(format!("{name}_graph")).expect("valid graph id");
        let volume_id = Id::new(format!("{name}_volume")).expect("valid volume node id");
        let output_id = Id::new(format!("{name}_output")).expect("valid output id");
        let mut graph = NodeGroup::new(name, GraphKind::Shader);
        graph.nodes.insert(
            volume_id.clone(),
            GraphNode::new("ShaderNodeVolumePrincipled"),
        );
        let mut output = GraphNode::new("ShaderNodeOutputMaterial");
        output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        graph.nodes.insert(output_id.clone(), output);
        graph.links.push(GraphLink {
            from_node: volume_id,
            from_socket: "Volume".to_owned(),
            to_node: output_id,
            to_socket: "Volume".to_owned(),
        });
        doc.node_groups.insert(graph_id.clone(), graph);
        Material {
            name: name.to_owned(),
            node_tree: Some(graph_id),
            use_raytrace_refraction: Some(use_raytrace_refraction),
            ..Material::default()
        }
    }

    fn source_registered_layer_summary(
        doc: &mut SceneDoc,
        materials: Vec<Material>,
        face_material_index: u32,
        ray_visibility: Option<RayVisibility>,
    ) -> crate::error::Result<crate::render::raster::EeveeRaytraceLayerSummary> {
        let mut slots = Vec::with_capacity(materials.len());
        for (index, material) in materials.into_iter().enumerate() {
            let material_id = Id::new(format!("source_slot_{index}"))?;
            doc.materials.insert(material_id.clone(), material);
            slots.push(Some(material_id));
        }
        let base_revision = doc.revision;
        let applied = crate::ops::apply_batch(
            doc,
            &json!({
                "schema_version": 1,
                "base_revision": base_revision,
                "operations": [
                    {"op":"node.create","id":"source_body","kind":"plane","params":{"size":1.0}}
                ]
            }),
        )?;
        let mut doc = applied.doc;
        let node_id = Id::new("source_body")?;
        let data_id = doc
            .nodes
            .get(&node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane data is missing"))?;
        let data = doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "plane data block is missing")
        })?;
        let face = data
            .mesh
            .as_mut()
            .and_then(|mesh| mesh.faces.first_mut())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane face is missing"))?;
        face.material_index = face_material_index;
        data.materials = slots;
        let node = doc
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "plane object is missing"))?;
        node.ray_visibility = ray_visibility;

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let geometry = extract_geometry_with_semantics(
            &doc,
            &snapshot,
            Mode::Beauty,
            false,
            &BTreeMap::new(),
            crate::node_kernels::MathSemantics::Eevee,
        )?;
        eevee_registered_raytrace_layer_summary(&doc, &snapshot, &geometry)
    }

    #[test]
    fn volume_only_slot_keeps_source_layer_emission_presence() -> crate::error::Result<()> {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let material = source_volume_only_material(&mut doc, "volume", true);
        let summary = source_registered_layer_summary(&mut doc, vec![material], 0, None)?;
        let refraction = summary.refraction.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "refraction layer is missing")
        })?;
        assert!(refraction.has_any_closure);
        assert!(!refraction.has_reflect_closure);
        assert!(!refraction.has_transmit_closure);
        Ok(())
    }

    #[test]
    fn simple_material_fields_compile_through_source_principled_flags() -> crate::error::Result<()>
    {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let generated_glass = source_simple_principled_material(
            &mut doc,
            Material {
                name: "generated_glass".to_owned(),
                transmission: 1.0,
                use_raytrace_refraction: Some(true),
                ..Material::default()
            },
        );
        let clip = Material {
            name: "simple_clip".to_owned(),
            alpha_mode: "clip".to_owned(),
            base_color: [0.7, 0.7, 0.7, 0.25],
            alpha_threshold: 0.5,
            ..Material::default()
        };
        let shader_to_rgb = source_shader_material(
            &mut doc,
            "simple_shader_to_rgb",
            "ShaderNodeShaderToRGB",
            "Color",
            &json!({}),
            Some(false),
        );
        let summary = source_registered_layer_summary(
            &mut doc,
            vec![clip, shader_to_rgb, generated_glass],
            2,
            None,
        )?;
        assert!(summary.opaque.has_transmission);
        assert!(!summary.opaque.has_transmit_closure);
        let refraction = summary.refraction.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "refraction layer is missing")
        })?;
        assert!(refraction.has_transmit_closure);
        assert!(refraction.has_reflect_closure);
        Ok(())
    }

    #[test]
    fn deferred_layer_group_inputs_use_inlined_default_link_state() -> crate::error::Result<()> {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let child_id = Id::new("transparent_group".to_owned()).expect("valid group id");
        let root_id = Id::new("group_material".to_owned()).expect("valid graph id");
        let mut child = NodeGroup::new("Transparent Group", GraphKind::Shader);
        child.interface = GraphInterface {
            inputs: vec![GraphSocket {
                id: "Color".to_owned(),
                name: "Color".to_owned(),
                socket_type: "NodeSocketColor".to_owned(),
                default: json!([0.0, 0.0, 0.0, 1.0]),
            }],
            outputs: vec![GraphSocket {
                id: "Shader".to_owned(),
                name: "Shader".to_owned(),
                socket_type: "shader".to_owned(),
                default: serde_json::Value::Null,
            }],
        };
        let input_id = Id::new("input".to_owned()).expect("valid node id");
        let transparent_id = Id::new("transparent".to_owned()).expect("valid shader id");
        let output_id = Id::new("group_output".to_owned()).expect("valid output id");
        child
            .nodes
            .insert(input_id.clone(), GraphNode::new("NodeGroupInput"));
        child.nodes.insert(
            transparent_id.clone(),
            GraphNode::new("ShaderNodeBsdfTransparent"),
        );
        let mut child_output = GraphNode::new("NodeGroupOutput");
        child_output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        child.nodes.insert(output_id.clone(), child_output);
        child.links.push(GraphLink {
            from_node: input_id,
            from_socket: "Color".to_owned(),
            to_node: transparent_id.clone(),
            to_socket: "Color".to_owned(),
        });
        child.links.push(GraphLink {
            from_node: transparent_id,
            from_socket: "BSDF".to_owned(),
            to_node: output_id,
            to_socket: "Shader".to_owned(),
        });
        doc.node_groups.insert(child_id.clone(), child);

        let group_node_id = Id::new("child".to_owned()).expect("valid node id");
        let material_output_id = Id::new("output".to_owned()).expect("valid output id");
        let mut root = NodeGroup::new("Group Material", GraphKind::Shader);
        let mut group_node = GraphNode::new("ShaderNodeGroup");
        group_node
            .properties
            .insert("node_tree".to_owned(), json!(child_id.as_str()));
        group_node
            .inputs
            .insert("Color".to_owned(), json!([0.0, 0.0, 0.0, 1.0]));
        root.nodes.insert(group_node_id.clone(), group_node);
        let mut output = GraphNode::new("ShaderNodeOutputMaterial");
        output
            .properties
            .insert("is_active_output".to_owned(), json!(true));
        root.nodes.insert(material_output_id.clone(), output);
        root.links.push(GraphLink {
            from_node: group_node_id,
            from_socket: "Shader".to_owned(),
            to_node: material_output_id,
            to_socket: "Surface".to_owned(),
        });
        doc.node_groups.insert(root_id.clone(), root);
        let group_material = Material {
            name: "Group Material".to_owned(),
            node_tree: Some(root_id),
            use_raytrace_refraction: Some(false),
            ..Material::default()
        };
        let shader_to_rgb = source_shader_material(
            &mut doc,
            "shader_to_rgb",
            "ShaderNodeShaderToRGB",
            "Color",
            &json!({}),
            Some(false),
        );
        let summary = source_registered_layer_summary(
            &mut doc,
            vec![group_material, shader_to_rgb],
            1,
            None,
        )?;
        assert!(!summary.opaque.has_transmission);
        assert!(!summary.opaque.has_transmit_closure);
        Ok(())
    }

    #[test]
    fn deferred_layer_flags_keep_camera_hidden_zero_weight_reflection() -> crate::error::Result<()>
    {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let refraction = source_shader_material(
            &mut doc,
            "visible_refraction",
            "ShaderNodeBsdfRefraction",
            "BSDF",
            &json!({}),
            Some(true),
        );
        let glossy = source_shader_material(
            &mut doc,
            "occluded_zero_weight_glossy",
            "ShaderNodeBsdfGlossy",
            "BSDF",
            &json!({"Weight":0.0}),
            Some(false),
        );
        let mut visibility = RayVisibility::default();
        visibility.set(RayKind::Camera, false);
        let summary = source_registered_layer_summary(
            &mut doc,
            vec![refraction, glossy],
            1,
            Some(visibility),
        )?;
        let refraction_layer = summary.refraction.ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "refraction layer is missing")
        })?;
        assert!(summary.opaque.has_reflect_closure);
        assert!(refraction_layer.has_transmit_closure);
        assert!(!refraction_layer.has_reflect_closure);
        Ok(())
    }

    #[test]
    fn deferred_layer_transmission_union_keeps_shader_to_rgb_distinct() -> crate::error::Result<()>
    {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let transparent = source_shader_material(
            &mut doc,
            "transparent",
            "ShaderNodeBsdfTransparent",
            "BSDF",
            &json!({"Color":[1.0,1.0,1.0,1.0]}),
            Some(false),
        );
        let shader_to_rgb = source_shader_material(
            &mut doc,
            "shader_to_rgb",
            "ShaderNodeShaderToRGB",
            "Color",
            &json!({}),
            Some(false),
        );
        let unused_refraction = source_shader_material(
            &mut doc,
            "unused_refraction_slot",
            "ShaderNodeBsdfRefraction",
            "BSDF",
            &json!({}),
            Some(true),
        );
        let summary = source_registered_layer_summary(
            &mut doc,
            vec![transparent, shader_to_rgb, unused_refraction],
            1,
            None,
        )?;
        assert!(summary.opaque.has_transmission);
        assert!(!summary.opaque.has_transmit_closure);
        assert!(summary.refraction.is_none());
        Ok(())
    }

    #[test]
    fn deferred_layer_sss_is_broad_transmission_not_a_transmit_closure() -> crate::error::Result<()>
    {
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let sss = source_shader_material(
            &mut doc,
            "subsurface",
            "ShaderNodeSubsurfaceScattering",
            "BSSRDF",
            &json!({}),
            Some(false),
        );
        let summary = source_registered_layer_summary(&mut doc, vec![sss], 0, None)?;
        assert!(summary.opaque.has_reflect_closure);
        assert!(!summary.opaque.has_transmit_closure);
        assert!(summary.opaque.has_transmission);
        Ok(())
    }

    #[test]
    fn triangle_vertices_map_to_their_own_face_corner_normals() {
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::new(1.0, 1.0, 0.0), DVec3::Y],
            vec![vec![0, 1, 2, 3]],
        )
        .expect("quad is valid");
        let face = &mesh.faces[0];
        // The second triangle of the quad, (v0, v2, v3), uses corners 0, 2 and 3.
        let ids = [face.vertices[0], face.vertices[2], face.vertices[3]];
        let corners = triangle_corners(face, &ids);
        assert_eq!(corners, [Some(0), Some(2), Some(3)]);
        let normals = [DVec3::X, DVec3::Y, DVec3::Z, DVec3::NEG_X];
        assert_eq!(
            corner_values(&normals, corners),
            Some([DVec3::X, DVec3::Z, DVec3::NEG_X])
        );
        assert_eq!(corner_values(&normals, [Some(0), None, Some(2)]), None);
    }

    #[test]
    fn displaced_neighbours_share_edge_points_and_shade_with_their_own_face_normals() {
        let tilt = DVec3::new(0.2, 0.1, 1.0).normalize();
        // The second triangle is not coplanar with the first, so face normals differ.
        let first = source_triangle([DVec3::ZERO, DVec3::X, DVec3::Y], [tilt; 3]);
        let second = source_triangle([DVec3::X, DVec3::new(1.0, 1.0, 0.5), DVec3::Y], [tilt; 3]);
        let first_out = subdivide_displaced(&first, DMat4::IDENTITY, |_| Ok(0.1)).expect("first");
        let second_out =
            subdivide_displaced(&second, DMat4::IDENTITY, |_| Ok(0.1)).expect("second");
        let shared_edge = [
            DVec3::X + tilt * 0.1,
            DVec3::Y + tilt * 0.1,
            (DVec3::X + DVec3::Y) * 0.5 + tilt * 0.1,
        ];
        for point in shared_edge {
            for output in [&first_out, &second_out] {
                assert!(
                    output
                        .iter()
                        .flat_map(|triangle| triangle.positions)
                        .any(|corner| corner.abs_diff_eq(point, 1e-12)),
                    "shared displaced point {point:?} missing"
                );
            }
        }
        for sub in first_out.iter().chain(&second_out) {
            let [a, b, c] = sub.positions;
            let face = (b - a).cross(c - a).normalize();
            let face = if face.dot(tilt) < 0.0 { -face } else { face };
            let normals = sub
                .normals
                .expect("smooth sub-triangle keeps shading normals");
            assert!(normals.iter().all(|normal| normal.abs_diff_eq(face, 1e-9)));
        }
    }

    #[test]
    fn solid_blends_a_viewport_material_only_below_full_viewport_alpha() -> crate::error::Result<()>
    {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let mut doc = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}}
                ]
            }),
        )?
        .doc;
        let body_id = Id::new("body")?;
        let material_id = Id::new("glass")?;
        let data_id = doc.nodes[&body_id]
            .data
            .clone()
            .expect("plane has mesh data");
        doc.data_blocks
            .get_mut(&data_id)
            .expect("plane data block exists")
            .materials = vec![Some(material_id.clone())];
        let body = doc.nodes.get_mut(&body_id).expect("body exists");
        body.materials = vec![None];
        body.material_links = vec![MaterialLink::Data];
        // Solid reads the viewport alpha; Beauty keeps the node tree's alpha mode ("opaque" here).
        for (alpha, transparent) in [(0.5, true), (1.0, false)] {
            doc.materials.insert(
                material_id.clone(),
                Material {
                    name: "Glass".to_owned(),
                    base_color: [0.1, 0.6, 0.9, 1.0],
                    viewport_display: Some(ViewportDisplay {
                        diffuse_color: [0.8, 0.3, 0.2, alpha],
                        roughness: 0.25,
                        metallic: 0.5,
                    }),
                    ..Material::default()
                },
            );
            for (mode, expected) in [(Mode::Solid, transparent), (Mode::Beauty, false)] {
                let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
                let geometry = extract_geometry(&doc, &snapshot, mode, false, &BTreeMap::new())?;
                assert!(!geometry.triangles.is_empty(), "the plane has triangles");
                for triangle in &geometry.triangles {
                    assert_eq!(
                        matches!(triangle.alpha_mode, AlphaMode::Blend),
                        expected,
                        "alpha {alpha} in {mode:?} must blend only when Solid's alpha is below one"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn renderer_resolves_shared_data_slots_and_object_overrides() -> crate::error::Result<()> {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let mut doc = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"node.create","id":"body","kind":"plane","params":{"size":2.0}},
                    {"op":"node.duplicate","target":{"id":"body"},"id":"linked","mode":"linked"}
                ]
            }),
        )?
        .doc;
        let body_id = Id::new("body")?;
        let linked_id = Id::new("linked")?;
        let data_id = doc.nodes[&body_id]
            .data
            .clone()
            .expect("plane has mesh data");
        assert_eq!(doc.nodes[&linked_id].data.as_ref(), Some(&data_id));

        let blue_id = Id::new("blue")?;
        let gold_id = Id::new("gold")?;
        let red_id = Id::new("red")?;
        doc.materials.insert(
            blue_id.clone(),
            Material {
                base_color: [0.0, 0.0, 1.0, 1.0],
                ..Material::default()
            },
        );
        doc.materials.insert(
            gold_id.clone(),
            Material {
                base_color: [1.0, 0.7, 0.0, 1.0],
                ..Material::default()
            },
        );
        doc.materials.insert(
            red_id.clone(),
            Material {
                base_color: [1.0, 0.0, 0.0, 1.0],
                ..Material::default()
            },
        );
        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-3.0, 0.0, 0.0),
                DVec3::new(-2.0, 0.0, 0.0),
                DVec3::new(-3.0, 1.0, 0.0),
                DVec3::new(-1.0, 0.0, 0.0),
                DVec3::ZERO,
                DVec3::new(-1.0, 1.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 0.0),
                DVec3::new(3.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9, 10, 11]],
        )
        .expect("slot regression mesh is valid");
        for (slot, face) in mesh.faces.iter_mut().enumerate() {
            face.material_index = u32::try_from(slot).expect("four slots fit in u32");
        }
        let data = doc
            .data_blocks
            .get_mut(&data_id)
            .expect("plane data block exists");
        data.mesh = Some(mesh);
        data.materials = vec![Some(blue_id), None, Some(gold_id.clone()), Some(gold_id)];

        let body = doc.nodes.get_mut(&body_id).expect("body exists");
        body.materials = vec![None; 4];
        body.material_links = vec![MaterialLink::Data; 4];
        // Blender's OBJECT lookup falls back to DATA when its raw override slot is null.
        let linked = doc.nodes.get_mut(&linked_id).expect("linked node exists");
        linked.materials = vec![None, None, Some(red_id), None];
        linked.material_links = vec![
            MaterialLink::Data,
            MaterialLink::Data,
            MaterialLink::Object,
            MaterialLink::Object,
        ];

        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let geometry = extract_geometry(&doc, &snapshot, Mode::Solid, false, &BTreeMap::new())?;
        let colors_for = |node_id: &Id| {
            let object = geometry
                .objects
                .iter()
                .find(|object| &object.node_id == node_id)
                .expect("rendered object exists");
            geometry
                .triangles
                .iter()
                .filter(|triangle| triangle.object_index == object.index)
                .map(|triangle| geometry.materials[triangle.material_index].base_color)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            colors_for(&body_id),
            vec![
                [0.0, 0.0, 1.0, 1.0],
                // Solid's no-material face takes Blender's empty default surface, not Potter's Beauty default.
                [0.8, 0.8, 0.8, 1.0],
                [1.0, 0.7, 0.0, 1.0],
                [1.0, 0.7, 0.0, 1.0],
            ]
        );
        assert_eq!(
            colors_for(&linked_id),
            vec![
                [0.0, 0.0, 1.0, 1.0],
                // Solid's no-material face takes Blender's empty default surface, not Potter's Beauty default.
                [0.8, 0.8, 0.8, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 0.7, 0.0, 1.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn horizon_crossing_area_keeps_the_visible_diffuse_lobe() {
        let axis_v = DVec3::new(
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let light_direction = DVec3::X.cross(axis_v);
        let shape = AreaShape {
            axis_u: DVec3::X,
            axis_v,
            half_u: 1.0,
            half_v: 2.0,
            ellipse: false,
            area: 8.0,
        };
        let point = DVec3::ZERO;
        let normal = DVec3::Z;
        let centre = DVec3::new(0.0, 0.0, -1.0);
        let irradiance =
            area_irradiance_per_radiance(&shape, centre, light_direction, point, normal)
                * area_radiance(1.0, shape.area);
        assert!(irradiance > 0.0);
        let (samples, count) =
            area_visibility_samples(&shape, centre, light_direction, point, normal);
        assert_eq!(
            count, 1,
            "the four below-horizon samples must not contribute"
        );
        assert!(
            samples[..count]
                .iter()
                .all(|sample| sample.weight > 0.0 && sample.direction.z > 0.0)
        );

        let params = BsdfParams::from_material(
            &Material {
                base_color: [0.8, 0.8, 0.8, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                ..Material::default()
            },
            normal,
        );
        let centre_direction = (centre - point).normalize_or_zero();
        assert!(centre_direction.dot(normal) < 0.0);
        assert_eq!(
            specular_coefficient(
                &params,
                &principled_layers(&params, normal, normal),
                centre_direction
            ),
            DVec3::ZERO,
            "the specular lobe towards a below-horizon centre is zero"
        );
        let output = diffuse_coefficient(
            &params,
            &principled_layers(&params, normal, normal),
            DVec3::Z,
        ) * irradiance;
        assert!(
            output.x > 0.0,
            "visible upper strip must retain diffuse light: {output:?}"
        );
    }
    #[test]
    fn view_layer_hidden_surfaces_do_not_block_realtime_shadow_rays() -> crate::error::Result<()> {
        let mut triangle = source_triangle(
            [
                DVec3::new(-1.0, -1.0, 0.0),
                DVec3::new(1.0, -1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            [DVec3::Z; 3],
        );
        triangle.object_index = 1;
        triangle.ray_mask = RayKind::Shadow.bit();
        let triangles = [triangle];
        let bvh = Bvh::build(&triangles)?;
        let point_maps = BTreeMap::new();
        let mut object = ObjectRecord {
            index: 1,
            node_id: Id::new("object").expect("valid object ID"),
            data_id: None,
            instance_path: Vec::new(),
            uv_layers: Vec::new(),
            uv_render: None,
            color_layers: Vec::new(),
            material_colors: MaterialColors::default(),
            color_layer_indices: Vec::new(),
            geometry_attributes: Vec::new(),
            thickness: 0.0,
            object_to_world: DMat4::IDENTITY,
            world_to_local: Some(DMat4::IDENTITY),
            object_color: [1.0; 4],
            object_pass_index: 0.0,
            object_random_number: None,
            is_holdout: false,
            shadow_terminator_normal_offset: 0.0,
            shadow_terminator_geometry_offset: 0.1,
            shadow_terminator_shading_offset: 0.0,
            view_layer_use_solid: false,
            view_layer_use_volumes: true,
        };
        let hidden = ShadowScene {
            bvh: &bvh,
            triangles: &triangles,
            objects: std::slice::from_ref(&object),
            prepared_materials: &[],
            has_volume_only: false,
            point_maps: &point_maps,
            occluders: None,
        };
        let origin = DVec3::Z;
        let direction = -DVec3::Z;
        assert_eq!(ray_transmittance(hidden, origin, direction, 2.0, true), 1.0);

        object.view_layer_use_solid = true;
        let visible = ShadowScene {
            bvh: &bvh,
            triangles: &triangles,
            objects: std::slice::from_ref(&object),
            prepared_materials: &[],
            has_volume_only: false,
            point_maps: &point_maps,
            occluders: None,
        };
        assert_eq!(
            ray_transmittance(visible, origin, direction, 2.0, true),
            0.0
        );
        Ok(())
    }
    fn eevee_sample(doc: &SceneDoc) -> crate::error::Result<SurfaceSample> {
        let snapshot = Snapshot::evaluate(doc, &EvaluationContext::default())?;
        let images = BTreeMap::new();
        let geometry = extract_geometry_with_semantics(
            doc,
            &snapshot,
            Mode::Beauty,
            false,
            &images,
            crate::node_kernels::MathSemantics::Eevee,
        )?;
        let view = ViewBasis::from_camera(&Camera {
            position: DVec3::new(0.0, 0.0, 2.5),
            target: DVec3::ZERO,
            up: DVec3::Y,
            near: 0.1,
            far: 100.0,
            projection: Projection::Orthographic { height: 2.0 },
            shift: [0.0; 2],
            depth_of_field: None,
            stereo_mode: StereoMode::None,
            interocular_distance: 0.065,
        })?;
        let shader = scene_shader_with_semantics(
            doc,
            &snapshot,
            &images,
            &geometry,
            Mode::Beauty,
            &view,
            crate::node_kernels::MathSemantics::Eevee,
        );
        let triangle = geometry.triangles.first().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "EEVEE isolation plane has no triangle",
            )
        })?;
        let point = triangle.positions.iter().copied().sum::<DVec3>() / 3.0;
        shader.shade(
            triangle,
            &SurfaceHit {
                point,
                shadow_point: point,
                normal: DVec3::Z,
                geometric: DVec3::Z,
                geometric_normal: DVec3::Z,
                back_facing: false,
                realtime_differentials: crate::shader::HitDifferentials {
                    world_dx: DVec3::ZERO,
                    world_dy: DVec3::ZERO,
                    uv_dx: [0.0; 2],
                    uv_dy: [0.0; 2],
                },
                camera_matrices: None,
                barycentric: [1.0 / 3.0; 3],
                gbuffer_frag_coord: Some([0.5, 0.5]),
                gbuffer_sample_offset: Some([0.5; 3]),
                gbuffer_closure_random: Some(0.5),
                view: DVec3::Z,
                eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
            },
        )
    }

    #[test]
    fn eevee_object_holdout_and_cycles_only_settings_are_isolated() -> crate::error::Result<()> {
        let initial = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let applied = crate::ops::apply_batch(
            &initial,
            &json!({
                "schema_version": 1,
                "base_revision": 0,
                "operations": [
                    {"op":"node.create","id":"plane","kind":"plane","params":{"size":1.0}}
                ]
            }),
        )?;
        let mut doc = applied.doc;
        let material_id = Id::new("eevee_emission")?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                emission_color: [0.5, 0.25, 0.125],
                emission_strength: 1.0,
                ..Material::default()
            },
        );
        let node = doc.nodes.get_mut(&Id::new("plane")?).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "EEVEE isolation plane is missing")
        })?;
        node.materials = vec![Some(material_id)];
        node.material_links = vec![MaterialLink::Object];
        let baseline = eevee_sample(&doc)?;
        assert!(baseline.color[..3].iter().any(|channel| *channel > 0.0));

        let mut changed = doc.clone();
        let scene = changed.scenes.values_mut().next().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "EEVEE isolation scene is missing")
        })?;
        scene.render.caustics_reflective = false;
        scene.render.caustics_refractive = false;
        scene.render.film_exposure = 8.0;
        scene.render.film_transparent_glass = true;
        scene.render.film_transparent_roughness = 0.0;
        let node_id = Id::new("plane")?;
        changed
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "EEVEE isolation plane is missing")
            })?
            .shadow_terminator_shading_offset = 2.0;
        let changed_sample = eevee_sample(&changed)?;
        assert_eq!(changed_sample.color, baseline.color);
        assert_eq!(changed_sample.holdout_weight, baseline.holdout_weight);

        let mut held_out = doc;
        held_out
            .nodes
            .get_mut(&node_id)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "EEVEE holdout plane is missing")
            })?
            .is_holdout = true;
        let holdout_sample = eevee_sample(&held_out)?;
        assert!(holdout_sample.has_surface);
        assert_eq!(holdout_sample.holdout_weight, 1.0);
        assert_eq!(holdout_sample.color, [0.0; 4]);
        Ok(())
    }
}

#[cfg(test)]
mod transparent_shadow_tests {
    use crate::model::{Material, Registry};
    use crate::shader::PreparedMaterial;

    use super::casts_transparent_shadow;

    fn casts(material: &Material) -> bool {
        casts_transparent_shadow(material, &PreparedMaterial::new(material, &Registry::new()))
    }

    #[test]
    fn an_opaque_material_casts_no_transparent_shadow() {
        assert!(!casts(&Material::default()));
    }

    #[test]
    fn a_non_opaque_material_with_opaque_base_alpha_casts_no_transparent_shadow() {
        // Its export is a Principled with Alpha 1, which EEVEE compiles as opaque.
        let blend = Material {
            alpha_mode: "blend".to_owned(),
            base_color: [0.6, 0.6, 0.6, 1.0],
            ..Material::default()
        };
        assert!(!casts(&blend));
    }

    #[test]
    fn a_non_opaque_material_casts_only_when_shadows_are_on() {
        let blend = Material {
            alpha_mode: "blend".to_owned(),
            base_color: [0.6, 0.6, 0.6, 0.5],
            ..Material::default()
        };
        assert!(casts(&blend));
        let blend_off = Material {
            use_transparent_shadow: false,
            ..blend
        };
        assert!(!casts(&blend_off));
    }
}

#[cfg(test)]
mod beauty_tint_tests {
    use super::{beauty_tint, weighted_tint};
    use crate::model::Material;

    fn material(alpha_mode: &str) -> Material {
        Material {
            alpha_mode: alpha_mode.to_owned(),
            ..Material::default()
        }
    }

    #[test]
    fn blended_material_keeps_each_channel() {
        assert_eq!(
            beauty_tint(Some(&material("blend")), [0.2, 0.8, 0.4], false),
            [0.2, 0.8, 0.4]
        );
    }

    /// A DITHERED material takes the mean of the Color's channels: (0.2 + 0.8 + 0.4) / 3 on every channel.
    #[test]
    fn dithered_material_takes_the_channel_mean() {
        let mean = (0.2 + 0.8 + 0.4) / 3.0;
        for alpha_mode in ["opaque", "clip"] {
            for channel in beauty_tint(Some(&material(alpha_mode)), [0.2, 0.8, 0.4], false) {
                assert!((channel - mean).abs() < 1.0e-12, "{alpha_mode}: {channel}");
            }
        }
    }

    /// Tints are clamped to [0, 1] first. BLENDED clamps each channel: 2 -> 1, -1 -> 0, 0.5 stays. DITHERED takes the
    /// mean of the clamped channels: (1 + 0 + 0.5) / 3 = 0.5.
    #[test]
    fn tints_are_clamped_before_use() {
        assert_eq!(
            beauty_tint(Some(&material("blend")), [2.0, -1.0, 0.5], false),
            [1.0, 0.0, 0.5]
        );
        let dithered = beauty_tint(Some(&material("clip")), [2.0, -1.0, 0.5], false);
        assert!((dithered[0] - 0.5).abs() < 1.0e-12, "{dithered:?}");
    }

    #[test]
    fn no_material_keeps_the_tint_clamped() {
        assert_eq!(beauty_tint(None, [0.5, 1.5, -0.2], false), [0.5, 1.0, 0.0]);
    }

    /// Mix(0.5) of a Transparent (colour (0.2, 0.8, 0.4)) and an opaque Principled. The transparent closure passes
    /// 0.5 * tint, the opaque one passes nothing, so the surface's tint is the Transparent colour and the composite passes
    /// `(1 - 0.5) * tint`: the transmittance EEVEE sums (`g_transmittance`) is `0.5 * tint`.
    #[test]
    fn mix_of_transparent_and_opaque_keeps_the_transparent_colour() {
        let tint = [0.2, 0.8, 0.4];
        let terms = [(0.5, 1.0, [1.0; 3]), (0.5, 0.0, tint)];
        let opacity = 0.5;
        let actual = weighted_tint(terms.into_iter(), opacity);
        for (channel, expected) in tint.iter().enumerate() {
            assert!((actual[channel] - expected).abs() < 1.0e-12, "{actual:?}");
        }
    }

    /// Two transparent closures, weights 0.5 and 0.5, alphas 0 and 0.5, tints (0.2, 0.8, 0.4) and white. The passed
    /// transmittance is 0.5 * (0.2, 0.8, 0.4) + 0.5 * 0.5 * (1, 1, 1) = (0.35, 0.65, 0.45), the opacity is
    /// 0.5 * 0 + 0.5 * 0.5 = 0.25, so the tint is (0.35, 0.65, 0.45) / 0.75.
    #[test]
    fn weighted_sum_over_two_transparent_closures() {
        let terms = [(0.5, 0.0, [0.2, 0.8, 0.4]), (0.5, 0.5, [1.0; 3])];
        let actual = weighted_tint(terms.into_iter(), 0.25);
        let expected = [0.35 / 0.75, 0.65 / 0.75, 0.45 / 0.75];
        for (channel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "channel {channel}: {actual}"
            );
        }
    }

    /// A surface that passes nothing (opacity 1) has no pass chance, so its tint is white.
    #[test]
    fn fully_opaque_surface_has_a_white_tint() {
        let actual = weighted_tint([(1.0, 1.0, [0.2, 0.8, 0.4])].into_iter(), 1.0);
        assert_eq!(actual, [1.0; 3]);
    }
}

#[cfg(test)]
mod rgb9e5_tests {
    use super::rgb9e5_roundtrip;
    use glam::DVec3;

    #[test]
    fn packs_the_sum_of_direct_lights_not_each_light_separately() {
        let first = DVec3::new(0.000_901, 0.000_40, 0.000_20);
        let second = DVec3::new(0.000_902, 0.000_41, 0.000_21);
        let summed = rgb9e5_roundtrip(first + second);
        let separately_packed = rgb9e5_roundtrip(first) + rgb9e5_roundtrip(second);

        assert_eq!(summed.x, 946.0 / 524_288.0);
        assert_ne!(summed, separately_packed);
    }
}

#[cfg(test)]
mod cycles_hair_matrix_tests {
    use glam::DVec3;

    use super::cycles_hair_object_matrix;

    #[test]
    fn row_major_cycles_matrix_preserves_blender_translation() {
        let matrix = cycles_hair_object_matrix([
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        assert_eq!(
            matrix.transform_point3(DVec3::ZERO),
            DVec3::new(3.0, 4.0, 5.0)
        );
        assert_eq!(matrix.transform_vector3(DVec3::X), DVec3::X);
    }
}
#[cfg(test)]
mod eevee_gbuffer_candidate_tests {
    use glam::{DMat4, DVec3, Vec3};

    use crate::{
        graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
        model::{Id, Material, Registry},
        render::raster::{EeveeRaytraceFeedback, SurfaceHit},
        shader::{
            BsdfModel, BsdfParams, EeveeGBufferCandidate, EeveeGBufferFeatures, EeveeGBufferLobe,
            HitContext, HitDifferentials, ShaderClosure, evaluate_surface,
        },
    };
    use serde_json::json;

    use super::super::eevee_raytrace_generate::generate_ray;

    use super::{
        EeveeGBufferMode, RayGenerateInput, RayLobe, RayThickness, ThicknessMode, beauty_tint,
        decode_eevee_gbuffer_bin, eevee_gbuffer_alpha, eevee_gbuffer_candidates,
        eevee_gbuffer_route, eevee_raytrace_sample, opaque_share,
    };

    #[test]
    fn gbuffer_alpha_reciprocal_uses_raw_transmittance_and_excludes_holdout() {
        let mut portal = BsdfParams::from_material(&Material::default(), DVec3::Z);
        portal.alpha = 0.0;
        portal.transparent_tint = [0.8, 0.4, 0.2];
        let mut holdout = BsdfParams::from_material(&Material::default(), DVec3::Z);
        holdout.model = BsdfModel::Holdout;
        holdout.alpha = 0.0;
        let closures = [
            ShaderClosure {
                weight: 0.5,
                params: portal,
            },
            ShaderClosure {
                weight: 0.5,
                params: holdout,
            },
        ];
        let (transparency, alpha, alpha_reciprocal) = eevee_gbuffer_alpha(&closures);
        let expected_transparency = (0.4_f32 + 0.2 + 0.1) / 3.0;
        assert_eq!(transparency.to_bits(), expected_transparency.to_bits());
        let expected = 1.0_f32 - expected_transparency;
        assert_eq!(alpha.to_bits(), expected.to_bits());
        assert_eq!(alpha_reciprocal.to_bits(), expected.recip().to_bits());
    }

    #[test]
    fn principled_reflection_candidate_keeps_the_source_closure() -> Result<(), String> {
        let closure = ShaderClosure {
            weight: 0.5,
            params: BsdfParams::from_material(&Material::default(), DVec3::Z),
        };
        let candidates = eevee_gbuffer_candidates(&[closure], DVec3::Z, DVec3::Z)
            .map_err(|error| format!("{error:?}"))?;
        let reflection = candidates
            .first()
            .ok_or_else(|| "Principled emits base reflection".to_owned())?;
        assert_eq!(reflection.source_closure_index, 0);
        assert_eq!(reflection.lobe, EeveeGBufferLobe::Reflection);
        assert_eq!(reflection.weight, 1.0);
        Ok(())
    }

    #[test]
    fn principled_distribution_controls_reflection_diffuse_and_coat_energy() -> Result<(), String> {
        let evaluate = |distribution: Option<&str>, coat_weight: f64| {
            let graph_id =
                Id::new("distribution_graph".to_owned()).map_err(|error| format!("{error:?}"))?;
            let principled_id =
                Id::new("principled".to_owned()).map_err(|error| format!("{error:?}"))?;
            let output_id = Id::new("output".to_owned()).map_err(|error| format!("{error:?}"))?;
            let mut principled = GraphNode::new("ShaderNodeBsdfPrincipled");
            if let Some(distribution) = distribution {
                principled
                    .properties
                    .insert("distribution".to_owned(), json!(distribution));
            }
            for (socket, value) in [
                ("Base Color", json!([0.12, 0.04, 0.8, 1.0])),
                ("Metallic", json!(0.0)),
                ("Roughness", json!(0.75)),
                ("IOR", json!(1.5)),
                ("Specular IOR Level", json!(0.5)),
                ("Specular Tint", json!([1.0, 1.0, 1.0, 1.0])),
                ("Transmission Weight", json!(0.0)),
                ("Subsurface Weight", json!(0.0)),
                ("Coat Weight", json!(coat_weight)),
                ("Sheen Weight", json!(0.0)),
            ] {
                principled.inputs.insert(socket.to_owned(), value);
            }

            let mut graph = NodeGroup::new("Distribution", GraphKind::Shader);
            graph.nodes.insert(principled_id.clone(), principled);
            graph.nodes.insert(
                output_id.clone(),
                GraphNode::new("ShaderNodeOutputMaterial"),
            );
            graph.links.push(GraphLink {
                from_node: principled_id,
                from_socket: "BSDF".to_owned(),
                to_node: output_id,
                to_socket: "Surface".to_owned(),
            });
            let mut groups = Registry::new();
            groups.insert(graph_id.clone(), graph);
            let material = Material {
                node_tree: Some(graph_id),
                ..Material::default()
            };
            let hit_context = HitContext::new(&groups);
            let shading =
                evaluate_surface(&material, &hit_context).map_err(|error| format!("{error:?}"))?;
            eevee_gbuffer_candidates(&shading.closures, DVec3::Z, DVec3::Z)
                .map_err(|error| format!("{error:?}"))
        };
        let candidate = |candidates: &[EeveeGBufferCandidate], lobe: EeveeGBufferLobe| {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.lobe == lobe)
                .ok_or_else(|| format!("missing {lobe:?} candidate"))?;
            Ok::<_, String>((
                DVec3::from_array(candidate.color.map(f64::from)),
                candidate.weight,
            ))
        };

        let multi = evaluate(None, 0.0)?;
        let ggx = evaluate(Some("GGX"), 0.0)?;
        let (multi_reflection, _) = candidate(&multi, EeveeGBufferLobe::Reflection)?;
        let (ggx_reflection, _) = candidate(&ggx, EeveeGBufferLobe::Reflection)?;
        let (multi_diffuse, _) = candidate(&multi, EeveeGBufferLobe::Diffuse)?;
        let (ggx_diffuse, _) = candidate(&ggx, EeveeGBufferLobe::Diffuse)?;
        assert!(multi_reflection.max_element() > ggx_reflection.max_element());
        assert!(multi_diffuse.max_element() < ggx_diffuse.max_element());

        let base_color = DVec3::new(0.12, 0.04, 0.8);
        for (reflection, diffuse) in [
            (multi_reflection, multi_diffuse),
            (ggx_reflection, ggx_diffuse),
        ] {
            let expected = base_color * (1.0 - reflection.max_element()).max(0.0);
            assert!((diffuse - expected).max_element().abs() < 1.0e-6);
        }

        let multi_coat = evaluate(None, 0.4)?;
        let ggx_coat = evaluate(Some("GGX"), 0.4)?;
        let (_, multi_coat_weight) = candidate(&multi_coat, EeveeGBufferLobe::Clearcoat)?;
        let (_, ggx_coat_weight) = candidate(&ggx_coat, EeveeGBufferLobe::Clearcoat)?;
        assert_eq!(multi_coat_weight, ggx_coat_weight);
        Ok(())
    }

    #[test]
    fn blended_eevee_transparency_uses_the_scene_color_target() {
        let material = Material {
            alpha_mode: "blend".to_owned(),
            ..Material::default()
        };
        let tint = [0.92, 0.87, 0.73];
        assert_eq!(beauty_tint(Some(&material), tint, true), [0.92; 3]);
        assert_eq!(beauty_tint(Some(&material), tint, false), tint);

        let dithered = Material::default();
        let mean = (tint[0] + tint[1] + tint[2]) / 3.0;
        assert_eq!(beauty_tint(Some(&dithered), tint, true), [mean; 3]);
    }

    #[test]
    fn principled_hair_gpu_weight_changes_radiance_without_changing_mix_coverage()
    -> Result<(), String> {
        let hair = |weight| {
            let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
            params.model = BsdfModel::Principled;
            params.base_color = [0.34, 0.12, 0.07, 1.0];
            params.eevee_gbuffer_features = EeveeGBufferFeatures::diffuse_only();
            params.eevee_diffuse_weight = Some(weight);
            ShaderClosure {
                weight: 0.5,
                params,
            }
        };
        let mut black_params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        black_params.model = BsdfModel::Principled;
        black_params.base_color = [0.0; 4];
        black_params.eevee_gbuffer_features = EeveeGBufferFeatures::diffuse_only();
        let black = ShaderClosure {
            weight: 0.5,
            params: black_params,
        };

        for (gpu_weight, expected) in [(0.62, 0.62_f32), (0.31, 0.31_f32)] {
            let closures = [hair(gpu_weight), black];
            assert_eq!(opaque_share(&closures), 1.0);
            let candidates = eevee_gbuffer_candidates(&closures, DVec3::Z, DVec3::Z)
                .map_err(|error| format!("{error:?}"))?;
            let hair_candidate = candidates
                .iter()
                .find(|candidate| candidate.source_closure_index == 0)
                .ok_or_else(|| "Hair diffuse G-buffer candidate is missing".to_owned())?;
            assert_eq!(hair_candidate.lobe, EeveeGBufferLobe::Diffuse);
            assert_eq!(hair_candidate.weight, expected);
        }
        Ok(())
    }

    #[test]
    fn rotated_camera_raytrace_producer_keeps_consumer_ray_in_source_world_frame()
    -> Result<(), String> {
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::X);
        params.model = BsdfModel::Principled;
        params.normal = DVec3::X;
        params.eevee_gbuffer_features = EeveeGBufferFeatures::diffuse_only();
        let closures = [ShaderClosure {
            weight: 1.0,
            params,
        }];
        let world_to_view = DMat4::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let camera_matrices = (world_to_view.inverse(), world_to_view);
        let hit = SurfaceHit {
            point: DVec3::new(0.0, 0.0, -2.0),
            shadow_point: DVec3::new(0.0, 0.0, -2.0),
            normal: DVec3::X,
            geometric: DVec3::X,
            geometric_normal: DVec3::X,
            back_facing: false,
            realtime_differentials: HitDifferentials {
                world_dx: DVec3::ZERO,
                world_dy: DVec3::ZERO,
                uv_dx: [0.0; 2],
                uv_dy: [0.0; 2],
            },
            camera_matrices: Some(&camera_matrices),
            barycentric: [1.0 / 3.0; 3],
            gbuffer_frag_coord: Some([0.5, 0.5]),
            gbuffer_sample_offset: Some([0.0; 3]),
            gbuffer_closure_random: Some(0.5),
            view: DVec3::X,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
        };
        let route = eevee_gbuffer_route(None, &closures, &hit, 0.825)
            .map_err(|error| format!("{error:?}"))?;
        let route_bin = route
            .layout
            .bins
            .iter()
            .flatten()
            .copied()
            .next()
            .ok_or_else(|| "diffuse G-buffer route has no selected bin".to_owned())?;
        assert_eq!(route_bin.mode, EeveeGBufferMode::Diffuse);
        assert_eq!(route.thickness, super::Thickness::None);
        let decoded = decode_eevee_gbuffer_bin(&route, &closures, route_bin)
            .map_err(|error| format!("{error:?}"))?;
        assert_ne!(
            world_to_view.transform_vector3(decoded.normal),
            decoded.normal
        );
        let sample = eevee_raytrace_sample(Some(&route), &closures, &hit, Some(false))
            .map_err(|error| format!("{error:?}"))?;
        let consumer_bin = sample.bins[route_bin.storage_layer]
            .ok_or_else(|| "raytrace producer left the selected bin empty".to_owned())?;

        let params = &decoded.closure.params;
        let source_input = RayGenerateInput {
            lobe: RayLobe::Diffuse,
            normal: Vec3::new(
                decoded.normal.x as f32,
                decoded.normal.y as f32,
                decoded.normal.z as f32,
            ),
            view: Vec3::new(hit.view.x as f32, hit.view.y as f32, hit.view.z as f32),
            roughness: decoded.roughness.unwrap_or(1.0) as f32,
            ior: decoded.ior.unwrap_or(1.0) as f32,
            thickness: RayThickness {
                mode: if params.thickness_slab {
                    ThicknessMode::Slab
                } else {
                    ThicknessMode::Sphere
                },
                value: params.thickness.unwrap_or(0.0).max(0.0) as f32,
            },
        };
        let noise = [0.23, 0.37];
        let consumer_ray =
            generate_ray(consumer_bin.world_input, noise).map_err(|error| format!("{error:?}"))?;
        let source_world_ray =
            generate_ray(source_input, noise).map_err(|error| format!("{error:?}"))?;
        assert_eq!(consumer_ray.direction, source_world_ray.direction);
        Ok(())
    }

    #[test]
    fn selected_transmission_ray_uses_quantized_source_thickness() -> Result<(), String> {
        let material = Material {
            base_color: [0.8, 0.3, 0.2, 1.0],
            transmission: 1.0,
            ..Material::default()
        };
        let mut params = BsdfParams::from_material(&material, DVec3::Z);
        params.thickness = Some(1.0e-6);
        let closures = [ShaderClosure {
            weight: 1.0,
            params,
        }];
        let hit = SurfaceHit {
            point: DVec3::new(0.0, 0.0, -2.0),
            shadow_point: DVec3::new(0.0, 0.0, -2.0),
            normal: DVec3::Z,
            geometric: DVec3::Z,
            geometric_normal: DVec3::Z,
            back_facing: false,
            realtime_differentials: HitDifferentials {
                world_dx: DVec3::ZERO,
                world_dy: DVec3::ZERO,
                uv_dx: [0.0; 2],
                uv_dy: [0.0; 2],
            },
            camera_matrices: None,
            barycentric: [1.0 / 3.0; 3],
            gbuffer_frag_coord: Some([0.5, 0.5]),
            gbuffer_sample_offset: Some([0.0; 3]),
            gbuffer_closure_random: Some(0.5),
            view: DVec3::Z,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
        };
        let route = eevee_gbuffer_route(None, &closures, &hit, 0.0)
            .map_err(|error| format!("{error:?}"))?;
        let bin = route
            .layout
            .bins
            .iter()
            .take(route.layout.bin_count)
            .flatten()
            .find(|bin| bin.lobe == EeveeGBufferLobe::Refraction)
            .copied()
            .ok_or_else(|| "refraction G-buffer route has no selected bin".to_owned())?;
        assert_eq!(route.thickness, super::Thickness::None);

        let sample = eevee_raytrace_sample(Some(&route), &closures, &hit, Some(false))
            .map_err(|error| format!("{error:?}"))?;
        let consumer_bin = sample.bins[bin.storage_layer]
            .ok_or_else(|| "raytrace producer left the refraction bin empty".to_owned())?;
        assert_eq!(consumer_bin.world_input.thickness.value, 0.0);
        let noise = [0.23, 0.37];
        let collapsed_ray =
            generate_ray(consumer_bin.world_input, noise).map_err(|error| format!("{error:?}"))?;
        assert_eq!(collapsed_ray.origin_offset, Vec3::ZERO);

        let mut raw_input = consumer_bin.world_input;
        raw_input.thickness = RayThickness {
            mode: ThicknessMode::Sphere,
            value: 1.0e-6,
        };
        let raw_ray = generate_ray(raw_input, noise).map_err(|error| format!("{error:?}"))?;
        assert_ne!(raw_ray.origin_offset, Vec3::ZERO);
        Ok(())
    }
}
#[cfg(test)]
mod eevee_world_sun_tests {
    use glam::{DVec3, Vec3};

    use crate::{
        eval::{EvaluationContext, Snapshot},
        model::{Material, SceneDoc, World},
        render::raster::Mode,
        shader::{BsdfModel, BsdfParams, ShaderClosure},
    };

    use super::{
        EeveeRaytraceFeedback, EeveeWorldSun, Receiver, SurfaceShadeContext, eevee_world_sun_light,
        shade,
    };

    #[test]
    fn extracted_world_sun_uses_native_surface_lighting_and_separate_shadow_angle()
    -> Result<(), String> {
        let world = World {
            eevee_sun_angle: Some(0.4),
            eevee_use_sun_shadow: Some(true),
            eevee_use_sun_shadow_jitter: Some(true),
            eevee_sun_shadow_jitter_overblur: Some(50.0),
            ..World::default()
        };
        let sun = eevee_world_sun_light(
            &world,
            EeveeWorldSun {
                radiance: Vec3::splat(2.0),
                direction: Vec3::Z,
            },
        )
        .map_err(str::to_owned)?;
        assert!((sun.light.sun_angle - 0.4).abs() < f64::EPSILON);
        let sun_radius = (0.4_f64 * 0.5).tan().clamp(0.001, 20.0);
        let expected_surface_energy =
            std::f64::consts::FRAC_1_PI * (1.0 + 1.0 / sun_radius.powi(2));
        assert_eq!(sun.light.energy, expected_surface_energy);
        let shadow_angle = 0.4_f64 * 0.5 * 0.5;
        let shadow_radius = sun
            .shadow_radius
            .ok_or_else(|| "enabled source Sun shadow radius is missing".to_owned())?;
        assert!((shadow_radius - shadow_angle.tan()).abs() < 1.0e-15);
        assert!(sun.ray_visibility.allows(crate::model::RayKind::Diffuse));
        assert!(sun.ray_visibility.allows(crate::model::RayKind::Glossy));
        assert!(!sun.ray_visibility.allows(crate::model::RayKind::Camera));
        assert!(
            sun.ray_visibility
                .allows(crate::model::RayKind::Transmission)
        );
        assert!(
            sun.ray_visibility
                .allows(crate::model::RayKind::VolumeScatter)
        );
        assert_eq!(sun.volume_energy, 1.0);

        let no_shadow_world = World {
            eevee_sun_angle: Some(0.0),
            eevee_use_sun_shadow: Some(false),
            ..World::default()
        };
        let sun = eevee_world_sun_light(
            &no_shadow_world,
            EeveeWorldSun {
                radiance: Vec3::splat(2.0),
                direction: Vec3::Z,
            },
        )
        .map_err(str::to_owned)?;
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())
            .map_err(|error| format!("{error:?}"))?;
        let mut params = BsdfParams::from_material(&Material::default(), DVec3::Z);
        params.model = BsdfModel::Principled;
        params.base_color = [1.0; 4];
        params.reflectance = 1.0;
        let closures = [ShaderClosure {
            weight: 1.0,
            params,
        }];

        let receiver = Receiver {
            point: DVec3::ZERO,
            shadow_point: DVec3::ZERO,
            geometric: DVec3::Z,
            view: DVec3::Z,
        };
        let lit = shade(&SurfaceShadeContext {
            doc: &doc,
            snapshot: &snapshot,
            receiver,
            normal: DVec3::Z,
            closures: &closures,
            mode: Mode::Beauty,
            shadow: None,
            object_thickness: 0.0,
            world: None,
            world_sun: Some(&sun),
            gbuffer: None,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
            material_refraction: Some(false),
        })
        .map_err(|error| format!("{error:?}"))?;
        let unlit = shade(&SurfaceShadeContext {
            doc: &doc,
            snapshot: &snapshot,
            receiver,
            normal: DVec3::Z,
            closures: &closures,
            mode: Mode::Beauty,
            shadow: None,
            object_thickness: 0.0,
            world: None,
            world_sun: None,
            gbuffer: None,
            eevee_raytrace_feedback: EeveeRaytraceFeedback::default(),
            material_refraction: Some(false),
        })
        .map_err(|error| format!("{error:?}"))?;
        assert!(lit[0] > unlit[0]);
        assert!(lit[1] > unlit[1]);
        assert!(lit[2] > unlit[2]);
        Ok(())
    }
    #[test]
    fn extracted_world_sun_matrix_matches_native_duff_basis() -> Result<(), String> {
        let world = World {
            eevee_sun_angle: Some(0.0),
            eevee_use_sun_shadow: Some(false),
            ..World::default()
        };
        let sun = eevee_world_sun_light(
            &world,
            EeveeWorldSun {
                radiance: Vec3::ONE,
                direction: Vec3::X,
            },
        )
        .map_err(str::to_owned)?;
        assert_eq!(sun.matrix.x_axis.truncate(), -DVec3::Z);
        assert_eq!(sun.matrix.y_axis.truncate(), DVec3::Y);
        assert_eq!(sun.matrix.z_axis.truncate(), DVec3::X);
        Ok(())
    }
}
