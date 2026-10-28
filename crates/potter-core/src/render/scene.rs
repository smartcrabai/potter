use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashSet},
    sync::{Arc, OnceLock},
};

use glam::{DMat3, DMat4, DVec3};
use serde::Serialize;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::{
        GeneratedCorners, GeneratedVertex, Mesh, TextureSpace,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        normals::{render_corner_normals, smooth_face_flags},
        uv_layers::{self, UvStore},
    },
    image::ImageData,
    model::{Id, Material, Node, RayKind, RayVisibility, SceneDoc},
    shader::{
        BsdfParams, EvaluatorWorkspace, HitContext, HitUvLayers, PreparedMaterial, ShaderClosure,
        SurfaceShading, dielectric_f0, evaluate_displacement_with_workspace,
        evaluate_surface_with_workspace, fill_color_layers, heaviest, make_orthonormals,
    },
};

use super::world_probe::WorldProbe;
use super::{
    bvh::{Bvh, BvhHit, WalkRay},
    eevee_ltc::Emitter,
    eevee_transmission::{
        self, GlassClosures, GlassLight, Thickness, irradiance_probe_radiance, probe_radiance,
    },
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
        AlphaMode, Line, Mode, SurfaceHit, SurfaceSample, SurfaceShader, SurfaceShaderWorker,
        Triangle, ViewBasis, ambient_visibility, blend_color, blend_corner_normals, blend_uv,
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
    material_pass_index: Option<f64>,
    render_layer: Option<usize>,
    available_mask: Option<Arc<[u64]>>,
}

impl MaterialColors {
    /// Records the entry of `material_index`; a later face of the same material replaces it.
    fn insert(
        &mut self,
        material_index: usize,
        material_pass_index: Option<f64>,
        render_layer: Option<usize>,
        available_mask: Option<Arc<[u64]>>,
    ) {
        let entry = MaterialColor {
            material_index,
            material_pass_index,
            render_layer,
            available_mask,
        };
        match self
            .entries
            .binary_search_by_key(&material_index, |stored| stored.material_index)
        {
            Ok(position) => self.entries[position] = entry,
            Err(position) => self.entries.insert(position, entry),
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

pub(super) struct Geometry {
    pub triangles: Vec<Triangle>,
    pub lines: Vec<Line>,
    /// Mesh edges for the wireframe preview, each once; empty outside Wire.
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
    if opaque_blocks && shadow.occluders.is_none() {
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

pub(super) fn extract_geometry(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    mode: Mode,
    path_traced: bool,
    images: &BTreeMap<String, ImageData>,
) -> Result<Geometry> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::TargetNotFound, "selected scene does not exist"))?;
    let view_layer = snapshot
        .view_layer
        .as_ref()
        .and_then(|id| scene.view_layers.get(id));
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
    let mut materials = vec![default_surface];
    let mut prepared = vec![PreparedMaterial::default()];
    let mut material_indices =
        std::collections::BTreeMap::<(Id, Option<usize>, Option<Arc<[u64]>>), usize>::new();
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
        if !visible_nodes.contains(node_id)
            || !snapshot.node_is_visible(node_id, node)
            || (matches!(mode, Mode::Beauty)
                && !crate::exchange::evaluated_visibility(snapshot, node_id, node).1)
        {
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
        let volume = node
            .data
            .as_ref()
            .and_then(|data_id| snapshot.volume_data.get(data_id))
            .or_else(|| data_block.and_then(|data| data.volume.as_ref()));
        let strokes = snapshot.strokes.get(node_id);
        let hair = data_block.and_then(|data| data.hair_curves.as_ref());
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
        let mesh = snapshot.meshes.get(node_id).or(raster_volume_mesh.as_ref());
        if mesh.is_none() && strokes.is_none() && hair.is_none() && volume.is_none() {
            continue;
        }
        let evaluated = snapshot.nodes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "render node is missing from snapshot",
            )
        })?;
        let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
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
            thickness,
            material_colors: MaterialColors::default(),
            object_to_world: matrix,
            world_to_local: object_inverse(matrix),
            object_color,
            object_pass_index,
            object_random_number,
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
            let face_normal_data = corner_normals(
                mesh,
                matrix,
                node_id,
                snapshot.adaptive_normals.get(node_id).map(Vec::as_slice),
                has_uv_map_tangent,
                &mut warnings,
            );
            let mut mikk_tangents = if has_uv_map_tangent && !uv_store.layers.is_empty() {
                Some(mesh_mikk_tangents(
                    mesh,
                    &mesh_triangles,
                    &vertex_positions,
                    &uv_store,
                    face_normal_data.local.as_deref(),
                    matrix,
                )?)
            } else {
                None
            }
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
            let vertex_slots: BTreeMap<u32, usize> = if carried_layers.is_empty() {
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
                let face_corner_normals = face_normals[face_index].as_deref();
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
                let material_id = snapshot.slot_material(node_id, node, data_block, binding_slot);
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
                    let mut positions = [DVec3::ZERO; 3];
                    let mut vertices = [GeneratedVertex::default(); 3];
                    let corners = triangle_corners(face, triangle);
                    let layer_uvs = triangle_layer_uvs(&uv_store, face_index, corners);
                    for (index, vertex_id) in triangle.iter().enumerate() {
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
                        face_corner_base,
                        corners,
                        *triangle,
                        &vertex_slots,
                    );
                    let layer_tangents = mikk_tangents
                        .as_mut()
                        .and_then(Iterator::next)
                        .unwrap_or_default();
                    let raw_normal =
                        (positions[1] - positions[0]).cross(positions[2] - positions[0]);
                    if raw_normal.length_squared() <= f64::EPSILON || !raw_normal.is_finite() {
                        continue;
                    }
                    let corner_normals =
                        face_corner_normals.and_then(|normals| corner_values(normals, corners));
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
                            shade(
                                doc, snapshot, center, center, normal, normal, normal, &closures,
                                mode, None, thickness, None,
                            )
                        };
                    let triangle = Triangle {
                        positions,
                        generated,
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
                                    // Potter keeps the DITHERED graph's expected coverage in the existing opacity/tint
                                    // model while retaining its separate depth-writing class for camera light shapes.
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
            let color = snapshot
                .slot_material(node_id, node, data_block, 0)
                .and_then(|material_id| doc.materials.get(material_id))
                .map_or([0.6, 0.6, 0.6, 1.0], |material| material.base_color);
            for (curve_index, curve) in hair.curves.iter().enumerate() {
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
                    continue;
                }
                let element_index = u32::try_from(elements.len() + 1).map_err(|_| {
                    PotError::new(
                        ErrorCode::LimitExceeded,
                        "too many elements for the ID buffer",
                    )
                })?;
                elements.push(ElementRecord {
                    index: element_index,
                    node_id: node_id.clone(),
                    domain: "hair_curve".to_owned(),
                    element_id: format!("h{curve_index}"),
                    uv_render: None,
                    uv_available_mask: None,
                });
                let radius = curve.radius * radius_scale;
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
                    let start = matrix.transform_point3(DVec3::from_array(curve.points[segment]));
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
                        element_index,
                        ray_mask: ray_visibility.mask(),
                    });
                }
            }
        }
    }
    let mut geometry = Geometry {
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
        let shadow_maps =
            build_point_shadow_maps(doc, snapshot, &geometry.triangles, &caster_materials)?;
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
    caster_materials: &[bool],
) -> Result<BTreeMap<Id, PointShadowMap>> {
    // A triangle that casts a transparent shadow is left out: its occlusion is the ray pass of `ray_transmittance`.
    let opaque: Vec<&Triangle> = triangles
        .iter()
        .filter(|triangle| {
            !caster_materials
                .get(triangle.material_index)
                .copied()
                .unwrap_or(false)
        })
        .collect();
    let mut maps = BTreeMap::new();
    for (id, node) in &doc.nodes {
        let Some(light) = node
            .data
            .as_ref()
            .and_then(|data_id| doc.data_blocks.get(data_id))
            .and_then(|block| block.light.as_ref())
        else {
            continue;
        };
        if !matches!(
            light.light_type,
            crate::model::LightType::Point | crate::model::LightType::Spot
        ) {
            continue;
        }
        let Some(evaluated) = snapshot.nodes.get(id) else {
            continue;
        };
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
        evaluate_displacement_with_workspace(material, prepared, &context, evaluator_workspace)
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

struct FaceNormalData {
    world: Vec<Option<Vec<DVec3>>>,
    local: Option<Vec<Option<Vec<DVec3>>>>,
}

/// Corner normals in object and world space; `None` marks a flat face. A mesh whose corner normals
/// cannot be built keeps flat face normals and records a warning.
fn corner_normals(
    mesh: &crate::geom::Mesh,
    matrix: DMat4,
    node_id: &Id,
    adaptive_normals: Option<&[Option<DVec3>]>,
    keep_local: bool,
    warnings: &mut Vec<serde_json::Value>,
) -> FaceNormalData {
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let mut local = match render_corner_normals(mesh, &positions) {
        Ok(faces) => faces,
        Err(error) => {
            warnings.push(serde_json::json!({
                "code": "shading.corner_normals_failed",
                "message": format!(
                    "smooth shading normals could not be computed ({error}); flat face normals were used"
                ),
                "node_id": node_id,
            }));
            let flat_faces = vec![None; mesh.faces.len()];
            return FaceNormalData {
                world: flat_faces.clone(),
                local: keep_local.then_some(flat_faces),
            };
        }
    };
    if let Some(normals) = adaptive_normals.filter(|normals| normals.len() == mesh.vertices.len())
        && let Ok(smooth_faces) = smooth_face_flags(mesh)
        && smooth_faces.iter().any(|smooth| *smooth)
    {
        let vertex_indices: BTreeMap<u32, usize> = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect();
        for (face_index, (face, smooth)) in mesh.faces.iter().zip(smooth_faces).enumerate() {
            if !smooth {
                continue;
            }
            let Some(face_normals) = local.get_mut(face_index).and_then(Option::as_mut) else {
                continue;
            };
            for (corner, vertex_id) in face.vertices.iter().enumerate() {
                if let Some(normal) = vertex_indices
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
    // Inverse-transpose keeps normals perpendicular to the surface under non-uniform scale.
    let normal_matrix = DMat3::from_mat4(matrix).inverse().transpose();
    let mut degenerate = false;
    let world = local
        .iter()
        .map(|face| {
            let world: Vec<DVec3> = face
                .as_ref()?
                .iter()
                .map(|normal| (normal_matrix * normal).normalize_or_zero())
                .collect();
            let valid = world
                .iter()
                .all(|normal| normal.is_finite() && *normal != DVec3::ZERO);
            degenerate |= !valid;
            valid.then_some(world)
        })
        .collect();
    if degenerate {
        warnings.push(serde_json::json!({
            "code": "shading.corner_normals_failed",
            "message": "smooth shading normals became zero-length under the object transform; affected faces were shaded flat",
            "node_id": node_id,
        }));
    }
    FaceNormalData {
        world,
        local: keep_local.then_some(local),
    }
}

#[cfg(test)]
fn world_face_normals(
    mesh: &crate::geom::Mesh,
    matrix: DMat4,
    node_id: &Id,
    adaptive_normals: Option<&[Option<DVec3>]>,
    warnings: &mut Vec<serde_json::Value>,
) -> Vec<Option<Vec<DVec3>>> {
    corner_normals(mesh, matrix, node_id, adaptive_normals, false, warnings).world
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

/// Calculates mesh-space Mikk tangents for every UV layer, then applies Cycles' inverse-transpose
/// object-normal transform to each corner before hits interpolate them.
fn mesh_mikk_tangents(
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    vertex_positions: &BTreeMap<u32, DVec3>,
    uv_store: &UvStore,
    local_face_normals: Option<&[Option<Vec<DVec3>>]>,
    matrix: DMat4,
) -> Result<Vec<Vec<[DVec3; 3]>>> {
    if triangles.is_empty() {
        return Ok(Vec::new());
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
                .and_then(Option::as_deref)
                .and_then(|face_normals| corner_values(face_normals, corners))
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
                let world = normal_matrix
                    * DVec3::new(tangent[0] as f64, tangent[1] as f64, tangent[2] as f64);
                if world.is_finite() {
                    world
                } else {
                    DVec3::ZERO
                }
            }));
        }
    }
    Ok(tangents_by_triangle)
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

struct SceneShaderCore<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: super::path::SharedGeometry<'a>,
    mode: Mode,
    /// The camera's view space, in which Solid's studio lights are fixed.
    view: &'a ViewBasis,
    evaluator_workspace: RefCell<EvaluatorWorkspace>,
    /// Lazily populated once per render and shared by the serial pass and all band workers.
    world_probe: Arc<OnceLock<Result<Option<Arc<WorldProbe>>>>>,
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
    match mode {
        Mode::Solid => {}
        Mode::Beauty if geometry.shadows.is_some() => {}
        _ => return None,
    }
    Some(Box::new(SceneShader {
        core: SceneShaderCore {
            doc,
            snapshot,
            images,
            geometry: super::path::SharedGeometry::of(geometry),
            mode,
            view,
            evaluator_workspace: RefCell::new(EvaluatorWorkspace::default()),
            world_probe: Arc::new(OnceLock::new()),
        },
        warnings: &geometry.shader_warnings,
    }))
}

impl SurfaceShader for SceneShader<'_> {
    fn shade(&self, triangle: &Triangle, hit: &SurfaceHit<'_>) -> Result<SurfaceSample> {
        self.core.shade(self.warnings, triangle, hit)
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

/// The transparent tint the Beauty composite applies to a surface. EEVEE keeps a Transparent closure's colour per channel
/// only for a BLENDED material (`ForwardPipeline::use_colored_transparency`, `GPU_MATFLAG_TRANSPARENT_MAYBE_COLORED`).
/// A DITHERED material stores one alpha, `1 - average(transmittance)` (`eevee_surf_deferred.bsl.hh`, `out_color.a`), so
/// its tint is the channel mean. `alpha_mode` `blend` is BLENDED; `opaque` and `clip` are DITHERED
/// (`blend_bridge.py`, `surface_render_method`).
fn beauty_tint(material: Option<&Material>, tint: [f64; 3]) -> [f64; 3] {
    // A transmittance is a fraction of the light: EEVEE clamps the Color at zero and saturates the result
    // (`gpu_shader_material_transparent.glsl`, `eevee_surf_deferred.bsl.hh`), so a tint above one or below zero is cut to
    // [0, 1] before it can amplify or invert the background.
    let tint = tint.map(|channel| channel.clamp(0.0, 1.0));
    match material {
        Some(material) if material.alpha_mode != "blend" => {
            let mean = (tint[0] + tint[1] + tint[2]) / 3.0;
            [mean; 3]
        }
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
            view: self.view,
            evaluator_workspace: RefCell::new(EvaluatorWorkspace::default()),
            world_probe: Arc::clone(&self.world_probe),
        }
    }

    /// The render's world probe holds radiance at every texel of a cube map. The first shade that needs it builds it
    /// lazily, then all band workers share it because the world cannot change during a render
    /// (`LightProbeModule::sync_world`, `eevee_lightprobe.cc`). `None` means the scene has no world.
    fn world_probe(&self) -> Result<Option<&WorldProbe>> {
        let cached = self.world_probe.get_or_init(|| {
            let world = self
                .doc
                .scenes
                .get(&self.snapshot.scene_id)
                .and_then(|scene| scene.world.as_ref())
                .and_then(|id| self.doc.worlds.get(id));
            world
                .map(|world| {
                    WorldProbe::build(world, &self.doc.node_groups, self.images).map(Arc::new)
                })
                .transpose()
        });
        match cached {
            Ok(probe) => Ok(probe.as_deref()),
            Err(error) => Err(error.clone()),
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
        if matches!(self.mode, Mode::Beauty)
            && prepared.is_some_and(PreparedMaterial::is_volume_only)
        {
            return Ok(SurfaceSample {
                color: [0.0; 4],
                has_surface: false,
                transparent_tint: [1.0; 3],
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
            fallback = [ShaderClosure {
                weight: 1.0,
                params: shading_params(material, base, hit.normal),
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
            prepared_materials: self.geometry.prepared,
            has_volume_only: shadows.has_volume_only,
            point_maps: &shadows.point_maps,
            occluders: shadows.transparent_casters.then_some(&occluders),
        });
        let thickness = object_for_index(self.geometry.objects, triangle.object_index)
            .map_or(0.0, |object| object.thickness);
        let color = shade(
            self.doc,
            self.snapshot,
            hit.point,
            hit.shadow_point,
            hit.normal,
            hit.geometric,
            hit.view,
            closures,
            self.mode,
            shadow,
            thickness,
            self.world_probe()?,
        );
        occluders.finish()?;
        Ok(SurfaceSample {
            color,
            has_surface: true,
            transparent_tint: beauty_tint(material, surface_tint(closures)),
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
        evaluate_surface_with_workspace(material, prepared, &context, &mut workspace)
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
/// vertex values and corner layers the face's corner values; a value that is not there reads as
/// opaque black.
fn triangle_layer_colors(
    layers: &[&ColorLayer],
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
            ColorDomain::Corner => corners.map(|corner| {
                corner
                    .and_then(|corner| layer.values.get(corner_base + corner))
                    .copied()
                    .unwrap_or(BLACK)
            }),
        })
        .collect()
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
    let mut context = HitContext::new(&doc.node_groups)
        .with_images(images)
        .with_uv_layers(HitUvLayers {
            barycentric: pose.barycentric,
            ..uv_layers
        })
        .with_uv_tangents(&triangle.layer_tangents)
        .with_color_layers(color_maps, color_render)
        .with_warnings(warnings);
    context.uv = uv_override.unwrap_or_else(|| blend_uv(&triangle.uv, pose.barycentric));
    context = context.with_generated_corners(triangle.generated, pose.barycentric);
    context.position = pose.point;
    context.object = pose.object;
    context.object_to_world = object.object_to_world;
    context.world_to_object = object
        .world_to_local
        .unwrap_or_else(|| DMat4::from_cols_array(&[f64::NAN; 16]));
    context.object_color = object.object_color;
    context.object_pass_index = object.object_pass_index;
    context.object_random_number = object.object_random_number;
    context.material_pass_index = material_colors.and_then(|colors| colors.material_pass_index);
    context.normal = pose.normal;
    context.geometric_normal = pose.geometric_normal;
    context.back_facing = pose.back_facing;
    context.path_traced = pose.path_traced;
    if !pose.path_traced {
        context = context.with_realtime_position_f32(triangle.positions, pose.barycentric);
    }
    context.light_falloff_distance = None;
    context.view_direction = pose.view;
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

/// The transparent tint of a hit, as the composite applies it: a hit passes `(1 - opacity) * tint` of the light behind
/// it. EEVEE accumulates every transparent closure into one transmittance, `g_transmittance += transmittance * weight`
/// (`eevee_nodetree_lib.bsl.hh`, `closure_eval(ClosureTransparency)`), and Cycles sums the closures' transparent weights
/// the same way. So a mix of closures passes `sum w (1 - alpha) tint`, and the tint is that over the pass chance.
fn surface_tint(closures: &[ShaderClosure]) -> [f64; 3] {
    if closures.is_empty() {
        return [1.0; 3];
    }
    weighted_tint(
        closures.iter().map(|closure| {
            (
                closure.weight,
                closure.params.alpha,
                closure.params.transparent_tint,
            )
        }),
        opaque_share(closures),
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

#[expect(
    clippy::too_many_arguments,
    reason = "flat material shading consumes explicit evaluated surface and light values"
)]
fn shade(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    point: DVec3,
    shadow_point: DVec3,
    normal: DVec3,
    geometric: DVec3,
    view: DVec3,
    closures: &[ShaderClosure],
    mode: Mode,
    shadow: ShadowContext<'_>,
    object_thickness: f64,
    world: Option<&WorldProbe>,
) -> [f64; 4] {
    let opaque = opaque_share(closures);
    if !matches!(mode, Mode::Beauty) {
        let albedo = heaviest(closures, |closure| closure.weight)
            .map_or(DVec3::ZERO, |closure| base_albedo(&closure.params));
        return [albedo.x, albedo.y, albedo.z, opaque];
    }
    let scene = doc.scenes.get(&snapshot.scene_id);
    // The view is the camera ray's reverse. An orthographic camera's rays are parallel, which is EEVEE's view too
    // (`world_incident_vector`); a perspective ray points at the camera. Without one, the normal.
    let view = view.try_normalize().unwrap_or(normal);
    // EEVEE's `eevee.light_threshold` sets each local light's influence radius; Blender's default is 0.01.
    let light_threshold = scene.map_or(0.01, |scene| scene.render.eevee_light_threshold);
    let receiver = Receiver {
        point,
        shadow_point,
        geometric,
        view,
    };
    // The diffuse environment depends on the normal alone, so every closure shares it.
    // The diffuse environment depends on the normal alone, so every closure shares it.
    let diffuse_environment =
        world.map(|world| prefiltered_world(world, normal, 1.0, irradiance_probe_radiance));
    // Ambient occlusion attenuates the environment (indirect) light only, as EEVEE's horizon scan does; EEVEE's direct
    // light is never occluded by it. The hemisphere is the one the raster's occlusion pass casts (`ambient_visibility`).
    let ambient = shadow.map_or(1.0, |scene| {
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
    });
    // A closure's reflective share is its weight times its alpha. The raster composite multiplies a straight colour
    // by the opaque share (`render/raster.rs`, `composite_sample`), so the premultiplied sums are divided by it below.
    let mut reflected = DVec3::ZERO;
    let mut emitted = DVec3::ZERO;
    for closure in closures {
        let params = &closure.params;
        let share = closure.weight * params.alpha.clamp(0.0, 1.0);
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
        let mut lobes = DVec3::ZERO;
        if let (Some(world), Some(diffuse_environment)) = (world, diffuse_environment) {
            let diffuse_albedo = if params.thin_wall {
                let subsurface = params.subsurface_weight.clamp(0.0, 1.0);
                let reflection = f64::midpoint(1.0, -params.subsurface_anisotropy).clamp(0.0, 1.0);
                albedo * (1.0 - subsurface + subsurface * reflection)
            } else {
                albedo
            };
            let mut environment = diffuse_albedo * diffuse_environment * dielectric * lower;
            let reflection = (-view + normal * (2.0 * normal.dot(view))).normalize_or_zero();
            let specular_environment =
                prefiltered_world(world, reflection, roughness, probe_radiance);
            let specular_f0 = dielectric_f0(params.ior, params.specular_ior_level);
            let f0 = albedo * metallic + DVec3::splat(specular_f0 * dielectric);
            environment += specular_environment * f0 * (1.0_f64 - roughness).powi(2) * lower;
            lobes += environment * ambient;
            if params.thin_wall && params.subsurface_weight > 0.0 {
                let sphere_thickness = matches!(thickness, Thickness::Sphere(value) if value > 0.0);
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
                lobes += glass_environment(world, &glass);
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
            lobes += light_contribution(
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
            ) * influence;
            if let Some(glass) = &glass {
                lobes += glass_direct_contribution(
                    light,
                    matrix,
                    receiver,
                    glass,
                    thickness,
                    ray_visibility,
                    shadow,
                    point_shadow,
                ) * influence;
            }
        }
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
    let reflected = reflected.max(DVec3::ZERO).min(DVec3::splat(1.0));
    let result = reflected + emitted;
    [result.x, result.y, result.z, opaque]
}

/// The environment's share of the glass closures: the reflection at its mirror direction and the transmission at the
/// probe direction, each blurred with its roughness as EEVEE's light probe is (`eevee_transmission`).
fn glass_environment(world: &WorldProbe, closures: &GlassClosures) -> DVec3 {
    let reflected = prefiltered_world(
        world,
        closures.reflection_direction,
        closures.reflection_roughness,
        probe_radiance,
    );
    let transmitted = prefiltered_world(
        world,
        closures.transmission_direction,
        closures.transmission_roughness,
        probe_radiance,
    );
    reflected * closures.reflection + transmitted * closures.transmission
}

/// The world's filtered radiance along `axis` for `roughness`: five taps spread over a lobe of that roughness, read from
/// the render's world probe. Each tap's texel is `stored` (the capture format of the probe it is read from) before the
/// blur: a diffuse read stores irradiance and a glossy or glass read stores radiance (`eevee_transmission`).
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
        let Some(direct) = direct_emitter(light, matrix, point) else {
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
    let color = DVec3::from_array(light.color).max(DVec3::ZERO);
    let energy = light.energy.max(0.0);
    let position = matrix.transform_point3(DVec3::ZERO);
    let axis = matrix.transform_vector3(DVec3::Z).normalize_or_zero();
    match light.light_type {
        crate::model::LightType::Sun => {
            let radius = sun_disc_radius(light.sun_angle);
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
                sun_radius: radius,
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

/// EEVEE's direct light through the glass closures (`light_eval_single`, with `eevee_light_eval.bsl.hh`'s attenuation):
/// each closure's colour, times the light's colour and radiance, its attenuation, its visibility, and the LTC form
/// factor of the emitter through the closure's inverse matrix. The reflection is shadowed from the shading point; the
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
) -> DVec3 {
    let Some(direct) = direct_emitter(light, matrix, receiver.point) else {
        return DVec3::ZERO;
    };
    let factors = LightFactors::of(light);
    let mut result = DVec3::ZERO;

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
        result += glass.reflection
            * direct.radiance
            * (direct.attenuation * facing * visibility * form * factors.specular);
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
        result += glass.transmission
            * direct.radiance
            * (direct.attenuation * facing * visibility * form * factors.transmission);
    }
    result
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
    let radius = sun_disc_radius(light.sun_angle);
    let weight = energy / SUN_DISC_SAMPLES as f64;
    let mut mean_visibility = 0.0;
    let mut specular = DVec3::ZERO;
    let mut translucent = DVec3::ZERO;
    for sample in sun_disc_directions(toward_light, radius) {
        let visibility = match shadow {
            Some(shadow) => ray_transmittance(
                shadow,
                shadow_origin(shadow_point, geometric, sample),
                sample,
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
        ColorSlotReferences, carried_color_layers, color_layer_references, corner_values,
        extract_geometry, source_has_color_layer, subdivide_displaced, surface_shader,
        triangle_corners, triangle_layer_colors, triangle_layer_uvs, world_face_normals,
    };
    use crate::render::principled_bsdf::{
        diffuse_coefficient, layers as principled_layers, specular_coefficient,
    };
    use crate::render::raster::{Camera, Projection, StereoMode, ViewBasis};
    use crate::shader::BsdfParams;
    use crate::{
        eval::{EvaluationContext, Snapshot},
        geom::{
            Mesh,
            color_layers::{ColorDomain, ColorLayer, ColorStore},
            uv_layers::{UvLayer, UvStore},
        },
        graph::{GraphKind, GraphNode, NodeGroup},
        image::{ImageData, ImageInterpolation},
        model::{Id, Material, MaterialLink, RayVisibility, SceneDoc, ViewportDisplay},
        render::{
            light_model::{
                AreaShape, area_irradiance_per_radiance, area_radiance, area_visibility_samples,
            },
            raster::{AlphaMode, Mode, SurfaceHit, Triangle},
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
                    view: DVec3::Z,
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
            lit_per_pixel: false,
            ray_mask: RayVisibility::default().mask(),
        }
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
            beauty_tint(Some(&material("blend")), [0.2, 0.8, 0.4]),
            [0.2, 0.8, 0.4]
        );
    }

    /// A DITHERED material takes the mean of the Color's channels: (0.2 + 0.8 + 0.4) / 3 on every channel.
    #[test]
    fn dithered_material_takes_the_channel_mean() {
        let mean = (0.2 + 0.8 + 0.4) / 3.0;
        for alpha_mode in ["opaque", "clip"] {
            for channel in beauty_tint(Some(&material(alpha_mode)), [0.2, 0.8, 0.4]) {
                assert!((channel - mean).abs() < 1.0e-12, "{alpha_mode}: {channel}");
            }
        }
    }

    /// Tints are clamped to [0, 1] first. BLENDED clamps each channel: 2 -> 1, -1 -> 0, 0.5 stays. DITHERED takes the
    /// mean of the clamped channels: (1 + 0 + 0.5) / 3 = 0.5.
    #[test]
    fn tints_are_clamped_before_use() {
        assert_eq!(
            beauty_tint(Some(&material("blend")), [2.0, -1.0, 0.5]),
            [1.0, 0.0, 0.5]
        );
        let dithered = beauty_tint(Some(&material("clip")), [2.0, -1.0, 0.5]);
        assert!((dithered[0] - 0.5).abs() < 1.0e-12, "{dithered:?}");
    }

    #[test]
    fn no_material_keeps_the_tint_clamped() {
        assert_eq!(beauty_tint(None, [0.5, 1.5, -0.2]), [0.5, 1.0, 0.0]);
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
