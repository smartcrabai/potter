use std::{
    cell::RefCell,
    collections::{BTreeMap, HashSet},
};

use glam::{DMat3, DMat4, DVec3};
use serde::Serialize;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    geom::{
        normals::render_corner_normals,
        uv_layers::{self, UvStore},
    },
    image::ImageData,
    model::{Id, Material, SceneDoc},
    shader::{HitContext, HitUvLayers, evaluate_displacement},
};

use super::{
    bvh::Bvh,
    raster::{AlphaMode, Line, Mode, SurfaceShader, Triangle, blend_corner_normals, blend_uv},
};

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
    pub materials: Vec<Material>,
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
}

/// Shadow acceleration for the realtime Beauty engine; built once per geometry and shared by
/// the flat-triangle baking and the per-pixel shading of smooth triangles.
pub(super) struct RealtimeShadows {
    bvh: Bvh,
    point_maps: BTreeMap<Id, PointShadowMap>,
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

type ShadowContext<'a> = Option<(&'a Bvh, &'a [Triangle], &'a BTreeMap<Id, PointShadowMap>)>;
pub(super) fn extract_geometry(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    mode: Mode,
    render_visibility: bool,
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

    let mut materials = vec![Material::default()];
    let mut material_indices = std::collections::BTreeMap::<Id, usize>::new();
    let mut triangles = Vec::new();
    let mut lines = Vec::new();
    let mut objects = Vec::new();
    let mut elements = Vec::new();
    let mut bounds: Option<(DVec3, DVec3)> = None;
    let mut warnings = Vec::new();
    let shader_warnings: RefCell<Vec<serde_json::Value>> = RefCell::new(Vec::new());
    for (node_id, node) in &doc.nodes {
        if !visible_nodes.contains(node_id)
            || !node.visible
            || (render_visibility && !node.render_visible)
        {
            continue;
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
        objects.push(ObjectRecord {
            index: object_index,
            node_id: node_id.clone(),
            data_id: node.data.clone(),
            instance_path: snapshot
                .instance_paths
                .get(node_id)
                .and_then(|paths| paths.first())
                .cloned()
                .unwrap_or_default(),
            uv_layers: uv_names.clone(),
            uv_render,
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
            let mesh_triangles = mesh.triangulate().map_err(|error| {
                PotError::with_details(
                    ErrorCode::EvaluationFailed,
                    format!("mesh triangulation failed: {error}"),
                    serde_json::json!({ "node_id": node_id }),
                )
            })?;
            let face_normals = world_face_normals(mesh, matrix, node_id, &mut warnings);
            let mut triangle_offset = 0_usize;
            for (face_index, face) in mesh.faces.iter().enumerate() {
                let material_slot = usize::try_from(face.material_index).ok();
                let binding = material_slot
                    .and_then(|slot| snapshot.slot_binding(node_id, node, slot));
                let render_layer = binding.map_or(uv_render, |binding| {
                    binding.uv.render_index.map(usize::from)
                });
                let uv_available_mask = binding.map(|binding| binding.uv.available_mask);
                let face_corner_normals = face_normals[face_index].as_deref();
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
                let material_id = material_slot
                    .and_then(|slot| snapshot.slot_material(node_id, node, slot));
                let material_index = material_id.map_or(0, |id| {
                    if let Some(index) = material_indices.get(id).copied() {
                        index
                    } else if let Some(material) = doc.materials.get(id) {
                        let index = materials.len();
                        materials.push(material.clone());
                        material_indices.insert(id.clone(), index);
                        index
                    } else {
                        0
                    }
                });
                let material = materials.get(material_index);
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
                    }
                    let uv = render_layer
                        .and_then(|index| layer_uvs.get(index).copied())
                        .unwrap_or([[0.0, 1.0]; 3]);
                    let raw_normal =
                        (positions[1] - positions[0]).cross(positions[2] - positions[0]);
                    if raw_normal.length_squared() <= f64::EPSILON || !raw_normal.is_finite() {
                        continue;
                    }
                    let corner_normals =
                        face_corner_normals.and_then(|normals| corner_values(normals, corners));
                    let normal = raw_normal.normalize();
                    let center = (positions[0] + positions[1] + positions[2]) / 3.0;
                    // Smooth triangles take lighting per pixel (raster) or per hit (path tracer).
                    let color = if corner_normals.is_some()
                        || (path_traced && matches!(mode, Mode::Beauty))
                    {
                        base_color
                    } else {
                        shade(
                            doc,
                            snapshot,
                            center,
                            center,
                            normal,
                            normal,
                            base_color,
                            material.map_or(0.0, |value| value.metallic),
                            material.map_or(0.8, |value| value.roughness),
                            mode,
                            None,
                            images,
                        )?
                    };
                    let triangle = Triangle {
                        positions,
                        object_index,
                        element_index,
                        color,
                        alpha_mode: material.map_or(AlphaMode::Opaque, |material| {
                            match material.alpha_mode.as_str() {
                                "blend" => AlphaMode::Blend,
                                "clip" => AlphaMode::Clip,
                                _ => AlphaMode::Opaque,
                            }
                        }),
                        alpha_threshold: material.map_or(0.5, |material| material.alpha_threshold),
                        double_sided: material.is_some_and(|value| value.double_sided),
                        material_index,
                        uv,
                        layer_uvs,
                        normals: corner_normals,
                    };
                    if let Some(material) = material.filter(|material| {
                        material.displacement_method == "displacement"
                            && material.node_tree.is_some()
                    }) {
                        let layers = HitUvLayers {
                            names: &uv_names,
                            corners: &triangle.layer_uvs,
                            positions: triangle.positions,
                            barycentric: [1.0, 0.0, 0.0],
                            render: render_layer,
                            available_mask: uv_available_mask,
                        };
                        triangles.extend(subdivide_displaced_triangle(
                            doc,
                            material,
                            matrix,
                            &triangle,
                            images,
                            layers,
                            &shader_warnings,
                        )?);
                    } else {
                        triangles.push(triangle);
                    }
                }
                triangle_offset = face_triangle_end;
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
            let default_point_color = node
                .materials
                .first()
                .and_then(|material_id| doc.materials.get(material_id))
                .map_or([0.6, 0.6, 0.6, 1.0], |material| material.base_color);
            let world_scale = [
                matrix.transform_vector3(DVec3::X).length(),
                matrix.transform_vector3(DVec3::Y).length(),
                matrix.transform_vector3(DVec3::Z).length(),
            ]
            .into_iter()
            .fold(0.0_f64, f64::max);
            for vertex in mesh.vertices.iter().filter(|vertex| {
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
                let color = read_vertex_color(mesh, vertex.id).unwrap_or(default_point_color);
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
                    color,
                    object_index,
                    element_index,
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
                    lines.push(Line {
                        start: DVec3::from_array(stroke.points_world[point_index]),
                        end: DVec3::from_array(stroke.points_world[next_index]),
                        radius_start: stroke.radius[point_index],
                        radius_end: stroke.radius[next_index],
                        color: stroke.color,
                        object_index,
                        element_index: stroke_element_index,
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
            let color = node
                .materials
                .first()
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
                    });
                }
            }
        }
    }
    let mut geometry = Geometry {
        triangles,
        lines,
        materials,
        objects,
        elements,
        bounds,
        warnings,
        shader_warnings,
        shadows: None,
    };
    if !path_traced && matches!(mode, Mode::Beauty) && !geometry.triangles.is_empty() {
        geometry.triangles = subdivide_realtime_triangles(&geometry.triangles)?;
        let shadow_maps = build_point_shadow_maps(doc, snapshot, &geometry.triangles)?;
        let bvh = Bvh::build(&geometry.triangles)?;
        for index in 0..geometry.triangles.len() {
            let triangle = &geometry.triangles[index];
            if triangle.normals.is_some() {
                // Smooth triangles are shaded per pixel from their corner normals.
                continue;
            }
            let material = geometry
                .materials
                .get(triangle.material_index)
                .unwrap_or(&geometry.materials[0]);
            let normal = (triangle.positions[1] - triangle.positions[0])
                .cross(triangle.positions[2] - triangle.positions[0])
                .normalize_or_zero();
            let center =
                (triangle.positions[0] + triangle.positions[1] + triangle.positions[2]) / 3.0;
            let color = shade(
                doc,
                snapshot,
                center,
                center,
                normal,
                normal,
                material.base_color,
                material.metallic,
                material.roughness,
                mode,
                Some((&bvh, &geometry.triangles, &shadow_maps)),
                images,
            )?;
            if let Some(target) = geometry.triangles.get_mut(index) {
                target.color = color;
            }
        }
        geometry.shadows = Some(RealtimeShadows {
            bvh,
            point_maps: shadow_maps,
        });
    }
    Ok(geometry)
}

fn build_point_shadow_maps(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    triangles: &[Triangle],
) -> Result<BTreeMap<Id, PointShadowMap>> {
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
        maps.insert(id.clone(), PointShadowMap::build(position, triangles)?);
    }
    Ok(maps)
}

impl PointShadowMap {
    fn build(position: DVec3, triangles: &[Triangle]) -> Result<Self> {
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
        for triangle in triangles {
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
        let bias = (depth * 0.001).max(1.0e-4);
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

fn subdivide_realtime_triangles(triangles: &[Triangle]) -> Result<Vec<Triangle>> {
    let triangle_capacity = triangles.len().checked_mul(16).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "realtime shadow subdivision exceeds the supported triangle count",
        )
    })?;
    let mut refined = Vec::new();
    refined.try_reserve_exact(triangle_capacity).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "realtime shadow subdivision exceeds available memory",
        )
    })?;
    for triangle in triangles {
        // Smooth triangles are shaded per pixel from their corner normals, so splitting them
        // would only multiply the raster and shadow work without changing the lighting.
        if triangle.normals.is_some() {
            refined.push(triangle.clone());
            continue;
        }
        for half in subdivide_triangle_once(triangle) {
            refined.extend(subdivide_triangle_once(&half));
        }
    }
    Ok(refined)
}

fn subdivide_triangle_once(triangle: &Triangle) -> [Triangle; 4] {
    let positions = [
        triangle.positions[0],
        triangle.positions[1],
        triangle.positions[2],
        (triangle.positions[0] + triangle.positions[1]) * 0.5,
        (triangle.positions[1] + triangle.positions[2]) * 0.5,
        (triangle.positions[2] + triangle.positions[0]) * 0.5,
    ];
    let uv = corner_midpoints(triangle.uv);
    let layer_uvs: Vec<[[f64; 2]; 6]> = triangle
        .layer_uvs
        .iter()
        .map(|corners| corner_midpoints(*corners))
        .collect();
    let subtriangles: [[usize; 3]; 4] = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]];
    subtriangles.map(|corners| {
        triangle.with_corners(
            corners.map(|vertex| positions[vertex]),
            corners.map(|vertex| uv[vertex]),
            layer_uvs
                .iter()
                .map(|table| corners.map(|vertex| table[vertex]))
                .collect(),
        )
    })
}

/// The three corners of a triangle followed by the midpoints of its edges (0-1, 1-2, 2-0).
fn corner_midpoints(corners: [[f64; 2]; 3]) -> [[f64; 2]; 6] {
    let midpoint =
        |a: [f64; 2], b: [f64; 2]| [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])];
    [
        corners[0],
        corners[1],
        corners[2],
        midpoint(corners[0], corners[1]),
        midpoint(corners[1], corners[2]),
        midpoint(corners[2], corners[0]),
    ]
}

fn subdivide_displaced_triangle(
    doc: &SceneDoc,
    material: &Material,
    object_to_world: DMat4,
    triangle: &Triangle,
    images: &BTreeMap<String, ImageData>,
    layers: HitUvLayers<'_>,
    warnings: &RefCell<Vec<serde_json::Value>>,
) -> Result<[Triangle; 4]> {
    subdivide_displaced(triangle, object_to_world, |sample| {
        let mut context = HitContext::new(&doc.node_groups)
            .with_images(images)
            .with_uv_layers(HitUvLayers {
                barycentric: sample.barycentric,
                ..layers
            })
            .with_warnings(warnings);
        context.uv = sample.uv;
        context.position = sample.point;
        context.object = sample.object;
        context.generated = sample.object;
        context.normal = sample.normal;
        evaluate_displacement(material, &context)
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
    let mut uvs = [[0.0; 2]; 6];
    let mut layer_samples = vec![[[0.0; 2]; 6]; triangle.layer_uvs.len()];
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
        let displacement = displacement(&sample)?;
        positions[index] = point + sample_normal * displacement;
        sample_normals[index] = sample_normal;
        uvs[index] = uv;
        for (table, corners) in layer_samples.iter_mut().zip(&triangle.layer_uvs) {
            table[index] = blend_uv(corners, weights);
        }
    }
    let indices: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 4], [2, 4, 5], [1, 4, 2]];
    Ok(indices.map(|corners| {
        let mut subtriangle = triangle.with_corners(
            corners.map(|vertex| positions[vertex]),
            corners.map(|vertex| uvs[vertex]),
            layer_samples
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

/// World-space corner normals per face; `None` marks a flat face. A mesh whose corner normals
/// cannot be built keeps flat face normals and records a warning.
fn world_face_normals(
    mesh: &crate::geom::Mesh,
    matrix: DMat4,
    node_id: &Id,
    warnings: &mut Vec<serde_json::Value>,
) -> Vec<Option<Vec<DVec3>>> {
    let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let local = match render_corner_normals(mesh, &positions) {
        Ok(faces) => faces,
        Err(error) => {
            warnings.push(serde_json::json!({
                "code": "shading.corner_normals_failed",
                "message": format!(
                    "smooth shading normals could not be computed ({error}); flat face normals were used"
                ),
                "node_id": node_id,
            }));
            return vec![None; mesh.faces.len()];
        }
    };
    // Inverse-transpose keeps normals perpendicular to the surface under non-uniform scale.
    let normal_matrix = DMat3::from_mat4(matrix).inverse().transpose();
    let mut degenerate = false;
    let faces = local
        .into_iter()
        .map(|face| {
            let world: Vec<DVec3> = face?
                .into_iter()
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
    faces
}

/// Per-pixel lighting for smooth triangles in raster Solid and realtime Beauty output. Flat
/// triangles keep the lighting baked into `Triangle::color`, which equals per-pixel lighting
/// for a constant normal.
pub(super) struct SceneShader<'a> {
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: &'a Geometry,
    mode: Mode,
}

/// The per-pixel shader for `mode`, or `None` when the raster outputs unlit or per-triangle colour.
pub(super) fn surface_shader<'a>(
    doc: &'a SceneDoc,
    snapshot: &'a Snapshot,
    images: &'a BTreeMap<String, ImageData>,
    geometry: &'a Geometry,
    mode: Mode,
) -> Option<Box<dyn SurfaceShader + 'a>> {
    match mode {
        Mode::Solid => {}
        Mode::Beauty if geometry.shadows.is_some() => {}
        _ => return None,
    }
    Some(Box::new(SceneShader {
        doc,
        snapshot,
        images,
        geometry,
        mode,
    }))
}

impl SurfaceShader for SceneShader<'_> {
    fn shade(
        &self,
        triangle: &Triangle,
        point: DVec3,
        shadow_point: DVec3,
        normal: DVec3,
        geometric: DVec3,
    ) -> Result<[f64; 3]> {
        let material = self.geometry.materials.get(triangle.material_index);
        let shadow = self.geometry.shadows.as_ref().map(|shadows| {
            (
                &shadows.bvh,
                self.geometry.triangles.as_slice(),
                &shadows.point_maps,
            )
        });
        let color = shade(
            self.doc,
            self.snapshot,
            point,
            shadow_point,
            normal,
            geometric,
            triangle.color,
            material.map_or(0.0, |value| value.metallic),
            material.map_or(0.8, |value| value.roughness),
            self.mode,
            shadow,
            self.images,
        )?;
        Ok([color[0], color[1], color[2]])
    }
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

fn read_vertex_color(mesh: &crate::geom::Mesh, vertex_id: u32) -> Option<[f64; 4]> {
    let key = format!("v{vertex_id}");
    let values = mesh
        .attributes
        .get("color")?
        .get("values")?
        .get(key)?
        .as_array()?;
    let color = [
        values.first()?.as_f64()?,
        values.get(1)?.as_f64()?,
        values.get(2)?.as_f64()?,
        values
            .get(3)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0),
    ];
    color
        .iter()
        .all(|channel| channel.is_finite())
        .then_some(color)
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
    base: [f64; 4],
    metallic: f64,
    roughness: f64,
    mode: Mode,
    shadow: ShadowContext<'_>,
    images: &BTreeMap<String, ImageData>,
) -> Result<[f64; 4]> {
    if !matches!(mode, Mode::Solid | Mode::Beauty) {
        return Ok(base);
    }
    let scene = doc.scenes.get(&snapshot.scene_id);
    let mut result = DVec3::ZERO;
    let albedo = DVec3::new(base[0], base[1], base[2]);
    if matches!(mode, Mode::Solid) {
        result += albedo * 0.30;
        for (direction, intensity, color) in [
            (
                DVec3::new(-0.45, -0.55, 0.70).normalize(),
                1.05,
                DVec3::splat(1.0),
            ),
            (
                DVec3::new(0.65, 0.35, 0.55).normalize(),
                0.45,
                DVec3::new(0.72, 0.82, 1.0),
            ),
        ] {
            result += direct_brdf(
                albedo, metallic, roughness, normal, -direction, normal, color, intensity,
            );
        }
    } else {
        let has_lights = doc.nodes.iter().any(|(id, light_node)| {
            light_node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .is_some_and(|block| block.light.is_some())
                && snapshot.nodes.contains_key(id)
        });
        if let Some(world) = scene
            .and_then(|value| value.world.as_ref())
            .and_then(|id| doc.worlds.get(id))
        {
            let diffuse_environment =
                prefiltered_world(world, &doc.node_groups, images, normal, 1.0)?;
            let metallic = metallic.clamp(0.0, 1.0);
            let roughness = roughness.clamp(0.0, 1.0);
            result += albedo * diffuse_environment * (1.0 - metallic);
            let view_direction = scene
                .and_then(|scene| scene.camera.as_ref())
                .and_then(|camera_id| snapshot.nodes.get(camera_id))
                .map(|camera| {
                    (DMat4::from_cols_array(&camera.world_matrix).transform_point3(DVec3::ZERO)
                        - point)
                        .normalize_or_zero()
                })
                .filter(|direction| direction.length_squared() > f64::EPSILON)
                .unwrap_or(normal);
            let reflection =
                (-view_direction + normal * (2.0 * normal.dot(view_direction))).normalize_or_zero();
            let specular_environment =
                prefiltered_world(world, &doc.node_groups, images, reflection, roughness)?;
            let f0 = albedo * metallic + DVec3::splat(0.04 * (1.0 - metallic));
            result += specular_environment * f0 * (1.0 - roughness).powi(2);
        } else if !has_lights {
            result += albedo * 0.08;
        }
        for (id, light_node) in &doc.nodes {
            let Some(light) = light_node
                .data
                .as_ref()
                .and_then(|data_id| doc.data_blocks.get(data_id))
                .and_then(|block| block.light.as_ref())
            else {
                continue;
            };
            let Some(evaluated) = snapshot.nodes.get(id) else {
                continue;
            };
            let matrix = DMat4::from_cols_array(&evaluated.world_matrix);
            let point_shadow = shadow.and_then(|(_, _, maps)| maps.get(id));
            result += light_contribution(
                light,
                matrix,
                point,
                shadow_point,
                normal,
                geometric,
                albedo,
                metallic,
                roughness,
                shadow,
                point_shadow,
            );
        }
    }
    let result = result.max(DVec3::ZERO).min(DVec3::splat(1.0));
    Ok([result.x, result.y, result.z, base[3]])
}

fn prefiltered_world(
    world: &crate::model::World,
    groups: &crate::model::Registry<crate::graph::NodeGroup>,
    images: &BTreeMap<String, ImageData>,
    axis: DVec3,
    roughness: f64,
) -> Result<DVec3> {
    const FILTER_OFFSETS: [[f64; 2]; 5] =
        [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]];
    let axis = axis.normalize_or_zero();
    if axis == DVec3::ZERO {
        return Ok(DVec3::ZERO);
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
        irradiance += crate::shader::evaluate_world(world, groups, images, direction)?;
    }
    Ok(irradiance / 5.0)
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
    albedo: DVec3,
    metallic: f64,
    roughness: f64,
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
) -> DVec3 {
    let color = DVec3::from_array(light.color).max(DVec3::ZERO);
    let energy = light.energy.max(0.0);
    let position = matrix.transform_point3(DVec3::ZERO);
    match light.light_type {
        crate::model::LightType::Sun => {
            let toward_light = matrix.transform_vector3(DVec3::Z).normalize_or_zero();
            direct_brdf(
                albedo,
                metallic,
                roughness,
                normal,
                toward_light,
                normal,
                color,
                energy,
            ) * visibility_for_light(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                toward_light,
                f64::INFINITY,
            )
        }
        crate::model::LightType::Point => {
            let delta = position - point;
            let distance_squared = delta.length_squared().max(1.0e-6);
            let toward_light = delta.normalize_or_zero();
            direct_brdf(
                albedo,
                metallic,
                roughness,
                normal,
                toward_light,
                normal,
                color,
                energy / distance_squared,
            ) * visibility_for_light(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                toward_light,
                distance_squared.sqrt(),
            )
        }
        crate::model::LightType::Spot => {
            let delta = position - point;
            let distance_squared = delta.length_squared().max(1.0e-6);
            let toward_light = delta.normalize_or_zero();
            let direction = matrix.transform_vector3(-DVec3::Z).normalize_or_zero();
            let cosine = direction.dot(-toward_light).clamp(-1.0, 1.0);
            let outer_cosine = (light.spot_size.clamp(0.0, std::f64::consts::PI) * 0.5).cos();
            if cosine <= outer_cosine {
                return DVec3::ZERO;
            }
            let inner_angle = light.spot_size.clamp(0.0, std::f64::consts::PI)
                * (1.0 - light.spot_blend.clamp(0.0, 1.0))
                * 0.5;
            let inner_cosine = inner_angle.cos();
            let falloff = if inner_cosine <= outer_cosine + f64::EPSILON {
                1.0
            } else {
                ((cosine - outer_cosine) / (inner_cosine - outer_cosine)).clamp(0.0, 1.0)
            };
            direct_brdf(
                albedo,
                metallic,
                roughness,
                normal,
                toward_light,
                normal,
                color,
                (energy / distance_squared) * falloff,
            ) * visibility_for_light(
                shadow,
                point_shadow,
                shadow_point,
                geometric,
                toward_light,
                distance_squared.sqrt(),
            )
        }
        crate::model::LightType::Area => {
            let radius = light.radius.max(0.0);
            let offsets = [
                DVec3::ZERO,
                DVec3::new(radius, 0.0, 0.0),
                DVec3::new(-radius, 0.0, 0.0),
                DVec3::new(0.0, radius, 0.0),
                DVec3::new(0.0, -radius, 0.0),
            ];
            let mut sum = DVec3::ZERO;
            for offset in offsets {
                let sample_position = position + matrix.transform_vector3(offset);
                let delta = sample_position - point;
                let distance_squared = delta.length_squared().max(1.0e-6);
                let toward_light = delta.normalize_or_zero();
                sum += direct_brdf(
                    albedo,
                    metallic,
                    roughness,
                    normal,
                    toward_light,
                    normal,
                    color,
                    energy / distance_squared,
                ) * visibility_for_light(
                    shadow,
                    point_shadow,
                    shadow_point,
                    geometric,
                    toward_light,
                    distance_squared.sqrt(),
                );
            }
            sum / 5.0
        }
    }
}

fn visibility_for_light(
    shadow: ShadowContext<'_>,
    point_shadow: Option<&PointShadowMap>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
) -> f64 {
    point_shadow.map_or_else(
        || shadow_visibility(shadow, point, normal, direction, distance),
        |shadow_map| shadow_map.visibility(point),
    )
}

fn shadow_visibility(
    shadow: ShadowContext<'_>,
    point: DVec3,
    normal: DVec3,
    direction: DVec3,
    distance: f64,
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
    let Some((bvh, triangles, _)) = shadow else {
        return 1.0;
    };
    let direction = direction.normalize_or_zero();
    if direction == DVec3::ZERO {
        return 0.0;
    }
    let epsilon = 1.0e-5;
    let normal = normal.normalize_or_zero();
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(DVec3::Z).normalize_or_zero()
    } else {
        normal.cross(DVec3::Y).normalize_or_zero()
    };
    let bitangent = normal.cross(tangent).normalize_or_zero();
    let angular_radius = if distance.is_finite() {
        (0.02 / distance.max(0.02)).min(0.02)
    } else {
        0.003
    };
    let maximum_distance = if distance.is_finite() {
        (distance - epsilon).max(0.0)
    } else {
        f64::INFINITY
    };
    if maximum_distance <= epsilon {
        return 1.0;
    }
    let mut visible = 0.0;
    for offset in PCF_OFFSETS {
        let sample_direction = (direction
            + tangent * (offset[0] * angular_radius)
            + bitangent * (offset[1] * angular_radius))
            .normalize_or_zero();
        let side = if normal.dot(sample_direction) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let origin = point + normal * (side * epsilon);
        if bvh
            .closest_hit_two_sided(
                triangles,
                origin,
                sample_direction,
                epsilon,
                maximum_distance,
            )
            .is_none()
        {
            visible += 1.0;
        }
    }
    visible / 9.0
}

#[expect(
    clippy::too_many_arguments,
    reason = "Cook-Torrance evaluation consumes explicit BRDF and light inputs"
)]
fn direct_brdf(
    albedo: DVec3,
    metallic: f64,
    roughness: f64,
    normal: DVec3,
    light_dir: DVec3,
    view_dir: DVec3,
    light_color: DVec3,
    intensity: f64,
) -> DVec3 {
    let normal_light_cosine = normal.dot(light_dir).max(0.0);
    let normal_view_cosine = normal.dot(view_dir).max(0.0);
    if normal_light_cosine <= 0.0 || normal_view_cosine <= 0.0 {
        return DVec3::ZERO;
    }
    let half_vector = (light_dir + view_dir).normalize_or_zero();
    let normal_half_cosine = normal.dot(half_vector).max(0.0);
    let view_half_cosine = view_dir.dot(half_vector).max(0.0);
    let alpha = roughness.max(0.04).powi(2);
    let alpha_squared = alpha * alpha;
    let distribution_denominator =
        (normal_half_cosine * normal_half_cosine * (alpha_squared - 1.0) + 1.0).powi(2)
            * std::f64::consts::PI;
    let distribution = alpha_squared / distribution_denominator.max(1.0e-9);
    let smith_k = (roughness + 1.0).powi(2) / 8.0;
    let geometry = |cosine: f64| cosine / (cosine * (1.0 - smith_k) + smith_k).max(1.0e-9);
    let geometry_term = geometry(normal_light_cosine) * geometry(normal_view_cosine);
    let reflectance_at_normal = DVec3::splat(0.04) * (1.0 - metallic) + albedo * metallic;
    let fresnel = reflectance_at_normal
        + (DVec3::ONE - reflectance_at_normal) * (1.0 - view_half_cosine).powi(5);
    let specular = fresnel
        * (distribution * geometry_term
            / (4.0 * normal_view_cosine * normal_light_cosine).max(1.0e-9));
    let diffuse = (DVec3::ONE - fresnel) * albedo * ((1.0 - metallic) / std::f64::consts::PI);
    (diffuse + specular) * light_color * (intensity * normal_light_cosine)
}

#[cfg(test)]
mod shading_normal_tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use glam::{DMat4, DVec3};
    use serde_json::json;

    use super::{
        corner_values, subdivide_displaced, triangle_corners, triangle_layer_uvs,
        world_face_normals,
    };
    use crate::{
        geom::{
            uv_layers::{UvLayer, UvStore},
            Mesh,
        },
        model::Id,
        render::raster::{AlphaMode, Triangle},
    };

    fn node_id() -> Id {
        Id::new("node").expect("valid id")
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
        let faces = world_face_normals(&mesh, DMat4::IDENTITY, &node_id(), &mut warnings);
        assert_eq!(faces, vec![None]);
        assert_eq!(warnings, Vec::<serde_json::Value>::new());
    }

    #[test]
    fn smooth_corner_normals_use_the_inverse_transpose_of_the_object_matrix() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::Y, DVec3::new(1.0, 0.0, -1.0)], true);
        let matrix = DMat4::from_scale(DVec3::new(1.0, 1.0, 3.0));
        let mut warnings = Vec::new();
        let faces = world_face_normals(&mesh, matrix, &node_id(), &mut warnings);
        let corners = faces[0].as_ref().expect("smooth face has corner normals");
        let expected = DVec3::new(-3.0, 0.0, -1.0).normalize();
        assert_eq!(corners.len(), 3);
        for corner in corners {
            assert!(corner.abs_diff_eq(expected, 1e-9), "got {corner:?}");
        }
        assert_eq!(warnings, Vec::<serde_json::Value>::new());
    }

    #[test]
    fn degenerate_smooth_fans_fall_back_to_flat_with_a_warning() {
        let mesh = single_face_mesh([DVec3::ZERO, DVec3::X, DVec3::new(2.0, 0.0, 0.0)], true);
        let mut warnings = Vec::new();
        let faces = world_face_normals(&mesh, DMat4::IDENTITY, &node_id(), &mut warnings);
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
        let faces = world_face_normals(&mesh, matrix, &node_id(), &mut warnings);
        assert_eq!(faces, vec![None]);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["code"], "shading.corner_normals_failed");
    }
    fn source_triangle(positions: [DVec3; 3], normals: [DVec3; 3]) -> Triangle {
        Triangle {
            positions,
            object_index: 1,
            element_index: 1,
            color: [1.0; 4],
            alpha_mode: AlphaMode::Opaque,
            alpha_threshold: 0.5,
            double_sided: false,
            material_index: 0,
            uv: [[0.0; 2]; 3],
            layer_uvs: Vec::new(),
            normals: Some(normals),
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
}
