use std::collections::{BTreeMap, HashSet};

use glam::{DMat4, DVec3};
use serde::Serialize;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    image::ImageData,
    model::{Id, Material, SceneDoc},
    shader::{HitContext, evaluate_displacement},
};

use super::{
    bvh::Bvh,
    raster::{AlphaMode, Line, Mode, Triangle},
};

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub(super) struct ObjectRecord {
    pub index: u32,
    pub node_id: Id,
    pub data_id: Option<Id>,
    pub instance_path: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub(super) struct ElementRecord {
    pub index: u32,
    pub node_id: Id,
    pub domain: String,
    pub element_id: String,
}

pub(super) struct Geometry {
    pub triangles: Vec<Triangle>,
    pub lines: Vec<Line>,
    pub materials: Vec<Material>,
    pub objects: Vec<ObjectRecord>,
    pub elements: Vec<ElementRecord>,
    pub bounds: Option<(DVec3, DVec3)>,
    pub warnings: Vec<serde_json::Value>,
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
            let mut triangle_offset = 0_usize;
            for face in &mesh.faces {
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
                });
                let material_id = usize::try_from(face.material_index)
                    .ok()
                    .and_then(|index| node.materials.get(index));
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
                let face_uv = read_face_uvs(mesh, face.id);
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
                    let mut uv = [[0.0; 2]; 3];
                    for (index, vertex_id) in triangle.iter().enumerate() {
                        let local = vertex_positions.get(vertex_id).copied().ok_or_else(|| {
                            PotError::new(
                                ErrorCode::SceneInvalid,
                                "triangle references a missing vertex",
                            )
                        })?;
                        positions[index] = matrix.transform_point3(local);
                        uv[index] = face
                            .vertices
                            .iter()
                            .position(|face_vertex| face_vertex == vertex_id)
                            .and_then(|corner| {
                                face_uv.as_ref().and_then(|values| values.get(corner))
                            })
                            .copied()
                            .unwrap_or([0.0; 2]);
                    }
                    let raw_normal =
                        (positions[1] - positions[0]).cross(positions[2] - positions[0]);
                    if raw_normal.length_squared() <= f64::EPSILON || !raw_normal.is_finite() {
                        continue;
                    }
                    let normal = raw_normal.normalize();
                    let center = (positions[0] + positions[1] + positions[2]) / 3.0;
                    let color = if path_traced && matches!(mode, Mode::Beauty) {
                        base_color
                    } else {
                        shade(
                            doc,
                            snapshot,
                            node,
                            center,
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
                        uv_available: face_uv.is_some(),
                    };
                    if let Some(material) = material.filter(|material| {
                        material.displacement_method == "displacement"
                            && material.node_tree.is_some()
                    }) {
                        triangles.extend(subdivide_displaced_triangle(
                            doc, material, matrix, triangle, images,
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
    };
    if !path_traced && matches!(mode, Mode::Beauty) && !geometry.triangles.is_empty() {
        geometry.triangles = subdivide_realtime_triangles(&geometry.triangles)?;
        let shadow_maps = build_point_shadow_maps(doc, snapshot, &geometry.triangles)?;
        let bvh = Bvh::build(&geometry.triangles)?;
        for index in 0..geometry.triangles.len() {
            let triangle = geometry.triangles[index];
            let material = geometry
                .materials
                .get(triangle.material_index)
                .unwrap_or(&geometry.materials[0]);
            let object = geometry
                .objects
                .iter()
                .find(|object| object.index == triangle.object_index)
                .and_then(|object| doc.nodes.get(&object.node_id))
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::EvaluationFailed,
                        "render triangle references a missing scene object",
                    )
                })?;
            let normal = (triangle.positions[1] - triangle.positions[0])
                .cross(triangle.positions[2] - triangle.positions[0])
                .normalize_or_zero();
            let center =
                (triangle.positions[0] + triangle.positions[1] + triangle.positions[2]) / 3.0;
            let color = shade(
                doc,
                snapshot,
                object,
                center,
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
    for triangle in triangles.iter().copied() {
        for half in subdivide_triangle_once(triangle) {
            refined.extend(subdivide_triangle_once(half));
        }
    }
    Ok(refined)
}

fn subdivide_triangle_once(triangle: Triangle) -> [Triangle; 4] {
    let positions = [
        triangle.positions[0],
        triangle.positions[1],
        triangle.positions[2],
        (triangle.positions[0] + triangle.positions[1]) * 0.5,
        (triangle.positions[1] + triangle.positions[2]) * 0.5,
        (triangle.positions[2] + triangle.positions[0]) * 0.5,
    ];
    let uv = [
        triangle.uv[0],
        triangle.uv[1],
        triangle.uv[2],
        [
            f64::midpoint(triangle.uv[0][0], triangle.uv[1][0]),
            f64::midpoint(triangle.uv[0][1], triangle.uv[1][1]),
        ],
        [
            f64::midpoint(triangle.uv[1][0], triangle.uv[2][0]),
            f64::midpoint(triangle.uv[1][1], triangle.uv[2][1]),
        ],
        [
            f64::midpoint(triangle.uv[2][0], triangle.uv[0][0]),
            f64::midpoint(triangle.uv[2][1], triangle.uv[0][1]),
        ],
    ];
    let subtriangles = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]];
    let mut result = [triangle; 4];
    for (subtriangle, corners) in subtriangles.into_iter().enumerate() {
        for (corner, vertex) in corners.into_iter().enumerate() {
            result[subtriangle].positions[corner] = positions[vertex];
            result[subtriangle].uv[corner] = uv[vertex];
        }
    }
    result
}

fn subdivide_displaced_triangle(
    doc: &SceneDoc,
    material: &Material,
    object_to_world: DMat4,
    triangle: Triangle,
    images: &BTreeMap<String, ImageData>,
) -> Result<[Triangle; 4]> {
    let normal = (triangle.positions[1] - triangle.positions[0])
        .cross(triangle.positions[2] - triangle.positions[0])
        .normalize_or_zero();
    let uv1 = [
        triangle.uv[1][0] - triangle.uv[0][0],
        triangle.uv[1][1] - triangle.uv[0][1],
    ];
    let uv2 = [
        triangle.uv[2][0] - triangle.uv[0][0],
        triangle.uv[2][1] - triangle.uv[0][1],
    ];
    let determinant = uv1[0] * uv2[1] - uv1[1] * uv2[0];
    let (tangent, bitangent) = if determinant.abs() > f64::EPSILON {
        (
            ((triangle.positions[1] - triangle.positions[0]) * uv2[1]
                - (triangle.positions[2] - triangle.positions[0]) * uv1[1])
                / determinant,
            ((triangle.positions[2] - triangle.positions[0]) * uv1[0]
                - (triangle.positions[1] - triangle.positions[0]) * uv2[0])
                / determinant,
        )
    } else {
        let tangent = if normal.z.abs() < 0.999 {
            normal.cross(DVec3::Z)
        } else {
            normal.cross(DVec3::Y)
        };
        (tangent, normal.cross(tangent))
    };
    let tangent = tangent.normalize_or_zero();
    let bitangent = bitangent.normalize_or_zero();
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
        let uv = [
            triangle.uv[0][0] * weights[0]
                + triangle.uv[1][0] * weights[1]
                + triangle.uv[2][0] * weights[2],
            triangle.uv[0][1] * weights[0]
                + triangle.uv[1][1] * weights[1]
                + triangle.uv[2][1] * weights[2],
        ];
        let object_point = if inverse_is_finite {
            world_to_object.transform_point3(point)
        } else {
            point
        };
        let mut context = HitContext::new(&doc.node_groups)
            .with_images(images)
            .with_tangent_frame(tangent, bitangent);
        context.uv = uv;
        context.position = point;
        context.object = object_point;
        context.generated = object_point;
        context.normal = normal;
        let displacement = evaluate_displacement(material, &context)?;
        positions[index] = point + normal * displacement;
        uvs[index] = uv;
    }
    let indices = [[0, 1, 2], [1, 3, 4], [2, 4, 5], [1, 4, 2]];
    let mut result = [triangle; 4];
    for (subtriangle, corners) in indices.iter().enumerate() {
        for (corner, vertex) in corners.iter().copied().enumerate() {
            result[subtriangle].positions[corner] = positions[vertex];
            result[subtriangle].uv[corner] = uvs[vertex];
        }
    }
    Ok(result)
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

fn read_face_uvs(mesh: &crate::geom::Mesh, face_id: u32) -> Option<Vec<[f64; 2]>> {
    mesh.attributes
        .get("uv_map")?
        .as_array()?
        .iter()
        .find(|entry| {
            entry.get("face_id").and_then(serde_json::Value::as_u64) == Some(u64::from(face_id))
        })?
        .get("uv")?
        .as_array()?
        .iter()
        .map(|coordinate| Some([coordinate.get(0)?.as_f64()?, coordinate.get(1)?.as_f64()?]))
        .collect()
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
    _node: &crate::model::Node,
    point: DVec3,
    normal: DVec3,
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
                normal,
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
    normal: DVec3,
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
                point,
                normal,
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
                point,
                normal,
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
                point,
                normal,
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
                    point,
                    normal,
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
