// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/shrinkwrap.cc, source/blender/blenkernel/intern/bvhutils.cc,
//   source/blender/blenlib/intern/math_geom.cc, source/blender/blenlib/intern/math_solvers.cc,
//   SPDX-FileCopyrightText: Blender Authors, GPL-2.0-or-later.

//! Shrinkwrap in Blender 5.2.2's single-precision arithmetic. Nearest-surface, target-projection,
//! nearest-vertex, and normal-projection modes query the target's corner-triangle `KdopTree`;
//! mesh positions are converted to `f32` at the boundary and written back as `f64`.

use std::collections::HashMap;

use glam::{DMat4, DVec3, Vec3};
use serde_json::Value;

use crate::{
    error::{PotError, Result},
    geom::{
        Mesh,
        blender_matrix::blender_invert_m4,
        bvh::{KdopTree, NearestHit, Ray, RayHit},
        normals,
    },
    model::Modifier,
};

use super::{bool_param, invalid_parameter, number_param, string_param, uint_param, validated};

/// Blender's `BLI_bvhtree_new` epsilon floor: every BVH leaf box is inflated by this.
const TREE_EPSILON: f32 = f32::EPSILON;

/// Blender's `BVH_RAYCAST_DIST_MAX`, the initial ray length bound.
const RAYCAST_DIST_MAX: f32 = f32::MAX / 2.0;

/// Blender's `float[4][4]`, indexed `[column][row]`.
type Mat = [[f32; 4]; 4];

#[derive(Clone, Copy, PartialEq)]
enum WrapMode {
    OnSurface,
    Inside,
    Outside,
    OutsideSurface,
    AboveSurface,
}

#[derive(Clone, Copy, PartialEq)]
enum Cull {
    Front,
    Back,
}

impl Cull {
    fn flipped(self) -> Self {
        match self {
            Self::Front => Self::Back,
            Self::Back => Self::Front,
        }
    }
}

/// Blender's `SpaceTransform`: subject-local to target-local and back.
#[derive(Clone, Copy)]
struct Space {
    subject_to_target: Mat,
    target_to_subject: Mat,
}

#[derive(Clone, Copy)]
struct Tri {
    /// Vertex indices, in mesh-vertex order.
    vertices: [usize; 3],
    /// Flattened corner indices, for corner normals.
    corners: [usize; 3],
    face: usize,
    /// Boundary-edge index for each triangle edge that is a boundary edge of the mesh.
    boundary: [Option<usize>; 3],
}

#[derive(Clone, Copy)]
struct BoundaryEdge {
    positions: [Vec3; 2],
    vertex_normals: [Vec3; 2],
    /// Boundary plane normals, sign-matched to the edge direction (`target_project_edge`).
    normal_directions: [Vec3; 2],
}

/// Blender's `ShrinkwrapTreeData` for one target: its corner-triangle BVH and per-element data,
/// or a vertex BVH for `NEAREST_VERTEX`, which leaves the triangle data empty.
struct Surface {
    tree: KdopTree,
    positions: Vec<Vec3>,
    vertex_normals: Vec<Vec3>,
    corner_normals: Vec<Vec3>,
    face_normals: Vec<Vec3>,
    /// Blender's `sharp_faces` lookup: `None` when the attribute is absent (every face smooth).
    sharp_faces: Option<Vec<bool>>,
    tris: Vec<Tri>,
    boundary: Vec<BoundaryEdge>,
}

impl Surface {
    fn triangle(&self, index: usize) -> [Vec3; 3] {
        self.tris[index]
            .vertices
            .map(|vertex| self.positions[vertex])
    }
}

/// A projection or snap hit in subject space; `index` is the corner triangle.
#[derive(Clone, Copy)]
struct Hit {
    index: usize,
    co: Vec3,
    normal: Vec3,
}

#[derive(Clone, Copy)]
struct Target<'a> {
    surface: &'a Surface,
    space: Space,
}

struct Projection<'a> {
    main: Target<'a>,
    auxiliary: Option<Target<'a>>,
    positive: bool,
    negative: bool,
    cull: Option<Cull>,
    invert_cull: bool,
    limit_squared: Option<f32>,
    mode: WrapMode,
    offset: f32,
}

enum Source {
    /// Blender's normalized projection axis; each vertex starts at its own position.
    Axis(Vec3),
    /// `MOD_SHRINKWRAP_PROJECT_OVER_NORMAL`: origins are the (subdivided) source positions.
    Normal {
        origins: Vec<Vec3>,
        directions: Vec<Vec3>,
    },
}

/// `use_apply_on_spline` (`eModifierMode_ApplyOnSpline`) only chooses curve control points over the
/// filled curve, so it has no effect on a mesh Shrinkwrap; Blender ignores it there too.
pub(super) fn evaluate(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let method = string_param(modifier, "wrap_method")?.to_ascii_uppercase();
    let Some(target) = modifier.runtime.shrinkwrap_target_mesh.as_ref() else {
        if modifier
            .params
            .get("target")
            .is_some_and(|value| !value.is_null())
        {
            return Err(invalid_parameter(
                modifier,
                "target",
                "a resolved mesh object operand",
            ));
        }
        return Ok(mesh.clone());
    };
    if modifier
        .params
        .get("auxiliary_target")
        .is_some_and(|value| !value.is_null())
        && modifier.runtime.shrinkwrap_auxiliary_target_mesh.is_none()
    {
        return Err(invalid_parameter(
            modifier,
            "auxiliary_target",
            "a resolved mesh object operand",
        ));
    }
    validated(target)?;
    if target.vertices.is_empty() || mesh.vertices.is_empty() {
        return Ok(mesh.clone());
    }

    let space = space_from_runtime(
        modifier.runtime.shrinkwrap_subject_to_target,
        modifier,
        "target",
    )?;
    let offset = number_param(modifier, "offset")? as f32;
    let positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect();
    let group = vertex_group(mesh, modifier)?;
    let invert_group = bool_param(modifier, "invert_vertex_group")?;
    let weights: Vec<f32> = mesh
        .vertices
        .iter()
        .map(|vertex| group_weight(group.as_ref(), vertex.id, invert_group))
        .collect();

    let updates = match method.as_str() {
        "NEAREST_VERTEX" => {
            let surface = build_vertex_surface(target);
            nearest_vertex_points(&positions, &weights, &surface, space, offset)
        }
        "NEAREST_SURFACEPOINT" | "TARGET_PROJECT" => {
            if target.faces.is_empty() {
                return Ok(mesh.clone());
            }
            let surface = build_surface(target)?;
            nearest_surface_points(
                &positions,
                &weights,
                &surface,
                space,
                wrap_mode(modifier)?,
                offset,
                method == "TARGET_PROJECT",
            )
        }
        "PROJECT" => {
            if target.faces.is_empty() {
                return Ok(mesh.clone());
            }
            let positive = bool_param(modifier, "use_positive_direction")?;
            let negative = bool_param(modifier, "use_negative_direction")?;
            let axis_flag = |name: &str| -> Result<f32> {
                Ok(if bool_param(modifier, name)? {
                    1.0
                } else {
                    0.0
                })
            };
            let axis = Vec3::new(
                axis_flag("use_project_x")?,
                axis_flag("use_project_y")?,
                axis_flag("use_project_z")?,
            );
            let subsurf_levels = uint_param(modifier, "subsurf_levels")?;
            if subsurf_levels > 6 {
                return Err(invalid_parameter(
                    modifier,
                    "subsurf_levels",
                    "an integer between 0 and 6",
                ));
            }
            if !positive && !negative {
                return Ok(mesh.clone());
            }
            let source = if axis == Vec3::ZERO {
                let points: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
                let origins = if subsurf_levels > 0 {
                    source_subdivision_positions(mesh, subsurf_levels)?
                } else {
                    positions.clone()
                };
                Source::Normal {
                    origins,
                    directions: to_vec3s(normals::geometric_vertex_normals(mesh, &points)?),
                }
            } else {
                Source::Axis(normalize(axis))
            };
            let cull = match string_param(modifier, "cull_face")?
                .to_ascii_uppercase()
                .as_str()
            {
                "OFF" => None,
                "FRONT" => Some(Cull::Front),
                "BACK" => Some(Cull::Back),
                _ => {
                    return Err(invalid_parameter(
                        modifier,
                        "cull_face",
                        "OFF, FRONT, or BACK",
                    ));
                }
            };
            let limit = number_param(modifier, "project_limit")? as f32;
            let limit_squared = limit * limit;
            let main = build_surface(target)?;
            let auxiliary =
                auxiliary_surface(modifier.runtime.shrinkwrap_auxiliary_target_mesh.as_ref())?;
            let auxiliary_space = space_from_runtime(
                modifier.runtime.shrinkwrap_auxiliary_subject_to_target,
                modifier,
                "auxiliary_target",
            )?;
            let projection = Projection {
                main: Target {
                    surface: &main,
                    space,
                },
                auxiliary: auxiliary.as_ref().map(|surface| Target {
                    surface,
                    space: auxiliary_space,
                }),
                positive,
                negative,
                cull,
                invert_cull: bool_param(modifier, "use_invert_cull")?,
                limit_squared: (limit_squared != 0.0).then_some(limit_squared),
                mode: wrap_mode(modifier)?,
                offset,
            };
            project_points(&projection, &positions, &weights, &source)
        }
        _ => {
            return Err(invalid_parameter(
                modifier,
                "wrap_method",
                "NEAREST_SURFACEPOINT, NEAREST_VERTEX, PROJECT, or TARGET_PROJECT",
            ));
        }
    };

    let mut result = mesh.clone();
    for (vertex, update) in result.vertices.iter_mut().zip(updates) {
        if let Some(co) = update {
            vertex.co = co.as_dvec3();
        }
    }
    validated(&result)?;
    Ok(result)
}

/// Blender's `BLI_space_transform_from_matrices` in float: `subject_to_target` is the f32
/// `local2target` the caller builds as `invert(target) * subject`, and `target_to_subject` is its
/// `invert_m4` (`blender_invert_m4`), as `target2local` is in Blender.
fn space_from_runtime(
    matrix: Option<[f64; 16]>,
    modifier: &Modifier,
    parameter: &str,
) -> Result<Space> {
    let subject_to_target =
        to_mat(matrix.map_or(DMat4::IDENTITY, |matrix| DMat4::from_cols_array(&matrix)));
    let finite = |matrix: &Mat| matrix.iter().flatten().all(|value| value.is_finite());
    match blender_invert_m4(&subject_to_target) {
        Some(target_to_subject) if finite(&subject_to_target) && finite(&target_to_subject) => {
            Ok(Space {
                subject_to_target,
                target_to_subject,
            })
        }
        _ => Err(invalid_parameter(
            modifier,
            parameter,
            "a finite, invertible object transform",
        )),
    }
}

fn wrap_mode(modifier: &Modifier) -> Result<WrapMode> {
    match string_param(modifier, "wrap_mode")?
        .to_ascii_uppercase()
        .as_str()
    {
        "ON_SURFACE" => Ok(WrapMode::OnSurface),
        "INSIDE" => Ok(WrapMode::Inside),
        "OUTSIDE" => Ok(WrapMode::Outside),
        "OUTSIDE_SURFACE" => Ok(WrapMode::OutsideSurface),
        "ABOVE_SURFACE" => Ok(WrapMode::AboveSurface),
        _ => Err(invalid_parameter(
            modifier,
            "wrap_mode",
            "ON_SURFACE, INSIDE, OUTSIDE, OUTSIDE_SURFACE, or ABOVE_SURFACE",
        )),
    }
}

fn vertex_group(mesh: &Mesh, modifier: &Modifier) -> Result<Option<HashMap<u32, f64>>> {
    let name = string_param(modifier, "vertex_group")?;
    let Some(group) = (!name.is_empty())
        .then(|| {
            mesh.attributes
                .get("vertex_groups")
                .and_then(Value::as_object)
                .and_then(|groups| groups.get(name))
                .and_then(Value::as_object)
        })
        .flatten()
    else {
        return Ok(None);
    };
    let weights = group
        .iter()
        .filter_map(|(key, value)| Some((key.strip_prefix('v')?.parse().ok()?, value.as_f64()?)))
        .collect();
    Ok(Some(weights))
}

/// Blender's `BKE_defvert_array_find_weight_safe`: no group is full weight; an inverted weight is
/// `1 - w`.
fn group_weight(group: Option<&HashMap<u32, f64>>, vertex_id: u32, invert: bool) -> f32 {
    let Some(group) = group else {
        return 1.0;
    };
    let weight = group.get(&vertex_id).map_or(0.0, |weight| *weight as f32);
    if invert { 1.0 - weight } else { weight }
}

/// Limit-surface positions of the source vertices at `levels` subdivisions, for
/// `OVER_NORMAL` projection (Blender's `shrinkwrap_calc_subdivided_positions`).
fn source_subdivision_positions(mesh: &Mesh, levels: u64) -> Result<Vec<Vec3>> {
    super::shrinkwrap_subdivision_positions(mesh, levels)
        .map(|positions| positions.into_iter().map(DVec3::as_vec3).collect())
}

fn build_vertex_surface(mesh: &Mesh) -> Surface {
    let positions: Vec<Vec3> = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect();
    // Blender inflates every leaf box by the tree epsilon, so a vertex leaf is its inflated box.
    let bounds: Vec<[f32; 6]> = positions
        .iter()
        .map(|point| {
            let [x, y, z] = point.to_array();
            [
                x - TREE_EPSILON,
                x + TREE_EPSILON,
                y - TREE_EPSILON,
                y + TREE_EPSILON,
                z - TREE_EPSILON,
                z + TREE_EPSILON,
            ]
        })
        .collect();
    Surface {
        tree: KdopTree::from_leaf_bounds(&bounds),
        positions,
        vertex_normals: Vec::new(),
        corner_normals: Vec::new(),
        face_normals: Vec::new(),
        sharp_faces: None,
        tris: Vec::new(),
        boundary: Vec::new(),
    }
}

/// Blender's `BKE_shrinkwrap_init_tree` for the corner-triangle tree and its per-element data.
fn build_surface(mesh: &Mesh) -> Result<Surface> {
    let points: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    let positions: Vec<Vec3> = points.iter().map(|point| point.as_vec3()).collect();
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let vertex_normals = to_vec3s(normals::geometric_vertex_normals(mesh, &points)?);
    let corner_normals = to_vec3s(normals::corner_normals(mesh, &points)?);
    let face_normals: Vec<Vec3> = mesh
        .faces
        .iter()
        .map(|face| {
            let face_points: Vec<DVec3> = face
                .vertices
                .iter()
                .map(|id| points[index_by_id[id]])
                .collect();
            normals::face_normal(&face_points).as_vec3()
        })
        .collect();
    let smooth = normals::smooth_face_flags(mesh)?;
    let sharp_faces: Option<Vec<bool>> =
        (!smooth.iter().all(|flag| *flag)).then(|| smooth.iter().map(|flag| !flag).collect());

    let (boundary, boundary_lookup) =
        boundary_edges(mesh, &index_by_id, &positions, &vertex_normals);
    let tessellation = mesh.tessellate().map_err(|error| {
        PotError::invalid_argument(format!("invalid target triangulation: {error}"))
    })?;
    let expected: usize = mesh
        .faces
        .iter()
        .map(|face| face.vertices.len().saturating_sub(2))
        .sum();
    if tessellation.len() != expected {
        return Err(PotError::invalid_argument(
            "target tessellation does not match its faces",
        ));
    }
    let mut tri_index = 0;
    let mut tris = Vec::with_capacity(tessellation.len());
    let mut corner_offset = 0;
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for _ in 0..face.vertices.len().saturating_sub(2) {
            let ids = *tessellation
                .get(tri_index)
                .ok_or_else(|| PotError::invalid_argument("target tessellation ended early"))?;
            tri_index += 1;
            let mut vertices = [0; 3];
            let mut corners = [0; 3];
            for (slot, id) in ids.into_iter().enumerate() {
                vertices[slot] = *index_by_id.get(&id).ok_or_else(|| {
                    PotError::invalid_argument("target face references a missing vertex")
                })?;
                corners[slot] = corner_offset
                    + face
                        .vertices
                        .iter()
                        .position(|face_id| *face_id == id)
                        .ok_or_else(|| {
                            PotError::invalid_argument("target triangle corner is missing")
                        })?;
            }
            tris.push(Tri {
                vertices,
                corners,
                face: face_index,
                boundary: [
                    edge_lookup(&boundary_lookup, vertices[0], vertices[1]),
                    edge_lookup(&boundary_lookup, vertices[1], vertices[2]),
                    edge_lookup(&boundary_lookup, vertices[2], vertices[0]),
                ],
            });
        }
        corner_offset += face.vertices.len();
    }
    let bounds: Vec<[f32; 6]> = tris
        .iter()
        .map(|tri| {
            KdopTree::leaf_bounds_for_triangle(
                tri.vertices.map(|vertex| positions[vertex].to_array()),
            )
        })
        .collect();
    Ok(Surface {
        tree: KdopTree::from_leaf_bounds(&bounds),
        positions,
        vertex_normals,
        corner_normals,
        face_normals,
        sharp_faces,
        tris,
        boundary,
    })
}

/// Blender's `shrinkwrap_build_boundary_data`: boundary edges (used by exactly one face), their
/// vertex-averaged boundary plane normals, and the edge lookup for triangle edges.
fn boundary_edges(
    mesh: &Mesh,
    index_by_id: &HashMap<u32, usize>,
    positions: &[Vec3],
    vertex_normals: &[Vec3],
) -> (Vec<BoundaryEdge>, HashMap<(usize, usize), usize>) {
    let mut use_counts = HashMap::<(usize, usize), usize>::new();
    for face in &mesh.faces {
        for corner in 0..face.vertices.len() {
            let first = index_by_id[&face.vertices[corner]];
            let second = index_by_id[&face.vertices[(corner + 1) % face.vertices.len()]];
            *use_counts
                .entry(edge_key_indices(first, second))
                .or_default() += 1;
        }
    }
    let mut directions = vec![Vec3::ZERO; positions.len()];
    let mut status = vec![0_i8; positions.len()];
    let mut boundary_pairs = Vec::new();
    for edge in &mesh.edges {
        let oriented = [
            index_by_id[&edge.vertices[0]],
            index_by_id[&edge.vertices[1]],
        ];
        let key = edge_key_indices(oriented[0], oriented[1]);
        if use_counts.get(&key) != Some(&1) {
            continue;
        }
        let direction = normalize(positions[oriented[1]] - positions[oriented[0]]);
        merge_boundary_direction(&mut directions, &mut status, oriented[0], direction, 1);
        merge_boundary_direction(&mut directions, &mut status, oriented[1], direction, 2);
        boundary_pairs.push(oriented);
    }
    for direction in &mut directions {
        *direction = normalize(*direction);
    }
    let plane_normals: Vec<Vec3> = (0..positions.len())
        .map(|vertex| {
            let normal = vertex_normals[vertex];
            normalize(normal.cross(directions[vertex]).cross(normal))
        })
        .collect();
    let mut lookup = HashMap::with_capacity(boundary_pairs.len());
    let mut edges = Vec::with_capacity(boundary_pairs.len());
    for [first, second] in boundary_pairs {
        let edge_direction = positions[second] - positions[first];
        let first_plane = if directions[first].dot(edge_direction) < 0.0 {
            -plane_normals[first]
        } else {
            plane_normals[first]
        };
        let second_plane = if directions[second].dot(edge_direction) < 0.0 {
            -plane_normals[second]
        } else {
            plane_normals[second]
        };
        lookup.insert(edge_key_indices(first, second), edges.len());
        edges.push(BoundaryEdge {
            positions: [positions[first], positions[second]],
            vertex_normals: [vertex_normals[first], vertex_normals[second]],
            normal_directions: [first_plane, second_plane],
        });
    }
    (edges, lookup)
}

/// Blender's `merge_vert_dir`.
fn merge_boundary_direction(
    directions: &mut [Vec3],
    status: &mut [i8],
    vertex: usize,
    edge_direction: Vec3,
    side: i8,
) {
    if (status[vertex] >= 0 && status[vertex] == side)
        || (status[vertex] < 0 && directions[vertex].dot(edge_direction) < 0.0)
    {
        directions[vertex] -= edge_direction;
    } else {
        directions[vertex] += edge_direction;
    }
    status[vertex] = if status[vertex] == 0 { side } else { -1 };
}

fn edge_key_indices(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

fn edge_lookup(
    edges: &HashMap<(usize, usize), usize>,
    first: usize,
    second: usize,
) -> Option<usize> {
    edges.get(&edge_key_indices(first, second)).copied()
}

fn auxiliary_surface(target: Option<&Mesh>) -> Result<Option<Surface>> {
    let Some(target) = target else {
        return Ok(None);
    };
    validated(target)?;
    if target.vertices.is_empty() || target.faces.is_empty() {
        return Ok(None);
    }
    build_surface(target).map(Some)
}

/// Blender's `shrinkwrap_calc_nearest_vertex`.
fn nearest_vertex_points(
    positions: &[Vec3],
    weights: &[f32],
    surface: &Surface,
    space: Space,
    offset: f32,
) -> Vec<Option<Vec3>> {
    let mut nearest = NearestHit {
        index: -1,
        co: [0.0; 3],
        dist_sq: f32::MAX,
    };
    positions
        .iter()
        .zip(weights)
        .map(|(&co, &weight)| {
            if weight == 0.0 {
                return None;
            }
            let target = transform_point(&space.subject_to_target, co);
            nearest.dist_sq = if nearest.index == -1 {
                f32::MAX
            } else {
                (target - Vec3::from(nearest.co)).length_squared()
            };
            surface.tree.find_nearest(
                target.to_array(),
                &mut nearest,
                &mut |index: usize, point: [f32; 3], hit: &mut NearestHit| {
                    // Blender's leaf has no callback: the leaf's inflated box gives the nearest point.
                    let vertex = surface.positions[index];
                    let query = Vec3::from(point);
                    let epsilon = Vec3::splat(TREE_EPSILON);
                    let nearest_point = query.clamp(vertex - epsilon, vertex + epsilon);
                    let dist_sq = (query - nearest_point).length_squared();
                    if dist_sq < hit.dist_sq {
                        hit.index = index as isize;
                        hit.dist_sq = dist_sq;
                        hit.co = nearest_point.to_array();
                    }
                },
                false,
            );
            if nearest.index < 0 {
                return None;
            }
            let mut weight = weight;
            if nearest.dist_sq > f32::EPSILON {
                let distance = nearest.dist_sq.sqrt();
                weight *= (distance - offset) / distance;
            }
            let snapped = transform_point(&space.target_to_subject, Vec3::from(nearest.co));
            Some(interp(co, snapped, weight))
        })
        .collect()
}

/// Blender's `shrinkwrap_calc_nearest_surface_point`: nearest surface point, or the target
/// projection for `TARGET_PROJECT`, snapped in target space and mapped back.
fn nearest_surface_points(
    positions: &[Vec3],
    weights: &[f32],
    surface: &Surface,
    space: Space,
    mode: WrapMode,
    offset: f32,
    target_project: bool,
) -> Vec<Option<Vec3>> {
    let mut nearest = NearestHit {
        index: -1,
        co: [0.0; 3],
        dist_sq: f32::MAX,
    };
    let mut normal = Vec3::ZERO;
    positions
        .iter()
        .zip(weights)
        .map(|(&co, &weight)| {
            if weight == 0.0 {
                return None;
            }
            let target = transform_point(&space.subject_to_target, co);
            if target_project || nearest.index == -1 {
                nearest.index = -1;
                nearest.dist_sq = f32::MAX;
            } else {
                nearest.dist_sq = (target - Vec3::from(nearest.co)).length_squared();
            }
            find_nearest_surface(surface, target_project, target, &mut nearest, &mut normal);
            if nearest.index < 0 {
                return None;
            }
            let hit = Hit {
                index: nearest.index as usize,
                co: Vec3::from(nearest.co),
                normal,
            };
            let snapped = snap_point(surface, None, mode, hit, offset, target);
            Some(interp(
                co,
                transform_point(&space.target_to_subject, snapped),
                weight,
            ))
        })
        .collect()
}

/// Blender's `BKE_shrinkwrap_find_nearest_surface`.
fn find_nearest_surface(
    surface: &Surface,
    target_project: bool,
    co: Vec3,
    nearest: &mut NearestHit,
    normal: &mut Vec3,
) {
    if target_project {
        surface.tree.find_nearest(
            co.to_array(),
            nearest,
            &mut |index: usize, point: [f32; 3], hit: &mut NearestHit| {
                target_project_triangle(surface, index, Vec3::from(point), hit, normal);
            },
            true,
        );
        if nearest.index >= 0 {
            return;
        }
    }
    surface.tree.find_nearest(
        co.to_array(),
        nearest,
        &mut |index: usize, point: [f32; 3], hit: &mut NearestHit| {
            nearest_triangle(surface, index, Vec3::from(point), hit, normal);
        },
        false,
    );
}

/// Blender's `mesh_corner_tris_nearest_point`.
fn nearest_triangle(
    surface: &Surface,
    index: usize,
    co: Vec3,
    hit: &mut NearestHit,
    normal: &mut Vec3,
) {
    let [v1, v2, v3] = surface.triangle(index);
    let closest = closest_on_triangle(co, v1, v2, v3);
    let dist_sq = (co - closest).length_squared();
    if dist_sq < hit.dist_sq {
        hit.index = index as isize;
        hit.dist_sq = dist_sq;
        hit.co = closest.to_array();
        *normal = normal_tri(v1, v2, v3);
    }
}

/// Blender's `mesh_corner_tris_target_project`, with its boundary-edge fallback.
fn target_project_triangle(
    surface: &Surface,
    index: usize,
    co: Vec3,
    hit: &mut NearestHit,
    normal: &mut Vec3,
) {
    let tri = surface.tris[index];
    let positions = surface.triangle(index);
    let closest = closest_on_triangle(co, positions[0], positions[1], positions[2]);
    let dist_sq = (co - closest).length_squared();
    if dist_sq >= hit.dist_sq {
        return;
    }
    let normals = tri.vertices.map(|vertex| surface.vertex_normals[vertex]);
    if let Some((hit_co, hit_no)) = project_on_triangle(co, positions, normals, closest, dist_sq) {
        update_hit(hit, normal, index, co, hit_co, hit_no);
    } else {
        for edge in tri.boundary.into_iter().flatten() {
            target_project_edge(&surface.boundary[edge], co, index, hit, normal);
        }
    }
}

/// Blender's `update_hit`: keeps the candidate only if it is strictly nearer than `hit`.
fn update_hit(
    hit: &mut NearestHit,
    normal: &mut Vec3,
    index: usize,
    co: Vec3,
    hit_co: Vec3,
    hit_no: Vec3,
) {
    let dist_sq = (hit_co - co).length_squared();
    if dist_sq < hit.dist_sq {
        hit.index = index as isize;
        hit.dist_sq = dist_sq;
        hit.co = hit_co.to_array();
        *normal = normalize(hit_no);
    }
}

/// Blender's `target_project_edge`: projects onto a non-manifold boundary edge treated as an
/// infinitely thin cylinder.
fn target_project_edge(
    edge: &BoundaryEdge,
    co: Vec3,
    index: usize,
    hit: &mut NearestHit,
    normal: &mut Vec3,
) {
    let [p0, p1] = edge.positions;
    let [d0, d1] = edge.normal_directions;
    let d0v0 = d0.dot(p0);
    let d0v1 = d0.dot(p1);
    let d1v0 = d1.dot(p0);
    let d1v1 = d1.dot(p1);
    let d0co = d0.dot(co);
    let a = d0v1 - d0v0 + d1v0 - d1v1;
    if a == 0.0 {
        return;
    }
    let b = 2.0 * d0v0 - d0v1 - d0co - d1v0 + d1.dot(co);
    let c = d0co - d0v0;
    let determinant = b * b - 4.0 * a * c;
    if determinant < 0.0 {
        return;
    }
    let root = determinant.sqrt();
    let signs: &[f32] = if determinant > 0.0 {
        &[1.0, -1.0]
    } else {
        &[-1.0]
    };
    for &sign in signs {
        let mut x = (-b + sign * root) / (2.0 * a);
        if (-1.0e-6_f32..=1.0 + 1.0e-6_f32).contains(&x) {
            x = x.clamp(0.0, 1.0);
            let hit_co = interp(p0, p1, x);
            let hit_no = interp(edge.vertex_normals[0], edge.vertex_normals[1], x);
            update_hit(hit, normal, index, co, hit_co, hit_no);
        }
    }
}

/// Blender's `target_project_solve_point_tri`: Newton's method on barycentric weights and the
/// normal distance. Returns the solved surface point and its interpolated normal.
fn project_on_triangle(
    point: Vec3,
    positions: [Vec3; 3],
    normals: [Vec3; 3],
    closest: Vec3,
    distance_squared: f32,
) -> Option<(Vec3, Vec3)> {
    let distance = distance_squared.sqrt();
    let magnitude = distance
        + len_manhattan(positions[0])
        + len_manhattan(positions[1])
        + len_manhattan(positions[2]);
    let epsilon = magnitude * 1.0e-6;
    let weights = tri_weights(positions[0], positions[1], positions[2], closest);
    let signed = if (point - closest).dot(interp3(normals, weights)) < 0.0 {
        -distance
    } else {
        distance
    };
    let mut x = [weights[0], weights[1], signed];
    clamp_barycentric(&mut x);

    let [p0, p1, p2] = positions;
    let [n0, n1, n2] = normals;
    let (c02, c12) = (p0 - p2, p1 - p2);
    let (n02, n12) = (n0 - n2, n1 - n2);
    let solved = newton_solve(
        x,
        epsilon,
        |x| {
            let w = [x[0], x[1], 1.0 - x[0] - x[1]];
            interp3(positions, w) + interp3(normals, w) * x[2] - point
        },
        |x| {
            [
                c02 + n02 * x[2],
                c12 + n12 * x[2],
                (n2 + n02 * x[0]) + n12 * x[1],
            ]
        },
        tri_correct,
    )?;
    let w = [solved[0], solved[1], 1.0 - solved[0] - solved[1]];
    Some((interp3(positions, w), interp3(normals, w)))
}

/// Blender's `target_project_tri_correct`: keeps a Newton step inside the triangle's barycentric
/// domain. Returns `false` when the step clearly points outside it.
fn tri_correct(x: [f32; 3], step: &mut Vec3, x_next: &mut [f32; 3]) -> bool {
    const EPSILON: f32 = 1.0e-5;
    const DIRECTION_EPSILON: f32 = 0.5;
    let sqrt_2 = std::f32::consts::SQRT_2;
    let mut fixed = false;
    let mut locked = false;
    let sum = x[0] + x[1];
    let sum_step = -(step.x + step.y);

    if sum + sum_step > 1.0 {
        let ldist = 1.0 - sum;
        if ldist < EPSILON * sqrt_2 {
            let step_length = (step.x * step.x + step.y * step.y).sqrt();
            if step_length > EPSILON && sum_step > step_length * DIRECTION_EPSILON * sqrt_2 {
                return false;
            }
            let correction = (sum + sum_step - 1.0) * 0.5;
            step.x += correction;
            step.y += correction;
            fixed = true;
            locked = true;
        } else {
            let scale = ldist / sum_step;
            *step *= scale;
            fixed = true;
        }
    }

    for index in 0..2 {
        if step[index] > x[index] {
            if x[index] < EPSILON {
                let step_length = (step.x * step.x + step.y * step.y).sqrt();
                if step_length > EPSILON
                    && (locked || step[index] > step_length * DIRECTION_EPSILON)
                {
                    return false;
                }
                step[index] = x[index];
                fixed = true;
            } else {
                let scale = x[index] / step[index];
                *step *= scale;
                fixed = true;
            }
        }
    }

    if fixed {
        *x_next = [x[0] - step.x, x[1] - step.y, x[2] - step.z];
        clamp_barycentric(x_next);
    }
    true
}

/// Blender's `BLI_newton3d_solve`. Returns `None` on a singular Jacobian, a rejected correction,
/// or non-convergence in `MAX_ITERATIONS`.
fn newton_solve(
    x_init: [f32; 3],
    epsilon: f32,
    deviation: impl Fn([f32; 3]) -> Vec3,
    jacobian: impl Fn([f32; 3]) -> [Vec3; 3],
    correct: impl Fn([f32; 3], &mut Vec3, &mut [f32; 3]) -> bool,
) -> Option<[f32; 3]> {
    const MAX_ITERATIONS: usize = 20;
    let epsilon = epsilon * epsilon;
    let mut x = x_init;
    let mut fdelta = deviation(x);
    let mut fdeltav = fdelta.length_squared();
    let mut iteration = 0;
    while iteration == 0 || (iteration < MAX_ITERATIONS && fdeltav > epsilon) {
        let inverse = invert_m3(jacobian(x).map(|column| column.to_array()))?;
        let mut step = mul_m3_v3(&inverse, fdelta);
        let mut x_next = [x[0] - step.x, x[1] - step.y, x[2] - step.z];
        if !correct(x, &mut step, &mut x_next) {
            return None;
        }
        fdelta = deviation(x_next);
        let mut next_fdeltav = fdelta.length_squared();
        while next_fdeltav > fdeltav && next_fdeltav > epsilon {
            let g0 = fdeltav.sqrt();
            let g1 = next_fdeltav.sqrt();
            let g01 = -g0 / step.length();
            let determinant = 2.0 * (g1 - g0 - g01);
            let mut scale = if determinant == 0.0 {
                0.1
            } else {
                -g01 / determinant
            };
            if scale < 0.1 {
                scale = 0.1;
            }
            step *= scale;
            x_next = [x[0] - step.x, x[1] - step.y, x[2] - step.z];
            fdelta = deviation(x_next);
            next_fdeltav = fdelta.length_squared();
        }
        x = x_next;
        fdeltav = next_fdeltav;
        iteration += 1;
    }
    (fdeltav <= epsilon).then_some(x)
}

/// Blender's `invert_m3`: adjugate over determinant, `None` when singular.
fn invert_m3(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = m;
    let determinant = m00 * (m11 * m22 - m12 * m21) - m10 * (m01 * m22 - m02 * m21)
        + m20 * (m01 * m12 - m02 * m11);
    if determinant == 0.0 {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    let adjugate = [
        [
            m11 * m22 - m12 * m21,
            -m01 * m22 + m02 * m21,
            m01 * m12 - m02 * m11,
        ],
        [
            -m10 * m22 + m12 * m20,
            m00 * m22 - m02 * m20,
            -m00 * m12 + m02 * m10,
        ],
        [
            m10 * m21 - m11 * m20,
            -m00 * m21 + m01 * m20,
            m00 * m11 - m01 * m10,
        ],
    ];
    Some(adjugate.map(|column| column.map(|value| value * inverse_determinant)))
}

/// Blender's `mul_v3_m3v3`.
fn mul_m3_v3(m: &[[f32; 3]; 3], v: Vec3) -> Vec3 {
    let [x, y, z] = v.to_array();
    Vec3::new(
        m[0][0] * x + m[1][0] * y + m[2][0] * z,
        m[0][1] * x + m[1][1] * y + m[2][1] * z,
        m[0][2] * x + m[1][2] * y + m[2][2] * z,
    )
}

/// Blender's `target_project_tri_clamp`: non-negative weights with `x + y <= 1`.
fn clamp_barycentric(x: &mut [f32; 3]) {
    x[0] = x[0].max(0.0);
    x[1] = x[1].max(0.0);
    if x[0] + x[1] > 1.0 {
        let sum = x[0] + x[1];
        x[0] /= sum;
        x[1] = 1.0 - x[0];
    }
}

/// Blender's `closest_on_tri_to_point_v3` (Ericson's region tests).
fn closest_on_triangle(point: Vec3, v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
    let ab = v2 - v1;
    let ac = v3 - v1;
    let ap = point - v1;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return v1;
    }
    let bp = point - v2;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return v2;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let ab_squared = d1 - d3;
        return if ab_squared == 0.0 {
            v1
        } else {
            v1 + ab * (d1 / ab_squared)
        };
    }
    let cp = point - v3;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return v3;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let ac_squared = d2 - d6;
        return if ac_squared == 0.0 {
            v1
        } else {
            v1 + ac * (d2 / ac_squared)
        };
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc_squared = (d4 - d3) + (d5 - d6);
        return if bc_squared == 0.0 {
            v2
        } else {
            v2 + (v3 - v2) * ((d4 - d3) / bc_squared)
        };
    }
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    v1 + ab * v + ac * w
}

/// Blender's `normal_tri_v3`, with its exact winding.
fn normal_tri(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3 {
    normalize((v1 - v2).cross(v2 - v3))
}

/// Blender's `interp_weights_tri_v3`: barycentric weights of `co` in the triangle.
fn tri_weights(v1: Vec3, v2: Vec3, v3: Vec3, co: Vec3) -> [f32; 3] {
    let normal = normal_tri(v1, v2, v3);
    let (x, y, z) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
    let (i, j): (usize, usize) = if z >= x && z >= y {
        (0, 1)
    } else if y >= x && y >= z {
        (0, 2)
    } else {
        (1, 2)
    };
    let area = |a: Vec3, b: Vec3, c: Vec3| {
        f32::midpoint((a[i] - b[i]) * (b[j] - c[j]), (a[j] - b[j]) * (c[i] - b[i]))
    };
    let mut weights = [area(v2, v3, co), area(v3, v1, co), area(v1, v2, co)];
    let total = weights[0] + weights[1] + weights[2];
    let inverse = 1.0 / total;
    weights = weights.map(|weight| weight * inverse);
    if weights.iter().all(|weight| weight.is_finite()) {
        weights
    } else {
        [1.0 / 3.0; 3]
    }
}

/// Blender's `BKE_shrinkwrap_snap_point_to_surface`: places the projected point at `offset` from
/// the hit.
fn snap_point(
    surface: &Surface,
    space: Option<&Space>,
    mode: WrapMode,
    hit: Hit,
    offset: f32,
    point: Vec3,
) -> Vec3 {
    match mode {
        WrapMode::OnSurface if offset != 0.0 => {
            snap_with_side(point, hit.co, hit.normal, offset, 0.0, true)
        }
        WrapMode::Inside => snap_with_side(point, hit.co, hit.normal, offset, -1.0, false),
        WrapMode::Outside => snap_with_side(point, hit.co, hit.normal, offset, 1.0, false),
        WrapMode::OutsideSurface if offset != 0.0 => {
            snap_with_side(point, hit.co, hit.normal, offset, 1.0, true)
        }
        WrapMode::AboveSurface if offset != 0.0 => {
            hit.co + smooth_normal(surface, space, hit) * offset
        }
        WrapMode::OnSurface | WrapMode::OutsideSurface | WrapMode::AboveSurface => hit.co,
    }
}

/// Blender's `shrinkwrap_snap_with_side`.
fn snap_with_side(
    point: Vec3,
    hit: Vec3,
    normal: Vec3,
    goal: f32,
    mut force_sign: f32,
    force_snap: bool,
) -> Vec3 {
    let mut delta = point - hit;
    let distance = delta.length();
    if distance < f32::EPSILON {
        return if force_snap || goal > 0.0 {
            hit + normal * (goal * force_sign)
        } else {
            hit
        };
    }
    let dsign = if delta.dot(normal) < 0.0 { -1.0 } else { 1.0 };
    if force_sign == 0.0 {
        force_sign = dsign;
    }
    if force_snap || dsign * distance * force_sign < goal {
        delta *= dsign / distance;
        let distance_epsilon = (goal.abs() + len_manhattan(hit)) * 1.0e-4;
        if distance < distance_epsilon {
            delta = interp(normal, delta, distance / distance_epsilon);
        }
        hit + delta * (goal * force_sign)
    } else {
        point
    }
}

/// Blender's `BKE_shrinkwrap_compute_smooth_normal`. Corner normals interpolate where the face's
/// `sharp_face` lookup allows it; otherwise the face normal is used. With a `space`, the result is
/// returned in subject space.
fn smooth_normal(surface: &Surface, space: Option<&Space>, hit: Hit) -> Vec3 {
    let tri = surface.tris[hit.index];
    let interpolate = surface
        .sharp_faces
        .as_ref()
        .is_none_or(|sharp| sharp[tri.face]);
    if interpolate {
        let target = space.map_or(hit.co, |space| {
            transform_point(&space.subject_to_target, hit.co)
        });
        let [v1, v2, v3] = surface.triangle(hit.index);
        let weights = tri_weights(v1, v2, v3, target);
        let corner_normals = tri.corners.map(|corner| surface.corner_normals[corner]);
        let normal = interp3(corner_normals, weights);
        match space {
            Some(space) => transform_normal_inverse(&space.subject_to_target, normal),
            None => normalize(normal),
        }
    } else {
        let normal = surface.face_normals[tri.face];
        space.map_or(normal, |space| {
            transform_normal_inverse(&space.subject_to_target, normal)
        })
    }
}

impl Projection<'_> {
    /// Blender's `shrinkwrap_calc_normal_projection_cb_ex` for one vertex: the positive pass and
    /// then the negative pass, each trying the auxiliary target before the main one, then the
    /// limit. Returns the snapped point, which the caller blends by weight.
    fn project_vertex(&self, co: Vec3, origin: Vec3, direction: Vec3) -> Option<Vec3> {
        let negative_cull = if self.invert_cull {
            self.cull.map(Cull::flipped)
        } else {
            self.cull
        };
        let passes = [
            (self.positive, direction, self.cull),
            (self.negative, -direction, negative_cull),
        ];
        let mut best = None;
        let mut auxiliary_won = false;
        for (enabled, pass_direction, cull) in passes {
            if !enabled {
                continue;
            }
            if let Some(auxiliary) = self.auxiliary
                && project_normal(auxiliary, None, origin, pass_direction, &mut best)
            {
                auxiliary_won = true;
            }
            if project_normal(self.main, cull, origin, pass_direction, &mut best) {
                auxiliary_won = false;
            }
        }
        let (_, hit) = best?;
        if let Some(limit) = self.limit_squared
            && (hit.co - co).length_squared() > limit
        {
            return None;
        }
        let target = match self.auxiliary {
            Some(auxiliary) if auxiliary_won => auxiliary,
            _ => self.main,
        };
        Some(snap_point(
            target.surface,
            Some(&target.space),
            self.mode,
            hit,
            self.offset,
            origin,
        ))
    }
}

/// Blender's `BKE_shrinkwrap_project_normal`: casts the subject-space `direction` against the
/// target. A hit replaces `best` only when it is strictly nearer; a hit that the face culling
/// rejects is dropped.
fn project_normal(
    target: Target,
    cull: Option<Cull>,
    origin: Vec3,
    direction: Vec3,
    best: &mut Option<(f32, Hit)>,
) -> bool {
    let space = &target.space;
    let ray_origin = transform_point(&space.subject_to_target, origin);
    let ray_direction = transform_direction(&space.subject_to_target, direction);
    let bound = match *best {
        Some((distance, _)) => distance,
        None => RAYCAST_DIST_MAX,
    };
    let mut ray = RayHit {
        index: -1,
        dist: bound,
    };
    let found = target.surface.tree.ray_cast(
        ray_origin.to_array(),
        ray_direction.to_array(),
        &mut ray,
        &mut |triangle: usize, ray: &Ray, hit: &mut RayHit| {
            let [v0, v1, v2] = target
                .surface
                .triangle(triangle)
                .map(|point| point.to_array());
            if let Some(distance) = ray.watertight(v0, v1, v2)
                && distance < hit.dist
            {
                hit.index = triangle as isize;
                hit.dist = distance;
            }
        },
    );
    if found < 0 {
        return false;
    }
    let triangle = found as usize;
    let distance = ray.dist;
    let [v1, v2, v3] = target.surface.triangle(triangle);
    let hit_co = ray_origin + ray_direction * distance;
    let normal = transform_normal_inverse(&space.subject_to_target, normal_tri(v1, v2, v3));
    if let Some(cull) = cull {
        let dot = direction.dot(normal);
        if (cull == Cull::Front && dot <= 0.0) || (cull == Cull::Back && dot >= 0.0) {
            return false;
        }
    }
    *best = Some((
        distance,
        Hit {
            index: triangle,
            co: transform_point(&space.target_to_subject, hit_co),
            normal,
        },
    ));
    true
}

fn project_points(
    projection: &Projection,
    positions: &[Vec3],
    weights: &[f32],
    source: &Source,
) -> Vec<Option<Vec3>> {
    positions
        .iter()
        .zip(weights)
        .enumerate()
        .map(|(index, (&co, &weight))| {
            if weight == 0.0 {
                return None;
            }
            let (origin, direction) = match source {
                Source::Axis(axis) => (co, *axis),
                Source::Normal {
                    origins,
                    directions,
                } => (origins[index], directions[index]),
            };
            let snapped = projection.project_vertex(co, origin, direction)?;
            Some(interp(co, snapped, weight))
        })
        .collect()
}

/// Blender's `mul_v3_m4v3` on `[column][row]` storage: a point transform without a perspective
/// divide.
fn transform_point(m: &Mat, v: Vec3) -> Vec3 {
    let [x, y, z] = v.to_array();
    Vec3::new(
        x * m[0][0] + y * m[1][0] + m[2][0] * z + m[3][0],
        x * m[0][1] + y * m[1][1] + m[2][1] * z + m[3][1],
        x * m[0][2] + y * m[1][2] + m[2][2] * z + m[3][2],
    )
}

/// Blender's `BLI_space_transform_apply_normal`: `mul_mat3_m4_v3`, then normalization.
fn transform_direction(m: &Mat, v: Vec3) -> Vec3 {
    let [x, y, z] = v.to_array();
    normalize(Vec3::new(
        x * m[0][0] + y * m[1][0] + m[2][0] * z,
        x * m[0][1] + y * m[1][1] + m[2][1] * z,
        x * m[0][2] + y * m[1][2] + m[2][2] * z,
    ))
}

/// Blender's `BLI_space_transform_invert_normal`: `mul_transposed_mat3_m4_v3`, then
/// normalization.
fn transform_normal_inverse(m: &Mat, v: Vec3) -> Vec3 {
    let [x, y, z] = v.to_array();
    normalize(Vec3::new(
        x * m[0][0] + y * m[0][1] + m[0][2] * z,
        x * m[1][0] + y * m[1][1] + m[1][2] * z,
        x * m[2][0] + y * m[2][1] + m[2][2] * z,
    ))
}

/// Blender's `interp_v3_v3v3`.
fn interp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a * (1.0 - t) + b * t
}

/// Blender's `interp_v3_v3v3v3`.
fn interp3(values: [Vec3; 3], weights: [f32; 3]) -> Vec3 {
    values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2]
}

/// Blender's `len_manhattan_v3`.
fn len_manhattan(v: Vec3) -> f32 {
    v.x.abs() + v.y.abs() + v.z.abs()
}

/// Blender's `normalize_v3`: vectors with no length stay zero.
fn normalize(v: Vec3) -> Vec3 {
    let length_squared = v.dot(v);
    if length_squared > 1.0e-35 {
        v * (1.0 / length_squared.sqrt())
    } else {
        Vec3::ZERO
    }
}

fn to_mat(matrix: DMat4) -> Mat {
    matrix
        .to_cols_array_2d()
        .map(|column| column.map(|value| value as f32))
}

fn to_vec3s(values: Vec<DVec3>) -> Vec<Vec3> {
    values.into_iter().map(DVec3::as_vec3).collect()
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DQuat, DVec3, Vec3};
    use serde_json::json;

    use super::evaluate;
    use crate::{
        geom::{
            Mesh, UvSphereParams,
            bvh::{KdopTree, Ray, RayHit},
        },
        model::{Id, Modifier, ModifierRuntime},
    };

    /// f32 evaluation error bound: Blender's float32 rounding, the same absolute bound as the
    /// differential corpus.
    const F32_TOLERANCE: f64 = 1.0e-5;

    trait TestValue<T> {
        fn test_value(self) -> T;
    }

    impl<T, E: std::fmt::Debug> TestValue<T> for Result<T, E> {
        fn test_value(self) -> T {
            match self {
                Ok(value) => value,
                Err(error) => panic!("invalid shrinkwrap test fixture: {error:?}"),
            }
        }
    }

    impl<T> TestValue<T> for Option<T> {
        fn test_value(self) -> T {
            match self {
                Some(value) => value,
                None => panic!("shrinkwrap test expected a value"),
            }
        }
    }

    fn modifier(params: serde_json::Value, target: Mesh) -> Modifier {
        let mut modifier = Modifier {
            id: Id::new("shrinkwrap_test".to_owned()).test_value(),
            modifier_type: "shrinkwrap".to_owned(),
            name: "Shrinkwrap".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: serde_json::from_value(params).test_value(),
            binding_data: None,
            runtime: ModifierRuntime::default(),
        };
        modifier.runtime.shrinkwrap_target_mesh = Some(target);
        modifier
    }

    fn plane(z: f64) -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-2.0, -2.0, z),
                DVec3::new(2.0, -2.0, z),
                DVec3::new(2.0, 2.0, z),
                DVec3::new(-2.0, 2.0, z),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .test_value()
    }

    fn one_point(point: DVec3) -> Mesh {
        Mesh::from_positions_and_faces(vec![point], Vec::new()).test_value()
    }

    #[test]
    fn ray_through_shared_edge_hits_one_of_two_triangles() {
        let triangles = [
            [Vec3::ZERO, Vec3::X, Vec3::Y],
            [Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
        ];
        let leaves: Vec<[f32; 6]> = triangles
            .iter()
            .map(|triangle| {
                KdopTree::leaf_bounds_for_triangle(triangle.map(|point| point.to_array()))
            })
            .collect();
        let tree = KdopTree::from_leaf_bounds(&leaves);
        let mut ray_hit = RayHit {
            index: -1,
            dist: f32::MAX,
        };
        // The ray crosses the shared diagonal exactly; watertight casting must not miss it.
        let found = tree.ray_cast(
            [0.5, 0.5, 1.0],
            [0.0, 0.0, -1.0],
            &mut ray_hit,
            &mut |triangle: usize, ray: &Ray, hit: &mut RayHit| {
                let [v0, v1, v2] = triangles[triangle].map(|point| point.to_array());
                if let Some(distance) = ray.watertight(v0, v1, v2)
                    && distance < hit.dist
                {
                    hit.index = triangle as isize;
                    hit.dist = distance;
                }
            },
        );
        assert!(found >= 0);
        assert!((ray_hit.dist - 1.0).abs() < 1.0e-6);
    }
    #[test]
    fn nearest_surface_respects_wrap_mode_and_offset() {
        let source = one_point(DVec3::new(0.25, 0.5, 2.0));
        let above = evaluate(
            &source,
            &modifier(
                json!({"wrap_method":"NEAREST_SURFACEPOINT","wrap_mode":"ABOVE_SURFACE","offset":0.25}),
                plane(0.0),
            ),
        )
        .test_value();
        assert!(above.vertices[0].co.distance(DVec3::new(0.25, 0.5, 0.25)) < F32_TOLERANCE);

        let outside = evaluate(
            &source,
            &modifier(
                json!({"wrap_method":"NEAREST_SURFACEPOINT","wrap_mode":"OUTSIDE","offset":0.25}),
                plane(0.0),
            ),
        )
        .test_value();
        assert_eq!(outside.vertices[0].co, source.vertices[0].co);
    }
    #[test]
    fn shrinkwrap_snap_modes_preserve_blender_side_constraints() {
        let target = plane(0.0);
        let apply = |point, mode| {
            let source = one_point(point);
            evaluate(
                &source,
                &modifier(
                    json!({
                        "wrap_method":"NEAREST_SURFACEPOINT",
                        "wrap_mode":mode,
                        "offset":0.25
                    }),
                    target.clone(),
                ),
            )
            .test_value()
            .vertices[0]
                .co
        };
        assert!(
            apply(DVec3::new(0.25, 0.5, 2.0), "ON_SURFACE").distance(DVec3::new(0.25, 0.5, 0.25))
                < F32_TOLERANCE
        );
        assert!(
            apply(DVec3::new(0.25, 0.5, 2.0), "INSIDE").distance(DVec3::new(0.25, 0.5, -0.25))
                < F32_TOLERANCE
        );
        assert!(
            apply(DVec3::new(0.25, 0.5, 0.1), "OUTSIDE").distance(DVec3::new(0.25, 0.5, 0.25))
                < F32_TOLERANCE
        );
        assert!(
            apply(DVec3::new(0.25, 0.5, -1.0), "OUTSIDE_SURFACE")
                .distance(DVec3::new(0.25, 0.5, 0.25))
                < F32_TOLERANCE
        );
    }

    #[test]
    fn project_uses_enabled_axes_directions_limit_and_subdivision_noop() {
        let source = one_point(DVec3::new(0.0, 0.0, 1.0));
        let target = plane(0.0);
        let project = |params| evaluate(&source, &modifier(params, target.clone())).test_value();
        let negative = project(json!({
            "wrap_method":"PROJECT","use_project_z":true,
            "use_positive_direction":false,"use_negative_direction":true
        }));
        assert_eq!(negative.vertices[0].co, DVec3::ZERO);
        let no_axis = project(json!({
            "wrap_method":"PROJECT","use_positive_direction":false,"use_negative_direction":true
        }));
        // Blender's normals_calc_verts gives a loose vertex normalize(position) = +Z, so the
        // negative cast from z = 1 reaches the plane at the origin.
        assert_eq!(no_axis.vertices[0].co, DVec3::ZERO);
        let no_direction = project(json!({
            "wrap_method":"PROJECT","use_project_z":true,
            "use_positive_direction":false,"use_negative_direction":false
        }));
        assert_eq!(no_direction.vertices[0].co, source.vertices[0].co);
        let limited = project(json!({
            "wrap_method":"PROJECT","use_project_z":true,"project_limit":0.5,
            "use_positive_direction":false,"use_negative_direction":true
        }));
        assert_eq!(limited.vertices[0].co, source.vertices[0].co);
        let subdivision_levels = project(json!({
            "wrap_method":"PROJECT","use_project_z":true,"subsurf_levels":3,
            "use_positive_direction":false,"use_negative_direction":true
        }));
        assert_eq!(subdivision_levels.vertices[0].co, negative.vertices[0].co);
    }
    #[test]
    fn project_uses_each_public_axis_and_normalized_axis_combinations() {
        let x_plane = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, -2.0, -2.0),
                DVec3::new(0.0, 2.0, -2.0),
                DVec3::new(0.0, 2.0, 2.0),
                DVec3::new(0.0, -2.0, 2.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .test_value();
        let y_plane = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-2.0, 0.0, -2.0),
                DVec3::new(-2.0, 0.0, 2.0),
                DVec3::new(2.0, 0.0, 2.0),
                DVec3::new(2.0, 0.0, -2.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .test_value();
        let diagonal_plane = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(-2.0, -2.0, 2.0),
                DVec3::new(2.0, -2.0, -2.0),
                DVec3::new(2.0, 2.0, -2.0),
                DVec3::new(-2.0, 2.0, 2.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .test_value();
        let project = |point, target, params| {
            let source = one_point(point);
            evaluate(&source, &modifier(params, target))
                .test_value()
                .vertices[0]
                .co
        };
        assert!(
            project(
                DVec3::new(-1.0, 0.0, 0.0),
                x_plane,
                json!({"wrap_method":"PROJECT","use_project_x":true})
            )
            .distance(DVec3::ZERO)
                < F32_TOLERANCE
        );
        assert!(
            project(
                DVec3::new(0.0, -1.0, 0.0),
                y_plane,
                json!({"wrap_method":"PROJECT","use_project_y":true})
            )
            .distance(DVec3::ZERO)
                < F32_TOLERANCE
        );
        assert!(
            project(
                DVec3::new(-1.0, 0.0, -1.0),
                diagonal_plane,
                json!({"wrap_method":"PROJECT","use_project_x":true,"use_project_z":true})
            )
            .distance(DVec3::ZERO)
                < F32_TOLERANCE
        );
    }

    #[test]
    fn project_culling_and_invert_cull_follow_ray_direction() {
        let source = one_point(DVec3::new(0.0, 0.0, 1.0));
        let target = plane(0.0);
        let evaluate_params =
            |params| evaluate(&source, &modifier(params, target.clone())).test_value();
        let culled = evaluate_params(json!({
            "wrap_method":"PROJECT","use_project_z":true,"cull_face":"FRONT",
            "use_positive_direction":false,"use_negative_direction":true
        }));
        assert_eq!(culled.vertices[0].co, source.vertices[0].co);
        let inverted = evaluate_params(json!({
            "wrap_method":"PROJECT","use_project_z":true,"cull_face":"FRONT",
            "use_positive_direction":false,"use_negative_direction":true,"use_invert_cull":true
        }));
        assert!(inverted.vertices[0].co.z.abs() < F32_TOLERANCE);
    }

    #[test]
    fn nearest_vertex_projects_uv_sphere_vertices_to_radius() {
        let source = Mesh::uv_sphere(UvSphereParams {
            segments: 8,
            ring_count: 4,
            radius: 2.0,
        })
        .test_value();
        // The 32 x 16 unit sphere is the `uv_sphere_default` oracle, whose largest radius error in
        // Blender's f32 output is 1.2177e-6 (`primitive_uv_oracle.json`); the bound below covers it.
        let target = Mesh::uv_sphere(UvSphereParams {
            segments: 32,
            ring_count: 16,
            radius: 1.0,
        })
        .test_value();
        let result = evaluate(
            &source,
            &modifier(json!({"wrap_method":"NEAREST_VERTEX"}), target),
        )
        .test_value();
        // Blender stores UV sphere vertices as f32: its own radius-1 sphere output is off by up to
        // 1.22e-6 (`tests/fixtures/primitive_uvs/primitive_uv_oracle.json`, `uv_sphere_default`).
        assert!(
            result
                .vertices
                .iter()
                .all(|vertex| (vertex.co.length() - 1.0).abs() < 2.0e-6)
        );
    }

    #[test]
    fn nearest_vertex_uses_vertex_group_and_inversion() {
        let mut source = one_point(DVec3::new(0.0, 0.0, 2.0));
        source
            .attributes
            .insert("vertex_groups".to_owned(), json!({"Move":{"v0":0.25}}));
        let target = Mesh::from_positions_and_faces(vec![DVec3::ZERO], Vec::new()).test_value();
        let apply = |invert| {
            evaluate(
                &source,
                &modifier(
                    json!({
                        "wrap_method":"NEAREST_VERTEX","vertex_group":"Move",
                        "invert_vertex_group":invert
                    }),
                    target.clone(),
                ),
            )
            .test_value()
        };
        let epsilon = f32::EPSILON;
        assert!(
            (apply(false).vertices[0].co.z - f64::from(2.0_f32 * 0.75 + epsilon * 0.25)).abs()
                < F32_TOLERANCE
        );
        assert!(
            (apply(true).vertices[0].co.z - f64::from(2.0_f32 * 0.25 + epsilon * 0.75)).abs()
                < F32_TOLERANCE
        );
    }

    #[test]
    fn target_project_uses_a_smooth_target_surface() {
        let source = one_point(DVec3::new(1.25, 0.0, 0.2));
        let target = Mesh::uv_sphere(UvSphereParams {
            segments: 32,
            ring_count: 16,
            radius: 1.0,
        })
        .test_value();
        let projected = evaluate(
            &source,
            &modifier(
                json!({"wrap_method":"TARGET_PROJECT","wrap_mode":"ON_SURFACE"}),
                target,
            ),
        )
        .test_value();
        let expected = DVec3::new(
            0.984_465_777_873_992_9,
            -1.150_836_865_804_194_5e-7,
            0.157_720_044_255_256_65,
        );
        assert!(projected.vertices[0].co.distance(expected) < 1.0e-5);
    }
    #[test]
    fn target_project_keeps_open_surface_boundary_projection() {
        let source = one_point(DVec3::new(2.5, 0.0, 0.5));
        let projected = evaluate(
            &source,
            &modifier(
                json!({"wrap_method":"TARGET_PROJECT","wrap_mode":"ON_SURFACE"}),
                plane(0.0),
            ),
        )
        .test_value();
        assert!(projected.vertices[0].co.distance(DVec3::new(2.0, 0.0, 0.0)) < F32_TOLERANCE);
    }

    #[test]
    fn target_transforms_and_auxiliary_projection_are_respected() {
        let source = one_point(DVec3::new(0.0, 0.0, 2.0));
        let mut transformed = modifier(
            json!({"wrap_method":"NEAREST_SURFACEPOINT","wrap_mode":"ON_SURFACE"}),
            plane(0.0),
        );
        transformed.runtime.shrinkwrap_subject_to_target =
            Some(DMat4::from_translation(DVec3::new(0.0, 0.0, -1.0)).to_cols_array());
        let projected = evaluate(&source, &transformed).test_value();
        assert!(projected.vertices[0].co.distance(DVec3::Z) < F32_TOLERANCE);

        let mut auxiliary = modifier(
            json!({
                "wrap_method":"PROJECT","use_project_z":true,
                "use_positive_direction":true,"use_negative_direction":false
            }),
            plane(0.0),
        );
        auxiliary.runtime.shrinkwrap_auxiliary_target_mesh = Some(plane(0.0));
        auxiliary.runtime.shrinkwrap_auxiliary_subject_to_target =
            Some(DMat4::from_translation(DVec3::new(0.0, 0.0, -3.0)).to_cols_array());
        let source = one_point(DVec3::new(0.0, 0.0, 1.0));
        let projected = evaluate(&source, &auxiliary).test_value();
        assert!(projected.vertices[0].co.distance(DVec3::new(0.0, 0.0, 3.0)) < F32_TOLERANCE);
    }
    #[test]
    fn above_surface_uses_custom_target_normals_under_reflected_nonuniform_transform() {
        let mut target = plane(0.0);
        crate::geom::normals::shade_faces_smooth(&mut target);
        let up = [Some(DVec3::Y); 4];
        crate::geom::clnor::encode_vectors(
            &mut target,
            Some(&up[..]),
            crate::geom::clnor::NormalCodeOptions {
                corner_order: None,
                setter_input: crate::geom::clnor::SetterInput::NormalizeInput,
            },
        )
        .test_value();
        let source = one_point(DVec3::new(0.25, 0.5, 1.0));
        let projected = evaluate(
            &source,
            &modifier(
                json!({
                    "wrap_method":"NEAREST_SURFACEPOINT",
                    "wrap_mode":"ABOVE_SURFACE",
                    "offset":0.25
                }),
                target,
            ),
        )
        .test_value();
        assert!(
            projected.vertices[0]
                .co
                .distance(DVec3::new(0.25, 0.75, 0.0))
                < F32_TOLERANCE
        );

        let scale = DVec3::new(-2.0, 0.5, 3.0);
        let rotation = DQuat::from_rotation_y(0.4);
        let transform = DMat4::from_translation(DVec3::new(1.0, -2.0, 3.0))
            * DMat4::from_quat(rotation)
            * DMat4::from_scale(scale);
        let source = one_point(transform.transform_point3(DVec3::new(0.25, 0.5, 1.0)));
        let mut modifier = modifier(
            json!({
                "wrap_method":"NEAREST_SURFACEPOINT",
                "wrap_mode":"ABOVE_SURFACE",
                "offset":0.25
            }),
            plane(0.0),
        );
        modifier.runtime.shrinkwrap_subject_to_target = Some(transform.inverse().to_cols_array());
        let projected = evaluate(&source, &modifier).test_value();
        let expected = transform.transform_point3(DVec3::new(0.25, 0.5, 0.25));
        assert!(projected.vertices[0].co.distance(expected) < F32_TOLERANCE);
    }

    #[test]
    fn deform_preserves_topology_and_non_position_attributes() {
        let mut source = one_point(DVec3::new(0.0, 0.0, 2.0));
        source
            .attributes
            .insert("color".to_owned(), json!({"v0":[1,0,0,1]}));
        source
            .attributes
            .insert("uv_map".to_owned(), json!({"v0":[0.25,0.75]}));
        let result = evaluate(
            &source,
            &modifier(json!({"wrap_method":"NEAREST_SURFACEPOINT"}), plane(0.0)),
        )
        .test_value();
        assert_eq!(result.edges, source.edges);
        assert_eq!(result.faces, source.faces);
        assert_eq!(
            result
                .vertices
                .iter()
                .map(|vertex| vertex.id)
                .collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(result.attributes, source.attributes);
    }

    #[test]
    fn empty_target_and_unprojectable_axes_are_no_ops() {
        let source = one_point(DVec3::new(0.0, 0.0, 1.0));
        let empty = evaluate(
            &source,
            &modifier(json!({"wrap_method":"NEAREST_SURFACEPOINT"}), Mesh::new()),
        )
        .test_value();
        assert_eq!(empty.vertices, source.vertices);
        let no_faces = evaluate(
            &source,
            &modifier(
                json!({"wrap_method":"PROJECT","use_project_z":true}),
                Mesh::from_positions_and_faces(vec![DVec3::ZERO], Vec::new()).test_value(),
            ),
        )
        .test_value();
        assert_eq!(no_faces.vertices, source.vertices);
    }
    #[test]
    fn unresolved_scene_operands_do_not_evaluate_as_empty_meshes() {
        let source = one_point(DVec3::new(0.0, 0.0, 1.0));
        let mut missing_target =
            modifier(json!({"wrap_method":"NEAREST_SURFACEPOINT"}), plane(0.0));
        missing_target.runtime.shrinkwrap_target_mesh = None;
        missing_target
            .params
            .insert("target".to_owned(), json!("target"));
        assert!(evaluate(&source, &missing_target).is_err());

        let mut missing_auxiliary =
            modifier(json!({"wrap_method":"NEAREST_SURFACEPOINT"}), plane(0.0));
        missing_auxiliary
            .params
            .insert("auxiliary_target".to_owned(), json!("auxiliary"));
        assert!(evaluate(&source, &missing_auxiliary).is_err());
    }
}
