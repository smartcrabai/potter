//! Blender 4.1+ corner normals for shading.
//!
//! Blender picks a normals domain per mesh. When every face is smooth, no edge is sharp
//! and no custom normal exists, each vertex gets one angle-weighted normal over all of its
//! faces (POINT). Otherwise corners form fans that stop at sharp, boundary and non-manifold
//! edges and at flipped-winding neighbours (CORNER). Flat faces use their Newell normal.
//! Custom-normal codes (see `clnor`) are evaluated against the current fan at the current positions:
//! Auto uses the shared Blender fan result, authored codes replace it, and explicit zero is literal.
//!
//! Degenerate geometry is never an error, matching Blender: a zero-area face takes the unit Z
//! normal, a corner whose fan cancels gets the zero vector, and a zero-length edge contributes a
//! right angle to its corner.

use std::collections::{HashMap, HashSet};

use glam::{DMat3, DMat4, DVec3, Vec3};
use serde_json::{Map, Value, json};

use super::clnor::{self, CornerNormalCode, NormalCodeOptions, SetterInput};
use super::{Mesh, MeshError, edge_key};
use crate::error::{ErrorCode, PotError, Result};

/// Local-to-world normal matrix: the cofactor matrix, signed by the determinant. For invertible
/// transforms it equals |det| times the inverse transpose; for singular ones it still exists and
/// keeps the axes the transform preserves, which is what Blender's PLY writer produces.
pub(crate) fn normal_matrix(matrix: DMat4) -> DMat3 {
    let linear = DMat3::from_mat4(matrix);
    // Columns of the cofactor matrix are cross products of the other two columns.
    let cofactor = DMat3::from_cols(
        linear.y_axis.cross(linear.z_axis),
        linear.z_axis.cross(linear.x_axis),
        linear.x_axis.cross(linear.y_axis),
    );
    if linear.determinant() < 0.0 {
        -cofactor
    } else {
        cofactor
    }
}

/// Shading of one mesh in mesh-local space, for export.
pub(crate) struct ExportNormals {
    /// One normal per face corner, in `mesh.faces` order.
    pub corners: Vec<DVec3>,
    /// One normal per vertex, indexed like `mesh.vertices`.
    pub vertices: Vec<DVec3>,
    /// Per face: whether a corner normal differs from the face's flat normal. Computed from local
    /// geometry, so singular transforms (which collapse world geometry) keep their flags.
    pub smooth: Vec<bool>,
}

/// Corner normals and per-vertex export normals in mesh-local space. Corners are flattened in
/// `mesh.faces` order, then face-corner order; vertices follow `mesh.vertices`.
///
/// `positions` is indexed like `mesh.vertices` (storage order) so callers can pass
/// shape-key positions without cloning the mesh. Authored custom-normal corners are decoded from
/// the stored code state against `positions`; they are derived views, never stored vectors.
///
/// # Errors
///
/// Returns an error for invalid topology, non-finite or mis-sized positions, and malformed
/// `shade_smooth`, `sharp`, or `custom_normal` attributes.
pub(crate) fn export_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<ExportNormals> {
    let shading = shading_normals(mesh, positions)?;
    Ok(ExportNormals {
        corners: shading.corners,
        vertices: shading.vertices,
        smooth: shading.data_smooth,
    })
}

/// Corner normals in mesh-local space, in the order [`export_normals`] returns them.
///
/// # Errors
///
/// Returns the errors of [`export_normals`].
pub(crate) fn corner_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    shading_normals(mesh, positions).map(|shading| shading.corners)
}

/// Blender's `Mesh::vert_normals()`: each vertex's angle-weighted sum of its faces' geometric
/// normals, computed in f32 as Blender computes it. Shading flags and custom normals never change
/// it, so Shrinkwrap projects along it even on meshes that carry custom normals.
///
/// # Errors
///
/// Returns an error for invalid topology, or non-finite or mis-sized positions.
pub(crate) fn geometric_vertex_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    check_shading_input(mesh, positions)?;
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    Ok(float_normals(mesh, positions, &index_by_id)?
        .vertices
        .into_iter()
        .map(Vec3::as_dvec3)
        .collect())
}

/// Corner and flat normals agree within this distance, matching the importers.
const FLAT_TOLERANCE: f64 = 1.0e-4;

/// Whether two normals agree within the flatness tolerance shared by export and import.
pub(crate) fn close(left: DVec3, right: DVec3) -> bool {
    (left - right).abs().max_element() <= FLAT_TOLERANCE
}

/// Shading of one mesh in local space.
struct Shading {
    /// One normal per face corner, in `mesh.faces` order.
    corners: Vec<DVec3>,
    /// Per face: whether the face is smooth or carries custom normals.
    smooth: Vec<bool>,
    /// Per face: whether a corner normal differs from the face's flat normal.
    data_smooth: Vec<bool>,
    /// One normal per vertex, indexed like `mesh.vertices`.
    vertices: Vec<DVec3>,
}

/// Rejects a mesh or positions that the shading math cannot evaluate.
fn check_shading_input(mesh: &Mesh, positions: &[DVec3]) -> Result<()> {
    mesh.validate().map_err(|error| mesh_error(&error))?;
    if positions.len() != mesh.vertices.len() {
        return Err(PotError::invalid_argument(
            "corner normal positions must match the mesh vertex count",
        ));
    }
    if !positions.iter().all(|point| point.is_finite()) {
        return Err(mesh_error(&MeshError::InvalidTopology(
            "vertex coordinate is not finite",
        )));
    }
    Ok(())
}

fn shading_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Shading> {
    check_shading_input(mesh, positions)?;
    let codes = clnor::corner_codes(mesh)?;
    let resolved = codes
        .as_deref()
        .map(|codes| {
            clnor::evaluate_codes(
                mesh,
                codes,
                positions,
                NormalCodeOptions {
                    corner_order: None,
                    setter_input: SetterInput::NormalizeInput,
                },
            )
        })
        .transpose()?;
    shade(mesh, positions, codes.as_deref(), resolved.as_deref(), true)
}

/// Corner normals of `mesh` in the corner (fan) domain, ignoring its custom-normal layer. This is the
/// base the custom-normal setter replaces zero inputs with: Blender's `mesh_normals_corner_custom_set`
/// takes it from `normals_calc_corners`, which always builds fans, never the POINT vertex normal.
pub(crate) fn setter_corner_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Vec<DVec3>> {
    check_shading_input(mesh, positions)?;
    Ok(shade(mesh, positions, None, None, false)?.corners)
}

/// Shading from geometry plus typed custom-normal codes and their shared, current-fan evaluation.
/// `point_domain_allowed` is false for the custom-normal setter, whose corner normals are always fans.
fn shade(
    mesh: &Mesh,
    positions: &[DVec3],
    custom_codes: Option<&[CornerNormalCode]>,
    custom_normals: Option<&[DVec3]>,
    point_domain_allowed: bool,
) -> Result<Shading> {
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let sharp_edges = sharp_edge_keys(mesh)?;
    let smooth = smooth_face_flags(mesh)?;

    // Blender's POINT domain: all faces smooth, no sharp edge, no custom-normal layer. The custom-normal
    // setter never uses it: its corner normals come from `normals_calc_corners` (fans) on every mesh.
    let point_domain = point_domain_allowed
        && custom_codes.is_none()
        && sharp_edges.is_empty()
        && smooth.iter().all(|smooth| *smooth);
    if point_domain {
        return point_domain_shading(mesh, positions, &index_by_id, smooth);
    }

    let mut offsets = Vec::with_capacity(mesh.faces.len());
    let mut corner_total = 0_usize;
    for face in &mesh.faces {
        offsets.push(corner_total);
        corner_total += face.vertices.len();
    }

    // Face normals and corner angles for the fans. Blender's vertex normals sum all faces and ignore
    // shading flags, so they come from `float_normals`.
    let mut face_normals = Vec::with_capacity(mesh.faces.len());
    let mut corner_angles = vec![0.0_f64; corner_total];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let indices = face
            .vertices
            .iter()
            .map(|id| {
                index_by_id.get(id).copied().ok_or_else(|| {
                    mesh_error(&MeshError::InvalidTopology(
                        "face references a missing vertex",
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let points = indices
            .iter()
            .map(|index| positions[*index])
            .collect::<Vec<_>>();
        face_normals.push(face_normal(&points));
        for corner in 0..indices.len() {
            corner_angles[offsets[face_index] + corner] = corner_angle(&points, corner);
        }
    }
    let mut parent: Vec<usize> = (0..corner_total).collect();
    join_smooth_fans(mesh, &offsets, &sharp_edges, &smooth, &mut parent);

    let roots: Vec<usize> = (0..corner_total)
        .map(|corner| find(&mut parent, corner))
        .collect();
    let mut fan_sums = vec![DVec3::ZERO; corner_total];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        if !smooth[face_index] {
            continue;
        }
        for corner in 0..face.vertices.len() {
            let index = offsets[face_index] + corner;
            fan_sums[roots[index]] += face_normals[face_index] * corner_angles[index];
        }
    }

    let mut corners = Vec::with_capacity(corner_total);
    let mut face_smooth = Vec::with_capacity(mesh.faces.len());
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let start = offsets[face_index];
        let has_authored = custom_codes.is_some_and(|codes| {
            codes[start..start + face.vertices.len()]
                .iter()
                .any(|code| *code != CornerNormalCode::Auto)
        });
        face_smooth.push(smooth[face_index] || has_authored);
        for corner in 0..face.vertices.len() {
            let index = start + corner;
            let computed = if smooth[face_index] {
                fan_sums[roots[index]].normalize_or_zero()
            } else {
                face_normals[face_index]
            };
            corners.push(custom_normals.map_or(computed, |normals| normals[index]));
        }
    }

    // Blender uses its corner-to-vertex mixer for an allocated custom-normal layer; meshes without
    // that layer keep Blender's float vertex normals.
    let vertices = if custom_codes.is_some() {
        clnor::mix_corner_normals_to_vertices(mesh, positions, &corners)?
    } else {
        float_normals(mesh, positions, &index_by_id)?
            .vertices
            .into_iter()
            .map(Vec3::as_dvec3)
            .collect()
    };
    let data_smooth = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(face_index, face)| {
            let start = offsets[face_index];
            (start..start + face.vertices.len())
                .any(|index| !close(corners[index], face_normals[face_index]))
        })
        .collect();
    Ok(Shading {
        corners,
        smooth: face_smooth,
        data_smooth,
        vertices,
    })
}

/// Blender's POINT domain: each corner takes its vertex's float normal, the angle-weighted sum over
/// every face at that vertex. Nothing can split the fan, so no fans are built.
fn point_domain_shading(
    mesh: &Mesh,
    positions: &[DVec3],
    index_by_id: &HashMap<u32, usize>,
    smooth: Vec<bool>,
) -> Result<Shading> {
    let normals = float_normals(mesh, positions, index_by_id)?;
    let mut start = 0;
    let data_smooth = mesh
        .faces
        .iter()
        .zip(&normals.faces)
        .map(|(face, flat_normal)| {
            let end = start + face.vertices.len();
            let differs = normals.corners[start..end]
                .iter()
                .any(|corner| !close(corner.as_dvec3(), flat_normal.as_dvec3()));
            start = end;
            differs
        })
        .collect();
    Ok(Shading {
        corners: normals.corners.into_iter().map(Vec3::as_dvec3).collect(),
        smooth,
        data_smooth,
        vertices: normals.vertices.into_iter().map(Vec3::as_dvec3).collect(),
    })
}

/// Blender's float normals (`normals_calc_faces` and `normals_calc_verts`), computed in f32 as
/// Blender's `float3` is. Shading flags and custom normals never change them.
struct FloatNormals {
    /// One normal per face, in `mesh.faces` order.
    faces: Vec<Vec3>,
    /// One normal per face corner, flattened in `mesh.faces` order.
    corners: Vec<Vec3>,
    /// One normal per vertex, indexed like `mesh.vertices`.
    vertices: Vec<Vec3>,
}

fn float_normals(
    mesh: &Mesh,
    positions: &[DVec3],
    index_by_id: &HashMap<u32, usize>,
) -> Result<FloatNormals> {
    let points: Vec<Vec3> = positions.iter().map(|point| point.as_vec3()).collect();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut corner_vertices = Vec::new();
    let mut sums = vec![Vec3::ZERO; mesh.vertices.len()];
    let mut has_faces = vec![false; mesh.vertices.len()];
    for face in &mesh.faces {
        let indices = face
            .vertices
            .iter()
            .map(|id| {
                index_by_id.get(id).copied().ok_or_else(|| {
                    mesh_error(&MeshError::InvalidTopology(
                        "face references a missing vertex",
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let face_points: Vec<Vec3> = indices.iter().map(|index| points[*index]).collect();
        let normal = float_face_normal(&face_points);
        faces.push(normal);
        let count = indices.len();
        for (corner, &vertex) in indices.iter().enumerate() {
            let current = face_points[corner];
            let to_previous = float_normalize(face_points[(corner + count - 1) % count] - current);
            let to_next = float_normalize(face_points[(corner + 1) % count] - current);
            let factor = float_safe_acos_approx(float_dot(to_previous, to_next));
            sums[vertex] += normal * factor;
            has_faces[vertex] = true;
        }
        corner_vertices.extend(indices);
    }
    // A vertex without faces takes its own direction, as Blender's `normals_calc_verts` does.
    let vertices: Vec<Vec3> = sums
        .into_iter()
        .zip(has_faces)
        .zip(&points)
        .map(|((sum, has_face), point)| float_normalize(if has_face { sum } else { *point }))
        .collect();
    let corners = corner_vertices
        .iter()
        .map(|vertex| vertices[*vertex])
        .collect();
    Ok(FloatNormals {
        faces,
        corners,
        vertices,
    })
}

/// Blender's `normal_calc_ngon`: Newell's method from the last vertex, normalized by `normalize_v3`,
/// which multiplies by the reciprocal length. A zero-area face takes the unit Z normal.
fn float_face_normal(points: &[Vec3]) -> Vec3 {
    let normal = float_newell(points);
    let length_squared = float_dot(normal, normal);
    if length_squared > 1.0e-35_f32 {
        normal * (1.0 / length_squared.sqrt())
    } else {
        Vec3::Z
    }
}

/// Newell's sum over the polygon edges, starting from the last vertex, before normalisation
/// (`add_newell_cross_v3_v3v3` in `normal_calc_ngon` and `cross_poly_v3`).
fn float_newell(points: &[Vec3]) -> Vec3 {
    let Some(&last) = points.last() else {
        return Vec3::ZERO;
    };
    let mut normal = Vec3::ZERO;
    let mut previous = last;
    for &current in points {
        normal.x += (previous.y - current.y) * (previous.z + current.z);
        normal.y += (previous.z - current.z) * (previous.x + current.x);
        normal.z += (previous.x - current.x) * (previous.y + current.y);
        previous = current;
    }
    normal
}

/// Blender's `math::normalize`: each component is divided by the length, and a vector at or below
/// the `1e-35` squared-length threshold becomes zero.
fn float_normalize(vector: Vec3) -> Vec3 {
    let length_squared = float_dot(vector, vector);
    if length_squared > 1.0e-35_f32 {
        let length = length_squared.sqrt();
        Vec3::new(vector.x / length, vector.y / length, vector.z / length)
    } else {
        Vec3::ZERO
    }
}

/// Blender's `dot_v3v3`, summed left to right in f32.
fn float_dot(left: Vec3, right: Vec3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

/// Blender's `safe_acos_approx` in f32. The constants are Blender's literals at their f32 values.
fn float_safe_acos_approx(x: f32) -> f32 {
    let magnitude = x.abs();
    let m = if magnitude < 1.0 {
        1.0 - (1.0 - magnitude)
    } else {
        1.0
    };
    let acos = (1.0 - m).sqrt()
        * (std::f32::consts::FRAC_PI_2
            + m * (-0.213_300_99_f32 + m * (0.077_980_48_f32 + m * -0.021_640_95_f32)));
    if x < 0.0 {
        std::f32::consts::PI - acos
    } else {
        acos
    }
}

/// Corner normals per face for render shading. `None` marks a flat face, whose triangles keep
/// their geometric normal; smooth and custom-normal faces return one normal per face corner.
///
/// # Errors
///
/// Returns the errors of [`corner_normals`].
pub(crate) fn render_corner_normals(
    mesh: &Mesh,
    positions: &[DVec3],
) -> Result<Vec<Option<Vec<DVec3>>>> {
    let Shading {
        corners, smooth, ..
    } = shading_normals(mesh, positions)?;
    let mut start = 0;
    Ok(mesh
        .faces
        .iter()
        .zip(smooth)
        .map(|(face, smooth)| {
            let end = start + face.vertices.len();
            let normals = smooth.then(|| corners[start..end].to_vec());
            start = end;
            normals
        })
        .collect())
}

/// Union the corners at the shared start and end vertices of two face uses of one edge.
/// Faces that traverse the edge in the same direction are flipped relative to each other;
/// their fans stay split, as Blender treats that edge as sharp.
fn join_shared_edge(
    parent: &mut [usize],
    mesh: &Mesh,
    offsets: &[usize],
    (first_face, first_corner): (usize, usize),
    (second_face, second_corner): (usize, usize),
) {
    let first = &mesh.faces[first_face];
    let second = &mesh.faces[second_face];
    let start = first.vertices[first_corner];
    if second.vertices[second_corner] == start {
        return;
    }
    let first_end = (first_corner + 1) % first.vertices.len();
    let second_end = (second_corner + 1) % second.vertices.len();
    union(
        parent,
        offsets[first_face] + first_corner,
        offsets[second_face] + second_end,
    );
    union(
        parent,
        offsets[first_face] + first_end,
        offsets[second_face] + second_corner,
    );
}

/// Edges flagged `sharp`, as [`edge_key`] pairs of vertex IDs.
fn sharp_edge_keys(mesh: &Mesh) -> Result<HashSet<(u32, u32)>> {
    Ok(mesh
        .edges
        .iter()
        .zip(sharp_edge_flags(mesh)?)
        .filter(|(_, sharp)| *sharp)
        .map(|(edge, _)| edge_key(edge.vertices[0], edge.vertices[1]))
        .collect())
}

/// Joins the corners of every smooth fan in `parent`: smooth faces that share a non-sharp manifold
/// edge join the corners at its two ends. Sharp, boundary and non-manifold edges end fans.
fn join_smooth_fans(
    mesh: &Mesh,
    offsets: &[usize],
    sharp_edges: &HashSet<(u32, u32)>,
    smooth: &[bool],
    parent: &mut [usize],
) {
    let mut incidence: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();
    for (face_index, face) in mesh.faces.iter().enumerate() {
        for corner in 0..face.vertices.len() {
            let next = (corner + 1) % face.vertices.len();
            incidence
                .entry(edge_key(face.vertices[corner], face.vertices[next]))
                .or_default()
                .push((face_index, corner));
        }
    }
    for (key, uses) in &incidence {
        // Sharp edges, boundary edges (one use) and non-manifold edges (3+ uses) end fans.
        if sharp_edges.contains(key) || uses.len() != 2 {
            continue;
        }
        let (first, second) = (uses[0], uses[1]);
        if smooth[first.0] && smooth[second.0] {
            join_shared_edge(parent, mesh, offsets, first, second);
        }
    }
}

/// Newell normal of a polygon, normalized. A polygon with no area has no direction; Blender gives
/// it the unit Z normal, so every face has a normal and no export fails on one.
pub(crate) fn face_normal(points: &[DVec3]) -> DVec3 {
    let normal = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .fold(DVec3::ZERO, |sum, (current, next)| {
            sum + DVec3::new(
                (current.y - next.y) * (current.z + next.z),
                (current.z - next.z) * (current.x + next.x),
                (current.x - next.x) * (current.y + next.y),
            )
        });
    let length = normal.length();
    let scale = points
        .iter()
        .fold(1.0_f64, |scale, point| scale.max(point.abs().max_element()));
    if length.is_finite() && length > f64::EPSILON * scale * scale * 32.0 {
        normal / length
    } else {
        DVec3::Z
    }
}

/// Blender's `safe_acos_approx` (max error 4.5e-5 rad). Blender weights fan and vertex normals by
/// this approximation, not by the exact arc cosine, so the corner weights use it too.
pub(crate) fn safe_acos_approx(x: f64) -> f64 {
    let magnitude = x.abs();
    let m = if magnitude < 1.0 {
        1.0 - (1.0 - magnitude)
    } else {
        1.0
    };
    let acos = (1.0 - m).sqrt()
        * (std::f64::consts::FRAC_PI_2
            + m * (-0.213_300_989 + m * (0.077_980_478 + m * -0.021_640_95)));
    if x < 0.0 {
        std::f64::consts::PI - acos
    } else {
        acos
    }
}

/// Angle at `points[corner]` between its two edges, weighted as Blender weights a fan face: the
/// approximate arc cosine of the normalized edge dot product. A zero-length edge has no direction,
/// so its dot product is zero and the angle is a right angle, as Blender's `acos` of zero gives.
pub(crate) fn corner_angle(points: &[DVec3], corner: usize) -> f64 {
    let count = points.len();
    let current = points[corner];
    let previous = (points[(corner + count - 1) % count] - current).normalize_or_zero();
    let next = (points[(corner + 1) % count] - current).normalize_or_zero();
    safe_acos_approx(previous.dot(next))
}

/// Blender's `normals_calc_verts` (`mesh_normals.cc`): the unit face normal of each polygon, weighted
/// at every vertex by `corner_angle`, summed and normalized. A vertex with no polygon takes the
/// direction of its position, which is Blender's loose-vertex rule.
pub(crate) fn vertex_normals(points: &[DVec3], faces: &[&[usize]]) -> Vec<DVec3> {
    let mut sums = vec![None::<DVec3>; points.len()];
    for face in faces.iter().filter(|face| face.len() >= 3) {
        let corners: Vec<DVec3> = face.iter().map(|&vertex| points[vertex]).collect();
        let normal = face_normal(&corners);
        for (corner, &vertex) in face.iter().enumerate() {
            *sums[vertex].get_or_insert(DVec3::ZERO) += normal * corner_angle(&corners, corner);
        }
    }
    points
        .iter()
        .zip(sums)
        .map(|(point, sum)| match sum {
            Some(sum) => sum.normalize_or_zero(),
            None => point.normalize_or_zero(),
        })
        .collect()
}

/// Values of a named attribute, if present. A present attribute must have the expected
/// domain and type, so a mislabeled attribute is an error rather than silently flat.
fn attribute_values<'a>(
    mesh: &'a Mesh,
    name: &str,
    domain: &str,
    kind: &str,
) -> Result<Option<&'a Map<String, Value>>> {
    let Some(attribute) = mesh.attributes.get(name) else {
        return Ok(None);
    };
    let fields = attribute.as_object().filter(|fields| {
        fields.get("domain").and_then(Value::as_str) == Some(domain)
            && fields.get("type").and_then(Value::as_str) == Some(kind)
    });
    fields
        .and_then(|fields| fields.get("values"))
        .and_then(Value::as_object)
        .map(Some)
        .ok_or_else(|| {
            PotError::with_details(
                ErrorCode::InvalidArgument,
                format!("{name} must be a {kind} {domain} attribute with object values"),
                json!({ "attribute": name }),
            )
        })
}

fn flag(values: Option<&Map<String, Value>>, key: &str) -> Result<bool> {
    match values.and_then(|values| values.get(key)) {
        None => Ok(false),
        Some(Value::Bool(flag)) => Ok(*flag),
        Some(_) => Err(PotError::invalid_argument(format!(
            "boolean attribute value {key} must be true or false"
        ))),
    }
}

/// Per face: the `shade_smooth` flag, in `mesh.faces` order. A missing attribute is flat.
pub(crate) fn smooth_face_flags(mesh: &Mesh) -> Result<Vec<bool>> {
    let values = attribute_values(mesh, "shade_smooth", "face", "bool")?;
    mesh.faces
        .iter()
        .map(|face| flag(values, &format!("f{}", face.id)))
        .collect()
}

/// Field of the `shade_smooth` attribute that records Blender's `sharp_face` attribute as present.
/// `use_smooth` on every face keeps that attribute (all false); `shade_smooth()` removes it. Shrinkwrap
/// reads the difference, so the importer, the operator and the exporter keep the field in step.
pub(crate) const SHARP_FACE_PRESENT: &str = "sharp_face_present";

/// Blender's `sharp_face` lookup for the normal domain and the Shrinkwrap projection: `None` when the
/// attribute is absent (every face smooth), otherwise one flag per face in `mesh.faces` order, `true`
/// when sharp. With a flat face the attribute exists, so it is derived from `shade_smooth`; with every
/// face smooth, its presence is the recorded `SHARP_FACE_PRESENT` field.
pub(crate) fn sharp_face_flags(mesh: &Mesh) -> Result<Option<Vec<bool>>> {
    let smooth = smooth_face_flags(mesh)?;
    if smooth.iter().all(|smooth| *smooth) {
        // The presence flag is a field of the `shade_smooth` attribute object, beside `values`
        // (`record_sharp_face`), not a key inside its per-face values.
        let present = mesh
            .attributes
            .get("shade_smooth")
            .and_then(Value::as_object)
            .and_then(|fields| fields.get(SHARP_FACE_PRESENT))
            .and_then(Value::as_bool)
            == Some(true);
        return Ok(present.then(|| vec![false; smooth.len()]));
    }
    Ok(Some(smooth.iter().map(|smooth| !smooth).collect()))
}

/// Records that Blender's `sharp_face` attribute exists on `mesh` after a write that creates or updates
/// it through the attribute API: `rna_MeshPolygon_use_smooth_set` (`lookup_or_add_for_write`), an
/// attribute write, and a generator that always calls `lookup_or_add` (`MOD_screw.cc` `sharp_faces`).
/// The flag lives on the `shade_smooth` attribute, so a mesh without that attribute is all flat and
/// already has the Blender attribute through its flat faces.
pub(crate) fn record_sharp_face(mesh: &mut Mesh) {
    if let Some(fields) = mesh
        .attributes
        .get_mut("shade_smooth")
        .and_then(Value::as_object_mut)
    {
        fields.insert(SHARP_FACE_PRESENT.to_owned(), json!(true));
    }
}

/// Settles the presence of `sharp_face` after an edit operator rebuilt the mesh through `BMesh`. Blender's
/// `BM_mesh_bm_to_me` writes `sharp_face` only when some face is not smooth (`bmesh_mesh_convert.cc`,
/// `need_sharp_face |= !smooth`), so an all-smooth result has no record of it.
pub(crate) fn settle_sharp_face(mesh: &mut Mesh) -> Result<()> {
    if smooth_face_flags(mesh)?.iter().all(|smooth| *smooth)
        && let Some(fields) = mesh
            .attributes
            .get_mut("shade_smooth")
            .and_then(Value::as_object_mut)
    {
        fields.remove(SHARP_FACE_PRESENT);
    }
    Ok(())
}

/// Blender's `Mesh::normals_domain` (`mesh_normals.cc`): which elements carry the corner normals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NormalDomain {
    Point,
    Face,
    Corner,
}

/// Blender's `Mesh::normals_domain`: a custom normal layer wins; all sharp faces or all sharp edges give
/// the Face domain; no sharp face and no sharp edge give Point; anything mixed is Corner.
pub(crate) fn normals_domain(mesh: &Mesh) -> Result<NormalDomain> {
    if mesh.faces.is_empty() {
        return Ok(NormalDomain::Point);
    }
    if mesh.attributes.contains_key("custom_normal") {
        return Ok(NormalDomain::Corner);
    }
    let faces = sharp_face_flags(mesh)?;
    if faces
        .as_ref()
        .is_some_and(|flags| flags.iter().all(|flag| *flag))
    {
        return Ok(NormalDomain::Face);
    }
    let edges = sharp_edge_flags(mesh)?;
    if !edges.is_empty() && edges.iter().all(|flag| *flag) {
        return Ok(NormalDomain::Face);
    }
    let faces_smooth = faces.is_none_or(|flags| flags.iter().all(|flag| !flag));
    if faces_smooth && edges.iter().all(|flag| !flag) {
        Ok(NormalDomain::Point)
    } else {
        Ok(NormalDomain::Corner)
    }
}

/// Per edge: the `sharp` flag, in `mesh.edges` order. A missing attribute is unsharp.
pub(crate) fn sharp_edge_flags(mesh: &Mesh) -> Result<Vec<bool>> {
    let values = attribute_values(mesh, "sharp", "edge", "bool")?;
    mesh.edges
        .iter()
        .map(|edge| flag(values, &format!("e{}", edge.id)))
        .collect()
}

/// Blender's `BM_face_calc_normal` (`bmesh_polygon.cc`): `normal_tri_v3` and `normal_quad_v3` for
/// triangles and quads, and `bm_face_calc_poly_normal` for other sizes. That polygon normal is the
/// normalised Newell sum with no fallback, so a zero-area n-gon has the zero normal (unlike the mesh
/// normal, `normal_calc_ngon`, which takes +Z). Edit-mode operators read these normals.
pub(crate) fn bmesh_face_normal(points: &[Vec3]) -> Vec3 {
    match points {
        [v1, v2, v3] => float_normalize(float_cross(*v1 - *v2, *v2 - *v3)),
        [v1, v2, v3, v4] => float_normalize(float_cross(*v1 - *v3, *v2 - *v4)),
        _ => float_normalize(float_newell(points)),
    }
}

/// The cross product in the expanded f32 form of `normal_tri_v3` and `normal_quad_v3`.
fn float_cross(left: Vec3, right: Vec3) -> Vec3 {
    Vec3::new(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
}

/// Indexed topology for the angle rules: corners and edge endpoints refer to vertex indices.
struct AngleTopology {
    faces: Vec<Vec<usize>>,
    edges: Vec<[usize; 2]>,
    edge_of: HashMap<(usize, usize), usize>,
    points: Vec<Vec3>,
}

fn unordered(first: usize, second: usize) -> (usize, usize) {
    (first.min(second), first.max(second))
}

fn angle_topology(mesh: &Mesh) -> Result<AngleTopology> {
    let index_of = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let vertex_index = |id: u32| -> Result<usize> {
        index_of.get(&id).copied().ok_or_else(|| {
            mesh_error(&MeshError::InvalidTopology(
                "face or edge references a missing vertex",
            ))
        })
    };
    let faces = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| vertex_index(*id))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let edges = mesh
        .edges
        .iter()
        .map(|edge| -> Result<[usize; 2]> {
            Ok([
                vertex_index(edge.vertices[0])?,
                vertex_index(edge.vertices[1])?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let edge_of = edges
        .iter()
        .enumerate()
        .map(|(index, [first, second])| (unordered(*first, *second), index))
        .collect::<HashMap<_, _>>();
    let points = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect();
    Ok(AngleTopology {
        faces,
        edges,
        edge_of,
        points,
    })
}

impl AngleTopology {
    /// Blender's mesh face normals (`normals_calc_faces`: `normal_calc_ngon` for every size).
    fn face_normals(&self) -> Vec<Vec3> {
        self.faces
            .iter()
            .map(|face| {
                float_face_normal(
                    &face
                        .iter()
                        .map(|&index| self.points[index])
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    /// The edge index of each corner: the edge from a corner to the next corner of its face.
    fn corner_edges(&self) -> Result<Vec<Vec<usize>>> {
        self.faces
            .iter()
            .map(|face| {
                (0..face.len())
                    .map(|position| {
                        let next = face[(position + 1) % face.len()];
                        self.edge_of
                            .get(&unordered(face[position], next))
                            .copied()
                            .ok_or_else(|| {
                                mesh_error(&MeshError::InvalidTopology("face corner has no edge"))
                            })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect()
    }
}

/// What `mesh_edges_sharp_tag` records about an edge's first corner (Blender's `e2l[0]`).
#[derive(Clone, Copy)]
struct EdgeFirstCorner {
    face: usize,
    vertex: usize,
    second: EdgeSecond,
}

/// The second slot of Blender's `e2l`: still unset, decided sharp, or a smooth corner.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeSecond {
    Unset,
    Invalid,
    Smooth,
}

/// Blender's `Mesh.sharp_from_angle_set` edge rule (`rna_mesh_api.cc`, `mesh_edges_sharp_tag` in
/// `blenkernel/intern/mesh_normals.cc`). `sharp_faces` is empty when every face is smooth. `sharp_edges`
/// holds the current flags and is updated in place: an edge becomes sharp when its two faces' normals
/// differ by more than `angle` (`dot < cos(angle)`, f32). A caller handles `angle >= pi` and `angle == 0`.
pub(crate) fn sharp_edges_from_angle(
    mesh: &Mesh,
    sharp_faces: &[bool],
    sharp_edges: &mut [bool],
    angle: f32,
) -> Result<()> {
    let topology = angle_topology(mesh)?;
    if sharp_edges.len() != topology.edges.len()
        || (!sharp_faces.is_empty() && sharp_faces.len() != topology.faces.len())
    {
        return Err(PotError::invalid_argument(
            "sharp flags must match the mesh face and edge counts",
        ));
    }
    if angle >= std::f32::consts::PI {
        return Ok(());
    }
    let face_normals = topology.face_normals();
    let corner_edges = topology.corner_edges()?;
    let limit = angle.cos();
    let mut first_corners: Vec<Option<EdgeFirstCorner>> = vec![None; topology.edges.len()];
    for (face_index, face) in topology.faces.iter().enumerate() {
        let smooth = sharp_faces.is_empty() || !sharp_faces[face_index];
        for (position, &vertex) in face.iter().enumerate() {
            let edge = corner_edges[face_index][position];
            let current = first_corners[edge];
            match current {
                None => {
                    first_corners[edge] = Some(EdgeFirstCorner {
                        face: face_index,
                        vertex,
                        second: if smooth {
                            EdgeSecond::Unset
                        } else {
                            EdgeSecond::Invalid
                        },
                    });
                }
                Some(mut first) if first.second == EdgeSecond::Unset => {
                    let is_angle_sharp =
                        float_dot(face_normals[first.face], face_normals[face_index]) < limit;
                    if !smooth || sharp_edges[edge] || vertex == first.vertex || is_angle_sharp {
                        first.second = EdgeSecond::Invalid;
                        if is_angle_sharp {
                            sharp_edges[edge] = true;
                        }
                    } else {
                        first.second = EdgeSecond::Smooth;
                    }
                    first_corners[edge] = Some(first);
                }
                Some(mut first) if first.second == EdgeSecond::Smooth => {
                    first.second = EdgeSecond::Invalid;
                    sharp_edges[edge] = false;
                    first_corners[edge] = Some(first);
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

/// Blender's `mesh.set_sharpness_by_angle` (`editmesh_set_sharpness_by_angle.cc`, edit mode) with
/// `extend` off (its RNA default). A selected edge with exactly two faces (`BM_edge_loop_pair`) is sharp
/// when its `BMesh` face normals (`BM_face_calc_normal`) have a dot at most `cos(angle)`. Returns the new
/// flag per edge, or `None` when the edge keeps its flag.
pub(crate) fn sharp_edges_by_face_angle(
    mesh: &Mesh,
    selected: &[bool],
    angle: f32,
) -> Result<Vec<Option<bool>>> {
    let topology = angle_topology(mesh)?;
    if selected.len() != topology.edges.len() {
        return Err(PotError::invalid_argument(
            "edge selection must match the mesh edge count",
        ));
    }
    let corner_edges = topology.corner_edges()?;
    let mut incident = vec![Vec::new(); topology.edges.len()];
    for (face_index, edges) in corner_edges.iter().enumerate() {
        for &edge in edges {
            incident[edge].push(face_index);
        }
    }
    let normals = topology
        .faces
        .iter()
        .map(|face| {
            bmesh_face_normal(
                &face
                    .iter()
                    .map(|&index| topology.points[index])
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let limit = angle.cos();
    Ok(incident
        .iter()
        .enumerate()
        .map(|(edge, faces)| match faces.as_slice() {
            [first, second] if selected[edge] => {
                Some(float_dot(normals[*first], normals[*second]) <= limit)
            }
            _ => None,
        })
        .collect())
}

fn union(parent: &mut [usize], first: usize, second: usize) {
    let first = find(parent, first);
    let second = find(parent, second);
    parent[second.max(first)] = second.min(first);
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn mesh_error(error: &MeshError) -> PotError {
    PotError::with_details(
        ErrorCode::InvalidArgument,
        format!("corner normals: {error}"),
        json!({ "mesh_error": error.to_string() }),
    )
}

/// Attributes keyed by face or edge ID. Modifiers remap them through provenance, and instances
/// remap them by element.
pub(crate) const SHADING_ATTRIBUTES: [&str; 4] = ["shade_smooth", "sharp", "seam", "custom_normal"];

/// Marks every face smooth as `Mesh.shade_smooth()` does (`mesh_smooth_set(true)`, `mesh.cc`): the
/// `sharp_face` attribute is removed, so no presence is recorded.
pub(crate) fn shade_faces_smooth(mesh: &mut Mesh) {
    let values: Map<String, Value> = mesh
        .faces
        .iter()
        .map(|face| (format!("f{}", face.id), json!(true)))
        .collect();
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":values}),
    );
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use glam::DVec3;

    use super::*;

    fn set_faces_attribute(mesh: &mut Mesh, name: &str, values: &[(u32, bool)]) {
        let values: Map<String, Value> = values
            .iter()
            .map(|(id, flag)| (format!("f{id}"), json!(flag)))
            .collect();
        mesh.attributes.insert(
            name.to_owned(),
            json!({"domain": "face", "type": "bool", "values": values}),
        );
    }

    fn set_edge_sharp(mesh: &mut Mesh, a: u32, b: u32) {
        let id = mesh
            .edges
            .iter()
            .find(|edge| edge_key(edge.vertices[0], edge.vertices[1]) == edge_key(a, b))
            .map(|edge| edge.id)
            .expect("edge exists");
        mesh.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": {format!("e{id}"): true}}),
        );
    }

    /// Two quads meeting at a 90 degree fold along edge (1,0,0)-(1,1,0).
    /// Face 0 (`[0,1,2,3]`) has normal +Z; face 1 (`[2,1,4,5]`) has normal -X.
    fn fold() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(1.0, 1.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3], vec![2, 1, 4, 5]],
        )
        .expect("fold mesh is valid")
    }

    fn positions(mesh: &Mesh) -> Vec<DVec3> {
        mesh.vertices.iter().map(|vertex| vertex.co).collect()
    }

    /// Blender stores normals as f32, so vectors compare at f32 epsilon; the axis-aligned and
    /// unit expectations here are exact in f32 up to one rounding step.
    fn assert_vec_eq(actual: DVec3, expected: DVec3) {
        assert!(
            actual.abs_diff_eq(expected, f64::from(f32::EPSILON)),
            "expected {expected:?}, got {actual:?}"
        );
    }

    /// Compares a Blender float normal at f32 epsilon; `expected` is the nearest f32 value.
    fn assert_f32_vec_eq(actual: DVec3, expected: glam::Vec3) {
        let actual = actual.as_vec3();
        assert!(
            (actual - expected).abs().max_element() <= f32::EPSILON,
            "expected {expected:?}, got {actual:?}"
        );
    }

    const FACE_A_NORMAL: DVec3 = DVec3::Z;
    const FACE_B_NORMAL: DVec3 = DVec3::NEG_X;

    #[test]
    fn flat_face_corners_use_polygon_normal() {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .expect("triangle is valid");
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        assert_eq!(normals.len(), 3);
        for normal in normals {
            assert_vec_eq(normal, DVec3::Z);
        }
    }

    #[test]
    fn smooth_cube_corners_are_normalized_diagonals() {
        let mut mesh = Mesh::box_mesh(crate::geom::BoxParams::default()).expect("cube");
        let face_ids: Vec<u32> = mesh.faces.iter().map(|face| face.id).collect();
        let all_smooth: Vec<(u32, bool)> = face_ids.into_iter().map(|id| (id, true)).collect();
        set_faces_attribute(&mut mesh, "shade_smooth", &all_smooth);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        let mut corners = normals.iter();
        for face in &mesh.faces {
            for id in &face.vertices {
                let corner = corners.next().expect("one normal per corner");
                let vertex = mesh.vertex(*id).expect("vertex exists").co;
                // Blender computes POINT vertex normals in float: the expected corner is the f32
                // nearest to the exact diagonal (0.57735026 for a unit cube corner).
                assert_f32_vec_eq(*corner, vertex.normalize().as_vec3());
            }
        }
        assert!(corners.next().is_none());
    }

    #[test]
    fn sharp_edge_splits_smooth_fan() {
        let mut mesh = fold();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true), (1, true)]);
        let shared = (FACE_A_NORMAL + FACE_B_NORMAL).normalize();

        let joined = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        // Face 0 corners: 0,1,2,3; shared-edge vertices are 1 and 2.
        assert_vec_eq(joined[1], shared);
        assert_vec_eq(joined[2], shared);
        assert_vec_eq(joined[0], FACE_A_NORMAL);

        set_edge_sharp(&mut mesh, 1, 2);
        let split = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        assert_vec_eq(split[1], FACE_A_NORMAL);
        assert_vec_eq(split[2], FACE_A_NORMAL);
        // Face 1 corners follow the same order as its vertices: 2,1,4,5.
        assert_vec_eq(split[4], FACE_B_NORMAL);
        assert_vec_eq(split[5], FACE_B_NORMAL);
    }

    #[test]
    fn flat_neighbor_bounds_smooth_fan() {
        let mut mesh = fold();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true)]);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        assert_vec_eq(normals[1], FACE_A_NORMAL);
        assert_vec_eq(normals[2], FACE_A_NORMAL);
        assert_vec_eq(normals[4], FACE_B_NORMAL);
    }

    #[test]
    fn geometric_vertex_normals_ignore_custom_normals() {
        let mut mesh = fold();
        set_custom_vectors(&mut mesh, &[Some(-DVec3::Z); 8]);
        let points = positions(&mesh);
        let geometric = geometric_vertex_normals(&mesh, &points).expect("normals");
        // Vertex 0 belongs only to face 0, so its geometric normal is that face's normal.
        assert_vec_eq(geometric[0], FACE_A_NORMAL);
        // Authored custom normals change the shading output but not Blender's vert_normals.
        let shaded = export_normals(&mesh, &points).expect("normals").vertices;
        assert_close(shaded[0], -DVec3::Z, CODEC_TOLERANCE);
    }

    /// Stores one vector per corner as the custom-normal layer (`None` is automatic). Authored
    /// vectors go through the shared setter, so the layer holds codes, not vectors.
    fn set_custom_vectors(mesh: &mut Mesh, vectors: &[Option<DVec3>]) {
        clnor::encode_vectors(
            mesh,
            Some(vectors),
            NormalCodeOptions {
                corner_order: None,
                setter_input: SetterInput::NormalizeInput,
            },
        )
        .expect("custom normals encode");
    }

    /// Authored vectors round-trip through 16-bit codes, so they match to code precision.
    const CODEC_TOLERANCE: f64 = 1.0e-3;

    #[test]
    fn custom_normal_overrides_face_corners() {
        let mut mesh = fold();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true), (1, true)]);
        let mut vectors = vec![None; 8];
        vectors[..4].fill(Some(DVec3::Y));
        set_custom_vectors(&mut mesh, &vectors);
        // A translation keeps every fan space, so the authored codes decode to the same vector.
        let shifted: Vec<DVec3> = positions(&mesh)
            .into_iter()
            .map(|point| point + DVec3::new(0.0, 0.0, 0.5))
            .collect();
        let normals = corner_normals(&mesh, &shifted).expect("normals");
        for corner in &normals[0..4] {
            assert_close(*corner, DVec3::Y, CODEC_TOLERANCE);
        }
    }

    #[test]
    fn allocated_auto_layer_uses_shared_code_fan_evaluation() {
        let mut mesh = fold();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true), (1, true)]);
        let positions = positions(&mesh);
        let codes = vec![clnor::CornerNormalCode::Auto; 8];
        clnor::write_codes(&mut mesh, Some(&codes), NormalCodeOptions::default())
            .expect("write allocated automatic codes");
        let expected =
            clnor::evaluate_codes(&mesh, &codes, &positions, NormalCodeOptions::default())
                .expect("evaluate shared fan codes");
        assert_eq!(
            corner_normals(&mesh, &positions).expect("render corners"),
            expected,
            "allocated Auto codes use the same Blender fan evaluator as authored codes"
        );
    }

    #[test]
    fn custom_vertex_normals_use_corner_angles_and_loose_position_fallback() {
        let positions = vec![
            DVec3::ZERO,
            DVec3::X,
            DVec3::Y,
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(2.0, 0.0, 0.0),
        ];
        let mut mesh =
            Mesh::from_positions_and_faces(positions.clone(), vec![vec![0, 1, 2], vec![0, 2, 3]])
                .expect("adjacent faces with one loose vertex");
        let face_ids = mesh.faces.iter().map(|face| face.id).collect::<Vec<_>>();
        set_faces_attribute(
            &mut mesh,
            "shade_smooth",
            &[(face_ids[0], true), (face_ids[1], true)],
        );
        let mut vectors = vec![Some(DVec3::Z); 6];
        vectors[3..].fill(Some(DVec3::X));
        set_custom_vectors(&mut mesh, &vectors);

        let shading = export_normals(&mesh, &positions).expect("custom vertex normals");
        let angle = |previous: DVec3, center: DVec3, next: DVec3| {
            (previous - center)
                .normalize()
                .dot((next - center).normalize())
                .clamp(-1.0, 1.0)
                .acos()
        };
        let first_angle = angle(positions[2], positions[0], positions[1]);
        let second_angle = angle(positions[3], positions[0], positions[2]);
        let weighted =
            (shading.corners[0] * first_angle + shading.corners[3] * second_angle).normalize();
        let unweighted = (shading.corners[0] + shading.corners[3]).normalize();
        assert!(
            weighted.distance(unweighted) > 0.05,
            "fixture distinguishes angle weighting from uniform averaging"
        );
        assert!(
            shading.vertices[0].abs_diff_eq(weighted, 2.0e-5),
            "shared vertex normal must use Blender's corner-angle weights: {:?} vs {weighted:?}",
            shading.vertices[0]
        );
        assert!(
            shading.vertices[4].abs_diff_eq(DVec3::X, 1.0e-6),
            "a loose vertex uses its normalized position"
        );
    }

    #[test]
    fn null_corner_is_automatic_in_a_mixed_face() {
        let mut mesh = Mesh::box_mesh(crate::geom::BoxParams::default()).expect("cube");
        shade_faces_smooth(&mut mesh);
        let computed = corner_normals(&mesh, &positions(&mesh)).expect("computed");
        let mut vectors = vec![None; 24];
        vectors[0] = Some(DVec3::Z);
        set_custom_vectors(&mut mesh, &vectors);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        assert_close(normals[0], DVec3::Z, CODEC_TOLERANCE);
        for corner in 1..4 {
            assert_vec_eq(normals[corner], computed[corner]);
        }
    }

    #[test]
    fn explicit_zero_corner_is_authored_not_automatic() {
        let mut mesh = Mesh::box_mesh(crate::geom::BoxParams::default()).expect("cube");
        shade_faces_smooth(&mut mesh);
        let mut vectors = vec![None; 24];
        vectors[..4].fill(Some(DVec3::ZERO));
        set_custom_vectors(&mut mesh, &vectors);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        for corner in &normals[0..4] {
            assert_vec_eq(*corner, DVec3::ZERO);
        }
    }

    #[test]
    fn all_null_layer_matches_computed_normals() {
        let mut mesh = Mesh::box_mesh(crate::geom::BoxParams::default()).expect("cube");
        shade_faces_smooth(&mut mesh);
        let computed = corner_normals(&mesh, &positions(&mesh)).expect("computed");
        set_custom_vectors(&mut mesh, &vec![None; 24]);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        for (normal, expected) in normals.iter().zip(&computed) {
            assert_vec_eq(*normal, *expected);
        }
    }

    #[test]
    fn shading_flags_ignore_blender_fallbacks() {
        let mut mesh = Mesh::box_mesh(crate::geom::BoxParams::default()).expect("cube");
        let faces = mesh.faces.len();
        let edges = mesh.edges.len();
        mesh.attributes.insert(
            "blender_polygon_smooth".to_owned(),
            json!(vec![true; faces]),
        );
        mesh.attributes.insert(
            "blender_edge_flags".to_owned(),
            json!(vec![json!({"sharp": true, "seam": false}); edges]),
        );
        assert!(
            smooth_face_flags(&mesh)
                .expect("smooth")
                .iter()
                .all(|flag| !flag)
        );
        assert!(
            sharp_edge_flags(&mesh)
                .expect("sharp")
                .iter()
                .all(|flag| !flag)
        );
    }

    #[test]
    fn shape_key_positions_change_result() {
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .expect("triangle is valid");
        let mut shape = positions(&mesh);
        shape[2] = DVec3::new(0.0, 1.0, 1.0);
        let normals = corner_normals(&mesh, &shape).expect("normals");
        let expected = DVec3::new(0.0, -1.0, 1.0).normalize();
        for normal in normals {
            assert_vec_eq(normal, expected);
        }
    }

    fn assert_close(actual: DVec3, expected: DVec3, tolerance: f64) {
        assert!(
            actual.abs_diff_eq(expected, tolerance),
            "expected {expected:?}, got {actual:?}"
        );
    }

    /// Three triangles sharing edge (0,1). Faces 0 and 2 traverse it in the same direction.
    fn fan_mesh() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.5, 1.0, 0.0),
                DVec3::new(0.5, -1.0, 0.5),
                DVec3::new(0.5, -0.2, 1.0),
            ],
            vec![vec![0, 1, 2], vec![1, 0, 3], vec![0, 1, 4]],
        )
        .expect("fan mesh is valid")
    }

    /// Unit square split into two triangles along (0,2); the second is wound the other way.
    fn flipped_pair() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![0, 3, 2]],
        )
        .expect("flipped pair is valid")
    }

    #[test]
    fn all_smooth_non_manifold_fan_uses_point_domain() {
        let mut mesh = fan_mesh();
        set_faces_attribute(
            &mut mesh,
            "shade_smooth",
            &[(0, true), (1, true), (2, true)],
        );
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        // Blender 5.2.2 POINT domain: one angle-weighted normal per vertex over all its faces.
        let vertex = [
            DVec3::new(0.0, -0.289_038, 0.957_318),
            DVec3::new(0.0, -0.289_038, 0.957_318),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 0.447_214, 0.894_427),
            DVec3::new(0.0, -0.980_581, -0.196_116),
        ];
        // Corners: face 0 [0,1,2], face 1 [1,0,3], face 2 [0,1,4].
        let expected = [0_usize, 1, 2, 1, 0, 3, 0, 1, 4].map(|index| vertex[index]);
        assert_eq!(normals.len(), expected.len());
        for (actual, expected) in normals.iter().zip(expected) {
            assert_close(*actual, expected, 1e-5);
        }
    }

    #[test]
    fn mixed_non_manifold_fan_splits_per_face() {
        let mut mesh = fan_mesh();
        set_faces_attribute(
            &mut mesh,
            "shade_smooth",
            &[(0, false), (1, true), (2, true)],
        );
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        // Blender 5.2.2 CORNER domain: the non-manifold edge ends every fan.
        let face1 = DVec3::new(0.0, 0.447_214, 0.894_427);
        let face2 = DVec3::new(0.0, -0.980_581, -0.196_116);
        let expected = [
            DVec3::Z,
            DVec3::Z,
            DVec3::Z,
            face1,
            face1,
            face1,
            face2,
            face2,
            face2,
        ];
        for (actual, expected) in normals.iter().zip(expected) {
            assert_close(*actual, expected, 1e-5);
        }
    }

    #[test]
    fn flipped_smooth_pair_fan_cancels_to_zero() {
        let mut mesh = flipped_pair();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true), (1, true)]);
        // Blender 5.2.2: the vertex normals at the shared diagonal cancel to zero, so the corners
        // at vertices 0 and 2 are zero; the far corners keep their face normals.
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        assert_vec_eq(normals[1], DVec3::Z);
        assert_vec_eq(normals[4], DVec3::NEG_Z);
        for index in [0, 2, 3, 5] {
            assert_vec_eq(normals[index], DVec3::ZERO);
        }
    }

    #[test]
    fn flipped_face_next_to_flat_face_keeps_its_own_normal() {
        let mut mesh = flipped_pair();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, false), (1, true)]);
        let normals = corner_normals(&mesh, &positions(&mesh)).expect("normals");
        // Blender 5.2.2 CORNER domain: the flipped shared edge ends the fan.
        for corner in &normals[0..3] {
            assert_vec_eq(*corner, DVec3::Z);
        }
        for corner in &normals[3..6] {
            assert_vec_eq(*corner, DVec3::NEG_Z);
        }
    }

    #[test]
    fn collapsed_quad_matches_blender() {
        // Four collinear vertices have zero area. Blender 5.2.2 gives a flat face the unit Z
        // normal, and a smooth face [0, Z, Z, 0]: the end corners have angle zero.
        let mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::ZERO,
                DVec3::X,
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(3.0, 0.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .expect("quad is valid");
        let points = positions(&mesh);
        let flat = corner_normals(&mesh, &points).expect("flat normals");
        for normal in flat {
            assert_vec_eq(normal, DVec3::Z);
        }
        let mut smooth_mesh = mesh.clone();
        set_faces_attribute(&mut smooth_mesh, "shade_smooth", &[(0, true)]);
        let smooth = corner_normals(&smooth_mesh, &points).expect("smooth normals");
        assert_vec_eq(smooth[0], DVec3::ZERO);
        assert_vec_eq(smooth[1], DVec3::Z);
        assert_vec_eq(smooth[2], DVec3::Z);
        assert_vec_eq(smooth[3], DVec3::ZERO);
    }

    #[test]
    fn vertex_normals_follow_blender_export_rule() {
        // Blender 5.2.2 PLY vertex normals for this flat zero-area fan: vertex 3 sits only on the
        // degenerate triangle at an angle of zero, so its normal is zero.
        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::new(2.0, 0.0, 0.0)],
            vec![vec![0, 1, 2], vec![1, 0, 3]],
        )
        .expect("fan is valid");
        let vertices = export_normals(&mesh, &positions(&mesh))
            .expect("normals")
            .vertices;
        assert_vec_eq(vertices[0], DVec3::Z);
        assert_vec_eq(vertices[1], DVec3::Z);
        assert_vec_eq(vertices[2], DVec3::Z);
        assert_vec_eq(vertices[3], DVec3::ZERO);
    }

    #[test]
    fn render_corner_normals_keep_custom_normals_on_flat_faces() {
        let mut mesh = fold();
        let mut vectors = vec![None; 8];
        vectors[4..].fill(Some(DVec3::Y));
        set_custom_vectors(&mut mesh, &vectors);
        let faces = render_corner_normals(&mesh, &positions(&mesh)).expect("fold is valid");
        assert!(faces[0].is_none());
        let face_one = faces[1].as_ref().expect("authored face renders smooth");
        assert_eq!(face_one.len(), 4);
        for normal in face_one {
            assert_close(*normal, DVec3::Y, CODEC_TOLERANCE);
        }
    }

    #[test]
    fn render_corner_normals_return_corners_only_for_smooth_faces() {
        let mut mesh = fold();
        let points = positions(&mesh);
        assert_eq!(
            render_corner_normals(&mesh, &points).expect("fold is valid"),
            vec![None, None]
        );
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true)]);
        let faces = render_corner_normals(&mesh, &points).expect("fold is valid");
        assert_eq!(faces[0].as_ref().map(Vec::len), Some(4));
        assert!(faces[1].is_none());
    }
}
