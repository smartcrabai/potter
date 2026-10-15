//! Blender 4.1+ corner normals for shading.
//!
//! Blender picks a normals domain per mesh. When every face is smooth, no edge is sharp
//! and no custom normal exists, each vertex gets one angle-weighted normal over all of its
//! faces (POINT). Otherwise corners form fans that stop at sharp, boundary and non-manifold
//! edges and at flipped-winding neighbours (CORNER). Flat faces use their Newell normal.
//! A `custom_normal` corner value replaces the computed normal.
//!
//! Degenerate geometry is never an error, matching Blender: a zero-area face takes the unit Z
//! normal, a corner whose fan cancels gets the zero vector, and a zero-length edge contributes a
//! right angle to its corner.

use std::collections::{BTreeMap, HashMap, HashSet, hash_map::Entry};
use std::f64::consts::FRAC_PI_2;

use glam::{DMat3, DMat4, DVec3};
use serde_json::{Map, Value, json};

use super::{Face, Mesh, MeshError, edge_key};
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
/// shape-key positions without cloning the mesh. Custom-normal corners keep their
/// authored vectors regardless of `positions`.
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

fn shading_normals(mesh: &Mesh, positions: &[DVec3]) -> Result<Shading> {
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
    let custom = attribute_values(mesh, "custom_normal", "corner", "float3")?;

    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut sharp_edges = HashSet::new();
    for (edge, sharp) in mesh.edges.iter().zip(sharp_edge_flags(mesh)?) {
        if sharp {
            sharp_edges.insert(edge_key(edge.vertices[0], edge.vertices[1]));
        }
    }
    let smooth = smooth_face_flags(mesh)?;

    let mut offsets = Vec::with_capacity(mesh.faces.len());
    let mut corner_total = 0_usize;
    for face in &mesh.faces {
        offsets.push(corner_total);
        corner_total += face.vertices.len();
    }
    // Blender's POINT domain: all faces smooth, no sharp edge, no custom normal.
    let point_domain =
        custom.is_none() && sharp_edges.is_empty() && smooth.iter().all(|smooth| *smooth);

    // Face normals, corner angles, and the angle-weighted sum of every face at each vertex.
    // Blender's vertex normal sums all faces and ignores shading flags.
    let mut face_normals = Vec::with_capacity(mesh.faces.len());
    let mut corner_angles = vec![0.0_f64; corner_total];
    let mut corner_vertices = Vec::with_capacity(corner_total);
    let mut geometric_sums = vec![DVec3::ZERO; mesh.vertices.len()];
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
        let normal = face_normal(&points);
        face_normals.push(normal);
        for (corner, index) in indices.iter().enumerate() {
            let angle = corner_angle(&points, corner);
            corner_angles[offsets[face_index] + corner] = angle;
            geometric_sums[*index] += normal * angle;
        }
        corner_vertices.extend(indices);
    }

    let mut parent: Vec<usize> = (0..corner_total).collect();
    if point_domain {
        let mut first_corner_at = HashMap::<u32, usize>::new();
        for (face_index, face) in mesh.faces.iter().enumerate() {
            for (corner, id) in face.vertices.iter().enumerate() {
                let index = offsets[face_index] + corner;
                match first_corner_at.entry(*id) {
                    Entry::Occupied(first) => union(&mut parent, *first.get(), index),
                    Entry::Vacant(slot) => {
                        slot.insert(index);
                    }
                }
            }
        }
    } else {
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
                join_shared_edge(&mut parent, mesh, &offsets, first, second);
            }
        }
    }

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
        let authored = custom_corner_normals(custom, face)?;
        face_smooth.push(authored.is_some() || smooth[face_index]);
        for corner in 0..face.vertices.len() {
            let index = offsets[face_index] + corner;
            corners.push(match &authored {
                Some(normals) => normals[corner],
                None if smooth[face_index] => fan_sums[roots[index]].normalize_or_zero(),
                None => face_normals[face_index],
            });
        }
    }

    // Blender averages corner normals when custom normals exist; otherwise it uses the sums above.
    let vertices = if custom.is_some_and(|values| !values.is_empty()) {
        let mut sums = vec![DVec3::ZERO; mesh.vertices.len()];
        for (corner, index) in corner_vertices.iter().enumerate() {
            sums[*index] += corners[corner];
        }
        sums
    } else {
        geometric_sums
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
        vertices: vertices.into_iter().map(DVec3::normalize_or_zero).collect(),
    })
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

/// Angle at `points[corner]` between its two edges. A zero-length edge has no direction, so the
/// angle is a right angle, as Blender's `acos` of a zero dot product gives.
pub(crate) fn corner_angle(points: &[DVec3], corner: usize) -> f64 {
    let count = points.len();
    let current = points[corner];
    let previous = (points[(corner + count - 1) % count] - current).try_normalize();
    let next = (points[(corner + 1) % count] - current).try_normalize();
    match (previous, next) {
        (Some(previous), Some(next)) => previous.cross(next).length().atan2(previous.dot(next)),
        _ => FRAC_PI_2,
    }
}

fn custom_corner_normals(
    custom: Option<&Map<String, Value>>,
    face: &Face,
) -> Result<Option<Vec<DVec3>>> {
    let Some(value) = custom.and_then(|values| values.get(&format!("f{}", face.id))) else {
        return Ok(None);
    };
    let invalid = || {
        PotError::invalid_argument(format!(
            "custom_normal for face f{} must have one finite vector per corner",
            face.id
        ))
    };
    let corners = value
        .as_array()
        .filter(|corners| corners.len() == face.vertices.len())
        .ok_or_else(invalid)?;
    // A zero vector stays zero: singular transforms rotate authored normals to zero, as Blender does.
    corners
        .iter()
        .map(|corner| {
            vector3(corner)
                .map(DVec3::normalize_or_zero)
                .ok_or_else(invalid)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn vector3(value: &Value) -> Option<DVec3> {
    let [x, y, z] = value.as_array()?.as_slice() else {
        return None;
    };
    let vector = DVec3::new(x.as_f64()?, y.as_f64()?, z.as_f64()?);
    vector.is_finite().then_some(vector)
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

/// Per face: the `shade_smooth` flag, in `mesh.faces` order.
pub(crate) fn smooth_face_flags(mesh: &Mesh) -> Result<Vec<bool>> {
    let values = attribute_values(mesh, "shade_smooth", "face", "bool")?;
    mesh.faces
        .iter()
        .map(|face| flag(values, &format!("f{}", face.id)))
        .collect()
}

/// Per edge: the `sharp` flag, in `mesh.edges` order.
pub(crate) fn sharp_edge_flags(mesh: &Mesh) -> Result<Vec<bool>> {
    let values = attribute_values(mesh, "sharp", "edge", "bool")?;
    mesh.edges
        .iter()
        .map(|edge| flag(values, &format!("e{}", edge.id)))
        .collect()
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

/// Attributes keyed by face or edge ID. Topology changes leave them stale; instances remap them.
pub(crate) const SHADING_ATTRIBUTES: [&str; 3] = ["shade_smooth", "sharp", "custom_normal"];

/// Whether a topology change leaves shading keys stale: unless `evaluated` keeps each face's vertex
/// sequence and each edge's endpoints, the face- and edge-keyed `SHADING_ATTRIBUTES` name other elements.
pub(crate) fn shading_keys_stale(source: &Mesh, evaluated: &Mesh) -> bool {
    let shaded = SHADING_ATTRIBUTES
        .iter()
        .any(|name| source.attributes.contains_key(*name));
    let faces = |mesh: &Mesh| {
        mesh.faces
            .iter()
            .map(|face| (face.id, face.vertices.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    let edges = |mesh: &Mesh| {
        mesh.edges
            .iter()
            .map(|edge| (edge.id, edge_key(edge.vertices[0], edge.vertices[1])))
            .collect::<BTreeMap<_, _>>()
    };
    shaded && (faces(source) != faces(evaluated) || edges(source) != edges(evaluated))
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

    fn assert_vec_eq(actual: DVec3, expected: DVec3) {
        assert!(
            actual.abs_diff_eq(expected, 1e-12),
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
                assert_vec_eq(*corner, vertex.normalize());
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
    fn custom_normal_overrides_face_corners() {
        let mut mesh = fold();
        set_faces_attribute(&mut mesh, "shade_smooth", &[(0, true), (1, true)]);
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({
                "domain": "corner",
                "type": "float3",
                "values": {"f0": [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]}
            }),
        );
        let shifted: Vec<DVec3> = positions(&mesh)
            .into_iter()
            .map(|point| point + DVec3::new(0.0, 0.0, 0.5))
            .collect();
        let normals = corner_normals(&mesh, &shifted).expect("normals");
        for corner in &normals[0..4] {
            assert_vec_eq(*corner, DVec3::Y);
        }
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
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({
                "domain": "corner",
                "type": "float3",
                "values": {"f1": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]}
            }),
        );
        let faces = render_corner_normals(&mesh, &positions(&mesh)).expect("fold is valid");
        assert!(faces[0].is_none());
        assert_eq!(faces[1], Some(vec![DVec3::Y; 4]));
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

    #[test]
    fn weld_that_keeps_face_ids_and_corner_counts_marks_shading_stale() {
        // Welding copies of vertices 3 and 4 onto 1 and 0 keeps face IDs and corner counts, but e3 now joins other vertices.
        let mut source = Mesh::from_positions_and_faces(
            vec![
                DVec3::ZERO,
                DVec3::X,
                DVec3::Y,
                DVec3::X,
                DVec3::ZERO,
                DVec3::NEG_Y,
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )
        .expect("source mesh");
        set_faces_attribute(&mut source, "shade_smooth", &[(0, true), (1, true)]);
        set_edge_sharp(&mut source, 3, 4);
        let welded = Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::NEG_Y],
            vec![vec![0, 1, 2], vec![1, 0, 3]],
        )
        .expect("welded mesh");
        assert!(shading_keys_stale(&source, &welded));
        assert!(!shading_keys_stale(&source, &source.clone()));
    }
}
