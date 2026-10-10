use glam::{DMat4, DVec3};
use serde::{Deserialize, Serialize};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Aabb, Mesh},
};

/// Interpolation used to sample lattice-point displacements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum LatticeInterpolation {
    /// Trilinear interpolation between the eight surrounding points.
    #[default]
    Linear,
    /// Catmull-Rom cubic interpolation.
    Cardinal,
    #[serde(rename = "bspline")]
    BSpline,
}

/// A regular 3D control lattice in an explicit mesh-local cage domain.
///
/// Point storage is u-fastest, then v, then w: `((w * points_v) + v) * points_u + u`.
/// Each point is a deformed position; the undeformed regular lattice is derived from
/// [`Self::domain`] and the three point counts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LatticeData {
    /// Number of control points along the local u axis.
    pub points_u: usize,
    /// Number of control points along the local v axis.
    pub points_v: usize,
    /// Number of control points along the local w axis.
    pub points_w: usize,
    /// Interpolation method for displacements from the undeformed lattice.
    pub interpolation: LatticeInterpolation,
    /// Deformed control-point positions in u-fastest storage order.
    pub points: Vec<DVec3>,
    /// Mesh-local cage bounds corresponding to normalized coordinates `[0, 1]^3`.
    pub domain: Aabb,
}

impl Default for LatticeData {
    fn default() -> Self {
        let domain = default_domain();
        Self {
            points_u: 2,
            points_v: 2,
            points_w: 2,
            interpolation: LatticeInterpolation::Linear,
            points: identity_points(2, 2, 2, domain),
            domain,
        }
    }
}

const MIN_POINTS_PER_AXIS: usize = 2;
const MAX_POINTS_PER_AXIS: usize = 256;
const MAX_LATTICE_POINTS: usize = 1_000_000;
const MAX_MESH_VERTICES: usize = 1_000_000;
const MAX_MESH_ELEMENTS: usize = 3_000_000;

fn default_domain() -> Aabb {
    Aabb {
        min: DVec3::splat(-1.0),
        max: DVec3::splat(1.0),
    }
}

fn identity_points(points_u: usize, points_v: usize, points_w: usize, domain: Aabb) -> Vec<DVec3> {
    let mut points = Vec::with_capacity(points_u * points_v * points_w);
    for w in 0..points_w {
        for v in 0..points_v {
            for u in 0..points_u {
                points.push(rest_position(u, v, w, points_u, points_v, points_w, domain));
            }
        }
    }
    points
}

fn rest_position(
    u: usize,
    v: usize,
    w: usize,
    points_u: usize,
    points_v: usize,
    points_w: usize,
    domain: Aabb,
) -> DVec3 {
    DVec3::new(
        axis_position(domain.min.x, domain.max.x, u, points_u),
        axis_position(domain.min.y, domain.max.y, v, points_v),
        axis_position(domain.min.z, domain.max.z, w, points_w),
    )
}

fn axis_position(min: f64, max: f64, index: usize, count: usize) -> f64 {
    min + (max - min) * (index as f64 / (count - 1) as f64)
}

/// Deform a mesh through the lattice, optionally blending each result by a vertex weight.
///
/// Weights must be finite values in `[0, 1]`; zero leaves a vertex unchanged and one
/// applies the full lattice displacement. Vertices beyond the cage domain use the
/// nearest cage boundary coordinate.
///
/// # Errors
///
/// Returns `InvalidArgument` for malformed lattice data, weights, or mesh topology, and
/// `LimitExceeded` when the lattice or mesh exceeds supported dimensions/counts.
pub fn deform_mesh(
    mesh: &Mesh,
    lattice: &LatticeData,
    vertex_weights: Option<&[f64]>,
) -> Result<Mesh> {
    deform_mesh_with_transform(mesh, lattice, vertex_weights, DMat4::IDENTITY)
}

/// Deform a mesh using a lattice-to-mesh-local transform.
///
/// The lattice cage remains in lattice-object-local coordinates. The matrix maps its
/// local points into the input mesh's local coordinates.
///
/// # Errors
///
/// Returns the same errors as [`deform_mesh`], as well as `InvalidArgument` when the
/// lattice-to-mesh transform is non-finite or singular.
pub fn deform_mesh_with_transform(
    mesh: &Mesh,
    lattice: &LatticeData,
    vertex_weights: Option<&[f64]>,
    lattice_to_mesh: DMat4,
) -> Result<Mesh> {
    let point_count = validate_lattice(lattice)?;
    let mesh_element_count = mesh
        .vertices
        .len()
        .checked_add(mesh.edges.len())
        .and_then(|count| count.checked_add(mesh.faces.len()));
    if mesh.vertices.len() > MAX_MESH_VERTICES
        || mesh_element_count.is_none_or(|count| count > MAX_MESH_ELEMENTS)
    {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "mesh element count exceeds the lattice deformation limit",
        ));
    }
    if let Some(weights) = vertex_weights {
        if weights.len() != mesh.vertices.len() {
            return Err(PotError::invalid_argument(
                "vertex weight count must match mesh vertex count",
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
        {
            return Err(PotError::invalid_argument(
                "vertex weights must be finite values in [0, 1]",
            ));
        }
    }
    if !lattice_to_mesh.is_finite() {
        return Err(PotError::invalid_argument(
            "lattice-to-mesh transform must be finite",
        ));
    }
    let mesh_to_lattice = lattice_to_mesh.inverse();
    if !mesh_to_lattice.is_finite() {
        return Err(PotError::invalid_argument(
            "lattice-to-mesh transform must be invertible",
        ));
    }
    mesh.validate()
        .map_err(|error| PotError::invalid_argument(format!("invalid input mesh: {error}")))?;

    if is_identity(lattice, point_count) {
        return Ok(mesh.clone());
    }

    let mut output = mesh.clone();
    for (index, vertex) in output.vertices.iter_mut().enumerate() {
        let weight = vertex_weights.map_or(1.0, |weights| weights[index]);
        if weight == 0.0 {
            continue;
        }
        let lattice_position = mesh_to_lattice.transform_point3(vertex.co);
        let lattice_displacement = sample_displacement(lattice_position, lattice);
        let mesh_displacement = lattice_to_mesh.transform_vector3(lattice_displacement);
        let deformed_position = vertex.co + mesh_displacement * weight;
        if !deformed_position.is_finite() {
            return Err(PotError::invalid_argument(
                "lattice deformation produced non-finite vertex coordinates",
            ));
        }
        vertex.co = deformed_position;
    }
    Ok(output)
}

fn validate_lattice(lattice: &LatticeData) -> Result<usize> {
    let dimensions = [lattice.points_u, lattice.points_v, lattice.points_w];
    if dimensions.iter().any(|&count| count < MIN_POINTS_PER_AXIS) {
        return Err(PotError::invalid_argument(
            "lattice dimensions must each contain at least two points",
        ));
    }
    if dimensions.iter().any(|&count| count > MAX_POINTS_PER_AXIS) {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "lattice dimension exceeds the supported per-axis limit",
        ));
    }
    let point_count = dimensions
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "lattice point count exceeds addressable limits",
            )
        })?;
    if point_count > MAX_LATTICE_POINTS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "lattice point count exceeds the supported limit",
        ));
    }
    if lattice.points.len() != point_count {
        return Err(PotError::invalid_argument(
            "lattice point count does not match lattice dimensions",
        ));
    }
    if !lattice.domain.min.is_finite()
        || !lattice.domain.max.is_finite()
        || !(lattice.domain.min.cmplt(lattice.domain.max)).all()
        || !(lattice.domain.max - lattice.domain.min).is_finite()
    {
        return Err(PotError::invalid_argument(
            "lattice domain must have finite, strictly increasing bounds",
        ));
    }
    if lattice.points.iter().any(|point| !point.is_finite()) {
        return Err(PotError::invalid_argument(
            "lattice points must have finite coordinates",
        ));
    }
    Ok(point_count)
}

fn is_identity(lattice: &LatticeData, point_count: usize) -> bool {
    let mut index = 0;
    for w in 0..lattice.points_w {
        for v in 0..lattice.points_v {
            for u in 0..lattice.points_u {
                if lattice.points[index]
                    != rest_position(
                        u,
                        v,
                        w,
                        lattice.points_u,
                        lattice.points_v,
                        lattice.points_w,
                        lattice.domain,
                    )
                {
                    return false;
                }
                index += 1;
            }
        }
    }
    index == point_count
}

#[derive(Clone, Copy)]
struct AxisSamples {
    indices: [usize; 4],
    weights: [f64; 4],
    count: usize,
}

fn axis_samples(coordinate: f64, count: usize, interpolation: LatticeInterpolation) -> AxisSamples {
    let lower = coordinate.floor() as usize;
    let fraction = coordinate - lower as f64;
    match interpolation {
        LatticeInterpolation::Linear => AxisSamples {
            indices: [lower, (lower + 1).min(count - 1), 0, 0],
            weights: [1.0 - fraction, fraction, 0.0, 0.0],
            count: 2,
        },
        LatticeInterpolation::Cardinal | LatticeInterpolation::BSpline => {
            let t = fraction;
            let t2 = t * t;
            let t3 = t2 * t;
            let weights = if interpolation == LatticeInterpolation::Cardinal {
                [
                    -0.5 * t + t2 - 0.5 * t3,
                    1.0 - 2.5 * t2 + 1.5 * t3,
                    0.5 * t + 2.0 * t2 - 1.5 * t3,
                    -0.5 * t2 + 0.5 * t3,
                ]
            } else {
                [
                    (1.0 - t).powi(3) / 6.0,
                    (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
                    (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
                    t3 / 6.0,
                ]
            };
            let base = lower as isize;
            AxisSamples {
                indices: [
                    clamp_index(base - 1, count),
                    clamp_index(base, count),
                    clamp_index(base + 1, count),
                    clamp_index(base + 2, count),
                ],
                weights,
                count: 4,
            }
        }
    }
}

fn clamp_index(index: isize, count: usize) -> usize {
    index.clamp(0, count as isize - 1) as usize
}

fn cage_coordinate(value: f64, min: f64, max: f64, count: usize) -> f64 {
    if value <= min {
        0.0
    } else if value >= max {
        (count - 1) as f64
    } else {
        ((value - min) / (max - min)) * (count - 1) as f64
    }
}

fn sample_displacement(position: DVec3, lattice: &LatticeData) -> DVec3 {
    let domain = lattice.domain;
    let x = cage_coordinate(position.x, domain.min.x, domain.max.x, lattice.points_u);
    let y = cage_coordinate(position.y, domain.min.y, domain.max.y, lattice.points_v);
    let z = cage_coordinate(position.z, domain.min.z, domain.max.z, lattice.points_w);
    let u_samples = axis_samples(x, lattice.points_u, lattice.interpolation);
    let v_samples = axis_samples(y, lattice.points_v, lattice.interpolation);
    let w_samples = axis_samples(z, lattice.points_w, lattice.interpolation);
    let mut displacement = DVec3::ZERO;
    for wi in 0..w_samples.count {
        for vi in 0..v_samples.count {
            for ui in 0..u_samples.count {
                let index = (w_samples.indices[wi] * lattice.points_v + v_samples.indices[vi])
                    * lattice.points_u
                    + u_samples.indices[ui];
                let rest = rest_position(
                    u_samples.indices[ui],
                    v_samples.indices[vi],
                    w_samples.indices[wi],
                    lattice.points_u,
                    lattice.points_v,
                    lattice.points_w,
                    domain,
                );
                let weight = u_samples.weights[ui] * v_samples.weights[vi] * w_samples.weights[wi];
                displacement += (lattice.points[index] - rest) * weight;
            }
        }
    }
    displacement
}

#[cfg(test)]
mod tests {
    use glam::DVec3;
    use proptest::prelude::*;

    use crate::geom::Mesh;

    use super::{LatticeData, LatticeInterpolation, deform_mesh};

    #[test]
    fn moved_corner_only_influences_its_local_region() -> Result<(), Box<dyn std::error::Error>> {
        let mut lattice = LatticeData::default();
        lattice.points[0] += DVec3::X;
        let positions = vec![
            DVec3::new(-1.0, -1.0, -1.0),
            DVec3::new(1.0, 1.0, 1.0),
            DVec3::new(0.0, 0.0, 0.0),
        ];
        let mesh = Mesh::from_positions_and_faces(positions.clone(), vec![])?;
        let result = deform_mesh(&mesh, &lattice, None)?;

        assert_eq!(result.vertices[0].co, positions[0] + DVec3::X);
        assert_eq!(result.vertices[1].co, positions[1]);
        assert_eq!(result.vertices[2].co, positions[2] + DVec3::X * 0.125);
        Ok(())
    }

    #[test]
    fn per_vertex_weights_blend_original_and_deformed_positions()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut lattice = LatticeData::default();
        lattice.points[0] += DVec3::X;
        let original = DVec3::new(-1.0, -1.0, -1.0);
        let mesh = Mesh::from_positions_and_faces(vec![original; 3], vec![])?;
        let result = deform_mesh(&mesh, &lattice, Some(&[0.0, 0.5, 1.0]))?;

        assert_eq!(result.vertices[0].co, original);
        assert_eq!(result.vertices[1].co, original + DVec3::X * 0.5);
        assert_eq!(result.vertices[2].co, original + DVec3::X);
        Ok(())
    }
    proptest! {
        #[test]
        fn identity_lattice_preserves_arbitrary_mesh_positions(
            points in prop::collection::vec(
                (-1.0e4_f64..1.0e4, -1.0e4_f64..1.0e4, -1.0e4_f64..1.0e4),
                0..32,
            ),
            interpolation in 0_u8..3,
        ) {
            let positions: Vec<_> = points
                .into_iter()
                .map(|(x, y, z)| DVec3::new(x, y, z))
                .collect();
            let mesh = Mesh::from_positions_and_faces(positions, vec![])
                .unwrap_or_else(|error| panic!("generated lattice test mesh is valid: {error}"));
            let lattice = LatticeData {
                interpolation: match interpolation {
                    0 => LatticeInterpolation::Linear,
                    1 => LatticeInterpolation::Cardinal,
                    _ => LatticeInterpolation::BSpline,
                },
                ..LatticeData::default()
            };

            let deformed = deform_mesh(&mesh, &lattice, None)
                .unwrap_or_else(|error| panic!("identity lattice deformation is valid: {error}"));
            prop_assert_eq!(deformed, mesh);
        }
    }
}
