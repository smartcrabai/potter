use std::collections::HashMap;

use glam::{DMat3, DQuat, DVec3, EulerRot};
use serde::{Deserialize, Serialize};

use super::Mesh;
use crate::error::{ErrorCode, PotError, Result};

const MAX_ELEMENTS: usize = 128;
const MAX_GRID_CELLS: usize = 200_000;
const MAX_FIELD_EVALUATIONS: usize = 2_000_000;
const MAX_GRID_AXIS: usize = 10_000;
const MAX_SURFACE_VERTICES: usize = 500_000;
const MAX_SURFACE_FACES: usize = 240_000;

/// Primitive shape used by a metaball field element.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    #[default]
    Ball,
    Capsule,
    Plane,
    Ellipsoid,
    Cube,
}

/// One signed implicit-field contributor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetaballElement {
    #[serde(rename = "type")]
    pub element_type: ElementType,
    pub co: DVec3,
    pub radius: f64,
    pub stiffness: f64,
    /// Half-extents for plane, ellipsoid, and cube; capsule length is `size.z`.
    pub size: DVec3,
    /// Euler rotation in radians, applied in XYZ order.
    pub rotation: DVec3,
    pub negative: bool,
}

impl Default for MetaballElement {
    fn default() -> Self {
        Self {
            element_type: ElementType::Ball,
            co: DVec3::ZERO,
            radius: 1.0,
            stiffness: 1.0,
            size: DVec3::ONE,
            rotation: DVec3::ZERO,
            negative: false,
        }
    }
}

/// Metaball elements and the sampling settings used to polygonize their field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetaballData {
    pub elements: Vec<MetaballElement>,
    /// Sampling cell width used to generate this mesh.
    pub resolution: f64,
    /// Render sampling cell width retained in the scene contract.
    pub render_resolution: f64,
    /// Isosurface value; the scalar field is the signed sum of element influences.
    pub threshold: f64,
}

impl Default for MetaballData {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            resolution: 0.2,
            render_resolution: 0.1,
            threshold: 0.5,
        }
    }
}

#[derive(Clone, Copy)]
struct PreparedElement {
    element_type: ElementType,
    co: DVec3,
    inverse_rotation: DQuat,
    radius: f64,
    stiffness: f64,
    size: DVec3,
    negative: bool,
}

/// Polygonize the signed metaball field with marching tetrahedra.
///
/// The returned polygons retain a consistent outward winding. Sampling and input sizes
/// are bounded before grid storage is allocated.
pub fn to_mesh(data: &MetaballData) -> Result<Mesh> {
    validate_settings(data)?;
    let elements = prepare_elements(data)?;
    let positive_count = elements.iter().filter(|element| !element.negative).count();
    if positive_count == 0 {
        return Ok(Mesh::new());
    }

    let (minimum, maximum) = sampling_bounds(data, &elements, positive_count)?;
    let cell_counts = grid_cell_counts(minimum, maximum, data.resolution)?;
    let cells = checked_product(&cell_counts, "metaball grid cell count")?;
    if cells > MAX_GRID_CELLS {
        return Err(limit_error("metaball grid exceeds the cell limit"));
    }
    let node_counts = [cell_counts[0] + 1, cell_counts[1] + 1, cell_counts[2] + 1];
    let node_count = checked_product(&node_counts, "metaball grid node count")?;
    let evaluations = node_count
        .checked_mul(elements.len())
        .ok_or_else(|| limit_error("metaball field evaluation count overflowed"))?;
    if evaluations > MAX_FIELD_EVALUATIONS {
        return Err(limit_error(
            "metaball grid exceeds the field evaluation limit",
        ));
    }

    let mut samples = Vec::new();
    samples
        .try_reserve_exact(node_count)
        .map_err(|_| limit_error("could not allocate metaball sample grid"))?;
    let spacing = data.resolution;
    for z in 0..node_counts[2] {
        for y in 0..node_counts[1] {
            for x in 0..node_counts[0] {
                let point = grid_point([x, y, z], minimum, spacing);
                samples.push(field_value(point, &elements));
            }
        }
    }

    let mut positions = Vec::new();
    let mut polygons = Vec::new();
    let mut intersections = HashMap::new();
    let stride_y = node_counts[0];
    let stride_z = node_counts[0] * node_counts[1];
    let tetrahedra = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ];
    let cube_offsets = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
    let tetra_edges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

    for z in 0..cell_counts[2] {
        for y in 0..cell_counts[1] {
            for x in 0..cell_counts[0] {
                let mut corner_ids = [0_usize; 8];
                let mut corner_values = [0.0_f64; 8];
                let mut corner_points = [DVec3::ZERO; 8];
                for (corner, [dx, dy, dz]) in cube_offsets.iter().copied().enumerate() {
                    let grid_index = [x + dx, y + dy, z + dz];
                    let id = grid_index[0] + grid_index[1] * stride_y + grid_index[2] * stride_z;
                    corner_ids[corner] = id;
                    corner_values[corner] = samples[id];
                    corner_points[corner] = grid_point(grid_index, minimum, spacing);
                }
                for tetrahedron in tetrahedra {
                    polygonize_tetrahedron(
                        tetrahedron,
                        tetra_edges,
                        &corner_ids,
                        &corner_values,
                        &corner_points,
                        data.threshold,
                        &mut positions,
                        &mut polygons,
                        &mut intersections,
                    )?;
                }
            }
        }
    }

    Mesh::from_positions_and_faces(positions, polygons).map_err(|error| {
        PotError::new(
            ErrorCode::InternalError,
            format!("metaball polygonization produced invalid topology: {error}"),
        )
    })
}

fn validate_settings(data: &MetaballData) -> Result<()> {
    if data.elements.len() > MAX_ELEMENTS {
        return Err(limit_error("metaball element count exceeds the limit"));
    }
    positive(data.resolution, "resolution")?;
    positive(data.render_resolution, "render_resolution")?;
    positive(data.threshold, "threshold")?;
    let mut total_stiffness = 0.0;
    for (index, element) in data.elements.iter().enumerate() {
        if !element.co.is_finite() {
            return Err(PotError::invalid_argument(format!(
                "elements[{index}].co must be finite"
            )));
        }
        if !element.rotation.is_finite() {
            return Err(PotError::invalid_argument(format!(
                "elements[{index}].rotation must be finite"
            )));
        }
        positive(element.radius, "element radius")?;
        positive(element.stiffness, "element stiffness")?;
        if !element.size.is_finite() || element.size.cmple(DVec3::ZERO).any() {
            return Err(PotError::invalid_argument(format!(
                "elements[{index}].size must have finite positive components"
            )));
        }
        total_stiffness += element.stiffness;
        if !total_stiffness.is_finite() {
            return Err(limit_error(
                "metaball field strength exceeds numeric limits",
            ));
        }
    }
    Ok(())
}

fn prepare_elements(data: &MetaballData) -> Result<Vec<PreparedElement>> {
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(data.elements.len())
        .map_err(|_| limit_error("could not allocate metaball element data"))?;
    for element in &data.elements {
        let rotation = DQuat::from_euler(
            EulerRot::XYZ,
            element.rotation.x,
            element.rotation.y,
            element.rotation.z,
        );
        if !rotation.is_finite() {
            return Err(PotError::invalid_argument(
                "metaball rotation is not representable",
            ));
        }
        prepared.push(PreparedElement {
            element_type: element.element_type,
            co: element.co,
            inverse_rotation: rotation.conjugate(),
            radius: element.radius,
            stiffness: element.stiffness,
            size: element.size,
            negative: element.negative,
        });
    }
    Ok(prepared)
}

fn sampling_bounds(
    data: &MetaballData,
    elements: &[PreparedElement],
    positive_count: usize,
) -> Result<(DVec3, DVec3)> {
    let multiplier = positive_count as f64;
    let mut minimum = DVec3::splat(f64::INFINITY);
    let mut maximum = DVec3::splat(f64::NEG_INFINITY);
    for element in elements.iter().filter(|element| !element.negative) {
        let ratio = element.stiffness / data.threshold * multiplier;
        if !ratio.is_finite() {
            return Err(limit_error(
                "metaball surface extent exceeds numeric limits",
            ));
        }
        let falloff = element.radius * (ratio - 1.0).max(0.0).sqrt();
        let core_extent = match element.element_type {
            ElementType::Ball => 0.0,
            ElementType::Capsule => element.size.z * 0.5,
            ElementType::Plane => element.size.x.max(element.size.y),
            ElementType::Ellipsoid | ElementType::Cube => element.size.max_element(),
        };
        let extent = DVec3::splat(falloff + core_extent + data.resolution);
        let element_min = element.co - extent;
        let element_max = element.co + extent;
        if !element_min.is_finite() || !element_max.is_finite() {
            return Err(limit_error("metaball surface bounds exceed numeric limits"));
        }
        minimum = minimum.min(element_min);
        maximum = maximum.max(element_max);
    }
    Ok((minimum, maximum))
}

fn grid_cell_counts(minimum: DVec3, maximum: DVec3, spacing: f64) -> Result<[usize; 3]> {
    let mut counts = [0; 3];
    for axis in 0..3 {
        let span = maximum[axis] - minimum[axis];
        let raw_count = (span / spacing).ceil();
        if !raw_count.is_finite() || raw_count < 1.0 || raw_count > MAX_GRID_AXIS as f64 {
            return Err(limit_error(
                "metaball grid axis exceeds the dimension limit",
            ));
        }
        counts[axis] = raw_count as usize;
    }
    Ok(counts)
}

fn checked_product(values: &[usize; 3], description: &'static str) -> Result<usize> {
    values[0]
        .checked_mul(values[1])
        .and_then(|product| product.checked_mul(values[2]))
        .ok_or_else(|| limit_error(description))
}

fn field_value(point: DVec3, elements: &[PreparedElement]) -> f64 {
    elements.iter().fold(0.0, |field, element| {
        let local = element.inverse_rotation * (point - element.co);
        let q = shape_distance(element, local) / element.radius;
        if !q.is_finite() {
            return field;
        }
        let influence = element.stiffness / (1.0 + q * q);
        if element.negative {
            field - influence
        } else {
            field + influence
        }
    })
}

fn shape_distance(element: &PreparedElement, local: DVec3) -> f64 {
    match element.element_type {
        ElementType::Ball => local.length(),
        ElementType::Capsule => {
            let axial = local.z.clamp(-element.size.z * 0.5, element.size.z * 0.5);
            (local - DVec3::new(0.0, 0.0, axial)).length()
        }
        ElementType::Plane => {
            let outside_x = (local.x.abs() - element.size.x).max(0.0);
            let outside_y = (local.y.abs() - element.size.y).max(0.0);
            DVec3::new(outside_x, outside_y, local.z).length()
        }
        ElementType::Ellipsoid => {
            let normalized = local / element.size;
            let k0 = normalized.length();
            if k0 == 0.0 {
                0.0
            } else {
                let k1 = (local / (element.size * element.size)).length();
                (k0 * (k0 - 1.0) / k1).max(0.0)
            }
        }
        ElementType::Cube => {
            let q = local.abs() - element.size;
            let outside = q.max(DVec3::ZERO).length();
            let inside = q.max_element().min(0.0);
            (outside + inside).max(0.0)
        }
    }
}

fn grid_point(index: [usize; 3], minimum: DVec3, spacing: f64) -> DVec3 {
    minimum + DVec3::new(index[0] as f64, index[1] as f64, index[2] as f64) * spacing
}

#[expect(
    clippy::too_many_arguments,
    reason = "polygonization needs each cell corner array"
)]
fn polygonize_tetrahedron(
    tetrahedron: [usize; 4],
    tetra_edges: [[usize; 2]; 6],
    corner_ids: &[usize; 8],
    corner_values: &[f64; 8],
    corner_points: &[DVec3; 8],
    threshold: f64,
    positions: &mut Vec<DVec3>,
    polygons: &mut Vec<Vec<usize>>,
    intersections: &mut HashMap<(usize, usize), usize>,
) -> Result<()> {
    let mut crossing_ids = [0_usize; 4];
    let mut crossing_count = 0;
    for [first, second] in tetra_edges {
        let first_corner = tetrahedron[first];
        let second_corner = tetrahedron[second];
        let first_inside = corner_values[first_corner] >= threshold;
        let second_inside = corner_values[second_corner] >= threshold;
        if first_inside == second_inside {
            continue;
        }
        let vertex = intersection_vertex(
            corner_ids[first_corner],
            corner_ids[second_corner],
            corner_points[first_corner],
            corner_points[second_corner],
            corner_values[first_corner],
            corner_values[second_corner],
            threshold,
            positions,
            intersections,
        )?;
        if !crossing_ids[..crossing_count].contains(&vertex) {
            crossing_ids[crossing_count] = vertex;
            crossing_count += 1;
        }
    }
    if crossing_count < 3 {
        return Ok(());
    }

    let tetra_points = tetrahedron.map(|corner| corner_points[corner]);
    let tetra_values = tetrahedron.map(|corner| corner_values[corner]);
    let outward = -scalar_gradient(tetra_points, tetra_values);
    let outward_length = outward.length();
    if !outward.is_finite() || !outward_length.is_finite() || outward_length <= f64::EPSILON {
        return Ok(());
    }
    let normal = outward / outward_length;
    let centroid = crossing_ids[..crossing_count]
        .iter()
        .map(|id| positions[*id])
        .fold(DVec3::ZERO, |sum, point| sum + point)
        / crossing_count as f64;
    let mut u = positions[crossing_ids[0]] - centroid;
    let u_length = u.length();
    if !u_length.is_finite() || u_length <= f64::EPSILON {
        return Ok(());
    }
    u /= u_length;
    let v = normal.cross(u);
    let v_length = v.length();
    if !v_length.is_finite() || v_length <= f64::EPSILON {
        return Ok(());
    }
    let v = v / v_length;
    crossing_ids[..crossing_count].sort_by(|first, second| {
        let first_delta = positions[*first] - centroid;
        let second_delta = positions[*second] - centroid;
        let first_angle = first_delta.dot(v).atan2(first_delta.dot(u));
        let second_angle = second_delta.dot(v).atan2(second_delta.dot(u));
        first_angle.total_cmp(&second_angle)
    });
    let mut polygon = Vec::with_capacity(crossing_count);
    polygon.extend_from_slice(&crossing_ids[..crossing_count]);
    if polygons.len() >= MAX_SURFACE_FACES {
        return Err(limit_error("metaball surface exceeds the face limit"));
    }
    polygons.push(polygon);
    Ok(())
}

fn scalar_gradient(points: [DVec3; 4], values: [f64; 4]) -> DVec3 {
    let edges = DMat3::from_cols(
        points[1] - points[0],
        points[2] - points[0],
        points[3] - points[0],
    );
    let scale = values
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return DVec3::ZERO;
    }
    let normalized = values.map(|value| value / scale);
    let differences = DVec3::new(
        normalized[1] - normalized[0],
        normalized[2] - normalized[0],
        normalized[3] - normalized[0],
    );
    edges.inverse().transpose() * differences
}

#[expect(
    clippy::too_many_arguments,
    reason = "edge intersection needs both sample endpoints"
)]
fn intersection_vertex(
    first_id: usize,
    second_id: usize,
    first_point: DVec3,
    second_point: DVec3,
    first_value: f64,
    second_value: f64,
    threshold: f64,
    positions: &mut Vec<DVec3>,
    intersections: &mut HashMap<(usize, usize), usize>,
) -> Result<usize> {
    let difference = second_value - first_value;
    let numerator = threshold - first_value;
    let t = if difference.is_finite() && numerator.is_finite() {
        numerator / difference
    } else {
        let scale = first_value
            .abs()
            .max(second_value.abs())
            .max(threshold.abs())
            .max(1.0);
        ((threshold / scale) - (first_value / scale))
            / ((second_value / scale) - (first_value / scale))
    };
    if !t.is_finite() {
        return Err(PotError::invalid_argument(
            "metaball field intersection is not numerically stable",
        ));
    }
    let (key, position) = if t <= 1.0e-12 {
        ((first_id, first_id), first_point)
    } else if t >= 1.0 - 1.0e-12 {
        ((second_id, second_id), second_point)
    } else {
        let key = if first_id < second_id {
            (first_id, second_id)
        } else {
            (second_id, first_id)
        };
        (key, first_point.lerp(second_point, t))
    };
    if let Some(index) = intersections.get(&key) {
        return Ok(*index);
    }
    let index = positions.len();
    if positions.len() >= MAX_SURFACE_VERTICES {
        return Err(limit_error("metaball surface exceeds the vertex limit"));
    }
    positions.push(position);
    if !position.is_finite() {
        return Err(PotError::invalid_argument(
            "metaball surface vertex is not finite",
        ));
    }
    intersections.insert(key, index);
    Ok(index)
}

fn positive(value: f64, name: &'static str) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PotError::invalid_argument(format!(
            "metaball {name} must be finite and positive"
        )))
    }
}

fn limit_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use glam::DVec3;
    use proptest::prelude::*;

    use super::*;

    fn one_ball() -> MetaballData {
        MetaballData {
            elements: vec![MetaballElement {
                element_type: ElementType::Ball,
                co: DVec3::ZERO,
                radius: 1.0,
                stiffness: 1.0,
                size: DVec3::ONE,
                rotation: DVec3::ZERO,
                negative: false,
            }],
            resolution: 0.05,
            render_resolution: 0.05,
            threshold: 0.5,
        }
    }

    #[test]
    fn single_ball_bounds_follow_its_radius_within_one_cell() {
        let mesh = to_mesh(&one_ball()).unwrap();
        let bounds = mesh.bounds().unwrap();
        for axis in 0..3 {
            assert!((bounds.min[axis] + 1.0).abs() <= 0.05);
            assert!((bounds.max[axis] - 1.0).abs() <= 0.05);
        }
        assert!(!mesh.faces.is_empty());
        mesh.validate().unwrap();
    }

    #[test]
    fn negative_element_cuts_a_combined_surface() {
        let mut positive = one_ball();
        positive.resolution = 0.15;
        positive.render_resolution = 0.15;
        let mut combined = positive.clone();
        combined.elements.push(MetaballElement {
            element_type: ElementType::Ball,
            co: DVec3::new(0.45, 0.0, 0.0),
            radius: 0.7,
            stiffness: 1.0,
            size: DVec3::ONE,
            rotation: DVec3::ZERO,
            negative: true,
        });
        let positive_mesh = to_mesh(&positive).unwrap();
        let combined_mesh = to_mesh(&combined).unwrap();
        let prepared = prepare_elements(&combined).unwrap();
        let center_of_negative = DVec3::new(0.45, 0.0, 0.0);
        let cut_value = field_value(center_of_negative, &prepared);
        let positive_value = field_value(center_of_negative, &prepared[..1]);
        assert!(cut_value < combined.threshold);
        assert!(positive_value > combined.threshold);
        assert_ne!(combined_mesh, positive_mesh);
        assert!(combined_mesh.validate().is_ok());
        positive.elements[0].negative = true;
        assert!(to_mesh(&positive).unwrap().faces.is_empty());
    }

    proptest! {
        #[test]
        fn finite_ball_samples_produce_bounded_field_values(
            radius in 0.5_f64..2.0,
            x in -10.0_f64..10.0,
            y in -10.0_f64..10.0,
            z in -10.0_f64..10.0,
        ) {
            let element = PreparedElement {
                element_type: ElementType::Ball,
                co: DVec3::ZERO,
                inverse_rotation: glam::DQuat::IDENTITY,
                radius,
                stiffness: 1.0,
                size: DVec3::ONE,
                negative: false,
            };
            let value = field_value(DVec3::new(x, y, z), &[element]);
            prop_assert!(value.is_finite());
            prop_assert!((0.0..=1.0).contains(&value));
        }
    }
}
