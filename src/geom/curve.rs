//! Curve and surface tessellation into persistent meshes.

use glam::{DMat3, DQuat, DVec3};

use crate::model::{
    CurveData, CurveDimensions, CurveFillMode, CurvePoint, CurveSpline, CurveSplineType,
    HairCurvesData, SurfaceData, SurfacePoint,
};

use super::{Mesh, MeshError};

const MAX_INPUT_POINTS: usize = 65_536;
const MAX_ORDER: usize = 64;
const MAX_RESOLUTION: usize = 1_024;
const MAX_BEVEL_RESOLUTION: usize = 32;
const MAX_MESH_ELEMENTS: usize = 100_000;
const MAX_FILL_POINTS: usize = 4_096;
const EPSILON_SQUARED: f64 = 1.0e-24;

#[derive(Clone, Copy, Debug)]
struct HomogeneousPoint {
    position: DVec3,
    radius: f64,
    tilt: f64,
    weight: f64,
}

impl HomogeneousPoint {
    fn from_curve(point: &CurvePoint) -> Self {
        let weight = point.weight;
        Self {
            position: DVec3::from_array(point.co) * weight,
            radius: point.radius * weight,
            tilt: point.tilt * weight,
            weight,
        }
    }

    fn from_surface(point: &SurfacePoint) -> Self {
        let weight = point.weight;
        Self {
            position: DVec3::from_array(point.co) * weight,
            radius: 0.0,
            tilt: 0.0,
            weight,
        }
    }

    fn interpolate(self, other: Self, factor: f64) -> Self {
        Self {
            position: self.position.lerp(other.position, factor),
            radius: self.radius + (other.radius - self.radius) * factor,
            tilt: self.tilt + (other.tilt - self.tilt) * factor,
            weight: self.weight + (other.weight - self.weight) * factor,
        }
    }

    fn divided(self) -> Result<Self, MeshError> {
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(MeshError::InvalidParameter(
                "rational spline evaluation has an invalid weight",
            ));
        }
        let result = Self {
            position: self.position / self.weight,
            radius: self.radius / self.weight,
            tilt: self.tilt / self.weight,
            weight: 1.0,
        };
        if !result.position.is_finite() || !result.radius.is_finite() || !result.tilt.is_finite() {
            return Err(MeshError::InvalidParameter(
                "rational spline evaluation is not finite",
            ));
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug)]
struct ParameterSample {
    span: usize,
    parameter: f64,
}

#[derive(Clone, Debug)]
struct SplineAxis {
    order: usize,
    knots: Vec<f64>,
    samples: Vec<ParameterSample>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CurvePathPoint {
    pub(crate) position: DVec3,
    pub(crate) radius: f64,
    pub(crate) tilt: f64,
    pub(crate) orientation: DQuat,
}

#[derive(Clone, Copy, Debug)]
struct CurveSample {
    position: DVec3,
    tangent: DVec3,
    radius: f64,
    tilt: f64,
}
#[derive(Clone, Copy, Debug)]
struct Frame {
    normal: DVec3,
    binormal: DVec3,
}

/// Tessellate all curve splines as line paths, filled 2D polygons, or bevel sweeps.
///
/// Bevelled splines form polygon tubes; un-bevelled splines produce loose edges.
/// 2D fills support one simple, closed spline at a time, and do not combine with bevels.
///
/// # Errors
///
/// Returns an error for invalid, non-finite, unsupported, or over-budget curve data.
pub fn evaluate_curve(data: &CurveData) -> Result<Mesh, MeshError> {
    validate_curve_settings(data)?;
    let has_fill = data.fill_mode != CurveFillMode::None;
    if has_fill && data.dimensions != CurveDimensions::TwoD {
        return Err(MeshError::InvalidParameter(
            "filled curves must use two-dimensional dimensions",
        ));
    }
    if data.extrude > 0.0 && data.dimensions != CurveDimensions::TwoD {
        return Err(MeshError::InvalidParameter(
            "curve extrusion is supported only for two-dimensional curves",
        ));
    }
    if data.extrude > 0.0 && !has_fill {
        return Err(MeshError::InvalidParameter(
            "curve extrusion requires a 2D fill mode",
        ));
    }
    if has_fill && data.bevel_depth > 0.0 {
        return Err(MeshError::InvalidParameter(
            "2D curve fill and bevel cannot be combined",
        ));
    }
    if data.taper.is_some() {
        return Err(MeshError::InvalidParameter(
            "curve taper objects are not supported by geometry evaluation",
        ));
    }
    if has_fill && data.splines.len() != 1 {
        return Err(MeshError::InvalidParameter(
            "2D fill requires exactly one closed spline; holes and multiple contours are unsupported",
        ));
    }

    let mut total_input_points = 0_usize;
    let mut predicted_vertices = 0_usize;
    let mut predicted_edges = 0_usize;
    let mut predicted_faces = 0_usize;
    let mut path_counts = Vec::with_capacity(data.splines.len());
    for spline in &data.splines {
        validate_spline(spline)?;
        total_input_points = checked_add(total_input_points, spline.points.len())?;
        if total_input_points > MAX_INPUT_POINTS {
            return Err(MeshError::InvalidParameter(
                "curve control-point count exceeds the tessellation limit",
            ));
        }
        let sample_count = estimated_spline_samples(spline)?;
        path_counts.push(sample_count);
        if has_fill {
            let points = spline.points.len();
            if !spline.cyclic || points < 3 {
                return Err(MeshError::InvalidParameter(
                    "2D fill requires a closed spline with at least three control points",
                ));
            }
            if sample_count > MAX_FILL_POINTS {
                return Err(MeshError::InvalidParameter(
                    "2D fill contour exceeds the supported fill limit",
                ));
            }
            if data.extrude > 0.0 {
                predicted_vertices =
                    checked_add(predicted_vertices, checked_mul(sample_count, 2)?)?;
                predicted_edges = checked_add(predicted_edges, checked_mul(sample_count, 3)?)?;
                predicted_faces = checked_add(predicted_faces, sample_count)?;
                predicted_faces = checked_add(predicted_faces, fill_cap_count(data.fill_mode))?;
            } else {
                predicted_vertices = checked_add(predicted_vertices, sample_count)?;
                predicted_edges = checked_add(predicted_edges, sample_count)?;
                predicted_faces = checked_add(predicted_faces, fill_cap_count(data.fill_mode))?;
            }
        } else if data.bevel_depth > 0.0 {
            let rings = bevel_ring_count(data.bevel_resolution)?;
            let vertices = checked_mul(sample_count, rings)?;
            let path_edges = if spline.cyclic {
                sample_count
            } else {
                sample_count.saturating_sub(1)
            };
            predicted_vertices = checked_add(predicted_vertices, vertices)?;
            predicted_edges = checked_add(
                predicted_edges,
                checked_add(vertices, checked_mul(path_edges, rings)?)?,
            )?;
            predicted_faces = checked_add(predicted_faces, checked_mul(path_edges, rings)?)?;
        } else {
            let edge_count = if spline.cyclic {
                sample_count
            } else {
                sample_count.saturating_sub(1)
            };
            predicted_vertices = checked_add(predicted_vertices, sample_count)?;
            predicted_edges = checked_add(predicted_edges, edge_count)?;
        }
    }
    check_mesh_budget(predicted_vertices, predicted_edges, predicted_faces)?;

    let mut mesh = Mesh::new();
    if has_fill {
        let spline = &data.splines[0];
        let samples = tessellate_spline(spline)?;
        insert_fill(&mut mesh, &samples, data.fill_mode, data.extrude)?;
        return Ok(mesh);
    }

    for (spline, expected_count) in data.splines.iter().zip(path_counts) {
        let samples = tessellate_spline(spline)?;
        if samples.len() != expected_count {
            return Err(MeshError::InvalidTopology(
                "curve tessellation sample count changed unexpectedly",
            ));
        }
        if data.bevel_depth > 0.0 {
            insert_tube(
                &mut mesh,
                &samples,
                spline.cyclic,
                data.twist_mode.as_str(),
                data.bevel_depth,
                bevel_ring_count(data.bevel_resolution)?,
            )?;
        } else {
            insert_path_edges(&mut mesh, &samples, spline.cyclic)?;
        }
    }
    Ok(mesh)
}

/// Represent hair curves as loose mesh edges for line-based consumers.
///
/// # Errors
///
/// Returns an error for non-finite data or geometry that exceeds the mesh budget.
pub fn evaluate_hair_curves(data: &HairCurvesData) -> Result<Mesh, MeshError> {
    let mut vertex_count = 0_usize;
    let mut edge_count = 0_usize;
    for curve in &data.curves {
        if !curve.radius.is_finite() || curve.radius < 0.0 {
            return Err(MeshError::InvalidParameter(
                "hair-curve radius must be finite and non-negative",
            ));
        }
        if curve
            .points
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(MeshError::InvalidParameter(
                "hair-curve coordinates must be finite",
            ));
        }
        if curve.points.len() > 1 {
            vertex_count = checked_add(vertex_count, curve.points.len())?;
            edge_count = checked_add(edge_count, curve.points.len() - 1)?;
        }
    }
    check_mesh_budget(vertex_count, edge_count, 0)?;

    let mut mesh = Mesh::new();
    for curve in &data.curves {
        if curve.points.len() < 2 {
            continue;
        }
        let mut previous = None;
        for point in &curve.points {
            let vertex = mesh.insert_vertex(DVec3::from_array(*point))?;
            if let Some(previous) = previous {
                mesh.insert_edge([previous, vertex])?;
            }
            previous = Some(vertex);
        }
    }
    mesh.validate()?;
    Ok(mesh)
}

/// Tessellate a rectangular rational NURBS surface into a quad mesh.
///
/// Rows of `points` are the U control direction and columns are the V control direction.
///
/// # Errors
///
/// Returns an error for invalid, non-finite, non-rectangular, or over-budget surface data.
pub fn evaluate_surface(data: &SurfaceData) -> Result<Mesh, MeshError> {
    let resolution_u = checked_resolution(data.resolution[0], "surface U resolution")?;
    let resolution_v = checked_resolution(data.resolution[1], "surface V resolution")?;
    if data.points.is_empty() {
        return Ok(Mesh::new());
    }
    let row_count = data.points.len();
    let column_count = data.points[0].len();
    if row_count < 2 || column_count < 2 {
        return Err(MeshError::InvalidParameter(
            "a NURBS surface requires at least two control points in each direction",
        ));
    }
    if data.points.iter().any(|row| row.len() != column_count) {
        return Err(MeshError::InvalidParameter(
            "surface control-point grid must be rectangular",
        ));
    }
    let input_count = checked_mul(row_count, column_count)?;
    if input_count > MAX_INPUT_POINTS {
        return Err(MeshError::InvalidParameter(
            "surface control-point count exceeds the tessellation limit",
        ));
    }
    let order_u = checked_order(data.order_u, row_count, "surface U order")?;
    let order_v = checked_order(data.order_v, column_count, "surface V order")?;
    for row in &data.points {
        for point in row {
            validate_surface_point(point)?;
        }
    }

    let axis_u = make_axis(
        row_count,
        order_u,
        resolution_u,
        data.use_endpoint_u,
        data.cyclic_u,
    )?;
    let axis_v = make_axis(
        column_count,
        order_v,
        resolution_v,
        data.use_endpoint_v,
        data.cyclic_v,
    )?;
    let vertex_count = checked_mul(axis_u.samples.len(), axis_v.samples.len())?;
    let u_cells = if data.cyclic_u {
        axis_u.samples.len()
    } else {
        axis_u.samples.len().saturating_sub(1)
    };
    let v_cells = if data.cyclic_v {
        axis_v.samples.len()
    } else {
        axis_v.samples.len().saturating_sub(1)
    };
    let face_count = checked_mul(u_cells, v_cells)?;
    let edge_count = checked_add(
        checked_mul(u_cells, axis_v.samples.len())?,
        checked_mul(v_cells, axis_u.samples.len())?,
    )?;
    check_mesh_budget(vertex_count, edge_count, face_count)?;

    let homogeneous_grid: Vec<Vec<HomogeneousPoint>> = data
        .points
        .iter()
        .map(|row| row.iter().map(HomogeneousPoint::from_surface).collect())
        .collect();
    let extended_u: Vec<Vec<HomogeneousPoint>> = if data.cyclic_u {
        homogeneous_grid
            .iter()
            .cloned()
            .chain(homogeneous_grid.iter().take(order_u - 1).cloned())
            .collect()
    } else {
        homogeneous_grid.clone()
    };
    let mut positions = Vec::with_capacity(vertex_count);
    for u_sample in &axis_u.samples {
        let mut intermediate = Vec::with_capacity(column_count);
        for column in 0..column_count {
            let controls: Vec<HomogeneousPoint> =
                extended_u.iter().map(|row| row[column]).collect();
            intermediate.push(de_boor(
                &controls,
                &axis_u.knots,
                axis_u.order,
                u_sample.span,
                u_sample.parameter,
            )?);
        }
        let v_controls: Vec<HomogeneousPoint> = if data.cyclic_v {
            intermediate
                .iter()
                .copied()
                .chain(intermediate.iter().take(order_v - 1).copied())
                .collect()
        } else {
            intermediate
        };
        for v_sample in &axis_v.samples {
            let point = de_boor(
                &v_controls,
                &axis_v.knots,
                axis_v.order,
                v_sample.span,
                v_sample.parameter,
            )?
            .divided()?;
            positions.push(point.position);
        }
    }

    let mut mesh = Mesh::new();
    let mut vertex_ids = Vec::with_capacity(vertex_count);
    for position in positions {
        vertex_ids.push(mesh.insert_vertex(position)?);
    }
    for u in 0..u_cells {
        let next_u = (u + 1) % axis_u.samples.len();
        for v in 0..v_cells {
            let next_v = (v + 1) % axis_v.samples.len();
            let index = u * axis_v.samples.len() + v;
            let next_u_index = next_u * axis_v.samples.len() + v;
            let next_v_index = u * axis_v.samples.len() + next_v;
            let diagonal_index = next_u * axis_v.samples.len() + next_v;
            mesh.insert_face(
                vec![
                    vertex_ids[index],
                    vertex_ids[next_u_index],
                    vertex_ids[diagonal_index],
                    vertex_ids[next_v_index],
                ],
                0,
            )?;
        }
    }
    Ok(mesh)
}

fn validate_curve_settings(data: &CurveData) -> Result<(), MeshError> {
    if !matches!(data.twist_mode.as_str(), "MINIMUM" | "Z_UP" | "TANGENT") {
        return Err(MeshError::InvalidParameter("curve twist mode is invalid"));
    }
    if !data.bevel_depth.is_finite() || data.bevel_depth < 0.0 {
        return Err(MeshError::InvalidParameter(
            "curve bevel depth must be finite and non-negative",
        ));
    }
    if !data.extrude.is_finite() || data.extrude < 0.0 {
        return Err(MeshError::InvalidParameter(
            "curve extrusion must be finite and non-negative",
        ));
    }
    if data.bevel_depth > 0.0
        && usize::try_from(data.bevel_resolution)
            .map_err(|_| MeshError::InvalidParameter("invalid bevel resolution"))?
            > MAX_BEVEL_RESOLUTION
    {
        return Err(MeshError::InvalidParameter(
            "bevel resolution must be between 0 and 32",
        ));
    }
    Ok(())
}
fn bevel_ring_count(resolution: u32) -> Result<usize, MeshError> {
    let resolution = usize::try_from(resolution)
        .map_err(|_| MeshError::InvalidParameter("invalid bevel resolution"))?;
    if resolution > MAX_BEVEL_RESOLUTION {
        return Err(MeshError::InvalidParameter(
            "bevel resolution must be between 0 and 32",
        ));
    }
    checked_add(checked_mul(resolution, 2)?, 4)
}

fn validate_spline(spline: &CurveSpline) -> Result<(), MeshError> {
    let point_count = spline.points.len();
    if point_count < 2 {
        return Err(MeshError::InvalidParameter(
            "each curve spline requires at least two control points",
        ));
    }
    for point in &spline.points {
        validate_curve_point(point)?;
    }
    let resolution = checked_resolution(spline.resolution, "curve spline resolution")?;
    match spline.spline_type {
        CurveSplineType::Poly | CurveSplineType::Bezier => {
            if spline.cyclic && point_count < 3 {
                return Err(MeshError::InvalidParameter(
                    "cyclic curve splines require at least three control points",
                ));
            }
        }
        CurveSplineType::Nurbs => {
            let order = checked_order(spline.order, point_count, "curve spline order")?;
            for point in &spline.points {
                if point.weight <= 0.0 {
                    return Err(MeshError::InvalidParameter(
                        "NURBS curve weights must be positive",
                    ));
                }
            }
            if spline.cyclic && point_count < order {
                return Err(MeshError::InvalidParameter(
                    "cyclic NURBS requires at least as many points as its order",
                ));
            }
        }
    }
    if resolution > MAX_RESOLUTION {
        return Err(MeshError::InvalidParameter(
            "curve spline resolution exceeds the tessellation limit",
        ));
    }
    Ok(())
}

fn validate_curve_point(point: &CurvePoint) -> Result<(), MeshError> {
    if point
        .co
        .iter()
        .chain(&point.handle_left)
        .chain(&point.handle_right)
        .any(|x| !x.is_finite())
        || !point.weight.is_finite()
        || !point.radius.is_finite()
        || !point.tilt.is_finite()
    {
        return Err(MeshError::InvalidParameter(
            "curve coordinates, handles, weights, radius, and tilt must be finite",
        ));
    }
    if point.radius < 0.0 {
        return Err(MeshError::InvalidParameter(
            "curve point radius must be non-negative",
        ));
    }
    Ok(())
}

fn validate_surface_point(point: &SurfacePoint) -> Result<(), MeshError> {
    if point.co.iter().any(|coordinate| !coordinate.is_finite()) || !point.weight.is_finite() {
        return Err(MeshError::InvalidParameter(
            "surface coordinates and weights must be finite",
        ));
    }
    if point.weight <= 0.0 {
        return Err(MeshError::InvalidParameter(
            "NURBS surface weights must be positive",
        ));
    }
    Ok(())
}

fn estimated_spline_samples(spline: &CurveSpline) -> Result<usize, MeshError> {
    let point_count = spline.points.len();
    let spans = match spline.spline_type {
        CurveSplineType::Poly | CurveSplineType::Bezier => {
            if spline.cyclic {
                point_count
            } else {
                point_count - 1
            }
        }
        CurveSplineType::Nurbs => {
            if spline.cyclic {
                point_count
            } else {
                point_count
                    - usize::try_from(spline.order)
                        .map_err(|_| MeshError::InvalidParameter("invalid curve spline order"))?
                    + 1
            }
        }
    };
    let samples = checked_mul(
        spans,
        usize::try_from(spline.resolution)
            .map_err(|_| MeshError::InvalidParameter("invalid curve spline resolution"))?,
    )?;
    let sample_count = if spline.cyclic {
        samples
    } else {
        checked_add(samples, 1)?
    };
    if sample_count > MAX_MESH_ELEMENTS {
        return Err(MeshError::InvalidParameter(
            "curve spline tessellation exceeds the mesh element limit",
        ));
    }
    Ok(sample_count)
}

fn checked_resolution(value: u32, name: &'static str) -> Result<usize, MeshError> {
    let resolution = usize::try_from(value).map_err(|_| MeshError::InvalidParameter(name))?;
    if resolution == 0 || resolution > MAX_RESOLUTION {
        return Err(MeshError::InvalidParameter(name));
    }
    Ok(resolution)
}

fn checked_order(value: u32, count: usize, name: &'static str) -> Result<usize, MeshError> {
    let order = usize::try_from(value).map_err(|_| MeshError::InvalidParameter(name))?;
    if !(2..=MAX_ORDER).contains(&order) || order > count {
        return Err(MeshError::InvalidParameter(name));
    }
    Ok(order)
}

fn checked_add(first: usize, second: usize) -> Result<usize, MeshError> {
    first.checked_add(second).ok_or(MeshError::InvalidParameter(
        "geometry tessellation size overflow",
    ))
}

fn checked_mul(first: usize, second: usize) -> Result<usize, MeshError> {
    first.checked_mul(second).ok_or(MeshError::InvalidParameter(
        "geometry tessellation size overflow",
    ))
}

fn check_mesh_budget(vertices: usize, edges: usize, faces: usize) -> Result<(), MeshError> {
    if checked_add(checked_add(vertices, edges)?, faces)? > MAX_MESH_ELEMENTS {
        return Err(MeshError::InvalidParameter(
            "evaluated geometry exceeds the mesh element limit",
        ));
    }
    Ok(())
}

fn fill_cap_count(mode: CurveFillMode) -> usize {
    match mode {
        CurveFillMode::Front | CurveFillMode::Back => 1,
        CurveFillMode::Both => 2,
        CurveFillMode::None => 0,
    }
}

fn tessellate_spline(spline: &CurveSpline) -> Result<Vec<CurveSample>, MeshError> {
    match spline.spline_type {
        CurveSplineType::Poly => tessellate_poly(spline),
        CurveSplineType::Bezier => tessellate_bezier(spline),
        CurveSplineType::Nurbs => tessellate_nurbs(spline),
    }
}

pub(crate) fn evaluate_curve_spline(
    spline: &CurveSpline,
    twist_mode: &str,
    with_orientation: bool,
) -> Result<Vec<CurvePathPoint>, MeshError> {
    validate_spline(spline)?;
    if estimated_spline_samples(spline)? > MAX_MESH_ELEMENTS {
        return Err(MeshError::InvalidParameter(
            "curve path exceeds the sampling limit",
        ));
    }
    let samples = tessellate_spline(spline)?;
    let frames = with_orientation
        .then(|| parallel_transport_frames(&samples, spline.cyclic, twist_mode))
        .transpose()?;
    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let orientation = if let Some(frames) = &frames {
                let frame = frames[index];
                let tangent = sample_tangent(&samples, index, spline.cyclic)?;
                let cosine = sample.tilt.cos();
                let sine = sample.tilt.sin();
                let normal = frame.normal * cosine + frame.binormal * sine;
                let binormal = frame.binormal * cosine - frame.normal * sine;
                DQuat::from_mat3(&DMat3::from_cols(normal, binormal, tangent))
            } else {
                DQuat::IDENTITY
            };
            Ok(CurvePathPoint {
                position: sample.position,
                radius: sample.radius,
                tilt: sample.tilt,
                orientation,
            })
        })
        .collect()
}

#[expect(
    clippy::cast_precision_loss,
    reason = "curve resolution is capped at 1024 steps per spline span"
)]
fn tessellate_poly(spline: &CurveSpline) -> Result<Vec<CurveSample>, MeshError> {
    let count = spline.points.len();
    let spans = if spline.cyclic { count } else { count - 1 };
    let resolution = usize::try_from(spline.resolution)
        .map_err(|_| MeshError::InvalidParameter("invalid curve spline resolution"))?;
    let mut samples = Vec::with_capacity(estimated_spline_samples(spline)?);
    for span in 0..spans {
        let first = &spline.points[span];
        let second = &spline.points[(span + 1) % count];
        for step in 0..resolution {
            let t = step as f64 / resolution as f64;
            let first_position = DVec3::from_array(first.co);
            let direction = DVec3::from_array(second.co) - first_position;
            samples.push(CurveSample {
                position: first_position.lerp(DVec3::from_array(second.co), t),
                tangent: direction,
                radius: first.radius + (second.radius - first.radius) * t,
                tilt: first.tilt + (second.tilt - first.tilt) * t,
            });
        }
    }
    if !spline.cyclic {
        samples.push(sample_from_point(&spline.points[count - 1]));
    }
    ensure_finite_samples(&samples)?;
    Ok(samples)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "curve resolution is capped at 1024 steps per spline span"
)]
fn tessellate_bezier(spline: &CurveSpline) -> Result<Vec<CurveSample>, MeshError> {
    let count = spline.points.len();
    let spans = if spline.cyclic { count } else { count - 1 };
    let resolution = usize::try_from(spline.resolution)
        .map_err(|_| MeshError::InvalidParameter("invalid curve spline resolution"))?;
    let mut samples = Vec::with_capacity(estimated_spline_samples(spline)?);
    for span in 0..spans {
        let first = &spline.points[span];
        let second = &spline.points[(span + 1) % count];
        let control = [
            DVec3::from_array(first.co),
            DVec3::from_array(first.handle_right),
            DVec3::from_array(second.handle_left),
            DVec3::from_array(second.co),
        ];
        for step in 0..resolution {
            let t = step as f64 / resolution as f64;
            let one_minus_t = 1.0 - t;
            let tangent = (control[1] - control[0]) * (3.0 * one_minus_t * one_minus_t)
                + (control[2] - control[1]) * (6.0 * one_minus_t * t)
                + (control[3] - control[2]) * (3.0 * t * t);
            samples.push(CurveSample {
                position: de_casteljau(control, t),
                tangent,
                radius: first.radius + (second.radius - first.radius) * t,
                tilt: first.tilt + (second.tilt - first.tilt) * t,
            });
        }
    }
    if !spline.cyclic {
        let last = &spline.points[count - 1];
        let mut sample = sample_from_point(last);
        let previous = &spline.points[count - 2];
        sample.tangent = DVec3::from_array(last.co) - DVec3::from_array(last.handle_left);
        if sample.tangent.length_squared() <= EPSILON_SQUARED {
            sample.tangent = DVec3::from_array(last.co) - DVec3::from_array(previous.co);
        }
        samples.push(sample);
    }
    ensure_finite_samples(&samples)?;
    Ok(samples)
}

fn de_casteljau(control: [DVec3; 4], parameter: f64) -> DVec3 {
    let mut work = control;
    for level in 1..4 {
        for index in 0..(4 - level) {
            work[index] = work[index].lerp(work[index + 1], parameter);
        }
    }
    work[0]
}

fn tessellate_nurbs(spline: &CurveSpline) -> Result<Vec<CurveSample>, MeshError> {
    let count = spline.points.len();
    let order = checked_order(spline.order, count, "curve spline order")?;
    let resolution = checked_resolution(spline.resolution, "curve spline resolution")?;
    let axis = make_axis(count, order, resolution, spline.use_endpoint, spline.cyclic)?;
    let mut controls: Vec<HomogeneousPoint> = spline
        .points
        .iter()
        .map(HomogeneousPoint::from_curve)
        .collect();
    if spline.cyclic {
        controls.extend(
            spline
                .points
                .iter()
                .take(order - 1)
                .map(HomogeneousPoint::from_curve),
        );
    }
    let mut samples = Vec::with_capacity(axis.samples.len());
    for parameter in &axis.samples {
        let point = de_boor(
            &controls,
            &axis.knots,
            axis.order,
            parameter.span,
            parameter.parameter,
        )?
        .divided()?;
        samples.push(CurveSample {
            position: point.position,
            tangent: DVec3::ZERO,
            radius: point.radius,
            tilt: point.tilt,
        });
    }
    ensure_finite_samples(&samples)?;
    Ok(samples)
}

fn sample_from_point(point: &CurvePoint) -> CurveSample {
    CurveSample {
        position: DVec3::from_array(point.co),
        tangent: DVec3::ZERO,
        radius: point.radius,
        tilt: point.tilt,
    }
}

fn ensure_finite_samples(samples: &[CurveSample]) -> Result<(), MeshError> {
    if samples.iter().any(|sample| {
        !sample.position.is_finite() || !sample.radius.is_finite() || !sample.tilt.is_finite()
    }) {
        return Err(MeshError::InvalidParameter(
            "curve tessellation produced non-finite coordinates or attributes",
        ));
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "control counts and resolutions are bounded below f64's exact integer range"
)]
fn make_axis(
    count: usize,
    order: usize,
    resolution: usize,
    use_endpoint: bool,
    cyclic: bool,
) -> Result<SplineAxis, MeshError> {
    let controls = if cyclic {
        checked_add(count, order - 1)?
    } else {
        count
    };
    let knot_count = checked_add(controls, order)?;
    let mut knots = Vec::with_capacity(knot_count);
    if use_endpoint && !cyclic {
        knots.resize(knot_count, 0.0);
        for knot in knots.iter_mut().take(knot_count).skip(controls) {
            *knot = 1.0;
        }
        let span_count = controls - order + 1;
        for (index, knot) in knots.iter_mut().enumerate().take(controls).skip(order) {
            *knot = (index - order + 1) as f64 / span_count as f64;
        }
    } else {
        knots.extend((0..knot_count).map(|index| index as f64));
    }

    let mut samples = Vec::new();
    let mut spans = 0_usize;
    for span in (order - 1)..controls {
        let low = knots[span];
        let high = knots[span + 1];
        if high <= low {
            continue;
        }
        spans += 1;
        for step in 0..resolution {
            let parameter = low + (high - low) * (step as f64 / resolution as f64);
            samples.push(ParameterSample { span, parameter });
        }
    }
    if spans == 0 {
        return Err(MeshError::InvalidParameter(
            "spline knot vector has no non-zero spans",
        ));
    }
    if !cyclic {
        let span = (order - 1..controls)
            .rev()
            .find(|span| knots[span + 1] > knots[*span])
            .ok_or(MeshError::InvalidParameter(
                "spline knot vector has no non-zero spans",
            ))?;
        samples.push(ParameterSample {
            span,
            parameter: knots[span + 1],
        });
    }
    Ok(SplineAxis {
        order,
        knots,
        samples,
    })
}

fn de_boor(
    controls: &[HomogeneousPoint],
    knots: &[f64],
    order: usize,
    span: usize,
    parameter: f64,
) -> Result<HomogeneousPoint, MeshError> {
    let degree = order - 1;
    let first = span - degree;
    let mut work = Vec::with_capacity(order);
    for offset in 0..order {
        work.push(
            *controls
                .get(first + offset)
                .ok_or(MeshError::InvalidTopology(
                    "spline knot span references a missing control point",
                ))?,
        );
    }
    for level in 1..order {
        for index in (level..order).rev() {
            let knot_index = span - degree + index;
            let denominator = knots[knot_index + degree - level + 1] - knots[knot_index];
            let factor = if denominator <= 0.0 {
                0.0
            } else {
                (parameter - knots[knot_index]) / denominator
            };
            work[index] = work[index - 1].interpolate(work[index], factor);
        }
    }
    work.get(order - 1)
        .copied()
        .ok_or(MeshError::InvalidTopology(
            "spline evaluation produced no control point",
        ))
}

fn insert_path_edges(
    mesh: &mut Mesh,
    samples: &[CurveSample],
    cyclic: bool,
) -> Result<(), MeshError> {
    let mut ids = Vec::with_capacity(samples.len());
    for sample in samples {
        ids.push(mesh.insert_vertex(sample.position)?);
    }
    let edge_count = if cyclic {
        ids.len()
    } else {
        ids.len().saturating_sub(1)
    };
    for index in 0..edge_count {
        mesh.insert_edge([ids[index], ids[(index + 1) % ids.len()]])?;
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "bevel ring count is capped at 68"
)]
fn insert_tube(
    mesh: &mut Mesh,
    samples: &[CurveSample],
    cyclic: bool,
    twist_mode: &str,
    depth: f64,
    ring_count: usize,
) -> Result<(), MeshError> {
    let frames = parallel_transport_frames(samples, cyclic, twist_mode)?;
    let mut rings = Vec::with_capacity(samples.len());
    for (sample, frame) in samples.iter().zip(frames) {
        let cosine = sample.tilt.cos();
        let sine = sample.tilt.sin();
        let normal = frame.normal * cosine + frame.binormal * sine;
        let binormal = frame.binormal * cosine - frame.normal * sine;
        let radius = depth * sample.radius;
        let mut ring = Vec::with_capacity(ring_count);
        for point in 0..ring_count {
            let angle = std::f64::consts::TAU * point as f64 / ring_count as f64;
            let offset = normal * angle.cos() + binormal * angle.sin();
            ring.push(mesh.insert_vertex(sample.position + offset * radius)?);
        }
        rings.push(ring);
    }
    let segment_count = if cyclic {
        samples.len()
    } else {
        samples.len().saturating_sub(1)
    };
    for segment in 0..segment_count {
        let next_segment = (segment + 1) % samples.len();
        for point in 0..ring_count {
            let next_point = (point + 1) % ring_count;
            mesh.insert_face(
                vec![
                    rings[segment][point],
                    rings[segment][next_point],
                    rings[next_segment][next_point],
                    rings[next_segment][point],
                ],
                0,
            )?;
        }
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "tessellated sample counts are bounded by the mesh-element limit"
)]
fn parallel_transport_frames(
    samples: &[CurveSample],
    cyclic: bool,
    twist_mode: &str,
) -> Result<Vec<Frame>, MeshError> {
    if samples.len() < 2 {
        return Err(MeshError::InvalidParameter(
            "a bevel sweep requires at least two path samples",
        ));
    }
    let tangents: Vec<DVec3> = (0..samples.len())
        .map(|index| sample_tangent(samples, index, cyclic))
        .collect::<Result<_, _>>()?;
    if twist_mode == "Z_UP" || twist_mode == "TANGENT" {
        return tangents.iter().copied().map(blender_curve_frame).collect();
    }
    if twist_mode != "MINIMUM" {
        return Err(MeshError::InvalidParameter("curve twist mode is invalid"));
    }
    let count = samples.len();
    let mut normals = vec![DVec3::ZERO; count];
    if cyclic {
        normals[count - 1] = blender_curve_frame(tangents[count - 1])?.normal;
        normals[0] = transport_normal(normals[count - 1], tangents[count - 1], tangents[0])?;
        for index in 1..count - 1 {
            normals[index] =
                transport_normal(normals[index - 1], tangents[index - 1], tangents[index])?;
        }
        let closure_normal =
            transport_normal(normals[count - 2], tangents[count - 2], tangents[count - 1])?;
        let seam_angle = signed_angle(closure_normal, normals[count - 1], tangents[count - 1]);
        for index in 1..count {
            normals[index] = rotate_about_axis(
                normals[index],
                tangents[index],
                seam_angle * index as f64 / count as f64,
            );
        }
    } else {
        normals[0] = blender_curve_frame(tangents[0])?.normal;
        for index in 1..count {
            normals[index] =
                transport_normal(normals[index - 1], tangents[index - 1], tangents[index])?;
        }
    }
    Ok(tangents
        .into_iter()
        .zip(normals)
        .map(|(tangent, normal)| Frame {
            normal,
            binormal: tangent.cross(normal),
        })
        .collect())
}

fn blender_curve_frame(tangent: DVec3) -> Result<Frame, MeshError> {
    let mut axis = DVec3::new(-tangent.y, tangent.x, 0.0);
    if axis.length_squared() <= EPSILON_SQUARED {
        axis = DVec3::X;
    }
    axis = safe_normalize(axis)?;
    let align_angle = tangent.z.clamp(-1.0, 1.0).acos();
    let align = DQuat::from_axis_angle(axis, align_angle);
    let roll_angle = -(-tangent.x).atan2(-tangent.y);
    let roll = DQuat::from_axis_angle(tangent, roll_angle);
    let orientation = (roll * align).normalize();
    Ok(Frame {
        normal: orientation * DVec3::X,
        binormal: orientation * DVec3::Y,
    })
}

fn sample_tangent(samples: &[CurveSample], index: usize, cyclic: bool) -> Result<DVec3, MeshError> {
    let count = samples.len();
    let center = samples[index].position;
    let before = if index > 0 {
        Some(samples[index - 1].position)
    } else if cyclic {
        Some(samples[count - 1].position)
    } else {
        None
    };
    let after = if index + 1 < count {
        Some(samples[index + 1].position)
    } else if cyclic {
        Some(samples[0].position)
    } else {
        None
    };
    let endpoint = !cyclic && (index == 0 || index + 1 == count);
    if endpoint && samples[index].tangent.length_squared() > EPSILON_SQUARED {
        return safe_normalize(samples[index].tangent);
    }
    if let (Some(before), Some(after)) = (before, after) {
        let incoming = safe_normalize(center - before)?;
        let outgoing = safe_normalize(after - center)?;
        if let Ok(bisector) = safe_normalize(incoming + outgoing) {
            return Ok(bisector);
        }
    }
    let candidates = [
        samples[index].tangent,
        after.map_or(DVec3::ZERO, |point| point - center),
        before.map_or(DVec3::ZERO, |point| center - point),
    ];
    for candidate in candidates {
        if let Ok(tangent) = safe_normalize(candidate) {
            return Ok(tangent);
        }
    }
    Err(MeshError::InvalidParameter(
        "bevel path contains no usable tangent",
    ))
}

fn safe_normalize(vector: DVec3) -> Result<DVec3, MeshError> {
    if !vector.is_finite() {
        return Err(MeshError::InvalidParameter(
            "curve geometry contains an unbounded direction",
        ));
    }
    let scale = vector.abs().max_element();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(MeshError::InvalidParameter(
            "bevel path contains a zero-length direction",
        ));
    }
    let scaled = vector / scale;
    let length_squared = scaled.length_squared();
    if length_squared <= EPSILON_SQUARED || !length_squared.is_finite() {
        return Err(MeshError::InvalidParameter(
            "bevel path contains a zero-length direction",
        ));
    }
    Ok(scaled / length_squared.sqrt())
}

fn transport_normal(normal: DVec3, from: DVec3, to: DVec3) -> Result<DVec3, MeshError> {
    let cross = from.cross(to);
    let sine = cross.length();
    let cosine = from.dot(to).clamp(-1.0, 1.0);
    let rotated = if sine > 1.0e-12 {
        let axis = cross / sine;
        let angle = sine.atan2(cosine);
        normal * angle.cos()
            + axis.cross(normal) * angle.sin()
            + axis * axis.dot(normal) * (1.0 - angle.cos())
    } else {
        normal
    };
    let projected = rotated - to * rotated.dot(to);
    safe_normalize(projected)
}

fn signed_angle(from: DVec3, to: DVec3, axis: DVec3) -> f64 {
    axis.dot(from.cross(to)).atan2(from.dot(to))
}

fn rotate_about_axis(vector: DVec3, axis: DVec3, angle: f64) -> DVec3 {
    vector * angle.cos()
        + axis.cross(vector) * angle.sin()
        + axis * axis.dot(vector) * (1.0 - angle.cos())
}

fn insert_fill(
    mesh: &mut Mesh,
    samples: &[CurveSample],
    mode: CurveFillMode,
    extrude: f64,
) -> Result<(), MeshError> {
    let mut contour: Vec<DVec3> = samples.iter().map(|sample| sample.position).collect();
    validate_fill_contour(&contour)?;
    if signed_area_xy(&contour)? < 0.0 {
        contour.reverse();
    }
    let base_z = contour[0].z;
    if extrude > 0.0 {
        let mut bottom = Vec::with_capacity(contour.len());
        let mut top = Vec::with_capacity(contour.len());
        for position in &contour {
            bottom.push(mesh.insert_vertex(DVec3::new(
                position.x,
                position.y,
                position.z - extrude,
            ))?);
            top.push(mesh.insert_vertex(DVec3::new(
                position.x,
                position.y,
                position.z + extrude,
            ))?);
        }
        for index in 0..contour.len() {
            let next = (index + 1) % contour.len();
            mesh.insert_face(vec![bottom[index], bottom[next], top[next], top[index]], 0)?;
        }
        if matches!(mode, CurveFillMode::Front | CurveFillMode::Both) {
            mesh.insert_face(top, 0)?;
        }
        if matches!(mode, CurveFillMode::Back | CurveFillMode::Both) {
            bottom.reverse();
            mesh.insert_face(bottom, 0)?;
        }
    } else {
        let mut ids = Vec::with_capacity(contour.len());
        for position in contour {
            ids.push(mesh.insert_vertex(position)?);
        }
        if matches!(mode, CurveFillMode::Front | CurveFillMode::Both) {
            mesh.insert_face(ids.clone(), 0)?;
        }
        if matches!(mode, CurveFillMode::Back | CurveFillMode::Both) {
            ids.reverse();
            mesh.insert_face(ids, 0)?;
        }
    }
    let _ = base_z;
    Ok(())
}

fn validate_fill_contour(contour: &[DVec3]) -> Result<(), MeshError> {
    if contour.len() < 3 || contour.len() > MAX_FILL_POINTS {
        return Err(MeshError::InvalidParameter(
            "2D fill contour must contain between three and 4096 samples",
        ));
    }
    let scale = contour
        .iter()
        .map(|point| point.abs().max_element())
        .fold(1.0_f64, f64::max);
    let reference_z = contour[0].z / scale;
    if contour
        .iter()
        .any(|point| (point.z / scale - reference_z).abs() > 1.0e-9)
    {
        return Err(MeshError::InvalidParameter(
            "2D curve fill requires a planar contour parallel to the XY plane",
        ));
    }
    let normalized: Vec<[f64; 2]> = contour
        .iter()
        .map(|point| [point.x / scale, point.y / scale])
        .collect();
    for first in 0..normalized.len() {
        let first_next = (first + 1) % normalized.len();
        for second in (first + 1)..normalized.len() {
            let second_next = (second + 1) % normalized.len();
            if first == second || first_next == second || second_next == first {
                continue;
            }
            if segments_intersect(
                normalized[first],
                normalized[first_next],
                normalized[second],
                normalized[second_next],
            ) {
                return Err(MeshError::InvalidParameter(
                    "2D fill contours must not self-intersect",
                ));
            }
        }
    }
    if crate::float::equal_f64(signed_area_xy(contour)?, 0.0) {
        return Err(MeshError::InvalidParameter(
            "2D fill contour must enclose a non-zero area",
        ));
    }
    Ok(())
}

fn segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    fn orient(first: [f64; 2], second: [f64; 2], third: [f64; 2]) -> f64 {
        (second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0])
    }
    let first = orient(a, b, c);
    let second = orient(a, b, d);
    let third = orient(c, d, a);
    let fourth = orient(c, d, b);
    (crate::float::equal_f64(first, 0.0)
        || crate::float::equal_f64(second, 0.0)
        || first.is_sign_positive() != second.is_sign_positive())
        && (crate::float::equal_f64(third, 0.0)
            || crate::float::equal_f64(fourth, 0.0)
            || third.is_sign_positive() != fourth.is_sign_positive())
}

fn signed_area_xy(contour: &[DVec3]) -> Result<f64, MeshError> {
    let origin = contour[0];
    let mut twice_area = 0.0;
    for index in 0..contour.len() {
        let current = contour[index] - origin;
        let next = contour[(index + 1) % contour.len()] - origin;
        twice_area += current.x * next.y - current.y * next.x;
    }
    if !twice_area.is_finite() {
        return Err(MeshError::InvalidParameter(
            "2D fill contour area is not finite",
        ));
    }
    Ok(twice_area * 0.5)
}
