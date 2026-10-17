// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/modifiers/intern/MOD_laplaciansmooth.cc, source/blender/modifiers/intern/MOD_correctivesmooth.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

use super::{
    DVec3, ErrorCode, HashMap, Mesh, Modifier, PotError, Result, Value, bool_param, id_indices,
    invalid_parameter, json, number_param, positions, safe_acos_approx, string_param_with_expected,
    uint_param, validated,
};

/// Blender's `FLT_EPSILON`; both smoothers compare against it in single precision.
const FLT_EPSILON: f64 = f32::EPSILON as f64;
/// Blender's `min_area` for degenerate Laplacian Smooth edges and corners.
const LAPLACIAN_MIN_AREA: f64 = 1.0e-5;
/// Largest `iterations` value the schema accepts for both smoothers.
const MAX_ITERATIONS: u64 = 32_767;
/// Bind data format written by the corrective smooth bind operation.
const CORRECTIVE_BIND_FORMAT: &str = "blender_corrective_bind_v1";

/// Vertex-index topology shared by both smoothers, in mesh storage order.
struct Topology {
    edges: Vec<[usize; 2]>,
    faces: Vec<Vec<usize>>,
}

impl Topology {
    fn new(mesh: &Mesh, modifier: &Modifier) -> Result<Self> {
        let index_by_id = id_indices(mesh);
        let index = |id: u32| -> Result<usize> {
            index_by_id.get(&id).copied().ok_or_else(|| {
                invalid_parameter(
                    modifier,
                    "mesh",
                    "edges and faces referencing existing vertices",
                )
            })
        };
        let edges = mesh
            .edges
            .iter()
            .map(|edge| -> Result<[usize; 2]> {
                Ok([index(edge.vertices[0])?, index(edge.vertices[1])?])
            })
            .collect::<Result<Vec<_>>>()?;
        let faces = mesh
            .faces
            .iter()
            .map(|face| {
                face.vertices
                    .iter()
                    .map(|id| index(*id))
                    .collect::<Result<Vec<usize>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { edges, faces })
    }

    /// Blender's `mesh_get_boundaries`: zero the weight of every vertex on a boundary edge, an
    /// edge used by exactly one face.
    fn pin_boundary(&self, weights: &mut [f64]) {
        let mut face_uses: HashMap<[usize; 2], usize> = HashMap::new();
        for face in &self.faces {
            for (corner, &vertex) in face.iter().enumerate() {
                let next = face[(corner + 1) % face.len()];
                *face_uses
                    .entry([vertex.min(next), vertex.max(next)])
                    .or_default() += 1;
            }
        }
        for edge in &self.edges {
            let key = [edge[0].min(edge[1]), edge[0].max(edge[1])];
            if face_uses.get(&key) == Some(&1) {
                weights[edge[0]] = 0.0;
                weights[edge[1]] = 0.0;
            }
        }
    }
}

/// Blender's per-vertex group factor: 1 without a usable group, otherwise the weight, inverted when
/// requested. A group name that is not on the mesh is ignored, as `MOD_get_vgroup` does.
fn vertex_group_factors(mesh: &Mesh, modifier: &Modifier) -> Result<Vec<f64>> {
    let group_name = string_param_with_expected(modifier, "vertex_group", "a vertex-group name")?;
    let weights = if group_name.is_empty() {
        None
    } else {
        mesh.attributes
            .get("vertex_groups")
            .and_then(Value::as_object)
            .and_then(|groups| groups.get(group_name))
            .and_then(Value::as_object)
    };
    let Some(weights) = weights else {
        return Ok(vec![1.0; mesh.vertices.len()]);
    };
    let invert = bool_param(modifier, "invert_vertex_group")?;
    mesh.vertices
        .iter()
        .map(|vertex| {
            let weight = weights
                .get(&format!("v{}", vertex.id))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if !(weight.is_finite() && (0.0..=1.0).contains(&weight)) {
                return Err(invalid_parameter(
                    modifier,
                    "vertex_group",
                    "finite weights in [0, 1]",
                ));
            }
            Ok(if invert { 1.0 - weight } else { weight })
        })
        .collect()
}

fn iteration_count(modifier: &Modifier) -> Result<usize> {
    let iterations = uint_param(modifier, "iterations")?;
    let expected = "an integer from 0 to 32767";
    if iterations > MAX_ITERATIONS {
        return Err(invalid_parameter(modifier, "iterations", expected));
    }
    usize::try_from(iterations).map_err(|_| invalid_parameter(modifier, "iterations", expected))
}

fn with_positions(mesh: &Mesh, points: &[DVec3]) -> Result<Mesh> {
    let mut result = mesh.clone();
    for (vertex, point) in result.vertices.iter_mut().zip(points) {
        vertex.co = *point;
    }
    validated(&result)?;
    Ok(result)
}

#[derive(Clone, Copy)]
struct LaplacianSettings {
    iterations: usize,
    lambda: f64,
    lambda_border: f64,
    normalized: bool,
    preserve_volume: bool,
    axes: [bool; 3],
}

pub(super) fn laplacian_smooth(mesh: &Mesh, modifier: &Modifier) -> Result<Mesh> {
    let settings = LaplacianSettings {
        iterations: iteration_count(modifier)?,
        lambda: number_param(modifier, "lambda_factor")?,
        lambda_border: number_param(modifier, "lambda_border")?,
        normalized: bool_param(modifier, "use_normalized")?,
        preserve_volume: bool_param(modifier, "use_volume_preserve")?,
        axes: [
            bool_param(modifier, "use_x")?,
            bool_param(modifier, "use_y")?,
            bool_param(modifier, "use_z")?,
        ],
    };
    let factors = vertex_group_factors(mesh, modifier)?;
    let no_axes = !settings.axes.iter().any(|enabled| *enabled);
    if settings.iterations == 0 || mesh.vertices.is_empty() || no_axes {
        return Ok(mesh.clone());
    }
    let topology = Topology::new(mesh, modifier)?;
    let smoothed = laplacian_points(&topology, &positions(mesh), &factors, settings);
    with_positions(mesh, &smoothed)
}

/// Blender's `laplaciansmoothModifier_do`: builds the cotangent Laplacian once, then solves it per
/// iteration and moves the enabled axes of the non-degenerate vertices toward the solution.
fn laplacian_points(
    topology: &Topology,
    points: &[DVec3],
    factors: &[f64],
    settings: LaplacianSettings,
) -> Vec<DVec3> {
    let vertex_count = points.len();
    let mut edge_count = vec![0_usize; vertex_count];
    let mut face_count = vec![0_usize; vertex_count];
    let mut degenerate = vec![false; vertex_count];
    let mut ring_area = vec![0.0; vertex_count];
    let mut vertex_weight = vec![0.0; vertex_count];
    let mut vertex_length = vec![0.0; vertex_count];
    let mut edge_weight = Vec::with_capacity(topology.edges.len());
    for edge in &topology.edges {
        let (first, second) = (edge[0], edge[1]);
        edge_count[first] += 1;
        edge_count[second] += 1;
        let length = points[first].distance(points[second]);
        if length < LAPLACIAN_MIN_AREA {
            degenerate[first] = true;
            degenerate[second] = true;
            edge_weight.push(length);
        } else {
            edge_weight.push(length.recip());
        }
    }

    let mut corner_weights = Vec::with_capacity(topology.faces.len());
    for face in &topology.faces {
        let count = face.len();
        let mut weights = Vec::with_capacity(count);
        for (corner, &current) in face.iter().enumerate() {
            let previous = face[(corner + count - 1) % count];
            let next = face[(corner + 1) % count];
            let (before, middle, after) = (points[previous], points[current], points[next]);
            face_count[current] += 1;
            let area = (middle - before).cross(after - before).length() * 0.5;
            if area < LAPLACIAN_MIN_AREA {
                degenerate[current] = true;
            }
            ring_area[previous] += area;
            ring_area[current] += area;
            ring_area[next] += area;
            let w1 = cotangent_weight(middle, after, before) * 0.5;
            let w2 = cotangent_weight(after, before, middle) * 0.5;
            let w3 = cotangent_weight(before, middle, after) * 0.5;
            vertex_weight[current] += w2 + w3;
            vertex_weight[next] += w1 + w3;
            vertex_weight[previous] += w1 + w2;
            weights.push([w1, w2, w3]);
        }
        corner_weights.push(weights);
    }
    for (edge, weight) in topology.edges.iter().zip(&edge_weight) {
        let (first, second) = (edge[0], edge[1]);
        if edge_count[first] != face_count[first] && edge_count[second] != face_count[second] {
            vertex_length[first] += weight;
            vertex_length[second] += weight;
        }
    }

    let abs_lambda = settings.lambda.abs();
    let abs_border = settings.lambda_border.abs();
    let mut rows = vec![Vec::<(usize, f64)>::new(); vertex_count];
    let mut row_weight = vec![0.0; vertex_count];
    let mut border_weight = vec![0.0; vertex_count];
    for (index, &factor) in factors.iter().enumerate() {
        if degenerate[index] {
            laplacian_matrix_add(&mut rows, index, index, 1.0);
        } else if edge_count[index] == face_count[index] {
            let (weight, diagonal) = if settings.normalized {
                let total = vertex_weight[index];
                let weight = if total == 0.0 {
                    0.0
                } else {
                    -abs_lambda * factor / total
                };
                (weight, 1.0 + abs_lambda * factor)
            } else {
                let total = vertex_weight[index] * ring_area[index];
                let weight = if total == 0.0 {
                    0.0
                } else {
                    -abs_lambda * factor / (4.0 * total)
                };
                let diagonal = if ring_area[index] > 0.0 {
                    abs_lambda * factor / (4.0 * ring_area[index])
                } else {
                    0.0
                };
                (weight, 1.0 + diagonal)
            };
            row_weight[index] = weight;
            laplacian_matrix_add(&mut rows, index, index, diagonal);
        } else {
            let length = vertex_length[index];
            border_weight[index] = if length == 0.0 {
                0.0
            } else {
                -abs_border * factor * 2.0 / length
            };
            let diagonal = 1.0 + abs_border * factor * 2.0;
            laplacian_matrix_add(&mut rows, index, index, diagonal);
        }
    }

    let is_ring = |vertex: usize| edge_count[vertex] == face_count[vertex] && !degenerate[vertex];
    for (face, weights) in topology.faces.iter().zip(&corner_weights) {
        let count = face.len();
        for (corner, &[w1, w2, w3]) in weights.iter().enumerate() {
            let previous = face[(corner + count - 1) % count];
            let current = face[corner];
            let next = face[(corner + 1) % count];
            if is_ring(current) {
                laplacian_matrix_add(&mut rows, current, next, w3 * row_weight[current]);
                laplacian_matrix_add(&mut rows, current, previous, w2 * row_weight[current]);
            }
            if is_ring(next) {
                laplacian_matrix_add(&mut rows, next, current, w3 * row_weight[next]);
                laplacian_matrix_add(&mut rows, next, previous, w1 * row_weight[next]);
            }
            if is_ring(previous) {
                laplacian_matrix_add(&mut rows, previous, next, w1 * row_weight[previous]);
                laplacian_matrix_add(&mut rows, previous, current, w2 * row_weight[previous]);
            }
        }
    }
    for (edge, weight) in topology.edges.iter().zip(&edge_weight) {
        let (first, second) = (edge[0], edge[1]);
        if edge_count[first] != face_count[first]
            && edge_count[second] != face_count[second]
            && !degenerate[first]
            && !degenerate[second]
        {
            laplacian_matrix_add(&mut rows, first, second, weight * border_weight[first]);
            laplacian_matrix_add(&mut rows, second, first, weight * border_weight[second]);
        }
    }

    let centroid = points.iter().copied().sum::<DVec3>() / vertex_count as f64;
    let mut current = points.to_vec();
    let mut solver = SparseLaplacianSolver::new(rows, vertex_count);
    let mut right_hand_side = vec![0.0; vertex_count];
    for _ in 0..settings.iterations {
        let initial_volume = if settings.preserve_volume {
            mesh_volume(centroid, &current, &topology.faces)
        } else {
            0.0
        };
        let mut solutions = Vec::with_capacity(3);
        for axis in 0..3 {
            for (slot, point) in right_hand_side.iter_mut().zip(&current) {
                *slot = point[axis];
            }
            if !solver.solve(&right_hand_side) {
                break;
            }
            solutions.push(solver.solution.clone());
        }
        // Blender validates the solution only when all three axes solved.
        if solutions.len() != 3 {
            continue;
        }
        for (index, point) in current.iter_mut().enumerate() {
            if degenerate[index] {
                continue;
            }
            let lambda = if edge_count[index] == face_count[index] {
                settings.lambda
            } else {
                settings.lambda_border
            };
            let direction = if lambda >= 0.0 { 1.0 } else { -1.0 };
            for (axis, enabled) in settings.axes.into_iter().enumerate() {
                if enabled {
                    point[axis] += direction * (solutions[axis][index] - point[axis]);
                }
            }
        }
        if settings.preserve_volume {
            let final_volume = mesh_volume(centroid, &current, &topology.faces);
            if final_volume != 0.0 {
                let beta = (initial_volume / final_volume).cbrt();
                for point in &mut current {
                    for (axis, enabled) in settings.axes.into_iter().enumerate() {
                        if enabled {
                            point[axis] = (point[axis] - centroid[axis]) * beta + centroid[axis];
                        }
                    }
                }
            }
        }
    }
    current
}

/// Blender's `cotangent_tri_weight_v3`: cotangent of the angle at `first`.
fn cotangent_weight(first: DVec3, second: DVec3, third: DVec3) -> f64 {
    let first_edge = second - first;
    let second_edge = third - first;
    let area = first_edge.cross(second_edge).length();
    if area > FLT_EPSILON {
        first_edge.dot(second_edge) / area
    } else {
        0.0
    }
}

fn laplacian_matrix_add(rows: &mut [Vec<(usize, f64)>], row: usize, column: usize, value: f64) {
    if let Some((_, current)) = rows[row].iter_mut().find(|(index, _)| *index == column) {
        *current += value;
    } else {
        rows[row].push((column, value));
    }
}

/// Blender's `compute_volume`: absolute signed volume of the fan tetrahedra about `center`.
fn mesh_volume(center: DVec3, points: &[DVec3], faces: &[Vec<usize>]) -> f64 {
    let mut volume = 0.0;
    for face in faces {
        let first = points[face[0]] - center;
        for index in 1..face.len().saturating_sub(1) {
            let previous = points[face[index]] - center;
            let current = points[face[index + 1]] - center;
            volume += first.dot(previous.cross(current)) / 6.0;
        }
    }
    volume.abs()
}

struct SparseLaplacianSolver {
    rows: Vec<Vec<(usize, f64)>>,
    solution: Vec<f64>,
    residual: Vec<f64>,
    shadow_residual: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
    intermediate: Vec<f64>,
    second_product: Vec<f64>,
}

impl SparseLaplacianSolver {
    fn new(rows: Vec<Vec<(usize, f64)>>, vertex_count: usize) -> Self {
        Self {
            rows,
            solution: vec![0.0; vertex_count],
            residual: vec![0.0; vertex_count],
            shadow_residual: vec![0.0; vertex_count],
            direction: vec![0.0; vertex_count],
            product: vec![0.0; vertex_count],
            intermediate: vec![0.0; vertex_count],
            second_product: vec![0.0; vertex_count],
        }
    }

    fn solve(&mut self, right_hand_side: &[f64]) -> bool {
        self.solution.copy_from_slice(right_hand_side);
        sparse_matrix_vector(&self.rows, &self.solution, &mut self.product);
        for (index, &right_hand_side) in right_hand_side.iter().enumerate() {
            self.residual[index] = right_hand_side - self.product[index];
            self.shadow_residual[index] = self.residual[index];
        }
        let tolerance = 1.0e-11 * sparse_norm(right_hand_side).max(1.0);
        if sparse_norm(&self.residual) <= tolerance {
            return true;
        }
        let mut previous_rho = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        // A breakdown is an inner product that vanished. These products shrink with the residual
        // squared, so an absolute epsilon would abort a converging solve before `tolerance`.
        let max_iterations = self.solution.len().saturating_mul(4).max(32);
        for iteration in 0..max_iterations {
            let rho = sparse_dot(&self.shadow_residual, &self.residual);
            if rho.abs() <= f64::MIN_POSITIVE {
                return false;
            }
            if iteration == 0 {
                self.direction.copy_from_slice(&self.residual);
            } else {
                let beta = (rho / previous_rho) * (alpha / omega);
                for index in 0..self.direction.len() {
                    self.direction[index] = self.residual[index]
                        + beta * (self.direction[index] - omega * self.product[index]);
                }
            }
            sparse_matrix_vector(&self.rows, &self.direction, &mut self.product);
            let divisor = sparse_dot(&self.shadow_residual, &self.product);
            if divisor.abs() <= f64::MIN_POSITIVE {
                return false;
            }
            alpha = rho / divisor;
            for index in 0..self.intermediate.len() {
                self.intermediate[index] = self.residual[index] - alpha * self.product[index];
            }
            if sparse_norm(&self.intermediate) <= tolerance {
                for index in 0..self.solution.len() {
                    self.solution[index] += alpha * self.direction[index];
                }
                return true;
            }
            sparse_matrix_vector(&self.rows, &self.intermediate, &mut self.second_product);
            let denominator = sparse_dot(&self.second_product, &self.second_product);
            if denominator <= f64::MIN_POSITIVE {
                return false;
            }
            omega = sparse_dot(&self.second_product, &self.intermediate) / denominator;
            if omega.abs() <= f64::MIN_POSITIVE {
                return false;
            }
            for index in 0..self.solution.len() {
                self.solution[index] +=
                    alpha * self.direction[index] + omega * self.intermediate[index];
                self.residual[index] =
                    self.intermediate[index] - omega * self.second_product[index];
            }
            if sparse_norm(&self.residual) <= tolerance {
                return true;
            }
            previous_rho = rho;
        }
        false
    }
}

fn sparse_matrix_vector(rows: &[Vec<(usize, f64)>], input: &[f64], output: &mut [f64]) {
    for (row_index, row) in rows.iter().enumerate() {
        output[row_index] = row
            .iter()
            .map(|(column, coefficient)| coefficient * input[*column])
            .sum();
    }
}

fn sparse_dot(first: &[f64], second: &[f64]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(first, second)| first * second)
        .sum()
}

fn sparse_norm(vector: &[f64]) -> f64 {
    sparse_dot(vector, vector).sqrt()
}

#[derive(Clone, Copy)]
struct Smoothing {
    iterations: usize,
    factor: f64,
    length_weighted: bool,
}

/// Blender's `smooth_iter` (simple and edge-length weighted). Every vertex moves from the positions
/// at the start of its iteration, so the edge deltas never see partially updated neighbors.
fn smooth_points(
    topology: &Topology,
    points: &[DVec3],
    weights: &[f64],
    smoothing: Smoothing,
) -> Vec<DVec3> {
    let vertex_count = points.len();
    let mut current = points.to_vec();
    let mut edge_count = vec![0.0_f64; vertex_count];
    for edge in &topology.edges {
        edge_count[edge[0]] += 1.0;
        edge_count[edge[1]] += 1.0;
    }
    if smoothing.length_weighted {
        let lambda = smoothing.factor * 2.0;
        for _ in 0..smoothing.iterations {
            let mut delta = vec![DVec3::ZERO; vertex_count];
            let mut length_sum = vec![0.0; vertex_count];
            for edge in &topology.edges {
                let (first, second) = (edge[0], edge[1]);
                let direction = current[second] - current[first];
                let length = direction.length();
                let weighted = direction * length;
                delta[first] += weighted;
                delta[second] -= weighted;
                length_sum[first] += length;
                length_sum[second] += length;
            }
            for (index, point) in current.iter_mut().enumerate() {
                let divisor = length_sum[index] * edge_count[index];
                if divisor > FLT_EPSILON * 10.0 {
                    *point += delta[index] * (lambda * weights[index] / divisor);
                }
            }
        }
    } else {
        let divisors: Vec<f64> = weights
            .iter()
            .zip(&edge_count)
            .map(|(weight, count)| {
                let inverse_count = if *count > 0.0 { 1.0 / count } else { 1.0 };
                weight * smoothing.factor * inverse_count
            })
            .collect();
        for _ in 0..smoothing.iterations {
            let mut delta = vec![DVec3::ZERO; vertex_count];
            for edge in &topology.edges {
                let (first, second) = (edge[0], edge[1]);
                let direction = current[second] - current[first];
                delta[first] += direction;
                delta[second] -= direction;
            }
            for ((point, step), divisor) in current.iter_mut().zip(&delta).zip(&divisors) {
                *point += *step * *divisor;
            }
        }
    }
    current
}

/// Blender's `calc_tangent_loop`: orthonormal frame of one face corner from its two unit edge
/// directions, plus the corner's fan weight. Degenerate corners get the identity frame and weight 0.
#[derive(Clone, Copy)]
struct CornerFrame {
    axes: [DVec3; 3],
    weight: f64,
}

impl CornerFrame {
    fn to_local(self, offset: DVec3) -> DVec3 {
        DVec3::new(
            self.axes[0].dot(offset),
            self.axes[1].dot(offset),
            self.axes[2].dot(offset),
        )
    }

    fn to_world(self, local: DVec3) -> DVec3 {
        self.axes[0] * local.x + self.axes[1] * local.y + self.axes[2] * local.z
    }
}

struct TangentFrames {
    corners: Vec<Vec<CornerFrame>>,
    /// Blender's `tangent_weights_per_vertex`.
    vertex_weight: Vec<f64>,
}

/// Unit vector, or zero when the input is too short to normalize (Blender's `normalize_v3`).
fn unit_or_zero(vector: DVec3) -> DVec3 {
    let length_squared = vector.length_squared();
    if length_squared > 1.0e-35 {
        vector / length_squared.sqrt()
    } else {
        DVec3::ZERO
    }
}

/// Blender's `calc_tangent_loop`. `previous` is the unit direction from the corner to its
/// predecessor; `next` is the unit direction from the successor to the corner.
fn corner_frame(previous: DVec3, next: DVec3) -> CornerFrame {
    if (previous - next).abs().max_element() < FLT_EPSILON * 10.0 {
        return CornerFrame {
            axes: [DVec3::X, DVec3::Y, DVec3::Z],
            weight: 0.0,
        };
    }
    let normal = unit_or_zero(previous.cross(next));
    let tangent = unit_or_zero(previous + next);
    let bitangent = normal.cross(tangent);
    CornerFrame {
        axes: [tangent, bitangent, normal],
        weight: safe_acos_approx(next.dot(previous)).abs(),
    }
}

/// Blender's `calc_tangent_spaces` for every face corner, with the per-vertex weight sums.
fn tangent_frames(faces: &[Vec<usize>], points: &[DVec3]) -> TangentFrames {
    let mut corners = Vec::with_capacity(faces.len());
    let mut vertex_weight = vec![0.0; points.len()];
    for face in faces {
        let count = face.len();
        let mut frames = Vec::with_capacity(count);
        for (corner, &current) in face.iter().enumerate() {
            let previous = face[(corner + count - 1) % count];
            let next = face[(corner + 1) % count];
            let frame = corner_frame(
                unit_or_zero(points[previous] - points[current]),
                unit_or_zero(points[current] - points[next]),
            );
            vertex_weight[current] += frame.weight;
            frames.push(frame);
        }
        corners.push(frames);
    }
    TangentFrames {
        corners,
        vertex_weight,
    }
}

pub(super) fn corrective_smooth(
    mesh: &Mesh,
    rest_mesh: &Mesh,
    modifier: &Modifier,
) -> Result<Mesh> {
    let length_weighted =
        match string_param_with_expected(modifier, "smooth_type", "SIMPLE or LENGTH_WEIGHTED")? {
            "SIMPLE" => false,
            "LENGTH_WEIGHTED" => true,
            _ => {
                return Err(invalid_parameter(
                    modifier,
                    "smooth_type",
                    "SIMPLE or LENGTH_WEIGHTED",
                ));
            }
        };
    let smoothing = Smoothing {
        iterations: iteration_count(modifier)?,
        factor: number_param(modifier, "factor")?,
        length_weighted,
    };
    let scale = number_param(modifier, "scale")?;
    let only_smooth = bool_param(modifier, "use_only_smooth")?;
    let topology = Topology::new(mesh, modifier)?;
    let mut weights = vertex_group_factors(mesh, modifier)?;
    if bool_param(modifier, "use_pin_boundary")? {
        topology.pin_boundary(&mut weights);
    }
    let input = positions(mesh);
    let smoothed = smooth_points(&topology, &input, &weights, smoothing);
    if only_smooth {
        return with_positions(mesh, &smoothed);
    }

    let rest = rest_positions(modifier, mesh, rest_mesh)?;
    let rest_smoothed = smooth_points(&topology, &rest, &weights, smoothing);
    let rest_frames = tangent_frames(&topology.faces, &rest_smoothed);
    let frames = tangent_frames(&topology.faces, &smoothed);
    let mut result = smoothed;
    for ((face, corners), rest_corners) in topology
        .faces
        .iter()
        .zip(&frames.corners)
        .zip(&rest_frames.corners)
    {
        for ((&vertex, frame), rest_frame) in face.iter().zip(corners).zip(rest_corners) {
            let share = frame.weight / frames.vertex_weight[vertex];
            if share > 0.0 {
                let local = rest_frame.to_local(rest[vertex] - rest_smoothed[vertex]);
                result[vertex] += frame.to_world(local) * (share * scale);
            }
        }
    }
    with_positions(mesh, &result)
}

/// Rest positions for the corrective delta: the original mesh for ORCO, the bind coordinates for
/// BIND. An unbound BIND is an error, not a pass-through.
fn rest_positions(modifier: &Modifier, mesh: &Mesh, rest_mesh: &Mesh) -> Result<Vec<DVec3>> {
    match string_param_with_expected(modifier, "rest_source", "ORCO or BIND")? {
        "ORCO" => {
            if rest_mesh.vertices.len() != mesh.vertices.len() {
                return Err(invalid_parameter(
                    modifier,
                    "rest_source",
                    "an original mesh with the input vertex count",
                ));
            }
            Ok(positions(rest_mesh))
        }
        "BIND" => bind_positions(modifier, mesh.vertices.len()),
        _ => Err(invalid_parameter(modifier, "rest_source", "ORCO or BIND")),
    }
}

fn bind_positions(modifier: &Modifier, vertex_count: usize) -> Result<Vec<DVec3>> {
    let bind = modifier.binding_data.as_ref().ok_or_else(|| {
        PotError::with_details(
            ErrorCode::DependencyMissing,
            "corrective smooth BIND has no bind data; bind the modifier first",
            json!({"modifier_id": modifier.id, "parameter": "rest_source"}),
        )
    })?;
    let coords = bind
        .as_object()
        .filter(|object| {
            object.get("format").and_then(Value::as_str) == Some(CORRECTIVE_BIND_FORMAT)
        })
        .and_then(|object| object.get("coords"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            invalid_parameter(modifier, "bind_data", "blender_corrective_bind_v1 coords")
        })?;
    if coords.len() != vertex_count {
        return Err(invalid_parameter(
            modifier,
            "bind_data",
            "one rest coordinate per input vertex (bind vertex count mismatch)",
        ));
    }
    coords
        .iter()
        .map(|coord| bind_coordinate(modifier, coord))
        .collect()
}

fn bind_coordinate(modifier: &Modifier, value: &Value) -> Result<DVec3> {
    let expected = "finite [x, y, z] rest coordinates";
    let Some([x, y, z]) = value.as_array().map(Vec::as_slice) else {
        return Err(invalid_parameter(modifier, "bind_data", expected));
    };
    let component = |item: &Value| -> Result<f64> {
        item.as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| invalid_parameter(modifier, "bind_data", expected))
    };
    Ok(DVec3::new(component(x)?, component(y)?, component(z)?))
}
