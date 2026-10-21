//! Blender's harmonic-coordinates bind for Mesh Deform, ported from `meshlaplacian.cc`:
//!
//! - cage ray tests and the inside rule: `meshdeform_ray_tree_intersect`, `harmonic_ray_callback` and
//!   `meshdeform_inside_cage` (meshlaplacian.cc:939-1082), run on the Blender cage tree in `cage_bvh`
//!   (corner tris, binary k-DOP tree, watertight `dfs_raycast`, with the callback's normalised distance).
//!   Facing is `dot(n, dir) <= 0` on the polygon normal.
//! - boundary weights: `interp_weights_poly_v3` (`math_geom.cc` 4209-4292), used for the polygon of each hit.
//! - cell tagging and flood fill: `meshdeform_add_intersections`, `meshdeform_bind_floodfill`,
//!   `meshdeform_check_semibound` (lines 1119-1210, 1289-1303).
//! - linear system: `meshdeform_boundary_total_weight`, `meshdeform_matrix_add_*` (lines 1305-1443) and
//!   `meshdeform_matrix_solve` (lines 1445-1571).
//! - bind store: `harmonic_coordinates_bind` (lines 1573-1741) and the static weights (1514-1523).
//!
//! Blender solves the system with Eigen `SparseLU` (`linear_solver.cc` 293-328). This port assembles the
//! same matrix as triplets, factors it once with the same solver through the native shim
//! `native/meshdeform_harmonic_solve.cc`, then solves one cage vertex's right-hand side at a time.
//!
//! The cage ray tests (`cage_bvh`) run in `f32` as Blender does, including the tree traversal, so culling
//! matches. Elsewhere arithmetic is `f64` where Blender uses `f32`. Comparisons, branch structure, and
//! accumulation order follow the source.

use std::{ffi::c_void, ptr::NonNull};

use glam::DVec3;

use super::super::id_indices;
use super::cage_bvh::CageRayCache;
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, bvh::Vec3},
};

/// `MESHDEFORM_MIN_INFLUENCE` (meshlaplacian.cc:850).
const MIN_INFLUENCE: f64 = 0.0005;
/// `MESHDEFORM_LEN_THRESHOLD` (meshlaplacian.cc:848).
const LEN_THRESHOLD: f64 = 1e-6;
/// `FLT_EPSILON` as used by `interp_weights_poly_v3` (`math_geom.cc:4223`).
const FLT_EPSILON: f64 = f32::EPSILON as f64;
/// `MESHDEFORM_OFFSET` (meshlaplacian.cc:852-860); index 0 is the cell itself.
const OFFSET: [[isize; 3]; 7] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];
/// Padding on each side of the cage box, in cells (meshlaplacian.cc:1634-1635, `2.1f`).
const PAD_CELLS: f64 = 2.1;

/// Blender's stored harmonic bind for one cage: the dynamic grid (`mmd->dyngrid`) or the static
/// weights (`mmd->bindweights`), plus the per-vertex inside flags (`mmd->dynverts`).
pub(super) struct HarmonicBindGrid {
    /// Cells per axis, `(1 << precision) + 2` (`mdb->size`, meshlaplacian.cc:1589).
    pub(super) size: usize,
    /// Padded grid origin in cage-local space (`mmd->dyncellmin` = `mdb->min`).
    pub(super) cell_min: DVec3,
    /// Cell width used by evaluation (`mmd->dyncellwidth` = `mdb->width[0]`). Evaluation uses this
    /// single value on every axis.
    pub(super) cell_width: f64,
    /// Per source vertex: inside the cage (`mdb->inside`, stored as `dynverts`).
    pub(super) inside: Vec<bool>,
    /// Static bind only (empty when dynamic). Per source vertex, the nonzero `(cage vertex, weight)`
    /// pairs in cage index order, uncompacted; deform.rs compacts them.
    pub(super) static_rows: Vec<Vec<(usize, f64)>>,
    /// Dynamic bind only (empty when static). Per grid cell, index `x + y*size + z*size*size`: the
    /// influences with `phi >= MIN_INFLUENCE`, normalized to sum to one. Order is Blender's linked-list
    /// order (latest cage vertex first).
    pub(super) cells: Vec<Vec<(usize, f64)>>,
}

/// Cage geometry in the form Blender's cage mesh cache provides: positions and polygons for the boundary
/// weights, and the cage ray structures (`cage_bvh`) for the inside test and the cell-edge rays.
struct CageGeometry {
    points: Vec<DVec3>,
    faces: Vec<Vec<usize>>,
    rays: CageRayCache,
}

/// Cage-space `f32` coordinates, as Blender's bind stores them.
fn to_vec3(point: DVec3) -> Vec3 {
    point.as_vec3().to_array()
}

/// Cell classification, `MESHDEFORM_TAG_*` (meshlaplacian.cc:842-845).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tag {
    Untyped,
    Boundary,
    Interior,
    Exterior,
}

/// Blender's `MDefBoundIsect` (meshlaplacian.cc:862-873): the first cage polygon hit by a cell-edge ray.
struct BoundaryHit {
    /// Distance from the cell center to the hit, floored at `LEN_THRESHOLD` (`isect->len`).
    len: f64,
    /// Mean-value weights paired with the hit polygon's corner vertices, in polygon order.
    corners: Vec<(usize, f64)>,
}

/// Padded grid: Blender's `mdb->min/max/width/halfwidth` and `mdb->size` (meshlaplacian.cc:1581-1639).
#[derive(Clone, Copy)]
struct Grid {
    size: usize,
    min: DVec3,
    max: DVec3,
    width: DVec3,
    halfwidth: DVec3,
}

/// Unknowns of the linear system: `mdb->varidx` (meshlaplacian.cc:1453-1456).
struct Variables {
    index: Vec<Option<usize>>,
    count: usize,
}

/// Square system `A x = b` in triplet form (`mdb->varidx` unknowns, `meshdeform_matrix_add_*` entries),
/// handed to the native Eigen `SparseLU` solve.
struct SparseSystem {
    size: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
}

/// Mean-value direction from a cage polygon vertex to the hit point (`Float3_Len`, `math_geom.cc:4157`).
#[derive(Clone, Copy)]
struct Direction {
    vector: DVec3,
    length: f64,
}

/// Degenerate hit cases in `interp_weights_poly_v3` (math_geom.cc:4139-4141).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Degenerate {
    Point,
    Segment,
}

/// Blender's `harmonic_coordinates_bind` (meshlaplacian.cc:1573-1741) and its bind store. Builds the
/// harmonic bind for one cage against the source vertices.
///
/// `source_cage_space` are the source vertices in cage-local space. `dynamic` selects the dynamic grid
/// (`cells`) or the static per-vertex weights (`static_rows`).
pub(super) fn bind_harmonic_grid(
    source_cage_space: &[DVec3],
    cage: &Mesh,
    precision: u32,
    dynamic: bool,
) -> Result<HarmonicBindGrid> {
    if !(2..=10).contains(&precision) {
        return Err(PotError::invalid_argument(
            "mesh deform harmonic precision must be an integer from 2 to 10",
        ));
    }
    let geometry = cage_geometry(cage)?;
    let size = (1_usize << precision) + 2;
    let grid = Grid::padded(&geometry.points, size)?;
    let cell_count = grid_cells(size)?;
    let inside: Vec<bool> = source_cage_space
        .iter()
        .map(|&point| inside_cage(&geometry, &grid, point))
        .collect();
    let cells = CellBind::new(&geometry, grid, cell_count)?;
    let variables = Variables::new(&cells.tags)?;
    let mut lists: Vec<Vec<(usize, f64)>> =
        try_vec_with(if dynamic { cell_count } else { 0 }, Vec::new)?;
    let mut static_rows: Vec<Vec<(usize, f64)>> =
        try_vec_with(if dynamic { 0 } else { source_cage_space.len() }, Vec::new)?;
    if variables.count > 0 {
        let system = cells.assemble(&variables)?;
        let factor = HarmonicFactor::factorize(&system)?;
        // The triplets are only needed by the factorization.
        drop(system);
        // One right-hand side per cage vertex (meshlaplacian.cc:1657-1683), solved against the single
        // factorization. Each solution is expanded to cell values and recorded before the next column, so
        // only one column of unknowns and one of cells is held at a time.
        let mut rhs = try_vec_with(variables.count, || 0.0)?;
        let mut solution = try_vec_with(variables.count, || 0.0)?;
        let mut phi = try_vec_with(cell_count, || 0.0)?;
        for cage_vertex in 0..geometry.points.len() {
            rhs.fill(0.0);
            cells.fill_rhs(&variables, cage_vertex, &mut rhs);
            factor.solve_column(&rhs, &mut solution)?;
            cells.finish_phi(&variables, &solution, cage_vertex, &mut phi);
            if dynamic {
                push_dynamic(&mut lists, &phi, cage_vertex)?;
            } else {
                push_static(
                    &mut static_rows,
                    source_cage_space,
                    &inside,
                    &grid,
                    &phi,
                    cage_vertex,
                )?;
            }
        }
    }
    Ok(HarmonicBindGrid {
        size,
        cell_min: grid.min,
        cell_width: grid.width.x,
        inside,
        static_rows,
        cells: normalize_cells(lists),
    })
}

/// Cage positions, polygon corners, and the cage ray structures. Mirrors Blender's `cagemesh_cache` and the
/// cage tree (meshlaplacian.cc:1611-1619, bvhutils.cc), in `cage_bvh`. Corner triangles are Blender's loop
/// triangles from `Mesh::tessellate`.
fn cage_geometry(cage: &Mesh) -> Result<CageGeometry> {
    let indices = id_indices(cage);
    let storage_index = |id: &u32| {
        indices.get(id).copied().ok_or_else(|| {
            PotError::invalid_operation("mesh deform cage contains invalid face vertices")
        })
    };
    let points: Vec<DVec3> = cage.vertices.iter().map(|vertex| vertex.co).collect();
    let mut faces = Vec::with_capacity(cage.faces.len());
    for face in &cage.faces {
        let corners: Vec<usize> = face
            .vertices
            .iter()
            .map(storage_index)
            .collect::<Result<_>>()?;
        if corners.len() < 3 {
            return Err(PotError::invalid_operation(
                "mesh deform cage polygons need at least three vertices",
            ));
        }
        faces.push(corners);
    }
    if faces.is_empty() {
        return Err(PotError::invalid_operation(
            "mesh deform harmonic binding requires cage faces",
        ));
    }
    let corner_tris = cage
        .tessellate()
        .map_err(|error| {
            PotError::invalid_operation(format!("mesh deform cage triangulation failed: {error}"))
        })?
        .into_iter()
        .map(|triangle| {
            Ok([
                storage_index(&triangle[0])?,
                storage_index(&triangle[1])?,
                storage_index(&triangle[2])?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let cage_points: Vec<Vec3> = points.iter().map(|&point| to_vec3(point)).collect();
    let rays = CageRayCache::new(&cage_points, &faces, corner_tris)?;
    Ok(CageGeometry {
        points,
        faces,
        rays,
    })
}

fn limit_error() -> PotError {
    PotError::new(
        ErrorCode::LimitExceeded,
        "Mesh Deform harmonic bind is too large to represent; lower the precision",
    )
}

/// Builds a vector of `len` values, failing with a limit error instead of aborting when memory is short.
fn try_vec_with<T>(len: usize, mut make: impl FnMut() -> T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| limit_error())?;
    values.extend((0..len).map(|_| make()));
    Ok(values)
}

/// Appends to `values`, failing with a limit error instead of aborting when memory is short.
fn try_push<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    values.try_reserve(1).map_err(|_| limit_error())?;
    values.push(value);
    Ok(())
}

/// Cell count for `size` cells per axis. Only arithmetic overflow is rejected.
fn grid_cells(size: usize) -> Result<usize> {
    size.checked_mul(size)
        .and_then(|area| area.checked_mul(size))
        .ok_or_else(limit_error)
}

fn cell_index(size: usize, x: usize, y: usize, z: usize) -> usize {
    x + y * size + z * size * size
}

fn cell_coords(size: usize, cell: usize) -> (usize, usize, usize) {
    (cell % size, (cell / size) % size, cell / (size * size))
}

/// Cell at `dir` offset from `(x, y, z)`, or `None` outside the grid (`meshdeform_index`, 1086-1105).
fn neighbour_coords(
    size: usize,
    (x, y, z): (usize, usize, usize),
    dir: usize,
) -> Option<(usize, usize, usize)> {
    let [dx, dy, dz] = OFFSET[dir];
    let nx = x.checked_add_signed(dx)?;
    let ny = y.checked_add_signed(dy)?;
    let nz = z.checked_add_signed(dz)?;
    (nx < size && ny < size && nz < size).then_some((nx, ny, nz))
}

/// Padded bounds and cell width (meshlaplacian.cc:1581-1639).
impl Grid {
    fn padded(points: &[DVec3], size: usize) -> Result<Self> {
        let mut low = DVec3::splat(f64::INFINITY);
        let mut high = DVec3::splat(f64::NEG_INFINITY);
        for &point in points {
            low = low.min(point);
            high = high.max(point);
        }
        let maxwidth = (high - low).max_element().max(-1.0);
        if !(maxwidth.is_finite() && maxwidth > 0.0) {
            return Err(PotError::invalid_operation(
                "mesh deform cage must enclose a three-dimensional region",
            ));
        }
        let size_f = size as f64;
        let (min_x, max_x, width_x) = pad_axis(low.x, high.x, maxwidth, size_f);
        let (min_y, max_y, width_y) = pad_axis(low.y, high.y, maxwidth, size_f);
        let (min_z, max_z, width_z) = pad_axis(low.z, high.z, maxwidth, size_f);
        let width = DVec3::new(width_x, width_y, width_z);
        Ok(Self {
            size,
            min: DVec3::new(min_x, min_y, min_z),
            max: DVec3::new(max_x, max_y, max_z),
            width,
            halfwidth: width * 0.5,
        })
    }

    /// Cell center (meshlaplacian.cc:1107-1117): `min + x * width + halfwidth`.
    fn center(&self, x: usize, y: usize, z: usize) -> DVec3 {
        DVec3::new(
            self.min.x + x as f64 * self.width.x + self.halfwidth.x,
            self.min.y + y as f64 * self.width.y + self.halfwidth.y,
            self.min.z + z as f64 * self.width.z + self.halfwidth.z,
        )
    }
}

/// One axis of `harmonic_coordinates_bind`'s padding: returns `(min, max, width)`.
#[expect(
    clippy::manual_midpoint,
    reason = "mirrors Blender's (low + high) * 0.5 evaluation order for parity"
)]
fn pad_axis(low: f64, high: f64, maxwidth: f64, size: f64) -> (f64, f64, f64) {
    let center = (low + high) * 0.5;
    let mut min = center - maxwidth * 0.5;
    let mut max = center + maxwidth * 0.5;
    let cell = (max - min) / (size - 4.0);
    min -= PAD_CELLS * cell;
    max += PAD_CELLS * cell;
    (min, max, (max - min) / size)
}

/// Inside test (`meshdeform_inside_cage`, meshlaplacian.cc:1060-1082), on the cage ray structures.
fn inside_cage(geometry: &CageGeometry, grid: &Grid, point: DVec3) -> bool {
    geometry
        .rays
        .point_inside(to_vec3(point), to_vec3(grid.min), to_vec3(grid.max))
}

/// Boundary record for one cell-edge ray (`meshdeform_ray_tree_intersect`, meshlaplacian.cc:985-1058).
fn boundary_hit(geometry: &CageGeometry, start: DVec3, end: DVec3) -> Option<BoundaryHit> {
    let hit = geometry.rays.closest_hit(to_vec3(start), to_vec3(end))?;
    let point = start + (end - start) * f64::from(hit.lambda);
    let len = start.distance(point).max(LEN_THRESHOLD);
    let corners = &geometry.faces[geometry.rays.triangle_face(hit.triangle)];
    let positions: Vec<DVec3> = corners
        .iter()
        .map(|&corner| geometry.points[corner])
        .collect();
    let weights = mean_value_weights(&positions, point);
    Some(BoundaryHit {
        len,
        corners: corners.iter().copied().zip(weights).collect(),
    })
}

/// Weight of `cage_vertex` in a boundary record (`meshdeform_boundary_phi`, meshlaplacian.cc:1212-1226).
fn boundary_phi(hit: &BoundaryHit, cage_vertex: usize) -> f64 {
    hit.corners
        .iter()
        .find(|(vertex, _)| *vertex == cage_vertex)
        .map_or(0.0, |&(_, weight)| weight)
}

impl Direction {
    /// `DIR_V3_SET` (`math_geom.cc:4143`): `vertex - origin` and its length.
    fn between(vertex: DVec3, origin: DVec3) -> Self {
        let vector = vertex - origin;
        Self {
            vector,
            length: vector.length(),
        }
    }
}

/// `mean_value_half_tan_v3` (math_geom.cc:4167-4182).
fn half_tangent(current: Direction, next: Direction) -> f64 {
    let area = current.vector.cross(next.vector).length();
    if area != 0.0 {
        let dot = current.vector.dot(next.vector);
        let len = current.length * next.length;
        let result = (len - dot) / area;
        if result.is_finite() {
            return result;
        }
    }
    0.0
}

/// `line_point_factor_v3` (math_geom.cc:3354-3372), with epsilon 0 and fallback 0.
fn line_point_factor(point: DVec3, a: DVec3, b: DVec3) -> f64 {
    let u = b - a;
    let h = point - a;
    let len_sq = u.length_squared();
    if len_sq > 0.0 { u.dot(h) / len_sq } else { 0.0 }
}

/// `dist_squared_to_line_segment_v3` (math_geom.cc:535-542) through `closest_to_line_segment_v3`
/// (math_geom.cc:403-423).
fn dist_squared_to_segment(point: DVec3, a: DVec3, b: DVec3) -> f64 {
    let lambda = line_point_factor(point, a, b);
    let closest = if lambda <= 0.0 {
        a
    } else if lambda >= 1.0 {
        b
    } else {
        a + (b - a) * lambda
    };
    closest.distance_squared(point)
}

/// Mean-value weights of `point` over a polygon (`interp_weights_poly_v3`, math_geom.cc:4209-4292).
/// Degenerate points on a vertex or an edge fall back to linear weights, as in the source.
fn mean_value_weights(points: &[DVec3], point: DVec3) -> Vec<f64> {
    let n = points.len();
    let max_value = points.iter().fold(0.0_f64, |max, p| {
        max.max((p.x - point.x).abs())
            .max((p.y - point.y).abs())
            .max((p.z - point.z).abs())
    });
    let eps = 16.0 * FLT_EPSILON * max_value;
    let eps_sq = eps * eps;
    let mut weights = vec![0.0; n];
    let mut total = 0.0;
    let mut i_curr = n - 1;
    let mut i_next = 0;
    let mut v_curr = points[n - 1];
    let mut v_next = points[0];
    let mut d_next = Direction::between(points[n - 1], point);
    let mut ht_prev = half_tangent(Direction::between(points[n - 2], point), d_next);
    let mut degenerate = None;
    while i_next < n {
        if d_next.length < eps {
            degenerate = Some(Degenerate::Point);
            break;
        }
        if dist_squared_to_segment(point, v_curr, v_next) < eps_sq {
            degenerate = Some(Degenerate::Segment);
            break;
        }
        let d_curr = d_next;
        d_next = Direction::between(v_next, point);
        let ht = half_tangent(d_curr, d_next);
        weights[i_curr] = (ht_prev + ht) / d_curr.length;
        total += weights[i_curr];
        i_curr = i_next;
        i_next += 1;
        v_curr = v_next;
        v_next = points[i_next % n];
        ht_prev = ht;
    }
    match degenerate {
        Some(Degenerate::Point) => {
            weights.fill(0.0);
            weights[i_curr] = 1.0;
        }
        Some(Degenerate::Segment) => {
            weights.fill(0.0);
            let fac = line_point_factor(point, v_curr, v_next).clamp(0.0, 1.0);
            weights[i_curr] = 1.0 - fac;
            weights[i_next] = fac;
        }
        None => {
            if total != 0.0 {
                for weight in &mut weights {
                    *weight /= total;
                }
            }
        }
    }
    weights
}

/// Cell classification and the per-cage-vertex solve (`harmonic_coordinates_bind`, 1657-1683).
impl<'a> CellBind<'a> {
    fn new(geometry: &'a CageGeometry, grid: Grid, cell_count: usize) -> Result<Self> {
        let mut cells = Self {
            geometry,
            grid,
            tags: try_vec_with(cell_count, || Tag::Untyped)?,
            boundary: try_vec_with(cell_count, Default::default)?,
            semibound: try_vec_with(cell_count, || false)?,
        };
        cells.tag_boundaries();
        cells.flood_fill();
        cells.mark_semibound();
        Ok(cells)
    }

    /// `meshdeform_add_intersections` over every cell (meshlaplacian.cc:1663-1669, 1119-1142).
    fn tag_boundaries(&mut self) {
        let size = self.grid.size;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let cell = cell_index(size, x, y, z);
                    let center = self.grid.center(x, y, z);
                    for dir in 1..=6 {
                        let Some(next) = neighbour_coords(size, (x, y, z), dir) else {
                            continue;
                        };
                        let target = self.grid.center(next.0, next.1, next.2);
                        if let Some(hit) = boundary_hit(self.geometry, center, target) {
                            self.boundary[cell][dir - 1] = Some(hit);
                            self.tags[cell] = Tag::Boundary;
                        }
                    }
                }
            }
        }
    }

    /// `meshdeform_bind_floodfill` (meshlaplacian.cc:1144-1210). Cell 0 is exterior by padding; untyped
    /// cells left after the fill are interior.
    fn flood_fill(&mut self) {
        self.tags[0] = Tag::Exterior;
        let mut stack = vec![0_usize];
        while let Some(cell) = stack.pop() {
            for dir in 1..=6 {
                let Some(next) = self.neighbour(cell, dir) else {
                    continue;
                };
                let crosses = self.boundary[cell][dir - 1].is_some();
                let open = match self.tags[next] {
                    Tag::Untyped => true,
                    Tag::Boundary => !crosses,
                    Tag::Interior | Tag::Exterior => false,
                };
                if open {
                    self.tags[next] = Tag::Exterior;
                    stack.push(next);
                }
            }
        }
        for tag in &mut self.tags {
            if *tag == Tag::Untyped {
                *tag = Tag::Interior;
            }
        }
    }

    /// `meshdeform_check_semibound` (meshlaplacian.cc:1289-1303): exterior cells with a crossed edge.
    fn mark_semibound(&mut self) {
        for (cell, tag) in self.tags.iter().enumerate() {
            if *tag == Tag::Exterior && self.boundary[cell].iter().any(Option::is_some) {
                self.semibound[cell] = true;
            }
        }
    }

    fn neighbour(&self, cell: usize, dir: usize) -> Option<usize> {
        let size = self.grid.size;
        let coords = neighbour_coords(size, cell_coords(size, cell), dir)?;
        Some(cell_index(size, coords.0, coords.1, coords.2))
    }

    /// Matrix coupling of `cell` to its neighbour in `dir`: the neighbour is a unknown and the edge is
    /// not crossed (meshlaplacian.cc:1349-1360).
    fn coupling(&self, cell: usize, dir: usize) -> Option<usize> {
        let next = self.neighbour(cell, dir)?;
        (self.tags[next] != Tag::Exterior && self.boundary[cell][dir - 1].is_none()).then_some(next)
    }

    /// `meshdeform_boundary_total_weight` (meshlaplacian.cc:1305-1332).
    fn total_weight(&self, cell: usize) -> f64 {
        let mut total = 0.0;
        for dir in 1..=6 {
            if self.neighbour(cell, dir).is_none() {
                continue;
            }
            let weight = match &self.boundary[cell][dir - 1] {
                Some(hit) => 1.0 / hit.len,
                None if !self.semibound[cell] => 1.0 / self.grid.width.x,
                None => 0.0,
            };
            total += weight;
        }
        total
    }

    /// Builds the system matrix in triplet form (`meshdeform_matrix_add_cell`, meshlaplacian.cc:1334-1361).
    fn assemble(&self, variables: &Variables) -> Result<SparseSystem> {
        let mut system = SparseSystem {
            size: variables.count,
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
        };
        for (cell, slot) in variables.index.iter().enumerate() {
            let Some(row) = *slot else {
                continue;
            };
            system.push(row, row, 1.0)?;
            let total = self.total_weight(cell);
            for dir in 1..=6 {
                let Some(next) = self.coupling(cell, dir) else {
                    continue;
                };
                let Some(col) = variables.index[next] else {
                    continue;
                };
                let weight = (1.0 / self.grid.width.x) / total;
                system.push(row, col, -weight)?;
            }
        }
        Ok(system)
    }

    /// Right-hand side for one cage vertex (`meshdeform_matrix_add_rhs`, meshlaplacian.cc:1363-1390),
    /// written into `rhs` with one value per unknown. `rhs` must start zeroed.
    fn fill_rhs(&self, variables: &Variables, cage_vertex: usize, rhs: &mut [f64]) {
        for (cell, slot) in variables.index.iter().enumerate() {
            let Some(row) = *slot else {
                continue;
            };
            let total = self.total_weight(cell);
            for dir in 1..=6 {
                if self.neighbour(cell, dir).is_none() {
                    continue;
                }
                if let Some(hit) = &self.boundary[cell][dir - 1] {
                    let weight = (1.0 / hit.len) / total;
                    rhs[row] += weight * boundary_phi(hit, cage_vertex);
                }
            }
        }
    }

    /// Cell values for one cage vertex from its solved unknowns (`solution`, one per unknown), written into
    /// `phi`. Adds the semibound cells (meshlaplacian.cc:1392-1416) and the exterior cells next to them
    /// (1418-1443).
    fn finish_phi(
        &self,
        variables: &Variables,
        solution: &[f64],
        cage_vertex: usize,
        phi: &mut [f64],
    ) {
        phi.fill(0.0);
        for (cell, slot) in variables.index.iter().enumerate() {
            if let Some(row) = *slot {
                phi[cell] = solution[row];
            }
        }
        for (cell, &is_semibound) in self.semibound.iter().enumerate() {
            if !is_semibound {
                continue;
            }
            let total = self.total_weight(cell);
            let mut value = 0.0;
            for hit in self.boundary[cell].iter().flatten() {
                let weight = (1.0 / hit.len) / total;
                value += weight * boundary_phi(hit, cage_vertex);
            }
            phi[cell] = value;
        }
        for (cell, &tag) in self.tags.iter().enumerate() {
            if tag != Tag::Exterior || self.semibound[cell] {
                continue;
            }
            let mut sum = 0.0;
            let mut count = 0.0;
            for dir in 1..=6 {
                if let Some(next) = self
                    .neighbour(cell, dir)
                    .filter(|&next| self.semibound[next])
                {
                    sum += phi[next];
                    count += 1.0;
                }
            }
            if count != 0.0 {
                phi[cell] = sum / count;
            }
        }
    }
}

/// Per-cell bind state: `mdb->tag`, `mdb->boundisect`, `mdb->semibound`.
struct CellBind<'a> {
    geometry: &'a CageGeometry,
    grid: Grid,
    tags: Vec<Tag>,
    boundary: Vec<[Option<BoundaryHit>; 6]>,
    semibound: Vec<bool>,
}

impl Variables {
    /// `mdb->varidx` (meshlaplacian.cc:1453-1456): exterior cells carry no unknown.
    fn new(tags: &[Tag]) -> Result<Self> {
        let mut index: Vec<Option<usize>> = Vec::new();
        index
            .try_reserve_exact(tags.len())
            .map_err(|_| limit_error())?;
        let mut count = 0;
        for &tag in tags {
            if tag == Tag::Exterior {
                index.push(None);
            } else {
                index.push(Some(count));
                count += 1;
            }
        }
        Ok(Self { index, count })
    }
}

impl SparseSystem {
    fn push(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        try_push(&mut self.rows, row)?;
        try_push(&mut self.cols, col)?;
        try_push(&mut self.values, value)
    }
}

unsafe extern "C" {
    fn potter_meshdeform_harmonic_factor(
        size: usize,
        entry_count: usize,
        rows: *const usize,
        cols: *const usize,
        values: *const f64,
        factor: *mut *mut c_void,
    ) -> i32;
    fn potter_meshdeform_harmonic_solve_column(
        factor: *const c_void,
        rhs: *const f64,
        solution: *mut f64,
    ) -> i32;
    fn potter_meshdeform_harmonic_free(factor: *mut c_void);
}

/// Eigen `SparseLU` factorization of the bind system (`EIG_linear_solver_solve`, `linear_solver.cc`
/// 293-328), held by the native shim. Every cage vertex's right-hand side is solved against this one
/// factorization, one column at a time.
struct HarmonicFactor {
    handle: NonNull<c_void>,
    size: usize,
}

impl HarmonicFactor {
    /// Factors `system` once. Eigen stores matrix indices as 32-bit integers, so larger systems are
    /// rejected.
    fn factorize(system: &SparseSystem) -> Result<Self> {
        let size = system.size;
        let entry_count = system.values.len();
        if system.rows.len() != entry_count || system.cols.len() != entry_count {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "Mesh Deform harmonic bind system entries disagree in length",
            ));
        }
        if i32::try_from(size).is_err() || i32::try_from(entry_count).is_err() {
            return Err(limit_error());
        }
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `rows`, `cols`, and `values` each hold `entry_count` initialized elements (checked above).
        // The native side copies the triplets before returning and keeps no pointer to them; `handle` is
        // written only on success.
        let status = unsafe {
            potter_meshdeform_harmonic_factor(
                size,
                entry_count,
                system.rows.as_ptr(),
                system.cols.as_ptr(),
                system.values.as_ptr(),
                &raw mut handle,
            )
        };
        if status != 0 {
            return Err(solver_error(status));
        }
        let handle = NonNull::new(handle).ok_or_else(|| solver_error(status))?;
        Ok(Self { handle, size })
    }

    /// Solves one right-hand side (`size` values) into `solution` (`size` values).
    fn solve_column(&self, rhs: &[f64], solution: &mut [f64]) -> Result<()> {
        if rhs.len() != self.size || solution.len() != self.size {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "Mesh Deform harmonic bind right-hand side disagrees with its factorization",
            ));
        }
        // SAFETY: `handle` is live until `Drop`, and both slices hold `size` initialized values.
        let status = unsafe {
            potter_meshdeform_harmonic_solve_column(
                self.handle.as_ptr(),
                rhs.as_ptr(),
                solution.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(solver_error(status));
        }
        Ok(())
    }
}

impl Drop for HarmonicFactor {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `potter_meshdeform_harmonic_factor` and is released exactly once.
        unsafe { potter_meshdeform_harmonic_free(self.handle.as_ptr()) };
    }
}

fn solver_error(status: i32) -> PotError {
    PotError::invalid_operation(format!(
        "Failed to find Mesh Deform harmonic bind solution (native status {status}; increase precision?)"
    ))
}

/// Dynamic bind: records the cage vertex's influence on each cell (`meshdeform_matrix_solve`,
/// meshlaplacian.cc:1525-1539 dynamic branch). Pushing in cage order and reversing afterwards gives
/// Blender's head-inserted linked-list order.
fn push_dynamic(lists: &mut [Vec<(usize, f64)>], phi: &[f64], cage_vertex: usize) -> Result<()> {
    for (list, &value) in lists.iter_mut().zip(phi) {
        if value >= MIN_INFLUENCE {
            try_push(list, (cage_vertex, value))?;
        }
    }
    Ok(())
}

/// Static bind: interpolates `phi` at each inside source vertex (meshlaplacian.cc:1512-1523 with
/// `meshdeform_interp_w`, 1228-1287).
fn push_static(
    rows: &mut [Vec<(usize, f64)>],
    source: &[DVec3],
    inside: &[bool],
    grid: &Grid,
    phi: &[f64],
    cage_vertex: usize,
) -> Result<()> {
    for ((row, &point), &is_inside) in rows.iter_mut().zip(source).zip(inside) {
        if !is_inside {
            continue;
        }
        let gridvec = (point - grid.min - grid.halfwidth) / grid.width;
        let weight = interpolate_phi(grid.size, phi, gridvec);
        if weight != 0.0 {
            try_push(row, (cage_vertex, weight))?;
        }
    }
    Ok(())
}

/// `meshdeform_interp_w` (meshlaplacian.cc:1228-1287): trilinear interpolation with clamped corners.
fn interpolate_phi(size: usize, phi: &[f64], gridvec: DVec3) -> f64 {
    let (ix, dx) = split_axis(gridvec.x);
    let (iy, dy) = split_axis(gridvec.y);
    let (iz, dz) = split_axis(gridvec.z);
    let mut result = 0.0;
    let mut total = 0.0;
    for corner in 0..8_usize {
        let (x, wx) = if corner & 1 != 0 {
            (ix + 1, dx)
        } else {
            (ix, 1.0 - dx)
        };
        let (y, wy) = if corner & 2 != 0 {
            (iy + 1, dy)
        } else {
            (iy, 1.0 - dy)
        };
        let (z, wz) = if corner & 4 != 0 {
            (iz + 1, dz)
        } else {
            (iz, 1.0 - dz)
        };
        let cell = cell_index(
            size,
            clamp_axis(x, size),
            clamp_axis(y, size),
            clamp_axis(z, size),
        );
        let weight = wx * wy * wz;
        result += weight * phi[cell];
        total += weight;
    }
    if total > 0.0 {
        result /= total;
    }
    result
}

fn split_axis(value: f64) -> (i64, f64) {
    let base = value as i64;
    (base, value - base as f64)
}

fn clamp_axis(value: i64, size: usize) -> usize {
    value.clamp(0, size as i64 - 1) as usize
}

/// Blender normalizes each dynamic cell after building its list (meshlaplacian.cc:1700-1721). Lists are
/// reversed to Blender's head-inserted order before the sum.
fn normalize_cells(lists: Vec<Vec<(usize, f64)>>) -> Vec<Vec<(usize, f64)>> {
    lists
        .into_iter()
        .map(|mut list| {
            list.reverse();
            let total: f64 = list.iter().map(|&(_, weight)| weight).sum();
            if total > 0.0 {
                for (_, weight) in &mut list {
                    *weight /= total;
                }
            }
            list
        })
        .collect()
}
