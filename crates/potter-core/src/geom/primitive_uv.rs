//! The `UVMap` layer that Blender's primitive operators write by default (`calc_uvs`, and always for
//! the torus). Each function takes a mesh built in Blender's storage order and writes one layer of
//! per-loop values, in Blender's convention (v up); [`attach`] stores them in Potter's (v down).
//!
//! References (Blender 5.2.2): `bmesh/operators/bmo_primitive.cc` (`BM_mesh_calc_uvs_cube`,
//! `BM_mesh_calc_uvs_grid`, `BM_mesh_calc_uvs_circle`, `BM_mesh_calc_uvs_cone`, the `icouvs` table),
//! and `scripts/startup/bl_operators/add_mesh_torus.py` (`add_uvs`), and the UV sphere's
//! `bm_mesh_calc_uvs_sphere_face` and `BM_mesh_calc_uvs_sphere`.

use std::collections::HashMap;

use glam::DVec3;

use super::uv_layers::{self, UvLayer};
use super::{EndFillType, Mesh, MeshError};

/// Name of the UV layer Blender's primitives create.
const LAYER_NAME: &str = "UVMap";

/// Blender's `icouvs` table: one `[u, v]` per loop of the 20 base faces, in creation order.
const ICOSPHERE_UVS: [[f64; 2]; 60] = [
    [0.181_819, 0.0],
    [0.272_728, 0.157_461],
    [0.090_910, 0.157_461],
    [0.272_728, 0.157_461],
    [0.363_637, 0.0],
    [0.454_546, 0.157_461],
    [0.909_091, 0.0],
    [1.0, 0.157_461],
    [0.818_182, 0.157_461],
    [0.727_273, 0.0],
    [0.818_182, 0.157_461],
    [0.636_364, 0.157_461],
    [0.545_455, 0.0],
    [0.636_364, 0.157_461],
    [0.454_546, 0.157_461],
    [0.272_728, 0.157_461],
    [0.454_546, 0.157_461],
    [0.363_637, 0.314_921],
    [0.090_910, 0.157_461],
    [0.272_728, 0.157_461],
    [0.181_819, 0.314_921],
    [0.818_182, 0.157_461],
    [1.0, 0.157_461],
    [0.909_091, 0.314_921],
    [0.636_364, 0.157_461],
    [0.818_182, 0.157_461],
    [0.727_273, 0.314_921],
    [0.454_546, 0.157_461],
    [0.636_364, 0.157_461],
    [0.545_455, 0.314_921],
    [0.272_728, 0.157_461],
    [0.363_637, 0.314_921],
    [0.181_819, 0.314_921],
    [0.090_910, 0.157_461],
    [0.181_819, 0.314_921],
    [0.0, 0.314_921],
    [0.818_182, 0.157_461],
    [0.909_091, 0.314_921],
    [0.727_273, 0.314_921],
    [0.636_364, 0.157_461],
    [0.727_273, 0.314_921],
    [0.545_455, 0.314_921],
    [0.454_546, 0.157_461],
    [0.545_455, 0.314_921],
    [0.363_637, 0.314_921],
    [0.181_819, 0.314_921],
    [0.363_637, 0.314_921],
    [0.272_728, 0.472_382],
    [0.0, 0.314_921],
    [0.181_819, 0.314_921],
    [0.090_910, 0.472_382],
    [0.727_273, 0.314_921],
    [0.909_091, 0.314_921],
    [0.818_182, 0.472_382],
    [0.545_455, 0.314_921],
    [0.727_273, 0.314_921],
    [0.636_364, 0.472_382],
    [0.363_637, 0.314_921],
    [0.545_455, 0.314_921],
    [0.454_546, 0.472_382],
];

/// Stores per-face loop values (Blender's convention) as the mesh's `UVMap` layer, one entry per
/// face in storage order. Blender's v axis points up; Potter's stored convention points down.
fn attach(mesh: &mut Mesh, loops: Vec<Vec<[f64; 2]>>) {
    let faces = loops
        .into_iter()
        .map(|corners| {
            Some(
                corners
                    .into_iter()
                    .map(|[u, v]| [u, 1.0 - v])
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    uv_layers::write_layers(mesh, &[UvLayer::new(LAYER_NAME, faces)]);
}

/// Positions by vertex ID, for faces that name vertices by ID.
fn positions_by_id(mesh: &Mesh) -> HashMap<u32, DVec3> {
    mesh.vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect()
}

/// `BM_mesh_calc_uvs_cube`: a cross-shaped strip of six squares, walked face by face.
pub(super) fn cube(mesh: &mut Mesh) {
    let mut x = 0.375;
    let mut y = 0.0;
    let mut loops = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let mut corners = Vec::with_capacity(face.vertices.len());
        for index in 0..face.vertices.len().min(4) {
            corners.push([x, y]);
            match index {
                0 => x += 0.25,
                1 => y += 0.25,
                2 => x -= 0.25,
                _ => y -= 0.25,
            }
        }
        loops.push(corners);
        if y >= 0.75 && x > 0.125 {
            x = 0.125;
            y = 0.5;
        } else if x <= 0.125 {
            x = 0.625;
            y = 0.5;
        } else {
            y += 0.25;
        }
    }
    attach(mesh, loops);
}

/// `BM_mesh_calc_uvs_grid`: each face takes its cell of the unit square, row by row.
#[expect(
    clippy::cast_precision_loss,
    reason = "cell indices derive from u32 counts and are exactly representable in f64"
)]
pub(super) fn grid(mesh: &mut Mesh, x_segments: usize, y_segments: usize) {
    let dx = 1.0 / x_segments as f64;
    let dy = 1.0 / y_segments as f64;
    let loops = (0..mesh.faces.len())
        .map(|face_index| {
            let ix = (face_index % x_segments) as f64;
            let iy = (face_index / x_segments) as f64;
            vec![
                [ix * dx, iy * dy],
                [(ix + 1.0) * dx, iy * dy],
                [(ix + 1.0) * dx, (iy + 1.0) * dy],
                [ix * dx, (iy + 1.0) * dy],
            ]
        })
        .collect();
    attach(mesh, loops);
}

/// `BM_mesh_calc_uvs_circle`: the unit disc in the XY plane, so a loop's UV is its position.
pub(super) fn circle(mesh: &mut Mesh, radius: f64) {
    let positions = positions_by_id(mesh);
    let scale = 0.5 / radius;
    let loops: Vec<Vec<[f64; 2]>> = mesh
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    let co = positions[id];
                    [0.5 + scale * co.x, 0.5 + scale * co.y]
                })
                .collect()
        })
        .collect();
    attach(mesh, loops);
}

/// `BM_mesh_calc_uvs_cone` for a cylinder or cone with the given end radii (`radius_top` is the
/// `z = +depth / 2` radius). Sides with both radii unwrap as a strip; caps and collapsed sides
/// project onto their end's disc. The strip's `x` and `y` carry from one face to the next.
#[expect(
    clippy::cast_precision_loss,
    reason = "segment counts are small and exactly representable in f64"
)]
pub(super) fn cone(
    mesh: &mut Mesh,
    radius_top: f64,
    radius_bottom: f64,
    segments: usize,
    end_fill_type: EndFillType,
) {
    let cap = end_fill_type != EndFillType::Nothing;
    let positions = positions_by_id(mesh);
    let uv_width = 1.0 / segments as f64;
    let uv_height = if cap { 0.5 } else { 1.0 };
    let uv_center_y = if cap { 0.25 } else { 0.5 };
    let uv_center_x_top = if cap { 0.25 } else { 0.5 };
    let uv_center_x_bottom = if cap { 0.75 } else { 0.5 };
    let uv_radius = if cap { 0.24 } else { 0.5 };
    // A zero radius takes the other end's scale, then the unscaled radius.
    let scale = |radius: f64, fallback: f64| {
        if radius == 0.0 {
            fallback
        } else {
            uv_radius / radius
        }
    };
    let uv_scale_top = scale(radius_top, scale(radius_bottom, uv_radius));
    let uv_scale_bottom = scale(radius_bottom, uv_scale_top);
    let mut x = 1.0;
    let mut y = 1.0 - uv_height;
    let mut loops = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let points: Vec<DVec3> = face.vertices.iter().map(|id| positions[id]).collect();
        // Blender runs this before its caps are dissolved, when a cap is a triangle fan, so only a
        // quad that spans both ends is a side. A four-sided N-gon cap lies in one plane.
        let spans_both_ends =
            points.iter().any(|co| co.z > 0.0) && points.iter().any(|co| co.z < 0.0);
        if points.len() == 4 && radius_top != 0.0 && radius_bottom != 0.0 && spans_both_ends {
            let mut corners = Vec::with_capacity(4);
            for corner in 0..4 {
                match corner {
                    1 => y += uv_height,
                    2 => x -= uv_width,
                    3 => y -= uv_height,
                    _ => {}
                }
                corners.push([x, y]);
            }
            loops.push(corners);
        } else {
            let top = super::normals::face_normal(&points).z > 0.0;
            let corners: Vec<[f64; 2]> = points
                .iter()
                .map(|co| {
                    if top {
                        [
                            uv_center_x_top + co.x * uv_scale_top,
                            uv_center_y + co.y * uv_scale_top,
                        ]
                    } else {
                        [
                            uv_center_x_bottom + co.x * uv_scale_bottom,
                            uv_center_y + co.y * uv_scale_bottom,
                        ]
                    }
                })
                .collect();
            loops.push(corners);
        }
    }
    attach(mesh, loops);
}

/// `add_uvs` of the torus script: each quad's loops take the cell's corner of a `[0, 1]` square,
/// with the rounding and wrap rules the script applies.
#[expect(
    clippy::cast_precision_loss,
    reason = "segment counts are small and exactly representable in f64"
)]
pub(super) fn torus(mesh: &mut Mesh, major_segments: usize, minor_segments: usize) {
    let u_step = 1.0 / major_segments as f64;
    let v_step = 1.0 / minor_segments as f64;
    let u_init = 0.5 + 0.5_f64 % u_step;
    let v_init = 0.5 + 0.5_f64 % v_step;
    let u_wrap = 1.0 - (u_step / 2.0);
    let v_wrap = 1.0 - (v_step / 2.0);
    let mut loops = Vec::with_capacity(mesh.faces.len());
    let mut u_prev = u_init;
    let mut u_next = u_prev + u_step;
    for _ in 0..major_segments {
        let mut v_prev = v_init;
        let mut v_next = v_prev + v_step;
        for _ in 0..minor_segments {
            // Loops 0, 1, 2, 3 are the quad's corners in face order.
            loops.push(vec![
                [u_prev, v_prev],
                [u_next, v_prev],
                [u_next, v_next],
                [u_prev, v_next],
            ]);
            v_prev = if v_next > v_wrap {
                v_next - 1.0
            } else {
                v_next
            };
            v_next = v_prev + v_step;
        }
        u_prev = if u_next > u_wrap {
            u_next - 1.0
        } else {
            u_next
        };
        u_next = u_prev + u_step;
    }
    attach(mesh, loops);
}

/// Level-1 icosphere: the `icouvs` table, in loop order of the 20 base faces.
pub(super) fn icosphere_level_one(mesh: &mut Mesh) {
    let loops: Vec<Vec<[f64; 2]>> = (0..mesh.faces.len())
        .map(|face| ICOSPHERE_UVS[face * 3..face * 3 + 3].to_vec())
        .collect();
    attach(mesh, loops);
}

/// `BM_mesh_calc_uvs_sphere`: longitude and latitude per loop, a triangle's pole loops taking the
/// longitude of its two other corners' average, a seam fix per face, then the whole map shifted so
/// its smallest `u` is zero.
pub(super) fn uv_sphere(mesh: &mut Mesh) {
    let positions = positions_by_id(mesh);
    let mut min_x = 1.0_f64;
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let points: Vec<DVec3> = face.vertices.iter().map(|id| positions[id]).collect();
        let count = points.len();
        let (avg_x, avg_y) = if count == 3 {
            let sum = points
                .iter()
                .fold((0.0_f64, 0.0_f64), |(x, y), co| (x + co.x, y + co.y));
            (sum.0 / 3.0, sum.1 / 3.0)
        } else {
            (0.0, 0.0)
        };
        let mut uvs: Vec<[f64; 2]> = points
            .iter()
            .map(|co| {
                let length = co.length();
                let mut theta = if count == 3 && co.x.abs() < 0.0001 && co.y.abs() < 0.0001 {
                    avg_y.atan2(avg_x)
                } else {
                    co.y.atan2(co.x)
                };
                if (theta - std::f64::consts::PI).abs() < 0.0001 {
                    theta = -std::f64::consts::PI;
                }
                let phi = (co.z / length).clamp(-1.0, 1.0).acos();
                [
                    0.5 + theta / (std::f64::consts::PI * 2.0),
                    1.0 - phi / std::f64::consts::PI,
                ]
            })
            .collect();
        // Fix awkwardly wrapping UVs: a corner more than half a turn behind the face's largest u
        // is moved forward by one turn.
        let mut largest = 0;
        for (index, uv) in uvs.iter().enumerate().skip(1) {
            if uv[0] > uvs[largest][0] {
                largest = index;
            }
        }
        let largest_u = uvs[largest][0];
        for (index, uv) in uvs.iter_mut().enumerate() {
            if index != largest && largest_u - uv[0] > 0.5 {
                uv[0] += 1.0;
            }
        }
        for uv in &uvs {
            min_x = min_x.min(uv[0]);
        }
        loops.push(uvs);
    }
    for corners in &mut loops {
        for uv in corners.iter_mut() {
            uv[0] -= min_x;
        }
    }
    attach(mesh, loops);
}

fn lerp(first: [f64; 2], second: [f64; 2], fac: f64) -> [f64; 2] {
    [
        fac * first[0] + (1.0 - fac) * second[0],
        fac * first[1] + (1.0 - fac) * second[1],
    ]
}

/// Loop UVs of an icosphere subdivided to level 2 or up, keyed by (base face, vertex).
///
/// Blender's `subdivide_edges` splits each edge with `BM_edge_split`, which gives the new vertex, in
/// every face beside the edge, `fac` of the far vertex's UV and `1 - fac` of the split-from vertex's
/// (`BM_data_interp_face_vert_edge`). `fac` is `1 / (cuts + 1 - cut)`. The face grid's diagonals
/// are split the same way, from the diagonal's first end towards its second. Connecting edges copy
/// the UVs of the face's loops (`bmesh_kernel_split_face_make_edge`).
#[expect(
    clippy::cast_precision_loss,
    reason = "cut counts are small and exactly representable in f64"
)]
pub(super) fn icosphere_uv_table(
    base: &[[usize; 3]],
    edges: &[([usize; 2], Vec<usize>)],
    rows: &[Vec<Vec<usize>>],
) -> HashMap<(usize, usize), [f64; 2]> {
    let mut table = HashMap::new();
    for (face, corners) in base.iter().enumerate() {
        for (corner, &vertex) in corners.iter().enumerate() {
            table.insert((face, vertex), ICOSPHERE_UVS[face * 3 + corner]);
        }
    }
    for (endpoints, points) in edges {
        let cuts = points.len();
        let [start, end] = *endpoints;
        for (face, corners) in base.iter().enumerate() {
            if !(corners.contains(&start) && corners.contains(&end)) {
                continue;
            }
            let mut previous = start;
            for (cut, &point) in points.iter().enumerate() {
                let fac = 1.0 / (cuts + 1 - cut) as f64;
                let value = lerp(table[&(face, end)], table[&(face, previous)], fac);
                table.insert((face, point), value);
                previous = point;
            }
        }
    }
    let cuts = edges.first().map_or(0, |(_, points)| points.len());
    for (face, face_rows) in rows.iter().enumerate() {
        for row in 0..cuts {
            let line = &face_rows[row + 1];
            let far = line[row + 1];
            for split in 0..row {
                let fac = 1.0 / (row + 1 - split) as f64;
                let value = lerp(table[&(face, far)], table[&(face, line[split])], fac);
                table.insert((face, line[split + 1]), value);
            }
        }
    }
    table
}

/// Writes the icosphere's UVs: each final polygon takes, at each corner, the UV of its origin base
/// face's loop at that vertex.
pub(super) fn attach_by_origin(
    mesh: &mut Mesh,
    polygons: &[Vec<usize>],
    origins: &[usize],
    table: &HashMap<(usize, usize), [f64; 2]>,
) -> Result<(), MeshError> {
    let mut loops = Vec::with_capacity(polygons.len());
    for (polygon, &origin) in polygons.iter().zip(origins) {
        let mut corners = Vec::with_capacity(polygon.len());
        for &vertex in polygon {
            corners.push(table.get(&(origin, vertex)).copied().ok_or(
                MeshError::InvalidTopology("icosphere corner has no UV from its base face"),
            )?);
        }
        loops.push(corners);
    }
    attach(mesh, loops);
    Ok(())
}
