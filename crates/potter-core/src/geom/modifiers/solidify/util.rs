//! Vector helpers shared by the Solidify algorithms. Blender evaluates Solidify in 32-bit floats and
//! the offsets are sensitive to rounding, so these mirror `BLI_math_vector.cc`,
//! `math_geom_inline.cc`, and `MOD_solidify_extrude.cc`'s `mesh_calc_hq_normal` in `float`.

use glam::Vec3;

/// Blender's `normalize_v3_v3_length` with unit length 1: the normalized vector and the original
/// length. A vector whose squared length is at most `1e-35` becomes zero, with length zero.
pub(super) fn normalize_length(vector: Vec3) -> (Vec3, f32) {
    let squared = vector.dot(vector);
    if squared > 1.0e-35 {
        let length = squared.sqrt();
        (vector * (1.0 / length), length)
    } else {
        (Vec3::ZERO, 0.0)
    }
}

/// Blender's `normalize_v3`.
pub(super) fn normalized(vector: Vec3) -> Vec3 {
    normalize_length(vector).0
}

/// Blender's `angle_normalized_v3v3`: the angle between two unit vectors, from their chord.
pub(super) fn angle_normalized(first: Vec3, second: Vec3) -> f32 {
    if first.dot(second) >= 0.0 {
        2.0 * safe_asin((first - second).length() / 2.0)
    } else {
        std::f32::consts::PI - 2.0 * safe_asin((first + second).length() / 2.0)
    }
}

fn safe_asin(value: f32) -> f32 {
    value.clamp(-1.0, 1.0).asin()
}

/// Blender's `angle_v3v3`: the angle between two vectors of any length.
pub(super) fn angle_between(first: Vec3, second: Vec3) -> f32 {
    angle_normalized(normalized(first), normalized(second))
}

/// Blender's `angle_signed_on_axis_v3v3_v3`: the angle from `first` to `second` seen along the unit
/// `axis`, in `[0, 2π)`.
pub(super) fn angle_signed_on_axis(first: Vec3, second: Vec3, axis: Vec3) -> f32 {
    let first_projected = first - axis * first.dot(axis);
    let second_projected = second - axis * second.dot(axis);
    let angle = angle_between(first_projected, second_projected);
    if second_projected.cross(first_projected).dot(axis) < 0.0 {
        std::f32::consts::TAU - angle
    } else {
        angle
    }
}

/// Blender's `mid_v3_v3v3_angle_weighted`: the direction midway between two unit normals, scaled
/// by their angle as `(2/π)·2 · acos(|a+b|/2)`. This is the normal of an edge shared by two faces.
pub(super) fn mid_angle_weighted(first: Vec3, second: Vec3) -> Vec3 {
    let (direction, length) = normalize_length(first + second);
    direction * (std::f32::consts::FRAC_2_PI * 2.0 * (length / 2.0).acos())
}

/// Blender's `shell_v3v3_normalized_to_dist`: how far a shell vertex moves along its normal to keep
/// the face's thickness, `1 / |cos|` of the angle between the normal and the face normal.
pub(super) fn shell_distance(normal: Vec3, face_normal: Vec3) -> f32 {
    let cosine = normal.dot(face_normal).abs();
    if cosine < 1.0e-8 { 1.0 } else { 1.0 / cosine }
}

/// Blender's `mesh_calc_hq_normal`. `face_edges[f][c]` is the edge of face `f`'s corner `c`. Each
/// edge contributes its normal to both endpoints: the angle-weighted midpoint of its two faces, or
/// the lone face's normal. A vertex whose sum vanishes keeps `vert_normals`. Also returns, per edge,
/// whether three or more faces use it; Blender's even-offset pass treats those corners as plain.
pub(super) fn high_quality_vertex_normals(
    vertex_count: usize,
    edges: &[[usize; 2]],
    face_edges: &[Vec<usize>],
    face_normals: &[Vec3],
    vert_normals: &[Vec3],
) -> (Vec<Vec3>, Vec<bool>) {
    #[derive(Clone, Copy)]
    enum EdgeUse {
        Unused,
        One(usize),
        Two(usize, usize),
        Many,
    }
    let mut uses = vec![EdgeUse::Unused; edges.len()];
    let mut non_manifold = vec![false; edges.len()];
    for (face, corner_edges) in face_edges.iter().enumerate() {
        for &edge in corner_edges {
            uses[edge] = match uses[edge] {
                EdgeUse::Unused => EdgeUse::One(face),
                EdgeUse::One(first) => EdgeUse::Two(first, face),
                EdgeUse::Two(..) | EdgeUse::Many => {
                    non_manifold[edge] = true;
                    EdgeUse::Many
                }
            };
        }
    }
    let mut sums = vec![Vec3::ZERO; vertex_count];
    for (edge, &[first, second]) in edges.iter().enumerate() {
        let normal = match uses[edge] {
            EdgeUse::One(face) => face_normals[face],
            EdgeUse::Two(first_face, second_face) => {
                mid_angle_weighted(face_normals[first_face], face_normals[second_face])
            }
            EdgeUse::Unused | EdgeUse::Many => continue,
        };
        sums[first] += normal;
        sums[second] += normal;
    }
    let normals = sums
        .iter()
        .enumerate()
        .map(|(vertex, sum)| {
            let (direction, length) = normalize_length(*sum);
            if length == 0.0 {
                vert_normals[vertex]
            } else {
                direction
            }
        })
        .collect();
    (normals, non_manifold)
}
