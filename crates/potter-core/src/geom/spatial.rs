//! Blender's `mesh_apply_spatial_organization` (`blenkernel/intern/mesh.cc`), which the voxel remesh operator
//! runs on its result (`object_remesh.cc`). It decides the stored order of vertices and faces: faces are
//! partitioned into a k-d tree of leaves of at most `TARGET_GROUP_SIZE` faces (split at the midpoint of the
//! dominant axis of their centres, and by material index), and each leaf, in tree order, adds its vertices
//! that no earlier leaf added, followed by its faces. Vertices no face uses come last.

use std::collections::BTreeSet;

use glam::Vec3;

/// `TARGET_GROUP_SIZE` in `mesh.cc`.
const TARGET_GROUP_SIZE: usize = 2500;
/// `STACK_FIXED_DEPTH` in `mesh.cc`: the recursion stops one level before this depth.
const STACK_FIXED_DEPTH: usize = 100;

/// Blender's `NonContiguousGroup`, for one tree node. Leaves carry faces and vertex lists; inner nodes
/// carry the index of their first child (children are always created in pairs after their parent).
#[derive(Default)]
struct Group {
    faces: Vec<usize>,
    children: Option<usize>,
    unique_verts: Vec<usize>,
    shared_verts: Vec<usize>,
}

/// The order Blender's `mesh_apply_spatial_organization` stores the mesh in: the vertex order and the
/// face order, each as indices into the input.
///
/// `faces` holds each face's corner vertex indices (into `positions`), and `materials` each face's material
/// index. A mesh without faces is returned unchanged, as Blender returns early.
pub(crate) fn spatial_order(
    positions: &[Vec3],
    faces: &[Vec<usize>],
    materials: &[u32],
) -> (Vec<usize>, Vec<usize>) {
    let vertex_count = positions.len();
    if vertex_count == 0 || faces.is_empty() {
        return ((0..vertex_count).collect(), (0..faces.len()).collect());
    }
    // `calc_face_bounds` per face, and its centre (`Bounds::center`, the f32 midpoint).
    let face_bounds: Vec<(Vec3, Vec3)> = faces
        .iter()
        .map(|face| {
            let mut min = positions[face[0]];
            let mut max = min;
            for &vertex in &face[1..] {
                min = min.min(positions[vertex]);
                max = max.max(positions[vertex]);
            }
            (min, max)
        })
        .collect();
    let centers: Vec<Vec3> = face_bounds
        .iter()
        .map(|(min, max)| midpoint(*min, *max))
        .collect();
    // The root's bounds are the merged face bounds of the whole mesh.
    let root_min = face_bounds
        .iter()
        .fold(Vec3::splat(f32::MAX), |min, (face_min, _)| {
            min.min(*face_min)
        });
    let root_max = face_bounds
        .iter()
        .fold(Vec3::splat(f32::MIN), |max, (_, face_max)| {
            max.max(*face_max)
        });

    let mut order: Vec<usize> = (0..faces.len()).collect();
    let mut groups = vec![Group::default()];
    partition(
        &centers,
        materials,
        &mut order,
        &mut groups,
        0,
        0,
        Some((root_min, root_max)),
    );

    // `build_vertex_groups_for_leaves`: a vertex belongs to the first leaf (in tree order) that uses it.
    let mut used = vec![false; vertex_count];
    for group in &mut groups {
        if group.children.is_some() || group.faces.is_empty() {
            continue;
        }
        let verts: BTreeSet<usize> = group
            .faces
            .iter()
            .flat_map(|&face| faces[face].iter().copied())
            .collect();
        for vertex in verts {
            if used[vertex] {
                group.shared_verts.push(vertex);
            } else {
                used[vertex] = true;
                group.unique_verts.push(vertex);
            }
        }
    }

    // The reorder: each group in tree order adds its unique vertices, then its shared ones not yet added,
    // then its faces; unused vertices come last.
    let mut added = vec![false; vertex_count];
    let mut vertex_order = Vec::with_capacity(vertex_count);
    let mut face_order = Vec::with_capacity(faces.len());
    for group in &groups {
        for &vertex in group.unique_verts.iter().chain(&group.shared_verts) {
            if !added[vertex] {
                added[vertex] = true;
                vertex_order.push(vertex);
            }
        }
        face_order.extend(group.faces.iter().copied());
    }
    for (vertex, done) in added.iter().enumerate() {
        if !done {
            vertex_order.push(vertex);
        }
    }
    (vertex_order, face_order)
}

/// Blender's `partition_faces_recursively`. `faces` is the slice of the face order this node owns, `node`
/// its tree index, and `bounds` the precomputed bounds of the root (children recompute from their centres).
fn partition(
    centers: &[Vec3],
    materials: &[u32],
    faces: &mut [usize],
    groups: &mut Vec<Group>,
    node: usize,
    depth: usize,
    bounds: Option<(Vec3, Vec3)>,
) {
    let small = faces.len() <= TARGET_GROUP_SIZE || depth >= STACK_FIXED_DEPTH - 1;
    if small && !needs_material_split(faces, materials) {
        groups[node].faces = faces.to_vec();
        return;
    }
    let children = groups.len();
    groups[node].children = Some(children);
    groups.push(Group::default());
    groups.push(Group::default());

    let split = if small {
        partition_material(materials, faces)
    } else {
        let (min, max) = bounds.unwrap_or_else(|| {
            faces.iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(min, max), &face| (min.min(centers[face]), max.max(centers[face])),
            )
        });
        let axis = dominant_axis(max - min);
        let middle = midpoint_component(min[axis], max[axis]);
        partition_along_axis(centers, faces, axis, middle)
    };
    let (front, back) = faces.split_at_mut(split);
    partition(centers, materials, front, groups, children, depth + 1, None);
    partition(
        centers,
        materials,
        back,
        groups,
        children + 1,
        depth + 1,
        None,
    );
}

/// Blender's `leaf_needs_material_split`: some face's material differs from the first face's.
fn needs_material_split(faces: &[usize], materials: &[u32]) -> bool {
    let Some(&first) = faces.first() else {
        return false;
    };
    faces
        .iter()
        .any(|&face| materials[face] != materials[first])
}

/// Blender's `partition_along_axis`: faces whose centre is at or beyond `middle` move to the front.
fn partition_along_axis(centers: &[Vec3], faces: &mut [usize], axis: usize, middle: f32) -> usize {
    std_partition(faces, |face| centers[face][axis] >= middle)
}

/// Blender's `partition_material_indices`: faces with the first face's material move to the front.
fn partition_material(materials: &[u32], faces: &mut [usize]) -> usize {
    let first = materials[faces[0]];
    std_partition(faces, |face| materials[face] == first)
}

/// `std::partition` with the swap sequence of libc++ and libstdc++ (both use the same two-pointer scan),
/// so the order of each side matches Blender's on the platforms it is built for.
fn std_partition(faces: &mut [usize], predicate: impl Fn(usize) -> bool) -> usize {
    let mut first = 0;
    let mut last = faces.len();
    loop {
        loop {
            if first == last {
                return first;
            }
            if !predicate(faces[first]) {
                break;
            }
            first += 1;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if predicate(faces[last]) {
                break;
            }
        }
        faces.swap(first, last);
        first += 1;
    }
}

/// Blender's `math::midpoint` for float3, component by component.
fn midpoint(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        midpoint_component(a.x, b.x),
        midpoint_component(a.y, b.y),
        midpoint_component(a.z, b.z),
    )
}

/// Blender's `math::midpoint` for one float component, `(a + b) * 0.5`. `f32::midpoint` is `(a + b) / 2`
/// for every finite value that does not overflow, which is the same operation (halving is exact).
fn midpoint_component(a: f32, b: f32) -> f32 {
    f32::midpoint(a, b)
}

/// Blender's `math::dominant_axis`: the axis of the largest absolute component, ties going to x, then y.
fn dominant_axis(v: Vec3) -> usize {
    let b = v.abs();
    if b.x > b.y {
        if b.x > b.z { 0 } else { 2 }
    } else if b.y > b.z {
        1
    } else {
        2
    }
}
