// SPDX-License-Identifier: GPL-3.0-or-later
//
// Portions derived from Blender (https://www.blender.org):
//   source/blender/blenkernel/intern/mesh_mirror.cc, source/blender/modifiers/intern/MOD_mirror.cc, source/blender/bmesh/tools/bmesh_bisect_plane.cc, source/blender/blenlib/intern/string_utils.cc, source/blender/blenkernel/intern/deform.cc, SPDX-FileCopyrightText: 2005 Blender Authors, GPL-2.0-or-later.

//! Mirror modifier with Blender 5.2.2 semantics.
//!
//! Each enabled axis is one pass over the mesh the previous pass left. A pass optionally bisects the
//! mesh at the axis plane (`use_bisect_axis`), copies every vertex through the mirror transform, merges
//! each copy into its own original when the two lie strictly within the merge threshold, and drops
//! polygons that the merge degenerates or duplicates. Copies reverse their winding, take side-swapped
//! vertex group names, reflect their custom normals, and flip their UVs. Blender's merge compares
//! original and copy only, so coincident vertices of different originals are never welded.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
};

use glam::{DMat3, DMat4, DVec3, DVec4};
use serde_json::{Value, json};

use super::super::{
    Mesh,
    clnor::{
        CornerNormalCode, Fans, NormalCodeOptions, SetterInput, corner_codes, corner_order,
        encode_vector_in_space, evaluate_fan_codes, fans, stored_corner_order, write_codes,
    },
    edge_key,
    normals::{face_normal, sharp_edge_flags, smooth_face_flags},
    provenance::{Origin, remap_attributes_with_edges},
};
use super::{
    Polygon, bool_param, build_topology, clean_polygon, edge_pairs, invalid_parameter,
    map_uv_values, number_param, parameter_value, polygons, positions, schema_value, validated,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    model::{Id, Modifier},
};

/// Parameters of one Mirror modifier, read once per evaluation.
struct Settings {
    axes: [bool; 3],
    bisect: [bool; 3],
    flip: [bool; 3],
    /// `Some(threshold)` when `use_mirror_merge` is on.
    merge: Option<f64>,
    bisect_threshold: f64,
    vertex_groups: bool,
    uv_flip: [bool; 2],
    uv_mirror_offset: [f64; 2],
    uv_offset: [f64; 2],
    uv_udim: bool,
    object: Option<ObjectMirror>,
}

/// The mirror object's transform relative to the owner, as Blender's `mesh_mirror.cc` builds it.
struct ObjectMirror {
    /// Owner local to mirror-object local space.
    to_mirror: DMat4,
    /// Mirror-object local to owner local space.
    from_mirror: DMat4,
    /// Owner's squared axis lengths, normalised by their maximum; scales the bisect normal.
    owner_scale: DVec3,
}

/// One axis pass: the copy transform and, when bisected, the cut plane as `(origin, normal)`.
struct AxisPlane {
    matrix: DMat4,
    bisect: Option<(DVec3, DVec3)>,
}

pub(super) fn mirror(
    mesh: &Mesh,
    modifier: &Modifier,
    owner_world_matrix: DMat4,
    object_world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<Mesh> {
    validated(mesh)?;
    let settings = Settings::parse(modifier, owner_world_matrix, object_world_matrices)?;
    let mut codes = corner_codes(mesh)?;
    let mut order = corner_order(mesh)?;
    let mut current: Option<Mesh> = None;
    for axis in (0..3).filter(|&axis| settings.axes[axis]) {
        let (result, output_codes, output_order) = mirror_axis(
            current.as_ref().unwrap_or(mesh),
            codes.as_deref(),
            order.as_deref(),
            axis,
            &settings,
        )?;
        current = Some(result);
        codes = output_codes;
        order = output_order;
    }
    let Some(mut result) = current else {
        return Ok(mesh.clone());
    };
    if codes.is_some() || order.is_some() {
        write_codes(
            &mut result,
            codes.as_deref(),
            NormalCodeOptions {
                corner_order: order.as_deref(),
                setter_input: SetterInput::AlreadyNormalized,
            },
        )?;
    }
    Ok(result)
}

impl Settings {
    fn parse(
        modifier: &Modifier,
        owner_world_matrix: DMat4,
        object_world_matrices: &BTreeMap<Id, DMat4>,
    ) -> Result<Self> {
        // Blender stores both thresholds as float32; the comparisons below are exact at their boundaries.
        let merge_threshold = number_param(modifier, "merge_threshold")? as f32 as f64;
        if merge_threshold < 0.0 {
            return Err(invalid_parameter(
                modifier,
                "merge_threshold",
                "a non-negative finite number",
            ));
        }
        let bisect_threshold = number_param(modifier, "bisect_threshold")? as f32 as f64;
        if bisect_threshold < 0.0 {
            return Err(invalid_parameter(
                modifier,
                "bisect_threshold",
                "a non-negative finite number",
            ));
        }
        // `use_clip` only constrains interactive vertex transforms in Blender; the modifier ignores it.
        bool_param(modifier, "use_clip")?;
        let object = match parameter_value(modifier, "mirror_object") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let name = value.as_str().ok_or_else(|| {
                    invalid_parameter(modifier, "mirror_object", "an object ID or null")
                })?;
                let id = Id::new(name).map_err(|_| {
                    invalid_parameter(modifier, "mirror_object", "a valid object ID")
                })?;
                let mirror_world = object_world_matrices.get(&id).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::TargetNotFound,
                        "mirror object was not found",
                        json!({"modifier_id": modifier.id, "mirror_object": id}),
                    )
                })?;
                let to_mirror = mirror_world.inverse() * owner_world_matrix;
                let from_mirror = to_mirror.inverse();
                if !to_mirror
                    .to_cols_array()
                    .iter()
                    .all(|value| value.is_finite())
                    || !from_mirror
                        .to_cols_array()
                        .iter()
                        .all(|value| value.is_finite())
                {
                    return Err(invalid_parameter(
                        modifier,
                        "mirror_object",
                        "an object with an invertible transform",
                    ));
                }
                let squared = [
                    owner_world_matrix.x_axis,
                    owner_world_matrix.y_axis,
                    owner_world_matrix.z_axis,
                ]
                .map(|axis| axis.truncate().length_squared());
                let largest = squared.into_iter().fold(0.0, f64::max);
                let owner_scale = if largest == 0.0 {
                    DVec3::ONE
                } else {
                    DVec3::from_array(squared.map(|value| value / largest))
                };
                Some(ObjectMirror {
                    to_mirror,
                    from_mirror,
                    owner_scale,
                })
            }
        };
        Ok(Self {
            axes: flags3(modifier, "use_axis")?,
            bisect: flags3(modifier, "use_bisect_axis")?,
            flip: flags3(modifier, "use_bisect_flip_axis")?,
            merge: bool_param(modifier, "use_mirror_merge")?.then_some(merge_threshold),
            bisect_threshold,
            vertex_groups: bool_param(modifier, "use_mirror_vertex_groups")?,
            uv_flip: [
                bool_param(modifier, "use_mirror_u")?,
                bool_param(modifier, "use_mirror_v")?,
            ],
            uv_mirror_offset: [
                number_param(modifier, "mirror_offset_u")?,
                number_param(modifier, "mirror_offset_v")?,
            ],
            uv_offset: [
                number_param(modifier, "offset_u")?,
                number_param(modifier, "offset_v")?,
            ],
            uv_udim: bool_param(modifier, "use_mirror_udim")?,
            object,
        })
    }

    /// The copy transform and cut plane of the pass over `axis`, as `MOD_mirror`/`mesh_mirror.cc` build
    /// them. Without a mirror object the mirror passes through the origin.
    fn plane(&self, axis: usize) -> AxisPlane {
        let mut diagonal = DVec4::ONE;
        diagonal[axis] = -1.0;
        let mut matrix = DMat4::from_diagonal(diagonal);
        let (origin, mut normal) = if let Some(object) = &self.object {
            matrix = object.from_mirror * matrix * object.to_mirror;
            let normal = object.from_mirror.col(axis).truncate() * object.owner_scale;
            (object.from_mirror.w_axis.truncate(), normal)
        } else {
            let mut unit = DVec3::ZERO;
            unit[axis] = 1.0;
            (DVec3::ZERO, unit)
        };
        // Blender keeps the side away from the mirrored copy unless the flip is set.
        if !self.flip[axis] {
            normal = -normal;
        }
        AxisPlane {
            matrix,
            bisect: self.bisect[axis].then_some((origin, normal)),
        }
    }
}

fn flags3(modifier: &Modifier, name: &str) -> Result<[bool; 3]> {
    let values = schema_value(modifier, name)?
        .as_array()
        .filter(|values| values.len() == 3)
        .ok_or_else(|| invalid_parameter(modifier, name, "three booleans"))?;
    let mut flags = [false; 3];
    for (flag, value) in flags.iter_mut().zip(values) {
        *flag = value
            .as_bool()
            .ok_or_else(|| invalid_parameter(modifier, name, "three booleans"))?;
    }
    Ok(flags)
}

/// One mirror pass: optional bisect, copy and merge, then rebuilt provenance and the copy-side
/// attributes (vertex groups, custom normals, UVs).
fn mirror_axis(
    mesh: &Mesh,
    input_codes: Option<&[CornerNormalCode]>,
    input_order: Option<&[usize]>,
    axis: usize,
    settings: &Settings,
) -> Result<(Mesh, Option<Vec<CornerNormalCode>>, Option<Vec<usize>>)> {
    let plane = settings.plane(axis);
    let needs_order = input_codes.is_some() || input_order.is_some();
    let bisected;
    let (source, source_codes, source_order): (
        &Mesh,
        Option<Cow<[CornerNormalCode]>>,
        Option<Cow<[usize]>>,
    ) = match plane.bisect {
        Some((origin, normal)) => {
            let (cut, corners) =
                bisect(mesh, origin, normal, settings.bisect_threshold, input_order)?;
            bisected = cut;
            let codes = input_codes.map(|codes| {
                Cow::Owned(
                    corners
                        .iter()
                        .map(|&corner| codes[corner])
                        .collect::<Vec<_>>(),
                )
            });
            let order = needs_order.then(|| Cow::Owned(stored_corner_order(&bisected)));
            (&bisected, codes, order)
        }
        None => {
            let order = match (needs_order, input_order) {
                (false, _) => None,
                (true, Some(order)) => Some(Cow::Borrowed(order)),
                (true, None) => Some(Cow::Owned(stored_corner_order(mesh))),
            };
            (mesh, input_codes.map(Cow::Borrowed), order)
        }
    };
    let count = source.vertices.len();

    // The doubled mesh before the merge: the originals, then the copies at their mirror images.
    let mut points: Vec<DVec3> = Vec::with_capacity(2 * count);
    points.extend(source.vertices.iter().map(|vertex| vertex.co));
    for vertex in 0..count {
        points.push(plane.matrix.transform_point3(points[vertex]));
    }
    // Blender merges copy `v + count` into original `v` when they are strictly closer than the
    // threshold, and both move to their midpoint. Copies that do not merge keep their own vertex.
    let mut target: Vec<usize> = (0..2 * count).collect();
    if let Some(threshold) = settings.merge {
        let limit = threshold * threshold;
        for vertex in 0..count {
            let copy = count + vertex;
            if points[vertex].distance_squared(points[copy]) < limit {
                let midpoint = points[vertex].lerp(points[copy], 0.5);
                points[vertex] = midpoint;
                points[copy] = midpoint;
                target[copy] = vertex;
            }
        }
    }
    let mut point_of = vec![0_usize; 2 * count];
    let mut out_points = Vec::with_capacity(2 * count);
    let mut out_origin = Vec::with_capacity(2 * count);
    for point in 0..2 * count {
        if target[point] == point {
            point_of[point] = out_points.len();
            out_points.push(points[point]);
            out_origin.push(vec![(point % count, 1.0)]);
        }
    }
    for point in 0..2 * count {
        point_of[point] = point_of[target[point]];
    }

    // The doubled faces in Blender's pre-merge order: the originals, then the copies. A copy keeps its
    // first loop and walks the rest backwards, so each corner names the source corner it copies.
    let source_polygons = polygons(source)?;
    let mut polygon_offsets = Vec::with_capacity(source_polygons.len());
    let mut corner_total = 0;
    for (vertices, _) in &source_polygons {
        polygon_offsets.push(corner_total);
        corner_total += vertices.len();
    }
    let mut doubled: Vec<Vec<usize>> = Vec::with_capacity(2 * source_polygons.len());
    let mut doubled_corners: Vec<Vec<usize>> = Vec::with_capacity(2 * source_polygons.len());
    // (material, source polygon, copy) per doubled face.
    let mut doubled_faces: Vec<(u32, usize, bool)> = Vec::with_capacity(2 * source_polygons.len());
    for copy in [false, true] {
        for (face, (vertices, material)) in source_polygons.iter().enumerate() {
            let order: Vec<usize> = if copy {
                std::iter::once(0)
                    .chain((1..vertices.len()).rev())
                    .collect()
            } else {
                (0..vertices.len()).collect()
            };
            let base = if copy { count } else { 0 };
            doubled.push(
                order
                    .iter()
                    .map(|&position| base + vertices[position])
                    .collect(),
            );
            doubled_corners.push(
                order
                    .iter()
                    .map(|&position| polygon_offsets[face] + position)
                    .collect(),
            );
            doubled_faces.push((*material, face, copy));
        }
    }

    // Blender's weld context removes duplicate affected faces by their mapped edge sets; untouched
    // faces stay outside it (`weld_poly_find_doubles` in `mesh_merge_verts.cc`).
    let mut touched_face_edge_sets = HashSet::<Vec<(usize, usize)>>::new();
    let mut out_faces: Vec<(Vec<usize>, u32, usize)> = Vec::with_capacity(doubled.len());
    let mut out_copy: Vec<bool> = Vec::with_capacity(doubled.len());
    let mut corner_map: Vec<usize> = Vec::with_capacity(2 * corner_total);
    let mut corner_base = 0;
    for (face, &(material, source_face, copy)) in doubled.iter().zip(&doubled_faces) {
        let base = corner_base;
        corner_base += face.len();
        if copy && face.iter().all(|&vertex| target[vertex] != vertex) {
            continue;
        }
        let mapped: Vec<usize> = face.iter().map(|&vertex| point_of[vertex]).collect();
        let Some(clean) = clean_polygon(mapped) else {
            continue;
        };
        if clean.len() != face.len() {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "mirror face lost a corner while merging",
            ));
        }
        let touches_weld = settings.merge.is_some()
            && face.iter().any(|&vertex| {
                let source_vertex = if vertex < count {
                    vertex
                } else {
                    vertex - count
                };
                target[count + source_vertex] != count + source_vertex
            });
        if touches_weld {
            let mut edge_set: Vec<_> = clean
                .iter()
                .zip(clean.iter().cycle().skip(1))
                .map(|(&first, &second)| edge_key(first, second))
                .collect();
            edge_set.sort_unstable();
            if !touched_face_edge_sets.insert(edge_set) {
                continue;
            }
        }
        corner_map.extend(base..base + face.len());
        out_faces.push((clean, material, source_face));
        out_copy.push(copy);
    }

    // Blender `mesh_mirror.cc` copies each source edge's attributes onto its mirrored copy
    // (`edge_interp.copy(0, src_edges_num, src_edges_num)`). Merged edges: first source wins here.
    let edges = edge_pairs(source)?;
    let mut edge_list = Vec::with_capacity(2 * edges.len());
    for offset in [0, count] {
        for (edge, [first, second]) in edges.iter().enumerate() {
            edge_list.push((
                [point_of[offset + first], point_of[offset + second]],
                Some(edge),
            ));
        }
    }
    let mut result = assemble(source, out_points, out_origin, out_faces, edge_list)?;

    if settings.vertex_groups {
        let mut members = vec![Vec::new(); result.vertices.len()];
        for point in 0..2 * count {
            members[point_of[point]].push(point);
        }
        mirror_vertex_groups(source, &mut result, &members, count);
    }
    let doubled_order = source_order.as_ref().map(|source_order| {
        doubled_loop_order(
            source_order,
            &source_polygons,
            &polygon_offsets,
            &doubled_corners,
            &doubled_faces,
        )
    });
    let output_order = doubled_order.as_ref().map(|numbering| {
        let mut corner_at_ordinal = vec![usize::MAX; numbering.len()];
        for (output_corner, &doubled_corner) in corner_map.iter().enumerate() {
            corner_at_ordinal[numbering[doubled_corner]] = output_corner;
        }
        let mut order = vec![0; corner_map.len()];
        let mut ordinal = 0;
        for output_corner in corner_at_ordinal {
            if output_corner != usize::MAX {
                order[output_corner] = ordinal;
                ordinal += 1;
            }
        }
        order
    });
    let output_codes = if let Some(source_codes) = &source_codes {
        let smooth = smooth_face_flags(source)?;
        let mut sharp = HashSet::new();
        for (&[first, second], flag) in edges.iter().zip(sharp_edge_flags(source)?) {
            if flag {
                sharp.insert(edge_key(first, second));
                sharp.insert(edge_key(count + first, count + second));
            }
        }
        let doubled_smooth: Vec<bool> = doubled_faces
            .iter()
            .map(|&(_, face, _)| smooth[face])
            .collect();
        let doubled_order = doubled_order.as_deref().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "mirror codes have no loop numbering",
            )
        })?;
        let doubled_fans = fans(&points, &doubled, &doubled_smooth, &sharp, doubled_order);
        let corner_source: Vec<usize> = doubled_corners.iter().flatten().copied().collect();
        let copy_corner: Vec<bool> = doubled_corners
            .iter()
            .zip(&doubled_faces)
            .flat_map(|(corners, &(_, _, copy))| std::iter::repeat_n(copy, corners.len()))
            .collect();
        let normal_matrix = DMat3::from_mat4(plane.matrix.inverse()).transpose();
        let doubled_codes = doubled_custom_codes(
            &doubled_fans,
            source_codes,
            &corner_source,
            &copy_corner,
            normal_matrix,
        )?;
        Some(
            corner_map
                .iter()
                .map(|&corner| doubled_codes[corner])
                .collect(),
        )
    } else {
        None
    };
    result.attributes.remove("custom_normal");
    map_uv_values(&mut result, |face, _, uv| {
        if out_copy[face] {
            mirror_uv_pass(
                uv,
                settings.uv_flip,
                settings.uv_mirror_offset,
                settings.uv_offset,
                settings.uv_udim,
            )
        } else {
            uv
        }
    });
    Ok((result, output_codes, output_order))
}

/// Blender's `BKE_mesh_mirror_bisect_on_mirror_plane_for_modifier`: cut the mesh at the plane, move
/// vertices within `threshold` of it onto it, and remove everything more than `threshold` beyond it.
/// `normal` is not normalised, as in Blender, so the threshold scales with it. Returns the cut mesh and,
/// for each of its corners, the corner of `mesh` whose custom-normal code it copies.
/// A stored incoming loop basis also determines face slots and first loops; the rebuilt mesh then
/// uses the bisect output order.
fn bisect(
    mesh: &Mesh,
    origin: DVec3,
    normal: DVec3,
    threshold: f64,
    input_order: Option<&[usize]>,
) -> Result<(Mesh, Vec<usize>)> {
    let count = mesh.vertices.len();
    let offset = -normal.dot(origin);
    let side_of = |point: DVec3| normal.dot(point) + offset;
    let mut points = positions(mesh);
    let edges = edge_pairs(mesh)?;
    let side: Vec<f64> = points.iter().map(|&point| side_of(point)).collect();
    let mut directions = Vec::with_capacity(count + edges.len());
    directions.extend(side.iter().map(|&value| {
        if value <= -threshold {
            -1
        } else {
            i8::from(value >= threshold)
        }
    }));
    let length_squared = normal.length_squared();
    if length_squared > 0.0 {
        for vertex in 0..count {
            if directions[vertex] == 0 {
                points[vertex] -= normal * (side[vertex] / length_squared);
            }
        }
    }

    // Cut every edge whose endpoints lie strictly on opposite sides.
    let mut cuts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut origins: Vec<Vec<(usize, f64)>> =
        (0..count).map(|vertex| vec![(vertex, 1.0)]).collect();
    for &[first, second] in &edges {
        if directions[first] * directions[second] == -1 {
            let fraction = side[first] / (side[first] - side[second]);
            let cut = points.len();
            points.push(points[first].lerp(points[second], fraction));
            directions.push(0);
            origins.push(vec![(first, 1.0 - fraction), (second, fraction)]);
            cuts.insert(edge_key(first, second), cut);
        }
    }

    let mut source_polygons = polygons(mesh)?;
    let mut source_offsets = Vec::with_capacity(source_polygons.len());
    let mut first_loops = Vec::with_capacity(source_polygons.len());
    let mut rotations = Vec::with_capacity(source_polygons.len());
    let mut source_offset = 0;
    for (vertices, _) in &source_polygons {
        source_offsets.push(source_offset);
        let (first, rotation) = if let Some(order) = input_order {
            order[source_offset..source_offset + vertices.len()]
                .iter()
                .enumerate()
                .min_by_key(|(_, ordinal)| **ordinal)
                .map_or((source_offset, 0), |(rotation, &first)| (first, rotation))
        } else {
            (source_offset, 0)
        };
        first_loops.push(first);
        rotations.push(rotation);
        source_offset += vertices.len();
    }
    let mut face_order: Vec<usize> = (0..source_polygons.len()).collect();
    if input_order.is_some() {
        face_order.sort_by_key(|&face| first_loops[face]);
    }
    let stack_pop_order = bisect_face_pop_order(&edges, &source_polygons, &directions, count)?;

    let mut chords = Vec::new();
    // Blender keeps a split face's first piece in its slot; the other pieces are created after the
    // original faces (see below).
    let mut slots = Vec::with_capacity(mesh.faces.len());
    let mut added_by_face = vec![Vec::new(); source_polygons.len()];
    for face in face_order {
        let (mut vertices, material) = std::mem::take(&mut source_polygons[face]);
        let offset = source_offsets[face];
        let rotation = rotations[face];
        vertices.rotate_left(rotation);
        let mut cycle = Vec::with_capacity(2 * vertices.len());
        // A cut vertex copies the corner at the start of its edge: `BM_edge_split` copies the loop it
        // splits, and that loop sits at the edge's start in the face's winding.
        let mut cycle_source = Vec::with_capacity(2 * vertices.len());
        for (index, &vertex) in vertices.iter().enumerate() {
            let next = vertices[(index + 1) % vertices.len()];
            let source_corner = offset + (rotation + index) % vertices.len();
            cycle.push(vertex);
            cycle_source.push(source_corner);
            if let Some(&cut) = cuts.get(&edge_key(vertex, next)) {
                cycle.push(cut);
                cycle_source.push(source_corner);
            }
        }
        let source_of: HashMap<usize, usize> = cycle.iter().copied().zip(cycle_source).collect();
        let mut created = Vec::new();
        for (index, piece) in split_face(&cycle, &points, &directions, normal, &mut chords)
            .into_iter()
            .enumerate()
        {
            let sources: Vec<usize> = piece.iter().map(|vertex| source_of[vertex]).collect();
            let entry = (piece, material, face, sources);
            if index == 0 {
                slots.push(entry);
            } else {
                created.push(entry);
            }
        }
        added_by_face[face] = created;
    }
    // Blender appends created faces in the LIFO bisect stack's pop order; the kept pieces above
    // remain in their original face slots.
    let mut added = Vec::new();
    for face in stack_pop_order {
        added.extend(std::mem::take(&mut added_by_face[face]));
    }

    // Vertices beyond the threshold go, and every polygon, edge, or chord that uses them with them.
    let killed: Vec<bool> = (0..points.len())
        .map(|vertex| vertex < count && side_of(points[vertex]) - threshold > 0.0)
        .collect();
    let mut compact = vec![usize::MAX; points.len()];
    let mut out_points = Vec::new();
    let mut out_origin = Vec::new();
    for vertex in 0..points.len() {
        if !killed[vertex] {
            compact[vertex] = out_points.len();
            out_points.push(points[vertex]);
            out_origin.push(std::mem::take(&mut origins[vertex]));
        }
    }
    let mut out_faces: Vec<(Vec<usize>, u32, usize)> =
        Vec::with_capacity(slots.len() + added.len());
    let mut out_sources: Vec<usize> = Vec::new();
    for (piece, material, face, sources) in slots.into_iter().chain(added) {
        let mapped: Option<Vec<usize>> = piece
            .iter()
            .map(|&vertex| (!killed[vertex]).then(|| compact[vertex]))
            .collect();
        if let Some(mapped) = mapped {
            out_faces.push((mapped, material, face));
            out_sources.extend(sources);
        }
    }
    // Blender 5.2.2 `BM_edge_split` (bmesh/intern/bmesh_mods.cc) copies the split edge's hflag and
    // CustomData onto the new half (`BM_elem_attrs_copy(bm, e, e_new)`), so both halves keep the
    // source. A chord comes from `bm_face_bisect_verts` (bmesh/tools/bmesh_bisect_plane.cc) calling
    // `BM_face_split(..., nullptr, true)`, a null example, so `BM_edge_create` sets defaults: no source.
    let mut edge_list = Vec::with_capacity(edges.len() + chords.len());
    for (edge, &[first, second]) in edges.iter().enumerate() {
        match cuts.get(&edge_key(first, second)) {
            Some(&cut) => {
                edge_list.push(([first, cut], Some(edge)));
                edge_list.push(([cut, second], Some(edge)));
            }
            None => edge_list.push(([first, second], Some(edge))),
        }
    }
    edge_list.extend(chords.into_iter().map(|chord| (chord, None)));
    let out_edges: Vec<([usize; 2], Option<usize>)> = edge_list
        .into_iter()
        .filter(|([first, second], _)| !killed[*first] && !killed[*second])
        .map(|([first, second], source)| ([compact[first], compact[second]], source))
        .collect();
    let result = assemble(mesh, out_points, out_origin, out_faces, out_edges)?;
    Ok((result, out_sources))
}

/// Returns faces in the order `BM_mesh_bisect_plane` pops them: the mesh-edge traversal first pushes
/// each incident face once, then the BMesh face stack reverses that insertion order.
fn bisect_face_pop_order(
    edges: &[[usize; 2]],
    polygons: &[(Vec<usize>, u32)],
    directions: &[i8],
    vertex_count: usize,
) -> Result<Vec<usize>> {
    let edge_index: HashMap<_, _> = edges
        .iter()
        .enumerate()
        .map(|(index, &[first, second])| (edge_key(first, second), index))
        .collect();
    let mut edge_face_counts = vec![0; edges.len()];
    let mut vertex_edge_counts = vec![0; vertex_count];
    for &[first, second] in edges {
        vertex_edge_counts[first] += 1;
        vertex_edge_counts[second] += 1;
    }
    for (vertices, _) in polygons {
        for (&first, &second) in vertices.iter().zip(vertices.iter().cycle().skip(1)) {
            let edge = *edge_index.get(&edge_key(first, second)).ok_or_else(|| {
                PotError::new(
                    ErrorCode::InvalidArgument,
                    "mirror face references a missing edge",
                )
            })?;
            edge_face_counts[edge] += 1;
        }
    }

    let mut edge_face_offsets = Vec::with_capacity(edges.len() + 1);
    let mut edge_face_total = 0;
    edge_face_offsets.push(edge_face_total);
    for count in edge_face_counts {
        edge_face_total += count;
        edge_face_offsets.push(edge_face_total);
    }
    let mut vertex_edge_offsets = Vec::with_capacity(vertex_count + 1);
    let mut vertex_edge_total = 0;
    vertex_edge_offsets.push(vertex_edge_total);
    for count in vertex_edge_counts {
        vertex_edge_total += count;
        vertex_edge_offsets.push(vertex_edge_total);
    }

    let mut edge_face_cursors = edge_face_offsets[..edges.len()].to_vec();
    let mut vertex_edge_cursors = vertex_edge_offsets[..vertex_count].to_vec();
    let mut edge_faces = vec![0; edge_face_offsets[edges.len()]];
    let mut vertex_edges = vec![0; vertex_edge_offsets[vertex_count]];
    // BMesh radial iteration starts at the most recently appended face loop and walks backward.
    for (face, (vertices, _)) in polygons.iter().enumerate().rev() {
        for (&first, &second) in vertices.iter().zip(vertices.iter().cycle().skip(1)) {
            let edge = edge_index[&edge_key(first, second)];
            edge_faces[edge_face_cursors[edge]] = face;
            edge_face_cursors[edge] += 1;
        }
    }
    for (edge, &[first, second]) in edges.iter().enumerate() {
        vertex_edges[vertex_edge_cursors[first]] = edge;
        vertex_edge_cursors[first] += 1;
        vertex_edges[vertex_edge_cursors[second]] = edge;
        vertex_edge_cursors[second] += 1;
    }

    let mut face_stacked = vec![false; polygons.len()];
    let mut center_processed = vec![false; vertex_count];
    let mut stack = Vec::with_capacity(polygons.len());
    for (edge, &[first, second]) in edges.iter().enumerate() {
        if directions[first] * directions[second] == -1 {
            for &face in &edge_faces[edge_face_offsets[edge]..edge_face_offsets[edge + 1]] {
                if !face_stacked[face] {
                    face_stacked[face] = true;
                    stack.push(face);
                }
            }
        } else {
            for vertex in [first, second] {
                if directions[vertex] != 0 || center_processed[vertex] {
                    continue;
                }
                center_processed[vertex] = true;
                for &edge in
                    &vertex_edges[vertex_edge_offsets[vertex]..vertex_edge_offsets[vertex + 1]]
                {
                    for &face in &edge_faces[edge_face_offsets[edge]..edge_face_offsets[edge + 1]] {
                        if !face_stacked[face] {
                            face_stacked[face] = true;
                            stack.push(face);
                        }
                    }
                }
            }
        }
    }
    stack.reverse();
    Ok(stack)
}

/// Splits a face at the plane. `cycle` holds the face's vertices in order, with the cut vertices on
/// its plane-crossing edges already inserted. The cut follows `bmesh_bisect_plane.cc`: chords join
/// on-plane vertices and each chord is recorded in `chords`. Faces without two on-plane vertices and
/// vertices on both sides are returned unchanged.
fn split_face(
    cycle: &[usize],
    points: &[DVec3],
    directions: &[i8],
    plane_normal: DVec3,
    chords: &mut Vec<[usize; 2]>,
) -> Vec<Vec<usize>> {
    let length = cycle.len();
    let unchanged = || vec![cycle.to_vec()];
    let mut centers = Vec::new();
    let (mut has_negative, mut has_positive) = (false, false);
    for (index, &vertex) in cycle.iter().enumerate() {
        match directions[vertex] {
            -1 => has_negative = true,
            0 => centers.push(index),
            1 => has_positive = true,
            _ => unreachable!("bisect directions are -1, 0, or 1"),
        }
    }
    if centers.len() < 2 || !has_negative || !has_positive {
        return unchanged();
    }
    let direction = |index: usize| directions[cycle[index]];
    let prev = |index: usize| (index + length - 1) % length;
    let next = |index: usize| (index + 1) % length;
    let mut skip = vec![false; length];
    for &index in &centers {
        skip[index] = direction(prev(index)) == direction(next(index));
    }
    if centers.len() == 2 {
        let (first, second) = (cycle[centers[0]], cycle[centers[1]]);
        return match split_cycle(cycle, first, second) {
            Some((kept, added)) => {
                chords.push([first, second]);
                vec![kept, added]
            }
            None => unchanged(),
        };
    }

    let has_center_edge = centers.iter().any(|&index| direction(prev(index)) == 0);
    if has_center_edge {
        // Spans of on-plane vertices whose outer neighbours share a side are skipped as a whole.
        let Some(first_non_center) = (0..length).find(|&index| direction(index) != 0) else {
            return unchanged();
        };
        let mut current = first_non_center;
        loop {
            if !skip[current] && direction(current) == 0 && direction(next(current)) == 0 {
                let before = prev(current);
                let mut after = next(next(current));
                while direction(after) == 0 {
                    after = next(after);
                }
                if direction(before) == direction(after) {
                    let mut inner = next(before);
                    while inner != after {
                        skip[inner] = true;
                        inner = next(inner);
                    }
                }
                current = prev(after);
            }
            current = next(current);
            if current == first_non_center {
                break;
            }
        }
    }

    let face_points: Vec<DVec3> = cycle.iter().map(|&vertex| points[vertex]).collect();
    let sort_direction = face_normal(&face_points).cross(plane_normal);
    let Some(sort_direction) = sort_direction.try_normalize() else {
        return unchanged();
    };
    centers.sort_by(|&first, &second| {
        let key = |index: usize| sort_direction.dot(points[cycle[index]]);
        key(first).total_cmp(&key(second))
    });

    let adjacent_in_cycle = |first: usize, second: usize| {
        let delta = first.abs_diff(second);
        delta == 1 || delta == length - 1
    };
    let mut pieces = vec![cycle.to_vec()];
    let mut inside = false;
    for pair in centers.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        if has_center_edge && adjacent_in_cycle(first, second) {
            continue;
        }
        if !skip[first] {
            inside = !inside;
        }
        if !inside {
            continue;
        }
        let (first_vertex, second_vertex) = (cycle[first], cycle[second]);
        let Some(index) = pieces
            .iter()
            .position(|piece| piece.contains(&first_vertex) && piece.contains(&second_vertex))
        else {
            continue;
        };
        if let Some((kept, added)) = split_cycle(&pieces[index], first_vertex, second_vertex) {
            pieces[index] = kept;
            pieces.push(added);
            chords.push([first_vertex, second_vertex]);
        }
    }
    pieces
}

/// Splits a polygon along a chord as `BM_face_split` does. The old face follows `first` to `second`
/// in loop order; the new face follows the complement. Each face starts at the loop BMesh keeps as
/// its `l_first`, which is significant for closed custom-normal fans.
fn split_cycle(cycle: &[usize], first: usize, second: usize) -> Option<(Vec<usize>, Vec<usize>)> {
    let a = cycle.iter().position(|&vertex| vertex == first)?;
    let b = cycle.iter().position(|&vertex| vertex == second)?;
    let length = cycle.len();
    if (a + 1) % length == b || (b + 1) % length == a {
        return None;
    }
    let walk = |start: usize, end: usize| {
        let mut path = vec![cycle[start]];
        let mut index = start;
        while index != end {
            index = (index + 1) % length;
            path.push(cycle[index]);
        }
        path
    };
    let mut old_face = walk(a, b);
    let mut new_face = walk(b, a);
    let original_first = cycle[0];
    let old_contains_first = old_face.contains(&original_first);
    let old_start = if old_contains_first {
        original_first
    } else if original_first == cycle[b] {
        cycle[(b + length - 1) % length]
    } else if original_first == cycle[(a + length - 1) % length] {
        cycle[a]
    } else {
        cycle[b]
    };
    let new_start = if !old_contains_first {
        original_first
    } else if original_first == cycle[a] {
        cycle[(a + length - 1) % length]
    } else if original_first == cycle[(b + length - 1) % length] {
        cycle[b]
    } else {
        cycle[a]
    };
    let old_index = old_face.iter().position(|&vertex| vertex == old_start)?;
    let new_index = new_face.iter().position(|&vertex| vertex == new_start)?;
    old_face.rotate_left(old_index);
    new_face.rotate_left(new_index);
    Some((old_face, new_face))
}

/// Builds the rebuilt mesh of a pass with its provenance. `faces` are `(vertices, material, source
/// polygon)` in output order. `edges` are `(vertices, source edge)` with the first entry of a
/// duplicate pair winning; an edge no face uses is passed as loose.
fn assemble(
    source: &Mesh,
    points: Vec<DVec3>,
    origin_vertices: Vec<Vec<(usize, f64)>>,
    faces: Vec<(Vec<usize>, u32, usize)>,
    edges: Vec<([usize; 2], Option<usize>)>,
) -> Result<Mesh> {
    let mut source_of: HashMap<(usize, usize), Option<usize>> = HashMap::new();
    let mut unique = Vec::with_capacity(edges.len());
    for (pair, edge_source) in edges {
        if pair[0] == pair[1] {
            continue;
        }
        let key = edge_key(pair[0], pair[1]);
        if let std::collections::hash_map::Entry::Vacant(entry) = source_of.entry(key) {
            entry.insert(edge_source);
            unique.push((pair, edge_source));
        }
    }
    let face_edges: HashSet<(usize, usize)> = faces
        .iter()
        .flat_map(|(vertices, _, _)| {
            vertices
                .iter()
                .zip(vertices.iter().cycle().skip(1))
                .map(|(&first, &second)| edge_key(first, second))
        })
        .collect();
    let loose: Vec<[usize; 2]> = unique
        .iter()
        .filter(|(pair, _)| !face_edges.contains(&edge_key(pair[0], pair[1])))
        .map(|(pair, _)| *pair)
        .collect();
    let origin_faces: Vec<Option<usize>> = faces.iter().map(|(_, _, face)| Some(*face)).collect();
    let polygons: Vec<Polygon> = faces
        .into_iter()
        .map(|(vertices, material, _)| (vertices, material))
        .collect();
    let mut result = build_topology(points, polygons, loose)?;
    let index_of: HashMap<u32, usize> = result
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let explicit: Vec<Vec<(usize, f64)>> = result
        .edges
        .iter()
        .map(|edge| {
            let (Some(&first), Some(&second)) = (
                index_of.get(&edge.vertices[0]),
                index_of.get(&edge.vertices[1]),
            ) else {
                return Vec::new();
            };
            match source_of.get(&edge_key(first, second)) {
                Some(Some(edge_source)) => vec![(*edge_source, 1.0)],
                _ => Vec::new(),
            }
        })
        .collect();
    let origin = Origin {
        vertices: origin_vertices,
        faces: origin_faces,
        corners: None,
    };
    remap_attributes_with_edges(source, &mut result, &origin, explicit)?;
    validated(&result)?;
    Ok(result)
}

/// Mirror copies swap `.L`/`.R` group names and merged plane vertices average them. Blender's flip
/// map pairs each group with its side-swapped name, and a merged vertex gets the average of its
/// original's and copy's weights for each pair (`BKE_defvert_flip_merged`).
///
/// `members[o]` lists the pass's input points merged into output vertex `o`; point `c * count + v` is
/// source vertex `v` in copy `c`, and an odd `c` is a flipped copy.
fn mirror_vertex_groups(source: &Mesh, output: &mut Mesh, members: &[Vec<usize>], count: usize) {
    let Some(source_groups) = source
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
    else {
        return;
    };
    let names: Vec<&str> = source_groups.keys().map(String::as_str).collect();
    let flips = flip_map(&names);
    let group_values: Vec<&Value> = source_groups.values().collect();
    let source_keys: Vec<String> = source
        .vertices
        .iter()
        .map(|vertex| format!("v{}", vertex.id))
        .collect();
    // Summed weight per group and output vertex; `None` when no copy has an entry for the group.
    let mut sums = vec![vec![None::<f64>; output.vertices.len()]; names.len()];
    for (vertex, copies) in members.iter().enumerate().take(output.vertices.len()) {
        let share = 1.0 / copies.len() as f64;
        for &copy in copies {
            let flipped = copy / count % 2 == 1;
            let source_key = source_keys[copy % count].as_str();
            for (group, values) in group_values.iter().enumerate() {
                let Some(weight) = values.get(source_key).and_then(Value::as_f64) else {
                    continue;
                };
                let target = if flipped {
                    flips[group].unwrap_or(group)
                } else {
                    group
                };
                let slot = &mut sums[target][vertex];
                *slot = Some(slot.unwrap_or(0.0) + weight * share);
            }
        }
    }
    let groups: serde_json::Map<String, Value> = names
        .iter()
        .zip(sums)
        .map(|(name, sums)| {
            let weights: serde_json::Map<String, Value> = sums
                .into_iter()
                .enumerate()
                .filter_map(|(vertex, sum)| {
                    Some((format!("v{}", output.vertices[vertex].id), json!(sum?)))
                })
                .collect();
            ((*name).to_owned(), Value::Object(weights))
        })
        .collect();
    output
        .attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));
}

/// `BKE_object_defgroup_flip_map`: group `i` pairs with the group whose name is its side-swapped
/// name, when that group exists. Pairs are symmetric; unpaired groups map to `None`.
fn flip_map(names: &[&str]) -> Vec<Option<usize>> {
    let position: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect();
    let mut map = vec![None; names.len()];
    for (index, name) in names.iter().enumerate() {
        if map[index].is_some() {
            continue;
        }
        let flipped = flip_side_name(name);
        if let Some(&other) = position.get(flipped.as_str())
            && flipped != *name
        {
            map[index] = Some(other);
            map[other] = Some(index);
        }
    }
    map
}

/// `BLI_string_flip_side_name` with `strip_number` off: swaps a left/right marker in `name`. Blender
/// works on bytes, so the marker search and its lengths count bytes too. Every cut falls on an ASCII
/// byte, so the result stays UTF-8 and non-ASCII text around a marker is kept.
fn flip_side_name(name: &str) -> String {
    if name.len() < 3 {
        return name.to_owned();
    }
    let bytes = name.as_bytes();
    let mut body = name;
    let mut number = "";
    if let Some(dot) = name.rfind('.')
        && bytes.last().is_some_and(u8::is_ascii_digit)
        && bytes.get(dot + 1).is_some_and(u8::is_ascii_digit)
    {
        number = &name[dot..];
        body = &name[..dot];
    }
    let len = body.len();
    let body_bytes = body.as_bytes();
    let is_sep = |byte: u8| matches!(byte, b'.' | b' ' | b'-' | b'_');
    let side = |byte: u8| match byte {
        b'l' => Some("r"),
        b'r' => Some("l"),
        b'L' => Some("R"),
        b'R' => Some("L"),
        _ => None,
    };
    let mut prefix = body.to_owned();
    let mut suffix = String::new();
    let mut replace: Option<&str> = None;

    if len > 1 && is_sep(body_bytes[len - 2]) {
        if let Some(replacement) = side(body_bytes[len - 1]) {
            prefix = body[..len - 1].to_owned();
            replace = Some(replacement);
        }
    }
    if replace.is_none() && len > 1 && is_sep(body_bytes[1]) {
        if let Some(replacement) = side(body_bytes[0]) {
            suffix = body[1..].to_owned();
            prefix.clear();
            replace = Some(replacement);
        }
    }
    if replace.is_none() && len > 5 {
        let lower = body.to_ascii_lowercase();
        let found_right = lower.find("right");
        let found_left = lower.find("left");
        if let Some(index) = found_right.filter(|&index| index == 0 || index == len - 5) {
            let marker = &body[index..index + 5];
            let replacement = if marker.starts_with('r') {
                "left"
            } else if marker.as_bytes()[1] == b'I' {
                "LEFT"
            } else {
                "Left"
            };
            suffix = body[index + 5..].to_owned();
            prefix = body[..index].to_owned();
            replace = Some(replacement);
        } else if let Some(index) = found_left.filter(|&index| index == 0 || index == len - 4) {
            let marker = &body[index..index + 4];
            let replacement = if marker.starts_with('l') {
                "right"
            } else if marker.as_bytes()[1] == b'E' {
                "RIGHT"
            } else {
                "Right"
            };
            suffix = body[index + 4..].to_owned();
            prefix = body[..index].to_owned();
            replace = Some(replacement);
        }
    }
    match replace {
        Some(replacement) => format!("{prefix}{replacement}{suffix}{number}"),
        None => name.to_owned(),
    }
}

/// Blender's custom-normal codes after a mirror pass (`BKE_mesh_mirror_apply_mirror_on_axis_for_modifier`).
/// Each original corner keeps its source code; each copy encodes the reflected evaluated normal of its
/// source corner in the copy's pre-merge fan. The final merged fans evaluate and average these codes once.
fn doubled_custom_codes(
    fans: &Fans,
    source_codes: &[CornerNormalCode],
    corner_source: &[usize],
    copy_corner: &[bool],
    normal_matrix: DMat3,
) -> Result<Vec<CornerNormalCode>> {
    let mut codes: Vec<CornerNormalCode> = corner_source
        .iter()
        .zip(copy_corner)
        .map(|(&source, &copy)| {
            if copy {
                CornerNormalCode::Auto
            } else {
                source_codes[source]
            }
        })
        .collect();
    let evaluated = evaluate_fan_codes(fans, &codes)?;
    let mut original = vec![0_usize; source_codes.len()];
    for (corner, (&source, &copy)) in corner_source.iter().zip(copy_corner).enumerate() {
        if !copy {
            original[source] = corner;
        }
    }
    for (corner, (&source, &copy)) in corner_source.iter().zip(copy_corner).enumerate() {
        if copy {
            codes[corner] = if source_codes[source] == CornerNormalCode::ExplicitZero {
                CornerNormalCode::ExplicitZero
            } else {
                let mirrored = normal_matrix * evaluated[original[source]];
                if mirrored == DVec3::ZERO {
                    CornerNormalCode::Auto
                } else {
                    encode_vector_in_space(&fans.spaces[fans.fan_of[corner]], mirrored)
                }
            };
        }
    }
    Ok(codes)
}

/// Mirrors each source face's logical loop ranks, rather than its storage-order corner indices.
fn doubled_loop_order(
    source_order: &[usize],
    source_polygons: &[Polygon],
    polygon_offsets: &[usize],
    doubled_corners: &[Vec<usize>],
    doubled_faces: &[(u32, usize, bool)],
) -> Vec<usize> {
    let corner_total = source_order.len();
    let mut source_corners: Vec<usize> = (0..corner_total).collect();
    let mut loop_rank = vec![0; corner_total];
    let mut face_base = vec![0; source_polygons.len()];
    for face in 0..source_polygons.len() {
        let start = polygon_offsets[face];
        let end = start + source_polygons[face].0.len();
        source_corners[start..end].sort_unstable_by_key(|&corner| source_order[corner]);
        face_base[face] = source_order[source_corners[start]];
        for (rank, &corner) in source_corners[start..end].iter().enumerate() {
            loop_rank[corner] = rank;
        }
    }

    let mut face_order: Vec<usize> = (0..source_polygons.len()).collect();
    face_order.sort_unstable_by_key(|&face| face_base[face]);
    let mut copy_base = vec![0; source_polygons.len()];
    let mut next = 0;
    for face in face_order {
        copy_base[face] = next;
        next += source_polygons[face].0.len();
    }

    let mut numbering = Vec::with_capacity(2 * corner_total);
    for (corners, &(_, face, copy)) in doubled_corners.iter().zip(doubled_faces) {
        if copy {
            let size = corners.len();
            for &source_corner in corners {
                let slot = (size - loop_rank[source_corner]) % size;
                numbering.push(corner_total + copy_base[face] + slot);
            }
        } else {
            numbering.extend(corners.iter().map(|&corner| source_order[corner]));
        }
    }
    numbering
}

/// One mirror pass over a UV coordinate. A flipped axis is reflected about 0.5, or about the centre
/// of its UDIM tile with `udim`; the mirror offset applies to flipped axes only. The UV offset applies
/// to every copy.
pub(super) fn mirror_uv_pass(
    uv: [f64; 2],
    flip: [bool; 2],
    mirror_offset: [f64; 2],
    offset: [f64; 2],
    udim: bool,
) -> [f64; 2] {
    std::array::from_fn(|axis| {
        let value = if flip[axis] {
            // Blender's UDIM reflection: a whole-number coordinate maps to itself.
            let reflected = if udim {
                uv[axis].ceil() - uv[axis] % 1.0
            } else {
                1.0 - uv[axis]
            };
            reflected + mirror_offset[axis]
        } else {
            uv[axis]
        };
        value + offset[axis]
    })
}

#[cfg(test)]
mod tests {

    use super::{flip_side_name, mirror_uv_pass, split_cycle};

    #[test]
    fn side_names_flip_like_blender() {
        assert_eq!(flip_side_name("Arm.L"), "Arm.R");
        assert_eq!(flip_side_name("Arm_r"), "Arm_l");
        assert_eq!(flip_side_name("Arm.L.001"), "Arm.R.001");
        assert_eq!(flip_side_name("L.Arm"), "R.Arm");
        assert_eq!(flip_side_name("LeftHand"), "RightHand");
        assert_eq!(flip_side_name("hand_RIGHT"), "hand_LEFT");
        assert_eq!(flip_side_name("Body"), "Body");
        assert_eq!(flip_side_name(".L"), ".L");
        assert_eq!(flip_side_name("腕_Left"), "腕_Right");
        assert_eq!(flip_side_name("Left_腕"), "Right_腕");
        assert_eq!(flip_side_name("腕Left"), "腕Right");
        assert_eq!(flip_side_name("腕.L"), "腕.R");
        assert_eq!(flip_side_name("手_右"), "手_右");
    }

    #[test]
    fn bisected_pieces_follow_bmesh_slot_and_first_loop() {
        // Quad a, b, c, d; cut vertex 10 on a-b and 11 on c-d. The kept piece starts at the cut on a-b.
        let kept = split_cycle(&[0, 10, 1, 2, 11, 3], 10, 11);
        assert_eq!(kept, Some((vec![10, 1, 2, 11], vec![0, 10, 11, 3])));
        // The old face keeps the first original loop when it lies between the sorted cut endpoints.
        let split = split_cycle(&[0, 1, 10, 2, 3, 11], 10, 11);
        assert_eq!(split, Some((vec![11, 10, 2, 3], vec![0, 1, 10, 11])));
        // When the old piece contains loop 0, that loop stays first; the new face starts at l_f2.
        let split = split_cycle(&[0, 1, 2, 3, 10, 4, 11, 5], 11, 10);
        assert_eq!(split, Some((vec![0, 1, 2, 3, 10, 11, 5], vec![11, 10, 4])));
    }

    #[test]
    fn reverse_paired_faces_survive_when_merge_is_off() {
        use std::collections::BTreeMap;

        use glam::{DMat4, DVec3};
        use serde_json::json;

        use crate::{
            geom::Mesh,
            model::{Id, Modifier},
        };

        let mesh = Mesh::from_positions_and_faces(
            vec![DVec3::X, DVec3::X + DVec3::Y, DVec3::X + DVec3::Z],
            vec![vec![0, 1, 2], vec![2, 1, 0]],
        )
        .unwrap();
        let modifier = Modifier {
            id: Id::new("mirror").unwrap(),
            modifier_type: "mirror".to_owned(),
            name: "Mirror".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: json!({"use_axis":[true,false,false],"use_mirror_merge":false})
                .as_object()
                .unwrap()
                .clone(),
            binding_data: None,
            runtime: Default::default(),
        };
        let result = super::mirror(&mesh, &modifier, DMat4::IDENTITY, &BTreeMap::new()).unwrap();
        assert_eq!(result.faces.len(), 4);
    }

    #[test]
    fn bisect_uses_incoming_nonidentity_face_loop_basis() {
        use std::collections::BTreeMap;

        use glam::{DMat4, DVec3};
        use serde_json::json;

        use crate::{
            geom::{
                Mesh,
                clnor::{
                    NormalCodeOptions, SetterInput, corner_codes, corner_order, encode_vectors,
                },
            },
            model::{Id, Modifier},
        };

        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.0),
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(1.0, 2.0, 0.0),
                DVec3::new(1.0, 3.0, 0.0),
                DVec3::new(1.0, 3.0, 1.0),
                DVec3::new(1.0, 2.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        )
        .unwrap();
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            json!({"domain":"face","type":"bool","values":{"f0":true,"f1":true}}),
        );
        let source_order = [4, 5, 6, 7, 0, 1, 2, 3];
        let vectors = vec![
            Some(DVec3::new(1.0, 0.1, 0.2)),
            Some(DVec3::new(1.0, 0.2, 0.3)),
            Some(DVec3::new(1.0, 0.3, 0.4)),
            Some(DVec3::new(1.0, 0.4, 0.5)),
            Some(DVec3::new(1.0, 0.5, 0.6)),
            Some(DVec3::new(1.0, 0.6, 0.7)),
            Some(DVec3::new(1.0, 0.7, 0.8)),
            Some(DVec3::new(1.0, 0.8, 0.9)),
        ];
        encode_vectors(
            &mut mesh,
            Some(&vectors),
            NormalCodeOptions {
                corner_order: Some(&source_order),
                setter_input: SetterInput::NormalizeInput,
            },
        )
        .unwrap();
        let source_codes = corner_codes(&mesh).unwrap().unwrap();
        let modifier = Modifier {
            id: Id::new("mirror").unwrap(),
            modifier_type: "mirror".to_owned(),
            name: "Mirror".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: json!({
                "use_axis":[true,false,false],
                "use_bisect_axis":[true,false,false],
                "use_mirror_merge":false
            })
            .as_object()
            .unwrap()
            .clone(),
            binding_data: None,
            runtime: Default::default(),
        };
        let result = super::mirror(&mesh, &modifier, DMat4::IDENTITY, &BTreeMap::new()).unwrap();
        let result_codes = corner_codes(&result).unwrap().unwrap();
        let expected_originals = source_codes[4..]
            .iter()
            .chain(&source_codes[..4])
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(&result_codes[..8], expected_originals.as_slice());
        assert_eq!(corner_order(&result).unwrap(), None);
    }

    #[test]
    fn invalid_zero_space_does_not_turn_auto_copies_into_explicit_zero() {
        use std::collections::BTreeMap;

        use glam::{DMat4, DVec3};
        use serde_json::json;

        use crate::{
            geom::{
                Mesh,
                clnor::{CornerNormalCode, NormalCodeOptions, corner_codes, write_codes},
            },
            model::{Id, Modifier},
        };

        let mut mesh = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(1.0, 0.0, 2.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .unwrap();
        let automatic = vec![CornerNormalCode::Auto; 3];
        write_codes(&mut mesh, Some(&automatic), NormalCodeOptions::default()).unwrap();
        let modifier = Modifier {
            id: Id::new("mirror").unwrap(),
            modifier_type: "mirror".to_owned(),
            name: "Mirror".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: json!({"use_axis":[true,false,false],"use_mirror_merge":false})
                .as_object()
                .unwrap()
                .clone(),
            binding_data: None,
            runtime: Default::default(),
        };
        let result = super::mirror(&mesh, &modifier, DMat4::IDENTITY, &BTreeMap::new()).unwrap();
        assert_eq!(
            corner_codes(&result).unwrap(),
            Some(vec![CornerNormalCode::Auto; 6])
        );
    }

    /// Blender's UDIM mirror (values from a Blender 5.2.2 probe): whole-number coordinates map to
    /// themselves, a tile's interior reflects about its centre, and negative values follow `fmod`.
    #[test]
    fn mirror_udim_reflection_matches_blender_on_tile_edges_and_negatives() {
        let mirrored = |uv| mirror_uv_pass(uv, [true, true], [0.0, 0.0], [0.0, 0.0], true);
        assert_eq!(mirrored([0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(mirrored([1.0, 1.0]), [1.0, 1.0]);
        assert_eq!(mirrored([2.0, 0.5]), [2.0, 0.5]);
        assert_eq!(mirrored([-0.25, 1.25]), [0.25, 1.75]);
        assert_eq!(mirrored([1.25, 0.25]), [1.75, 0.75]);
    }
}
