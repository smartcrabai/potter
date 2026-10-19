//! Blender `MOD_solidify_extrude_modifyMesh` (`EXTRUDE` mode). The output keeps Blender's element
//! numbering: the originals, then the shell copies (each copied face reversed), then the rim faces
//! and the rim edges joining open boundaries to their copies. The attribute writers address that
//! numbering through [`Layout`], so the two must stay in step.

use std::collections::{HashMap, HashSet};
use std::f64::consts::{PI, TAU};

use glam::{DVec3, Vec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{Mesh, edge_key, normals::face_normal, provenance::Origin},
    model::Modifier,
};

use super::super::{
    Polygon, build_mesh, cyclic_pairs, edge_pairs, id_indices, mesh_attribute_values, polygons,
    positions, vertex_normals,
};
use super::{Params, has_deform_verts, has_vertex_group, set_group_weight, util};

/// The input in Blender's array form: storage-order elements, and per-face corner data.
struct Source {
    positions: Vec<Vec3>,
    edges: Vec<[usize; 2]>,
    faces: Vec<Vec<usize>>,
    materials: Vec<u32>,
    /// Per face, per corner `c`: the edge from corner `c` to corner `c + 1` (`corner_edges`).
    face_edges: Vec<Vec<usize>>,
    /// Flat index of each face's first corner (`face_offsets`).
    face_starts: Vec<usize>,
    /// Face normals (`face_normals_true`) and vertex normals (`vert_normals_true`). Computed in
    /// `f64` by the shared helpers and rounded once to `f32`.
    face_normals: Vec<Vec3>,
    vert_normals: Vec<Vec3>,
}

impl Source {
    fn new(mesh: &Mesh) -> Result<Self> {
        let points = positions(mesh);
        let polys = polygons(mesh)?;
        let edges = edge_pairs(mesh)?;
        let edge_index: HashMap<(usize, usize), usize> = edges
            .iter()
            .enumerate()
            .map(|(index, [first, second])| (edge_key(*first, *second), index))
            .collect();
        let mut face_starts = Vec::with_capacity(polys.len());
        let mut face_edges = Vec::with_capacity(polys.len());
        let mut start = 0;
        for (vertices, _) in &polys {
            face_starts.push(start);
            start += vertices.len();
            let corner_edges = cyclic_pairs(vertices)
                .map(|(first, second)| {
                    edge_index
                        .get(&edge_key(*first, *second))
                        .copied()
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "solidify face edge is missing from the mesh",
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            face_edges.push(corner_edges);
        }
        let mut has_face = vec![false; points.len()];
        for (vertices, _) in &polys {
            for &vertex in vertices {
                has_face[vertex] = true;
            }
        }
        let face_normals: Vec<Vec3> = polys
            .iter()
            .map(|(vertices, _)| {
                let corners: Vec<DVec3> = vertices.iter().map(|&vertex| points[vertex]).collect();
                face_normal(&corners).as_vec3()
            })
            .collect();
        let vert_normals: Vec<Vec3> = vertex_normals(&points, &polys)
            .into_iter()
            .enumerate()
            .map(|(vertex, normal)| {
                if has_face[vertex] {
                    normal.as_vec3()
                } else {
                    points[vertex].as_vec3().normalize_or_zero()
                }
            })
            .collect();
        let (faces, materials): (Vec<Vec<usize>>, Vec<u32>) = polys.into_iter().unzip();
        Ok(Self {
            positions: points.iter().map(|point| point.as_vec3()).collect(),
            edges,
            faces,
            materials,
            face_edges,
            face_starts,
            face_normals,
            vert_normals,
        })
    }
}

/// Blender's `EdgeUser`: which faces use an edge.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeUser {
    Unused,
    /// Two or more faces use the edge (Blender's `INVALID_PAIR`).
    Pair,
    /// Exactly one face uses the edge. `flip` is set when the face walks the edge against its
    /// stored direction (Blender's `face + faces_num`).
    Face {
        face: usize,
        flip: bool,
    },
}

/// Blender's `edge_users` and `edge_order`. `order[edge]` is the corner that ends the edge in the
/// face that uses it (the edge runs from corner `order - 1` to corner `order`).
fn edge_users(src: &Source) -> (Vec<EdgeUser>, Vec<usize>) {
    let mut users = vec![EdgeUser::Unused; src.edges.len()];
    let mut order = vec![0; src.edges.len()];
    for (face, vertices) in src.faces.iter().enumerate() {
        let mut previous = vertices.len().saturating_sub(1);
        for (corner, &vertex) in vertices.iter().enumerate() {
            let edge = src.face_edges[face][previous];
            if users[edge] == EdgeUser::Unused {
                let [first, second] = src.edges[edge];
                let forward = (vertices[previous] > vertex) == (first < second);
                users[edge] = EdgeUser::Face {
                    face,
                    flip: !forward,
                };
                order[edge] = corner;
            } else {
                users[edge] = EdgeUser::Pair;
            }
            previous = corner;
        }
    }
    (users, order)
}

/// Blender's `edge_user_pairs`: for each edge used by exactly one face per walking direction, the
/// `[face walking one way, face walking the other way]` pair. Other edges have no pair.
fn direction_pairs(src: &Source) -> Vec<Option<[usize; 2]>> {
    let mut slots = vec![[None::<usize>; 2]; src.edges.len()];
    let mut invalid = vec![false; src.edges.len()];
    for (face, vertices) in src.faces.iter().enumerate() {
        let mut previous = vertices.len().saturating_sub(1);
        for (corner, &vertex) in vertices.iter().enumerate() {
            let edge = src.face_edges[face][previous];
            let [first, second] = src.edges[edge];
            let slot = usize::from((vertices[previous] > vertex) == (first < second));
            if slots[edge][slot].is_none() && !invalid[edge] {
                slots[edge][slot] = Some(face);
            } else {
                invalid[edge] = true;
            }
            previous = corner;
        }
    }
    slots
        .iter()
        .zip(&invalid)
        .map(|(pair, &bad)| match (bad, pair) {
            (false, [Some(first), Some(second)]) => Some([*first, *second]),
            _ => None,
        })
        .collect()
}

/// Blender's `vert_lens`: per vertex, the squared length of its shortest edge (`f32::MAX` if none).
fn shortest_edge_squared(src: &Source) -> Vec<f32> {
    let mut lengths = vec![f32::MAX; src.positions.len()];
    for &[first, second] in &src.edges {
        let length = (src.positions[first] - src.positions[second]).length_squared();
        lengths[first] = lengths[first].min(length);
        lengths[second] = lengths[second].min(length);
    }
    lengths
}

/// The rim: one quad per boundary edge, and the vertices that join each boundary to its copy.
struct Rim {
    /// `(edge, face, flip)` per boundary edge, ascending (`new_edge_arr`).
    edges: Vec<(usize, usize, bool)>,
    /// Vertices on boundary edges, ascending (`new_vert_arr`).
    verts: Vec<usize>,
    /// Position of each vertex in `verts` (`old_vert_arr`). `usize::MAX` for other vertices, which
    /// are never read.
    slot: Vec<usize>,
}

impl Rim {
    fn new(src: &Source, users: &[EdgeUser], enabled: bool) -> Self {
        let edges: Vec<(usize, usize, bool)> = if enabled {
            users
                .iter()
                .enumerate()
                .filter_map(|(edge, user)| match *user {
                    EdgeUser::Face { face, flip } => Some((edge, face, flip)),
                    EdgeUser::Unused | EdgeUser::Pair => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut marked = vec![false; src.positions.len()];
        for &(edge, _, _) in &edges {
            for vertex in src.edges[edge] {
                marked[vertex] = true;
            }
        }
        let verts: Vec<usize> = marked
            .iter()
            .enumerate()
            .filter_map(|(vertex, &mark)| mark.then_some(vertex))
            .collect();
        let mut slot = vec![usize::MAX; marked.len()];
        for (index, &vertex) in verts.iter().enumerate() {
            slot[vertex] = index;
        }
        Self { edges, verts, slot }
    }
}

/// Blender's output numbering. Originals come first. With a shell, every vertex and edge is copied
/// after them, in order; without one, only the rim vertices and rim edges are copied. Rim vertices
/// and rim edges follow.
struct Layout {
    verts_num: usize,
    edges_num: usize,
    do_shell: bool,
    rim: Rim,
    /// Source edges copied after the originals, in output order.
    copy_edges: Vec<usize>,
}

impl Layout {
    fn new(src: &Source, users: &[EdgeUser], do_rim: bool, do_shell: bool) -> Self {
        let rim = Rim::new(src, users, do_rim);
        let edges_num = src.edges.len();
        let copy_edges: Vec<usize> = if do_shell {
            (0..edges_num).collect()
        } else {
            rim.edges.iter().map(|&(edge, _, _)| edge).collect()
        };
        Self {
            verts_num: src.positions.len(),
            edges_num,
            do_shell,
            rim,
            copy_edges,
        }
    }

    /// Source vertices copied after the originals: all of them with a shell, else the rim.
    fn copy_sources(&self) -> Vec<usize> {
        if self.do_shell {
            (0..self.verts_num).collect()
        } else {
            self.rim.verts.clone()
        }
    }

    /// Output index of the copy of source vertex `vertex`. Only called for copied vertices.
    fn copy_of(&self, vertex: usize) -> usize {
        if self.do_shell {
            vertex + self.verts_num
        } else {
            self.rim.slot[vertex] + self.verts_num
        }
    }

    /// Output edge index of the copy of `copy_edges[position]`.
    fn copy_edge(&self, position: usize) -> usize {
        self.edges_num + position
    }

    /// Output edge index of the rim edge joining rim vertex `index` (a position in `rim.verts`).
    fn vertical_edge(&self, index: usize) -> usize {
        self.edges_num + self.copy_edges.len() + index
    }

    /// Vertex pairs of every output edge, in output edge order.
    fn edge_pairs(&self, source_edges: &[[usize; 2]]) -> Vec<[usize; 2]> {
        let mut edges = source_edges.to_vec();
        edges.extend(self.copy_edges.iter().map(|&edge| {
            let [first, second] = source_edges[edge];
            [self.copy_of(first), self.copy_of(second)]
        }));
        edges.extend(
            self.rim
                .verts
                .iter()
                .map(|&vertex| [vertex, self.copy_of(vertex)]),
        );
        edges
    }
}

/// A block of output vertices that receive one offset pass: the originals, or the copies.
#[derive(Clone, Copy)]
struct Block {
    start: usize,
    count: usize,
    /// Whether the block's `i`-th vertex is the source vertex `rim.verts[i]` rather than `i`.
    rim: bool,
}

/// Blender's `MOD_solidify_extrude_modifyMesh`.
pub(super) fn solidify_extrude(mesh: &Mesh, _modifier: &Modifier, params: &Params) -> Result<Mesh> {
    let src = Source::new(mesh)?;
    let verts_num = src.positions.len();
    let edges_num = src.edges.len();
    let do_rim = params.rim;
    let do_shell = !(do_rim && params.rim_only);
    let (users, order) = edge_users(&src);
    let layout = Layout::new(&src, &users, do_rim, do_shell);
    let rim_count = layout.rim.verts.len();

    // Blender applies material offsets only when an evaluated object has at least two slots.
    let (max_material, mat_ofs, mat_ofs_rim) = if params.material_slot_count > 1 {
        (
            i64::try_from(params.material_slot_count - 1).unwrap_or(i64::MAX),
            params.material_offset,
            params.material_offset_rim,
        )
    } else {
        (0, 0, 0)
    };

    let thickness = params.thickness as f32;
    let offset_fac = params.offset_fac as f32;
    let offset_clamp = params.offset_clamp as f32;
    let bevel_convex = params.bevel_convex as f32;
    // The thickness is split between the original side (`ofs_orig`) and the shell (`ofs_new`).
    #[expect(
        clippy::manual_midpoint,
        reason = "matches Blender's float evaluation order for parity"
    )]
    let ofs_orig = -(((-offset_fac + 1.0) * 0.5) * thickness);
    let ofs_new = thickness + ofs_orig;
    let do_flip = params.flip_normals;
    let do_clamp = offset_clamp != 0.0;
    let do_angle_clamp = do_clamp && params.angle_clamp;
    let do_bevel_convex = bevel_convex != 0.0;
    let even = params.even_offset;

    let offset_fac_vg = params.offset_fac_vg as f32;
    let offset_fac_vg_inv = 1.0 - offset_fac_vg;
    // Blender's `MOD_get_vgroup` drops the group when the mesh has no deform-vert data at all.
    let group_weights: Option<Vec<f32>> = params
        .groups
        .thickness
        .as_ref()
        .filter(|_| has_deform_verts(mesh))
        .map(|weights| weights.iter().map(|&weight| weight as f32).collect());
    // Blender's per-vertex thickness factor, `None` without the thickness group.
    let group_scale = |index: usize| -> Option<f32> {
        group_weights.as_ref().map(|weights| {
            let weight = if params.invert_vertex_group {
                1.0 - weights[index]
            } else {
                weights[index]
            };
            offset_fac_vg + weight * offset_fac_vg_inv
        })
    };

    // Blender's `MOD_SOLIDIFY_NORMAL_CALC` (quality normals) is the only switch for the high
    // quality normals and for the non-manifold corners of the even-offset pass.
    let (normals, non_manifold) = if params.quality_normals {
        util::high_quality_vertex_normals(
            verts_num,
            &src.edges,
            &src.face_edges,
            &src.face_normals,
            &src.vert_normals,
        )
    } else {
        (src.vert_normals.clone(), vec![false; edges_num])
    };

    let pairs = direction_pairs(&src);
    let lens = shortest_edge_squared(&src);
    // Angles across each edge's fold, and per vertex the largest fold among its edges (`vert_angs`).
    let mut edge_angles = vec![0.0_f32; edges_num];
    let mut vert_angs = vec![std::f32::consts::FRAC_PI_2; verts_num];
    for (edge, pair) in pairs.iter().enumerate() {
        let Some([first_face, second_face]) = *pair else {
            continue;
        };
        let [first, second] = src.edges[edge];
        let (first_normal, second_normal) =
            (src.face_normals[first_face], src.face_normals[second_face]);
        let axis = util::normalized(src.positions[first] - src.positions[second]);
        let angle = util::angle_signed_on_axis(first_normal, second_normal, axis);
        edge_angles[edge] = angle;
        if do_angle_clamp {
            // EVEN mode measures the unsigned fold, `π − angle`.
            let clamp_angle = if even {
                (PI - f64::from(util::angle_normalized(first_normal, second_normal))) as f32
            } else {
                angle
            };
            vert_angs[first] = vert_angs[first].max(clamp_angle);
            vert_angs[second] = vert_angs[second].max(clamp_angle);
        }
    }

    // EVEN mode: each vertex's offset is the average of its corners' angle-weighted offsets.
    let mut vert_angles: Vec<f32> = Vec::new();
    let mut vert_accum: Vec<f32> = Vec::new();
    if even {
        vert_angles = vec![0.0; verts_num];
        vert_accum = vec![0.0; verts_num];
        for (face, vertices) in src.faces.iter().enumerate() {
            let count = vertices.len();
            for corner in 0..count {
                let vertex = vertices[corner];
                let previous = vertices[(corner + count - 1) % count];
                let next = vertices[(corner + 1) % count];
                let previous_direction =
                    util::normalized(src.positions[previous] - src.positions[vertex]);
                let next_direction = util::normalized(src.positions[vertex] - src.positions[next]);
                let angle =
                    util::angle_normalized(previous_direction, next_direction).max(f32::EPSILON);
                vert_accum[vertex] += angle;
                let non_manifold_corner = non_manifold[src.face_edges[face][corner]]
                    || non_manifold[src.face_edges[face][(corner + 1) % count]];
                let distance = if non_manifold_corner {
                    1.0
                } else {
                    util::shell_distance(normals[vertex], src.face_normals[face])
                };
                vert_angles[vertex] += distance * angle;
            }
        }
        for (index, angle) in vert_angles.iter_mut().enumerate() {
            if let Some(scale) = group_scale(index) {
                *angle *= scale;
            }
        }
        if do_clamp {
            let clamp_factor = 1.0
                + if do_angle_clamp {
                    offset_fac.abs()
                } else {
                    0.0
                };
            let offset = thickness.abs() * offset_clamp * clamp_factor;
            if offset > f32::EPSILON {
                let offset_sq = offset * offset;
                for index in 0..verts_num {
                    if do_angle_clamp {
                        let cos_ang = (vert_angs[index] * 0.5).cos();
                        if cos_ang > 0.0 {
                            let max_off = lens[index].sqrt() * 0.5 / cos_ang;
                            if max_off < offset * 0.5 {
                                vert_angles[index] *= max_off / offset * 2.0;
                            }
                        }
                    } else if lens[index] < offset_sq {
                        vert_angles[index] *= lens[index].sqrt() / offset;
                    }
                }
            }
        }
    }

    // Copies start at the original positions. Offsets then move each block, and the two offsets
    // target the two blocks that Blender's `INIT_VERT_ARRAY_OFFSETS` selects.
    let copy_sources = layout.copy_sources();
    let mut positions: Vec<Vec3> = src.positions.clone();
    positions.extend(copy_sources.iter().map(|&vertex| src.positions[vertex]));
    let block = |first: bool| -> Block {
        if first {
            Block {
                start: 0,
                count: verts_num,
                rim: false,
            }
        } else if do_shell {
            Block {
                start: verts_num,
                count: verts_num,
                rim: false,
            }
        } else {
            Block {
                start: verts_num,
                count: rim_count,
                rim: true,
            }
        }
    };
    let new_first = (ofs_new >= ofs_orig) != do_flip;
    let new_block = block(new_first);
    let orig_block = block(!new_first);

    let plain_clamp = thickness.abs() * offset_clamp;
    let plain_clamp_sq = plain_clamp * plain_clamp;
    let plain_pass = |positions: &mut [Vec3], block: Block, base: f32, is_new: bool| {
        if base == 0.0 {
            return;
        }
        for i_orig in 0..block.count {
            let index = if block.rim {
                layout.rim.verts[i_orig]
            } else {
                i_orig
            };
            let mut scale = match group_scale(index) {
                Some(factor) => factor * base,
                None => base,
            };
            if do_clamp && plain_clamp > f32::EPSILON {
                if do_angle_clamp {
                    // The shell side reads the fold from the far side, `2π − angle`. The original
                    // side reads by block position, as Blender does.
                    let cos_ang = if is_new {
                        (((TAU - f64::from(vert_angs[index])) * 0.5) as f32).cos()
                    } else {
                        (vert_angs[i_orig] * 0.5).cos()
                    };
                    if cos_ang > 0.0 {
                        let max_off = lens[index].sqrt() * 0.5 / cos_ang;
                        if max_off < plain_clamp * 0.5 {
                            scale *= max_off / plain_clamp * 2.0;
                        }
                    }
                } else if lens[index] < plain_clamp_sq {
                    scale *= lens[index].sqrt() / plain_clamp;
                }
            }
            positions[block.start + i_orig] += normals[index] * scale;
        }
    };
    let even_pass = |positions: &mut [Vec3], block: Block, base: f32| {
        if base == 0.0 {
            return;
        }
        for i_orig in 0..block.count {
            let index = if block.rim {
                layout.rim.verts[i_orig]
            } else {
                i_orig
            };
            if vert_accum[index] != 0.0 {
                positions[block.start + i_orig] +=
                    normals[index] * (base * (vert_angles[index] / vert_accum[index]));
            }
        }
    };
    if even {
        even_pass(positions.as_mut_slice(), new_block, ofs_new);
        even_pass(positions.as_mut_slice(), orig_block, ofs_orig);
    } else {
        plain_pass(positions.as_mut_slice(), new_block, ofs_new, true);
        plain_pass(positions.as_mut_slice(), orig_block, ofs_orig, false);
    }

    // Faces, with the source corner each output corner copies (Blender's `corner_interp`).
    let (out_polygons, face_sources, corners) =
        output_faces(&src, &layout, &order, mat_ofs, mat_ofs_rim, max_material);
    let logical = layout.edge_pairs(&src.edges);
    let logical_index: HashMap<(usize, usize), usize> = logical
        .iter()
        .enumerate()
        .map(|(index, [first, second])| (edge_key(*first, *second), index))
        .collect();
    let used: HashSet<(usize, usize)> = out_polygons
        .iter()
        .flat_map(|(vertices, _)| {
            cyclic_pairs(vertices).map(|(first, second)| edge_key(*first, *second))
        })
        .collect();
    let loose: Vec<[usize; 2]> = logical
        .iter()
        .filter(|pair| !used.contains(&edge_key(pair[0], pair[1])))
        .copied()
        .collect();
    let vertex_origin: Vec<usize> = (0..verts_num).chain(copy_sources.iter().copied()).collect();
    let origin = Origin {
        vertices: vertex_origin
            .iter()
            .map(|&vertex| vec![(vertex, 1.0)])
            .collect(),
        faces: face_sources,
        corners: Some(corners.iter().map(|&corner| vec![(corner, 1.0)]).collect()),
        edges: None,
    };
    let points: Vec<DVec3> = positions.iter().map(|point| point.as_dvec3()).collect();
    let mut result = build_mesh(points, out_polygons, loose, &origin, mesh)?;

    // Output edge index → Blender edge index.
    let output_ids = id_indices(&result);
    let logical_edge: Vec<usize> = result
        .edges
        .iter()
        .map(|edge| {
            output_ids
                .get(&edge.vertices[0])
                .copied()
                .zip(output_ids.get(&edge.vertices[1]).copied())
                .and_then(|(first, second)| logical_index.get(&edge_key(first, second)).copied())
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "solidify output edge is outside Blender's edge layout",
                    )
                })
        })
        .collect::<Result<_>>()?;

    if do_rim
        && (params.crease_rim != 0.0 || params.crease_outer != 0.0 || params.crease_inner != 0.0)
    {
        let mut creases = carried_edge_values(&source_edge_values(mesh, "crease_edge"), &layout);
        let crease_rim = params.crease_rim as f32;
        let crease_outer = params.crease_outer as f32;
        let crease_inner = params.crease_inner as f32;
        if crease_rim != 0.0 {
            for index in 0..layout.rim.verts.len() {
                creases[layout.vertical_edge(index)] = crease_rim;
            }
        }
        for (index, &(edge, _, _)) in layout.rim.edges.iter().enumerate() {
            if crease_outer != 0.0 {
                creases[edge] = add_crease(creases[edge], crease_outer);
            }
            if crease_inner != 0.0 {
                let position = if layout.do_shell { edge } else { index };
                let copy = layout.copy_edge(position);
                creases[copy] = add_crease(creases[copy], crease_inner);
            }
        }
        let mut values = Map::new();
        for (edge, &index) in result.edges.iter().zip(&logical_edge) {
            let value = creases[index];
            if value != 0.0 {
                values.insert(format!("e{}", edge.id), json!(f64::from(value)));
            }
        }
        remove_blender_attribute(&mut result, "crease_edge");
        result.attributes.insert(
            "crease_edge".to_owned(),
            json!({"domain": "edge", "type": "float", "values": values}),
        );
    }

    let source_bevel_vert = mesh_attribute_values(mesh, "bevel_weight_vert").map(|values| {
        values
            .iter()
            .map(|value| value.as_f64().map_or(0.0, |weight| weight as f32))
            .collect::<Vec<f32>>()
    });
    if mesh_attribute_values(mesh, "bevel_weight_edge").is_some()
        || do_bevel_convex
        || source_bevel_vert.is_some()
    {
        let mut bevel =
            carried_edge_values(&source_edge_values(mesh, "bevel_weight_edge"), &layout);
        if let Some(vertex_weights) = &source_bevel_vert {
            for (index, &vertex) in layout.rim.verts.iter().enumerate() {
                bevel[layout.vertical_edge(index)] =
                    vertex_weights.get(vertex).copied().unwrap_or(0.0);
            }
        }
        if do_bevel_convex {
            apply_bevel_convex(
                &mut bevel,
                &pairs,
                &edge_angles,
                bevel_convex,
                edges_num,
                do_shell,
            );
        }
        let values: Map<String, Value> = result
            .edges
            .iter()
            .zip(&logical_edge)
            .map(|(edge, &index)| (format!("e{}", edge.id), json!(f64::from(bevel[index]))))
            .collect();
        remove_blender_attribute(&mut result, "bevel_weight_edge");
        result.attributes.insert(
            "bevel_weight_edge".to_owned(),
            json!({"domain": "edge", "type": "float", "values": values}),
        );
    }

    // Output vertex groups: the shell is every copy; the rim is every rim vertex and its copy.
    // Blender only writes a group that exists on the mesh.
    if has_vertex_group(mesh, &params.shell_group) {
        for index in verts_num..result.vertices.len() {
            let id = result.vertices[index].id;
            set_group_weight(&mut result, &params.shell_group, id);
        }
    }
    if has_vertex_group(mesh, &params.rim_group) {
        for &vertex in &layout.rim.verts {
            let id = result.vertices[vertex].id;
            set_group_weight(&mut result, &params.rim_group, id);
            let copy_id = result.vertices[layout.copy_of(vertex)].id;
            set_group_weight(&mut result, &params.rim_group, copy_id);
        }
    }

    Ok(result)
}

/// Blender's output faces: originals, then reversed shell copies, then rim quads. The second value
/// is each face's source face. The third is each output corner's source corner, in output order.
fn output_faces(
    src: &Source,
    layout: &Layout,
    order: &[usize],
    shell_material: i64,
    rim_material: i64,
    max_material: i64,
) -> (Vec<Polygon>, Vec<Option<usize>>, Vec<usize>) {
    let mut out_polygons: Vec<Polygon> = Vec::new();
    let mut face_sources = Vec::new();
    let mut corners = Vec::new();
    for (face, vertices) in src.faces.iter().enumerate() {
        out_polygons.push((vertices.clone(), src.materials[face]));
        face_sources.push(Some(face));
        corners.extend(src.face_starts[face]..src.face_starts[face] + vertices.len());
    }
    if layout.do_shell {
        for (face, vertices) in src.faces.iter().enumerate() {
            let count = vertices.len();
            // Blender reverses each copy: copy corner `c` takes source corner `(count - c) % count`.
            let reversed: Vec<usize> = (0..count)
                .map(|corner| vertices[(count - corner) % count] + layout.verts_num)
                .collect();
            out_polygons.push((
                reversed,
                shifted_material(src.materials[face], shell_material, max_material),
            ));
            face_sources.push(Some(face));
            corners
                .extend((0..count).map(|corner| src.face_starts[face] + (count - corner) % count));
        }
    }
    for &(edge, face, flip) in &layout.rim.edges {
        let [first, second] = src.edges[edge];
        let count = src.faces[face].len();
        let end = src.face_starts[face] + order[edge];
        let start = src.face_starts[face] + (order[edge] + count - 1) % count;
        let quad = if flip {
            vec![second, first, layout.copy_of(first), layout.copy_of(second)]
        } else {
            vec![first, second, layout.copy_of(second), layout.copy_of(first)]
        };
        out_polygons.push((
            quad,
            shifted_material(src.materials[face], rim_material, max_material),
        ));
        face_sources.push(Some(face));
        corners.extend([end, start, start, end]);
    }
    (out_polygons, face_sources, corners)
}

/// Blender's material offset: shift by `offset` and clamp to the slot range. Blender only clamps
/// when the offset is nonzero, so a face outside the range with no offset keeps its index.
fn shifted_material(material: u32, offset: i64, max_material: i64) -> u32 {
    if offset == 0 {
        material
    } else {
        // The clamp keeps the value within `0..=max_material`, which fits `u32`.
        (i64::from(material) + offset).clamp(0, max_material) as u32
    }
}

/// Blender's `crease += amount` for rim edges, clamped above at 1 (no lower clamp).
fn add_crease(value: f32, amount: f32) -> f32 {
    let total = value + amount;
    if total > 1.0 { 1.0 } else { total }
}

/// Blender's `bevel_convex` on the edge weights of pair edges. A fold below π moves the original
/// edge, a fold above π moves the shell copy; each takes the factor on the matching side.
fn apply_bevel_convex(
    bevel: &mut [f32],
    pairs: &[Option<[usize; 2]>],
    angles: &[f32],
    bevel_convex: f32,
    edges_num: usize,
    shell: bool,
) {
    let positive = bevel_convex.clamp(0.0, 1.0);
    let negative = bevel_convex.clamp(-1.0, 0.0);
    for (edge, pair) in pairs.iter().enumerate() {
        if pair.is_none() {
            continue;
        }
        let angle = f64::from(angles[edge]);
        let original = if angle < PI { positive } else { negative };
        bevel[edge] = (bevel[edge] + original).clamp(0.0, 1.0);
        if shell {
            let copy = if angle > PI { positive } else { negative };
            bevel[edges_num + edge] = (bevel[edges_num + edge] + copy).clamp(0.0, 1.0);
        }
    }
}

/// A float edge attribute in source storage order: from a keyed `values` object (`e{id}`), or from
/// a storage-order array. Missing values are zero.
fn source_edge_values(mesh: &Mesh, name: &str) -> Vec<f32> {
    if let Some(values) = mesh
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object)
    {
        return mesh
            .edges
            .iter()
            .map(|edge| {
                values
                    .get(&format!("e{}", edge.id))
                    .and_then(Value::as_f64)
                    .map_or(0.0, |value| value as f32)
            })
            .collect();
    }
    let values = mesh_attribute_values(mesh, name).unwrap_or_default();
    (0..mesh.edges.len())
        .map(|index| {
            values
                .get(index)
                .and_then(Value::as_f64)
                .map_or(0.0, |value| value as f32)
        })
        .collect()
}

/// Output edge values carried over from the source: originals and their copies take the values of
/// the source edge they came from; rim vertical edges start at zero.
fn carried_edge_values(source: &[f32], layout: &Layout) -> Vec<f32> {
    let mut values =
        vec![0.0_f32; layout.edges_num + layout.copy_edges.len() + layout.rim.verts.len()];
    values[..layout.edges_num].copy_from_slice(&source[..layout.edges_num]);
    for (position, &edge) in layout.copy_edges.iter().enumerate() {
        values[layout.copy_edge(position)] = source[edge];
    }
    values
}

fn remove_blender_attribute(mesh: &mut Mesh, name: &str) {
    if let Some(entries) = mesh
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
    {
        entries.retain(|entry| entry.get("name").and_then(Value::as_str) != Some(name));
    }
}
