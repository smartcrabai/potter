//! Output assembly of the exact boolean: Blender `imesh_to_mesh`
//! (`geometry/intern/mesh_boolean.cc` l.716–842), `copy_or_interp_loop_attributes` (l.649–676),
//! and `set_material_from_map` (l.257–280).
//!
//! # Attribute gather
//!
//! Blender builds the output attributes by gathering from the joined mesh through one index map
//! per domain. This module builds the same joined mesh `J` (the analogue of Blender's
//! `joined_mesh`) and the same index maps, then hands `J` to [`remap_interpolated_corners_skipping`],
//! which owns the value rules of the ID-keyed stores and applies Blender's weighted bool mixer to
//! interpolated CORNER booleans:
//!
//! - POINT: an output vertex whose arena origin is `k` takes joined vertex `k`. Blender stores the
//!   LOCAL vertex index as the origin, so operand B's vertex `k` reads the joined element `k`, which
//!   is operand A's vertex `k` whenever A has one. This is Blender's EXACT behavior, reproduced and
//!   listed as a catalog caveat. Cut vertices take the default.
//! - FACE: an output polygon with source `Poly::orig = g` takes joined face `g`.
//! - CORNER and UV: an output ring that equals an unchanged source ring copies the source corners;
//!   any other output face interpolates its corners from its source face by mean value
//!   ([`mean_value_corners`]).
//!
//! Edge stores are gathered here instead: `crease_edge`, `sharp`, `blender_edge_flags`, and EDGE
//! entries of `blender_attributes` take the global edge index of each output edge (`Poly::edge_orig`,
//! last writer wins, as in Blender's `dst_to_src_edge`). The generic remap would derive edges from
//! the vertex origins, and the POINT origin above is not an edge origin. FACE stores
//! (`blender_polygon_smooth`, `shade_smooth`) are joined into `J` and gathered by the remap.
//!
//! `custom_normal` is a CORNER store of raw fan codes. [`join_custom_normal_codes`] copies each
//! operand's codes into the joined mesh untransformed, as Blender's join does (probes
//! `exact_clnor4`, `exact_rot90`, `exact_scale2`). [`mix_custom_normals`] then mixes every output
//! corner from its weighted source corners with Blender's `DefaultMixer<short2>` (`clnor::mix_codes`).
//! A copied corner keeps its code, and a flipped face keeps its codes: Blender does not negate
//! them, and the flip shows only when the codes decode on the flipped face's fan (probe
//! `flip.json`).
//!
//! Native generic attributes (`{"domain","type","values"}` stores keyed by element ID) join every
//! operand's values per element, as Blender's `join_meshes` concatenates each domain. Other stores
//! that are neither gathered nor structured (shading and metadata keys) are copied from operand A.
//!
//! # Deviations
//!
//! - Positions are the arena's double approximation `co_f` rounded to `f32`. Blender rounds its own
//!   double, which GMP truncates, so the two can differ in the last double bit before the `f32`
//!   rounding.
//! - Operand positions use `input::clean_transform` for the `f32` transform. Unchanged rings are
//!   recognized by comparing these positions with the arena's, so they must match exactly.
//! - Blender copies matching corners of a partly matching ring and interpolates the rest. Here a
//!   ring copies only when every corner matches. An interpolated corner that lies on a source vertex
//!   takes that vertex's value exactly, so both give the same values.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use glam::{DVec3, Mat4};
use serde_json::{Map, Value, json};

use crate::error::{PotError, Result};
use crate::geom::{
    Edge, Face, IdCounters, Mesh, Vertex,
    clnor::{self, NormalCodeOptions},
    color_layers, edge_key,
    provenance::{Origin, default_value, mean_value_corners, remap_interpolated_corners_skipping},
    uv_layers,
};

use super::dissolve::Poly;
use super::imesh::{Arena, NO_INDEX};
use super::input::{Operand, clean_transform};

/// Material tables of the operands (Blender `material_remaps`). Entry `k` maps operand `k`'s local
/// material index to an output material index. An empty slice keeps every local index, as Blender
/// does when no remap is computed.
pub(crate) struct OutputMaterials<'a> {
    /// One table per operand, in operand order.
    pub remap: &'a [Vec<u32>],
}

/// Assembles the output mesh of the exact boolean from the kept polygons (Blender `imesh_to_mesh`).
///
/// Vertices are the distinct arena vertices used by `polys`, in Blender's `populate_vert` order, at
/// their `f32` positions. Faces keep their ring order and take fresh IDs. Edges are stored in
/// Blender's `mesh_calc_edges` order. Attributes are gathered as described in the module
/// documentation.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` when the material tables do not match the operand count, a polygon
/// has no valid source face, or the output is invalid.
pub(crate) fn polys_to_mesh(
    polys: &[Poly],
    arena: &Arena,
    operands: &[Operand<'_>],
    materials: &OutputMaterials<'_>,
) -> Result<Mesh> {
    if !materials.remap.is_empty() && materials.remap.len() != operands.len() {
        return Err(PotError::invalid_argument(
            "boolean material tables must be empty or describe every operand",
        ));
    }
    let layout = Layout::new(operands);
    let mut joined = join(operands, &layout)?;
    joined.mesh.attributes = join_attributes(operands, &layout, &joined.mesh)?;
    join_custom_normal_codes(operands, &mut joined.mesh)?;
    uv_layers::write(&mut joined.mesh, &joined_uv_store(operands));
    let (arena_ids, planned) = plan(polys, arena, &layout)?;

    let positions: Vec<DVec3> = arena_ids
        .iter()
        .map(|&id| arena.vert(id).co_f.as_vec3().as_dvec3())
        .collect();
    let mut mesh = Mesh::new();
    let mut vertex_ids = Vec::with_capacity(positions.len());
    for &co in &positions {
        let id = mesh.insert_vertex(co).map_err(|error| {
            PotError::invalid_argument(format!("boolean output vertex failed: {error}"))
        })?;
        vertex_ids.push(id);
    }
    let face_materials: Vec<u32> = planned
        .iter()
        .map(|face| output_material(materials, face, operands))
        .collect();
    mesh.insert_faces(
        planned
            .iter()
            .zip(&face_materials)
            .map(|(face, &material)| {
                let ids: Vec<u32> = face.ring.iter().map(|&vertex| vertex_ids[vertex]).collect();
                Ok((ids, material))
            }),
    )
    .map_err(|error| PotError::invalid_argument(format!("boolean output face failed: {error}")))?;
    blender_edge_order(&mut mesh);

    let origin = Origin {
        vertices: arena_ids
            .iter()
            .map(|&id| point_origin(arena.vert(id).orig, joined.mesh.vertices.len()))
            .collect(),
        faces: planned.iter().map(|face| Some(face.joined_face)).collect(),
        corners: Some(corner_sources(&joined, &planned, &positions)),
    };
    remap_interpolated_corners_skipping(&joined.mesh, &mut mesh, &origin, &["custom_normal"])?;
    mix_custom_normals(
        &joined.mesh,
        &mut mesh,
        origin.corners.as_deref().unwrap_or(&[]),
    )?;

    let sources = edge_sources(&mesh, polys, &planned, &vertex_ids);
    let crease = keyed_edge_attribute(operands, &layout, &mesh, &sources, "crease_edge");
    let sharp = keyed_edge_attribute(operands, &layout, &mesh, &sources, "sharp");
    let flags = edge_flags_attribute(operands, &layout, &sources);
    insert_if(&mut mesh.attributes, "crease_edge", crease);
    insert_if(&mut mesh.attributes, "sharp", sharp);
    insert_if(&mut mesh.attributes, "blender_edge_flags", flags);
    merge_blender_attributes(&mut mesh, operands, &layout, &sources);

    mesh.validate().map_err(|error| {
        PotError::invalid_argument(format!("boolean output is invalid: {error}"))
    })?;
    Ok(mesh)
}

/// Joined index ranges of the operands (Blender `MeshOffsets`).
pub(super) struct Layout {
    vertices: Vec<Range<usize>>,
    edges: Vec<Range<usize>>,
    faces: Vec<Range<usize>>,
}

impl Layout {
    fn new(operands: &[Operand<'_>]) -> Self {
        let mut layout = Self {
            vertices: Vec::with_capacity(operands.len()),
            edges: Vec::with_capacity(operands.len()),
            faces: Vec::with_capacity(operands.len()),
        };
        let (mut vertex, mut edge, mut face) = (0, 0, 0);
        for operand in operands {
            layout
                .vertices
                .push(advance(&mut vertex, operand.mesh.vertices.len()));
            layout
                .edges
                .push(advance(&mut edge, operand.mesh.edges.len()));
            layout
                .faces
                .push(advance(&mut face, operand.mesh.faces.len()));
        }
        layout
    }
}

fn advance(next: &mut usize, len: usize) -> Range<usize> {
    let start = *next;
    *next += len;
    start..*next
}

/// Operand index and local index of a joined index.
fn locate(ranges: &[Range<usize>], index: usize) -> Option<(usize, usize)> {
    ranges.iter().enumerate().find_map(|(operand, range)| {
        range
            .contains(&index)
            .then(|| (operand, index - range.start))
    })
}

/// The joined mesh `J` (Blender `join_meshes_with_transforms`): every operand's vertices in the
/// joined space, and every operand's faces. Vertex and face IDs equal their joined indices.
struct Joined {
    mesh: Mesh,
    /// Joined vertex indices of each joined face, in ring order.
    rings: Vec<Vec<usize>>,
}

fn join(operands: &[Operand<'_>], layout: &Layout) -> Result<Joined> {
    let mut vertices = Vec::new();
    for (operand, range) in operands.iter().zip(&layout.vertices) {
        for (local, co) in joined_positions(operand)?.into_iter().enumerate() {
            vertices.push(Vertex {
                id: joined_id(range.start + local)?,
                co,
            });
        }
    }
    let mut faces = Vec::new();
    let mut rings = Vec::new();
    for (index, operand) in operands.iter().enumerate() {
        let local_of = local_index(operand.mesh.vertices.iter().map(|vertex| vertex.id));
        let base = layout.vertices[index].start;
        for (local_face, face) in operand.mesh.faces.iter().enumerate() {
            let ring = face
                .vertices
                .iter()
                .map(|id| {
                    local_of.get(id).map(|&local| base + local).ok_or_else(|| {
                        PotError::invalid_argument(
                            "boolean operand face references a missing vertex",
                        )
                    })
                })
                .collect::<Result<Vec<usize>>>()?;
            let ids = ring
                .iter()
                .map(|&vertex| joined_id(vertex))
                .collect::<Result<Vec<u32>>>()?;
            faces.push(Face {
                id: joined_id(layout.faces[index].start + local_face)?,
                vertices: ids,
                material_index: face.material_index,
            });
            rings.push(ring);
        }
    }
    let next_id = IdCounters {
        vertex: joined_id(vertices.len())?,
        edge: 0,
        face: joined_id(faces.len())?,
    };
    let mut mesh = Mesh {
        vertices,
        faces,
        next_id,
        ..Mesh::default()
    };
    add_boundary_edges(&mut mesh)?;
    Ok(Joined { mesh, rings })
}

fn joined_id(index: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        PotError::invalid_argument("boolean output exceeds the supported element count")
    })
}

/// Adds one edge per distinct face-boundary vertex pair, so the joined mesh has the complete topology
/// its normal codes are evaluated on. Edge IDs follow first-seen order over the faces.
fn add_boundary_edges(mesh: &mut Mesh) -> Result<()> {
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut edges = Vec::new();
    for face in &mesh.faces {
        let count = face.vertices.len();
        for corner in 0..count {
            let (first, second) = (face.vertices[corner], face.vertices[(corner + 1) % count]);
            if seen.insert(edge_key(first, second)) {
                edges.push(Edge {
                    id: joined_id(edges.len())?,
                    vertices: [first, second],
                });
            }
        }
    }
    mesh.next_id.edge = joined_id(edges.len())?;
    mesh.edges = edges;
    Ok(())
}

/// Operand vertices in the joined space, as `meshes_to_imesh` computes them: rounded to `f32`,
/// then transformed in `f32` (skipped for the identity).
fn joined_positions(operand: &Operand<'_>) -> Result<Vec<DVec3>> {
    let transform = clean_transform(operand.transform.as_mat4());
    let identity = transform == Mat4::IDENTITY;
    let mut positions = Vec::with_capacity(operand.mesh.vertices.len());
    for (index, vertex) in operand.mesh.vertices.iter().enumerate() {
        let mut co = vertex.co.as_vec3();
        if !identity {
            co = transform.transform_point3(co);
        }
        if !co.is_finite() {
            return Err(PotError::invalid_argument(format!(
                "boolean operand vertex {index} is not finite after transform"
            )));
        }
        positions.push(co.as_dvec3());
    }
    Ok(positions)
}

/// One output polygon: its output vertices in ring order and its source face.
struct PlannedFace {
    /// Output vertex storage indices in ring order.
    ring: Vec<usize>,
    /// Source face as a joined face index (Blender `Face::orig`).
    joined_face: usize,
    operand: usize,
    local_face: usize,
}

/// Distinct arena vertices used by `polys`, sorted as Blender's `IMesh::populate_vert` sorts them,
/// and the polygons over them. Blender fills `vert_` in first-seen order (`mesh_intersect.cc:571-577`,
/// `vert_[index] = item.key`) and sorts it with the comparator at `mesh_intersect.cc:583-594`: input
/// vertices by ascending origin, created vertices after them by id. Two input vertices with the same
/// local origin (different operands) tie in that sort, and the tie order sets the stored direction of
/// an edge joining them. `libcxx_sort::parallel_sort_by` reproduces the tie order: libc++ below 500
/// vertices, and oneTBB's quicksort at 500 or more (Blender's TBB `parallel_sort`).
fn plan(polys: &[Poly], arena: &Arena, layout: &Layout) -> Result<(Vec<u32>, Vec<PlannedFace>)> {
    let mut seen_ids: Vec<u32> = Vec::new();
    let mut seen_slot: HashMap<u32, usize> = HashMap::new();
    let mut planned = Vec::with_capacity(polys.len());
    for poly in polys {
        let joined_face = usize::try_from(poly.orig)
            .map_err(|_| PotError::invalid_argument("boolean output polygon has no source face"))?;
        let Some((operand, local_face)) = locate(&layout.faces, joined_face) else {
            return Err(PotError::invalid_argument(
                "boolean output polygon references a missing input face",
            ));
        };
        let ring: Vec<usize> = poly
            .verts
            .iter()
            .map(|&id| {
                *seen_slot.entry(id).or_insert_with(|| {
                    seen_ids.push(id);
                    seen_ids.len() - 1
                })
            })
            .collect();
        planned.push(PlannedFace {
            ring,
            joined_face,
            operand,
            local_face,
        });
    }
    let mut order: Vec<usize> = (0..seen_ids.len()).collect();
    let key = |slot: usize| {
        let id = seen_ids[slot];
        let orig = arena.vert(id).orig;
        if orig == NO_INDEX {
            (true, 0, id)
        } else {
            (false, orig, 0)
        }
    };
    super::libcxx_sort::parallel_sort_by(&mut order, |&a, &b| key(a) < key(b));
    let mut position = vec![0; seen_ids.len()];
    for (index, &slot) in order.iter().enumerate() {
        position[slot] = index;
    }
    for face in &mut planned {
        for vertex in &mut face.ring {
            *vertex = position[*vertex];
        }
    }
    let arena_ids: Vec<u32> = order.iter().map(|&slot| seen_ids[slot]).collect();
    Ok((arena_ids, planned))
}

/// Blender `mesh_calc_edges` (`keep_existing_edges = false`): edges are stored in order of first
/// appearance over the faces, where corner `c` of a face contributes the edge from the previous
/// corner's vertex to its own, each stored with its lower vertex ID first (`OrderedEdge`). Only the
/// storage order and orientation change; edge IDs are kept.
fn blender_edge_order(mesh: &mut Mesh) {
    let mut first_appearance: HashMap<(u32, u32), usize> = HashMap::with_capacity(mesh.edges.len());
    for face in &mesh.faces {
        let ring = &face.vertices;
        let count = ring.len();
        for corner in 0..count {
            let key = edge_key(ring[(corner + count - 1) % count], ring[corner]);
            let next = first_appearance.len();
            first_appearance.entry(key).or_insert(next);
        }
    }
    for edge in &mut mesh.edges {
        let (low, high) = edge_key(edge.vertices[0], edge.vertices[1]);
        edge.vertices = [low, high];
    }
    mesh.edges.sort_by_key(|edge| {
        first_appearance
            .get(&(edge.vertices[0], edge.vertices[1]))
            .copied()
            .unwrap_or(usize::MAX)
    });
}

/// Blender `set_material_from_map`: the operand's table maps a local index inside it, and any other
/// index is kept.
fn output_material(
    materials: &OutputMaterials<'_>,
    face: &PlannedFace,
    operands: &[Operand<'_>],
) -> u32 {
    let local = operands[face.operand].mesh.faces[face.local_face].material_index;
    materials
        .remap
        .get(face.operand)
        .and_then(|table| {
            usize::try_from(local)
                .ok()
                .and_then(|index| table.get(index))
        })
        .copied()
        .unwrap_or(local)
}

/// Blender reads a point attribute at the output vertex's arena origin, a local index, from the
/// joined array. An origin outside the joined array (a cut vertex) has no source.
fn point_origin(orig: i32, joined_count: usize) -> Vec<(usize, f64)> {
    match usize::try_from(orig) {
        Ok(index) if index < joined_count => vec![(index, 1.0)],
        _ => Vec::new(),
    }
}

/// Source corners of each planned face, in output corner order (see [`Origin::corners`]).
///
/// A ring equal to its source ring copies the source corners. Every other face interpolates its
/// corners by mean value over its source face, in one batch so the source index is built once.
fn corner_sources(
    joined: &Joined,
    planned: &[PlannedFace],
    positions: &[DVec3],
) -> Vec<Vec<(usize, f64)>> {
    let mut face_start = Vec::with_capacity(joined.rings.len());
    let mut corners = 0;
    for ring in &joined.rings {
        face_start.push(corners);
        corners += ring.len();
    }
    let mut per_face: Vec<Vec<Vec<(usize, f64)>>> = Vec::with_capacity(planned.len());
    let mut interpolated = Vec::new();
    for (index, face) in planned.iter().enumerate() {
        let source: Vec<DVec3> = joined.rings[face.joined_face]
            .iter()
            .map(|&vertex| joined.mesh.vertices[vertex].co)
            .collect();
        let ring: Vec<DVec3> = face.ring.iter().map(|&vertex| positions[vertex]).collect();
        if let Some(sources) = unchanged_corners(&ring, &source, face_start[face.joined_face]) {
            per_face.push(sources);
        } else {
            per_face.push(Vec::new());
            interpolated.push(index);
        }
    }
    let mean = mean_value_corners(
        &joined.mesh,
        positions,
        interpolated.iter().map(|&index| {
            (
                planned[index].ring.as_slice(),
                Some(planned[index].joined_face),
            )
        }),
    );
    let mut mean = mean.into_iter();
    for index in interpolated {
        per_face[index] = (0..planned[index].ring.len())
            .map(|_| mean.next().unwrap_or_default())
            .collect();
    }
    per_face.into_iter().flatten().collect()
}

/// Blender `fill_orig_loops`: a ring that equals its source ring, from some start corner in the
/// source's loop order, copies each source corner. `None` for any other ring.
fn unchanged_corners(
    ring: &[DVec3],
    source: &[DVec3],
    offset: usize,
) -> Option<Vec<Vec<(usize, f64)>>> {
    let first = *ring.first()?;
    let count = source.len();
    if ring.len() != count {
        return None;
    }
    let start = source.iter().position(|point| point.cmpeq(first).all())?;
    let same = ring
        .iter()
        .enumerate()
        .all(|(index, point)| point.cmpeq(source[(start + index) % count]).all());
    same.then(|| {
        (0..count)
            .map(|index| vec![(offset + (start + index) % count, 1.0)])
            .collect()
    })
}

/// Blender `interpolate_corner_attributes` for `custom_normal` (`geometry_mesh_boolean.cc`): each
/// output corner mixes the raw codes of its weighted source corners with `DefaultMixer<short2>`
/// (`clnor::mix_codes`). A copied corner has one source at weight 1, so it keeps its code. Blender
/// does not negate the codes of a flipped face (probe `flip.json`: an unchanged code decodes to the
/// flipped direction on the flipped fan), so no negation is applied here.
fn mix_custom_normals(joined: &Mesh, mesh: &mut Mesh, sources: &[Vec<(usize, f64)>]) -> Result<()> {
    let Some(codes) = clnor::corner_codes(joined)? else {
        return Ok(());
    };
    let mixed: Vec<clnor::CornerNormalCode> =
        sources
            .iter()
            .map(|corner| {
                clnor::mix_codes(corner.iter().filter_map(|&(index, weight)| {
                    codes.get(index).map(|&code| (code, weight as f32))
                }))
            })
            .collect();
    clnor::write_codes(mesh, Some(mixed.as_slice()), NormalCodeOptions::default())
}

/// Global edge index each output edge takes its attributes from (Blender `dst_to_src_edge`). The
/// last ring edge that names an input edge wins.
fn edge_sources(
    mesh: &Mesh,
    polys: &[Poly],
    planned: &[PlannedFace],
    vertex_ids: &[u32],
) -> Vec<Option<usize>> {
    let slot: HashMap<(u32, u32), usize> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
        .collect();
    let mut sources = vec![None; mesh.edges.len()];
    for (poly, face) in polys.iter().zip(planned) {
        let ring = &face.ring;
        for (corner, &vertex) in ring.iter().enumerate() {
            let next = ring[(corner + 1) % ring.len()];
            let Some(&origin) = poly.edge_orig.get(corner) else {
                continue;
            };
            let Ok(origin) = usize::try_from(origin) else {
                continue;
            };
            if let Some(&index) = slot.get(&edge_key(vertex_ids[vertex], vertex_ids[next])) {
                sources[index] = Some(origin);
            }
        }
    }
    sources
}

/// Joined attributes that the generic remap rewrites, or that this module gathers itself. Other
/// keys are shading and metadata, copied from operand A.
fn is_structured_key(key: &str) -> bool {
    matches!(
        key,
        "vertex_groups"
            | uv_layers::ATTRIBUTE
            | color_layers::ATTRIBUTE
            | "blender_attributes"
            | "skin_radii"
            | "skin_roots"
            | "blender_skin_vertices"
            | "crease_edge"
            | "sharp"
            | "blender_edge_flags"
            | "blender_polygon_smooth"
            | "shade_smooth"
            | "custom_normal"
    )
}

/// The attribute store `J` carries into the generic remap. EDGE entries and the edge stores
/// gathered here are left out and written after the remap.
fn join_attributes(
    operands: &[Operand<'_>],
    layout: &Layout,
    joined: &Mesh,
) -> Result<Map<String, Value>> {
    let mut attributes = Map::new();
    if let Some(first) = operands.first() {
        for (key, value) in &first.mesh.attributes {
            if !is_structured_key(key) && !is_native_store(value) {
                attributes.insert(key.clone(), value.clone());
            }
        }
    }
    // Native generic attributes exist on any operand, so every operand's values join.
    let mut native: Vec<(&str, &Value)> = Vec::new();
    for operand in operands {
        for (key, value) in &operand.mesh.attributes {
            let name = key.as_str();
            if !is_structured_key(name)
                && is_native_store(value)
                && native.iter().all(|(seen, _)| *seen != name)
            {
                native.push((name, value));
            }
        }
    }
    for (key, template) in native {
        attributes.insert(
            key.to_owned(),
            joined_native_attribute(operands, layout, joined, key, template),
        );
    }
    let vertex_ids: Vec<HashMap<u32, usize>> = operands
        .iter()
        .map(|operand| local_index(operand.mesh.vertices.iter().map(|vertex| vertex.id)))
        .collect();
    let face_ids: Vec<HashMap<u32, usize>> = operands
        .iter()
        .map(|operand| local_index(operand.mesh.faces.iter().map(|face| face.id)))
        .collect();
    let vertex_counts = per_operand(operands, |mesh| mesh.vertices.len());
    let face_counts = per_operand(operands, |mesh| mesh.faces.len());
    insert_if(
        &mut attributes,
        "shade_smooth",
        joined_face_attribute(operands, &face_ids, layout, "shade_smooth"),
    );
    insert_if(
        &mut attributes,
        "blender_polygon_smooth",
        joined_array(operands, "blender_polygon_smooth", &face_counts, || {
            json!(false)
        }),
    );
    insert_if(
        &mut attributes,
        "vertex_groups",
        joined_vertex_groups(operands, layout, &vertex_ids),
    );
    insert_if(
        &mut attributes,
        "skin_radii",
        joined_vertex_object(operands, layout, "skin_radii"),
    );
    insert_if(
        &mut attributes,
        "skin_roots",
        joined_vertex_object(operands, layout, "skin_roots"),
    );
    insert_if(
        &mut attributes,
        "blender_skin_vertices",
        joined_array(operands, "blender_skin_vertices", &vertex_counts, || {
            json!({})
        }),
    );
    insert_if(
        &mut attributes,
        "blender_attributes",
        joined_blender_attributes(operands),
    );
    insert_if(
        &mut attributes,
        color_layers::ATTRIBUTE,
        joined_color_layers(operands),
    );
    let color_types = joined_color_data_types(operands);
    if let Some(metadata) = attributes.get_mut("blender_metadata") {
        let metadata = metadata.as_object_mut().ok_or_else(|| {
            PotError::invalid_argument("Boolean blender_metadata must be an object")
        })?;
        if color_types.is_empty() {
            metadata.remove("color_data_types");
        } else {
            metadata.insert("color_data_types".to_owned(), Value::Object(color_types));
        }
    } else if !color_types.is_empty() {
        attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types": color_types}),
        );
    }
    Ok(attributes)
}

/// A native generic attribute store: `{"domain","type","values"}` keyed by element ID.
fn is_native_store(value: &Value) -> bool {
    matches!(
        text_field(value, "domain"),
        "point" | "edge" | "face" | "corner"
    ) && value.get("values").is_some_and(Value::is_object)
}

/// The element ID in a store key such as `f12` for prefix `f`, or `None` for another key.
fn element_id(name: &str, prefix: char) -> Option<u32> {
    name.strip_prefix(prefix)?.parse().ok()
}

/// A native generic attribute joined over every operand, as Blender's `join_meshes` concatenates each
/// domain. Keys move to joined element IDs, which equal joined indices; a corner row is keyed by its
/// joined face. An element without a value has no key and takes the default on remap.
fn joined_native_attribute(
    operands: &[Operand<'_>],
    layout: &Layout,
    joined: &Mesh,
    key: &str,
    template: &Value,
) -> Value {
    let joined_edges: HashMap<(u32, u32), u32> = joined
        .edges
        .iter()
        .map(|edge| (edge_key(edge.vertices[0], edge.vertices[1]), edge.id))
        .collect();
    let promote = text_field(template, "type") == "color"
        && operands.iter().any(|operand| {
            operand
                .mesh
                .attributes
                .get(key)
                .filter(|value| is_native_store(value))
                .is_some_and(|value| text_field(value, "domain") == "corner")
        });
    let domain = if promote {
        "corner"
    } else {
        text_field(template, "domain")
    };
    let mut values = Map::new();
    for (index, operand) in operands.iter().enumerate() {
        let Some(store) = operand
            .mesh
            .attributes
            .get(key)
            .filter(|value| is_native_store(value))
        else {
            continue;
        };
        let Some(entries) = store.get("values").and_then(Value::as_object) else {
            continue;
        };
        if domain == "corner" && text_field(store, "domain") == "point" {
            // A promoted point layer gives every corner its vertex's value. A face with a missing
            // vertex value has no row, so its corners take the default.
            for (local, face) in operand.mesh.faces.iter().enumerate() {
                let row: Option<Vec<Value>> = face
                    .vertices
                    .iter()
                    .map(|id| entries.get(&format!("v{id}")).cloned())
                    .collect();
                if let Some(row) = row {
                    values.insert(
                        format!("f{}", layout.faces[index].start + local),
                        Value::Array(row),
                    );
                }
            }
            continue;
        }
        let vertices = local_index(operand.mesh.vertices.iter().map(|vertex| vertex.id));
        let faces = local_index(operand.mesh.faces.iter().map(|face| face.id));
        match domain {
            "point" => {
                for (name, value) in entries {
                    if let Some(&local) = element_id(name, 'v').and_then(|id| vertices.get(&id)) {
                        values.insert(
                            format!("v{}", layout.vertices[index].start + local),
                            value.clone(),
                        );
                    }
                }
            }
            "face" | "corner" => {
                for (name, value) in entries {
                    if let Some(&local) = element_id(name, 'f').and_then(|id| faces.get(&id)) {
                        values.insert(
                            format!("f{}", layout.faces[index].start + local),
                            value.clone(),
                        );
                    }
                }
            }
            "edge" => {
                for edge in &operand.mesh.edges {
                    let Some(value) = entries.get(&format!("e{}", edge.id)) else {
                        continue;
                    };
                    let [first, second] = edge.vertices.map(|id| {
                        let &local = vertices.get(&id)?;
                        u32::try_from(layout.vertices[index].start + local).ok()
                    });
                    let (Some(first), Some(second)) = (first, second) else {
                        continue;
                    };
                    if let Some(&joined_edge) = joined_edges.get(&edge_key(first, second)) {
                        values.insert(format!("e{joined_edge}"), value.clone());
                    }
                }
            }
            _ => {}
        }
    }
    let mut output = with_values(template, Value::Object(values));
    output["domain"] = json!(domain);
    output
}

fn insert_if(attributes: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        attributes.insert(key.to_owned(), value);
    }
}

/// `vertex_groups` (`{group: {"v{id}": weight}}`) with each member moved to its joined vertex ID.
fn joined_vertex_groups(
    operands: &[Operand<'_>],
    layout: &Layout,
    vertex_ids: &[HashMap<u32, usize>],
) -> Option<Value> {
    let sources = objects_of(operands, "vertex_groups");
    if sources.is_empty() {
        return None;
    }
    let mut names: Vec<&str> = Vec::new();
    for (_, groups) in &sources {
        for name in groups.keys() {
            if !names.contains(&name.as_str()) {
                names.push(name.as_str());
            }
        }
    }
    let mut joined = Map::new();
    for name in names {
        let mut members = Map::new();
        for &(operand, groups) in &sources {
            let Some(weights) = groups.get(name).and_then(Value::as_object) else {
                continue;
            };
            for (key, weight) in weights {
                let Some(local) = key
                    .strip_prefix('v')
                    .and_then(|id| id.parse::<u32>().ok())
                    .and_then(|id| vertex_ids[operand].get(&id))
                else {
                    continue;
                };
                members.insert(
                    format!("v{}", layout.vertices[operand].start + local),
                    weight.clone(),
                );
            }
        }
        joined.insert(name.to_owned(), Value::Object(members));
    }
    Some(Value::Object(joined))
}

/// A vertex-indexed object (`skin_radii`, `skin_roots`) with keys moved to joined vertex indices.
fn joined_vertex_object(operands: &[Operand<'_>], layout: &Layout, key: &str) -> Option<Value> {
    let sources = objects_of(operands, key);
    if sources.is_empty() {
        return None;
    }
    let mut joined = Map::new();
    for (operand, values) in sources {
        for (index, value) in values {
            let Ok(local) = index.parse::<usize>() else {
                continue;
            };
            if local < layout.vertices[operand].len() {
                joined.insert(
                    (layout.vertices[operand].start + local).to_string(),
                    value.clone(),
                );
            }
        }
    }
    Some(Value::Object(joined))
}

/// The joined `uv_layers` store: one layer per name over every operand, faces in joined order.
/// A face of an operand without that layer has no entry, so its corners take the canonical UV on
/// remap. The active and render names come from the subject, operand A.
fn joined_uv_store(operands: &[Operand<'_>]) -> uv_layers::UvStore {
    let stores: Vec<uv_layers::UvStore> = operands
        .iter()
        .map(|operand| uv_layers::read(operand.mesh))
        .collect();
    let mut names: Vec<&str> = Vec::new();
    for store in &stores {
        for layer in &store.layers {
            if !names.contains(&layer.name.as_str()) {
                names.push(layer.name.as_str());
            }
        }
    }
    let layers = names
        .into_iter()
        .map(|name| {
            let mut faces = Vec::new();
            let mut pinned = Vec::new();
            for (operand, store) in operands.iter().zip(&stores) {
                let source = store.layers.iter().find(|layer| layer.name == name);
                for face in 0..operand.mesh.faces.len() {
                    faces.push(
                        source
                            .and_then(|layer| layer.faces.get(face))
                            .cloned()
                            .flatten(),
                    );
                    pinned.push(
                        source
                            .and_then(|layer| layer.pinned.get(face))
                            .cloned()
                            .flatten(),
                    );
                }
            }
            uv_layers::UvLayer {
                name: name.to_owned(),
                faces,
                pinned,
            }
        })
        .collect();
    let subject = stores.first();
    uv_layers::UvStore {
        layers,
        active: subject.and_then(|store| store.active.clone()),
        render: subject.and_then(|store| store.render.clone()),
    }
}

/// The joined `color_layers` list: every operand's color layers in first-seen order, with the
/// subject's active and render names. The remap writes the store from these names and the joined
/// color stores, as it does for the subject's own layers. A name missing from the union falls back
/// to the first layer in `color_layers::write`, as `concat::merge_color_layers` does for members.
fn joined_color_layers(operands: &[Operand<'_>]) -> Option<Value> {
    let stores: Vec<color_layers::ColorStore> = operands
        .iter()
        .map(|operand| color_layers::read(operand.mesh))
        .collect();
    let mut names: Vec<&str> = Vec::new();
    for layer in stores.iter().flat_map(|store| &store.layers) {
        if !names.contains(&layer.name.as_str()) {
            names.push(layer.name.as_str());
        }
    }
    let subject = stores.first()?;
    (!names.is_empty())
        .then(|| json!({"layers": names, "active": subject.active, "render": subject.render}))
}

/// The joined `blender_metadata.color_data_types`: each colour layer's encoding over the operands that
/// hold it. A layer stays `BYTE_COLOR` only when every such operand stores `BYTE_COLOR`. A layer with
/// no entry is `FLOAT_COLOR`, so a mix of encodings joins as float, as Blender's `join_meshes` does.
fn joined_color_data_types(operands: &[Operand<'_>]) -> Map<String, Value> {
    let mut types = Map::new();
    for operand in operands {
        let layers = color_layers::read(operand.mesh).layers;
        if layers.is_empty() {
            continue;
        }
        let stored = operand
            .mesh
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(Value::as_object);
        for layer in layers {
            let byte = stored
                .and_then(|stored| stored.get(&layer.name))
                .and_then(Value::as_str)
                == Some("BYTE_COLOR");
            let encoding = types
                .entry(layer.name)
                .or_insert_with(|| json!("BYTE_COLOR"));
            if !byte {
                *encoding = json!("FLOAT_COLOR");
            }
        }
    }
    types
}

/// The `blender_attributes` entries of `J`: POINT, FACE, and CORNER entries joined per element,
/// and entries of other domains as they are. EDGE entries are gathered by [`merge_blender_attributes`].
fn joined_blender_attributes(operands: &[Operand<'_>]) -> Option<Value> {
    if operands
        .iter()
        .all(|operand| array_of(operand.mesh, "blender_attributes").is_none())
    {
        return None;
    }
    let mut entries = Vec::new();
    for template in attribute_templates(operands) {
        let entry = match template.domain {
            "POINT" | "FACE" | "CORNER" => joined_attribute(operands, &template),
            "EDGE" => continue,
            _ => template.entry.clone(),
        };
        entries.push(entry);
    }
    Some(Value::Array(entries))
}

fn joined_attribute(operands: &[Operand<'_>], template: &AttributeTemplate<'_>) -> Value {
    let counts = element_counts(operands, template.domain);
    let sources: Vec<Option<&[Value]>> = operands
        .iter()
        .map(|operand| {
            find_entry(operand.mesh, template.name, template.domain).and_then(entry_values)
        })
        .collect();
    let data_type = template.data_type;
    with_values(
        template.entry,
        Value::Array(concat_elements(&sources, &counts, || {
            default_value(data_type)
        })),
    )
}

/// The joined array of a per-element store, such as `blender_skin_vertices`.
fn joined_array(
    operands: &[Operand<'_>],
    key: &str,
    counts: &[usize],
    default: impl Fn() -> Value,
) -> Option<Value> {
    if operands
        .iter()
        .all(|operand| array_of(operand.mesh, key).is_none())
    {
        return None;
    }
    let sources: Vec<Option<&[Value]>> = operands
        .iter()
        .map(|operand| array_of(operand.mesh, key))
        .collect();
    Some(Value::Array(concat_elements(&sources, counts, default)))
}

/// Concatenates per-operand element arrays in operand order. An element missing from its operand's
/// array takes `default`.
fn concat_elements(
    sources: &[Option<&[Value]>],
    counts: &[usize],
    default: impl Fn() -> Value,
) -> Vec<Value> {
    let mut joined = Vec::with_capacity(counts.iter().sum::<usize>());
    for (values, &count) in sources.iter().zip(counts) {
        for index in 0..count {
            joined.push(
                values
                    .and_then(|values| values.get(index))
                    .cloned()
                    .unwrap_or_else(&default),
            );
        }
    }
    joined
}

/// Merges the joined `blender_attributes` entries from the generic remap with the EDGE entries,
/// in the order of the first operand that declares each entry.
fn merge_blender_attributes(
    mesh: &mut Mesh,
    operands: &[Operand<'_>],
    layout: &Layout,
    sources: &[Option<usize>],
) {
    if operands
        .iter()
        .all(|operand| array_of(operand.mesh, "blender_attributes").is_none())
    {
        return;
    }
    let remapped = match mesh.attributes.remove("blender_attributes") {
        Some(Value::Array(entries)) => entries,
        _ => Vec::new(),
    };
    let mut remapped = remapped.into_iter();
    let entries: Vec<Value> = attribute_templates(operands)
        .iter()
        .filter_map(|template| {
            if template.domain == "EDGE" {
                Some(edge_entry(operands, layout, sources, template))
            } else {
                remapped.next()
            }
        })
        .collect();
    mesh.attributes
        .insert("blender_attributes".to_owned(), Value::Array(entries));
}

/// An EDGE entry: each output edge takes its source edge's value, or the default.
fn edge_entry(
    operands: &[Operand<'_>],
    layout: &Layout,
    sources: &[Option<usize>],
    template: &AttributeTemplate<'_>,
) -> Value {
    let values = sources
        .iter()
        .map(|source| {
            source
                .and_then(|index| locate(&layout.edges, index))
                .and_then(|(operand, local)| {
                    find_entry(operands[operand].mesh, template.name, template.domain)
                        .and_then(entry_values)
                        .and_then(|values| values.get(local))
                        .cloned()
                })
                .unwrap_or_else(|| default_value(template.data_type))
        })
        .collect();
    with_values(template.entry, Value::Array(values))
}

/// A face-keyed object store (`{"domain","type","values":{"f{id}": value}}`, as `shade_smooth`),
/// with each face key moved to its joined face ID.
fn joined_face_attribute(
    operands: &[Operand<'_>],
    face_ids: &[HashMap<u32, usize>],
    layout: &Layout,
    key: &str,
) -> Option<Value> {
    let template = operands.iter().find_map(|operand| {
        operand
            .mesh
            .attributes
            .get(key)
            .filter(|value| value.is_object())
    })?;
    let mut values = Map::new();
    for (operand, source) in operands.iter().enumerate() {
        let Some(entries) = source
            .mesh
            .attributes
            .get(key)
            .and_then(|attribute| attribute.get("values"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        for (name, value) in entries {
            let Some(local) = name
                .strip_prefix('f')
                .and_then(|id| id.parse::<u32>().ok())
                .and_then(|id| face_ids[operand].get(&id))
            else {
                continue;
            };
            values.insert(
                format!("f{}", layout.faces[operand].start + local),
                value.clone(),
            );
        }
    }
    Some(with_values(template, Value::Object(values)))
}

/// Blender's joined `custom_normal` (`join_meshes`): each operand's raw codes are copied in joined
/// corner order, with no transform. The EXACT probes `exact_clnor4`, `exact_rot90` and `exact_scale2`
/// match every output corner from untransformed input codes under translation, a 90 degree rotation
/// and a non-uniform scale. An operand without custom normals contributes automatic corners, and
/// nothing is written when no operand holds a custom-normal layer (basis-only input has none).
pub(super) fn join_custom_normal_codes(operands: &[Operand<'_>], mesh: &mut Mesh) -> Result<()> {
    let mut codes = Vec::new();
    let mut any = false;
    for operand in operands {
        let count: usize = operand
            .mesh
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum();
        match clnor::corner_codes(operand.mesh)? {
            Some(own) => {
                if own.len() != count {
                    return Err(PotError::invalid_argument(
                        "boolean operand custom normal codes do not match the corners",
                    ));
                }
                any = true;
                codes.extend(own);
            }
            None => codes.extend(std::iter::repeat_n(clnor::CornerNormalCode::Auto, count)),
        }
    }
    if !any {
        return Ok(());
    }
    clnor::write_codes(mesh, Some(codes.as_slice()), NormalCodeOptions::default())
}

/// An edge-keyed object store (`{"domain","type","values":{"e{id}": value}}`, as `crease_edge` and
/// `sharp`). Each output edge takes its source edge's value under its own ID.
fn keyed_edge_attribute(
    operands: &[Operand<'_>],
    layout: &Layout,
    mesh: &Mesh,
    sources: &[Option<usize>],
    key: &str,
) -> Option<Value> {
    let template = operands.iter().find_map(|operand| {
        operand
            .mesh
            .attributes
            .get(key)
            .filter(|value| value.is_object())
    })?;
    let values: Map<String, Value> = mesh
        .edges
        .iter()
        .zip(sources)
        .filter_map(|(edge, source)| {
            let (operand, local) = locate(&layout.edges, (*source)?)?;
            let source_mesh = operands[operand].mesh;
            let id = format!("e{}", source_mesh.edges[local].id);
            let value = source_mesh
                .attributes
                .get(key)?
                .get("values")?
                .get(id.as_str())?;
            Some((format!("e{}", edge.id), value.clone()))
        })
        .collect();
    Some(with_values(template, Value::Object(values)))
}

/// `blender_edge_flags`: `sharp` and `seam` of each output edge's source edge; an edge with no
/// source has neither mark.
fn edge_flags_attribute(
    operands: &[Operand<'_>],
    layout: &Layout,
    sources: &[Option<usize>],
) -> Option<Value> {
    if operands
        .iter()
        .all(|operand| array_of(operand.mesh, "blender_edge_flags").is_none())
    {
        return None;
    }
    let flags = sources
        .iter()
        .map(|source| {
            let flag = source
                .and_then(|index| locate(&layout.edges, index))
                .and_then(|(operand, local)| {
                    array_of(operands[operand].mesh, "blender_edge_flags")
                        .and_then(|flags| flags.get(local))
                });
            json!({"sharp": flag_set(flag, "sharp"), "seam": flag_set(flag, "seam")})
        })
        .collect();
    Some(Value::Array(flags))
}

fn flag_set(flag: Option<&Value>, key: &str) -> bool {
    flag.and_then(|flag| flag.get(key)).and_then(Value::as_bool) == Some(true)
}

/// One distinct `blender_attributes` entry, identified by name and domain, taken from the first
/// operand that declares it.
struct AttributeTemplate<'a> {
    entry: &'a Value,
    name: &'a str,
    domain: &'a str,
    data_type: &'a str,
}

fn attribute_templates<'a>(operands: &[Operand<'a>]) -> Vec<AttributeTemplate<'a>> {
    let mut templates: Vec<AttributeTemplate<'a>> = Vec::new();
    for operand in operands {
        let mesh: &'a Mesh = operand.mesh;
        for entry in array_of(mesh, "blender_attributes").unwrap_or_default() {
            let name = text_field(entry, "name");
            let domain = text_field(entry, "domain");
            if !templates
                .iter()
                .any(|template| template.name == name && template.domain == domain)
            {
                templates.push(AttributeTemplate {
                    entry,
                    name,
                    domain,
                    data_type: text_field(entry, "data_type"),
                });
            }
        }
    }
    templates
}

/// The `blender_attributes` entry `name` in `domain` of `mesh`.
fn find_entry<'a>(mesh: &'a Mesh, name: &str, domain: &str) -> Option<&'a Value> {
    array_of(mesh, "blender_attributes")?
        .iter()
        .find(|entry| text_field(entry, "name") == name && text_field(entry, "domain") == domain)
}

/// Element counts of each operand in `domain`.
fn element_counts(operands: &[Operand<'_>], domain: &str) -> Vec<usize> {
    match domain {
        "POINT" => per_operand(operands, |mesh| mesh.vertices.len()),
        "FACE" => per_operand(operands, |mesh| mesh.faces.len()),
        "EDGE" => per_operand(operands, |mesh| mesh.edges.len()),
        "CORNER" => per_operand(operands, corner_count),
        _ => Vec::new(),
    }
}

fn per_operand(operands: &[Operand<'_>], count: impl Fn(&Mesh) -> usize) -> Vec<usize> {
    operands.iter().map(|operand| count(operand.mesh)).collect()
}

fn corner_count(mesh: &Mesh) -> usize {
    mesh.faces.iter().map(|face| face.vertices.len()).sum()
}

/// Maps each ID to its position in storage order.
fn local_index(ids: impl Iterator<Item = u32>) -> HashMap<u32, usize> {
    ids.enumerate().map(|(index, id)| (id, index)).collect()
}

fn text_field<'v>(entry: &'v Value, key: &str) -> &'v str {
    entry.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn entry_values(entry: &Value) -> Option<&[Value]> {
    entry.get("values")?.as_array().map(Vec::as_slice)
}

/// `entry` with its `values` replaced; the other fields are kept.
fn with_values(entry: &Value, values: Value) -> Value {
    let mut object: Map<String, Value> = entry
        .as_object()
        .map(|object| {
            object
                .iter()
                .filter(|(key, _)| key.as_str() != "values")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();
    object.insert("values".to_owned(), values);
    Value::Object(object)
}

/// The array stored under `key` in `mesh`.
fn array_of<'a>(mesh: &'a Mesh, key: &str) -> Option<&'a [Value]> {
    mesh.attributes.get(key)?.as_array().map(Vec::as_slice)
}

/// The object stored under `key` in `mesh`.
fn object_of<'a>(mesh: &'a Mesh, key: &str) -> Option<&'a Map<String, Value>> {
    mesh.attributes.get(key)?.as_object()
}

/// The object under `key` of each operand that has one, with the operand's index.
fn objects_of<'a>(operands: &[Operand<'a>], key: &str) -> Vec<(usize, &'a Map<String, Value>)> {
    operands
        .iter()
        .enumerate()
        .filter_map(|(index, operand)| Some((index, object_of(operand.mesh, key)?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DVec3};
    use serde_json::{Value, json};

    use super::{OutputMaterials, polys_to_mesh};
    use crate::geom::boolean::{
        dissolve::Poly,
        imesh::{Arena, NO_INDEX},
        input::Operand,
    };
    use crate::geom::{
        Mesh,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        uv_layers,
    };

    type TestResult = Result<(), String>;

    fn mesh_from(positions: Vec<DVec3>, polygons: Vec<Vec<usize>>) -> Result<Mesh, String> {
        Mesh::from_positions_and_faces(positions, polygons).map_err(|error| format!("{error:?}"))
    }

    /// Two triangles of a unit quad at height `z` (faces 0 and 1).
    fn quad(z: f64) -> Result<Mesh, String> {
        mesh_from(
            vec![
                DVec3::new(0.0, 0.0, z),
                DVec3::new(1.0, 0.0, z),
                DVec3::new(0.0, 1.0, z),
                DVec3::new(1.0, 1.0, z),
            ],
            vec![vec![0, 1, 2], vec![1, 3, 2]],
        )
    }

    fn triangle(z: f64) -> Result<Mesh, String> {
        mesh_from(
            vec![
                DVec3::new(0.0, 0.0, z),
                DVec3::new(1.0, 0.0, z),
                DVec3::new(0.0, 1.0, z),
            ],
            vec![vec![0, 1, 2]],
        )
    }

    fn operand(mesh: &Mesh) -> Operand<'_> {
        Operand {
            mesh,
            transform: DMat4::IDENTITY,
        }
    }

    /// Stores `vectors`, one per corner in face order, as `mesh`'s native custom-normal codes.
    fn set_authored_normals(mesh: &mut Mesh, vectors: &[DVec3]) -> TestResult {
        let vectors: Vec<Option<DVec3>> = vectors.iter().copied().map(Some).collect();
        crate::geom::clnor::encode_vectors(
            mesh,
            Some(&vectors),
            crate::geom::clnor::NormalCodeOptions::default(),
        )
        .map_err(|error| format!("{error:?}"))
    }

    /// The evaluated per-corner normals of `mesh`: authored codes and the Auto fan alike. A corner whose
    /// authored vector equals its fan normal is stored as Auto, so the evaluated view is the comparable one.
    fn authored_normals(mesh: &Mesh) -> Result<Vec<DVec3>, String> {
        let positions: Vec<DVec3> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
        crate::geom::normals::corner_normals(mesh, &positions).map_err(|error| format!("{error:?}"))
    }

    /// Fails unless `normal` points along `expected` to within code precision.
    fn assert_near(normal: DVec3, expected: DVec3) -> TestResult {
        if (normal - expected).length() < 1e-3 {
            Ok(())
        } else {
            Err(format!("normal {normal:?} is not {expected:?}"))
        }
    }

    /// Blender `meshes_to_imesh` for an untransformed operand: arena vertices with local origins.
    fn arena_vertices(arena: &mut Arena, mesh: &Mesh) -> Result<Vec<u32>, String> {
        let mut ids = Vec::with_capacity(mesh.vertices.len());
        for (index, vertex) in mesh.vertices.iter().enumerate() {
            let orig = i32::try_from(index).map_err(|error| error.to_string())?;
            ids.push(arena.add_or_find_dvec(vertex.co, orig));
        }
        Ok(ids)
    }

    fn poly(verts: Vec<u32>, orig: i32) -> Poly {
        let edge_orig = vec![NO_INDEX; verts.len()];
        Poly {
            verts,
            orig,
            edge_orig,
        }
    }

    fn build(
        polys: &[Poly],
        arena: &Arena,
        operands: &[Operand<'_>],
        remap: &[Vec<u32>],
    ) -> Result<Mesh, String> {
        polys_to_mesh(polys, arena, operands, &OutputMaterials { remap })
            .map_err(|error| format!("{error:?}"))
    }

    fn attribute(mesh: &Mesh, key: &str) -> Value {
        mesh.attributes.get(key).cloned().unwrap_or(Value::Null)
    }

    fn set_attribute(mesh: &mut Mesh, key: &str, value: Value) {
        mesh.attributes.insert(key.to_owned(), value);
    }

    /// Polygons of both quads: A's faces (origins 0, 1), then B's (origins 2, 3).
    fn quad_polys(arena: &mut Arena, a: &Mesh, b: &Mesh) -> Result<Vec<Poly>, String> {
        let a_ids = arena_vertices(arena, a)?;
        let b_ids = arena_vertices(arena, b)?;
        Ok(vec![
            poly(vec![a_ids[0], a_ids[1], a_ids[2]], 0),
            poly(vec![a_ids[1], a_ids[3], a_ids[2]], 1),
            poly(vec![b_ids[0], b_ids[1], b_ids[2]], 2),
            poly(vec![b_ids[1], b_ids[3], b_ids[2]], 3),
        ])
    }

    #[test]
    fn face_attributes_come_from_the_source_operand() -> TestResult {
        let mut a = quad(0.0)?;
        let mut b = quad(5.0)?;
        set_attribute(
            &mut a,
            "blender_attributes",
            json!([{"name": "fid", "domain": "FACE", "data_type": "INT", "values": [100, 101]}]),
        );
        set_attribute(
            &mut b,
            "blender_attributes",
            json!([{"name": "fid", "domain": "FACE", "data_type": "INT", "values": [200, 201]}]),
        );
        set_attribute(&mut a, "blender_polygon_smooth", json!([true, false]));
        set_attribute(&mut b, "blender_polygon_smooth", json!([false, true]));
        let operands = [operand(&a), operand(&b)];
        let mut arena = Arena::new();
        let polys = quad_polys(&mut arena, &a, &b)?;
        let mesh = build(&polys, &arena, &operands, &[])?;
        assert_eq!(
            attribute(&mesh, "blender_attributes")[0]["values"],
            json!([100, 101, 200, 201])
        );
        assert_eq!(
            attribute(&mesh, "blender_polygon_smooth"),
            json!([true, false, false, true])
        );
        Ok(())
    }

    #[test]
    fn native_face_attributes_join_every_operand() -> TestResult {
        let mut a = quad(0.0)?;
        let mut b = quad(5.0)?;
        // Potter's import keeps FACE INT attributes as native stores keyed by face ID, so every
        // operand's values join, not only the first operand's.
        set_attribute(
            &mut a,
            "fid",
            json!({"domain": "face", "type": "int", "values": {"f0": 100, "f1": 101}}),
        );
        set_attribute(
            &mut b,
            "fid",
            json!({"domain": "face", "type": "int", "values": {"f0": 200, "f1": 201}}),
        );
        let operands = [operand(&a), operand(&b)];
        let mut arena = Arena::new();
        let polys = quad_polys(&mut arena, &a, &b)?;
        let mesh = build(&polys, &arena, &operands, &[])?;
        let values = attribute(&mesh, "fid")["values"].clone();
        let fids: Vec<Value> = mesh
            .faces
            .iter()
            .map(|face| values[format!("f{}", face.id)].clone())
            .collect();
        assert_eq!(fids, vec![json!(100), json!(101), json!(200), json!(201)]);
        Ok(())
    }

    #[test]
    fn color_layers_list_every_operands_layers() -> TestResult {
        let mut a = quad(0.0)?;
        let mut b = quad(5.0)?;
        let body = ColorStore {
            layers: vec![ColorLayer {
                name: "Body".to_owned(),
                domain: ColorDomain::Point,
                values: vec![[1.0, 0.0, 0.0, 1.0]; 4],
            }],
            active: Some("Body".to_owned()),
            render: Some("Body".to_owned()),
        };
        color_layers::write(&mut a, &body);
        let paint = ColorStore {
            layers: vec![ColorLayer {
                name: "Paint".to_owned(),
                domain: ColorDomain::Corner,
                values: vec![[0.0, 1.0, 0.0, 1.0]; 6],
            }],
            active: None,
            render: Some("Paint".to_owned()),
        };
        color_layers::write(&mut b, &paint);
        let operands = [operand(&a), operand(&b)];
        let mut arena = Arena::new();
        let polys = quad_polys(&mut arena, &a, &b)?;
        let mesh = build(&polys, &arena, &operands, &[])?;
        let store = color_layers::read(&mesh);
        let names: Vec<&str> = store
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect();
        assert_eq!(names, ["Body", "Paint"]);
        assert_eq!(store.active.as_deref(), Some("Body"));
        assert_eq!(store.render.as_deref(), Some("Body"));
        color_layers::validate(&mesh).map_err(|error| format!("{error:?}"))?;
        Ok(())
    }

    #[test]
    fn material_tables_map_each_face_through_its_operand() -> TestResult {
        let mut a = quad(0.0)?;
        a.faces[1].material_index = 1;
        let mut b = quad(5.0)?;
        b.faces[1].material_index = 1;
        let operands = [operand(&a), operand(&b)];
        let mut arena = Arena::new();
        let polys = quad_polys(&mut arena, &a, &b)?;
        let materials = |mesh: &Mesh| -> Vec<u32> {
            mesh.faces.iter().map(|face| face.material_index).collect()
        };

        let index_tables = vec![vec![1, 0], vec![1, 0]];
        let mesh = build(&polys, &arena, &operands, &index_tables)?;
        assert_eq!(materials(&mesh), vec![1, 0, 1, 0]);

        let transfer_tables = vec![vec![0, 1], vec![2, 3]];
        let mesh = build(&polys, &arena, &operands, &transfer_tables)?;
        assert_eq!(materials(&mesh), vec![0, 1, 2, 3]);

        let partial_tables = vec![vec![7], vec![7, 8]];
        let mesh = build(&polys, &arena, &operands, &partial_tables)?;
        assert_eq!(materials(&mesh), vec![7, 1, 7, 8]);
        Ok(())
    }

    #[test]
    fn point_attributes_read_the_joined_array_at_the_local_origin() -> TestResult {
        let mut a = triangle(0.0)?;
        set_attribute(
            &mut a,
            "vertex_groups",
            json!({"Weights": {"v0": 0.1, "v1": 0.2, "v2": 0.3}}),
        );
        let mut b = triangle(5.0)?;
        set_attribute(
            &mut b,
            "vertex_groups",
            json!({"Weights": {"v0": 0.9, "v1": 0.9, "v2": 0.9}}),
        );
        let operands = [operand(&a), operand(&b)];
        let mut arena = Arena::new();
        arena_vertices(&mut arena, &a)?;
        let b_ids = arena_vertices(&mut arena, &b)?;
        // B's only face has global index 1, after A's single face.
        let polys = [poly(b_ids, 1)];
        let mesh = build(&polys, &arena, &operands, &[])?;
        assert_eq!(
            attribute(&mesh, "vertex_groups"),
            json!({"Weights": {"v0": 0.1, "v1": 0.2, "v2": 0.3}})
        );
        Ok(())
    }

    #[test]
    fn output_positions_are_the_f32_rounding_of_the_exact_coordinates() -> TestResult {
        let a = triangle(0.0)?;
        let mut arena = Arena::new();
        let exact = DVec3::new(0.1, 1.0 / 3.0, 2.000_000_001);
        let ids = [
            arena.add_or_find_dvec(exact, 0),
            arena.add_or_find_dvec(DVec3::new(1.0, 0.0, 0.0), 1),
            arena.add_or_find_dvec(DVec3::new(0.0, 1.0, 0.0), 2),
        ];
        let mesh = build(&[poly(ids.to_vec(), 0)], &arena, &[operand(&a)], &[])?;
        let expected = exact.as_vec3().as_dvec3();
        assert!(mesh.vertices[0].co.cmpeq(expected).all());
        assert!(!mesh.vertices[0].co.cmpeq(exact).all());
        Ok(())
    }

    #[test]
    fn ring_order_and_face_count_are_preserved() -> TestResult {
        let pentagon = mesh_from(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 1.0, 0.0),
                DVec3::new(1.0, 2.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2, 3, 4]],
        )?;
        let mut arena = Arena::new();
        let ids = arena_vertices(&mut arena, &pentagon)?;
        let ring = vec![ids[4], ids[2], ids[0], ids[3], ids[1]];
        let mesh = build(&[poly(ring, 0)], &arena, &[operand(&pentagon)], &[])?;
        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(mesh.faces[0].vertices, vec![4, 2, 0, 3, 1]);
        assert_eq!(mesh.edges.len(), 5);
        assert!(mesh.vertices[0].co.cmpeq(DVec3::new(0.0, 0.0, 0.0)).all());
        Ok(())
    }

    #[test]
    fn vertices_follow_blender_populate_order() -> TestResult {
        let a = triangle(0.0)?;
        let mut arena = Arena::new();
        let created = arena.add_or_find_dvec(DVec3::new(0.5, 0.5, 1.0), NO_INDEX);
        let second = arena.add_or_find_dvec(DVec3::new(1.0, 0.0, 0.0), 1);
        let first = arena.add_or_find_dvec(DVec3::new(0.0, 0.0, 0.0), 0);
        let mesh = build(
            &[poly(vec![created, second, first], 0)],
            &arena,
            &[operand(&a)],
            &[],
        )?;
        assert_eq!(mesh.faces[0].vertices, vec![2, 1, 0]);
        assert!(mesh.vertices[0].co.cmpeq(DVec3::new(0.0, 0.0, 0.0)).all());
        assert!(mesh.vertices[1].co.cmpeq(DVec3::new(1.0, 0.0, 0.0)).all());
        assert!(mesh.vertices[2].co.cmpeq(DVec3::new(0.5, 0.5, 1.0)).all());
        Ok(())
    }

    #[test]
    fn edges_follow_blender_mesh_calc_edges_order() -> TestResult {
        let a = triangle(0.0)?;
        let mut arena = Arena::new();
        let ids = [
            arena.add_or_find_dvec(DVec3::new(0.0, 0.0, 0.0), 0),
            arena.add_or_find_dvec(DVec3::new(1.0, 0.0, 0.0), 1),
            arena.add_or_find_dvec(DVec3::new(0.0, 1.0, 0.0), 2),
            arena.add_or_find_dvec(DVec3::new(1.0, 1.0, 0.0), 3),
        ];
        let polys = [
            poly(vec![ids[0], ids[1], ids[2]], 0),
            poly(vec![ids[2], ids[1], ids[3]], 0),
        ];
        let mesh = build(&polys, &arena, &[operand(&a)], &[])?;
        let edges: Vec<[u32; 2]> = mesh.edges.iter().map(|edge| edge.vertices).collect();
        assert_eq!(edges, vec![[0, 2], [0, 1], [1, 2], [2, 3], [1, 3]]);
        Ok(())
    }

    #[test]
    fn corner_values_copy_unchanged_rings_and_interpolate_cut_rings() -> TestResult {
        let mut a = mesh_from(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(0.0, 2.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )?;
        uv_layers::write_layers(
            &mut a,
            &[uv_layers::UvLayer::new(
                "UVMap",
                vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
            )],
        );
        let mut arena = Arena::new();
        let ids = arena_vertices(&mut arena, &a)?;
        let cut_a = arena.add_or_find_dvec(DVec3::new(1.0, 0.0, 0.0), NO_INDEX);
        let cut_b = arena.add_or_find_dvec(DVec3::new(0.0, 1.0, 0.0), NO_INDEX);
        let polys = [poly(ids.clone(), 0), poly(vec![ids[0], cut_a, cut_b], 0)];
        let mesh = build(&polys, &arena, &[operand(&a)], &[])?;
        let uv: Vec<[f64; 2]> = uv_layers::read(&mesh).layers[0]
            .faces
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let expected = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
        ];
        assert_eq!(uv.len(), expected.len());
        for (got, want) in uv.iter().zip(expected) {
            assert!((got[0] - want[0]).abs() < 1e-9, "{got:?} != {want:?}");
            assert!((got[1] - want[1]).abs() < 1e-9, "{got:?} != {want:?}");
        }
        Ok(())
    }

    #[test]
    fn edge_attributes_come_from_the_ring_edge_origin() -> TestResult {
        let mut a = triangle(0.0)?;
        set_attribute(
            &mut a,
            "blender_edge_flags",
            json!([
                {"sharp": false, "seam": false},
                {"sharp": false, "seam": false},
                {"sharp": true, "seam": false}
            ]),
        );
        let mut arena = Arena::new();
        let ids = arena_vertices(&mut arena, &a)?;
        let mut polygon = poly(ids, 0);
        // Ring edge 1 (from corner 1 to corner 2) is input edge 2 of A.
        polygon.edge_orig = vec![NO_INDEX, 2, NO_INDEX];
        let mesh = build(&[polygon], &arena, &[operand(&a)], &[])?;
        assert_eq!(
            attribute(&mesh, "blender_edge_flags"),
            json!([
                {"sharp": false, "seam": false},
                {"sharp": false, "seam": false},
                {"sharp": true, "seam": false}
            ])
        );
        Ok(())
    }

    #[test]
    fn boolean_joins_color_encodings_and_promotes_point_layers() -> TestResult {
        let mut a = triangle(0.0)?;
        let mut b = triangle(5.0)?;
        let corner = ColorStore {
            layers: vec![ColorLayer {
                name: "Body".to_owned(),
                domain: ColorDomain::Corner,
                values: vec![[0.1, 0.2, 0.3, 0.4]; 3],
            }],
            active: Some("Body".to_owned()),
            render: Some("Body".to_owned()),
        };
        color_layers::write(&mut a, &corner);
        let point = ColorStore {
            layers: vec![ColorLayer {
                name: "Body".to_owned(),
                domain: ColorDomain::Point,
                values: vec![
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
            }],
            active: None,
            render: None,
        };
        color_layers::write(&mut b, &point);
        a.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types": {"Body": "BYTE_COLOR"}}),
        );
        let mut arena = Arena::new();
        let a_ids = arena_vertices(&mut arena, &a)?;
        let b_ids = arena_vertices(&mut arena, &b)?;
        let polys = [poly(a_ids, 0), poly(b_ids, 1)];
        let mesh = build(&polys, &arena, &[operand(&a), operand(&b)], &[])?;

        // One operand is FLOAT_COLOR (no entry), so the joined layer is float.
        assert_eq!(
            mesh.attributes["blender_metadata"]["color_data_types"]["Body"],
            json!("FLOAT_COLOR")
        );
        let store = color_layers::read(&mesh);
        assert_eq!(store.layers.len(), 1);
        assert_eq!(store.layers[0].domain, ColorDomain::Corner);
        Ok(())
    }

    #[test]
    fn custom_normals_of_flipped_faces_keep_their_codes_and_decode_flipped() -> TestResult {
        let mut a = mesh_from(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
                DVec3::new(5.0, 0.0, 0.0),
                DVec3::new(6.0, 0.0, 0.0),
                DVec3::new(5.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )?;
        set_authored_normals(&mut a, &[DVec3::new(0.0, 0.0, 2.0); 6])?;
        let mut arena = Arena::new();
        let ids = arena_vertices(&mut arena, &a)?;
        let polys = [
            poly(vec![ids[0], ids[1], ids[2]], 0),
            poly(vec![ids[5], ids[4], ids[3]], 1),
        ];
        let mesh = build(&polys, &arena, &[operand(&a)], &[])?;
        let normals = authored_normals(&mesh)?;
        // The kept face copies its codes. The reversed face keeps its codes too; they decode on its
        // flipped fan to the flipped direction (Blender does not negate them).
        for normal in &normals[..3] {
            assert_near(*normal, DVec3::Z)?;
        }
        for normal in &normals[3..] {
            assert_near(*normal, -DVec3::Z)?;
        }
        Ok(())
    }

    #[test]
    fn custom_normal_codes_are_copied_without_the_operand_transform() -> TestResult {
        let mut a = triangle(0.0)?;
        set_authored_normals(&mut a, &[DVec3::new(0.0, 0.0, 2.0); 3])?;
        let mut arena = Arena::new();
        let mut ids = Vec::with_capacity(a.vertices.len());
        for (index, vertex) in a.vertices.iter().enumerate() {
            let orig = i32::try_from(index).map_err(|error| error.to_string())?;
            let co = DVec3::new(2.0 * vertex.co.x, vertex.co.y, vertex.co.z);
            ids.push(arena.add_or_find_dvec(co, orig));
        }
        let scaled = Operand {
            mesh: &a,
            transform: DMat4::from_scale(DVec3::new(2.0, 1.0, 1.0)),
        };
        let mesh = build(&[poly(ids, 0)], &arena, &[scaled], &[])?;
        // Blender copies the raw codes of a joined operand; the operand transform does not touch them.
        let source = crate::geom::clnor::corner_codes(&a).map_err(|error| format!("{error:?}"))?;
        let joined =
            crate::geom::clnor::corner_codes(&mesh).map_err(|error| format!("{error:?}"))?;
        assert_eq!(joined, source);
        Ok(())
    }
}
